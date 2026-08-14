import logging
from typing import Optional, Union

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    GenerationConfig,
    GenerationMixin,
    LlamaModel,
    PretrainedConfig,
    SmolVLMConfig,
    SmolVLMModel,
    SmolVLMPreTrainedModel,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.models.smolvlm.modeling_smolvlm import (
    SmolVLMCausalLMOutputWithPast,
    SmolVLMForConditionalGeneration,
)

logger = logging.getLogger(__name__)


class LlamaActionExpertConfig(PretrainedConfig):
    """
    Configuration for LlamaActionExpert model.

    The Action Expert is a Llama-based transformer that processes action tokens
    in the SmolVLA architecture. It shares attention computation with the VLM
    through shared Q/K/V projections, enabling cross-attention between vision-language
    features and action tokens.

    Key Parameters:
        vocab_size: Vocabulary size for action tokens (default: 2048)
        hidden_size: Hidden dimension of the action expert (can differ from VLM)
        intermediate_size: Intermediate dimension in MLP layers
        num_hidden_layers: Number of transformer layers (typically matches VLM)
        num_attention_heads: Number of attention heads (must match VLM for shared attention)
        num_key_value_heads: Number of KV heads for GQA (must match VLM)
        head_dim: Dimension per attention head (set to match VLM's head_dim for alignment)

    Note:
        The action expert's hidden_size can be smaller than VLM's, but attention
        dimensions (head_dim, num_heads) must align for joint attention computation.
        This is handled automatically by SmolVLAParallelDecoding._resize_expert_heads_to_match_vlm().
    """

    model_type = "llama_action_expert"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `LlamaModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: Optional[int] = 2048,
        hidden_size: Optional[int] = 2048,
        intermediate_size: Optional[int] = 5120,
        num_hidden_layers: Optional[int] = 32,
        num_attention_heads: Optional[int] = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 2048,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[int] = 1e-5,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pretraining_tp: Optional[int] = 1,
        tie_word_embeddings: Optional[bool] = False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        mlp_bias: Optional[bool] = False,
        head_dim: Optional[int] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = (
            head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        )
        # Validate the correctness of rotary position embeddings parameters
        # Backward compatibility: if there is a 'type' field, copy it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        kwargs["_attn_implementation"] = "eager"
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class LlamaActionExpert(LlamaModel):
    """
    Llama-based Action Expert model for processing action tokens.

    This model extends LlamaModel to serve as an action token processor in the
    SmolVLA architecture. It receives action embeddings and processes them through
    standard Llama transformer layers with bidirectional attention (unlike the
    causal attention in standard language models).

    Architecture:
        - Standard Llama decoder layers with configurable hidden_size
        - Supports GQA (Grouped Query Attention) for efficient inference
        - Bidirectional attention allows action tokens to attend to each other
        - RoPE (Rotary Position Embeddings) for position encoding

    The model is designed to work in conjunction with a VLM, sharing attention
    computation for cross-modal reasoning between vision-language features and actions.
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position: torch.Tensor = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class SmolVLAParallelDecodingConfig(PretrainedConfig):
    """
    Configuration for SmolVLAParallelDecoding model.

    This configuration manages the dual-path architecture consisting of a VLM
    (Vision-Language Model) and an Action Expert. The parallel decoding approach
    generates all action tokens in a single forward pass, enabling efficient
    action prediction.

    Configuration Flow:
        1. Create from existing VLM config using from_vlm_config()
        2. Auto-generate action_expert_config with aligned attention dimensions
        3. Action Expert's hidden_size can differ from VLM (default: VLM's 1/2)

    Key Parameters:
        vlm_config: Configuration for the VLM (SmolVLM)
        action_expert_config: Configuration for the Action Expert (LlamaActionExpert)
        token_budget: Number of action tokens to generate (default: 16)
        action_vocab_size: Size of action vocabulary (default: 2048)
        action_hidden_size: Hidden size for action expert (default: VLM's 1/2)
        action_intermediate_size: MLP intermediate size for action expert

    Note:
        Use from_vlm_config() class method to create a properly aligned configuration
        from an existing VLM configuration.
    """

    model_type = "smolvla_parallel_decoding"

    def __init__(
        self,
        vlm_config=None,
        action_expert_config=None,
        action_hidden_size: int = None,
        action_intermediate_size: int = None,
        token_budget: int = 16,
        action_vocab_size: int = 2048,
        **kwargs,
    ):
        # Handle dict inputs for nested configs (from JSON deserialization)
        if isinstance(vlm_config, dict):
            vlm_config = SmolVLMConfig(**vlm_config)
        if isinstance(action_expert_config, dict):
            action_expert_config = LlamaActionExpertConfig(**action_expert_config)

        super().__init__(**kwargs)
        self.vlm_config = vlm_config
        self.action_expert_config = action_expert_config
        self.token_budget = token_budget
        self.action_vocab_size = action_vocab_size
        self.action_hidden_size = action_hidden_size
        self.action_intermediate_size = action_intermediate_size

        # Get initializer_range from vlm_config for weight initialization
        if vlm_config is not None:
            if hasattr(vlm_config, "text_config") and hasattr(
                vlm_config.text_config, "initializer_range"
            ):
                self.initializer_range = vlm_config.text_config.initializer_range
            elif hasattr(vlm_config, "initializer_range"):
                self.initializer_range = vlm_config.initializer_range
            else:
                self.initializer_range = 0.02  # Default value
        else:
            self.initializer_range = 0.02  # Default value

    @classmethod
    def from_vlm_config(
        cls,
        vlm_config,
        *,
        action_hidden_size: int = None,
        action_intermediate_size: int = None,
        token_budget: int = 16,
        action_vocab_size: int = 2048,
        **kwargs,
    ):
        """Create config from VLM config, auto-generate action_expert_config"""
        # Extract info from vlm_config's text_config
        text_config = vlm_config.text_config if hasattr(vlm_config, "text_config") else vlm_config

        # Get VLM parameters
        vlm_hidden_size = getattr(text_config, "hidden_size", None)
        if vlm_hidden_size is None:
            raise ValueError(
                f"Cannot get hidden_size from text_config, text_config type: {type(text_config)}"
            )

        vlm_num_heads = getattr(text_config, "num_attention_heads", None)
        if vlm_num_heads is None:
            raise ValueError("Cannot get num_attention_heads from text_config")

        vlm_num_kv_heads = getattr(text_config, "num_key_value_heads", vlm_num_heads)

        num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
        if num_hidden_layers is None:
            raise ValueError("Cannot get num_hidden_layers from text_config")

        # If action_hidden_size not specified, default to VLM's 1/2
        if action_hidden_size is None:
            action_hidden_size = vlm_hidden_size // 2

        if action_intermediate_size is None:
            # Llama typically uses 4 * hidden_size * 2/3, simplified here
            action_intermediate_size = action_hidden_size * 4

        # Calculate VLM's head_dim (this is the anchor point for Attention alignment)
        vlm_head_dim = vlm_hidden_size // vlm_num_heads

        action_expert_config = LlamaActionExpertConfig(
            vocab_size=action_vocab_size,
            hidden_size=action_hidden_size,  # [Modified] Use decoupled size
            intermediate_size=action_intermediate_size,
            num_hidden_layers=num_hidden_layers,  # Number of layers typically kept consistent for one-to-one correspondence
            num_attention_heads=vlm_num_heads,  # [Must be consistent]
            num_key_value_heads=vlm_num_kv_heads,  # [Must be consistent]
            # Pass head_dim mainly for recording, LlamaModel may not recognize it by default, we fix in Model init
            head_dim=vlm_head_dim,
            # ... other parameters kept consistent ...
            hidden_act=getattr(text_config, "hidden_act", "silu"),
            max_position_embeddings=getattr(text_config, "max_position_embeddings", 2048),
            initializer_range=getattr(text_config, "initializer_range", 0.02),
            rms_norm_eps=getattr(text_config, "rms_norm_eps", 1e-5),
            use_cache=False,  # Usually don't use Cache during training
            _attn_implementation="eager",  # Bidirectional attention must use eager
        )

        return cls(
            vlm_config=vlm_config,
            action_expert_config=action_expert_config,
            action_hidden_size=action_hidden_size,
            action_intermediate_size=action_intermediate_size,
            token_budget=token_budget,
            action_vocab_size=action_vocab_size,
            **kwargs,
        )


class SmolVLAParallelDecoding(SmolVLMPreTrainedModel, GenerationMixin):
    """
    SmolVLA Parallel Decoding model for vision-language-action prediction.

    This model implements a dual-path architecture that combines a Vision-Language
    Model (VLM) with an Action Expert for efficient action token generation. Unlike
    autoregressive approaches, parallel decoding generates all action tokens in a
    single forward pass.

    Architecture Overview:
        ┌─────────────────────────────────────────────────────────────┐
        │  VLM (SmolVLM)          Action Expert (Llama)               │
        │  ┌───────────┐          ┌───────────────────┐               │
        │  │ Embedding │          │ Learnable Tokens  │               │
        │  │   +       │          │ (action_embedding)│               │
        │  │ Images    │          └─────────┬─────────┘               │
        │  └─────┬─────┘                    │                         │
        │        │                          │                         │
        │  ┌─────▼──────────────────────────▼─────┐                   │
        │  │     Shared Attention (Layer-wise)     │                  │
        │  │  - VLM causal, Action bidirectional   │                  │
        │  │  - Cross-attention: Action sees VLM   │                  │
        │  └─────┬──────────────────────────┬─────┘                   │
        │        │                          │                         │
        │  ┌─────▼─────┐          ┌─────────▼─────────┐               │
        │  │  VLM      │          │  Action Expert    │               │
        │  │  Output   │          │  LM Head          │               │
        │  └───────────┘          └───────────────────┘               │
        └─────────────────────────────────────────────────────────────┘

    Key Features:
        1. **Shared Attention**: Q/K/V from VLM and Action Expert are concatenated
           for joint attention computation, enabling cross-modal reasoning.

        2. **Dimension Alignment**: Action Expert's hidden_size can differ from VLM,
           but attention dimensions (head_dim, num_heads) are aligned through
           _resize_expert_heads_to_match_vlm().

        3. **Attention Mask Rules**:
           - VLM queries: causal within VLM prefix, cannot see action keys
           - Action queries: bidirectional within action, can see all VLM keys

        4. **Position Encoding Strategies**:
           - Default: continuous positions (0, 1, 2, ..., vlm_len, vlm_len+1, ...)
           - Random stride: random intervals for data augmentation during training
           - Fixed stride: configurable offset for generation

    Training:
        Use forward() with labels for cross-entropy loss on action token prediction.

    Generation:
        Use generate() for single-pass action token generation.

    Example:
        >>> config = SmolVLAParallelDecodingConfig.from_vlm_config(vlm_config)
        >>> model = SmolVLAParallelDecoding(config)
        >>> outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss  # Cross-entropy loss for training
        >>> generated = model.generate(pixel_values=pixel_values, input_ids=input_ids)
    """

    config_class = SmolVLAParallelDecodingConfig

    def __init__(self, config: SmolVLAParallelDecodingConfig, *, init_backbones: bool = True):
        super().__init__(config)
        # Note: in from_pretrained() we first load vlm/action_expert separately, then attach them to main model.
        # To avoid initializing an extra set of same-size parameters here, allow skipping backbone initialization.
        if init_backbones:
            logger.info("Initializing VLM backbone...")
            self.vlm = SmolVLMModel(config.vlm_config)
            logger.info("Initializing Action Expert backbone...")
            self.action_expert = LlamaActionExpert(config.action_expert_config)

            # [New] === Action Expert dimension alignment surgery ===
            self._resize_expert_heads_to_match_vlm()
        else:
            self.vlm = None
            self.action_expert = None
        self.image_token_id = self.config.vlm_config.image_token_id

        # Create trainable embedding, shape (1, token_budget, action_expert.hidden_size)
        # Use smaller initialization values to ensure gradients propagate correctly
        self.action_embedding = nn.Parameter(
            torch.randn(1, config.token_budget, config.action_expert_config.hidden_size) * 0.02
        )

        # action_expert's lm_head, output dimension is action_vocab_size
        self.action_lm_head = nn.Linear(
            config.action_expert_config.hidden_size,
            config.action_vocab_size,
            bias=False,
        )

        self.vocab_size = config.action_vocab_size
        self.token_budget = config.token_budget
        if self.vlm is not None:
            self.vlm.text_model.generation_config = GenerationConfig.from_model_config(
                config.vlm_config
            )

        # Initialize weights and apply final processing
        self.post_init()

    def _resize_expert_heads_to_match_vlm(self):
        """
        Resize Action Expert's Q/K/V projection layers to match VLM's head dimensions.

        This method ensures that even when Action Expert's hidden_size differs from VLM's,
        the attention computation occurs in the same dimensional space. This is critical
        for shared attention where Q/K/V from both models are concatenated.

        Alignment Logic:
            - VLM head_dim = vlm_hidden_size / num_attention_heads
            - Action Expert Q/K/V output dimensions = num_heads * vlm_head_dim
            - Action Expert O_proj input = num_heads * vlm_head_dim (same as Q/K/V output)
            - Action Expert O_proj output = action_expert_hidden_size (back to expert's space)

        The projection path becomes:
            Input (action_hidden_size) -> Q/K/V proj -> (num_heads * vlm_head_dim)
            -> Attention -> (num_heads * vlm_head_dim) -> O_proj -> (action_hidden_size)

        This surgery is performed only when the dimensions don't already match.
        """
        vlm_config = self.vlm.text_model.config
        expert_config = self.action_expert.config

        vlm_head_dim = vlm_config.hidden_size // vlm_config.num_attention_heads

        # Calculate Expert's current default head_dim (based on its own hidden_size)
        expert_default_head_dim = expert_config.hidden_size // expert_config.num_attention_heads

        # If they don't match, Expert was initialized with small size, need to replace projection layers
        if vlm_head_dim != expert_default_head_dim:
            logger.info(
                f"[Architecture] Resizing Expert Projection Layers: {expert_default_head_dim} -> {vlm_head_dim}"
            )

            # Target output dimensions (must match VLM)
            target_q_dim = expert_config.num_attention_heads * vlm_head_dim
            target_kv_dim = expert_config.num_key_value_heads * vlm_head_dim

            for layer in self.action_expert.layers:
                # Replace Q_proj
                # Input: expert_hidden_size (small)
                # Output: num_heads * vlm_head_dim (large)
                layer.self_attn.q_proj = nn.Linear(
                    expert_config.hidden_size, target_q_dim, bias=expert_config.attention_bias
                )

                # Replace K_proj
                layer.self_attn.k_proj = nn.Linear(
                    expert_config.hidden_size, target_kv_dim, bias=expert_config.attention_bias
                )

                # Replace V_proj
                layer.self_attn.v_proj = nn.Linear(
                    expert_config.hidden_size, target_kv_dim, bias=expert_config.attention_bias
                )

                # Replace O_proj (Attention output back to Expert space)
                # Input: num_heads * vlm_head_dim (large)
                # Output: expert_hidden_size (small)
                layer.self_attn.o_proj = nn.Linear(
                    target_q_dim, expert_config.hidden_size, bias=expert_config.attention_bias
                )

                # [Important] Force update layer's internal head_dim record to prevent RoPE calculation errors
                layer.self_attn.head_dim = vlm_head_dim

    def _build_joint_position_ids(
        self,
        *,
        batch_size: int,
        vlm_seq_len: int,
        action_pos_ids: torch.LongTensor,
        device: torch.device,
    ) -> torch.LongTensor:
        """
        Construct joint position IDs for the combined VLM + Action sequence.

        The position IDs determine the rotary position embeddings applied to Q/K vectors.
        This method concatenates fixed VLM positions with caller-provided action positions.

        Args:
            batch_size: Number of sequences in the batch
            vlm_seq_len: Length of the VLM prefix sequence
            action_pos_ids: Position IDs for action tokens, shape (batch_size, action_seq_len)
            device: Target device for tensor creation

        Returns:
            Joint position IDs with shape (batch_size, vlm_seq_len + action_seq_len)
            - VLM positions: [0, 1, 2, ..., vlm_seq_len-1] (fixed, same for all batches)
            - Action positions: provided by caller (may include random/strided offsets)

        Note:
            The action positions can be constructed with different strategies:
            - Random stride: for data augmentation during training
            - Fixed stride: for consistent generation
        """
        if vlm_seq_len <= 0:
            raise ValueError(f"vlm_seq_len must be > 0, current value is {vlm_seq_len}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, current value is {batch_size}")
        if action_pos_ids.dim() != 2 or action_pos_ids.shape[0] != batch_size:
            raise ValueError(
                f"action_pos_ids expected shape (b, L), actual shape is {tuple(action_pos_ids.shape)}"
            )
        if action_pos_ids.dtype != torch.long:
            action_pos_ids = action_pos_ids.to(torch.long)
        action_pos_ids = action_pos_ids.to(device)

        vlm_pos = (
            torch.arange(vlm_seq_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        return torch.cat([vlm_pos, action_pos_ids], dim=1)

    def _build_action_pos_ids_random(
        self,
        *,
        batch_size: int,
        base_pos: int,
        action_seq_len: int,
        device: torch.device,
        random_position_min: int = 1,
        random_position_max: int = 5,
    ) -> torch.LongTensor:
        """
        Build random stride position IDs for action tokens (data augmentation).

        Each token's position is offset by a random integer sampled from [min, max],
        then cumulative sum is applied. This creates variable-spaced positions that
        can help the model generalize to different position encodings during training.

        Args:
            batch_size: Number of sequences in the batch
            base_pos: Starting position (typically vlm_seq_len)
            action_seq_len: Number of action tokens
            device: Target device
            random_position_min: Minimum stride interval (default: 1)
            random_position_max: Maximum stride interval (default: 5)

        Returns:
            Position IDs with shape (batch_size, action_seq_len)
            Example output for base_pos=10, action_seq_len=3:
                [[10+2, 10+2+3, 10+2+3+1],   # random strides: [2,3,1]
                 [10+1, 10+1+4, 10+1+4+2]]   # random strides: [1,4,2]
        """
        steps = torch.randint(
            low=random_position_min,
            high=random_position_max + 1,
            size=(batch_size, action_seq_len),
            device=device,
            dtype=torch.long,
        )
        return base_pos + torch.cumsum(steps, dim=1)

    def _build_action_pos_ids_strided(
        self,
        *,
        batch_size: int,
        base_pos: int,
        action_seq_len: int,
        device: torch.device,
        position_offset: int = 0,
        initial_position_shift: int = 0,
    ) -> torch.LongTensor:
        """
        Build fixed stride position IDs for action tokens.

        Creates evenly-spaced position IDs with a configurable stride. This is
        typically used during generation for consistent position encoding.

        Args:
            batch_size: Number of sequences in the batch
            base_pos: Starting position (typically vlm_seq_len)
            action_seq_len: Number of action tokens
            device: Target device
            position_offset: Stride offset, determines spacing between positions
                - position_offset=0: stride=1, positions are [base, base+1, base+2, ...]
                - position_offset=1: stride=2, positions are [base+1, base+3, base+5, ...]
            initial_position_shift: Additional shift applied to all positions

        Returns:
            Position IDs with shape (batch_size, action_seq_len)

        Example:
            >>> # base_pos=10, action_seq_len=3, position_offset=0
            >>> # Output: [[10, 11, 12], [10, 11, 12], ...]
            >>> # base_pos=10, action_seq_len=3, position_offset=1
            >>> # Output: [[11, 13, 15], [11, 13, 15], ...]
        """
        if action_seq_len <= 0:
            raise ValueError(f"action_seq_len must be > 0, current value is {action_seq_len}")
        if not isinstance(position_offset, int) or isinstance(position_offset, bool):
            raise TypeError(f"position_offset must be int, actual type is {type(position_offset)}")
        if position_offset < 0:
            raise ValueError(f"position_offset must be >= 0, current value is {position_offset}")

        stride = position_offset + 1
        pos = (
            torch.arange(action_seq_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        return base_pos + position_offset + stride * pos + initial_position_shift

    def get_input_embeddings(self):
        return self.vlm.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.vlm.text_model.set_input_embeddings(value)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: Optional[torch.LongTensor] = None,
    ):
        return self.vlm.get_image_features(
            pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load from pretrained model.

        For hub IDs: downloads entire repo to local cache first, then loads like local.
        For local paths: loads directly from subdirectories.

        This simple approach avoids complex subfolder handling.
        """
        import os

        from huggingface_hub import snapshot_download

        logger.info(f"Loading SmolVLAParallelDecoding from {pretrained_model_name_or_path}...")

        # Check if it's a local path or a hub ID
        is_local = os.path.isdir(pretrained_model_name_or_path)

        if is_local:
            local_path = pretrained_model_name_or_path
        else:
            # Extract only valid snapshot_download kwargs
            snapshot_kwargs = {}
            for key in ["token", "use_auth_token", "revision", "repo_type", "cache_dir",
                        "local_dir", "force_download", "proxies", "etag_timeout",
                        "resume_download", "local_files_only"]:
                if key in kwargs:
                    snapshot_kwargs[key] = kwargs.pop(key)
            # Download entire repo to local cache
            local_path = snapshot_download(
                repo_id=pretrained_model_name_or_path,
                **snapshot_kwargs
            )

        # Check for subdirectories
        vlm_path = os.path.join(local_path, "vlm")
        action_expert_path = os.path.join(local_path, "action_expert")
        has_vlm_subfolder = os.path.isdir(vlm_path)
        has_action_expert_subfolder = os.path.isdir(action_expert_path)

        if has_vlm_subfolder and has_action_expert_subfolder:
            logger.info("Found VLM and Action Expert subdirectories. Loading separately...")

            # These parameters are only for this model's wrapper
            token_budget = kwargs.pop("token_budget", 16)
            action_vocab_size = kwargs.pop("action_vocab_size", 2048)
            action_hidden_size = kwargs.pop("action_hidden_size", None)
            action_intermediate_size = kwargs.pop("action_intermediate_size", None)

            # Load VLM config and model
            logger.info(f"Loading VLM config from {vlm_path}...")
            vlm_config = AutoConfig.from_pretrained(vlm_path)
            logger.info("Loading VLM weights... (this may take a while)")
            vlm = SmolVLMModel.from_pretrained(vlm_path, *model_args, **kwargs)

            # Load action_expert config and model
            logger.info(f"Loading Action Expert config from {action_expert_path}...")
            action_expert_config = LlamaActionExpertConfig.from_pretrained(action_expert_path)
            logger.info("Loading Action Expert weights...")
            action_expert = LlamaActionExpert.from_pretrained(
                action_expert_path, *model_args, **kwargs
            )

            # Create main config
            config = SmolVLAParallelDecodingConfig(
                vlm_config=vlm_config,
                action_expert_config=action_expert_config,
                action_hidden_size=action_hidden_size,
                action_intermediate_size=action_intermediate_size,
                token_budget=token_budget,
                action_vocab_size=action_vocab_size,
            )

            # Create model instance
            model = cls(config, init_backbones=False)
            model.vlm = vlm
            model.action_expert = action_expert
            model.vlm.text_model.generation_config = GenerationConfig.from_model_config(
                config.vlm_config
            )

            # Load action_embedding and action_lm_head
            action_components_path = os.path.join(local_path, "action_components.bin")
            if os.path.exists(action_components_path):
                logger.info("Loading action_embedding and action_lm_head...")
                action_components = torch.load(action_components_path, map_location="cpu")

                # Load action_embedding
                if "action_embedding" in action_components:
                    saved_embedding = action_components["action_embedding"]
                    if saved_embedding.shape != model.action_embedding.shape:
                        raise ValueError(
                            f"action_embedding shape mismatch: saved {saved_embedding.shape} != expected {model.action_embedding.shape}"
                        )
                    with torch.no_grad():
                        model.action_embedding.data.copy_(saved_embedding)

                # Load action_lm_head
                if "action_lm_head" in action_components:
                    model.action_lm_head.load_state_dict(action_components["action_lm_head"])
            else:
                logger.warning(
                    "action_components.bin not found, action_embedding and action_lm_head will use random initialization"
                )

            logger.info("Model loading complete.")
            return model
        else:
            # Default: only load vlm, action_expert randomly initialized
            logger.info(
                "No VLM/Action Expert subdirectories found. Loading from single checkpoint..."
            )
            try:
                token_budget = kwargs.pop("token_budget", 16)
                action_vocab_size = kwargs.pop("action_vocab_size", 2048)
                action_hidden_size = kwargs.pop("action_hidden_size", None)
                action_intermediate_size = kwargs.pop("action_intermediate_size", None)

                # Load config
                logger.info(f"Loading config from {local_path}...")
                original_config = AutoConfig.from_pretrained(local_path)

                # Create new config
                config = SmolVLAParallelDecodingConfig.from_vlm_config(
                    original_config,
                    action_hidden_size=action_hidden_size,
                    action_intermediate_size=action_intermediate_size,
                    token_budget=token_budget,
                    action_vocab_size=action_vocab_size,
                )

                # Create model instance
                model = cls(config, init_backbones=False)

                # Load vlm weights
                logger.info(
                    f"Loading SmolVLMForConditionalGeneration from {local_path}... (this may take a while)"
                )
                original_model = SmolVLMForConditionalGeneration.from_pretrained(
                    local_path,
                    *model_args,
                    **kwargs,
                )

                # Extract model attribute
                if hasattr(original_model, "model"):
                    model.vlm = original_model.model
                    model.vlm.text_model.generation_config = GenerationConfig.from_model_config(
                        config.vlm_config
                    )
                else:
                    raise ValueError(
                        "SmolVLMForConditionalGeneration has no 'model' attribute, cannot extract SmolVLMModel"
                    )

                # action_expert keeps random initialization
                logger.info("Initializing Action Expert with random weights...")
                model.action_expert = LlamaActionExpert(config.action_expert_config)
                logger.warning(
                    "action_expert not loaded from pretrained weights, will use random initialization"
                )

                # Perform dimension alignment surgery
                model._resize_expert_heads_to_match_vlm()

                logger.info("Model loading complete.")
                return model
            except Exception as e:
                raise ValueError(f"Cannot load model from {pretrained_model_name_or_path}: {e}")

    def save_pretrained(self, save_directory, **kwargs):
        """Save model, save vlm and action_expert to subfolders separately, and save action_embedding and action_lm_head"""
        import os

        logger.info(f"Saving SmolVLAParallelDecoding to {save_directory}...")
        os.makedirs(save_directory, exist_ok=True)

        # Save main config
        logger.info("Saving main config...")
        self.config.save_pretrained(save_directory)

        # Save vlm to subfolder
        logger.info("Saving VLM...")
        vlm_path = os.path.join(save_directory, "vlm")
        os.makedirs(vlm_path, exist_ok=True)
        self.vlm.save_pretrained(vlm_path, **kwargs)

        # Save action_expert to subfolder
        logger.info("Saving Action Expert...")
        action_expert_path = os.path.join(save_directory, "action_expert")
        os.makedirs(action_expert_path, exist_ok=True)
        self.action_expert.save_pretrained(action_expert_path, **kwargs)

        # Save action_embedding and action_lm_head to separate file
        logger.info("Saving action components...")
        action_components = {
            "action_embedding": self.action_embedding.data.clone(),
            "action_lm_head": self.action_lm_head.state_dict(),
        }
        action_components_path = os.path.join(save_directory, "action_components.bin")
        torch.save(action_components, action_components_path)
        logger.info("Model saved successfully.")

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _build_joint_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        vlm_seq_len: int,
        action_seq_len: int,
        device: torch.device,
        *,
        action_is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Construct the joint 4D additive attention mask for parallel decoding.

        The mask controls which positions can attend to which keys in the joint
        VLM + Action sequence. This is critical for maintaining the parallel
        decoding semantics where action tokens are generated simultaneously.

        Args:
            attention_mask: 2D mask for VLM padding, shape (batch_size, vlm_seq_len)
                - 1 for valid tokens, 0 for padding
            vlm_seq_len: Length of the VLM sequence
            action_seq_len: Number of action tokens
            device: Target device
            action_is_causal: If True, action tokens use causal attention
                (default: False for bidirectional attention within actions)

        Returns:
            4D additive attention mask, shape (batch_size, 1, total_len, total_len)
            - 0.0 for visible positions
            - -1e9 (neg_inf) for masked positions

        Attention Rules (default parallel decoding):
            1. VLM queries:
               - Can only attend to VLM prefix (causal within VLM)
               - Cannot see any action keys

            2. Action queries:
               - Can attend to all VLM keys (full cross-attention to VLM)
               - Within action: bidirectional by default (or causal if action_is_causal=True)

            3. Padding:
               - VLM padding positions (attention_mask==0) are masked as keys
               - Action tokens are always valid (no padding)

        Mask Structure (for total_len = vlm_seq_len + action_seq_len):
            ┌─────────────────┬──────────────┐
            │ VLM causal      │  -inf        │  <- VLM queries
            ├─────────────────┼──────────────┤
            │     0 (visible) │ action mask  │  <- Action queries
            └─────────────────┴──────────────┘
                  VLM keys      Action keys
        """
        if vlm_seq_len <= 0:
            raise ValueError(f"vlm_seq_len must be > 0, current value is {vlm_seq_len}")
        if action_seq_len <= 0:
            raise ValueError(f"action_seq_len must be > 0, current value is {action_seq_len}")

        if attention_mask is None:
            batch_size = 1
        else:
            batch_size = attention_mask.shape[0]

        total_len = vlm_seq_len + action_seq_len
        neg_inf = -1e9

        mask = torch.zeros(
            (batch_size, 1, total_len, total_len), device=device, dtype=torch.float32
        )

        # 1) VLM query cannot see action keys
        mask[:, :, :vlm_seq_len, vlm_seq_len:] = neg_inf

        # 2) VLM internally causal
        causal = torch.triu(
            torch.ones((vlm_seq_len, vlm_seq_len), device=device, dtype=torch.bool), diagonal=1
        )
        mask[:, :, :vlm_seq_len, :vlm_seq_len].masked_fill_(causal, neg_inf)

        # 3) action internally (optional causal)
        if action_is_causal:
            a_causal = torch.triu(
                torch.ones((action_seq_len, action_seq_len), device=device, dtype=torch.bool),
                diagonal=1,
            )
            mask[:, :, vlm_seq_len:, vlm_seq_len:].masked_fill_(a_causal, neg_inf)

        # 4) padding key mask
        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape[1] != vlm_seq_len:
                raise ValueError(
                    f"attention_mask expected shape (b, vlm_seq_len)=({batch_size},{vlm_seq_len}), actual shape is {tuple(attention_mask.shape)}"
                )
            action_key_mask = torch.ones(
                (batch_size, action_seq_len), device=device, dtype=attention_mask.dtype
            )
            key_mask = torch.cat([attention_mask, action_key_mask], dim=1)  # (b, total_len)
            pad_keys = (key_mask == 0).to(mask.dtype)  # (b, total_len)
            mask = mask + pad_keys[:, None, None, :] * neg_inf

        return mask

    def _shared_attention_forward(
        self,
        vlm_hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """
        Execute shared attention computation for one transformer layer.

        This is the core mechanism of parallel decoding: Q/K/V vectors from both
        VLM and Action Expert are concatenated for joint attention, enabling
        cross-modal reasoning while maintaining separate model spaces.

        Process Flow:
            1. Input LayerNorm on VLM and Action hidden states separately
            2. Project to Q/K/V in each model's projection space
            3. Reshape to (batch, heads, seq, head_dim) and concatenate
            4. Apply rotary position embeddings (RoPE) to joint Q/K
            5. Compute joint attention with 4D mask
            6. Split attention output and project back through separate O_proj
            7. Residual connection + MLP (separate for each model)

        Args:
            vlm_hidden_states: VLM hidden states, shape (batch, vlm_seq_len, vlm_hidden_size)
            action_hidden_states: Action hidden states, shape (batch, action_seq_len, action_hidden_size)
            layer_idx: Index of the transformer layer to process
            attention_mask: 4D additive attention mask
            position_ids: Joint position IDs for RoPE
            past_key_values: KV cache (not used in parallel decoding training)
            use_cache: Whether to use KV cache
            cache_position: Current cache position

        Returns:
            Tuple of (vlm_hidden_states, action_hidden_states) after the layer

        Note:
            This method intentionally avoids using transformers' Cache mechanism
            to prevent gradient issues from detach/in-place operations in the
            cache implementation. For training, we compute full attention each time.
        """
        vlm_layer = self.vlm.text_model.layers[layer_idx]
        action_layer = self.action_expert.layers[layer_idx]

        # input layernorm
        vlm_norm = vlm_layer.input_layernorm(vlm_hidden_states)
        action_norm = action_layer.input_layernorm(action_hidden_states)

        # QKV projections
        vlm_q = vlm_layer.self_attn.q_proj(vlm_norm)
        vlm_k = vlm_layer.self_attn.k_proj(vlm_norm)
        vlm_v = vlm_layer.self_attn.v_proj(vlm_norm)

        action_q = action_layer.self_attn.q_proj(action_norm)
        action_k = action_layer.self_attn.k_proj(action_norm)
        action_v = action_layer.self_attn.v_proj(action_norm)

        bsz = vlm_q.shape[0]
        vlm_seq_len = vlm_q.shape[1]
        action_seq_len = action_q.shape[1]
        total_len = vlm_seq_len + action_seq_len

        cfg = self.vlm.text_model.config
        num_heads = cfg.num_attention_heads
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
        head_dim = cfg.hidden_size // num_heads

        # (b, seq, hidden) -> (b, heads, seq, head_dim)
        vlm_q = vlm_q.view(bsz, vlm_seq_len, num_heads, head_dim).transpose(1, 2)
        vlm_k = vlm_k.view(bsz, vlm_seq_len, num_kv_heads, head_dim).transpose(1, 2)
        vlm_v = vlm_v.view(bsz, vlm_seq_len, num_kv_heads, head_dim).transpose(1, 2)

        action_q = action_q.view(bsz, action_seq_len, num_heads, head_dim).transpose(1, 2)
        action_k = action_k.view(bsz, action_seq_len, num_kv_heads, head_dim).transpose(1, 2)
        action_v = action_v.view(bsz, action_seq_len, num_kv_heads, head_dim).transpose(1, 2)

        q = torch.cat([vlm_q, action_q], dim=2)  # (b, heads, total, head_dim)
        k = torch.cat([vlm_k, action_k], dim=2)  # (b, kv_heads, total, head_dim)
        v = torch.cat([vlm_v, action_v], dim=2)  # (b, kv_heads, total, head_dim)

        if position_ids is None:
            position_ids = torch.arange(total_len, device=q.device).unsqueeze(0).expand(bsz, -1)
        if position_ids.shape[1] != total_len:
            raise ValueError(
                f"position_ids length should be {total_len}, actual is {position_ids.shape[1]}"
            )

        # rotary embeddings for joint sequence
        dummy = torch.empty(
            (bsz, total_len, cfg.hidden_size), device=q.device, dtype=vlm_hidden_states.dtype
        )
        cos, sin = self.vlm.text_model.rotary_emb(dummy, position_ids=position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA expand
        if num_kv_heads != num_heads:
            k = repeat_kv(k, num_heads // num_kv_heads)
            v = repeat_kv(v, num_heads // num_kv_heads)

        # attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * (head_dim**-0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask  # broadcast: (b,1,q,k)
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)  # (b, heads, total, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, total_len, -1)

        # split + o_proj
        vlm_attn = vlm_layer.self_attn.o_proj(attn_output[:, :vlm_seq_len])
        action_attn = action_layer.self_attn.o_proj(attn_output[:, vlm_seq_len:])

        # residual 1
        vlm_hidden_states = vlm_hidden_states + vlm_attn
        action_hidden_states = action_hidden_states + action_attn

        # MLP block (residual 2) — match HF LlamaDecoderLayer
        vlm_residual = vlm_hidden_states
        vlm_hidden_states = vlm_layer.post_attention_layernorm(vlm_hidden_states)
        vlm_hidden_states = vlm_residual + vlm_layer.mlp(vlm_hidden_states)

        action_residual = action_hidden_states
        action_hidden_states = action_layer.post_attention_layernorm(action_hidden_states)
        action_hidden_states = action_residual + action_layer.mlp(action_hidden_states)

        return vlm_hidden_states, action_hidden_states

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        random_position_offset: bool = False,
        random_position_min: int = 1,
        random_position_max: int = 5,
        **kwargs,
    ) -> Union[tuple, SmolVLMCausalLMOutputWithPast]:
        """
        Forward pass for training the parallel decoding model.

        Processes the VLM input and generates all action tokens in a single forward
        pass. If labels are provided, computes cross-entropy loss for training.

        Args:
            input_ids: Input token IDs for VLM, shape (batch_size, seq_len)
            attention_mask: Attention mask for VLM input, 1 for valid, 0 for padding
            position_ids: Position IDs (usually auto-generated)
            past_key_values: KV cache (not used in parallel decoding)
            inputs_embeds: Pre-computed input embeddings
            pixel_values: Image pixel values for vision encoder
            pixel_attention_mask: Attention mask for images
            image_hidden_states: Pre-computed image hidden states
            labels: Target action token IDs, shape (batch_size, token_budget)
            use_cache: Whether to use KV cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            cache_position: Cache position for incremental decoding
            return_dict: Whether to return a ModelOutput
            logits_to_keep: Number of logits to keep (optimization)
            random_position_offset: Enable random stride position encoding (data augmentation)
            random_position_min: Minimum random stride interval
            random_position_max: Maximum random stride interval

        Returns:
            SmolVLMCausalLMOutputWithPast containing:
                - loss: Cross-entropy loss if labels provided, else None
                - logits: Action token logits, shape (batch, token_budget, action_vocab_size)
                - past_key_values: None (not used in parallel decoding)
                - hidden_states: Action hidden states if output_hidden_states=True
                - image_hidden_states: Image features from VLM

        Example:
            >>> outputs = model(
            ...     pixel_values=images,
            ...     input_ids=input_ids,
            ...     labels=action_labels,
            ...     random_position_offset=True,  # Data augmentation
            ... )
            >>> loss = outputs.loss
            >>> logits = outputs.logits  # (batch, token_budget, vocab_size)
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.vlm_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.vlm_config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.vlm_config.use_return_dict
        )

        # -----------------------------
        # 1) Construct VLM's inputs_embeds (strictly align with SmolVLMModel.forward approach)
        # -----------------------------
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("You must provide either input_ids or inputs_embeds")

        if input_ids is not None:
            batch_size, vlm_seq_len = input_ids.shape
        else:
            batch_size, vlm_seq_len, _ = inputs_embeds.shape

        if inputs_embeds is None:
            vlm_inputs_embeds = self.vlm.text_model.get_input_embeddings()(input_ids).to(
                device=input_ids.device, dtype=self.vlm.text_model.dtype
            )
        else:
            vlm_inputs_embeds = inputs_embeds

        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states")

        if pixel_values is not None:
            if input_ids is None:
                raise ValueError(
                    "When using pixel_values, must provide input_ids (for locating image_token_id and inputs_merger)"
                )
            image_hidden_states = self.vlm.get_image_features(
                pixel_values, pixel_attention_mask
            ).to(vlm_inputs_embeds.device)
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(
                dtype=self.vlm.dtype, device=vlm_inputs_embeds.device
            )

        if image_hidden_states is not None:
            # Key: SmolVLM uses inputs_merger to replace image features at image_token position (doesn't change seq_len)
            vlm_inputs_embeds = self.vlm.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=vlm_inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        # -----------------------------
        # 2) action expert input (learnable tokens)
        # -----------------------------
        action_inputs_embeds = self.action_embedding.expand(batch_size, -1, -1)
        action_seq_len = action_inputs_embeds.shape[1]

        # -----------------------------
        # 3) Construct position_ids (optional augmentation)
        # -----------------------------
        total_seq_len = vlm_seq_len + action_seq_len
        base_pos = int(vlm_seq_len)

        if not isinstance(random_position_offset, bool):
            raise TypeError(
                f"random_position_offset must be bool, actual type is {type(random_position_offset)}"
            )

        if random_position_offset:
            # Construct random stride position_ids for action expert
            action_pos_ids = self._build_action_pos_ids_random(
                batch_size=batch_size,
                base_pos=base_pos,
                action_seq_len=action_seq_len,
                device=vlm_inputs_embeds.device,
                random_position_min=random_position_min,
                random_position_max=random_position_max,
            )
            position_ids = self._build_joint_position_ids(
                batch_size=batch_size,
                vlm_seq_len=vlm_seq_len,
                action_pos_ids=action_pos_ids,
                device=vlm_inputs_embeds.device,
            )
        else:
            # Default: continuous position_ids
            position_ids = (
                torch.arange(total_seq_len, device=vlm_inputs_embeds.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        # joint 4D mask: VLM causal + action bidirectional (default)
        attention_mask_4d = self._build_joint_attention_mask(
            attention_mask=attention_mask,
            vlm_seq_len=vlm_seq_len,
            action_seq_len=action_seq_len,
            device=vlm_inputs_embeds.device,
            action_is_causal=False,
        )

        # -----------------------------
        # 4) Forward layer by layer (shared attention)
        # -----------------------------
        vlm_hidden_states = vlm_inputs_embeds
        action_hidden_states = action_inputs_embeds

        num_layers = self.config.vlm_config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            vlm_hidden_states, action_hidden_states = self._shared_attention_forward(
                vlm_hidden_states=vlm_hidden_states,
                action_hidden_states=action_hidden_states,
                layer_idx=layer_idx,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        # Final norm
        vlm_hidden_states = self.vlm.text_model.norm(vlm_hidden_states)
        action_hidden_states = self.action_expert.norm(action_hidden_states)

        # Get logits through action_expert's lm_head
        action_logits = self.action_lm_head(action_hidden_states)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                action_logits.reshape(-1, action_logits.shape[-1]), labels.reshape(-1)
            )

        past_key_values_output = None

        return SmolVLMCausalLMOutputWithPast(
            loss=loss,
            logits=action_logits,
            past_key_values=past_key_values_output,
            hidden_states=action_hidden_states if output_hidden_states else None,
            attentions=None,  # Don't return attention for now
            image_hidden_states=image_hidden_states,  # Pass through, for external reuse
        )

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        position_offset: int = 0,
        initial_position_shift: int = 0,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generate action tokens using parallel decoding.

        Performs a single forward pass to generate all action tokens simultaneously.
        This is more efficient than autoregressive generation for fixed-length
        action sequences.

        Args:
            input_ids: Input token IDs for VLM
            attention_mask: Attention mask for VLM input
            position_ids: Position IDs (auto-generated if None)
            past_key_values: KV cache (not used)
            inputs_embeds: Pre-computed input embeddings
            pixel_values: Image pixel values for vision encoder
            pixel_attention_mask: Attention mask for images
            image_hidden_states: Pre-computed image hidden states
            use_cache: Whether to use KV cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            cache_position: Cache position
            return_dict: Whether to return dict
            position_offset: Stride offset for position encoding
                - position_offset=0: positions [base, base+1, base+2, ...]
                - position_offset=1: positions [base+1, base+3, base+5, ...]
            initial_position_shift: Additional shift for all positions

        Returns:
            Generated action token IDs, shape (batch_size, token_budget)

        Example:
            >>> action_tokens = model.generate(
            ...     pixel_values=images,
            ...     input_ids=input_ids,
            ...     position_offset=0,
            ... )
            >>> # action_tokens: (batch_size, token_budget)
        """
        _ = (
            position_ids,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            cache_position,
            return_dict,
            kwargs,
        )

        # -----------------------------
        # 1) Construct VLM's inputs_embeds
        # -----------------------------
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("You must provide either input_ids or inputs_embeds")

        if input_ids is not None:
            batch_size, vlm_seq_len = input_ids.shape
        else:
            batch_size, vlm_seq_len, _ = inputs_embeds.shape

        if inputs_embeds is None:
            vlm_inputs_embeds = self.vlm.text_model.get_input_embeddings()(input_ids).to(
                device=input_ids.device, dtype=self.vlm.text_model.dtype
            )
        else:
            vlm_inputs_embeds = inputs_embeds

        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states")

        if pixel_values is not None:
            if input_ids is None:
                raise ValueError(
                    "When using pixel_values, must provide input_ids (for locating image_token_id and inputs_merger)"
                )
            image_hidden_states = self.vlm.get_image_features(
                pixel_values, pixel_attention_mask
            ).to(vlm_inputs_embeds.device)
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(
                dtype=self.vlm.dtype, device=vlm_inputs_embeds.device
            )

        if image_hidden_states is not None:
            vlm_inputs_embeds = self.vlm.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=vlm_inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        device = vlm_inputs_embeds.device
        base_pos = int(vlm_seq_len)

        # -----------------------------
        # 2) Validate position_offset
        # -----------------------------
        if not isinstance(position_offset, int) or isinstance(position_offset, bool):
            raise TypeError(f"position_offset must be int, actual type is {type(position_offset)}")
        if position_offset < 0:
            raise ValueError(f"position_offset must be >= 0, current value is {position_offset}")

        # -----------------------------
        # 3) Construct position_ids
        # -----------------------------
        action_seq_len = self.token_budget

        action_pos_ids = self._build_action_pos_ids_strided(
            batch_size=batch_size,
            base_pos=base_pos,
            action_seq_len=action_seq_len,
            device=device,
            position_offset=position_offset,
            initial_position_shift=initial_position_shift,
        )
        position_ids_joint = self._build_joint_position_ids(
            batch_size=batch_size,
            vlm_seq_len=vlm_seq_len,
            action_pos_ids=action_pos_ids,
            device=device,
        )

        # -----------------------------
        # 4) action expert input
        # -----------------------------
        action_inputs_embeds = self.action_embedding.expand(batch_size, -1, -1)

        # joint 4D mask
        attention_mask_4d = self._build_joint_attention_mask(
            attention_mask=attention_mask,
            vlm_seq_len=vlm_seq_len,
            action_seq_len=action_seq_len,
            device=device,
            action_is_causal=False,
        )

        # -----------------------------
        # 5) Forward layer by layer
        # -----------------------------
        vlm_hidden_states = vlm_inputs_embeds
        action_hidden_states = action_inputs_embeds

        num_layers = self.config.vlm_config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            vlm_hidden_states, action_hidden_states = self._shared_attention_forward(
                vlm_hidden_states=vlm_hidden_states,
                action_hidden_states=action_hidden_states,
                layer_idx=layer_idx,
                attention_mask=attention_mask_4d,
                position_ids=position_ids_joint,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
            )

        # Final norm
        action_hidden_states = self.action_expert.norm(action_hidden_states)

        # logits -> argmax
        action_logits = self.action_lm_head(action_hidden_states)
        generated = torch.argmax(action_logits, dim=-1)

        return generated

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        pixel_values=None,
        pixel_attention_mask=None,
        image_hidden_states=None,
        logits_to_keep=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Call vlm's prepare_inputs_for_generation
        model_inputs = self.vlm.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if image_hidden_states is not None or not is_first_iteration:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_attention_mask"] = None

        return model_inputs


AutoConfig.register("smolvla_parallel_decoding", SmolVLAParallelDecodingConfig)
AutoModel.register(SmolVLAParallelDecodingConfig, SmolVLAParallelDecoding)
