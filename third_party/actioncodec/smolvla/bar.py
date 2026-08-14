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
    Configuration for LlamaActionExpert model in Blockwise AR architecture.

    The Action Expert in Blockwise AR architecture processes discrete action tokens
    using a blockwise autoregressive approach. Unlike parallel decoding where all
    tokens are generated simultaneously, blockwise AR generates tokens in fixed-size
    blocks with bidirectional attention within each block.

    Key Parameters:
        vocab_size: Action vocabulary size
        hidden_size: Hidden dimension (can differ from VLM)
        intermediate_size: MLP intermediate dimension
        num_hidden_layers: Number of transformer layers (matches VLM)
        num_attention_heads: Number of attention heads (must match VLM)
        num_key_value_heads: Number of KV heads for GQA (must match VLM)
        head_dim: Dimension per attention head (set to match VLM)

    Note:
        The hidden_size can be smaller than VLM's for efficiency, but attention
        dimensions are aligned for shared attention computation.
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

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        # Bidirectional attention must use eager implementation
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
    Llama-based Action Expert model for Blockwise AR architecture.

    This model extends LlamaModel to serve as a blockwise autoregressive action
    token predictor. It processes action tokens in blocks, enabling bidirectional
    attention within blocks while maintaining causal dependency across blocks.

    Architecture:
        - Standard Llama decoder layers with configurable hidden_size
        - Supports GQA (Grouped Query Attention)
        - Designed for blockwise attention patterns
        - RoPE (Rotary Position Embeddings) for position encoding

    The model works in conjunction with VLM through shared attention, generating
    action tokens block by block where each block can attend to all previous blocks.
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
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = (
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
            last_hidden_state=hidden_states, past_key_values=past_key_values
        )


class SmolVLABlockwiseARConfig(PretrainedConfig):
    """
    Configuration for SmolVLABlockwiseAR model.

    Blockwise AR generates action tokens in fixed-size blocks rather than all at
    once (parallel decoding) or one at a time (standard AR). Each block uses
    bidirectional attention internally, while maintaining causal order across blocks.

    Configuration Flow:
        1. Create from existing VLM config using from_vlm_config()
        2. Action Expert config auto-generated with aligned attention dimensions
        3. Block structure determined by token_budget and num_blocks

    Key Parameters:
        vlm_config: Configuration for the VLM (SmolVLM)
        action_expert_config: Configuration for the Action Expert
        token_budget: Total number of action tokens (default: 48)
        num_blocks: Number of blocks to divide tokens into (default: 3)
        action_vocab_size: Size of action vocabulary (default: 2048)
        tie_action_embeddings: Share input/output embeddings
        freeze_action_embeddings: Freeze embedding weights

    Block Structure:
        - block_size = token_budget / num_blocks
        - Each block contains block_size tokens
        - Within block: bidirectional attention
        - Across blocks: causal attention (block i can see blocks 0..i-1)

    Example:
        >>> # token_budget=48, num_blocks=3 -> block_size=16
        >>> # Block 0: tokens 0-15
        >>> # Block 1: tokens 16-31 (can see block 0)
        >>> # Block 2: tokens 32-47 (can see blocks 0-1)
    """

    model_type = "smolvla_blockwise_ar"

    def __init__(
        self,
        vlm_config=None,
        action_expert_config=None,
        action_hidden_size: int = None,
        action_intermediate_size: int = None,
        token_budget: int = 16,
        num_blocks: int = 4,
        action_vocab_size: int = 2048,
        tie_action_embeddings: bool = False,
        freeze_action_embeddings: bool = False,
        **kwargs,
    ):
        # Handle dict inputs for nested configs (from JSON deserialization)
        if isinstance(vlm_config, dict):
            vlm_config = SmolVLMConfig(**vlm_config)
        if isinstance(action_expert_config, dict):
            action_expert_config = LlamaActionExpertConfig(**action_expert_config)

        super().__init__(**kwargs)
        if token_budget <= 0:
            raise ValueError(f"token_budget must be > 0, current value is {token_budget}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, current value is {num_blocks}")
        if token_budget % num_blocks != 0:
            raise ValueError(
                f"token_budget({token_budget}) must be divisible by num_blocks({num_blocks})"
            )

        self.vlm_config = vlm_config
        self.action_expert_config = action_expert_config
        self.token_budget = token_budget
        self.num_blocks = num_blocks
        self.action_vocab_size = action_vocab_size
        self.tie_action_embeddings = tie_action_embeddings
        self.freeze_action_embeddings = freeze_action_embeddings
        self.action_hidden_size = action_hidden_size
        self.action_intermediate_size = action_intermediate_size

        if vlm_config is not None:
            if hasattr(vlm_config, "text_config") and hasattr(
                vlm_config.text_config, "initializer_range"
            ):
                self.initializer_range = vlm_config.text_config.initializer_range
            elif hasattr(vlm_config, "initializer_range"):
                self.initializer_range = vlm_config.initializer_range
            else:
                self.initializer_range = 0.02
        else:
            self.initializer_range = 0.02

    @classmethod
    def from_vlm_config(
        cls,
        vlm_config,
        *,
        action_hidden_size: int = None,
        action_intermediate_size: int = None,
        token_budget: int = 48,
        num_blocks: int = 3,
        action_vocab_size: int = 2048,
        tie_action_embeddings: bool = False,
        freeze_action_embeddings: bool = False,
        **kwargs,
    ):
        text_config = vlm_config.text_config if hasattr(vlm_config, "text_config") else vlm_config

        # Get VLM parameters
        vlm_hidden_size = getattr(text_config, "hidden_size", None)
        vlm_num_heads = getattr(text_config, "num_attention_heads", None)
        vlm_num_kv_heads = getattr(text_config, "num_key_value_heads", vlm_num_heads)

        # [Key change] Expert configuration logic
        # If action_hidden_size not specified, default to VLM's 1/4 or other ratio, or allow user to specify
        if action_hidden_size is None:
            action_hidden_size = (
                vlm_hidden_size // 2
            )  # Default to half of VLM's hidden size for efficiency, can be adjusted as needed

        if action_intermediate_size is None:
            # Llama typically uses 4 * hidden_size * 2/3, simplified here
            action_intermediate_size = action_hidden_size * 4

        # Calculate VLM's head_dim (this is the anchor point for Attention alignment)
        # Note: we only record this here, will use it to replace layers in Model init
        vlm_head_dim = vlm_hidden_size // vlm_num_heads

        action_expert_config = LlamaActionExpertConfig(
            vocab_size=action_vocab_size,
            hidden_size=action_hidden_size,
            intermediate_size=action_intermediate_size,
            num_hidden_layers=getattr(
                text_config, "num_hidden_layers", 12
            ),  # Number of layers typically kept consistent for one-to-one correspondence
            num_attention_heads=vlm_num_heads,  # [Must be consistent]
            num_key_value_heads=vlm_num_kv_heads,  # [Must be consistent]
            # Pass head_dim mainly for recording, LlamaModel may not recognize it by default, we fix in Model init
            head_dim=vlm_head_dim,
            # ... other parameters kept consistent ...
            hidden_act=getattr(text_config, "hidden_act", "silu"),
            max_position_embeddings=getattr(text_config, "max_position_embeddings", 2048),
            initializer_range=getattr(text_config, "initializer_range", 0.02),
            rms_norm_eps=getattr(text_config, "rms_norm_eps", 1e-5),
            use_cache=False,
            _attn_implementation="eager",
        )

        return cls(
            vlm_config=vlm_config,
            action_expert_config=action_expert_config,
            action_hidden_size=action_hidden_size,
            action_intermediate_size=action_intermediate_size,
            token_budget=token_budget,
            num_blocks=num_blocks,
            action_vocab_size=action_vocab_size,
            tie_action_embeddings=tie_action_embeddings,
            freeze_action_embeddings=freeze_action_embeddings,
            **kwargs,
        )


class SmolVLABlockwiseAR(SmolVLMPreTrainedModel, GenerationMixin):
    """
    SmolVLA Blockwise Autoregressive model for action token generation.

    This model implements blockwise autoregressive generation, a middle ground
    between parallel decoding (all tokens at once) and standard AR (one token
    at a time). Action tokens are generated in fixed-size blocks with bidirectional
    attention within each block.

    Architecture Overview:
        ┌─────────────────────────────────────────────────────────────────┐
        │  Input Sequence (per block):                                    │
        │  ┌─────────┬────────────────┬────────────────┬────────────────┐ │
        │  │  VLM    │   n BOS        │   Block 0      │   Block 1      │ │
        │  │  Prefix │   tokens       │   (16 tokens)  │   (16 tokens)  │ │
        │  └─────────┴────────────────┴────────────────┴────────────────┘ │
        │                                                                 │
        │  Blockwise Attention Mask:                                      │
        │  ┌────────────────────────────────────────────────────────────┐ │
        │  │   VLM   │  BOS  │ Blk0  │ Blk1  │ Blk2  │                  │ │
        │  │ ────────┼───────┼───────┼───────┼───────┤                  │ │
        │  │ VLM: Causal, no action keys                                │ │
        │  │ Action: Bidirectional within block, causal across blocks   │ │
        │  └────────────────────────────────────────────────────────────┘ │
        └─────────────────────────────────────────────────────────────────┘

    Key Features:
        1. **Blockwise Attention**:
           - Within block: bidirectional (all positions can see each other)
           - Across blocks: causal (block i can see blocks 0..i-1)
           - Enables modeling local dependencies while maintaining global order

        2. **BOS Prefix**: Each action sequence starts with n BOS tokens
           (n = block_size) as context for the first block.

        3. **Block Shift Prediction**:
           - Input: [BOS*n] + [Block 0] + ... + [Block k-1]
           - Output: Predicts [Block 0] + ... + [Block k]

        4. **Sampling Strategies**:
           - Greedy: argmax for each block
           - Top-k: Sample from top-k logits
           - Top-p (Nucleus): Sample from cumulative probability threshold
           - Temperature: Control randomness

    Training:
        Use forward() with labels for cross-entropy loss. Labels should be
        action token IDs with shape (batch, token_budget).

    Generation:
        Use generate() with optional sampling parameters.

    Example:
        >>> config = SmolVLABlockwiseARConfig.from_vlm_config(
        ...     vlm_config, token_budget=48, num_blocks=3
        ... )
        >>> model = SmolVLABlockwiseAR(config)
        >>> # Training
        >>> outputs = model(pixel_values=images, input_ids=input_ids, labels=action_labels)
        >>> loss = outputs.loss
        >>> # Generation (greedy)
        >>> tokens = model.generate(pixel_values=images, input_ids=input_ids)
        >>> # Generation (sampling)
        >>> tokens = model.generate(
        ...     pixel_values=images, input_ids=input_ids,
        ...     do_sample=True, temperature=0.8, top_k=50, top_p=0.95
        ... )
    """

    config_class = SmolVLABlockwiseARConfig

    def __init__(
        self, config: SmolVLABlockwiseARConfig, *, init_backbones: bool = True
    ):  # 1. Add parameter
        super().__init__(config)
        if config.token_budget % config.num_blocks != 0:
            raise ValueError(
                f"token_budget({config.token_budget}) must be divisible by num_blocks({config.num_blocks})"
            )

        # 2. Add conditional logic
        if init_backbones:
            self.vlm = SmolVLMModel(config.vlm_config)
            self.action_expert = LlamaActionExpert(config.action_expert_config)

            # [New] === Action Expert dimension alignment surgery ===
            self._resize_expert_heads_to_match_vlm()
        else:
            self.vlm = None
            self.action_expert = None

        self.image_token_id = self.config.vlm_config.image_token_id

        self.token_budget = config.token_budget
        self.num_blocks = config.num_blocks
        # n = token_budget / num_blocks
        self.block_size = config.token_budget // config.num_blocks

        self.action_vocab_size = config.action_vocab_size

        # - action token id: [0, action_vocab_size)
        # - BOS is not part of output, so use a separate trainable vector (repeated n times as input prefix)
        self.action_token_embedding = nn.Embedding(
            self.action_vocab_size, config.action_expert_config.hidden_size
        )
        self.bos_embedding = nn.Parameter(
            torch.randn(1, 1, config.action_expert_config.hidden_size) * 0.02
        )

        self.action_lm_head = nn.Linear(
            config.action_expert_config.hidden_size, self.action_vocab_size, bias=False
        )

        # Optional: share weights between input action embedding and output head (common LLM practice)
        if getattr(config, "tie_action_embeddings", False):
            self.action_lm_head.weight = self.action_token_embedding.weight

        # Optional: freeze action embedding (including lm_head.weight in tying scenario)
        if getattr(config, "freeze_action_embeddings", False):
            self.action_token_embedding.weight.requires_grad_(False)

        if self.vlm is not None:
            self.vlm.text_model.generation_config = GenerationConfig.from_model_config(
                config.vlm_config
            )

        self.post_init()

    def _resize_expert_heads_to_match_vlm(self):
        """
        Resize Action Expert's Q/K/V projection layers to match VLM's head dimensions.

        This ensures that even when Action Expert's hidden_size differs from VLM's,
        the attention computation occurs in the same dimensional space for shared attention.

        Alignment Logic:
            - VLM head_dim = vlm_hidden_size / num_attention_heads
            - Action Expert Q/K/V output dimensions = num_heads * vlm_head_dim
            - Action Expert O_proj output = action_expert_hidden_size

        This is called automatically during model initialization when dimensions differ.
        """
        vlm_config = self.vlm.text_model.config
        expert_config = self.action_expert.config

        vlm_head_dim = vlm_config.hidden_size // vlm_config.num_attention_heads

        # Calculate Expert's current default head_dim (based on its own hidden_size)
        expert_default_head_dim = expert_config.hidden_size // expert_config.num_attention_heads

        # If they don't match, Expert was initialized with small size, need to replace projection layers
        if vlm_head_dim != expert_default_head_dim:
            print(
                f"\033[96m[Architecture] Resizing Expert Projection Layers: {expert_default_head_dim} -> {vlm_head_dim}\033[0m"
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
        Construct joint position_ids: (b, vlm_seq_len + action_seq_len)
        - VLM part is fixed to [0..vlm_seq_len-1]
        - action part is provided by caller (only affects action expert)
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
        """action side position_ids: sample interval in [min,max] for each token and do cumulative sum, then add to base_pos."""
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
    ) -> torch.LongTensor:
        """
        action side position_ids: fixed stride (position_offset+1).
        - position_offset=0 => base, base+1, base+2, ...
        - position_offset=1 => base+1, base+3, base+5, ...
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
        return base_pos + position_offset + stride * pos

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

        logger.info(f"Loading SmolVLABlockwiseAR from {pretrained_model_name_or_path}...")

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

            # Load VLM
            logger.info(f"Loading VLM config from {vlm_path}...")
            vlm_config = AutoConfig.from_pretrained(vlm_path)
            logger.info("Loading VLM weights... (this may take a while)")
            vlm = SmolVLMModel.from_pretrained(vlm_path)

            # Load Action Expert
            logger.info(f"Loading Action Expert config from {action_expert_path}...")
            action_expert_config = LlamaActionExpertConfig.from_pretrained(action_expert_path)
            logger.info("Loading Action Expert weights...")
            action_expert = LlamaActionExpert.from_pretrained(action_expert_path)

            config = SmolVLABlockwiseARConfig(
                vlm_config=vlm_config,
                action_expert_config=action_expert_config,
                token_budget=kwargs.get("token_budget", 16),
                num_blocks=kwargs.get("num_blocks", 4),
                action_vocab_size=kwargs.get("action_vocab_size", 2048),
            )

            # Pass init_backbones=False to only create empty shell and bar components, not backbone
            model = cls(config, init_backbones=False)

            # Assign loaded models
            model.vlm = vlm
            model.action_expert = action_expert

            # Load action_components.bin
            action_components_path = os.path.join(local_path, "action_components.bin")
            if os.path.exists(action_components_path):
                logger.info("Loading action_token_embedding/bos_embedding and action_lm_head...")
                action_components = torch.load(action_components_path, map_location="cpu")

                if "action_token_embedding" in action_components:
                    try:
                        model.action_token_embedding.load_state_dict(
                            action_components["action_token_embedding"]
                        )
                    except Exception as e:
                        logger.warning(
                            f"action_token_embedding load failed (possibly old checkpoint dimension mismatch): {e}"
                        )
                if "bos_embedding" in action_components:
                    saved_bos = action_components["bos_embedding"]
                    if saved_bos.shape != model.bos_embedding.shape:
                        logger.warning(
                            f"bos_embedding shape mismatch, skipping load: saved={tuple(saved_bos.shape)} expect={tuple(model.bos_embedding.shape)}"
                        )
                    else:
                        with torch.no_grad():
                            model.bos_embedding.data.copy_(saved_bos)
                if "action_lm_head" in action_components:
                    try:
                        model.action_lm_head.load_state_dict(action_components["action_lm_head"])
                    except Exception as e:
                        logger.warning(
                            f"action_lm_head load failed (possibly old checkpoint dimension mismatch): {e}"
                        )
            else:
                logger.warning(
                    "action_components.bin not found, bar components will use random initialization"
                )

            logger.info("Model loading complete.")
            return model

        # fallback: only load vlm (and derive action_expert_config from vlm_config)
        logger.info("No VLM/Action Expert subdirectories found. Loading from single checkpoint...")
        try:
            logger.info(f"Loading config from {local_path}...")
            original_config = AutoConfig.from_pretrained(local_path)
            config = SmolVLABlockwiseARConfig.from_vlm_config(
                original_config,
                token_budget=kwargs.pop("token_budget", 48),
                num_blocks=kwargs.pop("num_blocks", 3),
                action_vocab_size=kwargs.pop("action_vocab_size", 2048),
                action_hidden_size=kwargs.pop("action_hidden_size", None),
                action_intermediate_size=kwargs.pop("action_intermediate_size", None),
            )

            # Pass init_backbones=False
            model = cls(config, init_backbones=False)

            logger.info(
                f"Loading SmolVLMForConditionalGeneration from {local_path}... (this may take a while)"
            )
            original_model = SmolVLMForConditionalGeneration.from_pretrained(
                local_path,
                *model_args,
                **kwargs,
            )
            if hasattr(original_model, "model"):
                model.vlm = original_model.model
            else:
                raise ValueError(
                    "SmolVLMForConditionalGeneration has no 'model' attribute, cannot extract SmolVLMModel"
                )

            # Must manually initialize action_expert because init_backbones=False skipped it
            logger.info("Initializing Action Expert...")
            model.action_expert = LlamaActionExpert(config.action_expert_config)

            logger.warning(
                "action_expert not loaded from pretrained weights, will use random initialization"
            )
            return model
        except Exception as e:
            raise ValueError(f"Cannot load model from {local_path}: {e}")

    def save_pretrained(self, save_directory, **kwargs):
        """Save model: vlm/action_expert in separate directories, bar components saved separately"""
        import os

        logger.info(f"Saving SmolVLABlockwiseAR to {save_directory}...")
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)

        logger.info("Saving VLM...")
        vlm_path = os.path.join(save_directory, "vlm")
        os.makedirs(vlm_path, exist_ok=True)
        self.vlm.save_pretrained(vlm_path, **kwargs)

        logger.info("Saving Action Expert...")
        action_expert_path = os.path.join(save_directory, "action_expert")
        os.makedirs(action_expert_path, exist_ok=True)
        self.action_expert.save_pretrained(action_expert_path, **kwargs)

        logger.info("Saving action components...")
        action_components = {
            "action_token_embedding": self.action_token_embedding.state_dict(),
            "bos_embedding": self.bos_embedding.data.clone(),
            "action_lm_head": self.action_lm_head.state_dict(),
        }
        torch.save(action_components, os.path.join(save_directory, "action_components.bin"))
        logger.info("Model saved successfully.")

    def _build_joint_attention_mask_blockwise_ar(
        self,
        attention_mask: Optional[torch.Tensor],
        vlm_seq_len: int,
        action_seq_len: int,
        device: torch.device,
        *,
        action_key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Construct the joint 4D attention mask for blockwise autoregressive generation.

        This mask implements the blockwise attention pattern where:
        - Within each block: bidirectional attention (all positions see each other)
        - Across blocks: causal attention (block i can see blocks 0..i-1)

        Args:
            attention_mask: 2D mask for VLM padding, shape (batch_size, vlm_seq_len)
            vlm_seq_len: Length of the VLM sequence
            action_seq_len: Total length of action sequence (BOS + action tokens)
            device: Target device
            action_key_mask: Mask for action keys, shape (batch, action_seq_len)

        Returns:
            4D additive attention mask, shape (batch_size, 1, total_len, total_len)

        Attention Rules:
            1. VLM queries:
               - Causal within VLM prefix
               - Cannot see any action keys

            2. Action queries:
               - Can attend to all VLM keys (full cross-attention)
               - Within same block: bidirectional
               - Across blocks: causal (can only see current and earlier blocks)

            3. Padding positions are masked as keys (set to -inf)

        Mask Structure Visualization (3 blocks, n tokens per block):
            Action Block IDs: [0,0,..0, 1,1,..1, 2,2,..2]
                              └─block0─┘ └─block1─┘ └─block2─┘

            For query in block 1:
            - Can see: VLM, BOS, block0, block1 (all tokens)
            - Cannot see: block2 (future block)
        """
        if vlm_seq_len <= 0:
            raise ValueError(f"vlm_seq_len must be > 0, current value is {vlm_seq_len}")
        if action_seq_len <= 0:
            raise ValueError(f"action_seq_len must be > 0, current value is {action_seq_len}")

        # batch_size inference:
        # - Prefer action_key_mask (because generate/certain calls may not pass attention_mask)
        # - Then use attention_mask
        # - Otherwise default to 1
        if action_key_mask is not None:
            if action_key_mask.dim() != 2 or action_key_mask.shape[1] != action_seq_len:
                raise ValueError(
                    f"action_key_mask expected shape (b, action_seq_len)=(b,{action_seq_len}), actual shape is {tuple(action_key_mask.shape)}"
                )
            batch_size = action_key_mask.shape[0]
        elif attention_mask is not None:
            batch_size = attention_mask.shape[0]
        else:
            batch_size = 1

        total_len = vlm_seq_len + action_seq_len
        neg_inf = -1e9
        mask = torch.zeros(
            (batch_size, 1, total_len, total_len), device=device, dtype=torch.float32
        )

        mask[:, :, :vlm_seq_len, vlm_seq_len:] = neg_inf

        causal = torch.triu(
            torch.ones((vlm_seq_len, vlm_seq_len), device=device, dtype=torch.bool), diagonal=1
        )
        mask[:, :, :vlm_seq_len, :vlm_seq_len].masked_fill_(causal, neg_inf)

        pos = torch.arange(action_seq_len, device=device)
        block_ids = pos // self.block_size

        q_block = block_ids.view(-1, 1)  # (aq, 1)
        k_block = block_ids.view(1, -1)  # (1, ak)
        block_future = k_block > q_block  # key block after query block => mask
        mask[:, :, vlm_seq_len:, vlm_seq_len:].masked_fill_(block_future, neg_inf)

        # 4) padding key mask（VLM + action）
        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape[1] != vlm_seq_len:
                raise ValueError(
                    f"attention_mask expected shape (b, vlm_seq_len)=({batch_size},{vlm_seq_len}), actual shape is {tuple(attention_mask.shape)}"
                )
        if action_key_mask is None:
            action_key_mask = torch.ones(
                (batch_size, action_seq_len), device=device, dtype=torch.long
            )
        else:
            if (
                action_key_mask.dim() != 2
                or action_key_mask.shape[0] != batch_size
                or action_key_mask.shape[1] != action_seq_len
            ):
                raise ValueError(
                    f"action_key_mask expected shape (b, action_seq_len)=({batch_size},{action_seq_len}), actual shape is {tuple(action_key_mask.shape)}"
                )

        if attention_mask is not None:
            key_mask = torch.cat([attention_mask.to(action_key_mask.dtype), action_key_mask], dim=1)
        else:
            key_mask = torch.cat(
                [
                    torch.ones(
                        (batch_size, vlm_seq_len), device=device, dtype=action_key_mask.dtype
                    ),
                    action_key_mask,
                ],
                dim=1,
            )

        pad_keys = (key_mask == 0).to(mask.dtype)  # (b, total)
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

        Similar to the parallel decoding version, but used with blockwise AR attention
        masks. Q/K/V vectors from both VLM and Action Expert are concatenated for
        joint attention computation.

        Args:
            vlm_hidden_states: VLM hidden states, shape (batch, vlm_seq_len, vlm_hidden_size)
            action_hidden_states: Action hidden states, shape (batch, action_seq_len, action_hidden_size)
            layer_idx: Index of the transformer layer
            attention_mask: 4D additive attention mask (blockwise AR pattern)
            position_ids: Joint position IDs for RoPE
            past_key_values: KV cache (not used in blockwise AR training)
            use_cache: Whether to use KV cache
            cache_position: Current cache position

        Returns:
            Tuple of (vlm_hidden_states, action_hidden_states) after the layer
        """
        vlm_layer = self.vlm.text_model.layers[layer_idx]
        action_layer = self.action_expert.layers[layer_idx]

        vlm_norm = vlm_layer.input_layernorm(vlm_hidden_states)
        action_norm = action_layer.input_layernorm(action_hidden_states)

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

        vlm_q = vlm_q.view(bsz, vlm_seq_len, num_heads, head_dim).transpose(1, 2)
        vlm_k = vlm_k.view(bsz, vlm_seq_len, num_kv_heads, head_dim).transpose(1, 2)
        vlm_v = vlm_v.view(bsz, vlm_seq_len, num_kv_heads, head_dim).transpose(1, 2)

        action_q = action_q.view(bsz, action_seq_len, num_heads, head_dim).transpose(1, 2)
        action_k = action_k.view(bsz, action_seq_len, num_kv_heads, head_dim).transpose(1, 2)
        action_v = action_v.view(bsz, action_seq_len, num_kv_heads, head_dim).transpose(1, 2)

        q = torch.cat([vlm_q, action_q], dim=2)
        k = torch.cat([vlm_k, action_k], dim=2)
        v = torch.cat([vlm_v, action_v], dim=2)

        if position_ids is None:
            position_ids = torch.arange(total_len, device=q.device).unsqueeze(0).expand(bsz, -1)
        if position_ids.shape[1] != total_len:
            raise ValueError(
                f"position_ids length should be {total_len}, actual is {position_ids.shape[1]}"
            )

        dummy = torch.empty(
            (bsz, total_len, cfg.hidden_size), device=q.device, dtype=vlm_hidden_states.dtype
        )
        cos, sin = self.vlm.text_model.rotary_emb(dummy, position_ids=position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if num_kv_heads != num_heads:
            k = repeat_kv(k, num_heads // num_kv_heads)
            v = repeat_kv(v, num_heads // num_kv_heads)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * (head_dim**-0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, total_len, -1)

        vlm_attn = vlm_layer.self_attn.o_proj(attn_output[:, :vlm_seq_len])
        action_attn = action_layer.self_attn.o_proj(attn_output[:, vlm_seq_len:])

        vlm_hidden_states = vlm_hidden_states + vlm_attn
        action_hidden_states = action_hidden_states + action_attn

        vlm_residual = vlm_hidden_states
        vlm_hidden_states = vlm_layer.post_attention_layernorm(vlm_hidden_states)
        vlm_hidden_states = vlm_residual + vlm_layer.mlp(vlm_hidden_states)

        action_residual = action_hidden_states
        action_hidden_states = action_layer.post_attention_layernorm(action_hidden_states)
        action_hidden_states = action_residual + action_layer.mlp(action_hidden_states)

        return vlm_hidden_states, action_hidden_states

    def _build_vlm_inputs_embeds(
        self,
        *,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        pixel_values: Optional[torch.FloatTensor],
        pixel_attention_mask: Optional[torch.BoolTensor],
        image_hidden_states: Optional[torch.FloatTensor],
    ):
        """
        Construct VLM input embeddings from various input formats.

        Handles the conversion of input_ids or inputs_embeds into proper VLM input
        embeddings, including optional image feature merging.

        Args:
            input_ids: Input token IDs for VLM
            inputs_embeds: Pre-computed input embeddings
            pixel_values: Image pixel values for vision encoder
            pixel_attention_mask: Attention mask for images
            image_hidden_states: Pre-computed image hidden states

        Returns:
            Tuple of (batch_size, vlm_seq_len, vlm_inputs_embeds, image_hidden_states)

        Note:
            This method strictly follows SmolVLMModel.forward logic for consistency.
        """
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
                    "When using pixel_values, input_ids must be provided (for locating image_token_id and inputs_merger)"
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

        return batch_size, vlm_seq_len, vlm_inputs_embeds, image_hidden_states

    def _run_action_sequence(
        self,
        *,
        vlm_inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        bos_len: int,
        action_input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Execute one forward pass for the action sequence.

        Constructs the full input sequence [BOS*n + action_input_ids], processes it
        through shared attention layers, and returns logits for all action positions.

        Args:
            vlm_inputs_embeds: VLM input embeddings
            attention_mask: VLM attention mask
            bos_len: Number of BOS tokens to prepend (typically = block_size)
            action_input_ids: Action token IDs for the sequence, shape (batch, L)
            position_ids: Optional position IDs

        Returns:
            Action logits for all positions (including BOS), shape (batch, bos_len+L, vocab_size)

        Note:
            Blockwise AR uses "block shift prediction":
            - Input: [BOS*n, blk0, blk1, ...]
            - Output positions 0..n-1 (BOS) predict block 0
            - Output positions n..2n-1 (block 0) predict block 1
            - And so on...
        """
        batch_size = vlm_inputs_embeds.shape[0]
        device = vlm_inputs_embeds.device

        if bos_len <= 0:
            raise ValueError(f"bos_len must be > 0, current value is {bos_len}")
        if action_input_ids.dim() != 2 or action_input_ids.shape[0] != batch_size:
            raise ValueError(
                f"action_input_ids expected shape (b, L), actual shape is {tuple(action_input_ids.shape)}"
            )
        if action_input_ids.dtype != torch.long:
            action_input_ids = action_input_ids.to(torch.long)
        action_input_ids = action_input_ids.to(device)
        if action_input_ids.numel() > 0:
            if (action_input_ids < 0).any() or (action_input_ids >= self.action_vocab_size).any():
                raise ValueError(
                    f"action_input_ids can only contain action token id (range [0, {self.action_vocab_size})), "
                    f"but detected min={int(action_input_ids.min().item())}, max={int(action_input_ids.max().item())}"
                )

        bos_embeds = self.bos_embedding.expand(batch_size, bos_len, -1).to(
            device=device, dtype=vlm_inputs_embeds.dtype
        )
        if action_input_ids.shape[1] == 0:
            action_token_embeds = torch.empty(
                (batch_size, 0, bos_embeds.shape[-1]), device=device, dtype=bos_embeds.dtype
            )
        else:
            action_token_embeds = self.action_token_embedding(action_input_ids).to(
                dtype=bos_embeds.dtype
            )
        action_inputs_embeds = torch.cat(
            [bos_embeds, action_token_embeds], dim=1
        )  # (b, bos_len + L, h)
        action_seq_len = action_inputs_embeds.shape[1]
        vlm_seq_len = vlm_inputs_embeds.shape[1]
        total_seq_len = vlm_seq_len + action_seq_len

        # No padding: all keys are valid
        action_key_mask = torch.ones((batch_size, action_seq_len), device=device, dtype=torch.long)

        if position_ids is None:
            # Default: action part grows continuously with VLM part (old behavior)
            position_ids = (
                torch.arange(total_seq_len, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        else:
            if (
                position_ids.dim() != 2
                or position_ids.shape[0] != batch_size
                or position_ids.shape[1] != total_seq_len
            ):
                raise ValueError(
                    f"position_ids expected shape (b, total_seq_len)=({batch_size},{total_seq_len}), actual shape is {tuple(position_ids.shape)}"
                )
            if position_ids.dtype != torch.long:
                position_ids = position_ids.to(torch.long)
            position_ids = position_ids.to(device)

        attention_mask_4d = self._build_joint_attention_mask_blockwise_ar(
            attention_mask=attention_mask,
            vlm_seq_len=vlm_seq_len,
            action_seq_len=action_seq_len,
            device=device,
            action_key_mask=action_key_mask,
        )

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
                past_key_values=None,
                use_cache=False,
                cache_position=None,
            )

        action_hidden_states = self.action_expert.norm(action_hidden_states)
        return self.action_lm_head(action_hidden_states)  # (b, L, action_vocab_size)

    def _predict_next_block_logits(
        self,
        *,
        vlm_inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        history_tokens: Optional[torch.LongTensor],
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Predict logits for the next block of action tokens.

        This is the core generation method for blockwise AR. Given the history of
        already-generated blocks, it returns logits for the next block.

        Args:
            vlm_inputs_embeds: VLM input embeddings
            attention_mask: VLM attention mask
            history_tokens: Previously generated action tokens, shape (batch, k*n)
                where k is the number of completed blocks
            position_ids: Optional position IDs

        Returns:
            Logits for the next block, shape (batch, block_size, action_vocab_size)

        Block Prediction Rules:
            - First block (history_tokens=None):
                Input: [BOS*n]
                Output: logits from BOS positions -> predict block 0

            - k-th block (k >= 1):
                Input: [BOS*n, block0, ..., block_{k-1}]
                Output: logits from block_{k-1} positions -> predict block k
        """
        bsz = vlm_inputs_embeds.shape[0]
        device = vlm_inputs_embeds.device
        n = self.block_size

        if history_tokens is None:
            logits = self._run_action_sequence(
                vlm_inputs_embeds=vlm_inputs_embeds,
                attention_mask=attention_mask,
                bos_len=n,
                action_input_ids=torch.empty((bsz, 0), device=device, dtype=torch.long),
                position_ids=position_ids,
            )
            return logits[:, :n, :]

        if history_tokens.dim() != 2 or history_tokens.shape[0] != bsz:
            raise ValueError(
                f"history_tokens expected shape (b, L), actual shape is {tuple(history_tokens.shape)}"
            )
        history_tokens = history_tokens.to(device=device, dtype=torch.long)
        if history_tokens.shape[1] % n != 0:
            raise ValueError(
                f"history_tokens length must be a multiple of n={n}, actual is {history_tokens.shape[1]}"
            )

        logits_all = self._run_action_sequence(
            vlm_inputs_embeds=vlm_inputs_embeds,
            attention_mask=attention_mask,
            bos_len=n,
            action_input_ids=history_tokens,
            position_ids=position_ids,
        )

        # logits_all action sequence length = n (BOS) + history_len
        # blk_{k-1} input position range is [n + (k-1)*n, n + k*n)
        history_blocks = history_tokens.shape[1] // n  # = k
        start = n + (history_blocks - 1) * n
        end = start + n
        return logits_all[:, start:end, :]

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
        Forward pass for training the blockwise AR model.

        Processes the VLM input and generates action tokens block by block. If labels
        are provided, computes cross-entropy loss for training.

        Args:
            input_ids: Input token IDs for VLM
            attention_mask: Attention mask for VLM input
            position_ids: Position IDs (auto-generated if None)
            past_key_values: KV cache (not used in blockwise AR training)
            inputs_embeds: Pre-computed input embeddings
            pixel_values: Image pixel values
            pixel_attention_mask: Image attention mask
            image_hidden_states: Pre-computed image features
            labels: Target action token IDs, shape (batch, token_budget)
            use_cache: Whether to use KV cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            cache_position: Cache position
            return_dict: Whether to return a ModelOutput
            logits_to_keep: Number of logits to keep (optimization)
            random_position_offset: Enable random stride position encoding
            random_position_min: Minimum random stride interval
            random_position_max: Maximum random stride interval

        Returns:
            SmolVLMCausalLMOutputWithPast containing:
                - loss: Cross-entropy loss if labels provided
                - logits: Action token logits, shape (batch, token_budget, vocab_size)
                - image_hidden_states: Image features from VLM

        Example:
            >>> outputs = model(
            ...     pixel_values=images,
            ...     input_ids=input_ids,
            ...     labels=action_labels,
            ...     random_position_offset=True,  # Data augmentation
            ... )
            >>> loss = outputs.loss
        """
        # Align with pd: these parameters are not currently used, but keep interface consistent
        _ = (
            position_ids,
            past_key_values,
            use_cache,
            output_attentions,
            cache_position,
            logits_to_keep,
            kwargs,
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.vlm_config.use_return_dict
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.vlm_config.output_hidden_states
        )

        batch_size, _, vlm_inputs_embeds, image_hidden_states = self._build_vlm_inputs_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
        )
        vlm_seq_len = vlm_inputs_embeds.shape[1]
        base_pos = int(vlm_seq_len)

        if not isinstance(random_position_offset, bool):
            raise TypeError(
                f"random_position_offset must be bool, but received {type(random_position_offset)}"
            )

        # Optional: construct augmented position_ids for action expert (only affects action part; VLM part remains [0..vlm_seq_len-1])
        action_pos_full = None
        if random_position_offset:
            action_pos_full = self._build_action_pos_ids_random(
                batch_size=batch_size,
                base_pos=base_pos,
                action_seq_len=self.token_budget,
                device=vlm_inputs_embeds.device,
                random_position_min=random_position_min,
                random_position_max=random_position_max,
            )

        logits = torch.empty(
            (batch_size, self.token_budget, self.action_vocab_size),
            device=vlm_inputs_embeds.device,
            dtype=vlm_inputs_embeds.dtype,
        )

        if labels is not None:
            if (
                labels.dim() != 2
                or labels.shape[0] != batch_size
                or labels.shape[1] != self.token_budget
            ):
                raise ValueError(
                    f"labels expected shape (b, token_budget)=({batch_size},{self.token_budget}), actual is {tuple(labels.shape)}"
                )
            if labels.dtype != torch.long:
                labels = labels.to(torch.long)
            # Training target must be action token only
            if (labels < 0).any() or (labels >= self.action_vocab_size).any():
                raise ValueError(
                    f"labels can only contain action token id (range [0, {self.action_vocab_size})), "
                    f"but detected min={int(labels.min().item())}, max={int(labels.max().item())}"
                )

            # By definition: input is [n BOS] + [blk0] + ... + [blk_{B-2}], output predicts [blk0] + ... + [blk_{B-1}]
            # Equivalent implementation: construct action_input_ids with length token_budget:
            #   action_input_ids = [BOS*n] + labels[:, :-n]
            n = self.block_size
            shifted = labels[:, :-n] if self.token_budget > n else labels[:, :0]
            if shifted.shape[1] + n != self.token_budget:
                raise ValueError(
                    f"Internal error: shifted length should be token_budget-n={self.token_budget - n}, actual is {shifted.shape[1]}"
                )
            position_ids_for_call = None
            if action_pos_full is not None:
                position_ids_for_call = self._build_joint_position_ids(
                    batch_size=batch_size,
                    vlm_seq_len=vlm_seq_len,
                    action_pos_ids=action_pos_full,
                    device=vlm_inputs_embeds.device,
                )
            logits = self._run_action_sequence(
                vlm_inputs_embeds=vlm_inputs_embeds,
                attention_mask=attention_mask,
                bos_len=n,
                action_input_ids=shifted,
                position_ids=position_ids_for_call,
            )
        else:
            # greedy unroll: first block uses BOS*n, subsequent use previous block (always with BOS*n as context)
            history_tokens = torch.empty(
                (batch_size, 0), device=vlm_inputs_embeds.device, dtype=torch.long
            )

            for block_idx in range(self.num_blocks):
                position_ids_for_call = None
                if action_pos_full is not None:
                    action_seq_len = (block_idx + 1) * self.block_size  # = bos_len + history_len
                    position_ids_for_call = self._build_joint_position_ids(
                        batch_size=batch_size,
                        vlm_seq_len=vlm_seq_len,
                        action_pos_ids=action_pos_full[:, :action_seq_len],
                        device=vlm_inputs_embeds.device,
                    )
                block_logits = self._predict_next_block_logits(
                    vlm_inputs_embeds=vlm_inputs_embeds,
                    attention_mask=attention_mask,
                    history_tokens=None if block_idx == 0 else history_tokens,
                    position_ids=position_ids_for_call,
                )
                start = block_idx * self.block_size
                end = (block_idx + 1) * self.block_size
                logits[:, start:end] = block_logits

                block_tokens = torch.argmax(
                    block_logits, dim=-1
                )  # (b, n), already only action vocab
                history_tokens = torch.cat([history_tokens, block_tokens], dim=1)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.action_vocab_size),
                labels.reshape(-1),
            )

        if not return_dict:
            return (loss, logits, None, None, image_hidden_states)

        return SmolVLMCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            image_hidden_states=image_hidden_states,
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
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generate action tokens using blockwise autoregressive decoding.

        Generates tokens block by block. Each block uses bidirectional attention
        internally, while maintaining causal order across blocks.

        Args:
            input_ids: Input token IDs for VLM
            attention_mask: Attention mask for VLM
            position_ids: Position IDs
            past_key_values: KV cache (not used)
            inputs_embeds: Pre-computed input embeddings
            pixel_values: Image pixel values
            pixel_attention_mask: Image attention mask
            image_hidden_states: Pre-computed image features
            use_cache: Whether to use KV cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            cache_position: Cache position
            return_dict: Whether to return dict
            position_offset: Stride offset for position encoding
            do_sample: Whether to sample (True) or use greedy (False)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Generated action token IDs, shape (batch_size, token_budget)

        Generation Process:
            Block 0: Input [BOS*n], predict block 0
            Block 1: Input [BOS*n, block0], predict block 1
            Block k: Input [BOS*n, block0, ..., block_{k-1}], predict block k

        Sampling Options:
            - Greedy (default): do_sample=False, argmax selection
            - Temperature: Higher values (e.g., 0.8) increase randomness
            - Top-k: Keep only top k logits, sample from them
            - Top-p: Keep smallest set of logits with cumulative prob >= p

        Example:
            >>> # Greedy generation
            >>> tokens = model.generate(pixel_values=images, input_ids=input_ids)
            >>> # Sampling with temperature
            >>> tokens = model.generate(
            ...     pixel_values=images, input_ids=input_ids,
            ...     do_sample=True, temperature=0.8, top_k=50
            ... )
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

        batch_size, _, vlm_inputs_embeds, _ = self._build_vlm_inputs_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
        )

        device = vlm_inputs_embeds.device
        vlm_seq_len = vlm_inputs_embeds.shape[1]
        base_pos = int(vlm_seq_len)

        if not isinstance(position_offset, int) or isinstance(position_offset, bool):
            raise TypeError(f"position_offset must be int, actual type is {type(position_offset)}")
        if position_offset < 0:
            raise ValueError(f"position_offset must be >= 0, current value is {position_offset}")

        generated = torch.empty((batch_size, self.token_budget), device=device, dtype=torch.long)
        history_tokens = torch.empty((batch_size, 0), device=device, dtype=torch.long)

        for block_idx in range(self.num_blocks):
            action_seq_len = (block_idx + 1) * self.block_size  # = bos_len + history_len
            action_pos_ids = self._build_action_pos_ids_strided(
                batch_size=batch_size,
                base_pos=base_pos,
                action_seq_len=action_seq_len,
                device=device,
                position_offset=position_offset,
            )
            position_ids_for_call = self._build_joint_position_ids(
                batch_size=batch_size,
                vlm_seq_len=vlm_seq_len,
                action_pos_ids=action_pos_ids,
                device=device,
            )
            block_logits = self._predict_next_block_logits(
                vlm_inputs_embeds=vlm_inputs_embeds,
                attention_mask=attention_mask,
                history_tokens=None if block_idx == 0 else history_tokens,
                position_ids=position_ids_for_call,
            )

            if do_sample:
                # Copy logits to avoid modifying original tensor
                logits_to_sample = block_logits.clone()

                # 1. Temperature Scaling
                if temperature > 0 and temperature != 1.0:
                    logits_to_sample = logits_to_sample / temperature

                # Flatten for unified processing (b*n, d)
                b, n, d = logits_to_sample.shape
                flat_logits = logits_to_sample.view(-1, d)

                # 2. Top-K Sampling
                if top_k > 0:
                    top_k = min(top_k, d)
                    # Get the k-th largest value
                    v, _ = torch.topk(flat_logits, top_k)
                    min_values = v[:, -1].unsqueeze(1)
                    # Set logits smaller than k-th largest to -inf
                    flat_logits = torch.where(
                        flat_logits < min_values,
                        torch.tensor(float("-inf"), device=device, dtype=flat_logits.dtype),
                        flat_logits,
                    )

                # 3. Top-P (Nucleus) Sampling
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(flat_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Restore to original index order
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    flat_logits = flat_logits.masked_fill(indices_to_remove, float("-inf"))

                # 4. Final Sampling
                probs = torch.softmax(flat_logits, dim=-1)
                flat_tokens = torch.multinomial(probs, num_samples=1)  # (b*n, 1)
                block_tokens = flat_tokens.view(b, n)

            else:
                block_tokens = torch.argmax(block_logits, dim=-1)  # (b, n)

            start = block_idx * self.block_size
            end = (block_idx + 1) * self.block_size
            generated[:, start:end] = block_tokens
            history_tokens = torch.cat([history_tokens, block_tokens], dim=1)

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
        # Call vlm's prepare_inputs_for_generation (same as pd)
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


AutoConfig.register("smolvla_blockwise_ar", SmolVLABlockwiseARConfig)
AutoModel.register(SmolVLABlockwiseARConfig, SmolVLABlockwiseAR)
