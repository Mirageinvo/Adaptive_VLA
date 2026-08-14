import logging
from typing import Optional, Union

import numpy as np
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
    Configuration for LlamaActionExpert model in Knowledge Isolation architecture.

    The Action Expert in KI architecture processes continuous action values through
    flow matching, unlike the discrete token prediction in parallel decoding. The
    hidden dimensions typically match the VLM, with only the intermediate_size being
    halved for a MoE-style architecture.

    Key Parameters:
        vocab_size: Vocabulary size (not used for flow matching, kept for compatibility)
        hidden_size: Hidden dimension (typically matches VLM)
        intermediate_size: MLP intermediate dimension (typically VLM's 1/2)
        num_hidden_layers: Number of transformer layers (matches VLM for layer alignment)
        num_attention_heads: Number of attention heads (matches VLM)
        num_key_value_heads: Number of KV heads for GQA

    Note:
        In KI architecture, the Action Expert shares the VLM's KV cache for efficient
        inference. The attention must use eager implementation for bidirectional attention.
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
        pad_token_id: Optional[int] = 128_002,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 2,
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
        # BC: if there is a 'type' field, copy it it to 'rope_type'.

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
    Llama-based Action Expert model for Knowledge Isolation architecture.

    This model extends LlamaModel to serve as an action predictor using continuous
    representations. In the KI architecture, it operates independently from the VLM
    during training but shares the VLM's KV cache during inference.

    Architecture:
        - Standard Llama decoder layers matching VLM's layer count
        - Bidirectional attention for action tokens
        - No language modeling head (uses separate output projection for flow matching)

    The model is designed for flow matching training, where it predicts velocity
    vectors in continuous action space rather than discrete tokens.
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


class FourierTimeEmbedding(nn.Module):
    """
    Fourier time encoding for flow matching timestamp encoding.

    Encodes a continuous timestamp t in [0, 1] to a hidden_size dimensional vector
    using random Fourier features. This allows the model to condition on the
    flow matching timestep.

    The encoding uses sin and cos of scaled random frequencies:
        encoding = concat([cos(2*pi*t*freqs), sin(2*pi*t*freqs)])

    Args:
        hidden_size: Output embedding dimension
        scale: Scale factor for random frequency initialization (default: 2.0)

    Example:
        >>> time_emb = FourierTimeEmbedding(hidden_size=256)
        >>> timestamps = torch.tensor([0.0, 0.5, 1.0])  # (3,)
        >>> encoding = time_emb(timestamps)  # (3, 256)
    """

    def __init__(self, hidden_size: int, scale: float = 2.0):
        super().__init__()
        self.hidden_size = hidden_size
        # Number of frequencies is half of hidden_size (because each frequency corresponds to sin and cos)
        self.register_buffer("freqs", torch.randn(hidden_size // 2) * scale, persistent=True)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (batch_size,) shaped tensor, values in [0, 1]
        Returns:
            (batch_size, hidden_size) shaped encoding
        """
        device = timestamps.device
        freqs = self.freqs.to(device)
        # timestamps: (b,), freqs: (d,)
        # Calculate 2 * pi * timestamps * freqs
        pos_emb = torch.einsum("b, d -> b d", timestamps, 2 * np.pi * freqs).to(
            device, torch.float32
        )
        # Concatenate cos and sin
        pos_emb = torch.cat([pos_emb.cos(), pos_emb.sin()], dim=-1)
        return pos_emb


class SmolVLAKnowledgeIsolationConfig(PretrainedConfig):
    """
    Configuration for SmolVLAKnowledgeIsolation model.

    The Knowledge Isolation (KI) architecture separates VLM and Action Expert paths,
    allowing them to specialize independently while sharing KV cache during inference.
    Action prediction uses flow matching on continuous action values.

    Configuration Flow:
        1. Create from existing VLM config using from_vlm_config()
        2. Action Expert config auto-generated with matching dimensions
        3. Only intermediate_size is halved (MoE-style architecture)

    Key Parameters:
        vlm_config: Configuration for the VLM (SmolVLM)
        action_expert_config: Configuration for the Action Expert
        action_horizon: Number of action steps to predict (default: 20)
        action_dim: Dimension of each action vector (default: 7)

    Key Differences from Parallel Decoding:
        - Actions are continuous vectors, not discrete tokens
        - Uses flow matching for training (continuous-time generative model)
        - VLM and Action Expert paths are isolated during forward pass
        - Shares KV cache during inference for efficiency
    """

    model_type = "smol_vla_knowledge_isolation"

    def __init__(
        self,
        vlm_config=None,
        action_expert_config=None,
        action_horizon: int = 20,
        action_dim: int = 7,
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
        self.action_horizon = action_horizon
        self.action_dim = action_dim

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
    def from_vlm_config(cls, vlm_config, action_horizon: int = 20, action_dim: int = 7, **kwargs):
        """Create config from VLM config, auto-generate action_expert_config"""
        # Extract info from vlm_config's text_config
        text_config = vlm_config.text_config if hasattr(vlm_config, "text_config") else vlm_config

        # Create action_expert_config, only intermediate_size is half, others remain consistent (MoE architecture)
        # Use getattr to safely get attributes
        hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                f"Cannot get hidden_size from text_config, text_config type: {type(text_config)}"
            )

        intermediate_size = getattr(text_config, "intermediate_size", None)
        if intermediate_size is None:
            raise ValueError("Cannot get intermediate_size from text_config")

        num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
        if num_hidden_layers is None:
            raise ValueError("Cannot get num_hidden_layers from text_config")

        num_attention_heads = getattr(text_config, "num_attention_heads", None)
        if num_attention_heads is None:
            raise ValueError("Cannot get num_attention_heads from text_config")

        num_key_value_heads = getattr(text_config, "num_key_value_heads", None)

        # For flow matching, vocab_size can be set to any value (won't be used)
        action_expert_config = LlamaActionExpertConfig(
            vocab_size=2048,  # Won't be used, but needs to be set
            hidden_size=hidden_size,  # Keep consistent
            intermediate_size=intermediate_size // 2,  # Only this is half
            num_hidden_layers=num_hidden_layers,  # Keep consistent
            num_attention_heads=num_attention_heads,  # Keep consistent
            num_key_value_heads=num_key_value_heads,  # Keep consistent
            hidden_act=getattr(text_config, "hidden_act", "silu"),
            max_position_embeddings=getattr(text_config, "max_position_embeddings", 2048),
            initializer_range=getattr(text_config, "initializer_range", 0.02),
            rms_norm_eps=getattr(text_config, "rms_norm_eps", 1e-5),
            use_cache=getattr(text_config, "use_cache", True),
            pad_token_id=getattr(text_config, "pad_token_id", 128_002),
            bos_token_id=getattr(text_config, "bos_token_id", 1),
            eos_token_id=getattr(text_config, "eos_token_id", 2),
            pretraining_tp=getattr(text_config, "pretraining_tp", 1),
            tie_word_embeddings=getattr(text_config, "tie_word_embeddings", False),
            attention_bias=getattr(text_config, "attention_bias", False),
            attention_dropout=getattr(text_config, "attention_dropout", 0.0),
            mlp_bias=getattr(text_config, "mlp_bias", False),
            _attn_implementation="eager",  # Bidirectional attention must use eager
        )

        return cls(
            vlm_config=vlm_config,
            action_expert_config=action_expert_config,
            action_horizon=action_horizon,
            action_dim=action_dim,
            **kwargs,
        )


class SmolVLAKnowledgeIsolation(SmolVLMPreTrainedModel, GenerationMixin):
    """
    SmolVLA Knowledge Isolation model for continuous action prediction.

    This model implements a knowledge isolation architecture where the VLM and
    Action Expert operate on separate paths. Action prediction uses flow matching,
    a continuous-time generative model that predicts velocity vectors to transform
    noise into target actions.

    Architecture Overview:
        ┌─────────────────────────────────────────────────────────────────┐
        │                        Training Phase                           │
        │  ┌───────────────┐              ┌────────────────────┐          │
        │  │     VLM       │              │   Action Expert    │          │
        │  │   Forward     │───KV Cache──▶│   Forward          │          │
        │  │  (independent)│              │ (bidirectional)    │          │
        │  └───────────────┘              └──────────┬─────────┘          │
        │                                            │                    │
        │                                 ┌──────────▼─────────┐          │
        │                                 │  Velocity Head     │          │
        │                                 │  (output_proj)     │          │
        │                                 └────────────────────┘          │
        └─────────────────────────────────────────────────────────────────┘

    Key Features:
        1. **Knowledge Isolation**: VLM and Action Expert have separate forward paths,
           allowing each to specialize in its domain.

        2. **Flow Matching Training**:
           - Sample timestamp t ~ Uniform(0, 1)
           - Sample noise x_0 ~ N(0, I)
           - Interpolate: x_t = (1-t) * x_0 + t * x_1 (target actions)
           - Predict velocity: v_t = x_1 - x_0
           - Loss: MSE(model(x_t, t), v_t)

        3. **KV Cache Sharing**: During inference, Action Expert reuses VLM's
           pre-computed KV cache, avoiding redundant computation.

        4. **Fourier Time Encoding**: Timestamp t is encoded using random Fourier
           features and added to action embeddings.

    Training:
        Use forward() with labels (target actions) for flow matching loss.

    Inference:
        Use sample() to generate actions via iterative flow matching sampling.

    Example:
        >>> config = SmolVLAKnowledgeIsolationConfig.from_vlm_config(vlm_config)
        >>> model = SmolVLAKnowledgeIsolation(config)
        >>> # Training
        >>> outputs = model(pixel_values=images, input_ids=input_ids, labels=actions)
        >>> loss = outputs.loss
        >>> # Inference
        >>> actions = model.sample(pixel_values=images, input_ids=input_ids, sampling_steps=10)
    """

    config_class = SmolVLAKnowledgeIsolationConfig

    def __init__(self, config: SmolVLAKnowledgeIsolationConfig, *, init_backbones: bool = True):
        super().__init__(config)
        # Note: in from_pretrained() we first load vlm/action_expert separately, then attach them to main model.
        # To avoid initializing an extra set of same-size parameters here, allow skipping backbone initialization.
        if init_backbones:
            logger.info("Initializing VLM backbone...")
            self.vlm = SmolVLMModel(config.vlm_config)
            logger.info("Initializing Action Expert backbone...")
            self.action_expert = LlamaActionExpert(config.action_expert_config)
        else:
            self.vlm = None
            self.action_expert = None
        self.image_token_id = self.config.vlm_config.image_token_id

        # input_proj: action_dim -> hidden_size
        self.input_proj = nn.Linear(
            config.action_dim, config.action_expert_config.hidden_size, bias=False
        )

        # output_proj: hidden_size -> action_dim
        self.output_proj = nn.Linear(
            config.action_expert_config.hidden_size, config.action_dim, bias=False
        )

        # Fourier time encoding
        self.time_embedding = FourierTimeEmbedding(config.action_expert_config.hidden_size)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        if self.vlm is not None:
            self.vlm.text_model.generation_config = GenerationConfig.from_model_config(
                config.vlm_config
            )

        # Initialize weights and apply final processing
        self.post_init()

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

        logger.info(f"Loading SmolVLAKnowledgeIsolation from {pretrained_model_name_or_path}...")

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
            action_horizon = kwargs.pop("action_horizon", 20)
            action_dim = kwargs.pop("action_dim", 7)

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
            config = SmolVLAKnowledgeIsolationConfig(
                vlm_config=vlm_config,
                action_expert_config=action_expert_config,
                action_horizon=action_horizon,
                action_dim=action_dim,
            )

            # Create model instance
            model = cls(config, init_backbones=False)
            model.vlm = vlm
            model.action_expert = action_expert
            model.vlm.text_model.generation_config = GenerationConfig.from_model_config(
                config.vlm_config
            )

            # Load KI side additional components
            action_components_path = os.path.join(local_path, "action_components.bin")
            if os.path.exists(action_components_path):
                logger.info("Loading KI action components (input_proj/output_proj/time_embedding)...")
                action_components = torch.load(action_components_path, map_location="cpu")
                if "input_proj" in action_components:
                    model.input_proj.load_state_dict(action_components["input_proj"])
                if "output_proj" in action_components:
                    model.output_proj.load_state_dict(action_components["output_proj"])
                if "time_embedding" in action_components:
                    model.time_embedding.load_state_dict(action_components["time_embedding"])
            else:
                logger.warning(
                    "action_components.bin not found, KI's input_proj/output_proj/time_embedding will use random initialization"
                )

            logger.info("Model loading complete.")
            return model
        else:
            # Default: only load vlm, action_expert uses random initialization
            logger.info("No VLM/Action Expert subdirectories found. Loading from single checkpoint...")
            try:
                action_horizon = kwargs.pop("action_horizon", 20)
                action_dim = kwargs.pop("action_dim", 7)

                # Load config
                logger.info(f"Loading config from {local_path}...")
                original_config = AutoConfig.from_pretrained(local_path)

                # Create new config
                config = SmolVLAKnowledgeIsolationConfig.from_vlm_config(
                    original_config,
                    action_horizon=action_horizon,
                    action_dim=action_dim,
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

                # action_expert must be manually initialized
                logger.info("Initializing Action Expert with random weights...")
                model.action_expert = LlamaActionExpert(config.action_expert_config)
                logger.warning(
                    "action_expert not loaded from pretrained weights, will use random initialization"
                )

                return model
            except Exception as e:
                raise ValueError(f"Cannot load model from {local_path}: {e}")

    def save_pretrained(self, save_directory, **kwargs):
        """Save model, save vlm / action_expert to subfolders, and save KI side additional components"""
        import os

        logger.info(f"Saving SmolVLAKnowledgeIsolation to {save_directory}...")
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

        # Save KI side additional components (not in vlm/action_expert subdirectories)
        # - input_proj / output_proj: Linear
        # - time_embedding: FourierTimeEmbedding (contains buffer `freqs`)
        logger.info("Saving action components (input_proj/output_proj/time_embedding)...")
        action_components = {
            "input_proj": self.input_proj.state_dict(),
            "output_proj": self.output_proj.state_dict(),
            "time_embedding": self.time_embedding.state_dict(),
        }
        action_components_path = os.path.join(save_directory, "action_components.bin")
        torch.save(action_components, action_components_path)
        logger.info("Model saved successfully.")

    def _action_expert_forward_with_vlm_cache(
        self,
        action_hidden_states: torch.Tensor,
        vlm_kv_cache: Optional[Cache],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """
        Execute Action Expert forward pass using VLM's pre-computed KV cache.

        This method implements KV cache sharing between VLM and Action Expert.
        For each layer, it retrieves VLM's cached K/V and concatenates them with
        Action Expert's own K/V for joint attention computation.

        Process Flow:
            1. For each layer:
               a. Apply input LayerNorm to action hidden states
               b. Compute Q, K, V projections for action tokens
               c. Retrieve K, V from VLM's cache for this layer
               d. Apply rotary embeddings to action's Q, K
               e. Concatenate: [VLM_K, action_K], [VLM_V, action_V]
               f. Compute attention: action_Q @ concatenated_K^T
               g. Apply O projection and MLP

        Args:
            action_hidden_states: Action embeddings, shape (batch, action_seq_len, hidden_size)
            vlm_kv_cache: VLM's pre-computed KV cache from a previous forward pass
            attention_mask: 4D attention mask, shape (batch, 1, action_seq_len, total_len)
            position_ids: Position IDs for action tokens, shape (batch, action_seq_len)
            cache_position: Cache positions for action tokens

        Returns:
            Processed action hidden states, shape (batch, action_seq_len, hidden_size)

        Note:
            This method intentionally bypasses transformers' Cache mechanism to
            avoid gradient issues. The VLM's cache is treated as a frozen tensor
            that we read from, not write to.
        """
        if vlm_kv_cache is None:
            raise ValueError("vlm_kv_cache cannot be None")

        num_layers = self.config.vlm_config.text_config.num_hidden_layers
        vlm_seq_len = vlm_kv_cache.get_seq_length()
        action_seq_len = action_hidden_states.shape[1]

        cfg = self.vlm.text_model.config
        num_heads = cfg.num_attention_heads
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
        head_dim = cfg.hidden_size // num_heads

        for layer_idx in range(num_layers):
            action_layer = self.action_expert.layers[layer_idx]

            # input layernorm
            action_norm = action_layer.input_layernorm(action_hidden_states)

            # QKV projections for action_expert
            action_q = action_layer.self_attn.q_proj(action_norm)
            action_k = action_layer.self_attn.k_proj(action_norm)
            action_v = action_layer.self_attn.v_proj(action_norm)

            bsz = action_q.shape[0]

            # Reshape to (b, heads, seq, head_dim)
            action_q = action_q.view(bsz, action_seq_len, num_heads, head_dim).transpose(1, 2)
            action_k = action_k.view(bsz, action_seq_len, num_kv_heads, head_dim).transpose(1, 2)
            action_v = action_v.view(bsz, action_seq_len, num_kv_heads, head_dim).transpose(1, 2)

            # Get k, v from VLM's kv_cache for this layer
            # kv_cache format: each layer is a tuple (k, v), where k, v have shape (batch_size, num_kv_heads, seq_len, head_dim)
            vlm_layer_cache = vlm_kv_cache[layer_idx]
            vlm_k = vlm_layer_cache[0]  # (batch_size, num_kv_heads, vlm_seq_len, head_dim)
            vlm_v = vlm_layer_cache[1]  # (batch_size, num_kv_heads, vlm_seq_len, head_dim)

            # Ensure vlm_k, vlm_v batch_size matches action's batch_size
            if vlm_k.shape[0] != bsz:
                raise ValueError(
                    f"VLM kv_cache batch_size {vlm_k.shape[0]} doesn't match action's batch_size {bsz}"
                )

            # Compute rotary embeddings for action part
            # position_ids should only contain position info for action part (provided by caller)
            if position_ids is None:
                if cache_position is None:
                    raise ValueError("position_ids and cache_position cannot both be None")
                action_position_ids = cache_position.unsqueeze(0).expand(
                    bsz, -1
                )  # (b, action_seq_len)
            else:
                # position_ids already only contains action part position info
                if position_ids.shape[1] != action_seq_len:
                    # If position_ids length doesn't match action_seq_len, try to extract action part
                    if position_ids.shape[1] >= vlm_seq_len + action_seq_len:
                        action_position_ids = position_ids[
                            :, vlm_seq_len : vlm_seq_len + action_seq_len
                        ]  # (b, action_seq_len)
                    else:
                        action_position_ids = (
                            position_ids  # Assume it's already action part positions
                        )
                else:
                    action_position_ids = position_ids  # (b, action_seq_len)

            # Compute rotary embeddings for action part
            dummy = torch.empty(
                (bsz, action_seq_len, cfg.hidden_size),
                device=action_q.device,
                dtype=action_hidden_states.dtype,
            )
            cos, sin = self.action_expert.rotary_emb(dummy, position_ids=action_position_ids)

            # Apply rotary embeddings to action's q, k
            # Note: VLM's k already has rotary embeddings applied in cache (using VLM's positions)
            # action's k needs to have rotary embeddings applied using action's positions
            action_q_rotated, action_k_rotated = apply_rotary_pos_emb(action_q, action_k, cos, sin)

            # Concatenate: VLM's k (already rotated) + action's k (just rotated)
            k = torch.cat(
                [vlm_k, action_k_rotated], dim=2
            )  # (b, kv_heads, vlm_seq_len+action_seq_len, head_dim)
            v = torch.cat(
                [vlm_v, action_v], dim=2
            )  # (b, kv_heads, vlm_seq_len+action_seq_len, head_dim)
            q = action_q_rotated  # (b, heads, action_seq_len, head_dim)

            # GQA expand
            if num_kv_heads != num_heads:
                k = repeat_kv(k, num_heads // num_kv_heads)
                v = repeat_kv(v, num_heads // num_kv_heads)

            # attention
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) * (
                head_dim**-0.5
            )  # (b, heads, action_seq_len, vlm_seq_len+action_seq_len)
            if attention_mask is not None:
                # attention_mask shape should be (b, 1, query_len, key_len)
                # query_len = action_seq_len, key_len = total_len
                attn_weights = attn_weights + attention_mask  # broadcast: (b,1,q,k)
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v)  # (b, heads, action_seq_len, head_dim)

            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, action_seq_len, -1)

            # o_proj
            action_attn = action_layer.self_attn.o_proj(attn_output)

            # residual 1
            action_hidden_states = action_hidden_states + action_attn

            # MLP block (residual 2)
            action_residual = action_hidden_states
            action_hidden_states = action_layer.post_attention_layernorm(action_hidden_states)
            action_hidden_states = action_residual + action_layer.mlp(action_hidden_states)

        return action_hidden_states

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
        labels: Optional[torch.FloatTensor] = None,  # (b, horizon, action_dim) shaped x_1
        timestamps: Optional[torch.FloatTensor] = None,  # (b,) shaped, values in [0, 1]
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, SmolVLMCausalLMOutputWithPast]:
        """
        Forward pass for flow matching training.

        Implements the flow matching training objective:
            1. Sample noise x_0 ~ N(0, I)
            2. Interpolate: x_t = (1-t) * x_0 + t * x_1 (where x_1 = labels)
            3. Predict velocity: v_t = model(x_t, t)
            4. Target velocity: x_1 - x_0
            5. Loss: MSE(v_t, x_1 - x_0)

        Args:
            input_ids: Input token IDs for VLM
            attention_mask: Attention mask for VLM input
            position_ids: Position IDs
            past_key_values: KV cache
            inputs_embeds: Pre-computed input embeddings
            pixel_values: Image pixel values
            pixel_attention_mask: Image attention mask
            image_hidden_states: Pre-computed image features
            labels: Target actions x_1, shape (batch, action_horizon, action_dim)
            timestamps: Flow matching timestamps in [0, 1], shape (batch,)
                - If None, randomly sampled
            use_cache: Whether to use KV cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            cache_position: Cache position
            return_dict: Whether to return dict

        Returns:
            SmolVLMCausalLMOutputWithPast containing:
                - loss: Flow matching MSE loss if labels provided
                - logits: Predicted velocity v_t, shape (batch, action_horizon, action_dim)
                - past_key_values: None
                - image_hidden_states: Image features from VLM

        Example:
            >>> outputs = model(
            ...     pixel_values=images,
            ...     input_ids=input_ids,
            ...     labels=target_actions,  # (batch, horizon, action_dim)
            ...     timestamps=torch.rand(batch_size),  # Optional
            ... )
            >>> loss = outputs.loss
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

        # First execute vlm forward
        vlm_outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )

        # Get kv_cache from vlm outputs
        vlm_kv_cache = vlm_outputs.past_key_values

        # Prepare action_expert inputs
        # Need to compute batch_size
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
        else:
            batch_size = 1

        # If timestamps not provided, randomly sample
        if timestamps is None:
            timestamps = torch.rand(batch_size, device=vlm_outputs.last_hidden_state.device)

        # If labels not provided, cannot compute loss, but can return predictions
        if labels is None:
            # Sample x_0 from Gaussian distribution
            x_0 = torch.randn(
                batch_size, self.action_horizon, self.action_dim, device=timestamps.device
            )
            # Compute x_t = (1-t) * x_0 + t * x_0 (since no x_1, use x_0 instead)
            x_t = x_0
        else:
            # Sample x_0 from Gaussian distribution
            x_0 = torch.randn_like(labels)  # (b, horizon, action_dim)
            # Compute x_t = (1-t) * x_0 + t * x_1
            t = timestamps.view(batch_size, 1, 1)  # (b, 1, 1) for broadcasting
            x_t = (1 - t) * x_0 + t * labels  # (b, horizon, action_dim)

        # Project x_t to hidden_size through input_proj
        # x_t: (b, horizon, action_dim) -> (b, horizon, hidden_size)
        action_inputs_embeds = self.input_proj(x_t)

        # Compute time encoding and add to inputs_embeds
        # timestamps: (b,) -> (b, hidden_size)
        time_emb = self.time_embedding(timestamps)  # (b, hidden_size)
        # Broadcast to horizon dimension: (b, hidden_size) -> (b, 1, hidden_size) -> (b, horizon, hidden_size)
        time_emb = time_emb.unsqueeze(1).expand(-1, self.action_horizon, -1)
        action_inputs_embeds = action_inputs_embeds + time_emb

        # Compute cache_position and position_ids
        # Get number of already processed tokens from vlm's kv_cache
        past_seen_tokens = vlm_kv_cache.get_seq_length() if vlm_kv_cache is not None else 0
        action_cache_position = (
            torch.arange(self.action_horizon, device=action_inputs_embeds.device) + past_seen_tokens
        )
        action_position_ids = action_cache_position.unsqueeze(0).expand(batch_size, -1)

        # Create 4D attention_mask (bidirectional attention, all positions visible)
        # Shape: (batch_size, 1, query_len, key_len)
        # 0.0 means visible, -inf means not visible
        query_len = self.action_horizon
        key_len = past_seen_tokens + self.action_horizon

        # Bidirectional attention, all positions visible, so all 0.0
        action_attention_mask = torch.zeros(
            (batch_size, 1, query_len, key_len),
            device=action_inputs_embeds.device,
            dtype=torch.float32,
        )

        # action_expert forward (using fixed method, ensuring gradient propagation)
        # Using new _action_expert_forward_with_vlm_cache method to avoid gradient detachment caused by Cache mechanism
        action_hidden_states = self._action_expert_forward_with_vlm_cache(
            action_hidden_states=action_inputs_embeds,
            vlm_kv_cache=vlm_kv_cache,
            attention_mask=action_attention_mask,
            position_ids=action_position_ids,
            cache_position=action_cache_position,
        )

        # Final norm
        action_hidden_states = self.action_expert.norm(action_hidden_states)

        # Get velocity estimate v_t through output_proj, shape (batch_size, horizon, action_dim)
        v_t = self.output_proj(action_hidden_states)

        loss = None
        if labels is not None:
            # Flow matching loss: MSE(v_t - (x_1 - x_0))
            target_velocity = labels - x_0  # (b, horizon, action_dim)
            loss = nn.functional.mse_loss(v_t, target_velocity)

        return SmolVLMCausalLMOutputWithPast(
            loss=loss,
            logits=v_t,  # Return velocity estimate as logits
            past_key_values=None,  # MoE architecture doesn't return past_key_values
            hidden_states=action_hidden_states if output_hidden_states else None,
            attentions=None,  # Don't return attention for now
            image_hidden_states=vlm_outputs.image_hidden_states
            if hasattr(vlm_outputs, "image_hidden_states")
            else None,
        )

    def sample(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        sampling_steps: int = 10,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample actions using flow matching (Euler method).

        Starting from Gaussian noise x_0 ~ N(0, I), iteratively apply the
        learned velocity field to transform noise into target actions.
        Uses Euler integration: x_{t+dt} = x_t + v_t * dt

        Sampling Process:
            1. Initialize: x_0 ~ N(0, I)
            2. For t = 0, dt, 2*dt, ..., 1:
               a. Predict velocity: v_t = model(x_t, t)
               b. Update: x_{t+dt} = x_t + v_t * dt
            3. Return x_1 as the generated actions

        Args:
            input_ids: Input token IDs for VLM
            attention_mask: Attention mask for VLM
            position_ids: Position IDs
            past_key_values: KV cache
            pixel_values: Image pixel values
            pixel_attention_mask: Image attention mask
            image_hidden_states: Pre-computed image features
            sampling_steps: Number of integration steps (default: 10)
            use_cache: Whether to use KV cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            cache_position: Cache position
            return_dict: Whether to return dict

        Returns:
            Generated actions x_1, shape (batch_size, action_horizon, action_dim)

        Example:
            >>> actions = model.sample(
            ...     pixel_values=images,
            ...     input_ids=input_ids,
            ...     sampling_steps=10,
            ... )
            >>> # actions: (batch_size, action_horizon, action_dim)
        """
        # Determine batch_size
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
        elif pixel_values is not None:
            batch_size = pixel_values.shape[0]
            device = pixel_values.device
        else:
            batch_size = 1
            device = next(self.parameters()).device

        # First execute vlm forward to get kv_cache
        vlm_outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=None,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )

        vlm_kv_cache = vlm_outputs.past_key_values

        # Sample x_0 from Gaussian distribution
        x_t = torch.randn(
            batch_size, self.action_horizon, self.action_dim, device=device, dtype=self.dtype
        )

        # Generate timestep sequence: from 0 to 1, divided into sampling_steps intervals
        timesteps = torch.linspace(0, 1, sampling_steps + 1, device=device)  # (sampling_steps + 1,)

        # Step-by-step sampling
        for i in range(sampling_steps):
            t_current = timesteps[i]  # Current timestep
            t_next = timesteps[i + 1]  # Next timestep
            dt = t_next - t_current  # Timestep size

            # Create timestamps for current timestep (batch_size,)
            timestamps = torch.full((batch_size,), t_current, device=device)

            # Project x_t to hidden_size through input_proj
            action_inputs_embeds = self.input_proj(x_t)  # (b, horizon, hidden_size)

            # Compute time encoding and add
            time_emb = self.time_embedding(timestamps).to(self.dtype)  # (b, hidden_size)
            time_emb = time_emb.unsqueeze(1).expand(
                -1, self.action_horizon, -1
            )  # (b, horizon, hidden_size)
            action_inputs_embeds = action_inputs_embeds + time_emb

            # Compute cache_position and position_ids
            past_seen_tokens = vlm_kv_cache.get_seq_length() if vlm_kv_cache is not None else 0
            action_cache_position = (
                torch.arange(self.action_horizon, device=device) + past_seen_tokens
            )
            action_position_ids = action_cache_position.unsqueeze(0).expand(batch_size, -1)

            # Create attention_mask
            query_len = self.action_horizon
            key_len = past_seen_tokens + self.action_horizon
            action_attention_mask = torch.zeros(
                (batch_size, 1, query_len, key_len),
                device=device,
                dtype=torch.float32,
            )

            # action_expert forward (using fixed method, ensuring gradient propagation)
            action_hidden_states = self._action_expert_forward_with_vlm_cache(
                action_hidden_states=action_inputs_embeds,
                vlm_kv_cache=vlm_kv_cache,
                attention_mask=action_attention_mask,
                position_ids=action_position_ids,
                cache_position=action_cache_position,
            )

            # Final norm
            action_hidden_states = self.action_expert.norm(action_hidden_states)

            # Get velocity estimate v_t
            v_t = self.output_proj(action_hidden_states)  # (b, horizon, action_dim)

            # Update x_t: x_t_next = x_t + v_t * dt
            x_t = x_t + v_t * dt

        # Return final x_1
        return x_t

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


AutoConfig.register("smol_vla_knowledge_isolation", SmolVLAKnowledgeIsolationConfig)
AutoModel.register(SmolVLAKnowledgeIsolationConfig, SmolVLAKnowledgeIsolation)
