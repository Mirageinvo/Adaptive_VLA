import copy
from typing import Any, Dict

from transformers import AutoConfig, PretrainedConfig


class ActionCodecConfig(PretrainedConfig):
    model_type = "action_codec"

    def __init__(
        self,
        embodiment_config: Dict[str, Any] = None,
        n_tokens: int = 16,
        n_quantizers: int = 1,
        z_dim: int = 512,
        vq_type: str = "vq",
        vq_codebook_size: int = 2048,
        vq_commitment_weight: float = 0.25,
        vq_decay: float = 0.99,
        vq_kmeans_init: bool = True,
        vq_threshold_ema_dead_code: int = 2,
        vq_quantizer_dropout: float = 0.25,
        encoder_dim: int = 256,
        encoder_n_layers: int = 6,
        encoder_n_heads: int = 8,
        encoder_add_self_attn: bool = False,
        encoder_add_causal_mask: bool = False,
        encoder_pos_encoding_type: str = "fourier",
        decoder_dim: int = 256,
        decoder_n_layers: int = 6,
        decoder_n_heads: int = 8,
        decoder_add_self_attn: bool = False,
        decoder_add_causal_mask: bool = False,
        decoder_pos_encoding_type: str = "fourier",
        decoder_cls_size: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if embodiment_config is None:
            default_config = {
                "franka_libero_20hz": {
                    "action_dim": 7,
                    "freq": 20,
                    "duration": 1,
                    "description": "20Hz 7-dim action for 1s. Delta eef position (xyz), orientation (rpy), and gripper position (1 open/0 close).",
                },
                "widowx_bridge_5hz": {
                    "action_dim": 7,
                    "freq": 5,
                    "duration": 1,
                    "description": "5Hz 7-dim action for 1s. Delta eef position (xyz), orientation (rpy), and gripper position (1 open/0 close).",
                },
                "franka_droid_15hz": {
                    "action_dim": 7,
                    "freq": 15,
                    "duration": 1,
                    "description": "15Hz 7-dim action for 1s. Delta eef position (xyz), orientation (rpy), and gripper position (1 open/0 close).",
                },
            }
            self.embodiment_config = copy.deepcopy(default_config)
        else:
            self.embodiment_config = copy.deepcopy(embodiment_config)

        self.n_tokens = n_tokens
        self.n_quantizers = n_quantizers
        self.z_dim = z_dim

        self.encoder_dim = encoder_dim
        self.encoder_n_layers = encoder_n_layers
        self.encoder_n_heads = encoder_n_heads
        self.encoder_add_self_attn = encoder_add_self_attn
        self.encoder_add_causal_mask = encoder_add_causal_mask
        self.encoder_pos_encoding_type = encoder_pos_encoding_type

        self.decoder_dim = decoder_dim
        self.decoder_n_layers = decoder_n_layers
        self.decoder_n_heads = decoder_n_heads
        self.decoder_add_self_attn = decoder_add_self_attn
        self.decoder_add_causal_mask = decoder_add_causal_mask
        self.decoder_pos_encoding_type = decoder_pos_encoding_type
        self.decoder_cls_size = decoder_cls_size

        self.vq_type = vq_type
        self.vq_codebook_size = vq_codebook_size
        self.vq_commitment_weight = vq_commitment_weight
        self.vq_decay = vq_decay
        self.vq_kmeans_init = vq_kmeans_init
        self.vq_threshold_ema_dead_code = vq_threshold_ema_dead_code
        self.vq_quantizer_dropout = vq_quantizer_dropout


class ActionCodecConfigOld(PretrainedConfig):
    model_type = "action_codec"

    def __init__(
        self,
        horizon: int = 20,
        action_dim: int = 7,
        action_encoding: str = "independent_v2",
        horizon_patch_size: int = 1,
        encoder_class: str = "actioncodec.modules.perceiver.PerceiverEncoder",
        decoder_class: str = "actioncodec.modules.perceiver.PerceiverDecoder",
        vq_class: str = "vector_quantize_pytorch.VectorQuantize",
        encoder_kwargs: Dict[str, Any] = None,
        decoder_kwargs: Dict[str, Any] = None,
        vq_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.horizon = horizon
        self.action_dim = action_dim
        self.action_encoding = action_encoding
        self.horizon_patch_size = horizon_patch_size
        self.encoder_class = encoder_class
        self.decoder_class = decoder_class
        self.vq_class = vq_class
        self.encoder_kwargs = (
            dict(encoder_kwargs)
            if encoder_kwargs is not None
            else {
                "dim": 384,
                "in_len": horizon,
                "out_len": 16,
                "num_layers": 12,
                "num_heads": 4,
                "output_round": -1.0,
            }
        )
        self.decoder_kwargs = (
            dict(decoder_kwargs)
            if decoder_kwargs is not None
            else {
                "dim": 384,
                "in_len": 16,
                "out_len": horizon,
                "num_layers": 12,
                "num_heads": 4,
            }
        )
        self.vq_kwargs = (
            dict(vq_kwargs)
            if vq_kwargs is not None
            else {
                "dim": 512,
                "codebook_size": 2048,
                "kmeans_init": True,
                "kmeans_iters": 10,
                "decay": 0.99,
                "commitment_weight": 0.25,
                "rotation_trick": False,
                "threshold_ema_dead_code": 2,
                "use_cosine_sim": False,
                "codebook_diversity_loss_weight": 0.0,
            }
        )


class BPEActionCodecConfig(PretrainedConfig):
    model_type = "bpe_action_codec"

    def __init__(
        self,
        horizon: int = 20,
        action_dim: int = 7,
        action_encoding: str = "independent_v2",
        horizon_patch_size: int = 1,
        encoder_class: str = "actioncodec.modules.perceiver.PerceiverEncoder",
        decoder_class: str = "actioncodec.modules.perceiver.PerceiverDecoder",
        vq_class: str = "vector_quantize_pytorch.VectorQuantize",
        encoder_kwargs: Dict[str, Any] = None,
        decoder_kwargs: Dict[str, Any] = None,
        vq_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.horizon = horizon
        self.action_dim = action_dim
        self.action_encoding = action_encoding
        self.horizon_patch_size = horizon_patch_size
        self.encoder_class = encoder_class
        self.decoder_class = decoder_class
        self.vq_class = vq_class
        self.encoder_kwargs = (
            dict(encoder_kwargs)
            if encoder_kwargs is not None
            else {
                "dim": 384,
                "in_len": horizon,
                "out_len": 16,
                "num_layers": 12,
                "num_heads": 4,
                "output_round": -1.0,
            }
        )
        self.decoder_kwargs = (
            dict(decoder_kwargs)
            if decoder_kwargs is not None
            else {
                "dim": 384,
                "in_len": 16,
                "out_len": horizon,
                "num_layers": 12,
                "num_heads": 4,
            }
        )
        self.vq_kwargs = (
            dict(vq_kwargs)
            if vq_kwargs is not None
            else {
                "dim": 512,
                "codebook_size": 2048,
                "kmeans_init": True,
                "kmeans_iters": 10,
                "decay": 0.99,
                "commitment_weight": 0.25,
                "rotation_trick": False,
                "threshold_ema_dead_code": 2,
                "use_cosine_sim": False,
                "codebook_diversity_loss_weight": 0.0,
            }
        )


AutoConfig.register("action_codec", ActionCodecConfig)
AutoConfig.register("bpe_action_codec", BPEActionCodecConfig)

__all__ = ["ActionCodecConfig", "BPEActionCodecConfig"]
