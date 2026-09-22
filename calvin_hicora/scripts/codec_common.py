"""Shared ActionCodec construction, checkpoint, and metric helpers."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIONCODEC_ROOT = REPO_ROOT / "third_party" / "actioncodec"
if not (ACTIONCODEC_ROOT / "actioncodec" / "configuration_actioncodec.py").exists():
    raise FileNotFoundError(
        f"Expected vendored ActionCodec at {ACTIONCODEC_ROOT}/actioncodec/"
    )
if str(ACTIONCODEC_ROOT) not in sys.path:
    sys.path.insert(0, str(ACTIONCODEC_ROOT))

from actioncodec.configuration_actioncodec import ActionCodecConfig  # noqa: E402
from actioncodec.modeling_actioncodec import ActionCodec  # noqa: E402
from actioncodec import rvq as actioncodec_rvq  # noqa: E402


LIBERO_EMBODIMENT_ALIASES = (
    "a_franka_libero_20hz",
    "franka_libero_20hz",
)

_SINGLE_PROCESS_CODEBOOK_OPS_PATCHED = False


def ensure_single_process_codebook_ops() -> None:
    """Vendored VectorQuantize only runs k-means / dead-code refresh on DDP rank 0.

    On the registered single-GPU path ``dist`` is not initialized, so the else
    branch writes zeros and marks the book ``inited=True``. Patch that behavior
    without mutating the vendored file SHA.
    """
    global _SINGLE_PROCESS_CODEBOOK_OPS_PATCHED
    if _SINGLE_PROCESS_CODEBOOK_OPS_PATCHED:
        return

    dist = actioncodec_rvq.dist
    kmeans = actioncodec_rvq.kmeans
    sample_vectors = actioncodec_rvq.sample_vectors

    def init_codebook(self, encodings):  # type: ignore[no-untyped-def]
        if self.inited.item():
            return
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            embed, cluster_sizes = kmeans(
                encodings.float(), self.codebook_size, self.kmeans_iters
            )
        else:
            embed = torch.zeros(
                self.codebook_size, self.codebook_dim, device=encodings.device
            ).float()
            cluster_sizes = torch.zeros(
                self.codebook_size, device=encodings.device, dtype=torch.float32
            )
        if dist.is_initialized():
            dist.broadcast(embed, src=0)
            dist.broadcast(cluster_sizes, src=0)
        self.codebook.copy_(embed)
        self.embed_avg.copy_(embed.clone())
        self.cluster_size.copy_(cluster_sizes.float())
        self.inited.fill_(True)

    def replace_dead_codes(self, encodings):  # type: ignore[no-untyped-def]
        if self.threshold_ema_dead == 0:
            return
        dead_mask = self.cluster_size < self.threshold_ema_dead
        if not dead_mask.any():
            return
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            samples = sample_vectors(encodings.float(), self.codebook_size)
            print(f"Replace {dead_mask.sum().item()} dead codes")
        else:
            samples = torch.zeros_like(self.codebook).float()
        if dist.is_initialized():
            dist.broadcast(samples, src=0)
        self.codebook[dead_mask] = samples[: dead_mask.sum()].to(self.codebook.dtype)

    actioncodec_rvq.VectorQuantize.init_codebook = init_codebook
    actioncodec_rvq.VectorQuantize.replace_dead_codes = replace_dead_codes
    _SINGLE_PROCESS_CODEBOOK_OPS_PATCHED = True


def freeze_primary_codebook(quantizer: torch.nn.Module) -> None:
    """Hard-freeze primary RVQ book: no EMA, no dead-code refresh, snapshot assert."""
    quantizer.eval()
    quantizer._calvin_frozen_codebook = quantizer.codebook.detach().clone()
    quantizer._calvin_frozen_embed_avg = quantizer.embed_avg.detach().clone()
    quantizer._calvin_frozen_cluster_size = quantizer.cluster_size.detach().clone()
    quantizer.ema_update = lambda *args, **kwargs: None  # type: ignore[method-assign]
    quantizer.replace_dead_codes = lambda *args, **kwargs: None  # type: ignore[method-assign]
    quantizer.init_codebook = lambda encodings: None  # type: ignore[method-assign]


def assert_residual_codebooks_initialized(model: ActionCodec) -> dict[str, Any]:
    if model.config.vq_type != "rvq":
        raise ValueError("Residual codebook check requires RVQ")
    report: dict[str, Any] = {"levels": []}
    for index, quantizer in enumerate(model.vq.quantizers[1:], start=1):
        unique_rows = int(torch.unique(quantizer.codebook, dim=0).shape[0])
        norm = float(quantizer.codebook.norm().item())
        level = {
            "level": index,
            "inited": bool(quantizer.inited.item()),
            "unique_rows": unique_rows,
            "codebook_norm": norm,
        }
        report["levels"].append(level)
        if not level["inited"] or unique_rows < 2 or norm <= 0.0:
            raise RuntimeError(f"Residual codebook {index} is degenerate: {level}")
    report["passed"] = True
    return report


@torch.no_grad()
def warm_residual_codebook_init(
    model: ActionCodec,
    actions: torch.Tensor,
    embodiment_id: int,
    padding_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run one quantize pass so residual books k-means-init on real actions."""
    ensure_single_process_codebook_ops()
    if model.config.vq_type != "rvq":
        raise ValueError("warm residual init requires RVQ")
    for quantizer in model.vq.quantizers[1:]:
        quantizer.kmeans_init = True
        quantizer.inited.fill_(False)
    was_training = model.training
    model.eval()
    ids = torch.full(
        (actions.shape[0],), embodiment_id, dtype=torch.long, device=actions.device
    )
    z_e = model._encode(actions, ids, padding_mask)
    _ = model.vq(z_e)
    used_randn_fallback = False
    for quantizer in model.vq.quantizers[1:]:
        unique_rows = int(torch.unique(quantizer.codebook, dim=0).shape[0])
        norm = float(quantizer.codebook.norm().item())
        if (not bool(quantizer.inited.item())) or unique_rows < 2 or norm <= 0.0:
            # Near-zero residuals (common after a tiny/smoke base VQ) collapse
            # k-means to the zero book. Re-seed so residual EMA can learn.
            quantizer.codebook.copy_(torch.randn_like(quantizer.codebook))
            quantizer.embed_avg.copy_(quantizer.codebook.detach().clone())
            quantizer.cluster_size.zero_()
            quantizer.inited.fill_(True)
            used_randn_fallback = True
    report = assert_residual_codebooks_initialized(model)
    report["used_randn_fallback"] = used_randn_fallback
    if was_training:
        model.train()
        enforce_rvq_frozen_primary(model)
    else:
        model.eval()
        enforce_rvq_frozen_primary(model)
    return report


def assert_dataset_matches_protocol(
    manifest: dict[str, Any], protocol: dict[str, Any]
) -> None:
    geometry = protocol["geometry"]
    representation = protocol["action_representation"]
    semantics = manifest.get("action_semantics", {})
    checks = [
        ("action_key", manifest.get("action_key"), representation["source_key"]),
        ("chunk_length", manifest.get("chunk_length"), geometry["chunk_length"]),
        (
            "execution_horizon",
            manifest.get("execution_horizon"),
            geometry["execution_horizon"],
        ),
        ("clip_min", semantics.get("clip", [None, None])[0], representation["clip_min"]),
        ("clip_max", semantics.get("clip", [None, None])[1], representation["clip_max"]),
        (
            "flip_gripper_sign",
            semantics.get("flip_gripper_sign"),
            representation["flip_gripper_sign"],
        ),
    ]
    if int(manifest.get("format_version", 0)) >= 3:
        checks.extend(
            [
                (
                    "short_episode_padding",
                    manifest.get("short_episode_padding"),
                    geometry["short_episode_padding"],
                ),
                (
                    "padding_mask_semantics",
                    manifest.get("padding_mask_semantics"),
                    geometry["padding_mask_semantics"],
                ),
            ]
        )
    mismatches = [
        f"{name}: manifest={actual!r} protocol={expected!r}"
        for name, actual, expected in checks
        if actual != expected
    ]
    if mismatches:
        raise RuntimeError(
            "Converted dataset does not match frozen protocol: "
            + "; ".join(mismatches)
        )


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def sha1_file(path: str | Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calvin_embodiment() -> dict[str, dict[str, Any]]:
    return {
        "franka_calvin_30hz": {
            "action_dim": 7,
            "freq": 30,
            "duration": 1,
            "description": (
                "CALVIN 30 Hz relative Cartesian xyz/rpy and binary gripper for one second"
            ),
        }
    }


def _architecture(protocol: dict[str, Any]) -> dict[str, Any]:
    if "architecture" not in protocol:
        raise KeyError("codec protocol missing architecture block (source of truth)")
    return protocol["architecture"]


def build_base_vq_codec(protocol: dict[str, Any]) -> tuple[ActionCodec, int]:
    """Build CALVIN-native single-layer VQ from the frozen protocol architecture."""
    ensure_single_process_codebook_ops()
    geometry = protocol["geometry"]
    arch = _architecture(protocol)
    config = ActionCodecConfig(
        embodiment_config=calvin_embodiment(),
        n_tokens=int(geometry["latent_positions"]),
        n_quantizers=1,
        z_dim=int(geometry["latent_dim"]),
        vq_type="vq",
        vq_codebook_size=int(geometry["codebook_size"]),
        vq_commitment_weight=float(arch["vq_commitment_weight"]),
        vq_decay=float(arch["vq_decay"]),
        vq_kmeans_init=bool(arch["vq_kmeans_init"]),
        vq_threshold_ema_dead_code=int(arch["vq_threshold_ema_dead_code"]),
        encoder_dim=int(arch["encoder_dim"]),
        encoder_n_layers=int(arch["encoder_n_layers"]),
        encoder_n_heads=int(arch["encoder_n_heads"]),
        encoder_add_self_attn=bool(arch["encoder_add_self_attn"]),
        encoder_add_causal_mask=bool(arch["encoder_add_causal_mask"]),
        encoder_pos_encoding_type=str(arch["encoder_pos_encoding_type"]),
        decoder_dim=int(arch["decoder_dim"]),
        decoder_n_layers=int(arch["decoder_n_layers"]),
        decoder_n_heads=int(arch["decoder_n_heads"]),
        decoder_add_self_attn=bool(arch["decoder_add_self_attn"]),
        decoder_add_causal_mask=bool(arch["decoder_add_causal_mask"]),
        decoder_pos_encoding_type=str(arch["decoder_pos_encoding_type"]),
        decoder_cls_size=int(arch["decoder_cls_size"]),
    )
    return ActionCodec(config), 0


def resolve_libero_embodiment_id(model: ActionCodec) -> tuple[int, str]:
    keys = list(model.config.embodiment_config)
    for alias in LIBERO_EMBODIMENT_ALIASES:
        if alias in keys:
            return keys.index(alias), alias
    raise ValueError(
        "Pretrained codec has no LIBERO Franka embodiment; "
        f"expected one of {LIBERO_EMBODIMENT_ALIASES}, got {keys}"
    )


@torch.no_grad()
def initialize_rvq_from_base(
    base: ActionCodec, protocol: dict[str, Any]
) -> tuple[ActionCodec, int, dict[str, Any]]:
    """Apply the exact ActionCodec Appendix C VQ -> RVQ initialization."""
    ensure_single_process_codebook_ops()
    geometry = protocol["geometry"]
    stage = protocol["training"]["codec"]["rvq_posttrain"]
    if not stage.get("freeze_encoder", True) or not stage.get(
        "freeze_primary_codebook", True
    ):
        raise RuntimeError(
            "Registered Appendix C path requires freeze_encoder and "
            "freeze_primary_codebook; refusing a divergent protocol"
        )
    if base.num_quantizers != 1 or base.config.vq_type != "vq":
        raise ValueError("RVQ post-training requires a single-layer VQ checkpoint")
    config = ActionCodecConfig(
        embodiment_config=base.config.embodiment_config,
        n_tokens=int(geometry["total_tokens"]),
        n_quantizers=int(geometry["rvq_levels"]),
        z_dim=int(geometry["latent_dim"]),
        vq_type="rvq",
        vq_codebook_size=int(geometry["codebook_size"]),
        vq_commitment_weight=float(base.config.vq_commitment_weight),
        vq_decay=float(base.config.vq_decay),
        vq_kmeans_init=bool(base.config.vq_kmeans_init),
        vq_threshold_ema_dead_code=int(base.config.vq_threshold_ema_dead_code),
        vq_quantizer_dropout=float(geometry["quantizer_dropout"]),
        encoder_dim=int(base.config.encoder_dim),
        encoder_n_layers=int(base.config.encoder_n_layers),
        encoder_n_heads=int(base.config.encoder_n_heads),
        encoder_add_self_attn=bool(base.config.encoder_add_self_attn),
        encoder_add_causal_mask=bool(base.config.encoder_add_causal_mask),
        encoder_pos_encoding_type=str(base.config.encoder_pos_encoding_type),
        decoder_dim=int(base.config.decoder_dim),
        decoder_n_layers=int(base.config.decoder_n_layers),
        decoder_n_heads=int(base.config.decoder_n_heads),
        decoder_add_self_attn=bool(base.config.decoder_add_self_attn),
        decoder_add_causal_mask=bool(base.config.decoder_add_causal_mask),
        decoder_pos_encoding_type=str(base.config.decoder_pos_encoding_type),
        decoder_cls_size=int(base.config.decoder_cls_size),
    )
    rvq = ActionCodec(config)
    rvq.encoder.load_state_dict(base.encoder.state_dict(), strict=True)
    rvq.decoder.load_state_dict(base.decoder.state_dict(), strict=True)

    source = base.vq._codebook
    primary = rvq.vq.quantizers[0]
    primary.codebook.copy_(source.embed[0])
    primary.embed_avg.copy_(source.embed_avg[0])
    primary.cluster_size.copy_(source.cluster_size[0])
    primary.inited.fill_(bool(source.initted.item()))
    # Align residual hyperparams that ActionCodec's RVQ ctor does not forward.
    # Residual books must k-means-init from real actions after inheritance even
    # when the base VQ checkpoint used kmeans_init=False in a unit test.
    for quantizer in rvq.vq.quantizers[1:]:
        quantizer.decay = float(base.config.vq_decay)
        quantizer.threshold_ema_dead = int(base.config.vq_threshold_ema_dead_code)
        quantizer.kmeans_init = True
        quantizer.inited.fill_(False)
        quantizer.codebook.zero_()
        quantizer.embed_avg.zero_()
        quantizer.cluster_size.zero_()
    primary.decay = float(base.config.vq_decay)
    primary.threshold_ema_dead = int(base.config.vq_threshold_ema_dead_code)

    for parameter in rvq.encoder.parameters():
        parameter.requires_grad_(False)
    freeze_primary_codebook(primary)

    embodiment_id = list(config.embodiment_config).index("franka_calvin_30hz")
    base.eval()
    rvq.eval()
    # Suppress residual k-means during the primary-token equality probe so the
    # random probe batch cannot zero-init residual books.
    for quantizer in rvq.vq.quantizers[1:]:
        quantizer.inited.fill_(True)
    horizon = int(geometry["chunk_length"])
    action = torch.randn(2, horizon, int(geometry["action_dim"]))
    base_codes = torch.as_tensor(base.encode(action, embodiment_ids=embodiment_id))
    rvq_codes = torch.as_tensor(rvq.encode(action, embodiment_ids=embodiment_id))
    primary_equal = torch.equal(base_codes, rvq_codes[:, : int(geometry["latent_positions"])])
    for quantizer in rvq.vq.quantizers[1:]:
        quantizer.inited.fill_(False)
    report = {
        "encoder_inherited_exactly": all(
            torch.equal(value, rvq.encoder.state_dict()[key])
            for key, value in base.encoder.state_dict().items()
        ),
        "decoder_inherited_exactly": all(
            torch.equal(value, rvq.decoder.state_dict()[key])
            for key, value in base.decoder.state_dict().items()
        ),
        "primary_codebook_inherited_exactly": bool(
            torch.equal(source.embed[0], primary.codebook)
        ),
        "primary_tokens_equal_before_posttraining": bool(primary_equal),
        "encoder_trainable_parameters": sum(
            parameter.numel() for parameter in rvq.encoder.parameters() if parameter.requires_grad
        ),
        "single_process_codebook_ops_patched": _SINGLE_PROCESS_CODEBOOK_OPS_PATCHED,
        "primary_hard_frozen": hasattr(primary, "_calvin_frozen_codebook"),
    }
    passed = (
        report["encoder_inherited_exactly"]
        and report["decoder_inherited_exactly"]
        and report["primary_codebook_inherited_exactly"]
        and report["primary_tokens_equal_before_posttraining"]
        and report["encoder_trainable_parameters"] == 0
        and report["primary_hard_frozen"]
    )
    report["passed"] = bool(passed)
    if not passed:
        raise RuntimeError(f"VQ -> RVQ inheritance gate failed: {report}")
    rvq.train()
    enforce_rvq_frozen_primary(rvq)
    return rvq, embodiment_id, report


def enforce_rvq_frozen_primary(model: ActionCodec) -> None:
    """Keep Appendix C frozen components frozen after every model.train()."""
    if model.config.vq_type != "rvq":
        return
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    primary = model.vq.quantizers[0]
    if not hasattr(primary, "_calvin_frozen_codebook"):
        freeze_primary_codebook(primary)
    else:
        primary.eval()
        # model.to(device) moves codebook but not the non-Parameter freeze snapshot
        frozen = primary._calvin_frozen_codebook.to(
            device=primary.codebook.device, dtype=primary.codebook.dtype
        )
        primary._calvin_frozen_codebook = frozen
        if not torch.equal(primary.codebook, frozen):
            raise RuntimeError(
                "Primary codebook mutated under Appendix C freeze contract"
            )
        if primary.training:
            primary.eval()


@torch.no_grad()
def expand_pretrained_with_gate(
    model: ActionCodec, protocol: dict[str, Any], device: torch.device
) -> tuple[int, dict[str, Any]]:
    """Descriptive pretrained-transfer gate only; not the primary CALVIN codec path.

    Probe convention matches ActionCodec: (B, horizon, max_action_dim) with
    zero-padding beyond the LIBERO action_dim. All current released embodiments
    are 7-D, so padding is usually a no-op, but the gate still constructs the
    tensor that way.
    """
    if "franka_calvin_30hz" in model.config.embodiment_config:
        raise ValueError("Pretrained codec already contains franka_calvin_30hz")

    model = model.to(device).eval()
    old_id, old_name = resolve_libero_embodiment_id(model)
    old_cfg = model.config.embodiment_config[old_name]
    old_horizon = int(old_cfg["freq"] * old_cfg["duration"])
    old_action_dim = int(old_cfg["action_dim"])
    max_action_dim = int(model.encoder.max_action_dim)
    generator = torch.Generator(device=device).manual_seed(1729)
    probe = torch.zeros(2, old_horizon, max_action_dim, device=device)
    probe[..., :old_action_dim] = torch.randn(
        2, old_horizon, old_action_dim, generator=generator, device=device
    )
    input_shape_before = tuple(model.encoder.input_proj.weight.shape)
    z_before = model._encode(probe, old_id).clone()
    decoded_before, mask_before = model._decode(z_before, old_id)

    model.expand_embodiment(calvin_embodiment())

    input_shape_after = tuple(model.encoder.input_proj.weight.shape)
    z_after = model._encode(probe, old_id)
    decoded_after, mask_after = model._decode(z_before, old_id)
    report = {
        "libero_embodiment_name": old_name,
        "libero_embodiment_id": old_id,
        "probe_shape": list(probe.shape),
        "old_action_dim": old_action_dim,
        "max_action_dim": max_action_dim,
        "input_projection_shape_before": input_shape_before,
        "input_projection_shape_after": input_shape_after,
        "encoder_latent_max_abs_delta": float((z_before - z_after).abs().max().item()),
        "encoder_latent_bitwise_equal": bool(torch.equal(z_before, z_after)),
        "decoder_output_bitwise_equal": bool(torch.equal(decoded_before, decoded_after)),
        "decoder_mask_bitwise_equal": bool(torch.equal(mask_before, mask_after)),
    }
    gate = protocol["expand_embodiment_gate"]
    passed = (
        report["encoder_latent_max_abs_delta"]
        <= float(gate["maximum_libero_encoder_latent_absolute_delta"])
        and (
            not gate["require_input_projection_shape_unchanged"]
            or input_shape_before == input_shape_after
        )
        and (
            not gate["require_libero_decoder_output_bitwise_equal"]
            or report["decoder_output_bitwise_equal"]
        )
        and report["decoder_mask_bitwise_equal"]
    )
    report["passed"] = passed
    if not passed:
        raise RuntimeError(f"expand_embodiment gate failed: {report}")
    calvin_id = list(model.config.embodiment_config).index("franka_calvin_30hz")
    return calvin_id, report


def encode_quantize_decode(
    model: ActionCodec,
    action: torch.Tensor,
    embodiment_id: int,
    padding_mask: torch.Tensor | None = None,
    n_quantizers: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = torch.full(
        (action.shape[0],), embodiment_id, dtype=torch.long, device=action.device
    )
    z_e = model._encode(action, ids, padding_mask)
    if n_quantizers is None:
        z_q, codes, _, quantization_loss = model._quantize(
            z_e, return_perplexity=False
        )
    else:
        if model.training:
            raise RuntimeError("n_quantizers evaluation requires model.eval()")
        z_q, codes, _, commitment_loss, codebook_loss = model.vq(
            z_e, n_quantizers=n_quantizers
        )
        quantization_loss = commitment_loss.mean() + codebook_loss.mean()
    reconstruction, reconstruction_mask = model._decode(z_q, ids)
    return reconstruction[..., :7], reconstruction_mask, codes, quantization_loss


def grouped_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    gripper_bce_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """CALVIN reconstruction loss: continuous MSE plus binary gripper BCE.

    The decoder's seventh output is treated as a logit. CALVIN gripper targets
    are mapped from {-1, +1} to {0, 1}; padded timesteps do not contribute.
    """
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            f"prediction and target must share shape (B,T,D), got "
            f"{prediction.shape} and {target.shape}"
        )
    if prediction.shape[-1] != 7:
        raise ValueError(f"CALVIN actions must have 7 channels, got {prediction.shape[-1]}")
    if mask.shape != prediction.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != {prediction.shape[:2]}")
    if gripper_bce_weight < 0:
        raise ValueError("gripper_bce_weight must be non-negative")

    valid = mask.to(dtype=prediction.dtype).unsqueeze(-1)
    denom_steps = valid.sum().clamp_min(1)
    squared = (prediction - target).square() * valid
    position = squared[..., :3].sum() / (denom_steps * 3)
    rotation = squared[..., 3:6].sum() / (denom_steps * 3)
    continuous_mse = squared[..., :6].sum() / (denom_steps * 6)

    gripper_target = (target[..., 6] + 1.0) * 0.5
    if not bool(((gripper_target == 0) | (gripper_target == 1)).all()):
        raise ValueError("CALVIN gripper targets must be exactly -1 or +1")
    gripper_bce_unreduced = F.binary_cross_entropy_with_logits(
        prediction[..., 6], gripper_target, reduction="none"
    )
    gripper_bce = (
        gripper_bce_unreduced * mask.to(dtype=prediction.dtype)
    ).sum() / denom_steps
    total = continuous_mse + float(gripper_bce_weight) * gripper_bce
    return total, {
        "position": position,
        "rotation": rotation,
        "continuous_mse": continuous_mse,
        "gripper_bce": gripper_bce,
    }


def cosine_with_warmup_factor(
    step: int, warmup_steps: int, max_steps: int, end_factor: float = 0.1
) -> float:
    if step < warmup_steps:
        start_factor = 1e-6
        return start_factor + (1 - start_factor) * step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    return end_factor + (1 - end_factor) * 0.5 * (1 + math.cos(math.pi * progress))
