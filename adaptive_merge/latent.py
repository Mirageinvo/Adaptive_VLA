"""ActionCodec latent helpers for merge experiments."""

from __future__ import annotations

import torch

from adaptive_rvq.codec import latent_from_depth
from adaptive_rvq.metrics import compute_metrics

from .merge_ops import expand_merged_latent, expand_subsampled_latent
from .segments import Segment


def full_depth_latent(E: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Return full-depth latent sum [B, P, D]."""

    depth = torch.full(codes.shape[:2], codes.shape[2], device=codes.device, dtype=torch.long)
    return latent_from_depth(E, codes, depth)


def decode_latent(model, latent: torch.Tensor, embodiment_id: int = 0) -> torch.Tensor:
    """Decode latent [B, P, D] to actions [B, T, 7]."""

    with torch.no_grad():
        rec, _ = model._decode(latent, embodiment_ids=embodiment_id, durations=None)
    return rec[..., :7]


def eval_merge_partition(
    model,
    E: torch.Tensor,
    codes: torch.Tensor,
    target_actions: torch.Tensor,
    segments: list[Segment],
    embodiment_id: int = 0,
) -> dict[str, float]:
    """Decode a merged latent partition and measure reconstruction error."""

    latent = full_depth_latent(E, codes.unsqueeze(0) if codes.ndim == 2 else codes)
    if codes.ndim == 2:
        target_actions = target_actions.unsqueeze(0)
    merged_latent = expand_merged_latent(latent, segments)
    rec = decode_latent(model, merged_latent, embodiment_id=embodiment_id)
    return compute_metrics(target_actions, rec)


def eval_subsample_partition(
    model,
    E: torch.Tensor,
    codes: torch.Tensor,
    target_actions: torch.Tensor,
    kept_indices: list[int],
    embodiment_id: int = 0,
) -> dict[str, float]:
    """Uniform subsample baseline via nearest-neighbor latent expansion."""

    latent = full_depth_latent(E, codes.unsqueeze(0) if codes.ndim == 2 else codes)
    if codes.ndim == 2:
        target_actions = target_actions.unsqueeze(0)
    expanded = expand_subsampled_latent(latent, kept_indices)
    rec = decode_latent(model, expanded, embodiment_id=embodiment_id)
    return compute_metrics(target_actions, rec)
