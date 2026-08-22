"""Latent merge / expand operations for temporal action positions."""

from __future__ import annotations

import torch

from .segments import Segment


def segments_from_lengths(lengths: tuple[int, ...]) -> list[Segment]:
    """Build contiguous segments from positive part lengths."""

    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError(f"Invalid segment lengths: {lengths}")
    segments: list[Segment] = []
    start = 0
    for length in lengths:
        end = start + length - 1
        segments.append(Segment(start, end))
        start = end + 1
    return segments


def expand_merged_latent(
    latent: torch.Tensor,
    segments: list[Segment],
    method: str = "mean",
) -> torch.Tensor:
    """Replace each segment with a merged latent and broadcast back to all positions.

    Args:
        latent: [B, P, D] or [P, D]
        segments: contiguous partition of [0, P)
        method: ``mean`` averages latents inside a segment
    """

    if method != "mean":
        raise ValueError(f"Unsupported merge method: {method}")
    if latent.ndim not in (2, 3):
        raise ValueError(f"Expected [P, D] or [B, P, D], got {tuple(latent.shape)}")

    batched = latent.ndim == 3
    work = latent.unsqueeze(0) if not batched else latent
    n_pos = work.shape[1]
    covered = sum(seg.length for seg in segments)
    if covered != n_pos:
        raise ValueError(f"Segments cover {covered} positions, expected {n_pos}")

    out = work.clone()
    for seg in segments:
        sl = slice(seg.start, seg.end + 1)
        merged = work[:, sl, :].mean(dim=1, keepdim=True)
        out[:, sl, :] = merged
    return out.squeeze(0) if not batched else out


def expand_subsampled_latent(latent: torch.Tensor, kept_indices: list[int]) -> torch.Tensor:
    """Uniform subsample baseline: copy nearest kept latent to every position."""

    if latent.ndim not in (2, 3):
        raise ValueError(f"Expected [P, D] or [B, P, D], got {tuple(latent.shape)}")
    batched = latent.ndim == 3
    work = latent.unsqueeze(0) if not batched else latent
    n_pos = work.shape[1]
    if not kept_indices:
        raise ValueError("kept_indices must be non-empty")
    kept = sorted(set(int(i) for i in kept_indices))
    if kept[0] < 0 or kept[-1] >= n_pos:
        raise ValueError(f"kept_indices out of range for n_positions={n_pos}: {kept}")

    out = work.clone()
    for pos in range(n_pos):
        nearest = min(kept, key=lambda k: abs(k - pos))
        out[:, pos, :] = work[:, nearest, :]
    return out.squeeze(0) if not batched else out
