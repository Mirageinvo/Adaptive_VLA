"""Oracle search over contiguous merge partitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .latent import eval_merge_partition
from .partitions import enumerate_partitions
from .segments import Segment


@dataclass
class OracleResult:
    n_segments: int
    segments: list[Segment]
    lengths: tuple[int, ...]
    metrics: dict[str, float]


def _n_positions(codes: torch.Tensor) -> int:
    if codes.ndim == 2:
        return int(codes.shape[0])
    if codes.ndim == 3:
        return int(codes.shape[1])
    raise ValueError(f"Expected codes [P, L] or [B, P, L], got {tuple(codes.shape)}")


def best_partition_for_budget(
    model,
    E: torch.Tensor,
    codes: torch.Tensor,
    target_actions: torch.Tensor,
    n_segments: int,
    embodiment_id: int = 0,
) -> OracleResult:
    """Find the contiguous partition with minimum reconstruction RMS."""

    n_pos = _n_positions(codes)
    best: OracleResult | None = None
    for segments in enumerate_partitions(n_pos, n_segments):
        metrics = eval_merge_partition(
            model,
            E,
            codes,
            target_actions,
            segments,
            embodiment_id=embodiment_id,
        )
        candidate = OracleResult(
            n_segments=n_segments,
            segments=segments,
            lengths=tuple(seg.length for seg in segments),
            metrics=metrics,
        )
        if best is None or metrics["rms"] < best.metrics["rms"]:
            best = candidate
    if best is None:
        raise RuntimeError(f"No partitions found for n_segments={n_segments}")
    return best


def oracle_curve_for_chunk(
    model,
    E: torch.Tensor,
    codes: torch.Tensor,
    target_actions: torch.Tensor,
    segment_budgets: list[int],
    embodiment_id: int = 0,
) -> dict[int, OracleResult]:
    """Oracle best partition for each segment budget."""

    n_pos = _n_positions(codes)
    out: dict[int, OracleResult] = {}
    for k in segment_budgets:
        if k < 1 or k > n_pos:
            raise ValueError(f"segment budget {k} out of range for n_positions={n_pos}")
        out[k] = best_partition_for_budget(
            model,
            E,
            codes,
            target_actions,
            n_segments=k,
            embodiment_id=embodiment_id,
        )
    return out
