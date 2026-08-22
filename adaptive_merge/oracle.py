"""Oracle search over contiguous merge partitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .latent import eval_merge_partition
from .partitions import enumerate_partitions
from .segments import Segment, identity_segments


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
    max_span: int | None = None,
) -> OracleResult:
    """Find the contiguous partition with minimum reconstruction RMS.

    ``max_span=None`` is the unrestricted search used by the live merge1 job.
    Pass ``max_span=4`` for the plan §1 constraint (segments of length 1–4).
    """

    n_pos = _n_positions(codes)
    best: OracleResult | None = None
    for segments in enumerate_partitions(n_pos, n_segments, max_span=max_span):
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
    max_span: int | None = None,
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
            max_span=max_span,
        )
    return out


def greedy_partition_for_budget(
    model,
    E: torch.Tensor,
    codes: torch.Tensor,
    target_actions: torch.Tensor,
    n_segments: int,
    embodiment_id: int = 0,
    max_span: int | None = None,
) -> OracleResult:
    """Greedy adjacent merges until ``n_segments`` remain (plan §8).

    Stops early if no legal adjacent merge remains under ``max_span``.
    ``OracleResult.n_segments`` is ``len(final segs)`` (may exceed the request).
    Metrics are always recomputed on that final segmentation.
    """

    n_pos = _n_positions(codes)
    if not (1 <= n_segments <= n_pos):
        raise ValueError(f"n_segments={n_segments} out of range for n_positions={n_pos}")
    segs = identity_segments(n_pos)
    while len(segs) > n_segments:
        best: tuple[float, list[Segment]] | None = None
        for i in range(len(segs) - 1):
            merged_len = segs[i].length + segs[i + 1].length
            if max_span is not None and merged_len > max_span:
                continue
            trial = segs[:i] + [Segment(segs[i].start, segs[i + 1].end)] + segs[i + 2 :]
            trial_metrics = eval_merge_partition(
                model, E, codes, target_actions, trial, embodiment_id=embodiment_id
            )
            if best is None or trial_metrics["rms"] < best[0]:
                best = (trial_metrics["rms"], trial)
        if best is None:
            break
        segs = best[1]
    metrics = eval_merge_partition(model, E, codes, target_actions, segs, embodiment_id=embodiment_id)
    return OracleResult(
        n_segments=len(segs),
        segments=segs,
        lengths=tuple(seg.length for seg in segs),
        metrics=metrics,
    )
