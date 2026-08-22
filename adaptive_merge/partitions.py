"""Enumerate contiguous partitions of temporal positions."""

from __future__ import annotations

from .merge_ops import segments_from_lengths
from .segments import Segment


def compositions(n: int, k: int, max_span: int | None = None) -> list[tuple[int, ...]]:
    """All ways to write n as sum of k positive integers.

    ``max_span`` limits each part (plan §1: merge spans of 2–4, or leave length 1).
    Default ``None`` keeps the unrestricted oracle used by the live merge1 run.
    """

    if k <= 0 or n <= 0:
        return []
    if k == 1:
        if max_span is not None and n > max_span:
            return []
        return [(n,)]
    hi = n - k + 1
    if max_span is not None:
        hi = min(hi, max_span)
    out: list[tuple[int, ...]] = []
    for first in range(1, hi + 1):
        for rest in compositions(n - first, k - 1, max_span=max_span):
            out.append((first,) + rest)
    return out


def enumerate_partitions(
    n_positions: int,
    n_segments: int,
    max_span: int | None = None,
) -> list[list[Segment]]:
    """All contiguous partitions of n_positions into exactly n_segments segments."""

    if not (1 <= n_segments <= n_positions):
        return []
    return [
        segments_from_lengths(lengths)
        for lengths in compositions(n_positions, n_segments, max_span=max_span)
    ]
