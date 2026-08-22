"""Enumerate contiguous partitions of temporal positions."""

from __future__ import annotations

from .merge_ops import segments_from_lengths
from .segments import Segment


def compositions(n: int, k: int) -> list[tuple[int, ...]]:
    """All ways to write n as sum of k positive integers."""

    if k <= 0 or n <= 0:
        return []
    if k == 1:
        return [(n,)]
    out: list[tuple[int, ...]] = []
    for first in range(1, n - k + 2):
        for rest in compositions(n - first, k - 1):
            out.append((first,) + rest)
    return out


def enumerate_partitions(n_positions: int, n_segments: int) -> list[list[Segment]]:
    """All contiguous partitions of n_positions into exactly n_segments segments."""

    if not (1 <= n_segments <= n_positions):
        return []
    return [segments_from_lengths(lengths) for lengths in compositions(n_positions, n_segments)]
