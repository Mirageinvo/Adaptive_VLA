"""Segment definitions and fixed merge schemes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    """Inclusive temporal segment [start, end] over action positions."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid segment [{self.start}, {self.end}]")

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def identity_segments(n_positions: int) -> list[Segment]:
    """One segment per position (no merge)."""

    return [Segment(i, i) for i in range(n_positions)]


def fixed_pair_segments(n_positions: int) -> list[Segment]:
    """Merge each adjacent pair: 16 -> 8 segments of length 2."""

    if n_positions % 2 != 0:
        raise ValueError(f"n_positions={n_positions} must be even for pair merge")
    return [Segment(i, i + 1) for i in range(0, n_positions, 2)]


def uniform_subsample_indices(n_positions: int, n_kept: int) -> list[int]:
    """Evenly spaced kept indices for uniform subsampling baseline."""

    if not (1 <= n_kept <= n_positions):
        raise ValueError(f"n_kept={n_kept} must be in [1, {n_positions}]")
    if n_kept == n_positions:
        return list(range(n_positions))
    # linspace endpoints included, rounded to unique indices.
    raw = [int(round(x)) for x in np.linspace(0, n_positions - 1, n_kept)]
    kept: list[int] = []
    for idx in raw:
        if not kept or idx != kept[-1]:
            kept.append(min(max(idx, 0), n_positions - 1))
    while len(kept) < n_kept:
        for candidate in range(n_positions):
            if candidate not in kept:
                kept.append(candidate)
                if len(kept) == n_kept:
                    break
    return sorted(kept[:n_kept])
