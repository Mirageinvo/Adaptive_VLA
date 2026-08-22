"""Lightweight tests for AATM merge primitives."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_merge.merge_ops import expand_merged_latent, segments_from_lengths
from adaptive_merge.partitions import compositions, enumerate_partitions
from adaptive_merge.segments import fixed_pair_segments, identity_segments, uniform_subsample_indices


def test_compositions_sum():
    for k in (1, 4, 8, 16):
        parts = compositions(16, k)
        assert len(parts) > 0
        for part in parts:
            assert sum(part) == 16
            assert len(part) == k
            assert all(x >= 1 for x in part)


def test_enumerate_partitions_count():
    # C(15, 7) = 6435 contiguous partitions of 16 into 8 segments
    parts = enumerate_partitions(16, 8)
    assert len(parts) == 6435


def test_expand_merged_latent_preserves_shape():
    latent = torch.randn(2, 16, 32)
    merged = expand_merged_latent(latent, fixed_pair_segments(16))
    assert merged.shape == latent.shape
    for seg in fixed_pair_segments(16):
        pair = merged[:, seg.start : seg.end + 1, :]
        assert torch.allclose(pair[:, 0, :], pair[:, 1, :])


def test_identity_segments_no_change():
    latent = torch.randn(16, 32)
    out = expand_merged_latent(latent, identity_segments(16))
    assert torch.allclose(out, latent)


def test_uniform_subsample_indices():
    kept = uniform_subsample_indices(16, 8)
    assert len(kept) == 8
    assert kept[0] == 0
    assert kept[-1] == 15


def test_segments_from_lengths():
    segs = segments_from_lengths((2, 3, 11))
    assert len(segs) == 3
    assert segs[0].start == 0 and segs[-1].end == 15


def main() -> None:
    test_compositions_sum()
    test_enumerate_partitions_count()
    test_expand_merged_latent_preserves_shape()
    test_identity_segments_no_change()
    test_uniform_subsample_indices()
    test_segments_from_lengths()
    print("all aatm tests passed")


if __name__ == "__main__":
    main()
