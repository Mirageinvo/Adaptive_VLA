"""Lightweight tests for AATM merge primitives."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_merge.merge_ops import expand_merged_latent, segments_from_lengths
from adaptive_merge.oracle import greedy_partition_for_budget
from adaptive_merge.partitions import compositions, enumerate_partitions
from adaptive_merge.plan_gaps import (
    adjacent_merge_mask,
    auroc,
    episode_grouped_split,
    filter_missing_episodes,
    lengths_to_position_ids,
    parse_oracle_lengths,
    rel_increase_pct,
    retained_gain,
    segment_length_by_start,
    span_is_plan_legal,
    spearman_corr,
    summarize_oracle_spans,
)
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


def test_max_span_compositions_are_subset():
    free = compositions(16, 8)
    limited = compositions(16, 8, max_span=4)
    assert 0 < len(limited) < len(free)
    assert all(max(part) <= 4 for part in limited)
    assert all(sum(part) == 16 and len(part) == 8 for part in limited)


def test_parse_and_span_legal():
    assert parse_oracle_lengths("2,2,2,2,2,2,2,2") == (2,) * 8
    assert parse_oracle_lengths("seed=3") is None
    assert parse_oracle_lengths("2x8") is None
    assert span_is_plan_legal((2, 2, 4, 1, 3, 1, 2, 1), max_span=4)
    assert not span_is_plan_legal((13, 1, 1, 1), max_span=4)


def test_span_summary_and_episode_split():
    length_by_chunk = {0: (2,) * 8, 1: (13, 1, 1, 1), 2: (4, 4, 4, 4)}
    rms_by_chunk = {0: 0.1, 1: 0.2, 2: 0.15}
    stats = summarize_oracle_spans([0, 1, 2], length_by_chunk, rms_by_chunk, max_span=4)
    assert stats["n_chunks_illegal_span"] == 1
    assert abs(stats["pct_chunks_plan_legal_span"] - 2 / 3) < 1e-9
    train, val, meta = episode_grouped_split(np.array([1, 1, 2, 2, 3, 3]), val_frac=0.34, seed=0)
    assert not meta["overlap_episodes"]
    assert set(train.tolist()).isdisjoint(set(val.tolist()))


def test_adjacent_merge_mask_and_retained_gain():
    mask = adjacent_merge_mask((2, 1, 13))
    assert mask.tolist() == [True, False, False] + [True] * 12
    ids = lengths_to_position_ids((2, 2, 12))
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[1] != ids[2]
    assert abs(retained_gain(0.12, 0.10, 0.20) - 0.8) < 1e-9
    assert abs(rel_increase_pct(0.12, 0.10) - 20.0) < 1e-9


def test_auroc_spearman_and_start_lengths():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == 1.0
    assert auroc(-scores, labels) == 0.0
    assert spearman_corr(np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 9.0])) > 0.99
    by_start = segment_length_by_start([0], {0: (2, 2, 12)})
    assert by_start["mean_length_if_segment_starts_here"][0] == 2
    assert by_start["mean_length_if_segment_starts_here"][2] == 2
    assert by_start["mean_length_if_segment_starts_here"][4] == 12
    assert by_start["n_segments_starting_here"][1] == 0


def test_filter_missing_episodes():
    kept, eps, excluded = filter_missing_episodes({0: 10, 2: 11}, [0, 1, 2, 3])
    assert kept.tolist() == [0, 2]
    assert eps.tolist() == [10, 11]
    assert excluded == [1, 3]
    assert -1 not in eps.tolist()
    kept_empty, eps_empty, excluded_all = filter_missing_episodes({}, [0, 1])
    assert kept_empty.size == 0
    assert eps_empty.size == 0
    assert excluded_all == [0, 1]


def test_greedy_max_span_stall():
    """Identity start cannot merge when max_span=1; metrics from identity, no crash."""

    seen: list[tuple[int, ...]] = []
    identity_rms = 0.42

    def fake_eval(model, E, codes, target_actions, segments, embodiment_id=0):
        lengths = tuple(int(seg.length) for seg in segments)
        seen.append(lengths)
        return {"rms": identity_rms}

    codes = torch.zeros(4, 2)
    actions = torch.zeros(4, 7)
    E = torch.zeros(2, 8)
    with patch("adaptive_merge.oracle.eval_merge_partition", side_effect=fake_eval):
        result = greedy_partition_for_budget(
            None, E, codes, actions, n_segments=1, max_span=1
        )
    assert result.lengths == (1, 1, 1, 1)
    assert result.n_segments == 4
    assert result.metrics["rms"] == identity_rms
    assert seen
    assert all(lens == (1, 1, 1, 1) for lens in seen)


def main() -> None:
    test_compositions_sum()
    test_enumerate_partitions_count()
    test_expand_merged_latent_preserves_shape()
    test_identity_segments_no_change()
    test_uniform_subsample_indices()
    test_segments_from_lengths()
    test_max_span_compositions_are_subset()
    test_parse_and_span_legal()
    test_span_summary_and_episode_split()
    test_adjacent_merge_mask_and_retained_gain()
    test_auroc_spearman_and_start_lengths()
    test_filter_missing_episodes()
    test_greedy_max_span_stall()
    print("all aatm tests passed")


if __name__ == "__main__":
    main()
