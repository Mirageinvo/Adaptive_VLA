"""Adaptive Action-Token Merging (AATM) for ActionCodec temporal compression."""

from .merge_ops import expand_merged_latent, expand_subsampled_latent, segments_from_lengths
from .oracle import best_partition_for_budget, enumerate_partitions
from .segments import Segment, fixed_pair_segments, identity_segments, uniform_subsample_indices

__all__ = [
    "Segment",
    "best_partition_for_budget",
    "enumerate_partitions",
    "expand_merged_latent",
    "expand_subsampled_latent",
    "fixed_pair_segments",
    "identity_segments",
    "segments_from_lengths",
    "uniform_subsample_indices",
]
