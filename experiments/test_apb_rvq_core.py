"""Lightweight tests for APB-RVQ allocation primitives."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_rvq.allocation import (  # noqa: E402
    enumerate_depths_for_budget,
    greedy_oracle_single_mode,
    validate_depth_map,
)


def test_budget_exact():
    states = enumerate_depths_for_budget(n_positions=4, budget=8, max_depth=3)
    assert states, "no states found"
    for s in states:
        assert len(s) == 4
        assert sum(s) == 8
        assert all(1 <= x <= 3 for x in s)


def test_validate_depth_map():
    validate_depth_map(torch.tensor([1, 2, 3, 1]), budget=7, max_depth=3)
    try:
        validate_depth_map(torch.tensor([0, 2, 3]), budget=5, max_depth=3)
        raise AssertionError("expected validation error for value 0")
    except ValueError:
        pass


def test_greedy_recovers_additive_oracle():
    rng = np.random.default_rng(0)
    gains_l2 = rng.uniform(0.1, 1.0, size=4)
    gains_l3 = rng.uniform(0.05, 0.4, size=4)

    def score(depth):
        depth = np.asarray(depth.tolist())
        total = 10.0
        for i in range(4):
            if depth[i] >= 2:
                total -= gains_l2[i]
            if depth[i] >= 3:
                total -= gains_l3[i]
        return total

    out, _ = greedy_oracle_single_mode(score, n_positions=4, budgets=[4, 5, 6, 7, 8], mode="exact-budget")
    # Monotonic improvement in additive setup.
    errs = [out[b]["error"] for b in [4, 5, 6, 7, 8]]
    assert all(errs[i + 1] <= errs[i] for i in range(len(errs) - 1)), errs


def main():
    test_budget_exact()
    test_validate_depth_map()
    test_greedy_recovers_additive_oracle()
    print("all tests passed")


if __name__ == "__main__":
    main()
