"""Reliability helpers for APB-RVQ runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.rate1_oracle_greedy import (  # noqa: E402
    CHECKPOINT_VERSION,
    EvalRow,
    _load_checkpoint,
    _run_fingerprint,
    _save_checkpoint,
    parse_args,
)


def test_run_fingerprint_stable():
    args = parse_args()
    args.n_chunks = 256
    args.budgets = "16,20,24"
    fp1 = _run_fingerprint(args, [16, 20, 24])
    fp2 = _run_fingerprint(args, [16, 20, 24])
    assert fp1 == fp2
    args.n_chunks = 512
    assert _run_fingerprint(args, [16, 20, 24]) != fp1


def test_checkpoint_roundtrip():
    args = parse_args()
    budgets = [16, 20, 24]
    fingerprint = _run_fingerprint(args, budgets)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        rows = [
            EvalRow(
                method="oracle",
                mode="exact-budget",
                budget=20,
                chunk_idx=0,
                episode_id=1,
                task_id=2,
                rms=0.1,
                mse=0.01,
                gripper_error=0.2,
            )
        ]
        arrays = {
            "global_rms_16": np.array([0.5]),
            "global_rms_32": np.array([0.4]),
            "global_rms_48": np.array([0.3]),
            "depth_maps_exact": np.zeros((1, len(budgets), 16), dtype=np.int16),
            "depth_maps_atmost": np.zeros((1, len(budgets), 16), dtype=np.int16),
            "train_idx": np.array([0], dtype=np.int64),
            "g12_sum": np.zeros(16),
            "g23_sum": np.zeros(16),
            "static_depth_16": np.ones(16, dtype=np.int16),
            "static_depth_20": np.ones(16, dtype=np.int16),
            "static_depth_24": np.ones(16, dtype=np.int16),
            "oracle_exact_16": np.array([]),
            "oracle_exact_20": np.array([0.1]),
            "oracle_exact_24": np.array([]),
            "oracle_atmost_16": np.array([]),
            "oracle_atmost_20": np.array([0.1]),
            "oracle_atmost_24": np.array([]),
            "random_16": np.array([]),
            "random_20": np.array([0.2]),
            "random_24": np.array([]),
            "uniform_16": np.array([]),
            "uniform_20": np.array([0.15]),
            "uniform_24": np.array([]),
            "static_16": np.array([]),
            "static_20": np.array([0.12]),
            "static_24": np.array([]),
            "global_mix_16": np.array([]),
            "global_mix_20": np.array([0.11]),
            "global_mix_24": np.array([]),
        }
        _save_checkpoint(
            out_dir,
            fingerprint=fingerprint,
            phase="oracle",
            meta_extra={"global_depth_done": 1, "static_prior_done": 1, "oracle_done": 1},
            arrays=arrays,
            rows=rows,
        )
        loaded = _load_checkpoint(out_dir, fingerprint, resume=True)
        assert loaded is not None
        assert loaded["meta"]["version"] == CHECKPOINT_VERSION
        assert loaded["meta"]["phase"] == "oracle"
        assert len(loaded["rows"]) == 1
        assert loaded["arrays"]["global_rms_16"][0] == 0.5


def test_preflight_strict_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "preflight.json"
        proc = subprocess.run(
            [
                sys.executable,
                "experiments/rate0_preflight.py",
                "--output",
                str(out),
                "--min-free-gb",
                "999999",
                "--strict",
            ],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["disk_guardrail_ok"] is False


def test_mean_random_per_chunk():
    from adaptive_rvq.metrics import mean_random_per_chunk

    # chunk-major, seed-minor: two chunks, three seeds
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    out = mean_random_per_chunk(values, n_chunks=2, n_seeds=3)
    assert np.allclose(out, [2.0, 5.0])
    try:
        mean_random_per_chunk(values, n_chunks=2, n_seeds=2)
        raise AssertionError("expected size mismatch error")
    except ValueError:
        pass


def test_bootstrap_ci_length_guard():
    from adaptive_rvq.metrics import bootstrap_ci

    try:
        bootstrap_ci(np.asarray([0.1, 0.2]), np.asarray([1, 2, 3]))
        raise AssertionError("expected length mismatch")
    except ValueError:
        pass


def main():
    test_run_fingerprint_stable()
    test_checkpoint_roundtrip()
    test_preflight_strict_blocks()
    test_mean_random_per_chunk()
    test_bootstrap_ci_length_guard()
    print("all reliability tests passed")


if __name__ == "__main__":
    main()
