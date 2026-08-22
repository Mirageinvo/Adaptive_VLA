#!/usr/bin/env python3
"""Merge-1 oracle compression curve and fixed vs adaptive baselines."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_merge.latent import eval_merge_partition, eval_subsample_partition
from adaptive_merge.oracle import best_partition_for_budget
from adaptive_merge.segments import fixed_pair_segments, identity_segments, uniform_subsample_indices
from adaptive_rvq.codec import encode_actions, load_codec, projected_codebooks
from adaptive_rvq.data import LIBERO_DATASET_ID, LIBERO_REVISION, load_libero_chunks
from adaptive_rvq.metrics import bootstrap_ci, mean_random_per_chunk


@dataclass
class EvalRow:
    method: str
    n_segments: int
    chunk_idx: int
    episode_id: int
    task_id: int
    rms: float
    mse: float
    gripper_error: float
    segment_lengths: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--dataset", default=LIBERO_DATASET_ID)
    ap.add_argument("--revision", default=LIBERO_REVISION)
    ap.add_argument("--n-chunks", type=int, default=256)
    ap.add_argument("--n-episodes", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--segment-budgets", default="16,14,12,10,8,6,4")
    ap.add_argument("--random-seeds", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="artifacts/merge/oracle")
    return ap.parse_args()


def _random_partition(n_positions: int, n_segments: int, rng: np.random.Generator) -> list[int]:
    """Sample random composition lengths summing to n_positions."""

    if n_segments == n_positions:
        return [1] * n_positions
    cuts = sorted(rng.choice(n_positions - 1, size=n_segments - 1, replace=False) + 1)
    bounds = [0, *cuts, n_positions]
    return [bounds[i + 1] - bounds[i] for i in range(n_segments)]


def _lengths_to_segments(lengths: list[int]):
    from adaptive_merge.merge_ops import segments_from_lengths

    return segments_from_lengths(tuple(lengths))


def main() -> None:
    args = parse_args()
    budgets = sorted(int(x) for x in args.segment_budgets.split(","), reverse=True)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_codec(model_id=args.model, device=args.device)
    batch = load_libero_chunks(
        n_chunks=args.n_chunks,
        n_episodes=args.n_episodes,
        seed=args.seed,
        dataset_id=args.dataset,
        revision=args.revision,
        device=args.device,
        gripper_mode=None,
        model=model,
        embodiment_id=args.embodiment,
    )
    codes = encode_actions(model, batch.actions, embodiment_id=args.embodiment)
    E = projected_codebooks(model, device=args.device)

    rows: list[EvalRow] = []
    oracle_by_budget: dict[int, list[float]] = {b: [] for b in budgets}
    fixed_pair_by_budget: dict[int, list[float]] = {b: [] for b in budgets}
    subsample_by_budget: dict[int, list[float]] = {b: [] for b in budgets}
    random_by_budget: dict[int, list[float]] = {b: [] for b in budgets}

    for chunk_idx in tqdm(range(codes.shape[0]), desc="merge1_oracle"):
        code_i = codes[chunk_idx]
        action_i = batch.actions[chunk_idx]
        episode_id = int(batch.episode_ids[chunk_idx])
        task_id = int(batch.task_ids[chunk_idx])

        baseline = eval_merge_partition(
            model, E, code_i, action_i, identity_segments(16), embodiment_id=args.embodiment
        )
        rows.append(
            EvalRow(
                method="no_merge",
                n_segments=16,
                chunk_idx=chunk_idx,
                episode_id=episode_id,
                task_id=task_id,
                rms=baseline["rms"],
                mse=baseline["mse"],
                gripper_error=baseline["gripper_error"],
                segment_lengths="1x16",
            )
        )

        for k in budgets:
            oracle = best_partition_for_budget(
                model, E, code_i, action_i, n_segments=k, embodiment_id=args.embodiment
            )
            oracle_by_budget[k].append(oracle.metrics["rms"])
            rows.append(
                EvalRow(
                    method="oracle_adaptive",
                    n_segments=k,
                    chunk_idx=chunk_idx,
                    episode_id=episode_id,
                    task_id=task_id,
                    rms=oracle.metrics["rms"],
                    mse=oracle.metrics["mse"],
                    gripper_error=oracle.metrics["gripper_error"],
                    segment_lengths=",".join(str(x) for x in oracle.lengths),
                )
            )

            if k == 8:
                fixed = eval_merge_partition(
                    model, E, code_i, action_i, fixed_pair_segments(16), embodiment_id=args.embodiment
                )
                fixed_pair_by_budget[k].append(fixed["rms"])
                rows.append(
                    EvalRow(
                        method="fixed_pair",
                        n_segments=8,
                        chunk_idx=chunk_idx,
                        episode_id=episode_id,
                        task_id=task_id,
                        rms=fixed["rms"],
                        mse=fixed["mse"],
                        gripper_error=fixed["gripper_error"],
                        segment_lengths="2x8",
                    )
                )

                kept = uniform_subsample_indices(16, k)
                sub = eval_subsample_partition(
                    model, E, code_i, action_i, kept, embodiment_id=args.embodiment
                )
                subsample_by_budget[k].append(sub["rms"])
                rows.append(
                    EvalRow(
                        method="uniform_subsample",
                        n_segments=8,
                        chunk_idx=chunk_idx,
                        episode_id=episode_id,
                        task_id=task_id,
                        rms=sub["rms"],
                        mse=sub["mse"],
                        gripper_error=sub["gripper_error"],
                        segment_lengths=f"kept={kept}",
                    )
                )

            for seed in range(args.random_seeds):
                rng = np.random.default_rng(args.seed + chunk_idx * 1000 + seed + k * 17)
                lengths = _random_partition(16, k, rng)
                rand = eval_merge_partition(
                    model, E, code_i, action_i, _lengths_to_segments(lengths), embodiment_id=args.embodiment
                )
                random_by_budget[k].append(rand["rms"])

    n_chunks = codes.shape[0]
    baseline_rms = np.asarray([r.rms for r in rows if r.method == "no_merge"], dtype=np.float64)
    baseline_mean = float(np.mean(baseline_rms))
    summary: dict[str, object] = {
        "experiment": "merge1_oracle_compression",
        "purpose": "ICRA Stage-1 oracle compression curve (preregistered)",
        "n_chunks": n_chunks,
        "segment_budgets": budgets,
        "baseline_rms_mean": baseline_mean,
        "oracle_curve": {},
        "adaptive_gain_vs_fixed_pair_k8": None,
        "pct_chunks_within_5pct_rms_at_budget": {},
        "stage1_gate": {},
    }
    for k in budgets:
        oracle_vals = np.asarray(oracle_by_budget[k], dtype=np.float64)
        oracle_mean = float(np.mean(oracle_vals))
        rel_increase = (oracle_mean - baseline_mean) / max(baseline_mean, 1e-12)
        lo, hi = bootstrap_ci(oracle_vals, batch.episode_ids[:n_chunks], seed=args.seed)
        entry = {
            "rms_mean": oracle_mean,
            "rms_median": float(np.median(oracle_vals)),
            "rel_error_increase_vs_no_merge": rel_increase,
            "bootstrap_ci": [lo, hi],
        }
        if k == 8 and fixed_pair_by_budget[k]:
            fixed_vals = np.asarray(fixed_pair_by_budget[k], dtype=np.float64)
            fixed_mean = float(np.mean(fixed_vals))
            gain = fixed_vals - oracle_vals
            gain_lo, gain_hi = bootstrap_ci(gain, batch.episode_ids[:n_chunks], seed=args.seed + 8)
            entry["fixed_pair_rms_mean"] = fixed_mean
            entry["uniform_subsample_rms_mean"] = float(np.mean(subsample_by_budget[k]))
            rand_mean = float(np.mean(mean_random_per_chunk(np.asarray(random_by_budget[k]), n_chunks, args.random_seeds)))
            entry["random_partition_rms_mean"] = rand_mean
            entry["adaptive_gain_vs_fixed_pair"] = fixed_mean - oracle_mean
            entry["adaptive_gain_vs_random"] = rand_mean - oracle_mean
            entry["adaptive_gain_bootstrap_ci"] = [gain_lo, gain_hi]
            summary["adaptive_gain_vs_fixed_pair_k8"] = entry["adaptive_gain_vs_fixed_pair"]

        summary["oracle_curve"][str(k)] = entry

        tol = 0.05
        ok = oracle_vals <= baseline_rms[:n_chunks] * (1.0 + tol) if len(baseline_rms) == n_chunks else oracle_vals <= baseline_mean * (1.0 + tol)
        compression_frac = 1.0 - (k / 16.0)
        summary["pct_chunks_within_5pct_rms_at_budget"][str(k)] = {
            "compression_fraction": compression_frac,
            "fraction": float(np.mean(ok)),
        }

    # Preregistered Stage-1 gate (ICRA)
    k8 = summary["oracle_curve"].get("8", {})
    k12 = summary["oracle_curve"].get("12", {})
    adaptive_gain = summary.get("adaptive_gain_vs_fixed_pair_k8")
    gain_ci = k8.get("adaptive_gain_bootstrap_ci") if isinstance(k8, dict) else None
    headroom_k8 = float(k8.get("rel_error_increase_vs_no_merge", 999)) if isinstance(k8, dict) else 999
    headroom_k12 = float(k12.get("rel_error_increase_vs_no_merge", 999)) if isinstance(k12, dict) else 999
    pct50 = summary["pct_chunks_within_5pct_rms_at_budget"].get("8", {}).get("fraction", 0.0)
    pct25 = summary["pct_chunks_within_5pct_rms_at_budget"].get("12", {}).get("fraction", 0.0)
    ci_excludes_zero = gain_ci is not None and gain_ci[0] > 0

    stage1_pass = (
        headroom_k8 <= 0.05
        and headroom_k12 <= 0.05
        and adaptive_gain is not None
        and adaptive_gain > 0
        and ci_excludes_zero
        and pct50 >= 0.50
    )
    stage1_kill = headroom_k8 > 0.20 or (adaptive_gain is not None and adaptive_gain <= 0)

    summary["stage1_gate"] = {
        "decision": "GO" if stage1_pass else ("KILL" if stage1_kill else "OPEN"),
        "criteria": {
            "headroom_k8_rel_rms_le_5pct": headroom_k8 <= 0.05,
            "headroom_k12_rel_rms_le_5pct": headroom_k12 <= 0.05,
            "adaptive_gain_k8_positive": adaptive_gain is not None and adaptive_gain > 0,
            "adaptive_gain_ci_excludes_zero": ci_excludes_zero,
            "pct_chunks_50pct_compression_at_5pct_tol": pct50 >= 0.50,
            "pct_chunks_25pct_compression_at_5pct_tol": pct25,
        },
        "note": "Stage-1 GO enables locality/labels; learned merger requires Stage-2 causal GO.",
    }

    pq.write_table(pa.Table.from_pylist([asdict(r) for r in rows]), out_dir / "eval_rows.parquet")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
