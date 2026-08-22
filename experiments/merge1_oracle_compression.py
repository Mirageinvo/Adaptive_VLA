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

N_POSITIONS = 16


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
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="artifacts/merge/oracle")
    return ap.parse_args()


def _random_partition(n_positions: int, n_segments: int, rng: np.random.Generator) -> list[int]:
    if n_segments == n_positions:
        return [1] * n_positions
    cuts = sorted(rng.choice(n_positions - 1, size=n_segments - 1, replace=False) + 1)
    bounds = [0, *cuts, n_positions]
    return [bounds[i + 1] - bounds[i] for i in range(n_segments)]


def _lengths_to_segments(lengths: list[int]):
    from adaptive_merge.merge_ops import segments_from_lengths

    return segments_from_lengths(tuple(lengths))


def summarize_rows(
    rows: list[EvalRow],
    episode_ids: list[int],
    chunk_ids: list[int],
    budgets: list[int],
    seed: int,
    random_seeds: int,
) -> dict[str, object]:
    """Build Stage-1 summary from eval rows (single process or merged shards)."""

    n_chunks = len(chunk_ids)
    baseline_by_chunk = {r.chunk_idx: r.rms for r in rows if r.method == "no_merge"}
    baseline_rms = np.asarray([baseline_by_chunk[i] for i in chunk_ids], dtype=np.float64)
    baseline_mean = float(np.mean(baseline_rms))

    oracle_by_budget: dict[int, list[float]] = {b: [] for b in budgets}
    for cid in chunk_ids:
        for k in budgets:
            match = [r for r in rows if r.method == "oracle_adaptive" and r.chunk_idx == cid and r.n_segments == k]
            if len(match) != 1:
                raise ValueError(f"Expected one oracle row for chunk={cid} k={k}, got {len(match)}")
            oracle_by_budget[k].append(match[0].rms)

    random_by_budget: dict[int, list[float]] = {b: [] for b in budgets}
    for cid in chunk_ids:
        for k in budgets:
            for s in range(random_seeds):
                match = [
                    r
                    for r in rows
                    if r.method == "random_partition"
                    and r.chunk_idx == cid
                    and r.n_segments == k
                    and r.segment_lengths == f"seed={s}"
                ]
                if len(match) != 1:
                    raise ValueError(f"Missing random row chunk={cid} k={k} seed={s}")
                random_by_budget[k].append(match[0].rms)

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

    episode_arr = np.asarray(episode_ids, dtype=np.int64)
    for k in budgets:
        oracle_vals = np.asarray(oracle_by_budget[k], dtype=np.float64)
        oracle_mean = float(np.mean(oracle_vals))
        rel_increase = (oracle_mean - baseline_mean) / max(baseline_mean, 1e-12)
        lo, hi = bootstrap_ci(oracle_vals, episode_arr, seed=seed)
        entry: dict[str, object] = {
            "rms_mean": oracle_mean,
            "rms_median": float(np.median(oracle_vals)),
            "rel_error_increase_vs_no_merge": rel_increase,
            "bootstrap_ci": [lo, hi],
        }

        if k == 8:
            fixed_vals = np.asarray(
                [r.rms for cid in chunk_ids for r in rows if r.method == "fixed_pair" and r.chunk_idx == cid],
                dtype=np.float64,
            )
            sub_vals = np.asarray(
                [
                    r.rms
                    for cid in chunk_ids
                    for r in rows
                    if r.method == "uniform_subsample" and r.chunk_idx == cid
                ],
                dtype=np.float64,
            )
            if fixed_vals.size == n_chunks:
                fixed_mean = float(np.mean(fixed_vals))
                gain = fixed_vals - oracle_vals
                gain_lo, gain_hi = bootstrap_ci(gain, episode_arr, seed=seed + 8)
                rand_mean = float(
                    np.mean(mean_random_per_chunk(np.asarray(random_by_budget[k]), n_chunks, random_seeds))
                )
                entry["fixed_pair_rms_mean"] = fixed_mean
                entry["uniform_subsample_rms_mean"] = float(np.mean(sub_vals))
                entry["random_partition_rms_mean"] = rand_mean
                entry["adaptive_gain_vs_fixed_pair"] = fixed_mean - oracle_mean
                entry["adaptive_gain_vs_random"] = rand_mean - oracle_mean
                entry["adaptive_gain_bootstrap_ci"] = [gain_lo, gain_hi]
                summary["adaptive_gain_vs_fixed_pair_k8"] = entry["adaptive_gain_vs_fixed_pair"]

        summary["oracle_curve"][str(k)] = entry
        tol = 0.05
        ok = oracle_vals <= baseline_rms * (1.0 + tol)
        summary["pct_chunks_within_5pct_rms_at_budget"][str(k)] = {
            "compression_fraction": 1.0 - (k / float(N_POSITIONS)),
            "fraction": float(np.mean(ok)),
        }

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
    return summary


def main() -> None:
    args = parse_args()
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"shard_id={args.shard_id} must be in [0, {args.num_shards})")

    budgets = sorted((int(x) for x in args.segment_budgets.split(",")), reverse=True)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

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

    chunk_indices = [i for i in range(codes.shape[0]) if i % args.num_shards == args.shard_id]
    rows: list[EvalRow] = []

    desc = f"merge1_oracle s{args.shard_id}/{args.num_shards}"
    for chunk_idx in tqdm(chunk_indices, desc=desc):
        code_i = codes[chunk_idx]
        action_i = batch.actions[chunk_idx]
        episode_id = int(batch.episode_ids[chunk_idx])
        task_id = int(batch.task_ids[chunk_idx])

        baseline = eval_merge_partition(
            model, E, code_i, action_i, identity_segments(N_POSITIONS), embodiment_id=args.embodiment
        )
        rows.append(
            EvalRow(
                method="no_merge",
                n_segments=N_POSITIONS,
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
                    model, E, code_i, action_i, fixed_pair_segments(N_POSITIONS), embodiment_id=args.embodiment
                )
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
                kept = uniform_subsample_indices(N_POSITIONS, k)
                sub = eval_subsample_partition(
                    model, E, code_i, action_i, kept, embodiment_id=args.embodiment
                )
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
                lengths = _random_partition(N_POSITIONS, k, rng)
                rand = eval_merge_partition(
                    model, E, code_i, action_i, _lengths_to_segments(lengths), embodiment_id=args.embodiment
                )
                rows.append(
                    EvalRow(
                        method="random_partition",
                        n_segments=k,
                        chunk_idx=chunk_idx,
                        episode_id=episode_id,
                        task_id=task_id,
                        rms=rand["rms"],
                        mse=rand["mse"],
                        gripper_error=rand["gripper_error"],
                        segment_lengths=f"seed={seed}",
                    )
                )

    shard_path = shard_dir / f"shard_{args.shard_id}.parquet"
    pq.write_table(pa.Table.from_pylist([asdict(r) for r in rows]), shard_path)

    meta = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "n_chunks_total": int(codes.shape[0]),
        "n_chunks_this_shard": len(chunk_indices),
        "chunk_indices": chunk_indices,
        "output_shard": str(shard_path),
    }
    (shard_dir / f"shard_{args.shard_id}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

    if args.num_shards == 1:
        chunk_ids = list(range(codes.shape[0]))
        episode_ids = [int(batch.episode_ids[i]) for i in chunk_ids]
        summary = summarize_rows(rows, episode_ids, chunk_ids, budgets, args.seed, args.random_seeds)
        pq.write_table(pa.Table.from_pylist([asdict(r) for r in rows]), out_dir / "eval_rows.parquet")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
