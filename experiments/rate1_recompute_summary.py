#!/usr/bin/env python3
"""Recompute Gate-1 summary metrics and per-task tables from per_chunk.parquet.

Uses only numpy + pyarrow (no pandas dependency).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_rvq.metrics import bootstrap_ci  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", default="artifacts/apb_rvq/oracle_full")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _improvement(base: float, oracle: float) -> float:
    return float((base - oracle) / max(base, 1e-12) * 100.0)


def _col(table, name: str) -> np.ndarray:
    return table.column(name).to_numpy(zero_copy_only=False)


def _filter_eq(table, col: str, value):
    return table.filter(pc.equal(table.column(col), value))


def _filter_and(table, predicates: list):
    mask = predicates[0]
    for pred in predicates[1:]:
        mask = pc.and_(mask, pred)
    return table.filter(mask)


def _sorted_unique_chunk_meta(oracle_budget0):
    chunk = _col(oracle_budget0, "chunk_idx").astype(np.int64)
    ep = _col(oracle_budget0, "episode_id").astype(np.int64)
    task = _col(oracle_budget0, "task_id").astype(np.int64)
    order = np.argsort(chunk, kind="mergesort")
    chunk_s, ep_s, task_s = chunk[order], ep[order], task[order]
    # drop duplicate chunk rows if any
    _, uniq_idx = np.unique(chunk_s, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    return chunk_s[uniq_idx], ep_s[uniq_idx], task_s[uniq_idx]


def _rms_by_chunk(table, chunk_ids: np.ndarray) -> np.ndarray:
    chunks = _col(table, "chunk_idx").astype(np.int64)
    rms = _col(table, "rms").astype(np.float64)
    order = np.argsort(chunks, kind="mergesort")
    chunks_s, rms_s = chunks[order], rms[order]
    # map expected chunk_ids -> values (assumes one row per chunk)
    if chunks_s.shape[0] != chunk_ids.shape[0] or not np.array_equal(chunks_s, chunk_ids):
        # align by dictionary
        mapping = {int(c): float(v) for c, v in zip(chunks_s, rms_s)}
        out = np.asarray([mapping[int(c)] for c in chunk_ids], dtype=np.float64)
        return out
    return rms_s


def _mean_random_by_chunk(table, chunk_ids: np.ndarray) -> np.ndarray:
    chunks = _col(table, "chunk_idx").astype(np.int64)
    rms = _col(table, "rms").astype(np.float64)
    sums = defaultdict(float)
    counts = defaultdict(int)
    for c, v in zip(chunks, rms):
        ci = int(c)
        sums[ci] += float(v)
        counts[ci] += 1
    return np.asarray([sums[int(c)] / max(counts[int(c)], 1) for c in chunk_ids], dtype=np.float64)


def main() -> None:
    args = parse_args()
    oracle_dir = Path(args.oracle_dir)
    metrics_path = oracle_dir / "metrics.json"
    parquet_path = oracle_dir / "per_chunk.parquet"
    if not metrics_path.exists() or not parquet_path.exists():
        raise FileNotFoundError(f"Need metrics.json and per_chunk.parquet under {oracle_dir}")

    old = json.loads(metrics_path.read_text(encoding="utf-8"))
    table = pq.read_table(parquet_path)
    budgets = [int(b) for b in old["meta"]["budgets"]]

    oracle_all = _filter_and(
        table,
        [
            pc.equal(table.column("method"), "oracle"),
            pc.equal(table.column("mode"), "exact-budget"),
        ],
    )
    oracle_b0 = _filter_eq(oracle_all, "budget", budgets[0])
    chunk_ids, ep, task = _sorted_unique_chunk_meta(oracle_b0)

    summary = {
        "meta": {
            **old["meta"],
            "recomputed_from": "per_chunk.parquet",
            "ci_method": "cluster bootstrap by episode on (oracle - mean_random_per_chunk)",
            "n_boot": args.n_boot,
        },
        "oracle_exact": {},
        "oracle_at_most": old.get("oracle_at_most", {}),
        "baselines": {},
        "gates": {},
    }

    key_budgets = [24, 32, 40]
    for budget in budgets:
        o_tbl = _filter_eq(oracle_all, "budget", budget)
        rnd_tbl = _filter_and(
            table,
            [
                pc.equal(table.column("method"), "random"),
                pc.equal(table.column("budget"), budget),
            ],
        )
        uni_tbl = _filter_and(
            table,
            [
                pc.equal(table.column("method"), "uniform"),
                pc.equal(table.column("budget"), budget),
            ],
        )
        st_tbl = _filter_and(
            table,
            [
                pc.equal(table.column("method"), "static"),
                pc.equal(table.column("budget"), budget),
            ],
        )
        gm_tbl = _filter_and(
            table,
            [
                pc.equal(table.column("method"), "global_mix"),
                pc.equal(table.column("budget"), budget),
            ],
        )

        o = _rms_by_chunk(o_tbl, chunk_ids)
        rnd = _mean_random_by_chunk(rnd_tbl, chunk_ids)
        uni = _rms_by_chunk(uni_tbl, chunk_ids)
        st = _rms_by_chunk(st_tbl, chunk_ids)
        gm = _rms_by_chunk(gm_tbl, chunk_ids)
        if not (len(o) == len(rnd) == len(uni) == len(st) == len(gm) == len(ep)):
            raise RuntimeError(f"Length mismatch at budget {budget}")

        delta = o - rnd
        ci = bootstrap_ci(delta, ep, n_boot=args.n_boot, seed=args.seed)
        best = np.minimum(np.minimum(uni, st), gm)
        best_mean = float(best.mean())
        summary["oracle_exact"][str(budget)] = {
            "rms_mean": float(o.mean()),
            "rms_ci_vs_random_delta": [float(ci[0]), float(ci[1])],
            "improvement_vs_random_pct": _improvement(float(rnd.mean()), float(o.mean())),
            "improvement_vs_uniform_pct": _improvement(float(uni.mean()), float(o.mean())),
            "improvement_vs_static_pct": _improvement(float(st.mean()), float(o.mean())),
            "improvement_vs_best_nonadaptive_pct": _improvement(best_mean, float(o.mean())),
            "ci_vs_random_excludes_zero": bool(ci[1] < 0.0 or ci[0] > 0.0),
            "gap_closed_mean": old.get("oracle_exact", {}).get(str(budget), {}).get("gap_closed_mean"),
        }
        summary["baselines"][str(budget)] = {
            "random_rms_mean": float(rnd.mean()),
            "uniform_rms_mean": float(uni.mean()),
            "static_rms_mean": float(st.mean()),
            "global_mix_rms_mean": float(gm.mean()),
            "best_nonadaptive_rms_mean": best_mean,
        }

    summary["gates"] = {
        "criterion_a_random_ge_15pct_on_24_32_40": all(
            summary["oracle_exact"][str(b)]["improvement_vs_random_pct"] >= 15.0 for b in key_budgets
        ),
        "criterion_a_ci_excludes_zero_on_24_32_40": all(
            summary["oracle_exact"][str(b)]["ci_vs_random_excludes_zero"] for b in key_budgets
        ),
        "criterion_b_best_nonadaptive_ge_10pct_on_24_32_40": all(
            summary["oracle_exact"][str(b)]["improvement_vs_best_nonadaptive_pct"] >= 10.0
            for b in key_budgets
        ),
    }

    # Per-task breakdown at key budgets.
    per_task = {"meta": {"n_chunks": int(len(ep)), "n_tasks": int(len(np.unique(task)))}, "budgets": {}}
    for budget in key_budgets:
        o_tbl = _filter_eq(oracle_all, "budget", budget)
        o_chunks = _col(o_tbl, "chunk_idx").astype(np.int64)
        o_tasks = _col(o_tbl, "task_id").astype(np.int64)
        o_rms = _col(o_tbl, "rms").astype(np.float64)
        rnd_tbl = _filter_and(
            table,
            [
                pc.equal(table.column("method"), "random"),
                pc.equal(table.column("budget"), budget),
            ],
        )
        rnd_chunks = _col(rnd_tbl, "chunk_idx").astype(np.int64)
        rnd_tasks = _col(rnd_tbl, "task_id").astype(np.int64)
        rnd_rms = _col(rnd_tbl, "rms").astype(np.float64)

        # mean random per (chunk, task)
        rnd_sum = defaultdict(float)
        rnd_cnt = defaultdict(int)
        for c, t, v in zip(rnd_chunks, rnd_tasks, rnd_rms):
            key = (int(c), int(t))
            rnd_sum[key] += float(v)
            rnd_cnt[key] += 1

        by_task_o = defaultdict(list)
        by_task_r = defaultdict(list)
        for c, t, v in zip(o_chunks, o_tasks, o_rms):
            key = (int(c), int(t))
            r = rnd_sum[key] / max(rnd_cnt[key], 1)
            by_task_o[int(t)].append(float(v))
            by_task_r[int(t)].append(float(r))

        rows = []
        for task_id in sorted(by_task_o):
            o_mean = float(np.mean(by_task_o[task_id]))
            r_mean = float(np.mean(by_task_r[task_id]))
            rows.append(
                {
                    "task_id": int(task_id),
                    "n_chunks": int(len(by_task_o[task_id])),
                    "oracle_rms_mean": o_mean,
                    "random_rms_mean": r_mean,
                    "improvement_vs_random_pct": _improvement(r_mean, o_mean),
                }
            )
        rows.sort(key=lambda x: x["improvement_vs_random_pct"], reverse=True)
        imps = np.asarray([r["improvement_vs_random_pct"] for r in rows], dtype=np.float64)
        per_task["budgets"][str(budget)] = {
            "n_tasks_positive": int((imps > 0).sum()),
            "n_tasks_ge_15pct": int((imps >= 15).sum()),
            "improvement_mean_over_tasks": float(imps.mean()),
            "improvement_median_over_tasks": float(np.median(imps)),
            "tasks": rows,
        }

    out_metrics = oracle_dir / "metrics_corrected.json"
    out_tasks = oracle_dir / "per_task.json"
    out_metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_tasks.write_text(json.dumps(per_task, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": [str(out_metrics), str(out_tasks)], "gates": summary["gates"]}, indent=2))
    for b in key_budgets:
        row = summary["oracle_exact"][str(b)]
        print(
            f"B={b}: vs_random={row['improvement_vs_random_pct']:.1f}% "
            f"CI_delta={row['rms_ci_vs_random_delta']} "
            f"vs_best={row['improvement_vs_best_nonadaptive_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
