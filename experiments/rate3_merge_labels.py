#!/usr/bin/env python3
"""Merge Phase A rate3 shards into labels.parquet + stats."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="artifacts/apb_rvq/labels_phase_a")
    ap.add_argument("--output", default="artifacts/apb_rvq/labels_phase_a/labels.parquet")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    shards = sorted(in_dir.glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.parquet in {in_dir}")

    tables = [pq.read_table(p) for p in shards]
    # concat
    table = tables[0]
    for t in tables[1:]:
        import pyarrow as pa

        table = pa.concat_tables([table, t])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Coverage / uniqueness guards
    chunk_idx = np.asarray(table.column("chunk_idx").to_pylist(), dtype=np.int64)
    budgets_arr = np.asarray(table.column("budget").to_pylist(), dtype=np.int64)
    keys = list(zip(chunk_idx.tolist(), budgets_arr.tolist()))
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate (chunk_idx, budget) rows in merged labels")
    unique_chunks = sorted(set(chunk_idx.tolist()))
    unique_budgets = sorted(set(budgets_arr.tolist()))
    expected = len(unique_chunks) * len(unique_budgets)
    if table.num_rows != expected:
        raise RuntimeError(
            f"Row count {table.num_rows} != n_chunks({len(unique_chunks)}) * n_budgets({len(unique_budgets)})"
        )
    if unique_chunks != list(range(unique_chunks[0], unique_chunks[-1] + 1)):
        raise RuntimeError("chunk_idx coverage is not contiguous")

    pq.write_table(table, out)

    # stats
    budgets = budgets_arr
    depths = np.asarray(table.column("depth_map").to_pylist(), dtype=np.int16)
    depth_counts = Counter(depths.reshape(-1).tolist())
    stats = {
        "n_rows": int(table.num_rows),
        "n_shards": len(shards),
        "shards": [p.name for p in shards],
        "budgets": unique_budgets,
        "n_chunks": len(unique_chunks),
        "chunk_range": [int(unique_chunks[0]), int(unique_chunks[-1])],
        "depth_value_counts": {str(k): int(v) for k, v in sorted(depth_counts.items())},
        "mean_depth": float(depths.mean()),
        "mean_budget_used": float(depths.sum(axis=1).mean()),
    }
    stats_path = out.parent / "label_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    features_cfg = {
        "phase": "A",
        "causal_features": ["coarse_codes", "position_embeddings", "state_first", "budget"],
        "non_causal_or_oracle_adjacent": ["codes_full", "state_mean", "depth_map"],
        "primary_target": "depth_map",
        "derived_targets": ["nested_gate_ge2", "nested_gate_ge3"],
        "forbidden_in_deployable_router": [
            "future_actions",
            "state_mean",
            "codes_full",
            "fine_logits",
            "soft_marginal_utility",
            "depth_map",
        ],
    }
    (out.parent / "features_config.json").write_text(json.dumps(features_cfg, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": [str(out), str(stats_path)], "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
