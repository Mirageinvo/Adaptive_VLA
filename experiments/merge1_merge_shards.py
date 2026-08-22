#!/usr/bin/env python3
"""Merge shard outputs into a single Stage-1 summary."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.merge1_oracle_compression import EvalRow, summarize_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-seeds", type=int, default=8)
    ap.add_argument("--segment-budgets", default="16,14,12,10,8,6,4")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    shard_dir = in_dir / "shards"
    shard_files = sorted(shard_dir.glob("shard_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No shard parquet files in {shard_dir}")

    rows: list[EvalRow] = []
    for path in shard_files:
        table = pq.read_table(path)
        rows.extend(EvalRow(**row) for row in table.to_pylist())

    budgets = sorted((int(x) for x in args.segment_budgets.split(",")), reverse=True)
    chunk_ids = sorted({r.chunk_idx for r in rows if r.method == "no_merge"})
    episode_ids = [next(r.episode_id for r in rows if r.chunk_idx == cid and r.method == "no_merge") for cid in chunk_ids]

    summary = summarize_rows(
        rows,
        episode_ids=episode_ids,
        chunk_ids=chunk_ids,
        budgets=budgets,
        seed=args.seed,
        random_seeds=args.random_seeds,
    )
    summary["n_shards"] = len(shard_files)
    summary["shard_files"] = [p.name for p in shard_files]

    pq.write_table(pa.Table.from_pylist([asdict(r) for r in rows]), in_dir / "eval_rows.parquet")
    out_path = Path(args.output) if args.output else in_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
