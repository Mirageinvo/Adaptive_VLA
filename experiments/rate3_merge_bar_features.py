#!/usr/bin/env python3
"""Merge BAR feature shards and compare Phase A vs Phase B eval JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def merge_features(in_dir: Path, out_path: Path) -> dict:
    files = sorted(in_dir.glob("feat_*.npz"))
    if not files:
        raise FileNotFoundError(f"No feat_*.npz in {in_dir}")
    chunks, feats = [], []
    for f in files:
        d = np.load(f)
        chunks.append(d["chunk_idx"])
        feats.append(d["obs_pooled_ctx"])
    chunk_idx = np.concatenate(chunks)
    obs = np.concatenate(feats, axis=0)
    order = np.argsort(chunk_idx)
    chunk_idx = chunk_idx[order]
    obs = obs[order]
    if len(np.unique(chunk_idx)) != len(chunk_idx):
        raise RuntimeError("Duplicate chunk_idx in BAR features")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, chunk_idx=chunk_idx, obs_pooled_ctx=obs.astype(np.float16))
    meta = {
        "n": int(len(chunk_idx)),
        "dim": int(obs.shape[1]),
        "chunk_range": [int(chunk_idx[0]), int(chunk_idx[-1])],
        "shards": [p.name for p in files],
        "output": str(out_path),
    }
    (out_path.parent / "features_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-dir", default="")
    ap.add_argument("--merge-out", default="artifacts/apb_rvq/bar_features_phase_b/obs_pooled_ctx.npz")
    ap.add_argument("--phase-a-dir", default="artifacts/apb_rvq")
    ap.add_argument("--phase-b-dir", default="artifacts/apb_rvq")
    ap.add_argument("--output", default="artifacts/apb_rvq/phase_ab_compare.json")
    args = ap.parse_args()

    if args.merge_dir:
        meta = merge_features(Path(args.merge_dir), Path(args.merge_out))
        print(json.dumps({"merged": meta}, indent=2))

    rows = []
    a_root = Path(args.phase_a_dir)
    b_root = Path(args.phase_b_dir)
    for bpath in sorted(b_root.glob("router_phase_b_b*/eval_retained.json")):
        budget = int(bpath.parent.name.split("_b")[-1])
        apath = a_root / f"router_phase_a_b{budget}" / "eval_retained.json"
        if not apath.exists():
            continue
        a = json.loads(apath.read_text())
        b = json.loads(bpath.read_text())
        rows.append(
            {
                "budget": budget,
                "n_eval_a": a.get("n_eval"),
                "n_eval_b": b.get("n_eval"),
                "retained_a": a.get("retained_gain_ratio_of_means"),
                "retained_b": b.get("retained_gain_ratio_of_means"),
                "delta_retained": (b.get("retained_gain_ratio_of_means") or 0)
                - (a.get("retained_gain_ratio_of_means") or 0),
                "pred_vs_random_a": a.get("pred_vs_random_pct"),
                "pred_vs_random_b": b.get("pred_vs_random_pct"),
                "oracle_vs_random_a": a.get("oracle_vs_random_pct"),
                "oracle_vs_random_b": b.get("oracle_vs_random_pct"),
                "position_acc_a": a.get("position_acc"),
                "position_acc_b": b.get("position_acc"),
            }
        )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"compare": rows, "wrote": str(out)}, indent=2))


if __name__ == "__main__":
    main()
