#!/usr/bin/env python3
"""Evaluate Phase-A/B router by reconstruction retained gain vs oracle/random."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_rvq.codec import decode_with_depth, encode_actions, load_codec, projected_codebooks
from adaptive_rvq.data import load_libero_chunks_indexed
from adaptive_rvq.metrics import compute_metrics
from experiments.rate4_train_router import DepthRouterMLP, allocate_from_logits, load_bar_feature_table


def random_depth_maps(
    n: int,
    budget: int,
    n_pos: int = 16,
    n_seeds: int = 20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    out = np.ones((n, n_seeds, n_pos), dtype=np.int64)
    rem = budget - n_pos
    for i in range(n):
        for s in range(n_seeds):
            for _ in range(rem):
                cand = np.where(out[i, s] < 3)[0]
                out[i, s, rng.choice(cand)] += 1
    return out


def batch_rms(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (recon[..., :6] - target[..., :6]).pow(2).mean(dim=(-1, -2)).sqrt()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="artifacts/apb_rvq/labels_phase_a/labels.parquet")
    ap.add_argument("--checkpoint", default="artifacts/apb_rvq/router_phase_a_b28/best.pt")
    ap.add_argument("--bar-features", default="", help="override; else from ckpt/split")
    ap.add_argument("--split", default="")
    ap.add_argument("--oracle-dir", default="artifacts/apb_rvq/oracle_full")
    ap.add_argument("--budget", type=int, default=28)
    ap.add_argument("--n-eval", type=int, default=-1)
    ap.add_argument("--random-seeds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    ctx_dim = int(ckpt.get("ctx_dim", 0) or 0)
    bar_path = args.bar_features or ckpt_args.get("bar_features") or ""
    bar_feats = load_bar_feature_table(bar_path) if (ctx_dim > 0 and bar_path) else None
    if ctx_dim > 0 and bar_feats is None:
        raise RuntimeError("Checkpoint expects BAR features but none provided")

    model = DepthRouterMLP(
        vocab=int(ckpt_args.get("codebook_size", 2048)),
        ctx_dim=ctx_dim,
        ctx_proj=int(ckpt_args.get("ctx_proj", 128)),
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    split_path = Path(args.split) if args.split else Path(ckpt.get("split_path", ""))
    if not args.split and (Path(args.checkpoint).parent / "split.json").exists():
        split_path = Path(args.checkpoint).parent / "split.json"

    rows = [r for r in pq.read_table(args.labels).to_pylist() if int(r["budget"]) == args.budget]
    held_out = False
    if split_path and split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        val_eps = set(int(x) for x in split["val_episode_ids"])
        rows = [r for r in rows if int(r["episode_id"]) in val_eps]
        held_out = True

    rng = np.random.default_rng(args.seed)
    if args.n_eval > 0 and len(rows) > args.n_eval:
        pick = rng.choice(len(rows), size=args.n_eval, replace=False)
        rows = [rows[i] for i in pick.tolist()]
    if not rows:
        raise RuntimeError("No eval rows after split/filter")

    cfg = json.loads((Path(args.oracle_dir) / "config.json").read_text(encoding="utf-8"))
    metrics_meta = json.loads((Path(args.oracle_dir) / "metrics.json").read_text(encoding="utf-8"))
    episode_ids = np.asarray([r["episode_id"] for r in rows], dtype=np.int64)
    starts = np.asarray([r["start"] for r in rows], dtype=np.int64)
    oracle_depth = np.asarray([r["depth_map"] for r in rows], dtype=np.int64)

    batch = load_libero_chunks_indexed(
        episode_ids=episode_ids,
        starts=starts,
        dataset_id=cfg["dataset"],
        revision=cfg["revision"],
        device=args.device,
        gripper_mode=metrics_meta["meta"]["gripper_mode"],
        include_state=True,
    )
    codec = load_codec(model_id=cfg["model"], device=args.device)
    E = projected_codebooks(codec, device=args.device)
    codes = encode_actions(codec, batch.actions, embodiment_id=int(cfg.get("embodiment", 0)))

    coarse = torch.tensor([r["coarse_codes"] for r in rows], dtype=torch.long, device=args.device)
    pos = torch.tensor([r["position_embeddings"] for r in rows], dtype=torch.float32, device=args.device)
    state = torch.tensor([r["state_first"] for r in rows], dtype=torch.float32, device=args.device)
    budget_feat = torch.full((len(rows), 1), args.budget / 48.0, device=args.device)
    ctx = None
    if bar_feats is not None:
        ctx = torch.tensor(
            np.stack([bar_feats[int(r["chunk_idx"])] for r in rows], axis=0),
            dtype=torch.float32,
            device=args.device,
        )

    with torch.no_grad():
        if ctx is None:
            logits = model(coarse, pos, state, budget_feat)
        else:
            logits = model(coarse, pos, state, budget_feat, ctx=ctx)
        pred = allocate_from_logits(logits, args.budget)

    rnd_maps = random_depth_maps(len(rows), args.budget, n_seeds=args.random_seeds, rng=rng)
    emb = int(cfg.get("embodiment", 0))
    oracle_t = torch.from_numpy(oracle_depth).to(args.device)

    with torch.no_grad():
        rms_oracle = batch_rms(decode_with_depth(codec, E, codes, oracle_t, embodiment_id=emb), batch.actions)
        rms_pred = batch_rms(decode_with_depth(codec, E, codes, pred, embodiment_id=emb), batch.actions)
        rms_rand_seeds = []
        for s in range(args.random_seeds):
            rnd = torch.from_numpy(rnd_maps[:, s]).to(args.device)
            rms_rand_seeds.append(
                batch_rms(decode_with_depth(codec, E, codes, rnd, embodiment_id=emb), batch.actions)
            )
        rms_rand = torch.stack(rms_rand_seeds, dim=1).mean(dim=1)

    eps = 1e-8
    mean_o = float(rms_oracle.mean().item())
    mean_p = float(rms_pred.mean().item())
    mean_r = float(rms_rand.mean().item())
    retained_ratio_of_means = (mean_r - mean_p) / (mean_r - mean_o + eps)
    per_chunk = ((rms_rand - rms_pred) / (rms_rand - rms_oracle + eps)).clamp(-1, 2)

    out = {
        "phase": "B" if ctx_dim else "A",
        "budget": args.budget,
        "n_eval": len(rows),
        "held_out_episodes": held_out,
        "random_seeds": args.random_seeds,
        "mean_rms_oracle": mean_o,
        "mean_rms_pred": mean_p,
        "mean_rms_random": mean_r,
        "retained_gain_ratio_of_means": retained_ratio_of_means,
        "retained_gain_mean_of_ratios": float(per_chunk.mean().item()),
        "retained_gain_median_of_ratios": float(per_chunk.median().item()),
        "position_acc": float((pred == oracle_t).float().mean().item()),
        "hamming": float((pred != oracle_t).float().mean().item()),
        "exact_map_acc": float((pred == oracle_t).all(dim=1).float().mean().item()),
        "budget_match_rate": float((pred.sum(dim=1) == args.budget).float().mean().item()),
        "pred_depth_frac": {str(d): float((pred == d).float().mean().item()) for d in (1, 2, 3)},
        "oracle_depth_frac": {str(d): float((oracle_t == d).float().mean().item()) for d in (1, 2, 3)},
        "pred_vs_random_pct": float((mean_r - mean_p) / (mean_r + eps) * 100.0),
        "oracle_vs_random_pct": float((mean_r - mean_o) / (mean_r + eps) * 100.0),
        "aggregate_metrics_pred": compute_metrics(
            batch.actions, decode_with_depth(codec, E, codes, pred, embodiment_id=emb)
        ),
    }
    print(json.dumps(out, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
