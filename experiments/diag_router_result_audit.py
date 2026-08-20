#!/usr/bin/env python3
"""Diagnostics: are weak router results bugs or real?"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from experiments.rate4_train_router import DepthRouterMLP, allocate_from_logits, load_bar_feature_table


def main() -> None:
    root = Path("artifacts/apb_rvq")
    d = np.load(root / "oracle_full/depth_maps.npz")
    cfg = json.loads((root / "oracle_full/config.json").read_text())
    all_b = [int(x) for x in cfg["budgets"].split(",")]
    rows = pq.read_table(root / "labels_phase_a/labels.parquet").to_pylist()

    rng = np.random.default_rng(0)
    sample = rng.choice(len(rows), size=200, replace=False)
    mism = 0
    bad_sum = 0
    for i in sample:
        r = rows[i]
        bi = all_b.index(int(r["budget"]))
        dep = np.asarray(r["depth_map"])
        ref = d["depth_exact"][int(r["chunk_idx"]), bi]
        if not np.array_equal(dep, ref):
            mism += 1
        if int(dep.sum()) != int(r["budget"]):
            bad_sum += 1
    print("label_oracle_mismatches", mism, "bad_budget_sum", bad_sum, "of", len(sample))

    feats = load_bar_feature_table(root / "bar_features_phase_b")
    cids = np.array(sorted(feats.keys()))
    X = np.stack([feats[c] for c in cids])
    print("bar_n", len(cids), "range", int(cids[0]), int(cids[-1]), "dim", X.shape[1])
    print("bar_finite", bool(np.isfinite(X).all()), "feat_std_mean", float(X.std(0).mean()))
    diffs = np.linalg.norm(X[1:] - X[:-1], axis=1)
    print("bar_adj_l2_mean", float(diffs.mean()), "frac_near0", float((diffs < 1e-3).mean()))

    for phase in ["a", "b"]:
        ck = torch.load(root / f"router_phase_{phase}_b28/best.pt", map_location="cpu", weights_only=False)
        ctx_dim = int(ck.get("ctx_dim", 0) or 0)
        model = DepthRouterMLP(
            vocab=2048, ctx_dim=ctx_dim, ctx_proj=int(ck.get("args", {}).get("ctx_proj", 128))
        )
        model.load_state_dict(ck["model"])
        model.eval()
        split = json.loads((root / f"router_phase_{phase}_b28/split.json").read_text())
        val = set(split["val_episode_ids"])
        train = set(split["train_episode_ids"])
        print(phase, "train_val_overlap", len(train & val), "n_val_ep", len(val))
        sub = [r for r in rows if int(r["budget"]) == 28 and int(r["episode_id"]) in val][:256]
        coarse = torch.tensor([r["coarse_codes"] for r in sub], dtype=torch.long)
        pos = torch.tensor([r["position_embeddings"] for r in sub], dtype=torch.float32)
        state = torch.tensor([r["state_first"] for r in sub], dtype=torch.float32)
        budget = torch.full((len(sub), 1), 28 / 48.0)
        with torch.no_grad():
            if ctx_dim:
                ctx = torch.tensor(np.stack([feats[int(r["chunk_idx"])] for r in sub]), dtype=torch.float32)
                logits = model(coarse, pos, state, budget, ctx=ctx)
            else:
                logits = model(coarse, pos, state, budget)
            pred = allocate_from_logits(logits, 28)
            oracle = torch.tensor([r["depth_map"] for r in sub])
            probs = logits.softmax(-1)
            ent = float((-(probs * probs.clamp_min(1e-8).log()).sum(-1).mean()).item())
        print(
            phase,
            "budget_ok",
            float((pred.sum(1) == 28).float().mean()),
            "pred_frac",
            {d: round(float((pred == d).float().mean()), 3) for d in (1, 2, 3)},
            "oracle_frac",
            {d: round(float((oracle == d).float().mean()), 3) for d in (1, 2, 3)},
            "pos_acc",
            round(float((pred == oracle).float().mean()), 3),
            "entropy",
            round(ent, 3),
        )

    ck = torch.load(root / "router_phase_b_b28/best.pt", map_location="cpu", weights_only=False)
    model = DepthRouterMLP(vocab=2048, ctx_dim=2048, ctx_proj=128)
    model.load_state_dict(ck["model"])
    model.eval()
    split = json.loads((root / "router_phase_b_b28/split.json").read_text())
    val = set(split["val_episode_ids"])
    sub = [r for r in rows if int(r["budget"]) == 28 and int(r["episode_id"]) in val][:256]
    coarse = torch.tensor([r["coarse_codes"] for r in sub], dtype=torch.long)
    pos = torch.tensor([r["position_embeddings"] for r in sub], dtype=torch.float32)
    state = torch.tensor([r["state_first"] for r in sub], dtype=torch.float32)
    budget = torch.full((len(sub), 1), 28 / 48.0)
    ctx = torch.tensor(np.stack([feats[int(r["chunk_idx"])] for r in sub]), dtype=torch.float32)
    with torch.no_grad():
        p1 = allocate_from_logits(model(coarse, pos, state, budget, ctx=ctx), 28)
        p0 = allocate_from_logits(model(coarse, pos, state, budget, ctx=torch.zeros_like(ctx)), 28)
        pS = allocate_from_logits(model(coarse, pos, state, budget, ctx=ctx[torch.randperm(len(ctx))]), 28)
    print("ctx_vs_zero_hamming", round(float((p1 != p0).float().mean()), 4))
    print("ctx_vs_shuffle_hamming", round(float((p1 != pS).float().mean()), 4))

    # eval json sanity
    for phase in ["a", "b"]:
        ev = json.loads((root / f"router_phase_{phase}_b28/eval_retained.json").read_text())
        print(
            phase,
            "eval",
            "held_out",
            ev.get("held_out_episodes"),
            "n",
            ev.get("n_eval"),
            "oracle_vs_rand",
            round(ev.get("oracle_vs_random_pct", -1), 2),
            "pred_vs_rand",
            round(ev.get("pred_vs_random_pct", -1), 2),
            "retained",
            round(100 * ev.get("retained_gain_ratio_of_means", 0), 1),
        )


if __name__ == "__main__":
    main()
