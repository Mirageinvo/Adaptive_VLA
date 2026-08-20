#!/usr/bin/env python3
"""Phase-A/B router trainer: depth_map from causal features (+ optional BAR ctx)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_bar_feature_table(path: str | Path) -> dict[int, np.ndarray]:
    """Load obs_pooled_ctx keyed by chunk_idx from feat_*.npz or a single merged npz."""
    path = Path(path)
    files = sorted(path.glob("feat_*.npz")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No BAR feature files under {path}")
    out: dict[int, np.ndarray] = {}
    for f in files:
        d = np.load(f)
        for i, cid in enumerate(d["chunk_idx"].tolist()):
            out[int(cid)] = np.asarray(d["obs_pooled_ctx"][i], dtype=np.float32)
    return out


class LabelDataset(Dataset):
    def __init__(self, rows: list[dict], bar_feats: dict[int, np.ndarray] | None = None):
        self.rows = rows
        self.bar_feats = bar_feats
        if bar_feats is not None:
            missing = [int(r["chunk_idx"]) for r in rows if int(r["chunk_idx"]) not in bar_feats]
            if missing:
                raise KeyError(f"Missing BAR features for {len(missing)} rows, e.g. {missing[:5]}")

    @classmethod
    def from_parquet(
        cls,
        path: str,
        budgets: list[int] | None = None,
        bar_feats: dict[int, np.ndarray] | None = None,
    ) -> "LabelDataset":
        table = pq.read_table(path)
        rows = table.to_pylist()
        if budgets is not None:
            bset = set(budgets)
            rows = [r for r in rows if int(r["budget"]) in bset]
        return cls(rows, bar_feats=bar_feats)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        state = torch.tensor(r["state_first"], dtype=torch.float32)
        coarse = torch.tensor(r["coarse_codes"], dtype=torch.long)
        pos = torch.tensor(r["position_embeddings"], dtype=torch.float32)
        budget = torch.tensor([r["budget"]], dtype=torch.float32) / 48.0
        depth = torch.tensor(r["depth_map"], dtype=torch.long) - 1
        item = {
            "coarse": coarse,
            "pos": pos,
            "state": state,
            "budget": budget,
            "depth": depth,
            "chunk_idx": int(r["chunk_idx"]),
            "episode_id": int(r["episode_id"]),
        }
        if self.bar_feats is not None:
            item["ctx"] = torch.tensor(self.bar_feats[int(r["chunk_idx"])], dtype=torch.float32)
        return item


class DepthRouterMLP(nn.Module):
    """Per-position depth classifier; optional frozen BAR context."""

    def __init__(
        self,
        vocab: int = 2048,
        emb: int = 64,
        state_dim: int = 8,
        pos_dim: int = 32,
        hidden: int = 256,
        ctx_dim: int = 0,
        ctx_proj: int = 128,
    ):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.code_emb = nn.Embedding(vocab, emb)
        self.budget_mlp = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32))
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, 64), nn.GELU(), nn.Linear(64, 64))
        self.ctx_mlp = None
        extra = 0
        if ctx_dim > 0:
            self.ctx_mlp = nn.Sequential(
                nn.Linear(ctx_dim, ctx_proj),
                nn.GELU(),
                nn.Linear(ctx_proj, ctx_proj),
            )
            extra = ctx_proj
        in_dim = emb + pos_dim + 32 + 64 + extra
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, coarse, pos, state, budget, ctx=None):
        x = self.code_emb(coarse)
        b = self.budget_mlp(budget).unsqueeze(1).expand(-1, 16, -1)
        s = self.state_mlp(state).unsqueeze(1).expand(-1, 16, -1)
        parts = [x, pos, b, s]
        if self.ctx_mlp is not None:
            if ctx is None:
                raise ValueError("ctx required for vision router")
            c = self.ctx_mlp(ctx).unsqueeze(1).expand(-1, 16, -1)
            parts.append(c)
        return self.trunk(torch.cat(parts, dim=-1))


def ordinal_depth_loss(logits: torch.Tensor, depth_cls: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(logits.reshape(-1, 3), depth_cls.reshape(-1))
    probs = logits.softmax(dim=-1)
    p_ge2 = probs[..., 1] + probs[..., 2]
    p_ge3 = probs[..., 2]
    y = depth_cls
    t_ge2 = (y >= 1).float()
    t_ge3 = (y >= 2).float()
    bce = F.binary_cross_entropy(p_ge2.clamp(1e-6, 1 - 1e-6), t_ge2) + F.binary_cross_entropy(
        p_ge3.clamp(1e-6, 1 - 1e-6), t_ge3
    )
    return ce + 0.5 * bce


def allocate_from_logits(logits: torch.Tensor, budget: int) -> torch.Tensor:
    """Greedy nested allocation: score12=P(class==1), score23=P(class==2)."""
    bsz, n_pos, _ = logits.shape
    probs = logits.softmax(dim=-1)
    depth = torch.ones((bsz, n_pos), dtype=torch.long, device=logits.device)
    remaining = int(budget) - n_pos
    score12 = probs[..., 1]
    score23 = probs[..., 2]
    for _ in range(max(0, remaining)):
        can12 = (depth == 1).float() * score12
        can23 = (depth == 2).float() * (score23 + 1e-6)
        flat = torch.stack([can12, can23], dim=-1).reshape(bsz, -1)
        choice = flat.argmax(dim=-1)
        for bi in range(bsz):
            c = int(choice[bi].item())
            pos = c // 2
            kind = c % 2
            if kind == 0 and int(depth[bi, pos].item()) == 1:
                depth[bi, pos] = 2
            elif kind == 1 and int(depth[bi, pos].item()) == 2:
                depth[bi, pos] = 3
            else:
                legal = (depth[bi] < 3).nonzero(as_tuple=False).view(-1)
                if legal.numel() == 0:
                    continue
                scores = torch.where(depth[bi] == 1, score12[bi], score23[bi])
                j = int(legal[scores[legal].argmax()].item())
                depth[bi, j] = int(depth[bi, j].item()) + 1
    return depth


def episode_grouped_split(
    rows: list[dict],
    val_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    rng = np.random.default_rng(seed)
    episodes = sorted({int(r["episode_id"]) for r in rows})
    rng.shuffle(episodes)
    n_val_ep = max(1, int(math.floor(len(episodes) * val_frac)))
    val_eps = set(episodes[:n_val_ep])
    train_eps = set(episodes[n_val_ep:])
    train_rows = [r for r in rows if int(r["episode_id"]) in train_eps]
    val_rows = [r for r in rows if int(r["episode_id"]) in val_eps]
    meta = {
        "seed": seed,
        "val_frac": val_frac,
        "n_episodes": len(episodes),
        "n_train_episodes": len(train_eps),
        "n_val_episodes": len(val_eps),
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "train_episode_ids": sorted(train_eps),
        "val_episode_ids": sorted(val_eps),
        "overlap_episodes": sorted(train_eps & val_eps),
    }
    if meta["overlap_episodes"]:
        raise RuntimeError(f"Episode leakage in split: {meta['overlap_episodes'][:10]}")
    return train_rows, val_rows, meta


def _forward(model, batch, device):
    kwargs = dict(
        coarse=batch["coarse"].to(device),
        pos=batch["pos"].to(device),
        state=batch["state"].to(device),
        budget=batch["budget"].to(device),
    )
    if "ctx" in batch:
        kwargs["ctx"] = batch["ctx"].to(device)
    return model(**kwargs)


@torch.no_grad()
def eval_maps(model, loader, device) -> dict:
    model.eval()
    exact = 0
    total = 0
    budget_ok = 0
    pos_correct = 0
    pos_total = 0
    hamming_sum = 0.0
    depth_hist = torch.zeros(3, dtype=torch.long)
    for batch in loader:
        depth = batch["depth"].to(device) + 1
        logits = _forward(model, batch, device)
        budgets = (batch["budget"].view(-1).cpu().numpy() * 48.0).round().astype(np.int64)
        preds = [allocate_from_logits(logits[i : i + 1], int(B))[0] for i, B in enumerate(budgets.tolist())]
        pred = torch.stack(preds, dim=0)
        exact += int((pred == depth).all(dim=1).sum().item())
        total += depth.size(0)
        budget_t = torch.tensor(budgets, device=pred.device, dtype=pred.dtype)
        budget_ok += int((pred.sum(dim=1) == budget_t).sum().item())
        pos_correct += int((pred == depth).sum().item())
        pos_total += int(depth.numel())
        hamming_sum += float((pred != depth).float().mean(dim=1).sum().item())
        for d in (1, 2, 3):
            depth_hist[d - 1] += int((pred == d).sum().item())
    return {
        "exact_map_acc": exact / max(total, 1),
        "position_acc": pos_correct / max(pos_total, 1),
        "hamming": hamming_sum / max(total, 1),
        "budget_match_rate": budget_ok / max(total, 1),
        "pred_depth_frac": {
            "1": float(depth_hist[0] / max(pos_total, 1)),
            "2": float(depth_hist[1] / max(pos_total, 1)),
            "3": float(depth_hist[2] / max(pos_total, 1)),
        },
        "n": total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="artifacts/apb_rvq/labels_phase_a/labels.parquet")
    ap.add_argument("--bar-features", default="", help="dir or npz with obs_pooled_ctx (Phase B)")
    ap.add_argument("--output", default="artifacts/apb_rvq/router_phase_a")
    ap.add_argument("--budgets", default="28")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--codebook-size", type=int, default=2048)
    ap.add_argument("--ctx-proj", type=int, default=128)
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    bar_feats = load_bar_feature_table(args.bar_features) if args.bar_features else None
    ds = LabelDataset.from_parquet(args.labels, budgets=budgets, bar_feats=bar_feats)
    if len(ds) == 0:
        raise RuntimeError("Empty label dataset")

    train_rows, val_rows, split_meta = episode_grouped_split(ds.rows, args.val_frac, args.seed)
    train_loader = DataLoader(
        LabelDataset(train_rows, bar_feats=bar_feats), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        LabelDataset(val_rows, bar_feats=bar_feats), batch_size=args.batch_size, shuffle=False
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    split_path = out / "split.json"
    split_path.write_text(
        json.dumps(
            {
                **{k: v for k, v in split_meta.items() if k not in ("train_episode_ids", "val_episode_ids")},
                "train_episode_ids": split_meta["train_episode_ids"],
                "val_episode_ids": split_meta["val_episode_ids"],
                "budgets": budgets,
                "labels": args.labels,
                "bar_features": args.bar_features or None,
                "phase": "B" if bar_feats is not None else "A",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ctx_dim = int(next(iter(bar_feats.values())).shape[0]) if bar_feats else 0
    model = DepthRouterMLP(
        vocab=args.codebook_size, ctx_dim=ctx_dim, ctx_proj=args.ctx_proj
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    best = {"position_acc": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            logits = _forward(model, batch, args.device)
            loss = ordinal_depth_loss(logits, batch["depth"].to(args.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        metrics = eval_maps(model, val_loader, args.device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(row)
        print(json.dumps(row))
        if metrics["position_acc"] > best.get("position_acc", -1.0):
            best = row
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "metrics": row,
                    "split_path": str(split_path),
                    "ctx_dim": ctx_dim,
                },
                out / "best.pt",
            )

    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "best_metrics.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "best": best,
                "output": str(out),
                "phase": "B" if ctx_dim else "A",
                "ctx_dim": ctx_dim,
                "split": {k: split_meta[k] for k in ("n_train_episodes", "n_val_episodes", "n_train_rows", "n_val_rows")},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
