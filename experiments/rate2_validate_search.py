#!/usr/bin/env python3
"""Validate greedy oracle with beam and exact subset checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_rvq.allocation import beam_search_depth, enumerate_depths_for_budget
from adaptive_rvq.codec import decode_with_depth, encode_actions, load_codec, projected_codebooks
from adaptive_rvq.data import LIBERO_DATASET_ID, LIBERO_REVISION, load_libero_chunks
from adaptive_rvq.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", default="artifacts/apb_rvq/oracle")
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--n-beam-chunks", type=int, default=64)
    ap.add_argument("--exact-positions", type=int, default=8)
    ap.add_argument("--n-exact-chunks", type=int, default=32)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true", help="Resume from validate_checkpoint.json in --oracle-dir.")
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N completed validation chunks.",
    )
    return ap.parse_args()


def _checkpoint_path(oracle_dir: Path) -> Path:
    return oracle_dir / "validate_checkpoint.json"


def _load_checkpoint(oracle_dir: Path, resume: bool) -> dict | None:
    path = _checkpoint_path(oracle_dir)
    if not resume or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_checkpoint(
    oracle_dir: Path,
    *,
    budget: int,
    beam_width: int,
    beam_ids: list[int],
    exact_ids: list[int],
    beam_ratios: list[float],
    exact_ratios: list[float],
) -> None:
    payload = {
        "budget": budget,
        "beam_width": beam_width,
        "beam_ids": [int(x) for x in beam_ids],
        "exact_ids": [int(x) for x in exact_ids],
        "beam_ratios": [float(x) for x in beam_ratios],
        "exact_ratios": [float(x) for x in exact_ratios],
        "beam_done": len(beam_ratios),
        "exact_done": len(exact_ratios),
    }
    _checkpoint_path(oracle_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clear_checkpoint(oracle_dir: Path) -> None:
    path = _checkpoint_path(oracle_dir)
    if path.exists():
        path.unlink()


def score_one(model, E, codes_i, action_i, depth_1d, embodiment_id) -> float:
    rec = decode_with_depth(model, E, codes_i.unsqueeze(0), depth_1d.unsqueeze(0), embodiment_id=embodiment_id)
    return compute_metrics(action_i.unsqueeze(0), rec)["rms"]


def main() -> None:
    args = parse_args()
    oracle_dir = Path(args.oracle_dir)
    metrics = json.loads((oracle_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = json.loads((oracle_dir / "config.json").read_text(encoding="utf-8"))
    depth_npz = np.load(oracle_dir / "depth_maps.npz")
    budgets = [int(x) for x in cfg["budgets"].split(",")]
    if args.budget not in budgets:
        raise ValueError(f"Budget {args.budget} not found in {budgets}")
    budget_idx = budgets.index(args.budget)

    model = load_codec(model_id=cfg["model"], device=args.device)
    E = projected_codebooks(model, device=args.device)
    batch = load_libero_chunks(
        n_chunks=int(cfg["n_chunks"]),
        n_episodes=int(cfg["n_episodes"]),
        seed=int(cfg["seed"]),
        dataset_id=cfg.get("dataset", LIBERO_DATASET_ID),
        revision=cfg.get("revision", LIBERO_REVISION),
        device=args.device,
        gripper_mode=metrics["meta"]["gripper_mode"],
        model=model,
        embodiment_id=int(cfg["embodiment"]),
    )
    codes = encode_actions(model, batch.actions, embodiment_id=int(cfg["embodiment"]))
    depth_exact = torch.from_numpy(depth_npz["depth_exact"]).to(args.device)

    rng = np.random.default_rng(int(cfg["seed"]) + 99)
    chunk_ids = np.arange(codes.shape[0])
    rng.shuffle(chunk_ids)

    beam_ids = chunk_ids[: min(args.n_beam_chunks, len(chunk_ids))]
    exact_ids = chunk_ids[: min(args.n_exact_chunks, len(chunk_ids))]
    checkpoint = _load_checkpoint(oracle_dir, args.resume)
    beam_ratios: list[float] = []
    exact_ratios: list[float] = []
    beam_start = 0
    exact_start = 0

    if checkpoint is not None:
        if int(checkpoint.get("budget", -1)) != args.budget or int(checkpoint.get("beam_width", -1)) != args.beam_width:
            raise RuntimeError("Checkpoint budget/beam_width mismatch; delete validate_checkpoint.json to restart.")
        if checkpoint.get("beam_ids") != [int(x) for x in beam_ids] or checkpoint.get("exact_ids") != [int(x) for x in exact_ids]:
            raise RuntimeError("Checkpoint chunk selection mismatch; delete validate_checkpoint.json to restart.")
        beam_ratios = [float(x) for x in checkpoint.get("beam_ratios", [])]
        exact_ratios = [float(x) for x in checkpoint.get("exact_ratios", [])]
        beam_start = len(beam_ratios)
        exact_start = len(exact_ratios)

    for offset, idx in enumerate(
        tqdm(beam_ids[beam_start:], desc="beam", initial=beam_start, total=len(beam_ids))
    ):
        a = batch.actions[idx]
        c = codes[idx]

        def score(d):
            return score_one(model, E, c, a, d.to(args.device), int(cfg["embodiment"]))

        d1 = torch.ones(c.shape[0], dtype=torch.long, device=args.device)
        greedy = depth_exact[idx, budget_idx]
        greedy_err = score(greedy)
        base_err = score(d1)
        beam_depth, beam_err = beam_search_depth(
            score, n_positions=c.shape[0], budget=args.budget, beam_width=args.beam_width, max_depth=3
        )
        denom = max(base_err - beam_err, 1e-12)
        retained = (base_err - greedy_err) / denom
        beam_ratios.append(float(retained))
        done = beam_start + offset + 1
        if args.checkpoint_every > 0 and (done % args.checkpoint_every == 0 or done == len(beam_ids)):
            _save_checkpoint(
                oracle_dir,
                budget=args.budget,
                beam_width=args.beam_width,
                beam_ids=[int(x) for x in beam_ids],
                exact_ids=[int(x) for x in exact_ids],
                beam_ratios=beam_ratios,
                exact_ratios=exact_ratios,
            )

    for offset, idx in enumerate(
        tqdm(exact_ids[exact_start:], desc="exact_subset", initial=exact_start, total=len(exact_ids))
    ):
        a = batch.actions[idx]
        c = codes[idx]

        def score(d):
            return score_one(model, E, c, a, d.to(args.device), int(cfg["embodiment"]))

        d1 = torch.ones(c.shape[0], dtype=torch.long, device=args.device)
        greedy = depth_exact[idx, budget_idx]
        greedy_err = score(greedy)
        base_err = score(d1)

        subset = sorted(rng.choice(c.shape[0], size=min(args.exact_positions, c.shape[0]), replace=False).tolist())
        free_budget = args.budget - (c.shape[0] - len(subset))
        best_err = None
        for state in enumerate_depths_for_budget(len(subset), free_budget, max_depth=3):
            cand = torch.ones(c.shape[0], dtype=torch.long, device=args.device)
            for pos, dep in zip(subset, state):
                cand[pos] = int(dep)
            err = score(cand)
            if best_err is None or err < best_err:
                best_err = err
        denom = max(base_err - best_err, 1e-12)
        retained = (base_err - greedy_err) / denom
        exact_ratios.append(float(retained))
        done = exact_start + offset + 1
        if args.checkpoint_every > 0 and (done % args.checkpoint_every == 0 or done == len(exact_ids)):
            _save_checkpoint(
                oracle_dir,
                budget=args.budget,
                beam_width=args.beam_width,
                beam_ids=[int(x) for x in beam_ids],
                exact_ids=[int(x) for x in exact_ids],
                beam_ratios=beam_ratios,
                exact_ratios=exact_ratios,
            )

    out = {
        "budget": args.budget,
        "beam_width": args.beam_width,
        "n_beam_chunks": int(len(beam_ids)),
        "n_exact_chunks": int(len(exact_ids)),
        "beam_retained_mean": float(np.mean(beam_ratios)) if beam_ratios else float("nan"),
        "beam_retained_median": float(np.median(beam_ratios)) if beam_ratios else float("nan"),
        "exact_subset_retained_mean": float(np.mean(exact_ratios)) if exact_ratios else float("nan"),
        "exact_subset_retained_median": float(np.median(exact_ratios)) if exact_ratios else float("nan"),
        "passes_95pct_beam": bool(np.mean(beam_ratios) >= 0.95) if beam_ratios else False,
        "passes_95pct_exact_subset": bool(np.mean(exact_ratios) >= 0.95) if exact_ratios else False,
    }
    (oracle_dir / "validate_search.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _clear_checkpoint(oracle_dir)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
