#!/usr/bin/env python3
"""Gate-1 oracle experiment for APB-RVQ with matched-budget baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_rvq.allocation import (
    greedy_oracle_single_mode,
    random_depth_map,
    round_robin_depth_map,
    validate_depth_map,
)
from adaptive_rvq.codec import decode_with_depth, encode_actions, load_codec, projected_codebooks
from adaptive_rvq.data import LIBERO_DATASET_ID, LIBERO_REVISION, load_libero_chunks
from adaptive_rvq.metrics import (
    bootstrap_ci,
    compute_metrics,
    gap_closed_fraction,
    mean_random_per_chunk,
)

CHECKPOINT_VERSION = 1


@dataclass
class EvalRow:
    method: str
    mode: str
    budget: int
    chunk_idx: int
    episode_id: int
    task_id: int
    rms: float
    mse: float
    gripper_error: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--dataset", default=LIBERO_DATASET_ID)
    ap.add_argument("--revision", default=LIBERO_REVISION)
    ap.add_argument("--n-chunks", type=int, default=256)
    ap.add_argument("--n-episodes", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--budgets", default="16,20,24,28,32,36,40,44,48")
    ap.add_argument("--random-seeds", type=int, default=8)
    ap.add_argument("--static-prior-chunks", type=int, default=96)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="artifacts/apb_rvq/oracle")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint.json in --output.")
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N completed oracle chunks.",
    )
    return ap.parse_args()


def _run_fingerprint(args: argparse.Namespace, budgets: list[int]) -> str:
    payload = {
        "budgets": budgets,
        "dataset": args.dataset,
        "embodiment": args.embodiment,
        "model": args.model,
        "n_chunks": args.n_chunks,
        "n_episodes": args.n_episodes,
        "random_seeds": args.random_seeds,
        "revision": args.revision,
        "seed": args.seed,
        "static_prior_chunks": args.static_prior_chunks,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _checkpoint_paths(out_dir: Path) -> tuple[Path, Path, Path]:
    return out_dir / "checkpoint.json", out_dir / "checkpoint.npz", out_dir / "checkpoint_rows.parquet"


def _load_checkpoint(out_dir: Path, fingerprint: str, resume: bool) -> dict | None:
    meta_path, npz_path, rows_path = _checkpoint_paths(out_dir)
    if not resume or not meta_path.exists() or not npz_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError(f"Unsupported checkpoint version: {meta.get('version')}")
    if meta.get("fingerprint") != fingerprint:
        raise RuntimeError(
            "Checkpoint fingerprint mismatch; use the same run args or delete checkpoint files."
        )
    arrays = dict(np.load(npz_path))
    rows: list[EvalRow] = []
    if rows_path.exists():
        table = pq.read_table(rows_path)
        rows = [EvalRow(**row) for row in table.to_pylist()]
    return {"meta": meta, "arrays": arrays, "rows": rows}


def _save_checkpoint(
    out_dir: Path,
    *,
    fingerprint: str,
    phase: str,
    meta_extra: dict,
    arrays: dict[str, np.ndarray],
    rows: list[EvalRow],
) -> None:
    meta_path, npz_path, rows_path = _checkpoint_paths(out_dir)
    meta = {
        "version": CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "phase": phase,
        **meta_extra,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez_compressed(npz_path, **arrays)
    if rows:
        pq.write_table(pa.Table.from_pylist([asdict(r) for r in rows]), rows_path)


def _clear_checkpoint(out_dir: Path) -> None:
    for path in _checkpoint_paths(out_dir):
        if path.exists():
            path.unlink()


def _git_head() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return "unknown"


def eval_one(model, E, codes_i: torch.Tensor, action_i: torch.Tensor, depth_1d: torch.Tensor, embodiment_id: int):
    depth_1d = depth_1d.to(codes_i.device)
    validate_depth_map(depth_1d, budget=int(depth_1d.sum().item()))
    rec = decode_with_depth(
        model,
        E,
        codes_i.unsqueeze(0),
        depth_1d.unsqueeze(0),
        embodiment_id=embodiment_id,
    )
    return compute_metrics(action_i.unsqueeze(0), rec)


def build_static_prior_depth_maps(
    model,
    E,
    train_idx: np.ndarray,
    codes: torch.Tensor,
    actions: torch.Tensor,
    budgets: list[int],
    embodiment_id: int,
    *,
    g12_sum: np.ndarray | None = None,
    g23_sum: np.ndarray | None = None,
    start_at: int = 0,
) -> tuple[dict[int, torch.Tensor], np.ndarray, np.ndarray]:
    """Fixed depth maps from averaged singleton gains on train chunks."""

    n_pos = codes.shape[1]
    device = codes.device
    depth1 = torch.ones(n_pos, dtype=torch.long, device=device)
    depth2 = torch.full((n_pos,), 2, dtype=torch.long, device=device)
    g12 = np.zeros(n_pos, dtype=np.float64) if g12_sum is None else g12_sum.copy()
    g23 = np.zeros(n_pos, dtype=np.float64) if g23_sum is None else g23_sum.copy()
    train_idx = np.asarray(train_idx, dtype=np.int64)
    for offset, idx in enumerate(
        tqdm(train_idx[start_at:], desc="static_prior", initial=start_at, total=len(train_idx))
    ):
        m1 = eval_one(model, E, codes[idx], actions[idx], depth1, embodiment_id)["rms"]
        for pos in range(n_pos):
            d = depth1.clone()
            d[pos] = 2
            g12[pos] += m1 - eval_one(model, E, codes[idx], actions[idx], d, embodiment_id)["rms"]
        m2 = eval_one(model, E, codes[idx], actions[idx], depth2, embodiment_id)["rms"]
        for pos in range(n_pos):
            d = depth2.clone()
            d[pos] = 3
            g23[pos] += m2 - eval_one(model, E, codes[idx], actions[idx], d, embodiment_id)["rms"]

    g12_avg = g12 / max(len(train_idx), 1)
    g23_avg = g23 / max(len(train_idx), 1)

    out = {}
    for budget in budgets:
        depth = torch.ones(n_pos, dtype=torch.long, device=device)
        while int(depth.sum()) < budget:
            legal = []
            for pos in range(n_pos):
                if depth[pos] == 1:
                    legal.append((float(g12_avg[pos]), pos))
                elif depth[pos] == 2:
                    legal.append((float(g23_avg[pos]), pos))
            best = max(legal, key=lambda x: x[0])[1]
            depth[best] += 1
        out[budget] = depth.clone()
    return out, g12.copy(), g23.copy()


def _restore_error_dicts(arrays: dict[str, np.ndarray], budgets: list[int]) -> dict[str, dict[int, list[float]]]:
    out = {
        "oracle_exact": {b: arrays[f"oracle_exact_{b}"].tolist() for b in budgets},
        "oracle_atmost": {b: arrays[f"oracle_atmost_{b}"].tolist() for b in budgets},
        "random": {b: arrays[f"random_{b}"].tolist() for b in budgets},
        "uniform": {b: arrays[f"uniform_{b}"].tolist() for b in budgets},
        "static": {b: arrays[f"static_{b}"].tolist() for b in budgets},
        "global_mix": {b: arrays[f"global_mix_{b}"].tolist() for b in budgets},
    }
    return out


def _error_dicts_to_arrays(error_dicts: dict[str, dict[int, list[float]]], budgets: list[int]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for prefix in ("oracle_exact", "oracle_atmost", "random", "uniform", "static", "global_mix"):
        for budget in budgets:
            arrays[f"{prefix}_{budget}"] = np.asarray(error_dicts[prefix][budget], dtype=np.float64)
    return arrays


def main() -> None:
    args = parse_args()
    budgets = sorted(int(x) for x in args.budgets.split(","))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _run_fingerprint(args, budgets)
    checkpoint = _load_checkpoint(out_dir, fingerprint, args.resume)

    model = load_codec(model_id=args.model, device=args.device)
    E = projected_codebooks(model, device=args.device)
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
    actions = batch.actions
    codes = encode_actions(model, actions, embodiment_id=args.embodiment)
    n_chunks, n_pos, n_levels = codes.shape
    assert n_levels == 3

    depth1 = torch.ones(n_pos, dtype=torch.long, device=args.device)
    depth3 = torch.full((n_pos,), n_levels, dtype=torch.long, device=args.device)
    d2 = torch.full((n_pos,), 2, dtype=torch.long, device=args.device)

    global_metrics = {16: [], 32: [], 48: []}
    static_depth: dict[int, torch.Tensor] = {}
    rows: list[EvalRow] = []
    depth_maps_exact = np.zeros((n_chunks, len(budgets), n_pos), dtype=np.int16)
    depth_maps_atmost = np.zeros((n_chunks, len(budgets), n_pos), dtype=np.int16)
    oracle_exact_errors = {b: [] for b in budgets}
    oracle_atmost_errors = {b: [] for b in budgets}
    random_errors = {b: [] for b in budgets}
    uniform_errors = {b: [] for b in budgets}
    static_errors = {b: [] for b in budgets}
    global_mix_errors = {b: [] for b in budgets}

    phase = "global_depth"
    oracle_start = 0
    static_prior_start = 0
    g12_sum = None
    g23_sum = None
    train_idx = np.asarray([], dtype=np.int64)

    if checkpoint is not None:
        phase = checkpoint["meta"]["phase"]
        arrays = checkpoint["arrays"]
        rows = checkpoint["rows"]
        global_metrics[16] = [{"rms": float(x)} for x in arrays["global_rms_16"].tolist()]
        global_metrics[32] = [{"rms": float(x)} for x in arrays["global_rms_32"].tolist()]
        global_metrics[48] = [{"rms": float(x)} for x in arrays["global_rms_48"].tolist()]
        depth_maps_exact = arrays["depth_maps_exact"]
        depth_maps_atmost = arrays["depth_maps_atmost"]
        train_idx = arrays["train_idx"]
        error_dicts = _restore_error_dicts(arrays, budgets)
        oracle_exact_errors = error_dicts["oracle_exact"]
        oracle_atmost_errors = error_dicts["oracle_atmost"]
        random_errors = error_dicts["random"]
        uniform_errors = error_dicts["uniform"]
        static_errors = error_dicts["static"]
        global_mix_errors = error_dicts["global_mix"]
        oracle_start = int(checkpoint["meta"].get("oracle_done", 0))
        static_prior_start = int(checkpoint["meta"].get("static_prior_done", 0))
        if "g12_sum" in arrays:
            g12_sum = arrays["g12_sum"]
            g23_sum = arrays["g23_sum"]
        if phase in {"oracle", "done"} and all(f"static_depth_{budget}" in arrays for budget in budgets):
            static_depth = {
                budget: torch.from_numpy(arrays[f"static_depth_{budget}"]).to(args.device)
                for budget in budgets
            }

    if phase == "global_depth":
        for i in tqdm(range(n_chunks), desc="global_depth"):
            for budget, depth in ((16, depth1), (32, d2), (48, depth3)):
                global_metrics[budget].append(eval_one(model, E, codes[i], actions[i], depth, args.embodiment))
        _save_checkpoint(
            out_dir,
            fingerprint=fingerprint,
            phase="static_prior",
            meta_extra={"global_depth_done": n_chunks, "static_prior_done": 0, "oracle_done": 0},
            arrays={
                "global_rms_16": np.asarray([x["rms"] for x in global_metrics[16]], dtype=np.float64),
                "global_rms_32": np.asarray([x["rms"] for x in global_metrics[32]], dtype=np.float64),
                "global_rms_48": np.asarray([x["rms"] for x in global_metrics[48]], dtype=np.float64),
                "depth_maps_exact": depth_maps_exact,
                "depth_maps_atmost": depth_maps_atmost,
                "train_idx": train_idx,
                **_error_dicts_to_arrays(
                    {
                        "oracle_exact": oracle_exact_errors,
                        "oracle_atmost": oracle_atmost_errors,
                        "random": random_errors,
                        "uniform": uniform_errors,
                        "static": static_errors,
                        "global_mix": global_mix_errors,
                    },
                    budgets,
                ),
            },
            rows=rows,
        )
        phase = "static_prior"
        static_prior_start = 0

    if phase == "static_prior":
        if len(train_idx) == 0:
            uniq_eps = np.unique(batch.episode_ids)
            rng = np.random.default_rng(args.seed)
            rng.shuffle(uniq_eps)
            train_eps = set(uniq_eps[: max(1, int(0.7 * len(uniq_eps)))].tolist())
            train_idx = np.asarray([i for i, ep in enumerate(batch.episode_ids) if int(ep) in train_eps], dtype=np.int64)
            if len(train_idx) > args.static_prior_chunks:
                train_idx = rng.choice(train_idx, size=args.static_prior_chunks, replace=False)
        static_depth, g12_sum, g23_sum = build_static_prior_depth_maps(
            model,
            E,
            train_idx,
            codes,
            actions,
            budgets,
            embodiment_id=args.embodiment,
            g12_sum=g12_sum,
            g23_sum=g23_sum,
            start_at=static_prior_start,
        )
        static_arrays = {
            f"static_depth_{budget}": static_depth[budget].detach().cpu().numpy().astype(np.int16)
            for budget in budgets
        }
        _save_checkpoint(
            out_dir,
            fingerprint=fingerprint,
            phase="oracle",
            meta_extra={
                "global_depth_done": n_chunks,
                "static_prior_done": len(train_idx),
                "oracle_done": 0,
            },
            arrays={
                "global_rms_16": np.asarray([x["rms"] for x in global_metrics[16]], dtype=np.float64),
                "global_rms_32": np.asarray([x["rms"] for x in global_metrics[32]], dtype=np.float64),
                "global_rms_48": np.asarray([x["rms"] for x in global_metrics[48]], dtype=np.float64),
                "depth_maps_exact": depth_maps_exact,
                "depth_maps_atmost": depth_maps_atmost,
                "train_idx": train_idx,
                "g12_sum": g12_sum,
                "g23_sum": g23_sum,
                **static_arrays,
                **_error_dicts_to_arrays(
                    {
                        "oracle_exact": oracle_exact_errors,
                        "oracle_atmost": oracle_atmost_errors,
                        "random": random_errors,
                        "uniform": uniform_errors,
                        "static": static_errors,
                        "global_mix": global_mix_errors,
                    },
                    budgets,
                ),
            },
            rows=rows,
        )
        phase = "oracle"
        oracle_start = 0

    for chunk_idx in tqdm(range(oracle_start, n_chunks), desc="oracle", initial=oracle_start, total=n_chunks):
        action_i = actions[chunk_idx]
        codes_i = codes[chunk_idx]

        def score_depth(d1d: torch.Tensor) -> float:
            return eval_one(model, E, codes_i, action_i, d1d.to(args.device), args.embodiment)["rms"]

        path_exact, _ = greedy_oracle_single_mode(score_depth, n_pos, budgets, mode="exact-budget", max_depth=3)
        path_atmost, _ = greedy_oracle_single_mode(score_depth, n_pos, budgets, mode="at-most-budget", max_depth=3)
        last_atmost_depth = torch.ones(n_pos, dtype=torch.long, device=args.device)

        for bi, budget in enumerate(budgets):
            exact_depth = path_exact[budget]["depth"]
            exact_metrics = eval_one(model, E, codes_i, action_i, exact_depth, args.embodiment)
            oracle_exact_errors[budget].append(exact_metrics["rms"])
            depth_maps_exact[chunk_idx, bi] = exact_depth.cpu().numpy().astype(np.int16)
            rows.append(
                EvalRow(
                    method="oracle",
                    mode="exact-budget",
                    budget=budget,
                    chunk_idx=chunk_idx,
                    episode_id=int(batch.episode_ids[chunk_idx]),
                    task_id=int(batch.task_ids[chunk_idx]),
                    rms=exact_metrics["rms"],
                    mse=exact_metrics["mse"],
                    gripper_error=exact_metrics["gripper_error"],
                )
            )

            if budget in path_atmost:
                last_atmost_depth = path_atmost[budget]["depth"]
            atmost_metrics = eval_one(model, E, codes_i, action_i, last_atmost_depth, args.embodiment)
            oracle_atmost_errors[budget].append(atmost_metrics["rms"])
            depth_maps_atmost[chunk_idx, bi] = last_atmost_depth.cpu().numpy().astype(np.int16)
            rows.append(
                EvalRow(
                    method="oracle",
                    mode="at-most-budget",
                    budget=budget,
                    chunk_idx=chunk_idx,
                    episode_id=int(batch.episode_ids[chunk_idx]),
                    task_id=int(batch.task_ids[chunk_idx]),
                    rms=atmost_metrics["rms"],
                    mse=atmost_metrics["mse"],
                    gripper_error=atmost_metrics["gripper_error"],
                )
            )

            uni_depth = round_robin_depth_map(n_pos, budget).to(args.device)
            uni_metrics = eval_one(model, E, codes_i, action_i, uni_depth, args.embodiment)
            uniform_errors[budget].append(uni_metrics["rms"])

            st_depth = static_depth[budget]
            st_metrics = eval_one(model, E, codes_i, action_i, st_depth, args.embodiment)
            static_errors[budget].append(st_metrics["rms"])

            mix = None
            if budget <= 32:
                w = (budget - 16) / 16.0
                g1 = global_metrics[16][chunk_idx]["rms"]
                g2 = global_metrics[32][chunk_idx]["rms"]
                mix = (1.0 - w) * g1 + w * g2
            else:
                w = (budget - 32) / 16.0
                g2 = global_metrics[32][chunk_idx]["rms"]
                g3 = global_metrics[48][chunk_idx]["rms"]
                mix = (1.0 - w) * g2 + w * g3
            global_mix_errors[budget].append(float(mix))

            rows.extend(
                [
                    EvalRow(
                        method="uniform",
                        mode="exact-budget",
                        budget=budget,
                        chunk_idx=chunk_idx,
                        episode_id=int(batch.episode_ids[chunk_idx]),
                        task_id=int(batch.task_ids[chunk_idx]),
                        rms=uni_metrics["rms"],
                        mse=uni_metrics["mse"],
                        gripper_error=uni_metrics["gripper_error"],
                    ),
                    EvalRow(
                        method="static",
                        mode="exact-budget",
                        budget=budget,
                        chunk_idx=chunk_idx,
                        episode_id=int(batch.episode_ids[chunk_idx]),
                        task_id=int(batch.task_ids[chunk_idx]),
                        rms=st_metrics["rms"],
                        mse=st_metrics["mse"],
                        gripper_error=st_metrics["gripper_error"],
                    ),
                    EvalRow(
                        method="global_mix",
                        mode="exact-budget",
                        budget=budget,
                        chunk_idx=chunk_idx,
                        episode_id=int(batch.episode_ids[chunk_idx]),
                        task_id=int(batch.task_ids[chunk_idx]),
                        rms=float(mix),
                        mse=float("nan"),
                        gripper_error=float("nan"),
                    ),
                ]
            )

        for seed in range(args.random_seeds):
            rrng = np.random.default_rng(args.seed + 1000 * seed + chunk_idx)
            for budget in budgets:
                r_depth = random_depth_map(n_pos, budget, rrng).to(args.device)
                r_metrics = eval_one(model, E, codes_i, action_i, r_depth, args.embodiment)
                random_errors[budget].append(r_metrics["rms"])
                rows.append(
                    EvalRow(
                        method="random",
                        mode=f"seed-{seed}",
                        budget=budget,
                        chunk_idx=chunk_idx,
                        episode_id=int(batch.episode_ids[chunk_idx]),
                        task_id=int(batch.task_ids[chunk_idx]),
                        rms=r_metrics["rms"],
                        mse=r_metrics["mse"],
                        gripper_error=r_metrics["gripper_error"],
                    )
                )

        completed = chunk_idx + 1
        if args.checkpoint_every > 0 and (
            completed % args.checkpoint_every == 0 or completed == n_chunks
        ):
            static_arrays = {
                f"static_depth_{budget}": static_depth[budget].detach().cpu().numpy().astype(np.int16)
                for budget in budgets
            }
            _save_checkpoint(
                out_dir,
                fingerprint=fingerprint,
                phase="done" if completed == n_chunks else "oracle",
                meta_extra={
                    "global_depth_done": n_chunks,
                    "static_prior_done": len(train_idx),
                    "oracle_done": completed,
                },
                arrays={
                    "global_rms_16": np.asarray([x["rms"] for x in global_metrics[16]], dtype=np.float64),
                    "global_rms_32": np.asarray([x["rms"] for x in global_metrics[32]], dtype=np.float64),
                    "global_rms_48": np.asarray([x["rms"] for x in global_metrics[48]], dtype=np.float64),
                    "depth_maps_exact": depth_maps_exact,
                    "depth_maps_atmost": depth_maps_atmost,
                    "train_idx": train_idx,
                    "g12_sum": g12_sum if g12_sum is not None else np.zeros(n_pos),
                    "g23_sum": g23_sum if g23_sum is not None else np.zeros(n_pos),
                    **static_arrays,
                    **_error_dicts_to_arrays(
                        {
                            "oracle_exact": oracle_exact_errors,
                            "oracle_atmost": oracle_atmost_errors,
                            "random": random_errors,
                            "uniform": uniform_errors,
                            "static": static_errors,
                            "global_mix": global_mix_errors,
                        },
                        budgets,
                    ),
                },
                rows=rows,
            )

    summary = {
        "meta": {
            "model": args.model,
            "dataset": args.dataset,
            "revision": args.revision,
            "commit": _git_head(),
            "device": args.device,
            "seed": args.seed,
            "n_chunks": n_chunks,
            "budgets": budgets,
            "gripper_mode": batch.gripper_mode,
        },
        "oracle_exact": {},
        "oracle_at_most": {},
        "baselines": {},
    }

    for budget in budgets:
        exact_arr = np.asarray(oracle_exact_errors[budget], dtype=np.float64)
        atmost_arr = np.asarray(oracle_atmost_errors[budget], dtype=np.float64)
        rnd_arr = np.asarray(random_errors[budget], dtype=np.float64)
        uni_arr = np.asarray(uniform_errors[budget], dtype=np.float64)
        st_arr = np.asarray(static_errors[budget], dtype=np.float64)
        gm_arr = np.asarray(global_mix_errors[budget], dtype=np.float64)
        g1 = np.asarray([x["rms"] for x in global_metrics[16]], dtype=np.float64)
        g3 = np.asarray([x["rms"] for x in global_metrics[48]], dtype=np.float64)
        rnd_per_chunk = mean_random_per_chunk(rnd_arr, n_chunks=len(exact_arr), n_seeds=args.random_seeds)
        delta = exact_arr - rnd_per_chunk
        ci = bootstrap_ci(delta, batch.episode_ids, n_boot=200, seed=args.seed)
        best_nonadaptive = min(float(uni_arr.mean()), float(st_arr.mean()), float(gm_arr.mean()))
        improvement_vs_random = float(
            (rnd_per_chunk.mean() - exact_arr.mean()) / max(rnd_per_chunk.mean(), 1e-12) * 100.0
        )
        improvement_vs_best = float(
            (best_nonadaptive - exact_arr.mean()) / max(best_nonadaptive, 1e-12) * 100.0
        )

        summary["oracle_exact"][str(budget)] = {
            "rms_mean": float(exact_arr.mean()),
            "rms_ci_vs_random_delta": [float(ci[0]), float(ci[1])],
            "improvement_vs_random_pct": improvement_vs_random,
            "improvement_vs_best_nonadaptive_pct": improvement_vs_best,
            "ci_vs_random_excludes_zero": bool(ci[1] < 0.0 or ci[0] > 0.0),
            "gap_closed_mean": float(
                np.nanmean([gap_closed_fraction(a, b, c) for a, b, c in zip(g1, exact_arr, g3)])
            ),
        }
        summary["oracle_at_most"][str(budget)] = {
            "rms_mean": float(atmost_arr.mean()),
        }
        summary["baselines"][str(budget)] = {
            "random_rms_mean": float(rnd_per_chunk.mean()),
            "uniform_rms_mean": float(uni_arr.mean()),
            "static_rms_mean": float(st_arr.mean()),
            "global_mix_rms_mean": float(gm_arr.mean()),
            "best_nonadaptive_rms_mean": best_nonadaptive,
        }

    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out_dir / "git_commit.txt").write_text(summary["meta"]["commit"] + "\n", encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "depth_maps.npz",
        depth_exact=depth_maps_exact,
        depth_at_most=depth_maps_atmost,
        episode_ids=batch.episode_ids,
        task_ids=batch.task_ids,
        starts=batch.starts,
    )
    table = pa.Table.from_pylist([asdict(r) for r in rows])
    pq.write_table(table, out_dir / "per_chunk.parquet")
    _clear_checkpoint(out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
