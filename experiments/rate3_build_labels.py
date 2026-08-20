#!/usr/bin/env python3
"""Phase A rate3: build router labels from oracle depth maps + ActionCodec coarse codes.

No rate1 recompute. Reloads exact (episode_id, start) chunks from oracle artifacts.
Primary target: depth_map. Nested gates are derived diagnostics only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_rvq.codec import encode_actions, load_codec
from adaptive_rvq.data import load_libero_chunks_indexed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", default="artifacts/apb_rvq/oracle_full")
    ap.add_argument("--output", default="artifacts/apb_rvq/labels_phase_a")
    ap.add_argument("--budgets", default="20,24,28,32,36,40")
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-end", type=int, default=-1, help="Exclusive end; -1 = all")
    ap.add_argument("--window-size", type=int, default=128, help="Chunks per load/encode window")
    ap.add_argument("--encode-batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def _sin_cos_pos(n_pos: int = 16, dim: int = 32) -> np.ndarray:
    pos = np.arange(n_pos, dtype=np.float32)[:, None]
    i = np.arange(dim // 2, dtype=np.float32)[None, :]
    angles = pos / np.power(10000.0, (2 * i) / dim)
    out = np.zeros((n_pos, dim), dtype=np.float32)
    out[:, 0::2] = np.sin(angles)
    out[:, 1::2] = np.cos(angles)
    return out


def main() -> None:
    args = parse_args()
    oracle_dir = Path(args.oracle_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((oracle_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = json.loads((oracle_dir / "config.json").read_text(encoding="utf-8"))
    depth_npz = np.load(oracle_dir / "depth_maps.npz")
    depth_exact = depth_npz["depth_exact"]  # [N, B, P]
    episode_ids = depth_npz["episode_ids"].astype(np.int64)
    task_ids = depth_npz["task_ids"].astype(np.int64)
    starts = depth_npz["starts"].astype(np.int64)

    all_budgets = [int(x) for x in str(cfg["budgets"]).split(",")]
    budgets = [int(x) for x in args.budgets.split(",")]
    budget_indices = [all_budgets.index(b) for b in budgets]

    n_total = int(depth_exact.shape[0])
    shard_end = n_total if args.shard_end < 0 else min(args.shard_end, n_total)
    shard_start = max(0, args.shard_start)
    if shard_start >= shard_end:
        raise ValueError(f"Empty shard: [{shard_start}, {shard_end})")

    shard_name = f"shard_{shard_start:05d}_{shard_end:05d}"
    shard_path = out_dir / f"{shard_name}.parquet"
    ckpt_path = out_dir / f"{shard_name}.checkpoint.json"

    done_until = shard_start
    rows: list[dict] = []
    if args.resume and ckpt_path.exists() and shard_path.exists():
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        done_until = int(ckpt["done_until"])
        rows = pq.read_table(shard_path).to_pylist()
        print(f"[rate3] resume {shard_name} from {done_until} ({len(rows)} rows)")

    if done_until >= shard_end:
        print(f"[rate3] shard already complete: {shard_path}")
        return

    model = load_codec(model_id=cfg.get("model", metrics["meta"]["model"]), device=args.device)
    pos_emb = _sin_cos_pos(16, 32)
    dataset_id = cfg.get("dataset", metrics["meta"]["dataset"])
    revision = cfg.get("revision", metrics["meta"]["revision"])
    gripper_mode = metrics["meta"]["gripper_mode"]
    embodiment = int(cfg.get("embodiment", 0))

    cursor = done_until
    pbar = tqdm(total=shard_end - done_until, desc=f"labels[{shard_name}]")
    while cursor < shard_end:
        win_end = min(cursor + args.window_size, shard_end)
        idx = np.arange(cursor, win_end, dtype=np.int64)
        batch = load_libero_chunks_indexed(
            episode_ids=episode_ids[idx],
            starts=starts[idx],
            dataset_id=dataset_id,
            revision=revision,
            device=args.device,
            gripper_mode=gripper_mode,
            include_state=True,
        )
        codes_parts = []
        actions = batch.actions
        for i in range(0, len(actions), args.encode_batch_size):
            codes = encode_actions(model, actions[i : i + args.encode_batch_size], embodiment_id=embodiment)
            codes_parts.append(codes.detach().cpu().numpy().astype(np.int16))
        codes_all = np.concatenate(codes_parts, axis=0)
        coarse = codes_all[:, :, 0]
        states = None if batch.states is None else batch.states.detach().cpu().numpy().astype(np.float32)

        for local_i, global_idx in enumerate(idx.tolist()):
            state_mean = None if states is None else states[local_i].mean(axis=0)
            state_first = None if states is None else states[local_i][0]
            for budget, b_idx in zip(budgets, budget_indices):
                depth = depth_exact[global_idx, b_idx].astype(np.int16)
                rows.append(
                    {
                        "chunk_idx": int(global_idx),
                        "episode_id": int(episode_ids[global_idx]),
                        "task_id": int(task_ids[global_idx]),
                        "start": int(starts[global_idx]),
                        "budget": int(budget),
                        "coarse_codes": coarse[local_i].tolist(),
                        "codes_full": codes_all[local_i].tolist(),
                        "depth_map": depth.tolist(),
                        "nested_gate_ge2": (depth >= 2).astype(np.float32).tolist(),
                        "nested_gate_ge3": (depth >= 3).astype(np.float32).tolist(),
                        "position_embeddings": pos_emb.tolist(),
                        "state_first": None if state_first is None else state_first.tolist(),
                        "state_mean": None if state_mean is None else state_mean.tolist(),
                        "task_name": batch.task_names[local_i],
                    }
                )
        cursor = win_end
        pbar.update(len(idx))
        pq.write_table(pa.Table.from_pylist(rows), shard_path)
        ckpt_path.write_text(
            json.dumps(
                {
                    "done_until": cursor,
                    "shard_start": shard_start,
                    "shard_end": shard_end,
                    "n_rows": len(rows),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    pbar.close()

    if ckpt_path.exists():
        ckpt_path.unlink()
    meta = {
        "oracle_dir": str(oracle_dir),
        "phase": "A",
        "shard_start": shard_start,
        "shard_end": shard_end,
        "budgets": budgets,
        "n_rows": len(rows),
        "n_chunks": shard_end - shard_start,
        "features": [
            "coarse_codes",
            "codes_full",
            "position_embeddings",
            "state_first",
            "state_mean",
            "budget",
        ],
        "primary_target": "depth_map",
        "derived_targets": ["nested_gate_ge2", "nested_gate_ge3"],
        "notes": [
            "nested_gate_* are hard nested indicators derived from depth_map, not soft marginal utilities.",
            "No VLM/image features in Phase A.",
        ],
    }
    (out_dir / f"{shard_name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
