#!/usr/bin/env python3
"""Phase B prep: cache causal visual frames (first image + wrist) for oracle chunks.

Does not invent utilities. Writes compact uint8 JPEG-bytes or raw RGB npz shards.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_rvq.data import LIBERO_DATASET_ID, LIBERO_REVISION, _episode_parquet_path


def _to_uint8_image(x) -> np.ndarray:
    """Decode LIBERO parquet image cells (ndarray or HF dict with bytes)."""
    if isinstance(x, dict):
        if "bytes" in x and x["bytes"] is not None:
            from PIL import Image

            return np.asarray(Image.open(io.BytesIO(x["bytes"])).convert("RGB"), dtype=np.uint8)
        if "path" in x and x["path"]:
            from PIL import Image

            return np.asarray(Image.open(x["path"]).convert("RGB"), dtype=np.uint8)
        # Some HF rows nest under 'array'
        if "array" in x:
            x = x["array"]
    arr = np.asarray(x)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", default="artifacts/apb_rvq/oracle_full")
    ap.add_argument("--output", default="artifacts/apb_rvq/visual_cache_phase_b")
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-end", type=int, default=-1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--jpeg-quality", type=int, default=85)
    args = ap.parse_args()

    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit("Pillow required for Phase B visual cache") from e

    oracle_dir = Path(args.oracle_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((oracle_dir / "config.json").read_text(encoding="utf-8"))
    depth = np.load(oracle_dir / "depth_maps.npz")
    episode_ids = depth["episode_ids"].astype(np.int64)
    starts = depth["starts"].astype(np.int64)
    task_ids = depth["task_ids"].astype(np.int64)
    n_total = len(episode_ids)
    shard_end = n_total if args.shard_end < 0 else min(args.shard_end, n_total)
    shard_start = max(0, args.shard_start)
    shard_name = f"vis_{shard_start:05d}_{shard_end:05d}"
    out_path = out_dir / f"{shard_name}.npz"
    ckpt_path = out_dir / f"{shard_name}.checkpoint.json"

    done_until = shard_start
    if args.resume and ckpt_path.exists() and out_path.exists():
        done_until = int(json.loads(ckpt_path.read_text())["done_until"])
        print(f"[phaseb] resume from {done_until}")

    if done_until >= shard_end:
        print(f"[phaseb] complete {out_path}")
        return

    # accumulate lists then save periodically
    payload = {
        "chunk_idx": [],
        "episode_id": [],
        "task_id": [],
        "start": [],
        "image_jpeg": [],
        "wrist_jpeg": [],
    }
    if args.resume and out_path.exists() and done_until > shard_start:
        old = np.load(out_path, allow_pickle=True)
        for k in payload:
            payload[k] = old[k].tolist()

    dataset_id = cfg.get("dataset", LIBERO_DATASET_ID)
    revision = cfg.get("revision", LIBERO_REVISION)
    cache: dict[int, object] = {}

    for idx in tqdm(range(done_until, shard_end), desc=shard_name):
        eid = int(episode_ids[idx])
        start = int(starts[idx])
        if eid not in cache:
            path = _episode_parquet_path(dataset_id, revision, eid)
            cache[eid] = pq.read_table(path, columns=["image", "wrist_image"])
        table = cache[eid]
        img = _to_uint8_image(table.column("image")[start].as_py())
        wrist = _to_uint8_image(table.column("wrist_image")[start].as_py())

        def encode(arr: np.ndarray) -> bytes:
            bio = io.BytesIO()
            Image.fromarray(arr).save(bio, format="JPEG", quality=args.jpeg_quality)
            return bio.getvalue()

        payload["chunk_idx"].append(idx)
        payload["episode_id"].append(eid)
        payload["task_id"].append(int(task_ids[idx]))
        payload["start"].append(start)
        payload["image_jpeg"].append(encode(img))
        payload["wrist_jpeg"].append(encode(wrist))
        done_until = idx + 1
        if (done_until - shard_start) % 64 == 0 or done_until == shard_end:
            np.savez_compressed(
                out_path,
                chunk_idx=np.asarray(payload["chunk_idx"], dtype=np.int64),
                episode_id=np.asarray(payload["episode_id"], dtype=np.int64),
                task_id=np.asarray(payload["task_id"], dtype=np.int64),
                start=np.asarray(payload["start"], dtype=np.int64),
                image_jpeg=np.asarray(payload["image_jpeg"], dtype=object),
                wrist_jpeg=np.asarray(payload["wrist_jpeg"], dtype=object),
            )
            ckpt_path.write_text(json.dumps({"done_until": done_until}, indent=2), encoding="utf-8")

    if ckpt_path.exists():
        ckpt_path.unlink()
    meta = {
        "phase": "B_cache",
        "shard_start": shard_start,
        "shard_end": shard_end,
        "n": shard_end - shard_start,
        "fields": ["image_jpeg", "wrist_jpeg"],
        "note": "Causal first-frame visuals only; no future frames, no utility labels.",
    }
    (out_dir / f"{shard_name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
