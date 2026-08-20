#!/usr/bin/env python3
"""Phase B: extract frozen BAR VLM pooled context features from visual cache.

Feature: obs_pooled_ctx = masked-mean of _build_vlm_inputs_embeds (dim 2048).
Causal: first-frame image+wrist + task text + normalized state; no action/fine tokens.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "actioncodec"))
sys.path.insert(0, os.path.join(REPO, "experiments"))

from k3_bar_suffix_repair import STATE_Q01, STATE_Q99, build_batch  # noqa: E402


def _decode_jpeg(blob) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(bytes(blob))).convert("RGB"), dtype=np.uint8)


def _load_visual_index(cache_dir: Path, shard_start: int, shard_end: int) -> dict[int, tuple[bytes, bytes]]:
    """Map chunk_idx -> (image_jpeg, wrist_jpeg) for chunks in [start, end)."""
    out: dict[int, tuple[bytes, bytes]] = {}
    for path in sorted(cache_dir.glob("vis_*.npz")):
        # Filename vis_00000_01024.npz → optional skip if no overlap
        stem = path.stem  # vis_00000_01024
        parts = stem.split("_")
        try:
            f0, f1 = int(parts[1]), int(parts[2])
            if f1 <= shard_start or f0 >= shard_end:
                continue
        except (IndexError, ValueError):
            pass
        d = np.load(path, allow_pickle=True)
        for i, cid in enumerate(d["chunk_idx"].tolist()):
            cid = int(cid)
            if shard_start <= cid < shard_end:
                out[cid] = (d["image_jpeg"][i], d["wrist_jpeg"][i])
    return out


def _task_and_state_from_labels(labels_path: Path) -> dict[int, tuple[str, np.ndarray]]:
    """One (task_name, state_first) per chunk_idx (budget-invariant)."""
    rows = pq.read_table(labels_path, columns=["chunk_idx", "task_name", "state_first"]).to_pylist()
    out: dict[int, tuple[str, np.ndarray]] = {}
    for r in rows:
        cid = int(r["chunk_idx"])
        if cid in out:
            continue
        out[cid] = (str(r["task_name"]), np.asarray(r["state_first"], dtype=np.float32))
    return out


def _norm_state(raw: np.ndarray) -> np.ndarray:
    return ((raw - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--visual-cache", default="artifacts/apb_rvq/visual_cache_phase_b")
    ap.add_argument("--labels", default="artifacts/apb_rvq/labels_phase_a/labels.parquet")
    ap.add_argument("--output", default="artifacts/apb_rvq/bar_features_phase_b")
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-end", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[barfeat] shard=[{args.shard_start},{args.shard_end}) loading label meta...", flush=True)
    meta = _task_and_state_from_labels(Path(args.labels))
    shard_end = (max(meta.keys()) + 1) if args.shard_end < 0 else args.shard_end
    print(f"[barfeat] loading visual cache for shard...", flush=True)
    vis = _load_visual_index(Path(args.visual_cache), args.shard_start, shard_end)
    all_ids = sorted(set(vis.keys()) & set(meta.keys()))
    if not all_ids:
        raise RuntimeError("No overlapping chunk_idx between visual cache and labels")

    ids = [c for c in all_ids if args.shard_start <= c < shard_end]
    if not ids:
        raise RuntimeError(f"Empty shard [{args.shard_start}, {shard_end})")
    print(f"[barfeat] n_ids={len(ids)}", flush=True)
    shard_name = f"feat_{args.shard_start:05d}_{shard_end:05d}"
    out_path = out_dir / f"{shard_name}.npz"
    ckpt_path = out_dir / f"{shard_name}.checkpoint.json"

    done = set()
    feats: dict[int, np.ndarray] = {}
    if args.resume and out_path.exists():
        old = np.load(out_path)
        for i, cid in enumerate(old["chunk_idx"].tolist()):
            feats[int(cid)] = old["obs_pooled_ctx"][i]
            done.add(int(cid))
        print(f"[barfeat] resume {len(done)} chunks from {out_path}")

    todo = [c for c in ids if c not in done]
    if not todo:
        print(f"[barfeat] shard complete: {out_path}")
        return

    import importlib.util

    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR

    root = os.path.join(REPO, "third_party", "actioncodec")
    sp = importlib.util.spec_from_file_location(
        "ac_vla_tok", os.path.join(root, "utils", "vla_tokenizer.py")
    )
    mm = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mm)

    print(f"[barfeat] loading BAR {args.ckpt}", flush=True)
    proc = mm.VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete"
    )
    model = (
        SmolVLABlockwiseAR.from_pretrained(
            args.ckpt,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            token_budget=48,
            num_blocks=3,
            action_vocab_size=2048,
        )
        .to(args.device)
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)

    ns = types.SimpleNamespace(center_crop=True, tiled=True, source="lerobot", flip="")

    def save():
        cids = np.asarray(sorted(feats.keys()), dtype=np.int64)
        arr = np.stack([feats[c] for c in cids.tolist()], axis=0).astype(np.float16)
        np.savez_compressed(out_path, chunk_idx=cids, obs_pooled_ctx=arr)
        ckpt_path.write_text(json.dumps({"n_done": len(feats), "shard_end": shard_end}, indent=2))

    for i0 in tqdm(range(0, len(todo), args.batch_size), desc=shard_name):
        batch_ids = todo[i0 : i0 + args.batch_size]
        im1_list, im2_list, tasks, states = [], [], [], []
        for cid in batch_ids:
            img_b, wrist_b = vis[cid]
            task, raw_st = meta[cid]
            img = _decode_jpeg(img_b)
            wrist = _decode_jpeg(wrist_b)
            # (C,H,W) uint8 tensors
            im1_list.append(torch.from_numpy(img).permute(2, 0, 1).contiguous())
            im2_list.append(torch.from_numpy(wrist).permute(2, 0, 1).contiguous())
            tasks.append(task)
            states.append(_norm_state(raw_st))
        im1 = torch.stack(im1_list, dim=0)
        im2 = torch.stack(im2_list, dim=0)
        st = np.stack(states, axis=0)

        # Silence per-batch print from build_batch
        import builtins

        _print = builtins.print
        builtins.print = lambda *a, **k: None
        try:
            batch = build_batch(im1, im2, tasks, st, proc, ns, args.device, pad_side="left")
        finally:
            builtins.print = _print

        with torch.no_grad():
            _, _, VLM, _ = model._build_vlm_inputs_embeds(
                input_ids=batch["input_ids"],
                inputs_embeds=None,
                pixel_values=batch.get("pixel_values"),
                pixel_attention_mask=batch.get("pixel_attention_mask"),
                image_hidden_states=None,
            )
            am = batch.get("attention_mask")
            if am is None:
                pooled = VLM.float().mean(1)
            else:
                w = am[:, : VLM.shape[1]].to(dtype=torch.float32).unsqueeze(-1)
                pooled = (VLM.float() * w).sum(1) / w.sum(1).clamp_min(1.0)
            pv = pooled.detach().cpu().numpy().astype(np.float16)

        for j, cid in enumerate(batch_ids):
            feats[cid] = pv[j]

        if (i0 // args.batch_size) % 8 == 0 or i0 + args.batch_size >= len(todo):
            save()

    save()
    if ckpt_path.exists():
        ckpt_path.unlink()
    meta_out = {
        "feature": "obs_pooled_ctx",
        "dim": int(next(iter(feats.values())).shape[0]),
        "n": len(feats),
        "shard_start": args.shard_start,
        "shard_end": shard_end,
        "ckpt": args.ckpt,
        "causal": True,
        "notes": "masked-mean VLM prefix embeds; no action/fine tokens",
    }
    (out_dir / f"{shard_name}.meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(json.dumps(meta_out, indent=2))


if __name__ == "__main__":
    main()
