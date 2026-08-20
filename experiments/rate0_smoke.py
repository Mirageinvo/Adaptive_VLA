#!/usr/bin/env python3
"""Rate-0 smoke checks for APB-RVQ oracle pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_rvq.codec import (
    assert_full_depth_matches_native,
    encode_actions,
    load_codec,
    projected_codebooks,
)
from adaptive_rvq.data import LIBERO_DATASET_ID, LIBERO_REVISION, load_libero_chunks
from adaptive_rvq.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--dataset", default=LIBERO_DATASET_ID)
    ap.add_argument("--revision", default=LIBERO_REVISION)
    ap.add_argument("--n-chunks", type=int, default=32)
    ap.add_argument("--n-episodes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="artifacts/apb_rvq/rate0_smoke.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_codec(model_id=args.model, device=args.device)
    assert model.vocab_size == 2048, model.vocab_size
    assert model.num_quantizers == 3, model.num_quantizers
    assert model.n_tokens_per_quantizer == 16, model.n_tokens_per_quantizer

    tasks_path = hf_hub_download(
        repo_id=args.dataset,
        filename="meta/tasks.jsonl",
        repo_type="dataset",
        revision=args.revision,
    )

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

    codes = encode_actions(model, batch.actions, embodiment_id=args.embodiment)
    E = projected_codebooks(model, device=args.device)
    assert_full_depth_matches_native(model, E, codes, embodiment_id=args.embodiment)

    depth_full = torch.full(codes.shape[:2], model.num_quantizers, device=args.device, dtype=torch.long)
    from adaptive_rvq.codec import decode_with_depth

    rec = decode_with_depth(model, E, codes, depth_full, embodiment_id=args.embodiment)
    metrics = compute_metrics(batch.actions, rec)

    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "revision": args.revision,
        "tasks_path": tasks_path,
        "device": args.device,
        "gripper_mode": batch.gripper_mode,
        "n_chunks": int(batch.actions.shape[0]),
        "action_shape": list(batch.actions.shape),
        "codes_shape": list(codes.shape),
        "metrics": metrics,
        "checks": {
            "vocab_size": model.vocab_size,
            "rvq_levels": model.num_quantizers,
            "positions": model.n_tokens_per_quantizer,
            "level_major_layout": True,
            "full_depth_matches_native": True,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("rate0 smoke passed")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
