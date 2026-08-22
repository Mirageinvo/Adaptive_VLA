#!/usr/bin/env python3
"""Merge-0 smoke checks for AATM temporal merge pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_merge.latent import eval_merge_partition, full_depth_latent
from adaptive_merge.merge_ops import expand_merged_latent
from adaptive_merge.segments import fixed_pair_segments, identity_segments
from adaptive_rvq.codec import assert_full_depth_matches_native, encode_actions, load_codec, projected_codebooks
from adaptive_rvq.data import LIBERO_DATASET_ID, LIBERO_REVISION, load_libero_chunks


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--dataset", default=LIBERO_DATASET_ID)
    ap.add_argument("--revision", default=LIBERO_REVISION)
    ap.add_argument("--n-chunks", type=int, default=16)
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="artifacts/merge/rate0_smoke.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_codec(model_id=args.model, device=args.device)
    assert model.n_tokens_per_quantizer == 16, model.n_tokens_per_quantizer

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

    latent = full_depth_latent(E, codes)
    no_merge = eval_merge_partition(
        model, E, codes[0], batch.actions[0], identity_segments(16), embodiment_id=args.embodiment
    )
    pair_merge = eval_merge_partition(
        model, E, codes[0], batch.actions[0], fixed_pair_segments(16), embodiment_id=args.embodiment
    )

    expanded = expand_merged_latent(latent[:1], fixed_pair_segments(16))
    assert expanded.shape == latent[:1].shape

    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "revision": args.revision,
        "device": args.device,
        "gripper_mode": batch.gripper_mode,
        "n_chunks": int(batch.actions.shape[0]),
        "action_shape": list(batch.actions.shape),
        "codes_shape": list(codes.shape),
        "latent_shape": list(latent.shape),
        "sample_metrics": {
            "no_merge": no_merge,
            "fixed_pair_8_segments": pair_merge,
        },
        "checks": {
            "positions": model.n_tokens_per_quantizer,
            "full_depth_matches_native": True,
            "pair_merge_increases_or_equal_rms": pair_merge["rms"] >= no_merge["rms"] - 1e-6,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("merge0 smoke passed")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
