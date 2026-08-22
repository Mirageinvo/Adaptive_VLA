#!/usr/bin/env python3
"""Stage 0: architectural integration gate for AATM (ICRA preregistered).

Measures where temporal merge can reduce wall-clock in the ActionCodec/VLA
pipeline. Must pass BEFORE latency claims in the paper.

Reference BAR numbers from k4a3 (V100, batch=1, integrated SmolVLA BAR):
  - VLM prefix: ~70 ms
  - Full 3-block generation: ~199 ms
  - End-to-end: ~273 ms
  - Action-position share in one decoder pass: ~5.2%
  - t(16)/t(8) ≈ 1.033x

Run:
  python experiments/merge0_integration.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_merge.latent import decode_latent, full_depth_latent
from adaptive_merge.merge_ops import expand_merged_latent
from adaptive_merge.segments import fixed_pair_segments, identity_segments
from adaptive_rvq.codec import encode_actions, load_codec, projected_codebooks
from adaptive_rvq.data import LIBERO_DATASET_ID, LIBERO_REVISION, load_libero_chunks

# Preregistered BAR reference (k4a3, V100 batch=1). Used for ceiling analysis only.
BAR_REF = {
    "source": "experiments/k4a3_latency_probe.py + FINDINGS.md §7b",
    "device": "Tesla V100-SXM2-32GB",
    "batch": 1,
    "vlm_prefix_ms": 69.9,
    "full_generation_3blocks_ms": 198.9,
    "end_to_end_ms": 273.3,
    "linear_fit": {"intercept_ms": 62.39, "slope_ms_per_position": 0.2149},
    "action_position_share_at_q16": 0.052,
    "speedup_16_to_8_single_pass": 1.0333,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--dataset", default=LIBERO_DATASET_ID)
    ap.add_argument("--revision", default=LIBERO_REVISION)
    ap.add_argument("--n-chunks", type=int, default=64)
    ap.add_argument("--n-episodes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="artifacts/merge/stage0_integration.json")
    return ap.parse_args()


def timeit_ms(fn, warmup: int, iters: int, device: str) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
    }


def amdahl_speedup(fraction: float, local_speedup: float) -> float:
    """End-to-end speedup if `fraction` of pipeline speeds up by `local_speedup`."""

    if local_speedup <= 1.0:
        return 1.0
    return 1.0 / ((1.0 - fraction) + fraction / local_speedup)


def integration_paths(bar_ref: dict, codec_decode_ms: float, merger_ms: float) -> list[dict]:
    """Preregistered integration paths and theoretical ceilings."""

    e2e = float(bar_ref["end_to_end_ms"])
    gen = float(bar_ref["full_generation_3blocks_ms"])
    prefix = float(bar_ref["vlm_prefix_ms"])
    slope = float(bar_ref["linear_fit"]["slope_ms_per_position"])
    pos_share = float(bar_ref["action_position_share_at_q16"])

    codec_frac = codec_decode_ms / max(e2e + codec_decode_ms, 1e-9)

    paths = [
        {
            "id": "A_merge_after_vla_before_decode",
            "description": "VLA generates full tokens; merge latent; ActionCodec decode",
            "vla_retrain_required": False,
            "vla_compute_change": "0% (full token_budget unchanged)",
            "codec_decode_change": "~0% (latent shape stays [B,16,D])",
            "merger_overhead_ms": merger_ms,
            "max_e2e_speedup_vs_baseline": amdahl_speedup(codec_frac, 1.0),
            "icra_latency_claim_viable": False,
            "notes": "Valid for reconstruction oracle only; does not reduce VLA cost.",
        },
        {
            "id": "B_sparse_positions_in_bar_pass",
            "description": "Fewer active action positions inside one BAR decoder pass (bos_len)",
            "vla_retrain_required": False,
            "measured_single_pass_speedup_16_to_8": float(bar_ref["speedup_16_to_8_single_pass"]),
            "max_e2e_speedup_if_generation_scales_like_single_pass": amdahl_speedup(
                gen / e2e, float(bar_ref["speedup_16_to_8_single_pass"])
            ),
            "action_position_share": pos_share,
            "icra_latency_claim_viable": pos_share >= 0.10,
            "notes": "k4a3: ~3% single-pass gain; integrated BAR NO-GO for sparse pass.",
        },
        {
            "id": "C_retrain_fewer_temporal_tokens",
            "description": "Retrain BAR with K<P temporal positions (merge-before-generation)",
            "vla_retrain_required": True,
            "blocks_change": "token_budget and block layout must be redesigned",
            "theoretical_pass_savings_ms_16_to_8": 8 * slope,
            "icra_latency_claim_viable": "conditional_on_retrain_and_success",
            "notes": "Primary path to VLA compute reduction if oracle shows headroom.",
        },
        {
            "id": "D_adaptive_nfe",
            "description": "Early exit after block 0/1 when confidence high (fewer decoder passes)",
            "vla_retrain_required": True,
            "per_block_pass_ms_approx": gen / 3.0,
            "max_e2e_speedup_3_to_2_blocks": amdahl_speedup(gen / e2e, 3.0 / 2.0),
            "icra_latency_claim_viable": "orthogonal_to_temporal_merge",
            "notes": "Complementary axis; not the same as token merging.",
        },
        {
            "id": "E_action_only_refiner",
            "description": "Cached VLM prefix + lightweight action refiner with variable K",
            "vla_retrain_required": True,
            "refiner_fraction_needed_for_1_15x_at_K8": (1.0 - 1.0 / 1.15) / (1.0 - (16 + 7 * 8) / (8 * 16)),
            "icra_latency_claim_viable": "unverified_open_question",
            "notes": "Token sparsity may matter here; requires dedicated latency probe.",
        },
    ]
    return paths


def stage0_gate(paths: list[dict], bar_ref: dict) -> dict:
    """Preregistered Stage 0 decision."""

    pos_share = float(bar_ref["action_position_share_at_q16"])
    sparse_bar_viable = pos_share >= 0.10
    merge_after_viable = any(p["id"] == "A_merge_after_vla_before_decode" and p.get("icra_latency_claim_viable") for p in paths)
    retrain_path_open = True  # always open as research direction

    if merge_after_viable:
        latency_story = "GO"
    elif retrain_path_open and not sparse_bar_viable:
        latency_story = "CONDITIONAL"
    else:
        latency_story = "NO-GO"

    return {
        "decision": latency_story,
        "sparse_bar_in_integrated_architecture": "NO-GO" if not sparse_bar_viable else "OPEN",
        "merge_after_decode_for_speedup": "NO-GO",
        "oracle_redundancy_experiments": "GO",
        "paper_latency_claims_require": [
            "Path C and/or E measured on target hardware (batch=1)",
            "End-to-end LIBERO latency, not token-count proxy",
            "Merger overhead reported separately",
        ],
        "position_share_threshold": 0.10,
        "observed_position_share": pos_share,
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_codec(model_id=args.model, device=args.device)
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
    latent = full_depth_latent(E, codes)
    pair_segments = fixed_pair_segments(model.n_tokens_per_quantizer)

    def decode_baseline():
        decode_latent(model, latent[:1], embodiment_id=args.embodiment)

    def decode_merged():
        merged = expand_merged_latent(latent[:1], pair_segments)
        decode_latent(model, merged, embodiment_id=args.embodiment)

    def merger_only():
        expand_merged_latent(latent[:1], pair_segments)

    def encode_batch():
        encode_actions(model, batch.actions[:1], embodiment_id=args.embodiment)

    timings = {
        "codec_encode": timeit_ms(encode_batch, args.warmup, args.iters, args.device),
        "codec_decode_baseline": timeit_ms(decode_baseline, args.warmup, args.iters, args.device),
        "codec_decode_after_pair_merge": timeit_ms(decode_merged, args.warmup, args.iters, args.device),
        "merger_expand_pair_mean": timeit_ms(merger_only, args.warmup, args.iters, args.device),
    }

    decode_ms = timings["codec_decode_baseline"]["median_ms"]
    merger_ms = timings["merger_expand_pair_mean"]["median_ms"]
    paths = integration_paths(BAR_REF, decode_ms, merger_ms)
    gate = stage0_gate(paths, BAR_REF)

    payload = {
        "experiment": "merge0_integration",
        "purpose": "ICRA Stage-0 architectural gate (preregistered)",
        "model": args.model,
        "dataset": args.dataset,
        "device": args.device,
        "n_positions": int(model.n_tokens_per_quantizer),
        "n_chunks_profiled": int(batch.actions.shape[0]),
        "bar_reference": BAR_REF,
        "timings_batch1": timings,
        "codec_decode_fraction_of_bar_e2e": decode_ms / (BAR_REF["end_to_end_ms"] + decode_ms),
        "integration_paths": paths,
        "stage0_gate": gate,
        "sanity": {
            "identity_segments_count": len(identity_segments(16)),
            "pair_segments_count": len(pair_segments),
            "decode_baseline_vs_merged_ratio": timings["codec_decode_baseline"]["median_ms"]
            / max(timings["codec_decode_after_pair_merge"]["median_ms"], 1e-9),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"stage0_gate": gate, "timings": timings}, indent=2))


if __name__ == "__main__":
    main()
