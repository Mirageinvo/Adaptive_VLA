#!/usr/bin/env python3
"""Post-hoc plan analyses on a finished merge1 run. Does not touch the live job.

Covers plan gaps that merge1 did not compute:
  §3 C  state-independent global scheme (train-optimal among observed legal winners)
  §1    span 2–4 proxy vs unrestricted oracle
  §4    locality heatmap + segment-length histogram
  §5/11C cosine similarity vs oracle merge frequency
  §8    greedy vs full oracle (optional, extra decodes)
  §11 B uniform subsample at every budget
  quantiles / heavy chunks / comparison table

Not a full re-search: scheme C and span≤4 oracle use candidate schemes from saved
oracle winners, not all 16,369 partitions (that would be another multi-hour job).

Usage after medium:
  python experiments/merge1_posthoc_plan_gaps.py --oracle-dir artifacts/merge/oracle_medium --rows-only
  python experiments/merge1_posthoc_plan_gaps.py --oracle-dir artifacts/merge/oracle_medium --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_merge.latent import (
    adjacent_cosine_similarity,
    adjacent_l2_distance,
    eval_merge_partition,
    eval_subsample_partition,
    full_depth_latent,
)
from adaptive_merge.merge_ops import segments_from_lengths
from adaptive_merge.oracle import greedy_partition_for_budget
from adaptive_merge.plan_gaps import (
    adjacent_merge_mask,
    attention_compute_relative,
    cluster_bootstrap_pvalue_oracle_lower,
    co_segment_frequency,
    episode_grouped_split,
    filter_missing_episodes,
    heavy_chunk_report,
    heuristic_aurocs,
    parse_oracle_lengths,
    rel_increase_pct,
    retained_gain,
    rms_quantiles,
    scheme_frequency,
    segment_length_by_start,
    similarity_merge_bins,
    span_is_plan_legal,
    summarize_oracle_spans,
    unique_oracle_schemes,
)
from adaptive_merge.segments import fixed_pair_segments, uniform_subsample_indices
from adaptive_rvq.codec import encode_actions, load_codec, projected_codebooks
from adaptive_rvq.data import LIBERO_DATASET_ID, LIBERO_REVISION, load_libero_chunks
from adaptive_rvq.metrics import bootstrap_ci

N_POS = 16


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", default="artifacts/merge/oracle_medium")
    ap.add_argument("--model", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--dataset", default=LIBERO_DATASET_ID)
    ap.add_argument("--revision", default=LIBERO_REVISION)
    ap.add_argument("--n-chunks", type=int, default=256)
    ap.add_argument("--n-episodes", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-span", type=int, default=4)
    ap.add_argument("--budgets", default="16,14,12,10,8,6,4")
    ap.add_argument("--heavy-rel-threshold", type=float, default=0.10)
    ap.add_argument("--rows-only", action="store_true")
    ap.add_argument("--compute-greedy", action="store_true", help="Extra GPU: greedy merge vs oracle.")
    ap.add_argument("--greedy-budgets", default="12,8")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default=None)
    return ap.parse_args()


def load_eval_rows(oracle_dir: Path) -> list[dict]:
    merged = oracle_dir / "eval_rows.parquet"
    if merged.exists():
        return pq.read_table(merged).to_pylist()
    shards = sorted((oracle_dir / "shards").glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No eval_rows.parquet or shards in {oracle_dir}")
    rows: list[dict] = []
    for path in shards:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def method_rms(rows: list[dict], method: str, k: int | None = None) -> dict[int, float]:
    out: dict[int, float] = {}
    acc: dict[int, list[float]] = {}
    for row in rows:
        if row["method"] != method:
            continue
        if k is not None and int(row["n_segments"]) != k:
            continue
        cid = int(row["chunk_idx"])
        acc.setdefault(cid, []).append(float(row["rms"]))
    for cid, vals in acc.items():
        out[cid] = float(np.mean(vals))
    return out


def oracle_maps(rows: list[dict], k: int):
    chunk_ids: list[int] = []
    lengths: dict[int, tuple[int, ...]] = {}
    rms: dict[int, float] = {}
    episode: dict[int, int] = {}
    task: dict[int, int] = {}
    for row in rows:
        if row["method"] != "oracle_adaptive" or int(row["n_segments"]) != k:
            continue
        parsed = parse_oracle_lengths(row["segment_lengths"])
        if parsed is None:
            continue
        cid = int(row["chunk_idx"])
        chunk_ids.append(cid)
        lengths[cid] = parsed
        rms[cid] = float(row["rms"])
        episode[cid] = int(row["episode_id"])
        task[cid] = int(row["task_id"])
    return sorted(set(chunk_ids)), lengths, rms, episode, task


def cached_scheme_rms(rows: list[dict], k: int, lengths: tuple[int, ...]) -> dict[int, float]:
    want = ",".join(str(x) for x in lengths)
    pair_key = "2x8" if lengths == (2,) * 8 else None
    out: dict[int, float] = {}
    for row in rows:
        cid = int(row["chunk_idx"])
        if row["method"] == "oracle_adaptive" and int(row["n_segments"]) == k and row["segment_lengths"] == want:
            out[cid] = float(row["rms"])
        elif pair_key and row["method"] == "fixed_pair" and row["segment_lengths"] == pair_key:
            out[cid] = float(row["rms"])
    return out


def eval_scheme_all_chunks(model, E, codes, actions, lengths, cache, embodiment_id) -> np.ndarray:
    segments = segments_from_lengths(lengths)
    n = codes.shape[0]
    missing = [i for i in range(n) if i not in cache]
    for cid in tqdm(missing, desc=f"scheme {lengths}", leave=False):
        cache[cid] = eval_merge_partition(
            model, E, codes[cid], actions[cid], segments, embodiment_id=embodiment_id
        )["rms"]
    return np.asarray([cache[i] for i in range(n)], dtype=np.float64)


def aligned_array(mapping: dict[int, float], n: int) -> np.ndarray | None:
    if any(i not in mapping for i in range(n)):
        return None
    return np.asarray([mapping[i] for i in range(n)], dtype=np.float64)


def warn_excluded_episodes(excluded: list[int], context: str, warnings: list[str]) -> None:
    if not excluded:
        return
    preview = excluded[:32]
    msg = (
        f"{context}: excluded {len(excluded)} chunk indices missing from "
        f"episode_by_chunk (never bootstrap episode_id=-1): {preview}"
    )
    print(f"WARNING: {msg}", flush=True)
    warnings.append(msg)


def write_table_md(path: Path, table: list[dict], extra_lines: list[str] | None = None) -> None:
    keys = [
        "k",
        "oracle_full",
        "oracle_span_le4_proxy",
        "scheme_c",
        "fixed_pair",
        "uniform",
        "random",
        "greedy",
    ]
    lines = [
        "| k | Oracle (full) | Oracle (span≤4 proxy) | Global C | Fixed pair | Uniform | Random | Greedy |",
        "|---|--------------:|----------------------:|---------:|-----------:|--------:|-------:|-------:|",
    ]
    for row in table:
        cells = []
        for key in keys:
            val = row.get(key)
            if key == "k":
                cells.append(str(val))
            elif val is None:
                cells.append("—")
            else:
                cells.append(f"{val:.2f}%")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Values are relative RMS increase vs no-merge. Lower is better.")
    lines.append("Scheme C and span≤4 are post-hoc proxies over observed legal oracle winners, not a full re-enumeration.")
    lines.append("")
    lines.append("## Plan coverage")
    lines.extend(
        [
            "",
            "| Plan | Analysis | Status |",
            "|------|----------|--------|",
            "| §1 | max span 2–4 | post-hoc proxy |",
            "| §2 | oracle curves | merge1 |",
            "| §3A | no merge | merge1 |",
            "| §3B | fixed pair | merge1 (k=8) |",
            "| §3C | global train-optimal | post-hoc (observed winners) |",
            "| §3D | oracle adaptive | merge1 |",
            "| §3 | adaptive gain + bootstrap p | post-hoc; science uses *_val |",
            "| §4 | heatmap / span hist / start-pos length | post-hoc |",
            "| §4/§11C | similarity→merge bins + Spearman | post-hoc (codec reload) |",
            "| §5 | heuristic AUROC (cosine/L2/coarse) | post-hoc; entropy/margin N/A |",
            "| §7 | learned merger | not this stage |",
            "| §8 | greedy | `--compute-greedy` |",
            "| §9/§10 | V100 e2e | not this stage; quadratic compute estimate only |",
            "| §11B | uniform all k | post-hoc decode |",
            "| §12 | ablations | not this stage |",
            "| §13 | go/no-go | after this file |",
            "",
        ]
    )
    if extra_lines:
        lines.extend(extra_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    oracle_dir = Path(args.oracle_dir)
    budgets = sorted((int(x) for x in args.budgets.split(",")), reverse=True)
    greedy_budgets = {int(x) for x in args.greedy_budgets.split(",") if x}
    rows = load_eval_rows(oracle_dir)
    baseline = method_rms(rows, "no_merge")
    n_from_rows = max(baseline) + 1 if baseline else 0

    payload_warnings: list[str] = []
    payload: dict[str, object] = {
        "experiment": "merge1_posthoc_plan_gaps",
        "purpose": "Plan §1/§3C/§4/§5/§8/§11 post-hoc from merge1 eval_rows",
        "oracle_dir": str(oracle_dir),
        "max_span": args.max_span,
        "rows_only": args.rows_only,
        "compute_greedy": args.compute_greedy,
        "limitations": [
            "Scheme C searches unique train oracle winners (plus fixed-pair at k=8), not all partitions.",
            "span≤4 oracle is exact only when the unrestricted winner is already legal; otherwise min over observed legal schemes.",
            "Greedy was not saved by merge1; computed only if --compute-greedy.",
            "Uniform at k≠8 was not saved by merge1; computed if codec is loaded.",
            "Science-decision keys are *_val (held-out episodes). *_all is transparency only.",
            "Bootstrap never uses episode_id=-1; chunks missing from episode_by_chunk are excluded.",
        ],
        "warnings": payload_warnings,
        "by_budget": {},
        "similarity": None,
        "heuristic_aurocs": None,
        "compute_estimate": {str(k): attention_compute_relative(k) for k in (16, 14, 12, 10, 8, 6, 4)},
        "comparison_table": [],
    }

    model = E = codes = actions = latent = None
    if not args.rows_only:
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
        actions = batch.actions
        E = projected_codebooks(model, device=args.device)
        latent = full_depth_latent(E, codes)
        payload["n_chunks_reloaded"] = int(codes.shape[0])
        n_from_rows = int(codes.shape[0])

    cosine_np = None
    l2_np = None
    same_coarse_np = None
    if latent is not None:
        cosine_np = adjacent_cosine_similarity(latent).detach().cpu().numpy()
        l2_np = adjacent_l2_distance(latent).detach().cpu().numpy()
    if codes is not None:
        same_coarse_np = (codes[:, :-1, 0] == codes[:, 1:, 0]).detach().cpu().numpy().astype(np.float64)

    table_rows: list[dict] = []

    for k in budgets:
        chunk_ids, length_by_chunk, rms_by_chunk, episode_by_chunk, task_by_chunk = oracle_maps(rows, k)
        if not chunk_ids:
            payload["by_budget"][str(k)] = {"error": "no oracle rows"}
            continue

        n = n_from_rows or (max(chunk_ids) + 1)
        oracle_vals = aligned_array(rms_by_chunk, n)
        base_vals = aligned_array(baseline, n)
        random_vals = aligned_array(method_rms(rows, "random_partition", k), n)
        pair_vals = aligned_array(method_rms(rows, "fixed_pair", k), n) if k == 8 else None
        uniform_vals = aligned_array(method_rms(rows, "uniform_subsample", k), n)

        span_stats = summarize_oracle_spans(chunk_ids, length_by_chunk, rms_by_chunk, max_span=args.max_span)
        freq = scheme_frequency(chunk_ids, length_by_chunk)
        locality = co_segment_frequency(chunk_ids, length_by_chunk, n_positions=N_POS)
        entry: dict[str, object] = {
            "span_stats": span_stats,
            "locality": {
                "adjacent_merge_frequency": locality["adjacent_merge_frequency"],
                "mean_adjacent_merge_frequency": locality["mean_adjacent_merge_frequency"],
                "co_segment_heatmap": locality["co_segment_heatmap"],
                "segment_length_by_start": segment_length_by_start(chunk_ids, length_by_chunk, n_positions=N_POS),
            },
            "n_unique_oracle_schemes": len(freq),
            "top5_oracle_schemes": [
                {"lengths": list(lengths), "n_chunks": int(count)} for lengths, count in freq[:5]
            ],
            "oracle_rms_quantiles": rms_quantiles(np.asarray(list(rms_by_chunk.values()))),
        }
        if base_vals is not None and oracle_vals is not None:
            rel = (oracle_vals - base_vals) / np.maximum(base_vals, 1e-12)
            entry["oracle_rel_increase_quantiles"] = rms_quantiles(rel)
            if pair_vals is not None:
                entry["fixed_pair_rms_quantiles"] = rms_quantiles(pair_vals)
                entry["fixed_pair_rel_increase_quantiles"] = rms_quantiles(
                    (pair_vals - base_vals) / np.maximum(base_vals, 1e-12)
                )
            entry["heavy_chunks"] = heavy_chunk_report(
                chunk_ids,
                rms_by_chunk,
                baseline,
                episode_by_chunk,
                task_by_chunk,
                rel_threshold=args.heavy_rel_threshold,
            )

        merged_adj = np.stack([adjacent_merge_mask(length_by_chunk[cid], N_POS) for cid in range(n) if cid in length_by_chunk])
        if cosine_np is not None and merged_adj.shape[0] == cosine_np.shape[0]:
            sim = similarity_merge_bins(cosine_np, merged_adj.astype(np.float64))
            sim["heuristic_aurocs"] = heuristic_aurocs(
                merged_adj,
                cosine_np,
                l2=l2_np,
                same_coarse=same_coarse_np,
            )
            entry["similarity_vs_merge"] = sim
        elif cosine_np is None:
            entry["similarity_vs_merge"] = {
                "note": "requires codec reload (run without --rows-only)",
                "adjacent_merge_frequency_from_oracle": locality["adjacent_merge_frequency"],
            }

        scheme_c_mean = None
        span_proxy_mean = None
        greedy_mean = None
        uniform_mean = float(np.mean(uniform_vals)) if uniform_vals is not None else None
        pair_mean = float(np.mean(pair_vals)) if pair_vals is not None else None
        oracle_mean = float(np.mean(oracle_vals)) if oracle_vals is not None else None
        random_mean = float(np.mean(random_vals)) if random_vals is not None else None
        base_mean = float(np.mean(base_vals)) if base_vals is not None else None

        if not args.rows_only and model is not None and codes is not None:
            episode_arr = np.asarray([episode_by_chunk[cid] for cid in chunk_ids], dtype=np.int64)
            train_pos, val_pos, split_meta = episode_grouped_split(
                episode_arr, val_frac=args.val_frac, seed=args.seed
            )
            train_cids = [chunk_ids[i] for i in train_pos.tolist()]
            val_cids = [chunk_ids[i] for i in val_pos.tolist()]
            entry["split"] = split_meta

            candidates = unique_oracle_schemes(train_cids, length_by_chunk, max_span=args.max_span)
            if k == 8:
                pair = tuple(seg.length for seg in fixed_pair_segments(N_POS))
                if pair not in candidates:
                    candidates.append(pair)

            if candidates:
                scheme_warnings: list[str] = []
                candidate_rms: dict[tuple[int, ...], np.ndarray] = {}
                chosen_lengths = None
                best_train = float("inf")
                if not train_cids:
                    msg = "train empty; scheme C chosen on all chunks (selection fallback, not a val science decision)"
                    print(f"WARNING: {msg}", flush=True)
                    scheme_warnings.append(msg)
                for lengths in candidates:
                    cache = cached_scheme_rms(rows, k, lengths)
                    all_rms = eval_scheme_all_chunks(
                        model, E, codes, actions, lengths, cache, args.embodiment
                    )
                    candidate_rms[lengths] = all_rms
                    train_mean = float(np.mean(all_rms[train_cids])) if train_cids else float(np.mean(all_rms))
                    if train_mean < best_train:
                        best_train = train_mean
                        chosen_lengths = lengths

                assert chosen_lengths is not None
                scheme_c_vals = candidate_rms[chosen_lengths]
                if oracle_vals is None:
                    oracle_vals = np.asarray(
                        [
                            rms_by_chunk[i] if i in rms_by_chunk else np.nan
                            for i in range(len(scheme_c_vals))
                        ],
                        dtype=np.float64,
                    )
                proxy = np.empty(len(scheme_c_vals), dtype=np.float64)
                for cid in range(len(proxy)):
                    winner = length_by_chunk.get(cid)
                    if winner is not None and span_is_plan_legal(winner, max_span=args.max_span):
                        proxy[cid] = rms_by_chunk[cid]
                    else:
                        proxy[cid] = min(float(arr[cid]) for arr in candidate_rms.values())

                n_arr = len(scheme_c_vals)
                train_ok = [cid for cid in train_cids if cid < n_arr]
                val_in_range = [cid for cid in val_cids if cid < n_arr]
                all_idx = list(range(n_arr))
                all_kept, ep_all, all_excluded = filter_missing_episodes(episode_by_chunk, all_idx)
                warn_excluded_episodes(all_excluded, f"k={k} all-data bootstrap", scheme_warnings)
                val_kept, ep_val, val_excluded = filter_missing_episodes(episode_by_chunk, val_in_range)
                warn_excluded_episodes(val_excluded, f"k={k} val bootstrap", scheme_warnings)

                gain_all = scheme_c_vals - oracle_vals
                scheme_c_mean = float(np.mean(scheme_c_vals))
                span_proxy_mean = float(np.mean(proxy))
                oracle_all_mean = (
                    float(np.mean(oracle_vals[all_kept])) if all_kept.size else float(np.nanmean(oracle_vals))
                )
                scheme_c_all_aligned = (
                    float(np.mean(scheme_c_vals[all_kept])) if all_kept.size else scheme_c_mean
                )
                gain_lo_all = gain_hi_all = p_all = None
                if all_kept.size:
                    gain_lo_all, gain_hi_all = bootstrap_ci(
                        gain_all[all_kept], ep_all, seed=args.seed + k
                    )
                    p_all = cluster_bootstrap_pvalue_oracle_lower(
                        oracle_vals[all_kept], scheme_c_vals[all_kept], ep_all, seed=args.seed + 100 + k
                    )

                entry["scheme_c"] = {
                    "chosen_on_train": list(chosen_lengths),
                    "n_candidates": len(candidates),
                    "train_rms": float(np.mean(scheme_c_vals[train_ok])) if train_ok else best_train,
                    "all_rms": scheme_c_mean,
                    "oracle_unrestricted_all_rms": oracle_all_mean,
                    "adaptive_gain_vs_scheme_c_all": scheme_c_all_aligned - oracle_all_mean,
                    "adaptive_gain_bootstrap_ci_all": (
                        [gain_lo_all, gain_hi_all] if gain_lo_all is not None else None
                    ),
                    "oracle_span_le4_proxy_all_rms": span_proxy_mean,
                    "adaptive_gain_unrestricted_vs_span_proxy": span_proxy_mean - oracle_all_mean,
                    "bootstrap_p_oracle_better_than_scheme_c_all": p_all,
                    "n_chunks_bootstrap_all": int(all_kept.size),
                    "warnings": scheme_warnings,
                }

                if val_kept.size == 0:
                    msg = (
                        f"k={k}: val empty after episode alignment; science decision "
                        "fields are null (no silent fallback to all)"
                    )
                    print(f"WARNING: {msg}", flush=True)
                    scheme_warnings.append(msg)
                    entry["scheme_c"]["val_rms"] = None
                    entry["scheme_c"]["oracle_unrestricted_val_rms"] = None
                    entry["scheme_c"]["adaptive_gain_vs_scheme_c_val"] = None
                    entry["scheme_c"]["adaptive_gain_bootstrap_ci_val"] = None
                    entry["scheme_c"]["bootstrap_p_oracle_better_than_scheme_c_val"] = None
                    entry["scheme_c"]["n_chunks_bootstrap_val"] = 0
                else:
                    scheme_c_val = scheme_c_vals[val_kept]
                    oracle_val = oracle_vals[val_kept]
                    scheme_c_val_mean = float(np.mean(scheme_c_val))
                    oracle_val_mean = float(np.mean(oracle_val))
                    gain_lo_val, gain_hi_val = bootstrap_ci(
                        scheme_c_val - oracle_val, ep_val, seed=args.seed + k
                    )
                    entry["scheme_c"]["val_rms"] = scheme_c_val_mean
                    entry["scheme_c"]["oracle_unrestricted_val_rms"] = oracle_val_mean
                    entry["scheme_c"]["adaptive_gain_vs_scheme_c_val"] = scheme_c_val_mean - oracle_val_mean
                    entry["scheme_c"]["adaptive_gain_bootstrap_ci_val"] = [gain_lo_val, gain_hi_val]
                    entry["scheme_c"]["bootstrap_p_oracle_better_than_scheme_c_val"] = (
                        cluster_bootstrap_pvalue_oracle_lower(
                            oracle_val, scheme_c_val, ep_val, seed=args.seed + 100 + k
                        )
                    )
                    entry["scheme_c"]["n_chunks_bootstrap_val"] = int(val_kept.size)

                if pair_vals is not None:
                    best_fixed = np.minimum(pair_vals, scheme_c_vals)
                    entry["scheme_c"]["fixed_pair_all_rms"] = float(np.mean(pair_vals))
                    entry["scheme_c"]["best_fixed_all_rms"] = float(np.mean(best_fixed))
                    entry["scheme_c"]["adaptive_gain_vs_best_fixed_all"] = (
                        float(np.mean(best_fixed[all_kept]) - np.mean(oracle_vals[all_kept]))
                        if all_kept.size
                        else float(np.mean(best_fixed) - np.nanmean(oracle_vals))
                    )
                    entry["scheme_c"]["bootstrap_p_oracle_better_than_best_fixed_all"] = (
                        cluster_bootstrap_pvalue_oracle_lower(
                            oracle_vals[all_kept], best_fixed[all_kept], ep_all, seed=args.seed + k
                        )
                        if all_kept.size
                        else None
                    )
                    if val_kept.size == 0:
                        entry["scheme_c"]["best_fixed_val_rms"] = None
                        entry["scheme_c"]["adaptive_gain_vs_best_fixed_val"] = None
                        entry["scheme_c"]["bootstrap_p_oracle_better_than_best_fixed_val"] = None
                    else:
                        best_fixed_val = best_fixed[val_kept]
                        oracle_val = oracle_vals[val_kept]
                        entry["scheme_c"]["best_fixed_val_rms"] = float(np.mean(best_fixed_val))
                        entry["scheme_c"]["adaptive_gain_vs_best_fixed_val"] = float(
                            np.mean(best_fixed_val) - np.mean(oracle_val)
                        )
                        entry["scheme_c"]["bootstrap_p_oracle_better_than_best_fixed_val"] = (
                            cluster_bootstrap_pvalue_oracle_lower(
                                oracle_val, best_fixed_val, ep_val, seed=args.seed + k
                            )
                        )

                payload_warnings.extend(scheme_warnings)

            if uniform_vals is None:
                kept = uniform_subsample_indices(N_POS, k)
                uni = np.empty(n, dtype=np.float64)
                for cid in tqdm(range(n), desc=f"uniform k={k}", leave=False):
                    uni[cid] = eval_subsample_partition(
                        model, E, codes[cid], actions[cid], kept, embodiment_id=args.embodiment
                    )["rms"]
                uniform_vals = uni
                uniform_mean = float(np.mean(uni))
                entry["uniform_subsample"] = {"kept": kept, "rms_mean": uniform_mean}

            if args.compute_greedy and k in greedy_budgets:
                greedy_rms = np.empty(n, dtype=np.float64)
                for cid in tqdm(range(n), desc=f"greedy k={k}"):
                    greedy_rms[cid] = greedy_partition_for_budget(
                        model,
                        E,
                        codes[cid],
                        actions[cid],
                        n_segments=k,
                        embodiment_id=args.embodiment,
                        max_span=args.max_span,
                    ).metrics["rms"]
                greedy_mean = float(np.mean(greedy_rms))
                entry["greedy"] = {
                    "rms_mean": greedy_mean,
                    "max_span": args.max_span,
                    "retained_gain_vs_random": (
                        retained_gain(greedy_mean, float(np.mean(oracle_vals)), random_mean)
                        if oracle_vals is not None and random_mean is not None
                        else None
                    ),
                }

        table_rows.append(
            {
                "k": k,
                "oracle_full": rel_increase_pct(oracle_mean, base_mean) if oracle_mean is not None and base_mean else None,
                "oracle_span_le4_proxy": rel_increase_pct(span_proxy_mean, base_mean)
                if span_proxy_mean is not None and base_mean
                else None,
                "scheme_c": rel_increase_pct(scheme_c_mean, base_mean) if scheme_c_mean is not None and base_mean else None,
                "fixed_pair": rel_increase_pct(pair_mean, base_mean) if pair_mean is not None and base_mean else None,
                "uniform": rel_increase_pct(uniform_mean, base_mean) if uniform_mean is not None and base_mean else None,
                "random": rel_increase_pct(random_mean, base_mean) if random_mean is not None and base_mean else None,
                "greedy": rel_increase_pct(greedy_mean, base_mean) if greedy_mean is not None and base_mean else None,
            }
        )
        payload["by_budget"][str(k)] = entry

    if cosine_np is not None:
        # Aggregate similarity vs merge at k=8 if present, else first budget.
        k_sim = 8 if 8 in budgets else budgets[0]
        _, length_k, _, _, _ = oracle_maps(rows, k_sim)
        if length_k:
            merged = np.stack(
                [adjacent_merge_mask(length_k[cid], N_POS) for cid in range(cosine_np.shape[0]) if cid in length_k]
            )
            if merged.shape[0] == cosine_np.shape[0]:
                payload["similarity"] = similarity_merge_bins(cosine_np, merged.astype(np.float64))
                payload["similarity"]["budget"] = k_sim
                payload["heuristic_aurocs"] = heuristic_aurocs(
                    merged, cosine_np, l2=l2_np, same_coarse=same_coarse_np
                )
                payload["heuristic_aurocs"]["budget"] = k_sim

    payload["comparison_table"] = table_rows
    out_path = Path(args.output) if args.output else oracle_dir / "plan_gaps.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_table_md(out_path.with_suffix(".md"), table_rows)
    print(json.dumps({"comparison_table": table_rows, "output": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
