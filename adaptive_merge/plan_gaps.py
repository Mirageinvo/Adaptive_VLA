"""Post-hoc plan-gap helpers: span 2–4 validity and scheme C candidates.

Does not change the live merge1 search. Reads saved oracle ``segment_lengths``.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def parse_oracle_lengths(raw: str) -> tuple[int, ...] | None:
    """Parse merge1 oracle length strings. Returns None for non-oracle encodings."""

    text = str(raw).strip()
    if not text or text.startswith("seed=") or text.startswith("kept="):
        return None
    if "x" in text:
        return None
    try:
        parts = tuple(int(x) for x in text.split(",") if x)
    except ValueError:
        return None
    if not parts or any(p <= 0 for p in parts):
        return None
    return parts


def span_is_plan_legal(lengths: tuple[int, ...], max_span: int = 4) -> bool:
    """Plan §1: leave a position separate (1) or merge a block of 2–4."""

    return all(1 <= length <= max_span for length in lengths)


def filter_missing_episodes(
    episode_by_chunk: dict[int, int],
    chunk_indices: list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Keep chunk indices that have a real episode id.

    Missing keys are excluded. Never fabricates ``episode_id=-1`` (that would
    become a fake bootstrap cluster).
    """

    chunk_indices = [int(i) for i in np.asarray(chunk_indices).tolist()]
    kept: list[int] = []
    episode_ids: list[int] = []
    excluded: list[int] = []
    for cid in chunk_indices:
        if cid not in episode_by_chunk:
            excluded.append(cid)
            continue
        kept.append(cid)
        episode_ids.append(int(episode_by_chunk[cid]))
    return (
        np.asarray(kept, dtype=np.int64),
        np.asarray(episode_ids, dtype=np.int64),
        excluded,
    )


def episode_grouped_split(
    episode_ids: np.ndarray,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Episode-disjoint train/val split. Thresholds must be chosen on train only."""

    episode_ids = np.asarray(episode_ids)
    unique = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    n_val = max(1, int(round(len(unique) * val_frac))) if len(unique) > 1 else 0
    val_eps = set(perm[:n_val].tolist())
    train_eps = set(perm[n_val:].tolist())
    train_idx = np.nonzero(np.isin(episode_ids, list(train_eps)))[0]
    val_idx = np.nonzero(np.isin(episode_ids, list(val_eps)))[0]
    return (
        train_idx,
        val_idx,
        {
            "n_train_episodes": len(train_eps),
            "n_val_episodes": len(val_eps),
            "n_train_chunks": int(train_idx.size),
            "n_val_chunks": int(val_idx.size),
            "overlap_episodes": sorted(train_eps & val_eps),
        },
    )


def summarize_oracle_spans(
    chunk_ids: list[int],
    length_by_chunk: dict[int, tuple[int, ...]],
    rms_by_chunk: dict[int, float],
    max_span: int = 4,
) -> dict[str, object]:
    """Span statistics for one budget from saved unrestricted-oracle winners."""

    spans: list[int] = []
    max_spans: list[int] = []
    legal_flags: list[bool] = []
    legal_rms: list[float] = []
    illegal_rms: list[float] = []
    for cid in chunk_ids:
        lengths = length_by_chunk[cid]
        spans.extend(lengths)
        mx = max(lengths)
        max_spans.append(mx)
        legal = span_is_plan_legal(lengths, max_span=max_span)
        legal_flags.append(legal)
        if legal:
            legal_rms.append(rms_by_chunk[cid])
        else:
            illegal_rms.append(rms_by_chunk[cid])

    span_counts = Counter(spans)
    return {
        "n_chunks": len(chunk_ids),
        "mean_segment_length": float(np.mean(spans)) if spans else float("nan"),
        "median_segment_length": float(np.median(spans)) if spans else float("nan"),
        "mean_max_span": float(np.mean(max_spans)) if max_spans else float("nan"),
        "pct_chunks_plan_legal_span": float(np.mean(legal_flags)) if legal_flags else float("nan"),
        "n_chunks_illegal_span": int(sum(not x for x in legal_flags)),
        "span_histogram": {str(k): int(v) for k, v in sorted(span_counts.items())},
        "max_span_histogram": {str(k): int(v) for k, v in sorted(Counter(max_spans).items())},
        "rms_mean_legal_span_subset": float(np.mean(legal_rms)) if legal_rms else None,
        "rms_mean_illegal_span_subset": float(np.mean(illegal_rms)) if illegal_rms else None,
        "note": (
            "If pct_chunks_plan_legal_span is high, unrestricted oracle already "
            "matches plan §1 (max span 2–4). Illegal winners are an optimistic gap."
        ),
    }


def unique_oracle_schemes(
    chunk_ids: list[int],
    length_by_chunk: dict[int, tuple[int, ...]],
    max_span: int | None = None,
) -> list[tuple[int, ...]]:
    """Distinct oracle partitions, optionally filtered to plan-legal spans."""

    seen: set[tuple[int, ...]] = set()
    for cid in chunk_ids:
        lengths = length_by_chunk[cid]
        if max_span is not None and not span_is_plan_legal(lengths, max_span=max_span):
            continue
        seen.add(lengths)
    return sorted(seen, key=lambda t: (len(t), t))


def scheme_frequency(
    chunk_ids: list[int],
    length_by_chunk: dict[int, tuple[int, ...]],
) -> list[tuple[tuple[int, ...], int]]:
    counts: Counter[tuple[int, ...]] = Counter(length_by_chunk[cid] for cid in chunk_ids)
    return counts.most_common()


def lengths_to_position_ids(lengths: tuple[int, ...], n_positions: int = 16) -> np.ndarray:
    """Map each temporal position to its segment id."""

    ids = np.empty(n_positions, dtype=np.int64)
    pos = 0
    for seg_id, length in enumerate(lengths):
        ids[pos : pos + length] = seg_id
        pos += length
    if pos != n_positions:
        raise ValueError(f"lengths {lengths} cover {pos} positions, expected {n_positions}")
    return ids


def adjacent_merge_mask(lengths: tuple[int, ...], n_positions: int = 16) -> np.ndarray:
    """True at boundary i if positions i and i+1 are in the same oracle segment."""

    ids = lengths_to_position_ids(lengths, n_positions=n_positions)
    return ids[:-1] == ids[1:]


def co_segment_frequency(
    chunk_ids: list[int],
    length_by_chunk: dict[int, tuple[int, ...]],
    n_positions: int = 16,
) -> dict[str, object]:
    """Plan §4 locality: adjacent merge rate and 16×16 same-segment heatmap."""

    heat = np.zeros((n_positions, n_positions), dtype=np.float64)
    adj = np.zeros(n_positions - 1, dtype=np.float64)
    for cid in chunk_ids:
        ids = lengths_to_position_ids(length_by_chunk[cid], n_positions=n_positions)
        adj += (ids[:-1] == ids[1:]).astype(np.float64)
        for i in range(n_positions):
            heat[i] += (ids == ids[i]).astype(np.float64)
    n = max(len(chunk_ids), 1)
    heat = heat / n
    adj = adj / n
    return {
        "n_chunks": len(chunk_ids),
        "adjacent_merge_frequency": adj.tolist(),
        "mean_adjacent_merge_frequency": float(np.mean(adj)),
        "co_segment_heatmap": heat.tolist(),
        "note": "heatmap[i,j] = P(i and j in the same oracle segment).",
    }


def rms_quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def heavy_chunk_report(
    chunk_ids: list[int],
    oracle_rms: dict[int, float],
    baseline_rms: dict[int, float],
    episode_by_chunk: dict[int, int],
    task_by_chunk: dict[int, int] | None = None,
    rel_threshold: float = 0.10,
    top_n: int = 20,
) -> dict[str, object]:
    """Chunks where oracle compression raises RMS by more than ``rel_threshold``."""

    rows: list[dict] = []
    for cid in chunk_ids:
        base = float(baseline_rms[cid])
        err = float(oracle_rms[cid])
        rel = (err - base) / max(base, 1e-12)
        if rel > rel_threshold:
            rec = {
                "chunk_idx": int(cid),
                "episode_id": int(episode_by_chunk[cid]),
                "rel_rms_increase": rel,
                "oracle_rms": err,
                "baseline_rms": base,
            }
            if task_by_chunk is not None:
                rec["task_id"] = int(task_by_chunk[cid])
            rows.append(rec)
    rows.sort(key=lambda r: r["rel_rms_increase"], reverse=True)
    return {
        "rel_threshold": rel_threshold,
        "n_heavy": len(rows),
        "fraction_heavy": float(len(rows) / max(len(chunk_ids), 1)),
        "top": rows[:top_n],
    }


def similarity_merge_bins(
    cosine: np.ndarray,
    merged: np.ndarray,
    edges: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.01),
) -> dict[str, object]:
    """Plan §5/§11C: oracle merge frequency vs adjacent cosine similarity."""

    cosine = np.asarray(cosine, dtype=np.float64).reshape(-1)
    merged = np.asarray(merged, dtype=np.float64).reshape(-1)
    if cosine.shape != merged.shape:
        raise ValueError("cosine and merged masks must match")
    bins: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (cosine >= lo) & (cosine < hi)
        n = int(mask.sum())
        bins.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n_pairs": n,
                "oracle_merge_frequency": float(np.mean(merged[mask])) if n else None,
            }
        )
    if cosine.size == 0 or np.std(cosine) < 1e-12 or np.std(merged) < 1e-12:
        pearson = float("nan")
        spearman = float("nan")
    else:
        pearson = float(np.corrcoef(cosine, merged)[0, 1])
        spearman = spearman_corr(cosine, merged)
    return {
        "pearson_cosine_vs_merge": pearson,
        "spearman_cosine_vs_merge": spearman,
        "bins": bins,
    }


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann–Whitney AUROC. ``labels`` True = oracle merged."""

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    neg_sorted = np.sort(neg)
    lt = np.searchsorted(neg_sorted, pos, side="left")
    le = np.searchsorted(neg_sorted, pos, side="right")
    return float(np.mean((lt + 0.5 * (le - lt)) / neg.size))


def heuristic_aurocs(
    merged: np.ndarray,
    cosine: np.ndarray,
    l2: np.ndarray | None = None,
    same_coarse: np.ndarray | None = None,
) -> dict[str, object]:
    """Plan §5: causal heuristics vs oracle merge labels. No VLA logits in this pipeline."""

    merged = np.asarray(merged).reshape(-1)
    cosine = np.asarray(cosine, dtype=np.float64).reshape(-1)
    out: dict[str, object] = {
        "cosine": auroc(cosine, merged),
        "one_minus_cosine": auroc(1.0 - cosine, merged),
        "entropy": None,
        "margin": None,
        "unavailable": ["entropy", "margin"],
        "note": "entropy/margin need VLA action-head logits; ActionCodec post-hoc has codes+latents only.",
    }
    if l2 is not None:
        out["neg_l2"] = auroc(-np.asarray(l2, dtype=np.float64).reshape(-1), merged)
    if same_coarse is not None:
        out["same_coarse_code"] = auroc(np.asarray(same_coarse, dtype=np.float64).reshape(-1), merged)
    return out


def segment_length_by_start(
    chunk_ids: list[int],
    length_by_chunk: dict[int, tuple[int, ...]],
    n_positions: int = 16,
) -> dict[str, object]:
    """Plan §4: mean oracle segment length among segments that start at position i."""

    sum_len = np.zeros(n_positions, dtype=np.float64)
    count = np.zeros(n_positions, dtype=np.float64)
    heat = np.zeros((n_positions, n_positions + 1), dtype=np.int64)
    for cid in chunk_ids:
        pos = 0
        for length in length_by_chunk[cid]:
            sum_len[pos] += length
            count[pos] += 1
            heat[pos, length] += 1
            pos += length
    mean_len = np.divide(sum_len, count, out=np.full_like(sum_len, np.nan), where=count > 0)
    return {
        "mean_length_if_segment_starts_here": mean_len.tolist(),
        "n_segments_starting_here": count.astype(int).tolist(),
        "start_pos_by_length_counts": heat.tolist(),
    }


def cluster_bootstrap_pvalue_oracle_lower(
    oracle: np.ndarray,
    baseline: np.ndarray,
    episode_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> float:
    """One-sided cluster bootstrap p-value for H1: mean(oracle) < mean(baseline).

    p = fraction of resamples with mean(oracle) >= mean(baseline).
    """

    from collections import defaultdict

    oracle = np.asarray(oracle, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    episode_ids = np.asarray(episode_ids)
    rng = np.random.default_rng(seed)
    clusters = np.unique(episode_ids)
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for idx, ep in enumerate(episode_ids.tolist()):
        by_cluster[int(ep)].append(idx)
    n_ge = 0
    for _ in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        picks = np.concatenate([np.asarray(by_cluster[int(ep)], dtype=np.int64) for ep in chosen])
        if float(np.mean(oracle[picks])) >= float(np.mean(baseline[picks])):
            n_ge += 1
    return float(n_ge / n_boot)


def attention_compute_relative(k: int, k_full: int = 16) -> dict[str, float]:
    """Cheap compute proxy (plan §9/§10) until V100 e2e exists. Not wall-clock."""

    ratio = k / float(k_full)
    return {
        "k": k,
        "self_attn_quadratic": float(ratio**2),
        "linear_mlp": float(ratio),
    }


def rel_increase_pct(method_rms: float, baseline_rms: float) -> float:
    return 100.0 * (method_rms - baseline_rms) / max(baseline_rms, 1e-12)


def retained_gain(err_method: float, err_oracle: float, err_random: float) -> float:
    """(random - method) / (random - oracle); user's (g-r)/(o-r) with lower-is-better errors."""

    denom = err_random - err_oracle
    if abs(denom) < 1e-12:
        return float("nan")
    return float((err_random - err_method) / denom)
