"""Metrics and bootstrap helpers for APB-RVQ experiments."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    """Compute continuous-channel and gripper metrics."""

    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {tuple(original.shape)} vs {tuple(reconstructed.shape)}")
    diff = reconstructed - original
    cont = diff[..., :6]
    gripper_pred = reconstructed[..., -1] > 0.5
    gripper_true = original[..., -1] > 0.5
    mse = float(cont.pow(2).mean().item())
    rms = float(np.sqrt(mse))
    by_channel = cont.pow(2).mean(dim=(0, 1)).sqrt().cpu().numpy()
    return {
        "mse": mse,
        "rms": rms,
        "gripper_error": float((gripper_pred != gripper_true).float().mean().item()),
        "ch0_rms": float(by_channel[0]),
        "ch1_rms": float(by_channel[1]),
        "ch2_rms": float(by_channel[2]),
        "ch3_rms": float(by_channel[3]),
        "ch4_rms": float(by_channel[4]),
        "ch5_rms": float(by_channel[5]),
    }


def gap_closed_fraction(err_depth1: float, err_budget: float, err_depth3: float) -> float:
    denom = err_depth1 - err_depth3
    if denom <= 1e-12:
        return float("nan")
    return float((err_depth1 - err_budget) / denom)


def summarize_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    out = {}
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
        out[f"{key}_median"] = float(np.median(vals))
    return out


def mean_random_per_chunk(random_values: np.ndarray, n_chunks: int, n_seeds: int) -> np.ndarray:
    """Average multi-seed random RMS into one value per chunk.

    Random values must be stored in chunk-major, seed-minor order:
    ``[c0s0, c0s1, ..., c0sN, c1s0, ...]``.
    """

    values = np.asarray(random_values, dtype=np.float64)
    expected = n_chunks * n_seeds
    if values.size != expected:
        raise ValueError(f"Expected {expected} random values, got {values.size}")
    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive")
    return values.reshape(n_chunks, n_seeds).mean(axis=1)


def bootstrap_ci(
    values: np.ndarray,
    episode_ids: np.ndarray,
    n_boot: int = 400,
    seed: int = 0,
    q_low: float = 2.5,
    q_high: float = 97.5,
) -> tuple[float, float]:
    """Cluster bootstrap by episode id."""

    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    episode_ids = np.asarray(episode_ids)
    if values.shape[0] != episode_ids.shape[0]:
        raise ValueError(
            f"values/episode_ids length mismatch: {values.shape[0]} vs {episode_ids.shape[0]}"
        )
    clusters = np.unique(episode_ids)
    by_cluster = defaultdict(list)
    for idx, episode_id in enumerate(episode_ids):
        by_cluster[int(episode_id)].append(idx)
    boot = []
    for _ in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        picks = np.concatenate([np.asarray(by_cluster[int(ep)], dtype=np.int64) for ep in chosen])
        boot.append(float(np.mean(values[picks])))
    lo, hi = np.percentile(np.asarray(boot), [q_low, q_high])
    return float(lo), float(hi)
