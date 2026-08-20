"""LIBERO v2.0 loading and action preprocessing."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

LIBERO_DATASET_ID = "physical-intelligence/libero"
LIBERO_REVISION = "v2.0"

# Mirrors third_party/actioncodec/scripts/utils.py.
MAX_ACTION_Q = np.array(
    [
        0.9375,
        0.9107142686843872,
        0.9375,
        0.20357142388820648,
        0.26357144117355347,
        0.375,
        1.0,
    ],
    dtype=np.float32,
)


@dataclass
class ChunkBatch:
    actions: torch.Tensor
    raw_actions: torch.Tensor
    episode_ids: np.ndarray
    task_ids: np.ndarray
    task_names: list[str]
    starts: np.ndarray
    gripper_mode: str
    dataset_id: str = LIBERO_DATASET_ID
    revision: str = LIBERO_REVISION


def _load_tasks_map(dataset_id: str, revision: str) -> dict[int, str]:
    path = hf_hub_download(
        repo_id=dataset_id,
        filename="meta/tasks.jsonl",
        repo_type="dataset",
        revision=revision,
    )
    tasks: dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            tasks[int(row["task_index"])] = row["task"]
    return tasks


def normalize_actions(raw_actions: np.ndarray, gripper_mode: str = "invert") -> np.ndarray:
    """Normalize LIBERO actions for ActionCodec."""

    actions = raw_actions.astype(np.float32, copy=True)
    actions[..., :-1] = actions[..., :-1] / MAX_ACTION_Q[:-1]
    actions[..., :-1] = np.clip(actions[..., :-1], -1.0, 1.0)
    if gripper_mode == "invert":
        actions[..., -1] = (1.0 - raw_actions[..., -1]) / 2.0
    elif gripper_mode == "negate":
        actions[..., -1] = -raw_actions[..., -1]
    else:
        raise ValueError(f"Unknown gripper_mode: {gripper_mode}")
    actions[..., -1] = np.clip(actions[..., -1], 0.0, 1.0)
    return actions


def detect_gripper_convention(model, raw_actions: np.ndarray, device: str, embodiment_id: int = 0) -> str:
    """Pick the convention with the lower full-depth gripper floor."""

    from .codec import encode_actions, load_codec, projected_codebooks, decode_with_depth

    if raw_actions.ndim != 3 or raw_actions.shape[-1] != 7:
        raise ValueError(f"Expected [B, T, 7] raw actions, got {tuple(raw_actions.shape)}")

    sample = raw_actions[: min(len(raw_actions), 8)]
    scores: dict[str, float] = {}
    E = projected_codebooks(model, device=device)
    for mode in ("invert", "negate"):
        norm = torch.from_numpy(normalize_actions(sample, mode)).to(device)
        codes = encode_actions(model, norm, embodiment_id=embodiment_id)
        depth = torch.full(codes.shape[:2], codes.shape[2], device=device, dtype=torch.long)
        rec = decode_with_depth(model, E, codes, depth, embodiment_id=embodiment_id)
        pred = (rec[..., -1] > 0.5).float()
        true = (norm[..., -1] > 0.5).float()
        scores[mode] = float((pred != true).float().mean().item())
    return min(scores, key=scores.get)


def load_libero_chunks(
    n_chunks: int,
    chunk_len: int = 20,
    n_episodes: int | None = None,
    seed: int = 0,
    dataset_id: str = LIBERO_DATASET_ID,
    revision: str = LIBERO_REVISION,
    device: str = "cpu",
    gripper_mode: str | None = None,
    model=None,
    embodiment_id: int = 0,
) -> ChunkBatch:
    """Load a chunked subset directly from parquet files."""

    tasks_map = _load_tasks_map(dataset_id, revision)
    rng = np.random.default_rng(seed)
    total_eps = 1693
    n_episodes = n_episodes or min(max(24, n_chunks // 2), total_eps)
    episode_ids = rng.choice(total_eps, size=min(n_episodes, total_eps), replace=False)
    per_episode = int(np.ceil(n_chunks / len(episode_ids)))

    raw_actions: list[np.ndarray] = []
    chunk_episode_ids: list[int] = []
    task_ids: list[int] = []
    task_names: list[str] = []
    starts: list[int] = []

    for episode_id in episode_ids:
        path = hf_hub_download(
            repo_id=dataset_id,
            filename=f"data/chunk-{episode_id // 1000:03d}/episode_{episode_id:06d}.parquet",
            repo_type="dataset",
            revision=revision,
        )
        table = pq.read_table(path, columns=["actions", "task_index"])
        n_rows = table.num_rows
        if n_rows < chunk_len:
            continue
        valid_starts = np.arange(0, n_rows - chunk_len + 1)
        chosen = rng.choice(valid_starts, size=min(per_episode, len(valid_starts)), replace=False)
        actions_ep = np.asarray(table.column("actions").to_pylist(), dtype=np.float32)
        task_idx_ep = np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)
        for start in chosen:
            raw_actions.append(actions_ep[start : start + chunk_len])
            chunk_episode_ids.append(int(episode_id))
            task_ids.append(int(task_idx_ep[start]))
            task_names.append(tasks_map[int(task_idx_ep[start])])
            starts.append(int(start))
            if len(raw_actions) >= n_chunks:
                break
        if len(raw_actions) >= n_chunks:
            break

    if not raw_actions:
        raise RuntimeError("Failed to load any LIBERO chunks")

    raw = np.stack(raw_actions).astype(np.float32)
    if gripper_mode is None:
        if model is None:
            raise ValueError("model is required when gripper_mode is None")
        gripper_mode = detect_gripper_convention(model, raw, device=device, embodiment_id=embodiment_id)
    actions = normalize_actions(raw, gripper_mode=gripper_mode)
    return ChunkBatch(
        actions=torch.from_numpy(actions).to(device),
        raw_actions=torch.from_numpy(raw).to(device),
        episode_ids=np.asarray(chunk_episode_ids, dtype=np.int64),
        task_ids=np.asarray(task_ids, dtype=np.int64),
        task_names=task_names,
        starts=np.asarray(starts, dtype=np.int64),
        gripper_mode=gripper_mode,
    )
