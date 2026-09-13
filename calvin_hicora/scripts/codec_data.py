"""Indexed CALVIN action chunks for ActionCodec training and evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset


Split = Literal["train", "val", "dev", "all"]
Normalization = Literal["native", "quantile"]
SUPPORTED_FORMAT_VERSIONS = {2}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_converted_dataset(root: Path, verify_hashes: bool = False) -> dict[str, Any]:
    required = (
        "actions.npy",
        "chunk_index.npy",
        "episode_index.json",
        "norm_stats.json",
        "data_manifest.json",
    )
    if (root / ".conversion_incomplete").exists():
        raise RuntimeError(f"{root} contains an incomplete conversion")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing converted dataset files: {missing}")
    manifest = load_json(root / "data_manifest.json")
    if int(manifest.get("format_version", -1)) not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported converted-dataset format_version={manifest.get('format_version')!r}; "
            f"expected one of {sorted(SUPPORTED_FORMAT_VERSIONS)}"
        )
    development = manifest.get("development_split", {})
    if not manifest.get("is_pipeline_validation") and not development.get("enabled"):
        raise ValueError(
            "Scientific converted datasets must carve an enabled development split "
            "from official training/"
        )
    if verify_hashes:
        for name, expected in manifest["generated_sha256"].items():
            actual = sha256_file(root / name)
            if actual != expected:
                raise ValueError(f"SHA256 mismatch for {name}: {actual} != {expected}")
    return manifest


def quantile_scale(stats: dict[str, Any]) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    scale = np.maximum(np.abs(q01), np.abs(q99))
    scale[6] = 1.0
    if np.any(scale[:6] <= 0):
        raise ValueError(f"Invalid continuous-channel quantile scale: {scale}")
    return scale


def conditional_normalization_required(
    stats: dict[str, Any], protocol: dict[str, Any]
) -> bool:
    rule = protocol["action_representation"]["conditional_normalization_arm"]
    std = np.asarray(stats["std"], dtype=np.float64)[:6]
    q99_abs = np.asarray(stats["q99_abs_continuous"], dtype=np.float64)
    return bool(
        np.any(std < float(rule["enabled_if_any_continuous_std_below"]))
        or np.any(q99_abs < float(rule["enabled_if_any_continuous_q99_abs_below"]))
    )


def _episode_lookup(episodes: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(episode["episode_id"]): episode for episode in episodes}


def _load_or_build_split_indices(
    root: Path,
    split: Split,
    all_chunks: np.ndarray,
    split_codes: dict[str, int],
) -> np.ndarray:
    """Cache flatnonzero results so DataLoader workers do not recompute them."""
    chunk_path = root / "chunk_index.npy"
    digest = hashlib.sha1(chunk_path.read_bytes()).hexdigest()[:12]
    cache_path = root / f"indices_{split}_{digest}.npy"
    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")
    if split == "all":
        indices = np.arange(len(all_chunks), dtype=np.int64)
    else:
        if split not in split_codes:
            raise ValueError(f"Unknown split {split!r}")
        indices = np.flatnonzero(
            all_chunks["split_id"] == int(split_codes[split])
        ).astype(np.int64, copy=False)
    np.save(cache_path, indices, allow_pickle=False)
    return indices


class CalvinActionChunkDataset(Dataset):
    """Memory-mapped 30-step CALVIN relative-action chunks."""

    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        normalization: Normalization = "native",
        protocol_path: str | Path | None = None,
        verify_hashes: bool = False,
        cache_indices: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.manifest = validate_converted_dataset(self.root, verify_hashes)
        self.episode_index = load_json(self.root / "episode_index.json")
        self.stats = load_json(self.root / "norm_stats.json")
        self.actions = np.load(self.root / "actions.npy", mmap_mode="r")
        self.all_chunks = np.load(self.root / "chunk_index.npy", mmap_mode="r")
        self.chunk_length = int(self.manifest["chunk_length"])
        self.normalization = normalization
        self.episodes_by_id = _episode_lookup(self.episode_index["episodes"])

        if self.actions.ndim != 2 or self.actions.shape[1] != 7:
            raise ValueError(f"actions.npy must have shape (N, 7), got {self.actions.shape}")

        if cache_indices:
            self.indices = _load_or_build_split_indices(
                self.root,
                split,
                self.all_chunks,
                self.episode_index["split_codes"],
            )
        elif split == "all":
            self.indices = np.arange(len(self.all_chunks), dtype=np.int64)
        else:
            split_codes = self.episode_index["split_codes"]
            if split not in split_codes:
                raise ValueError(f"Unknown split {split!r}")
            self.indices = np.flatnonzero(
                self.all_chunks["split_id"] == int(split_codes[split])
            )

        if normalization == "quantile":
            if protocol_path is None:
                raise ValueError("quantile normalization requires protocol_path")
            protocol = load_json(Path(protocol_path))
            if not conditional_normalization_required(self.stats, protocol):
                raise RuntimeError(
                    "Quantile arm is forbidden: registered conditional threshold did not fire"
                )
            self.scale = quantile_scale(self.stats)
        elif normalization == "native":
            self.scale = np.ones(7, dtype=np.float32)
        else:
            raise ValueError(f"Unknown normalization {normalization!r}")

    def __len__(self) -> int:
        return len(self.indices)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root={str(self.root)!r}, split={self.split!r}, "
            f"n={len(self)}, chunk_length={self.chunk_length}, "
            f"normalization={self.normalization!r}, "
            f"pipeline_validation={self.manifest.get('is_pipeline_validation')})"
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        chunk = self.all_chunks[int(self.indices[index])]
        start = int(chunk["action_start"])
        stop = start + self.chunk_length
        action = np.array(self.actions[start:stop], dtype=np.float32, copy=True)
        if len(action) != self.chunk_length:
            raise IndexError(f"Short chunk at {start}: {len(action)}")
        action /= self.scale
        action = np.clip(action, -1.0, 1.0)
        episode_id = int(chunk["episode_id"])
        episode = self.episodes_by_id[episode_id]
        return {
            "action": torch.from_numpy(action),
            # Always True for CALVIN: every registered chunk is fully valid and
            # never zero-padded. Kept for ActionCodec encode/decode API parity.
            "padding_mask": torch.ones(self.chunk_length, dtype=torch.bool),
            "episode_id": torch.tensor(episode_id, dtype=torch.long),
            "source_start": torch.tensor(int(chunk["source_start"]), dtype=torch.long),
            "scene_id": torch.tensor(int(chunk["scene_id"]), dtype=torch.long),
            # Available here so a future VLA dataset can filter language-only
            # chunks without rebuilding the converted action cache.
            "has_lang_ann": torch.tensor(bool(episode["has_lang_ann"]), dtype=torch.bool),
        }
