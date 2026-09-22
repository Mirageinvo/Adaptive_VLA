#!/usr/bin/env python3
"""Smoke-check memmap DataLoader worker sharding for CALVIN action chunks.

Covers production-relevant DataLoader modes used by train_codec.py:
num_workers>0, persistent_workers, and pin_memory when CUDA is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Script lives in calvin_hicora/scripts/; keep sibling imports robust when the
# working directory is the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from codec_data import CalvinActionChunkDataset, validate_converted_dataset  # noqa: E402


def _consume(loader: DataLoader, max_batches: int = 8) -> dict[str, object]:
    batches = 0
    samples = 0
    episode_ids: set[int] = set()
    source_starts: set[int] = set()
    for batch in loader:
        assert batch["action"].shape[1:] == (30, 7)
        assert batch["padding_mask"].all()
        assert batch["scene_id"].ndim == 1
        assert batch["scene_id"].dtype == torch.int64
        assert batch["has_lang_ann"].dtype == torch.bool
        episode_ids.update(int(value) for value in batch["episode_id"].tolist())
        source_starts.update(int(value) for value in batch["source_start"].tolist())
        batches += 1
        samples += int(batch["action"].shape[0])
        if batches >= max_batches:
            break
    assert batches > 0
    # Debug train is one play episode, so episode diversity can be 1.
    # Distinct source_start values prove workers are feeding different chunks.
    assert len(source_starts) > 1, source_starts
    return {
        "batches": batches,
        "samples": samples,
        "unique_episode_ids": sorted(episode_ids),
        "num_unique_episode_ids": len(episode_ids),
        "num_unique_source_starts": len(source_starts),
    }


def main() -> int:
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/calvin_converted/debug")
    manifest = validate_converted_dataset(data_root)
    assert int(manifest["format_version"]) in (2, 3)

    dataset = CalvinActionChunkDataset(data_root, split="train", cache_indices=True)
    printed = repr(dataset)
    print(printed)
    assert printed.startswith("CalvinActionChunkDataset(")
    assert f"n={len(dataset)}" in printed

    digest = __import__("hashlib").sha1(
        (data_root / "chunk_index.npy").read_bytes()
    ).hexdigest()[:12]
    cache_path = data_root / f"indices_train_{digest}.npy"
    assert cache_path.exists(), cache_path
    cached = np.load(cache_path)
    assert len(cached) == len(dataset), (len(cached), len(dataset))
    assert cached.dtype == np.int64

    sample = dataset[0]
    # scene_id is stored as int8 in chunk_index.npy and must be widened here.
    assert sample["scene_id"].dtype == torch.int64
    # has_lang_ann is not in chunk_index.npy; it is looked up from episode_index.
    assert sample["has_lang_ann"].dtype == torch.bool
    assert sample["padding_mask"].all()
    assert sample["padding_mask"].dtype == torch.bool

    cuda = torch.cuda.is_available()
    report: dict[str, object] = {
        "cuda": cuda,
        "indices_cache": str(cache_path),
        "cache_len": int(len(cached)),
        "dataset_len": int(len(dataset)),
    }

    # Production train_codec uses persistent_workers when num_workers > 0.
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        shuffle=False,
        persistent_workers=True,
        pin_memory=cuda,
    )
    report["persistent_workers"] = _consume(loader)

    if cuda:
        # Explicit pin_memory path when CUDA is present.
        pinned = DataLoader(
            dataset,
            batch_size=32,
            num_workers=8,
            shuffle=False,
            persistent_workers=True,
            pin_memory=True,
        )
        report["pin_memory"] = _consume(pinned)
        first = next(iter(pinned))
        assert first["action"].is_pinned()

    # Close persistent workers cleanly.
    del loader
    if cuda:
        del pinned

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
