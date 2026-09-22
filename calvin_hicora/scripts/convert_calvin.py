#!/usr/bin/env python3
"""Convert official CALVIN transitions into an indexed action-chunk dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CHUNK_DTYPE = np.dtype(
    [
        ("action_start", "<i8"),
        ("episode_id", "<i4"),
        ("source_start", "<i8"),
        ("split_id", "i1"),
        ("scene_id", "i1"),
        ("valid_length", "<i2"),
    ]
)
SPLIT_CODES = {"train": 0, "val": 1, "dev": 2}
SCENE_CODES = {"A": 0, "B": 1, "C": 2, "D": 3}
CONVERTED_FORMAT_VERSION = 3
DEFAULT_DEV_FRACTION = 0.05
DEFAULT_DEV_SEED = 0


@dataclass(frozen=True)
class SourceSplit:
    name: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-length", type=int, default=30)
    parser.add_argument("--execution-horizon", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--action-key", default="rel_actions")
    parser.add_argument("--pipeline-validation", action="store_true")
    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=DEFAULT_DEV_FRACTION,
        help="Episode fraction carved from official training/ into development.",
    )
    parser.add_argument(
        "--dev-seed",
        type=int,
        default=DEFAULT_DEV_SEED,
        help="RNG seed for the episode-disjoint development carve.",
    )
    parser.add_argument(
        "--no-dev-split",
        action="store_true",
        help="Disable development carve (forbidden for scientific conversion).",
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        help="Official zip; its SHA256 is included in the manifest.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def script_sha1() -> str:
    return hashlib.sha1(Path(__file__).read_bytes()).hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def discover_splits(root: Path) -> list[SourceSplit]:
    aliases = (("train", "training"), ("val", "validation"))
    found = [
        SourceSplit(name, root / directory)
        for name, directory in aliases
        if (root / directory).is_dir()
    ]
    if found:
        if {split.name for split in found} != {"train", "val"}:
            raise ValueError(f"Expected both training/ and validation/ under {root}")
        return found

    basename = root.name.lower()
    if basename in {"training", "train"}:
        return [SourceSplit("train", root)]
    if basename in {"validation", "val"}:
        return [SourceSplit("val", root)]
    raise ValueError(f"Cannot find CALVIN training/ and validation/ under {root}")


def load_episode_bounds(split_dir: Path) -> np.ndarray:
    path = split_dir / "ep_start_end_ids.npy"
    bounds = np.asarray(np.load(path, allow_pickle=True), dtype=np.int64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"{path} must have shape (episodes, 2), got {bounds.shape}")
    if np.any(bounds[:, 1] < bounds[:, 0]):
        raise ValueError(f"Invalid episode bounds in {path}")
    # Official CALVIN ep_start_end_ids.npy is not guaranteed to be pre-sorted.
    order = np.argsort(bounds[:, 0], kind="mergesort")
    bounds = bounds[order]
    if len(bounds) > 1 and np.any(bounds[1:, 0] <= bounds[:-1, 1]):
        raise ValueError(f"Episode ranges overlap after sorting in {path}")
    return bounds


def normalize_scene_name(value: Any) -> str:
    match = re.search(r"(?:scene[_-]?)?([ABCD])$", str(value), flags=re.IGNORECASE)
    return match.group(1).upper() if match else "unknown"


def load_scene_ranges(split_dir: Path) -> list[tuple[str, int, int]]:
    path = split_dir / "scene_info.npy"
    if not path.exists():
        return []
    payload = np.load(path, allow_pickle=True).item()
    ranges = []
    for key, value in payload.items():
        start, end = map(int, value)
        scene = normalize_scene_name(key)
        if scene not in SCENE_CODES:
            raise ValueError(f"Unrecognized CALVIN scene name {key!r} in {path}")
        ranges.append((scene, start, end))
    return sorted(ranges, key=lambda row: row[1])


def default_scene_ranges_for_directory(split_dir: Path) -> list[tuple[str, int, int]]:
    text = str(split_dir).lower()
    if "task_d_d" in text or "calvin_debug_dataset" in text:
        return [("D", 0, 2**63 - 1)]
    raise FileNotFoundError(
        f"Missing required scene_info.npy in {split_dir} and cannot infer scene"
    )


def scene_for_episode(
    start: int, end: int, scene_ranges: Iterable[tuple[str, int, int]]
) -> str:
    matches = [name for name, low, high in scene_ranges if low <= start and end <= high]
    if len(matches) != 1:
        raise AssertionError(
            f"Episode [{start}, {end}] must be inside exactly one scene; matches={matches}"
        )
    return matches[0]


def load_language_ranges(split_dir: Path) -> list[tuple[int, int]]:
    path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not path.exists():
        return []
    payload = np.load(path, allow_pickle=True).item()
    return [(int(start), int(end)) for start, end in payload["info"]["indx"]]


def overlaps_any(start: int, end: int, ranges: Iterable[tuple[int, int]]) -> bool:
    return any(start <= high and low <= end for low, high in ranges)


def transition_path(split_dir: Path, frame_id: int) -> Path:
    return split_dir / f"episode_{frame_id:07d}.npz"


def read_action(path: Path, action_key: str) -> tuple[np.ndarray, bool]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as transition:
        if action_key not in transition:
            raise KeyError(f"{path} has no {action_key!r}; keys={transition.files}")
        action = np.asarray(transition[action_key], dtype=np.float32)
    if action.shape != (7,):
        raise ValueError(f"{path}:{action_key} must have shape (7,), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError(f"{path}:{action_key} contains NaN or inf")
    if np.any(action < -1.00001) or np.any(action > 1.00001):
        raise ValueError(f"{path}:{action_key} is outside CALVIN's [-1, 1] scale")
    clipped = np.clip(action, -1.0, 1.0)
    return clipped, bool(np.any(clipped != action))


def build_episode_records(splits: list[SourceSplit]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    action_offset = 0
    for split in splits:
        bounds = load_episode_bounds(split.path)
        scenes = load_scene_ranges(split.path)
        languages = load_language_ranges(split.path)
        if not scenes:
            scenes = default_scene_ranges_for_directory(split.path)
        for source_start, source_end in bounds:
            source_start, source_end = int(source_start), int(source_end)
            n_frames = source_end - source_start + 1
            scene = scene_for_episode(source_start, source_end, scenes)
            records.append(
                {
                    "episode_id": len(records),
                    "split": split.name,
                    "scene": scene,
                    "source_directory": str(split.path.resolve()),
                    "source_start": source_start,
                    "source_end_inclusive": source_end,
                    "action_start": action_offset,
                    "action_end_exclusive": action_offset + n_frames,
                    "n_frames": n_frames,
                    "has_lang_ann": overlaps_any(source_start, source_end, languages),
                }
            )
            action_offset += n_frames
    return records


def carve_development_split(
    records: list[dict[str, Any]],
    *,
    fraction: float,
    seed: int,
    stratify_by_scene: bool = True,
) -> dict[str, Any]:
    """Carve an episode-disjoint development split from official training episodes.

    Official CALVIN validation/ remains untouched and is not used for selection.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"dev fraction must be in (0, 1), got {fraction}")
    train_indices = [i for i, record in enumerate(records) if record["split"] == "train"]
    if len(train_indices) < 2:
        raise ValueError(
            "Need at least two official training episodes to carve a development split"
        )

    rng = np.random.RandomState(seed)
    selected: list[int] = []
    if stratify_by_scene:
        by_scene: dict[str, list[int]] = {}
        for index in train_indices:
            by_scene.setdefault(records[index]["scene"], []).append(index)
        for scene in sorted(by_scene):
            indices = list(by_scene[scene])
            rng.shuffle(indices)
            # Leave at least one train episode in the scene when possible.
            n_take = min(len(indices) - 1, max(0, int(round(len(indices) * fraction))))
            selected.extend(indices[:n_take])
    if not selected:
        indices = list(train_indices)
        rng.shuffle(indices)
        n_take = min(len(indices) - 1, max(1, int(round(len(indices) * fraction))))
        selected = indices[:n_take]

    if not selected:
        raise RuntimeError("Development carve selected no episodes")
    if len(selected) >= len(train_indices):
        raise RuntimeError("Development carve would empty the training split")

    for index in selected:
        records[index]["split"] = "dev"
        records[index]["carved_from"] = "training"

    remaining_train = sum(1 for record in records if record["split"] == "train")
    report = {
        "enabled": True,
        "source": "episode_disjoint_carve_from_official_training",
        "fraction": fraction,
        "seed": seed,
        "stratify_by_scene": stratify_by_scene,
        "dev_episode_ids": sorted(int(records[i]["episode_id"]) for i in selected),
        "dev_episode_count": len(selected),
        "train_episode_count_after_carve": remaining_train,
        "official_validation_untouched": True,
    }
    return report


def write_actions(
    records: list[dict[str, Any]], output: Path, action_key: str
) -> tuple[np.memmap, int]:
    total_frames = sum(record["n_frames"] for record in records)
    actions = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(total_frames, 7)
    )
    written = 0
    clipped_frames = 0
    for record in records:
        split_dir = Path(record["source_directory"])
        for source_id in range(
            record["source_start"], record["source_end_inclusive"] + 1
        ):
            action, was_clipped = read_action(
                transition_path(split_dir, source_id), action_key
            )
            actions[written] = action
            clipped_frames += int(was_clipped)
            written += 1
            if written % 10000 == 0:
                print(f"converted {written}/{total_frames} frames", flush=True)
    assert written == total_frames
    actions.flush()
    return actions, clipped_frames


def build_chunks(
    records: list[dict[str, Any]], chunk_length: int, stride: int
) -> np.ndarray:
    rows = []
    for record in records:
        episode_length = int(record["n_frames"])
        if episode_length <= 0:
            raise ValueError(f"Episode {record['episode_id']} has no actions")
        last_start = record["action_end_exclusive"] - chunk_length
        if episode_length < chunk_length:
            rows.append(
                (
                    record["action_start"],
                    record["episode_id"],
                    record["source_start"],
                    SPLIT_CODES[record["split"]],
                    SCENE_CODES[record["scene"]],
                    episode_length,
                )
            )
            continue
        for action_start in range(record["action_start"], last_start + 1, stride):
            source_start = record["source_start"] + action_start - record["action_start"]
            rows.append(
                (
                    action_start,
                    record["episode_id"],
                    source_start,
                    SPLIT_CODES[record["split"]],
                    SCENE_CODES[record["scene"]],
                    chunk_length,
                )
            )
    chunks = np.asarray(rows, dtype=CHUNK_DTYPE)
    for chunk in chunks:
        record = records[int(chunk["episode_id"])]
        start = int(chunk["action_start"])
        valid_length = int(chunk["valid_length"])
        assert record["action_start"] <= start
        assert 0 < valid_length <= chunk_length
        assert start + valid_length <= record["action_end_exclusive"]
        if valid_length < chunk_length:
            assert start == record["action_start"]
            assert valid_length == record["n_frames"]
        assert int(chunk["split_id"]) == SPLIT_CODES[record["split"]]
        assert int(chunk["scene_id"]) == SCENE_CODES[record["scene"]]
    return chunks


def train_statistics(actions: np.ndarray, records: list[dict[str, Any]]) -> dict[str, Any]:
    train_parts = [
        np.asarray(actions[r["action_start"] : r["action_end_exclusive"]])
        for r in records
        if r["split"] == "train"
    ]
    if not train_parts:
        raise ValueError("No train actions found")
    train = np.concatenate(train_parts, axis=0)
    continuous = train[:, :6]
    gripper_values = np.unique(train[:, 6])
    rounded = {float(np.round(value, decimals=6)) for value in gripper_values.tolist()}
    if not rounded.issubset({-1.0, 1.0}):
        raise ValueError(
            f"Train gripper values must be in {{-1, +1}}, got {sorted(rounded)}"
        )
    return {
        "source": "train_split_only",
        "n_frames": int(len(train)),
        "mean": train.mean(axis=0).tolist(),
        "std": train.std(axis=0).tolist(),
        "q01": np.quantile(train, 0.01, axis=0).tolist(),
        "q99": np.quantile(train, 0.99, axis=0).tolist(),
        "q99_abs_continuous": np.quantile(np.abs(continuous), 0.99, axis=0).tolist(),
        "gripper_values": gripper_values.tolist(),
    }


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.chunk_length <= 0 or args.stride <= 0 or args.execution_horizon <= 0:
        raise ValueError("chunk-length, execution-horizon, and stride must be positive")
    if args.execution_horizon > args.chunk_length:
        raise ValueError("execution-horizon cannot exceed chunk-length")
    if args.no_dev_split and not args.pipeline_validation:
        raise ValueError(
            "--no-dev-split is forbidden for scientific conversion; "
            "pass --pipeline-validation only for debug smoke trees"
        )
    if args.dev_fraction <= 0 or args.dev_fraction >= 1:
        raise ValueError("--dev-fraction must be in (0, 1)")
    conversion_started = time.perf_counter()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_root} is non-empty; pass --overwrite explicitly")
    output_root.mkdir(parents=True, exist_ok=True)
    for stale in output_root.glob("indices_*.npy"):
        stale.unlink()
    incomplete = output_root / ".conversion_incomplete"
    incomplete.write_text("conversion in progress\n")

    splits = discover_splits(dataset_root)
    records = build_episode_records(splits)
    if args.no_dev_split:
        development_split = {
            "enabled": False,
            "reason": "explicitly disabled for pipeline validation only",
        }
    else:
        train_episode_count = sum(1 for record in records if record["split"] == "train")
        if train_episode_count < 2:
            if not args.pipeline_validation:
                raise ValueError(
                    "Need at least two official training episodes to carve a "
                    "development split"
                )
            development_split = {
                "enabled": False,
                "reason": (
                    "pipeline_validation_source_has_fewer_than_two_training_episodes; "
                    "selection falls back to official val for smoke only"
                ),
                "train_episode_count": train_episode_count,
            }
        else:
            development_split = carve_development_split(
                records,
                fraction=float(args.dev_fraction),
                seed=int(args.dev_seed),
                stratify_by_scene=True,
            )
    actions_path = output_root / "actions.npy"
    actions, clipped_frames = write_actions(records, actions_path, args.action_key)
    chunks = build_chunks(records, args.chunk_length, args.stride)
    chunks_path = output_root / "chunk_index.npy"
    np.save(chunks_path, chunks, allow_pickle=False)

    episode_index = {
        "format_version": CONVERTED_FORMAT_VERSION,
        "split_codes": SPLIT_CODES,
        "scene_codes": SCENE_CODES,
        "development_split": development_split,
        "episodes": records,
    }
    episode_path = output_root / "episode_index.json"
    json_dump(episode_path, episode_index)
    stats = train_statistics(actions, records)
    stats_path = output_root / "norm_stats.json"
    json_dump(stats_path, stats)

    repo_root = Path(__file__).resolve().parents[2]
    source_metadata = {}
    for split in splits:
        for relative in (
            "ep_start_end_ids.npy",
            "scene_info.npy",
            "lang_annotations/auto_lang_ann.npy",
        ):
            path = split.path / relative
            if path.exists():
                source_metadata[str(path.relative_to(dataset_root))] = sha256_file(path)
    manifest = {
        "format_version": CONVERTED_FORMAT_VERSION,
        "is_pipeline_validation": bool(args.pipeline_validation),
        "dataset_root": str(dataset_root),
        "action_key": args.action_key,
        "action_semantics": {
            "additional_normalization": False,
            "clip": [-1.0, 1.0],
            "flip_gripper_sign": False,
        },
        "chunk_length": args.chunk_length,
        "execution_horizon": args.execution_horizon,
        "stride": args.stride,
        "short_episode_padding": "repeat_last_action",
        "padding_mask_semantics": "true_is_valid_false_is_padding",
        "development_split": development_split,
        "num_frames": int(actions.shape[0]),
        "clipped_frames": clipped_frames,
        "num_episodes": len(records),
        "num_chunks": len(chunks),
        "num_padded_chunks": int((chunks["valid_length"] < args.chunk_length).sum()),
        "split_counts": {
            name: sum(1 for record in records if record["split"] == name)
            for name in SPLIT_CODES
        },
        "source_archive": (
            {
                "path": str(args.source_archive.resolve()),
                "sha256": sha256_file(args.source_archive),
            }
            if args.source_archive
            else None
        ),
        "source_metadata_sha256": source_metadata,
        "generated_sha256": {
            "actions.npy": sha256_file(actions_path),
            "chunk_index.npy": sha256_file(chunks_path),
            "episode_index.json": sha256_file(episode_path),
            "norm_stats.json": sha256_file(stats_path),
        },
        "converter_script_sha1": script_sha1(),
        "adaptive_vla_commit": git_commit(repo_root),
        "python": sys.version,
        "numpy": np.__version__,
    }
    elapsed_seconds = time.perf_counter() - conversion_started
    manifest["conversion_elapsed_seconds"] = elapsed_seconds
    manifest["conversion_frames_per_second"] = int(actions.shape[0]) / max(
        elapsed_seconds, np.finfo(float).eps
    )
    manifest_path = output_root / "data_manifest.json"
    json_dump(manifest_path, manifest)
    incomplete.unlink()
    print(
        f"wrote {manifest['num_frames']} frames, {manifest['num_episodes']} episodes, "
        f"{manifest['num_chunks']} chunks to {output_root}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("conversion failed; partial output retains .conversion_incomplete", file=sys.stderr)
        raise
