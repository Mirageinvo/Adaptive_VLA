#!/usr/bin/env python3
"""Stream CALVIN actions from a local zip or directly from HTTP — no full unzip.

Variant B+ (network): Freiburg advertises ``Accept-Ranges: bytes``. We:

1. ``HEAD`` the archive and Range-read the central directory via ``HttpRangeFile``
   + ``zipfile.ZipFile`` (seekable; never loads the zip into a giant BytesIO).
2. Pull only metadata members (episode bounds / scene / language) over Range.
3. Sequentially stream the ZIP body once (with byte-offset resume), inflate only
   ``episode_*.npz`` members, keep ``rel_actions``, discard RGB payloads, and
   write into an ``actions.npy`` memmap with periodic ``flush()``.

Disk peak ≈ sizeof(actions.npy) + indices (tens of MiB), not the 177 GiB archive.
Run long jobs under ``tmux`` with ``PYTHONUNBUFFERED=1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from convert_calvin import (
    CONVERTED_FORMAT_VERSION,
    DEFAULT_DEV_FRACTION,
    DEFAULT_DEV_SEED,
    SCENE_CODES,
    SPLIT_CODES,
    build_chunks,
    carve_development_split,
    normalize_scene_name,
    overlaps_any,
    scene_for_episode,
    sha256_file,
    train_statistics,
)
from zip_http_stream import (
    HttpRangeFile,
    ResumableHttpStream,
    fetch_text,
    head_content_length,
    iter_zip_local_members,
    parse_official_sha256,
)


KNOWN_TRANSITION_KEYS = (
    "actions",
    "rel_actions",
    "robot_obs",
    "scene_obs",
    "rgb_static",
    "rgb_gripper",
    "rgb_tactile",
    "depth_static",
    "depth_gripper",
    "depth_tactile",
)

EPISODE_MEMBER_RE = re.compile(
    r"^(?:.*/)?(?P<split>training|validation)/episode_(?P<frame>\d{7})\.npz$"
)
CHECKPOINT_NAME = ".stream_checkpoint.json"
DEFAULT_CALVIN_BASE = "http://calvin.cs.uni-freiburg.de/dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-zip",
        type=Path,
        help="Local official zip (never fully extracted).",
    )
    source.add_argument(
        "--source-url",
        help="HTTP(S) URL of the official zip (streamed; not saved to disk).",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-length", type=int, default=30)
    parser.add_argument("--execution-horizon", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--action-key", default="rel_actions")
    parser.add_argument("--pipeline-validation", action="store_true")
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument("--dev-seed", type=int, default=DEFAULT_DEV_SEED)
    parser.add_argument("--no-dev-split", action="store_true")
    parser.add_argument(
        "--expected-sha256",
        help="Official archive digest; default: fetch sha256sum.txt next to --source-url.",
    )
    parser.add_argument(
        "--checksum-url",
        default=f"{DEFAULT_CALVIN_BASE}/sha256sum.txt",
        help="Official sha256sum.txt URL used when --expected-sha256 is omitted.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1000,
        help="Flush actions memmap every N written frames (tmux crash safety).",
    )
    parser.add_argument(
        "--checkpoint-every-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Persist sequential-stream byte offset this often.",
    )
    parser.add_argument(
        "--delete-zip-after",
        action="store_true",
        help="Delete local --source-zip after SHA-256 is recorded (URL mode never saves).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def script_sha1() -> str:
    return hashlib.sha1(Path(__file__).read_bytes()).hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_npy_bytes(raw: bytes) -> Any:
    import io

    return np.load(io.BytesIO(raw), allow_pickle=True)


def load_npy_from_zip(archive: zipfile.ZipFile, member: str) -> Any:
    with archive.open(member) as handle:
        return np.load(handle, allow_pickle=True)


def action_from_npz_bytes(
    raw: bytes, action_key: str, member: str
) -> tuple[np.ndarray, bool, tuple[str, ...]]:
    import io

    with np.load(io.BytesIO(raw), allow_pickle=True) as transition:
        present = tuple(transition.files)
        if action_key not in transition:
            raise KeyError(f"{member} has no {action_key!r}; keys={list(present)}")
        action = np.asarray(transition[action_key], dtype=np.float32)
    if action.shape != (7,):
        raise ValueError(f"{member}:{action_key} must have shape (7,), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError(f"{member}:{action_key} contains NaN or inf")
    if np.any(action < -1.00001) or np.any(action > 1.00001):
        raise ValueError(f"{member}:{action_key} is outside CALVIN's [-1, 1] scale")
    clipped = np.clip(action, -1.0, 1.0)
    return clipped, bool(np.any(clipped != action)), present


def read_action_from_zip(
    archive: zipfile.ZipFile, member: str, action_key: str
) -> tuple[np.ndarray, bool, tuple[str, ...]]:
    with archive.open(member) as handle:
        with np.load(handle, allow_pickle=True) as transition:
            present = tuple(transition.files)
            if action_key not in transition:
                raise KeyError(f"{member} has no {action_key!r}; keys={list(present)}")
            action = np.asarray(transition[action_key], dtype=np.float32)
    if action.shape != (7,):
        raise ValueError(f"{member}:{action_key} must have shape (7,), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError(f"{member}:{action_key} contains NaN or inf")
    if np.any(action < -1.00001) or np.any(action > 1.00001):
        raise ValueError(f"{member}:{action_key} is outside CALVIN's [-1, 1] scale")
    clipped = np.clip(action, -1.0, 1.0)
    return clipped, bool(np.any(clipped != action)), present


def discover_zip_splits(archive: zipfile.ZipFile) -> list[dict[str, str]]:
    meta_hits: dict[str, str] = {}
    for info in archive.infolist():
        name = info.filename
        if name.endswith("/ep_start_end_ids.npy"):
            parent = name[: -len("/ep_start_end_ids.npy")]
            split_name = Path(parent).name
            if split_name in {"training", "validation"}:
                meta_hits[split_name] = parent
    if "training" not in meta_hits or "validation" not in meta_hits:
        raise ValueError(
            "Zip must contain both training/ and validation/ with ep_start_end_ids.npy; "
            f"found={sorted(meta_hits)}"
        )
    print(
        f"[CD] central directory ok; splits={{{meta_hits['training']!r}, "
        f"{meta_hits['validation']!r}}}; n_members={len(archive.infolist())}",
        flush=True,
    )
    return [
        {"name": "train", "zip_dir": meta_hits["training"], "source_split": "training"},
        {"name": "val", "zip_dir": meta_hits["validation"], "source_split": "validation"},
    ]


def episode_member_path(split_zip_dir: str, frame_id: int) -> str:
    return f"{split_zip_dir}/episode_{frame_id:07d}.npz"


def load_episode_bounds(archive: zipfile.ZipFile, split_zip_dir: str) -> np.ndarray:
    member = f"{split_zip_dir}/ep_start_end_ids.npy"
    bounds = np.asarray(load_npy_from_zip(archive, member), dtype=np.int64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"{member} must have shape (episodes, 2), got {bounds.shape}")
    if np.any(bounds[:, 1] < bounds[:, 0]):
        raise ValueError(f"Invalid episode bounds in {member}")
    # Official CALVIN ep_start_end_ids.npy is not guaranteed to be pre-sorted.
    order = np.argsort(bounds[:, 0], kind="mergesort")
    bounds = bounds[order]
    if len(bounds) > 1 and np.any(bounds[1:, 0] <= bounds[:-1, 1]):
        raise ValueError(f"Episode ranges overlap after sorting in {member}")
    return bounds


def load_scene_ranges(
    archive: zipfile.ZipFile, split_zip_dir: str
) -> list[tuple[str, int, int]]:
    member = f"{split_zip_dir}/scene_info.npy"
    try:
        payload = load_npy_from_zip(archive, member).item()
    except KeyError:
        return []
    ranges = []
    for key, value in payload.items():
        start, end = map(int, value)
        scene = normalize_scene_name(key)
        if scene not in SCENE_CODES:
            raise ValueError(f"Unrecognized CALVIN scene name {key!r} in {member}")
        ranges.append((scene, start, end))
    return sorted(ranges, key=lambda row: row[1])


def default_scene_ranges_for_split(split_zip_dir: str) -> list[tuple[str, int, int]]:
    """task_D_D validation omits scene_info.npy; the package is scene D only."""
    lowered = split_zip_dir.lower()
    if "task_d_d" in lowered or "calvin_debug_dataset" in lowered:
        print(
            f"[meta] {split_zip_dir}: scene_info.npy absent; defaulting all episodes to scene D",
            flush=True,
        )
        return [("D", 0, 2**63 - 1)]
    raise FileNotFoundError(
        f"Missing required scene_info.npy under {split_zip_dir} and cannot infer scene"
    )


def load_language_ranges(
    archive: zipfile.ZipFile, split_zip_dir: str
) -> list[tuple[int, int]]:
    member = f"{split_zip_dir}/lang_annotations/auto_lang_ann.npy"
    try:
        payload = load_npy_from_zip(archive, member).item()
    except KeyError:
        return []
    return [(int(start), int(end)) for start, end in payload["info"]["indx"]]


def build_episode_records_from_zip(
    archive: zipfile.ZipFile, splits: list[dict[str, str]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    action_offset = 0
    for split in splits:
        bounds = load_episode_bounds(archive, split["zip_dir"])
        scenes = load_scene_ranges(archive, split["zip_dir"])
        if not scenes:
            scenes = default_scene_ranges_for_split(split["zip_dir"])
        languages = load_language_ranges(archive, split["zip_dir"])
        for source_start, source_end in bounds:
            source_start, source_end = int(source_start), int(source_end)
            n_frames = source_end - source_start + 1
            scene = scene_for_episode(source_start, source_end, scenes)
            records.append(
                {
                    "episode_id": len(records),
                    "split": split["name"],
                    "scene": scene,
                    "source_zip_directory": split["zip_dir"],
                    "source_split": split["source_split"],
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


def write_episode_index_npz(path: Path, records: list[dict[str, Any]]) -> None:
    n = len(records)
    episode_id = np.empty(n, dtype=np.int32)
    action_start = np.empty(n, dtype=np.int64)
    action_end_exclusive = np.empty(n, dtype=np.int64)
    source_start = np.empty(n, dtype=np.int64)
    source_end_inclusive = np.empty(n, dtype=np.int64)
    n_frames = np.empty(n, dtype=np.int32)
    split_id = np.empty(n, dtype=np.int8)
    scene_id = np.empty(n, dtype=np.int8)
    has_lang_ann = np.empty(n, dtype=np.bool_)
    for i, record in enumerate(records):
        episode_id[i] = record["episode_id"]
        action_start[i] = record["action_start"]
        action_end_exclusive[i] = record["action_end_exclusive"]
        source_start[i] = record["source_start"]
        source_end_inclusive[i] = record["source_end_inclusive"]
        n_frames[i] = record["n_frames"]
        split_id[i] = SPLIT_CODES[record["split"]]
        scene_id[i] = SCENE_CODES[record["scene"]]
        has_lang_ann[i] = bool(record["has_lang_ann"])
    np.savez_compressed(
        path,
        episode_id=episode_id,
        action_start=action_start,
        action_end_exclusive=action_end_exclusive,
        source_start=source_start,
        source_end_inclusive=source_end_inclusive,
        n_frames=n_frames,
        split_id=split_id,
        scene_id=scene_id,
        has_lang_ann=has_lang_ann,
        split_codes=np.asarray(list(SPLIT_CODES.items()), dtype=object),
        scene_codes=np.asarray(list(SCENE_CODES.items()), dtype=object),
        format_version=np.int32(CONVERTED_FORMAT_VERSION),
    )


def assert_chunks_respect_episode_bounds(
    chunks: np.ndarray, records: list[dict[str, Any]], chunk_length: int
) -> None:
    for chunk in chunks:
        record = records[int(chunk["episode_id"])]
        start = int(chunk["action_start"])
        valid_length = int(chunk["valid_length"])
        stop = start + valid_length
        if not (record["action_start"] <= start < stop <= record["action_end_exclusive"]):
            raise AssertionError(
                f"Chunk crosses episode boundary: chunk=[{start}, {stop}) "
                f"episode={record['episode_id']} "
                f"bounds=[{record['action_start']}, {record['action_end_exclusive']})"
            )
        if valid_length < chunk_length:
            if start != record["action_start"] or valid_length != record["n_frames"]:
                raise AssertionError(
                    f"Short-episode padding contract broken for episode {record['episode_id']}"
                )


def modalities_contract(action_key: str, observed_keys: Iterable[str]) -> dict[str, Any]:
    observed = list(observed_keys)
    discarded = [key for key in observed if key != action_key]
    known_discarded = [
        key for key in KNOWN_TRANSITION_KEYS if key != action_key and key not in discarded
    ]
    return {
        "actions_only": True,
        "extracted_modalities": [action_key],
        "discarded_modalities_observed": discarded,
        "discarded_modalities_known_calvin": known_discarded,
        "rgb_present_in_cache": False,
        "vla_image_read_forbidden": True,
        "note": (
            "This cache retains robot relative actions only. Future VLA training "
            "must not attempt to load rgb_static/rgb_gripper from this directory; "
            "RGB requires the official archive or a separate vision cache."
        ),
    }


def resolve_development_split(
    records: list[dict[str, Any]],
    *,
    no_dev_split: bool,
    pipeline_validation: bool,
    dev_fraction: float,
    dev_seed: int,
) -> dict[str, Any]:
    if no_dev_split:
        return {
            "enabled": False,
            "reason": "explicitly disabled for pipeline validation only",
        }
    train_episode_count = sum(1 for record in records if record["split"] == "train")
    if train_episode_count < 2:
        if not pipeline_validation:
            raise ValueError(
                "Need at least two official training episodes to carve a development split"
            )
        return {
            "enabled": False,
            "reason": (
                "pipeline_validation_source_has_fewer_than_two_training_episodes; "
                "selection falls back to official val for smoke only"
            ),
            "train_episode_count": train_episode_count,
        }
    return carve_development_split(
        records,
        fraction=float(dev_fraction),
        seed=int(dev_seed),
        stratify_by_scene=True,
    )


def frame_to_action_index(records: list[dict[str, Any]]) -> dict[tuple[str, int], int]:
    mapping: dict[tuple[str, int], int] = {}
    for record in records:
        split = record["source_split"]
        for offset, source_id in enumerate(
            range(record["source_start"], record["source_end_inclusive"] + 1)
        ):
            mapping[(split, source_id)] = int(record["action_start"] + offset)
    return mapping


def write_actions_from_local_zip(
    archive: zipfile.ZipFile,
    records: list[dict[str, Any]],
    output: Path,
    action_key: str,
    flush_every: int,
) -> tuple[np.memmap, int, tuple[str, ...]]:
    total_frames = sum(int(record["n_frames"]) for record in records)
    actions = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(total_frames, 7)
    )
    written = 0
    clipped_frames = 0
    observed_keys: tuple[str, ...] = ()
    for record in records:
        split_dir = record["source_zip_directory"]
        for source_id in range(record["source_start"], record["source_end_inclusive"] + 1):
            member = episode_member_path(split_dir, source_id)
            action, was_clipped, present = read_action_from_zip(
                archive, member, action_key
            )
            if not observed_keys:
                observed_keys = present
            actions[written] = action
            clipped_frames += int(was_clipped)
            written += 1
            if written % flush_every == 0:
                actions.flush()
                print(f"streamed {written}/{total_frames} actions (flushed)", flush=True)
    actions.flush()
    if written != total_frames:
        raise RuntimeError(f"Wrote {written} frames, expected {total_frames}")
    return actions, clipped_frames, observed_keys


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_actions_from_http_stream(
    url: str,
    size: int,
    records: list[dict[str, Any]],
    output: Path,
    action_key: str,
    *,
    flush_every: int,
    checkpoint_path: Path,
    checkpoint_every_bytes: int,
    expected_sha256: str | None,
) -> tuple[np.memmap, int, tuple[str, ...], str | None]:
    """Sequential body download: inflate episode members, memmap actions, resume."""
    total_frames = sum(int(record["n_frames"]) for record in records)
    index_map = frame_to_action_index(records)
    needed = set(index_map)
    actions_path = output
    seen_path = output.with_name("actions_seen_mask.npy")

    checkpoint = load_checkpoint(checkpoint_path) or {}
    start_offset = int(checkpoint.get("bytes_consumed", 0))
    if actions_path.exists() and seen_path.exists() and start_offset > 0:
        actions = np.lib.format.open_memmap(actions_path, mode="r+")
        seen = np.lib.format.open_memmap(seen_path, mode="r+")
        print(
            f"[stream] resume checkpoint bytes_consumed={start_offset} "
            f"frames_done={int(seen.sum())}/{total_frames}",
            flush=True,
        )
    else:
        actions = np.lib.format.open_memmap(
            actions_path, mode="w+", dtype=np.float32, shape=(total_frames, 7)
        )
        seen = np.lib.format.open_memmap(
            seen_path, mode="w+", dtype=np.bool_, shape=(total_frames,)
        )
        seen[:] = False
        seen.flush()
        start_offset = 0

    clipped_frames = int(checkpoint.get("clipped_frames", 0))
    observed_keys: tuple[str, ...] = tuple(checkpoint.get("observed_keys", ()))
    last_checkpoint_bytes = start_offset
    written_since_flush = 0

    def should_materialize(filename: str) -> bool:
        match = EPISODE_MEMBER_RE.match(filename)
        if match is None:
            return False
        key = (match.group("split"), int(match.group("frame")))
        if key not in needed:
            return False
        action_index = index_map[key]
        return not bool(seen[action_index])

    stream = ResumableHttpStream(
        url,
        size,
        start_offset=start_offset,
        hash_stream=(start_offset == 0),
    )
    try:
        for member in iter_zip_local_members(
            stream,
            start_offset=start_offset,
            should_materialize=should_materialize,
        ):
            start_offset = stream.bytes_consumed
            match = EPISODE_MEMBER_RE.match(member.filename)
            if match is None or not member.data:
                if stream.bytes_consumed - last_checkpoint_bytes >= checkpoint_every_bytes:
                    actions.flush()
                    seen.flush()
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "bytes_consumed": stream.bytes_consumed,
                            "clipped_frames": clipped_frames,
                            "observed_keys": list(observed_keys),
                            "frames_done": int(seen.sum()),
                            "total_frames": total_frames,
                        },
                    )
                    last_checkpoint_bytes = stream.bytes_consumed
                    print(
                        f"[stream] checkpoint offset={stream.bytes_consumed}/{size} "
                        f"frames={int(seen.sum())}/{total_frames}",
                        flush=True,
                    )
                continue

            split = match.group("split")
            frame_id = int(match.group("frame"))
            key = (split, frame_id)
            action_index = index_map[key]
            if seen[action_index]:
                continue
            action, was_clipped, present = action_from_npz_bytes(
                member.data, action_key, member.filename
            )
            if not observed_keys:
                observed_keys = present
            actions[action_index] = action
            seen[action_index] = True
            clipped_frames += int(was_clipped)
            written_since_flush += 1
            done = int(seen.sum())
            if written_since_flush >= flush_every:
                actions.flush()
                seen.flush()
                written_since_flush = 0
                print(
                    f"[stream] frames={done}/{total_frames} "
                    f"offset={stream.bytes_consumed}/{size}",
                    flush=True,
                )
            if stream.bytes_consumed - last_checkpoint_bytes >= checkpoint_every_bytes:
                actions.flush()
                seen.flush()
                save_checkpoint(
                    checkpoint_path,
                    {
                        "bytes_consumed": stream.bytes_consumed,
                        "clipped_frames": clipped_frames,
                        "observed_keys": list(observed_keys),
                        "frames_done": done,
                        "total_frames": total_frames,
                    },
                )
                last_checkpoint_bytes = stream.bytes_consumed
        # Drain CD + EOCD so a from-byte-0 stream SHA-256 matches the full object.
        remaining = size - stream.bytes_consumed
        if remaining > 0:
            print(f"[stream] draining trailing {remaining} CD/EOCD bytes", flush=True)
            stream.discard(remaining)
        actions.flush()
        seen.flush()
    finally:
        stream.close()

    missing = int((~np.asarray(seen)).sum())
    if missing:
        raise RuntimeError(
            f"HTTP stream finished but {missing}/{total_frames} action slots are empty; "
            "re-run to resume from checkpoint"
        )
    stream_digest = stream.hexdigest()
    if expected_sha256 and stream_digest and stream_digest != expected_sha256.lower():
        raise ValueError(
            f"Stream SHA-256 mismatch: {stream_digest} != {expected_sha256.lower()}"
        )
    if expected_sha256 and stream_digest is None:
        print(
            "[stream] SHA-256 not recomputed (resumed mid-archive); "
            f"recording official digest {expected_sha256}",
            flush=True,
        )
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    seen_path.unlink(missing_ok=True)
    return actions, clipped_frames, observed_keys, stream_digest


def finalize_dataset(
    *,
    output_root: Path,
    records: list[dict[str, Any]],
    actions: np.memmap,
    actions_path: Path,
    clipped_frames: int,
    observed_keys: tuple[str, ...],
    development_split: dict[str, Any],
    args: argparse.Namespace,
    conversion_started: float,
    source_manifest: dict[str, Any],
    conversion_mode: str,
) -> dict[str, Any]:
    chunks = build_chunks(records, args.chunk_length, args.stride)
    assert_chunks_respect_episode_bounds(chunks, records, args.chunk_length)
    chunks_path = output_root / "chunk_index.npy"
    np.save(chunks_path, chunks, allow_pickle=False)

    episode_index = {
        "format_version": CONVERTED_FORMAT_VERSION,
        "split_codes": SPLIT_CODES,
        "scene_codes": SCENE_CODES,
        "development_split": development_split,
        "episodes": records,
    }
    episode_json_path = output_root / "episode_index.json"
    json_dump(episode_json_path, episode_index)
    episode_npz_path = output_root / "episode_index.npz"
    write_episode_index_npz(episode_npz_path, records)

    stats = train_statistics(actions, records)
    stats_path = output_root / "norm_stats.json"
    json_dump(stats_path, stats)

    modalities = modalities_contract(args.action_key, observed_keys)
    repo_root = Path(__file__).resolve().parents[2]
    elapsed_seconds = time.perf_counter() - conversion_started
    manifest = {
        "format_version": CONVERTED_FORMAT_VERSION,
        "converter": "convert_from_zip_actions_only.py",
        "conversion_mode": conversion_mode,
        "is_pipeline_validation": bool(args.pipeline_validation),
        "source": source_manifest,
        "modalities": modalities,
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
        "window_boundary_rule": (
            "sliding windows with the configured stride stay inside a single "
            "episode; action_end_exclusive from episode_index is hard"
        ),
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
        "generated_sha256": {
            "actions.npy": sha256_file(actions_path),
            "chunk_index.npy": sha256_file(chunks_path),
            "episode_index.json": sha256_file(episode_json_path),
            "episode_index.npz": sha256_file(episode_npz_path),
            "norm_stats.json": sha256_file(stats_path),
        },
        "converter_script_sha1": script_sha1(),
        "adaptive_vla_commit": git_commit(repo_root),
        "python": sys.version,
        "numpy": np.__version__,
        "conversion_elapsed_seconds": elapsed_seconds,
        "conversion_frames_per_second": int(actions.shape[0])
        / max(elapsed_seconds, np.finfo(float).eps),
    }
    manifest_path = output_root / "data_manifest.json"
    json_dump(manifest_path, manifest)
    return manifest


def convert_from_local_zip(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    source_zip = args.source_zip.resolve()
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    conversion_started = time.perf_counter()
    print(f"hashing source zip {source_zip} ...", flush=True)
    source_sha256 = sha256_file(source_zip)
    print(f"source_zip sha256={source_sha256}", flush=True)
    with zipfile.ZipFile(source_zip, mode="r") as archive:
        splits = discover_zip_splits(archive)
        records = build_episode_records_from_zip(archive, splits)
        development_split = resolve_development_split(
            records,
            no_dev_split=args.no_dev_split,
            pipeline_validation=args.pipeline_validation,
            dev_fraction=args.dev_fraction,
            dev_seed=args.dev_seed,
        )
        actions_path = output_root / "actions.npy"
        actions, clipped_frames, observed_keys = write_actions_from_local_zip(
            archive, records, actions_path, args.action_key, args.flush_every
        )
    manifest = finalize_dataset(
        output_root=output_root,
        records=records,
        actions=actions,
        actions_path=actions_path,
        clipped_frames=clipped_frames,
        observed_keys=observed_keys,
        development_split=development_split,
        args=args,
        conversion_started=conversion_started,
        source_manifest={
            "path": str(source_zip),
            "sha256": source_sha256,
            "deleted_after_conversion": bool(args.delete_zip_after),
        },
        conversion_mode="local_zip_actions_only_no_full_unzip",
    )
    if args.delete_zip_after:
        source_zip.unlink()
        print(f"deleted source zip after recording sha256={source_sha256}", flush=True)
    return manifest


def convert_from_url(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    url = args.source_url
    conversion_started = time.perf_counter()
    print(f"[HEAD] {url}", flush=True)
    size = head_content_length(url)
    print(f"[HEAD] Content-Length={size} ({size / (1024**3):.2f} GiB); Range OK", flush=True)

    archive_name = Path(url.rstrip("/").split("/")[-1]).name
    if args.expected_sha256:
        expected_sha256 = args.expected_sha256.lower()
    else:
        print(f"[checksum] fetching {args.checksum_url}", flush=True)
        expected_sha256 = parse_official_sha256(fetch_text(args.checksum_url), archive_name)
    print(f"[checksum] expected sha256={expected_sha256}", flush=True)

    print("[CD] opening ZipFile over HttpRangeFile (EOCD from end)...", flush=True)
    range_file = HttpRangeFile(url, size)

    def on_request(start: int, end: int) -> None:
        if end - start > 1024 * 1024:
            print(f"[CD] Range bytes={start}-{end}", flush=True)

    range_file.on_request = on_request
    with zipfile.ZipFile(range_file, mode="r") as archive:
        splits = discover_zip_splits(archive)
        print("[meta] loading episode bounds / scene / language via Range...", flush=True)
        records = build_episode_records_from_zip(archive, splits)
    print(
        f"[meta] episodes={len(records)} frames={sum(r['n_frames'] for r in records)}",
        flush=True,
    )
    development_split = resolve_development_split(
        records,
        no_dev_split=args.no_dev_split,
        pipeline_validation=args.pipeline_validation,
        dev_fraction=args.dev_fraction,
        dev_seed=args.dev_seed,
    )

    actions_path = output_root / "actions.npy"
    checkpoint_path = output_root / CHECKPOINT_NAME
    actions, clipped_frames, observed_keys, stream_digest = write_actions_from_http_stream(
        url,
        size,
        records,
        actions_path,
        args.action_key,
        flush_every=args.flush_every,
        checkpoint_path=checkpoint_path,
        checkpoint_every_bytes=args.checkpoint_every_bytes,
        expected_sha256=expected_sha256,
    )
    return finalize_dataset(
        output_root=output_root,
        records=records,
        actions=actions,
        actions_path=actions_path,
        clipped_frames=clipped_frames,
        observed_keys=observed_keys,
        development_split=development_split,
        args=args,
        conversion_started=conversion_started,
        source_manifest={
            "url": url,
            "content_length": size,
            "sha256": stream_digest or expected_sha256,
            "sha256_source": (
                "stream_digest" if stream_digest else "official_sha256sum.txt"
            ),
            "expected_sha256": expected_sha256,
            "saved_on_disk": False,
        },
        conversion_mode="http_range_cd_plus_sequential_body_actions_only",
    )


def main() -> int:
    # Ensure tmux / redirected logs show progress immediately.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

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
    if not 0.0 < args.dev_fraction < 1.0:
        raise ValueError("--dev-fraction must be in (0, 1)")

    output_root = args.output_root.resolve()
    resume = (output_root / CHECKPOINT_NAME).exists()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite and not resume:
        raise FileExistsError(
            f"{output_root} is non-empty; pass --overwrite or resume from "
            f"{CHECKPOINT_NAME}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    if args.overwrite and not resume:
        for stale in output_root.glob("*"):
            if stale.is_file():
                stale.unlink()
    for stale in output_root.glob("indices_*.npy"):
        stale.unlink()
    incomplete = output_root / ".conversion_incomplete"
    incomplete.write_text("conversion in progress\n")

    if args.source_url:
        manifest = convert_from_url(args, output_root)
    else:
        manifest = convert_from_local_zip(args, output_root)

    incomplete.unlink(missing_ok=True)
    print(
        f"wrote {manifest['num_frames']} frames, {manifest['num_episodes']} episodes, "
        f"{manifest['num_chunks']} chunks to {output_root} "
        f"(actions-only; RGB discarded; mode={manifest['conversion_mode']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(
            "conversion failed; partial output retains .conversion_incomplete "
            "and may include .stream_checkpoint.json for resume",
            file=sys.stderr,
            flush=True,
        )
        raise
