#!/usr/bin/env python3
"""Stream a Freiburg CALVIN zip and materialize RGB+meta frames only.

Depth/tactile live *inside* episode_*.npz (not as separate zip members).
Filename --exclude patterns alone do not save disk. This tool:

1. Streams the zip over HTTP (Range + resume) — the archive is never saved.
2. Drops depth_* / *tactile* arrays from each episode npz before writing.
3. Skips zip members whose paths match depth/tactile.
4. Optionally verifies the full-body SHA-256 against sha256sum.txt.

Progress is printed every ``--log-every`` episode frames with flush=True.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from zip_http_stream import (  # noqa: E402
    ResumableHttpStream,
    fetch_text,
    head_content_length,
    iter_zip_local_members,
    parse_official_sha256,
)

EPISODE_RE = re.compile(
    r"^(?:.*/)?(?P<split>training|validation)/episode_(?P<frame>\d{7})\.npz$"
)
DROP_KEY_RE = re.compile(r"^(depth_.*|.*tactile.*)$")
DROP_MEMBER_RE = re.compile(r"(depth|tactile)", re.IGNORECASE)

KEEP_KEYS = (
    "rgb_static",
    "rgb_gripper",
    "actions",
    "rel_actions",
    "robot_obs",
    "scene_obs",
)

# Empirical RGB+meta payload from debug episode (~138 KiB/frame) + npz overhead.
BYTES_PER_FRAME_ESTIMATE = 150 * 1024
D_D_FRAMES_KNOWN = 611_099
D_D_ZIP_GIB = 165.20


def drop_member(filename: str) -> bool:
    base = filename.rsplit("/", 1)[-1]
    if base.endswith("/"):
        return False
    return bool(DROP_MEMBER_RE.search(base)) and not EPISODE_RE.match(filename)


def strip_episode_npz(raw: bytes) -> bytes:
    src = np.load(io.BytesIO(raw))
    out: dict[str, Any] = {}
    for key in src.files:
        if DROP_KEY_RE.match(key):
            continue
        if key in KEEP_KEYS or not (
            key.startswith("depth") or "tactile" in key or key.startswith("rgb_")
        ):
            # Keep protocol RGB + proprio/actions; also keep any unknown non-depth keys.
            if key.startswith("depth") or "tactile" in key:
                continue
            if key.startswith("rgb_") and key not in ("rgb_static", "rgb_gripper"):
                continue
            out[key] = src[key]
    if "rgb_static" not in out or "rgb_gripper" not in out:
        raise ValueError(f"episode npz missing required RGB keys; got {sorted(out)}")
    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    return buf.getvalue()


def estimate_frames(archive_name: str, zip_bytes: int) -> int:
    if archive_name == "task_D_D.zip":
        return D_D_FRAMES_KNOWN
    zip_gib = zip_bytes / (1024**3)
    return max(1, int(D_D_FRAMES_KNOWN * (zip_gib / D_D_ZIP_GIB)))


def preflight_disk(output_root: Path, archive_name: str, zip_bytes: int) -> int:
    frames = estimate_frames(archive_name, zip_bytes)
    need = int(frames * BYTES_PER_FRAME_ESTIMATE * 1.15) + 5 * 1024**3
    st = os.statvfs(output_root)
    free_bytes = st.f_bavail * st.f_frsize
    print(
        f"[preflight] archive={archive_name} zip={zip_bytes / 1024**3:.2f} GiB "
        f"est_frames={frames} est_rgb_on_disk={need / 1024**3:.1f} GiB "
        f"free={free_bytes / 1024**3:.1f} GiB",
        flush=True,
    )
    if free_bytes < need:
        raise SystemExit(
            f"ERROR: insufficient free space for RGB-only extract of {archive_name}. "
            f"Need ~{need / 1024**3:.0f} GiB, have {free_bytes / 1024**3:.0f} GiB. "
            "Depth/tactile stripping saves ~76% vs full npz but ABCD still needs ~320 GiB."
        )
    return frames


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "next_offset": 0,
            "frames_written": 0,
            "members_seen": 0,
            "members_skipped": 0,
            "bytes_written": 0,
        }
    return json.loads(path.read_text())


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-url", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--archive-name", required=True)
    p.add_argument("--expected-sha256", default="")
    p.add_argument("--checksum-url", default="")
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument("--state-file", type=Path, default=None)
    p.add_argument("--skip-disk-preflight", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.state_file or (output_root / ".stream_state.json")

    zip_bytes = head_content_length(args.source_url)
    expected = args.expected_sha256.lower().strip()
    if not expected and args.checksum_url:
        expected = parse_official_sha256(fetch_text(args.checksum_url), args.archive_name)
    if expected:
        print(f"[sha256] expected={expected}", flush=True)

    if not args.skip_disk_preflight:
        preflight_disk(output_root, args.archive_name, zip_bytes)

    state = load_state(state_path)
    start_offset = int(state.get("next_offset", 0))
    frames_written = int(state.get("frames_written", 0))
    members_seen = int(state.get("members_seen", 0))
    members_skipped = int(state.get("members_skipped", 0))
    bytes_written = int(state.get("bytes_written", 0))

    print(
        f"[start] url={args.source_url} -> {output_root} "
        f"resume_offset={start_offset}/{zip_bytes} frames_done={frames_written}",
        flush=True,
    )
    t0 = time.time()
    stream = ResumableHttpStream(
        args.source_url,
        zip_bytes,
        start_offset=start_offset,
        hash_stream=(start_offset == 0),
    )

    def should_materialize(filename: str) -> bool:
        if filename.endswith("/"):
            return False
        if drop_member(filename):
            return False
        # Always materialize episode npz (strip inside) and non-episode metadata.
        return True

    try:
        for member in iter_zip_local_members(
            stream,
            start_offset=start_offset,
            should_materialize=should_materialize,
        ):
            members_seen += 1
            name = member.filename
            if name.endswith("/") or not member.data:
                if drop_member(name) or name.endswith("/"):
                    members_skipped += 1
                state.update(
                    {
                        "next_offset": stream.bytes_consumed,
                        "frames_written": frames_written,
                        "members_seen": members_seen,
                        "members_skipped": members_skipped,
                        "bytes_written": bytes_written,
                    }
                )
                continue

            if drop_member(name):
                members_skipped += 1
                state["next_offset"] = stream.bytes_consumed
                continue

            dest = output_root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            payload = member.data
            ep = EPISODE_RE.match(name)
            if ep:
                payload = strip_episode_npz(member.data)
                frames_written += 1
                if frames_written % args.log_every == 0:
                    elapsed = max(time.time() - t0, 1e-6)
                    print(
                        f"[progress] frames={frames_written} members={members_seen} "
                        f"skipped={members_skipped} "
                        f"bytes_out={bytes_written / 1024**3:.2f}GiB "
                        f"stream={stream.bytes_consumed / 1024**3:.2f}/"
                        f"{zip_bytes / 1024**3:.2f}GiB "
                        f"rate={frames_written / elapsed:.1f} frames/s",
                        flush=True,
                    )

            dest.write_bytes(payload)
            bytes_written += len(payload)
            state.update(
                {
                    "next_offset": stream.bytes_consumed,
                    "frames_written": frames_written,
                    "members_seen": members_seen,
                    "members_skipped": members_skipped,
                    "bytes_written": bytes_written,
                }
            )
            if members_seen % 500 == 0:
                save_state(state_path, state)

        save_state(state_path, state)
        digest = stream.hexdigest()
        if expected and digest:
            if digest != expected:
                raise SystemExit(
                    f"ERROR: stream sha256 mismatch: got {digest} expected {expected}"
                )
            print(f"[sha256] OK {digest}", flush=True)
        elif expected and digest is None:
            print(
                "[sha256] WARN: mid-stream resume disabled local hash; "
                "rely on official sha256sum.txt provenance",
                flush=True,
            )
        manifest = {
            "archive_name": args.archive_name,
            "source_url": args.source_url,
            "expected_sha256": expected or None,
            "stream_sha256": digest,
            "frames_written": frames_written,
            "members_seen": members_seen,
            "members_skipped": members_skipped,
            "bytes_written": bytes_written,
            "modalities_kept": list(KEEP_KEYS),
            "modalities_dropped": ["depth_*", "*tactile*", "rgb_tactile"],
            "zip_saved_on_disk": False,
        }
        (output_root / "extract_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"[done] frames={frames_written} bytes_out={bytes_written / 1024**3:.2f} GiB "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    finally:
        stream.close()
        save_state(state_path, state)


if __name__ == "__main__":
    main()
