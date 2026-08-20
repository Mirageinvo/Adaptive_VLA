#!/usr/bin/env python3
"""Preflight capture for APB-RVQ runs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def _cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="artifacts/apb_rvq/preflight.json")
    ap.add_argument("--min-free-gb", type=float, default=50.0)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when disk guardrail fails.",
    )
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    usage = shutil.disk_usage(Path.home())
    free_gb = usage.free / (1024 ** 3)
    payload = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": _cmd(["python3", "-V"]),
        "git_head": _cmd(["git", "rev-parse", "HEAD"]),
        "cwd": os.getcwd(),
        "hf_home": os.environ.get("HF_HOME", ""),
        "disk_home_free_gb": round(free_gb, 2),
        "disk_guardrail_ok": bool(free_gb >= args.min_free_gb),
        "nvidia_smi": _cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader",
            ]
        ),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if args.strict and not payload["disk_guardrail_ok"]:
        raise SystemExit(
            f"Preflight failed: {payload['disk_home_free_gb']} GB free < {args.min_free_gb} GB required"
        )


if __name__ == "__main__":
    main()
