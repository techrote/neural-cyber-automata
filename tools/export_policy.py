#!/usr/bin/env python3
"""Convert an evaluated BC checkpoint, then run its frozen CPU export screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nca.export.build import convert, immutable_json, read_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    conversion = commands.add_parser("convert", help="verify complete T105 lineage and create immutable NumPy bundle")
    conversion.add_argument("--training-root", type=Path, required=True)
    conversion.add_argument("--accepted-t105-commit", required=True)
    conversion.add_argument("--profile", type=Path, default=ROOT / "configs/export/jev-profile-v1.json")
    conversion.add_argument("--config", type=Path, default=ROOT / "configs/export/cpu-bc-v1.json")
    conversion.add_argument("--output", type=Path, required=True)
    screen = commands.add_parser("screen", help="bounded real-host qualification, costs, failures and matched pose clips")
    screen.add_argument("--training-root", type=Path, required=True)
    screen.add_argument("--accepted-t105-commit", required=True)
    screen.add_argument("--bundle", type=Path, required=True)
    screen.add_argument("--sha256", required=True, help="expected bundle file-byte SHA-256 from conversion receipt")
    screen.add_argument("--config", type=Path, default=ROOT / "configs/export/cpu-bc-v1.json")
    screen.add_argument("--host", type=Path, default=os.environ.get("NCA_HOST_CHECKOUT"))
    screen.add_argument("--godot", type=Path, default=os.environ.get("NCA_GODOT"))
    screen.add_argument("--render-python", required=True, help="Python with NumPy and pinned Pillow for actual pose clips")
    screen.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "convert":
        profile, _ = read_json(args.profile)
        receipt = convert(args.training_root, profile, accepted_t105_commit=args.accepted_t105_commit,
                          output=args.output, screen_config=args.config)
        receipt_path = args.output.with_suffix(".conversion.json")
        immutable_json(receipt_path, receipt)
        print(json.dumps({key: receipt[key] for key in ("artifact", "artifact_sha256", "bundle_sha256", "runtime_qualified")}))
        return 0
    if not args.host or not args.godot:
        parser.error("--host and --godot or matching NCA environment variables are required")
    from nca.bench.deployment import run_screen
    result = run_screen(root=ROOT, training_root=args.training_root, bundle_path=args.bundle,
                        bundle_hash=args.sha256, config_path=args.config, host_path=args.host,
                        godot_path=args.godot, output=args.output,
                        accepted_t105_commit=args.accepted_t105_commit, render_python=args.render_python)
    print(json.dumps({key: result[key] for key in ("status", "runtime_qualified", "wall_seconds", "errors")}))
    return 0 if result["status"] == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
