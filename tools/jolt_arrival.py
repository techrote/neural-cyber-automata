#!/usr/bin/env python3
"""Run the single fixed Jolt teacher diagnostic from a published source freeze."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, default=os.environ.get("NCA_HOST_CHECKOUT"))
    parser.add_argument("--godot", type=Path, default=os.environ.get("NCA_GODOT"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    if not args.host or not args.godot:
        parser.error("configure NCA_HOST_CHECKOUT and NCA_GODOT or supply --host and --godot")
    from nca.evaluation.jolt_arrival import run_diagnostic
    result = run_diagnostic(root=ROOT, checkout=args.host, godot=args.godot, output=args.output,
        declaration_path=ROOT / "configs/baselines/jolt-arrival-diagnostic-v1.json",
        expected_revision=args.expected_revision)
    print(result["status"])
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
