#!/usr/bin/env python3
"""Prepare or execute the separately gated fixed morphology comparison."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run"))
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--prerequisite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--declaration", type=Path)
    parser.add_argument("--host", type=Path)
    parser.add_argument("--godot", type=Path)
    args = parser.parse_args()
    from nca.contracts import canonical_json
    from nca.train.morphology_screen import prepare_continuation, run_comparison
    if args.operation == "prepare":
        result = prepare_continuation(root=args.source, prerequisite_directory=args.prerequisite, output=args.output)
        print(canonical_json(result))
        return 0 if result["status"] == "ready_for_public_declaration" else 2
    if args.declaration is None or args.host is None or args.godot is None:
        parser.error("run requires a publicly committed --declaration, --host and --godot")
    result = run_comparison(root=args.source, prerequisite_directory=args.prerequisite,
        declaration_path=args.declaration, checkout=args.host, godot=args.godot, output=args.output)
    print(canonical_json({name: result[name] for name in ("status", "phase", "sha256", "error",
        "attempted_game_requests", "recorded_game_requests", "wall_seconds", "evaluation")}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
