#!/usr/bin/env python3
"""One declared Torch CPU qualification of the unchanged evaluated BC1103 policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--godot", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--render-python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from nca.bench.torch_screen import run_followup
    report = run_followup(root=ROOT, declaration_path=ROOT / "configs/export/torch-cpu-followup-v1.json",
        original_root=args.original_root, host_path=args.host, godot_path=args.godot, output=args.output,
        source_revision=args.source_revision, render_python=args.render_python)
    print(json.dumps({key: report[key] for key in ("status", "runtime_qualified", "work", "errors")}))
    return 0 if report["runtime_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
