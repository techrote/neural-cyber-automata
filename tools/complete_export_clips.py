#!/usr/bin/env python3
"""One declared zero-host completion of immutable T109 pose evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime-root", "launcher-root", "retained-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--render-python", required=True)
    args = parser.parse_args()
    from nca.bench.export_postprocess import run_completion
    result = run_completion(root=ROOT, declaration_path=ROOT / "configs/export/torch-cpu-clips-completion-v1.json",
        runtime_root=args.runtime_root, launcher_root=args.launcher_root, retained_root=args.retained_root,
        output=args.output, source_revision=args.source_revision, render_python=args.render_python, started=STARTED)
    print(json.dumps({key: result[key] for key in ("status", "composite_runtime_qualified", "work", "errors")}))
    return 0 if result["composite_runtime_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
