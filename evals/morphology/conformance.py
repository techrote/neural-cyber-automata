"""Run only the declared 32-request actual-profile interface diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from nca.contracts import canonical_json
from nca.controllers.morphology.conformance import run_conformance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--godot", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/morphology/comparison-v1.json"))
    parser.add_argument("--profiles", type=Path, default=Path("configs/morphology/profiles-v1.json"))
    parser.add_argument("--declaration", type=Path, default=Path("configs/morphology/entry-gate-v1.json"))
    parser.add_argument("--result", type=Path, default=Path("docs/evidence/t106-prerequisite-result.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--previous-summary", type=Path)
    args = parser.parse_args()
    result = run_conformance(root=args.source, checkout=args.host, godot=args.godot, config_path=args.config,
                             profiles_path=args.profiles, declaration_path=args.declaration,
                             result_path=args.result, output=args.output,
                             recovery_path=args.recovery, previous_summary_path=args.previous_summary)
    print(canonical_json({"status": result["status"], "budget": result["budget"], "sha256": result["sha256"],
                          "errors": result["errors"], "learned_transfer_measured": False}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
