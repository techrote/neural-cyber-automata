#!/usr/bin/env python3
"""Verify retained Jolt evidence and compute the frozen authored teacher gate."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nca.contracts import canonical_json  # noqa: E402
from nca.evaluation.jolt_recovery_acceptance import audit_recovery  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-declaration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from nca.evaluation.durable_journal import publish_json
    if args.output.exists() or args.output.resolve().is_relative_to(args.evidence.resolve()):
        parser.error("use a new review path outside the sealed original evidence directory")
    try:
        result = audit_recovery(args.evidence, expected_revision=args.expected_revision,
                               expected_declaration_sha256=args.expected_declaration_sha256)
    except Exception as exc:
        result = {"schema": "nca-jolt-recovery-behavior-review/v1", "teacher_accepted": False,
                  "evidence_verification": {"verified": False},
                  "source_revision": args.expected_revision,
                  "declaration_sha256": args.expected_declaration_sha256,
                  "error": {"type": type(exc).__name__, "code": getattr(exc, "code", None)},
                  "learned_skill_qualified": False, "morphology_admitted": False}
    publish_json(args.output, result)
    print(canonical_json(result))
    return 0 if result["teacher_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
