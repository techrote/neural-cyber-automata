#!/usr/bin/env python3
"""Read-only original metadata bindings; no repository imports or execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    base = args.original_root / "jolt-learn-002-20261021"
    manifest = json.loads((base / "manifest.json").read_text())
    result = json.loads((base / "result.json").read_text())
    original_review = json.loads((args.original_root /
        "jolt-learn-002-training-postmortem-review-20261021.json").read_text())
    reservations = json.loads((base / "closed-budget-reservations.json").read_text())["reservations"]
    found_reservation_hash = sha(canonical(reservations))
    checks = {
        "configuration_matches_manifest": sha(canonical(json.loads((base / "config.json").read_text()))) == manifest["config_sha256"],
        "declaration_matches_manifest": sha(canonical(json.loads((base / "declaration.json").read_text()))) == manifest["declaration_sha256"],
        "inner_outer_declaration_bytes_identical": (base / "declaration.json").read_bytes() == (args.original_root / "jolt-execution-learn-002-20261021.json").read_bytes(),
        "result_binds_original_worker": sha((base / "worker-result.json").read_bytes()) == result["worker_result_sha256"],
        "result_binds_original_failed_readback": sha((base / "independent-readback.json").read_bytes()) == result["audit_sha256"],
        "closed_and_original_reservation_bytes_identical": (base / "budget-reservations.json").read_bytes() == (base / "closed-budget-reservations.json").read_bytes(),
        "reservation_count_matches_result": len(reservations) == result["costs"]["reservation_count"],
        "canonical_reservations_hash_matches_result": found_reservation_hash == result["costs"]["reservations_sha256"],
    }
    mismatches = []
    for name, expected in original_review["examined_artifact_identities"].items():
        path = base / name
        if path.stat().st_size != expected["bytes"] or sha(path.read_bytes()) != expected["sha256"]:
            mismatches.append(name)
    report = {
        "schema": "nca-learn-jolt-002-artifact-bindings/v1",
        "reviewer": "independent-archive-byte-audit",
        "reviewed_utc_date": "2026-10-21",
        "source_revision": manifest["source_revision"],
        "source_sha256": manifest["source_sha256"],
        "checks": checks,
        "checks_passed": all(checks.values()) and not mismatches,
        "reservation_count": len(reservations),
        "reservation_canonical_sha256": found_reservation_hash,
        "original_training_postmortem_bound_files": len(original_review["examined_artifact_identities"]),
        "original_training_postmortem_binding_mismatches": mismatches,
        "reproducer_sha256": sha(Path(__file__).read_bytes()),
        "scope": "Independent standard-library metadata identity checks only. No claim of complete owner receipts, physical correctness, tensor contents or qualification.",
        "original_status": result["status"],
        "original_qualified": result["qualification"]["qualified"],
        "review_operations": {"host_starts": 0, "physical_ticks": 0, "optimizer_calls": 0,
            "model_constructions": 0, "checkpoint_deserializations": 0, "neural_forwards": 0,
            "original_artifact_mutations": 0, "torch_imports": 0},
        "later_review_wall_seconds": time.monotonic() - start,
    }
    with args.output.open("xb") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"checks_passed": report["checks_passed"], "sha256": sha(args.output.read_bytes())}))


if __name__ == "__main__":
    main()

# © 2026 techrote. All rights reserved.
