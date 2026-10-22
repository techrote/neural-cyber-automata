#!/usr/bin/env python3
"""Compare source-bound Jolt diagnostics on a declared reference CPU.

This is *ballpark telemetry only*. It never changes execution admission,
trained policies, quality thresholds or original physical results.
© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nca.contracts import strict_json_loads
from nca.train.jolt_runtime_v3 import (
    capability_sha256, normalize_diagnostic_seconds,
)


def _read(path):
    path = Path(path)
    if (not path.is_file() or path.is_symlink() or
            path.stat().st_size > 4 * 1024 * 1024):
        raise ValueError("bounded original CPU diagnostic record required")
    return strict_json_loads(path.read_bytes(), max_bytes=4 * 1024 * 1024)


def compare(reference, observed, *, duration_seconds):
    if not isinstance(reference, dict) or not isinstance(observed, dict):
        raise ValueError("two original CPU reports are required")
    for record in (reference, observed):
        if (record.get("cpu_calibration") is None or
                not isinstance(record.get("runtime"), dict) or
                not isinstance(record.get("source_revision"), str) or
                len(record["source_revision"]) != 40):
            raise ValueError("source-bound runtime and calibration are mandatory")
    cap = capability_sha256(reference["runtime"])
    if capability_sha256(observed["runtime"]) != cap:
        raise ValueError("measured CPU software contracts differ")
    timing = normalize_diagnostic_seconds(duration_seconds,
        observed["cpu_calibration"], reference["cpu_calibration"])
    return {
        "schema": "nca-jolt-cpu-reference-comparison/v1",
        "scope": "ballpark only; synthetic or preflight diagnostic time",
        "reference_source_revision": reference["source_revision"],
        "observed_source_revision": observed["source_revision"],
        "software_capability_sha256": cap,
        "reference_cpu_model": reference["runtime"]["cpu_model"],
        "observed_cpu_model": observed["runtime"]["cpu_model"],
        "reference_kernel": reference["runtime"]["kernel"],
        "observed_kernel": observed["runtime"]["kernel"],
        "reference_calibration": reference["cpu_calibration"],
        "observed_calibration": observed["cpu_calibration"],
        "normalization": timing,
        "quality_thresholds_unchanged": True,
        "training_allocation_consumed": False,
        "experimental_admission_inferred": False,
        "not_a_hard_resource_bound": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--duration-seconds", required=True, type=float,
                        help="Original measured diagnostic duration on the observed CPU")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("never overwrite an original comparison receipt")
    value = compare(_read(args.reference), _read(args.observed),
                    duration_seconds=args.duration_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(original)
        stream.flush()
    print(json.dumps({"completed": True, "reference_cpu": value["reference_cpu_model"],
                      "observed_cpu": value["observed_cpu_model"],
                      "reference_seconds_approximate":
                        value["normalization"]["reference_seconds_approximate"],
                      "experiment_consumed":False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
