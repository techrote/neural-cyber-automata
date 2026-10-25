#!/usr/bin/env python3
"""Prepare the 4LY-004 admission packet without executing the experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nca.contracts import canonical_json, sha256_json, strict_json_loads  # noqa: E402
from nca.train.flight_admission import (  # noqa: E402
    RUNTIME_THREADS,
    admission_candidate,
    load_split_manifest,
    runtime_identity,
    source_binding,
)


def _load_optional(path: Path | None):
    if path is None:
        return None
    return strict_json_loads(path.read_bytes(), max_bytes=1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--custody-record", type=Path)
    parser.add_argument("--independent-review", type=Path)
    parser.add_argument(
        "--require-admitted",
        action="store_true",
        help="Return nonzero unless all external custody/review gates are present.",
    )
    args = parser.parse_args()

    root = args.repository_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_num_threads(RUNTIME_THREADS)
    runtime = runtime_identity(enforce=True)
    source = source_binding(root)
    development, development_sha = load_split_manifest(
        root / "configs/flight/4ly004-development-v1.json",
        split="development",
    )
    tuning, tuning_sha = load_split_manifest(
        root / "configs/flight/4ly004-tuning-v1.json",
        split="tuning",
    )
    custody = _load_optional(args.custody_record)
    review = _load_optional(args.independent_review)
    packet = admission_candidate(
        source=source,
        development_sha256=development_sha,
        tuning_sha256=tuning_sha,
        custody=custody,
        independent_review=review,
    )
    packet["observed_runtime"] = runtime
    packet.pop("packet_sha256", None)
    packet["packet_sha256"] = sha256_json(packet)

    (output / "source-binding.json").write_text(
        canonical_json(source) + "\n", encoding="utf-8"
    )
    (output / "development-manifest.json").write_text(
        canonical_json(development) + "\n", encoding="utf-8"
    )
    (output / "tuning-manifest.json").write_text(
        canonical_json(tuning) + "\n", encoding="utf-8"
    )
    (output / "admission-packet.json").write_text(
        canonical_json(packet) + "\n", encoding="utf-8"
    )

    summary = {
        "status": packet["status"],
        "admitted": packet["admitted"],
        "learner_revision": source["learner_revision"],
        "learner_source_sha256": source["learner_source_sha256"],
        "development_sha256": development_sha,
        "tuning_sha256": tuning_sha,
        "packet_sha256": packet["packet_sha256"],
        "reasons": packet["reasons"],
        "training_executed": False,
        "final_access": False,
    }
    print(json.dumps(summary, sort_keys=True))
    if args.require_admitted and not packet["admitted"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
