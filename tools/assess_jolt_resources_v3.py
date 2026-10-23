#!/usr/bin/env python3
"""Prepare a source-bound *nonadmitted* LEARN-JOLT-003 resource assessment v2.

The original CPU/native-handshake memory report must match the complete clean
source, frozen configuration and pinned software/host identities. Static
serialization and disk bounds are independently checked. The report cannot
assert a full worker peak or invent independent reviewer approval.
No physical Jolt capture, training or allocation is opened.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nca.contracts import sha256_json, strict_json_loads, ContractError
from nca.train.jolt_admission_v3 import validate_config_v3, validate_resource_admission
from nca.train.jolt_execution_v3 import COUNTERS
from nca.train.jolt_memory_watchdog_v3 import watchdog_policy_v3, SOFT_THRESHOLD_BYTES
from nca.train.jolt_runtime_v3 import capability_sha256
from nca.train.jolt_screen import source_inventory, runtime_identity
from nca.train.jolt_storage_v3 import storage_bounds_v3

SCHEMA = "nca-jolt-003-resource-assessment-draft/v2"
ALLOCATION = "learn-jolt-003-20261022-a1"
PLANNING_MULTIPLIER = 4


def _load(path, *, max_bytes=1024 * 1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ContractError("jolt_resource_input", "bounded original regular JSON is required")
    return strict_json_loads(path.read_bytes(), max_bytes=max_bytes)


def _sha_bytes(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _need(condition, detail):
    if not condition:
        raise ContractError("jolt_resource_assessment", detail)


def assess(*, root, config_path, measured_report_path, output_parent):
    """Independent readback of ORIGINAL nonexperimental memory evidence."""
    root, output_parent = Path(root), Path(output_parent)
    config = _load(config_path)
    validate_config_v3(root, config)
    _need(config["allocation_id"] == ALLOCATION, "unknown experiment allocation")
    _need(config["budget"]["maximum_wall_seconds"] is None,
          "risk-based resource review requires explicit unlimited clock")
    _need(len(config["budget"]["counters"]) == len(COUNTERS) == 39,
          "operation ceilings changed")
    measured = _load(measured_report_path, max_bytes=4*1024*1024)
    _need(measured.get("schema") == "nca-jolt-003-memory-diagnostic/v1"
          and measured.get("admission", {}).get("accepted") is False
          and measured.get("physical_experiment_started") is False
          and measured.get("allocation_consumed") is False
          and measured.get("native_handshake_only") is True
          and measured.get("sealed_case_inspection_or_training") is False,
          "memory evidence must be an original nonexperimental specimen")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    _need(len(revision) == 40, "exact executable SHA is required")
    sources = source_inventory(root)
    source_sha256 = sha256_json(sources)
    config_sha256 = sha256_json(config)
    _need(measured["source_revision"] == revision
          and measured["source_files"] == sources
          and measured["source_sha256"] == source_sha256
          and measured["config_sha256"] == config_sha256,
          "original memory measurement does not bind this source/configuration")
    exact_runtime = runtime_identity()
    _need(capability_sha256(exact_runtime) == capability_sha256(measured["runtime"])
          and measured["runtime_software_contract_sha256"] == capability_sha256(exact_runtime),
          "pinned CPU software contract differs from the measurement")
    bounds = storage_bounds_v3()
    _need(measured["storage_bounds"] == bounds, "original source-derived storage bound differs")
    _need(measured["host_identity"]["host_revision"] == config["host_revision"],
          "nonexperimental measurement used another pinned native host")
    memory = measured["memory"]
    measured_peak = memory.get("maximum_overlapping_worker_plus_native_rss_bytes")
    _need(type(measured_peak) is int and 0 < measured_peak < SOFT_THRESHOLD_BYTES
          and type(memory.get("native_samples")) is int
          and memory["native_samples"] > 0,
          "complete positive native/worker memory sample is unavailable")
    _need(type(memory.get("worker_ru_maxrss_bytes")) is int
          and memory["worker_ru_maxrss_bytes"] > 0,
          "original worker high-water observation is missing")
    _need(measured["synthetic_shape"]["results"] == 6464
          and measured["synthetic_shape"]["ppo_proxy_decisions"] == 256
          and measured["synthetic_shape"]["actual_training_or_benchmark_performed"] is False,
          "synthetic retention inventory is incomplete or mislabeled")
    disk_free = shutil.disk_usage(output_parent).free
    disk_required = bounds["required_disk_bytes_including_1_gib_reserve"]
    _need(type(disk_free) is int and disk_free > 0, "disk sample unavailable")
    observed_disk_sufficient = disk_free >= disk_required
    fourfold = PLANNING_MULTIPLIER * measured_peak
    # The watched 8 GiB limit is an early-abort sampled process-tree RSS
    # threshold, not a proven hard memory upper bound. Keep the memory gate
    # **false**, even if the heuristic margin looks exceptionally large.
    gates = {
        "canonical_reservation_maximum": True, "reader_limits": True,
        "journal_aggregate_bytes": True, "total_artifact_bytes": True,
        "total_file_count": True, "peak_resident_memory": False,
        "available_disk": observed_disk_sufficient,
        "original_clock_phase_enforcement": None,
        "complete_pipeline_time_feasibility": None,
    }
    unresolved = [
        {"id": "RESOURCE-MEM-01",
         "finding": "Observed peak combines synthetic worker retention and native capability handshake, not complete stepped-native/validated-PPO/independent-auditor lifecycle. A 4x multiplier is planning, not a hard bound."},
        {"id": "RESOURCE-MEM-02",
         "finding": "The 8 GiB watchdog samples process-tree RSS and fails closed; subinterval allocations and kernel OOM can precede detection. It is not a kernel-enforced hard memory limit."},
        {"id": "RESOURCE-REVIEWS-01",
         "finding": "Two independently authored accepted v2 final-source review records and their prospective_binding_sha256 are absent."},
        {"id": "RESOURCE-PROSPECTIVE-01",
         "finding": "A passing clean-source declaration and execution-runner free-disk recheck have not been independently published and read back."},
    ]
    if not observed_disk_sufficient:
        unresolved.append({"id": "RESOURCE-DISK-01", "finding": "Current sampled runner free disk is below the source-derived complete artifact + 1 GiB reserve."})
    record = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "allocation_id": ALLOCATION,
        "source_revision": revision, "source_sha256": source_sha256,
        "source_file_count": len(sources), "config_sha256": config_sha256,
        "host_identity_sha256": sha256_json(measured["host_identity"]),
        "runtime_contract_sha256": capability_sha256(exact_runtime),
        "original_cpu_hardware": measured["runtime"],
        "observed_runtime_in_assessment": exact_runtime,
        "original_report_sha256": _sha_bytes(measured_report_path),
        "original_report_scope": "synthetic GRU32 retention and pinned idle native handshake only",
        "serialization": bounds,
        "operation_counter_ceiling_sha256": sha256_json(COUNTERS),
        "memory_watchdog": watchdog_policy_v3(),
        "memory": {
            "sampled_original_combined_rss_bytes": measured_peak,
            "sampled_original_worker_highwater_rss_bytes": memory["worker_ru_maxrss_bytes"],
            "sampled_native_sample_count": memory["native_samples"],
            "planning_multiplier": PLANNING_MULTIPLIER,
            "illustrative_fourfold_rss_bytes": fourfold,
            "fourfold_below_soft_threshold": fourfold < SOFT_THRESHOLD_BYTES,
            "sampled_peak_to_soft_threshold_ratio": measured_peak / SOFT_THRESHOLD_BYTES,
            "kernel_memory_bound_proven": False,
            "complete_actual_worker_peak_observed": False,
        },
        "disk": {
            "sampled_free_bytes": disk_free,
            "required_free_disk_bytes": disk_required,
            "sampled_sufficient": observed_disk_sufficient,
            "must_recheck_on_execution_runner": True,
        },
        "gates": gates,
        "accepted": False, "offline_scope_only": True,
        "prospective_experiment_admitted": False,
        "unresolved_blocking_findings": unresolved,
        "independent_admission_reviews": [],
        "execution_declaration": None,
        "physical_host_session_for_training_started": False,
        "experiment_allocation_consumed": False,
        "single_attempt_claim_published": False,
        "full_peak_memory_claimed": False,
        "note": "Prepare a v2 resource proposal for independent risk assessment; this draft must fail validate_resource_admission. A full worker/native peak is not inferred from an idle native handshake.",
    }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/learning/jolt-learn-003-v2.json")
    parser.add_argument("--memory-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("resource assessment is immutable; never overwrite")
    result = assess(root=ROOT, config_path=args.config,
                    measured_report_path=args.memory_report,
                    output_parent=args.output.parent)
    # Prevent accidental promotion of a nonadmitted resource draft.
    try:
        validate_resource_admission({"resource_admission": result})
    except (ContractError, ValueError):
        pass
    else:
        raise RuntimeError("negative resource assessment was unexpectedly admitted")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, sort_keys=True, indent=2, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    print(json.dumps({"schema": SCHEMA, "status": "not_admitted",
        "source_revision": result["source_revision"],
        "sampled_peak_bytes": result["memory"]["sampled_original_combined_rss_bytes"],
        "fourfold_illustrative_bytes": result["memory"]["illustrative_fourfold_rss_bytes"],
        "watchdog_soft_threshold_bytes": SOFT_THRESHOLD_BYTES,
        "disk_sample_sufficient": result["disk"]["sampled_sufficient"],
        "allocation_consumed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
