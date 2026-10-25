"""4LY-004 prospective experiment-admission packaging and resource enforcement.

This module prepares source/runtime/split/resource/custody bindings for issue #74.
It does not collect teacher data, optimize a policy, open a final payload, or
grant execution by itself. A separately authored independent review must bind
the complete review candidate before an executable declaration can pass.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping

from nca.contracts import ContractError, canonical_json, sha256_json, strict_json_loads
from nca.train.flight_accounting import (
    ACCEPTED_DEPENDENCY_EVIDENCE,
    DESIGN_WORK_MAXIMA,
    DEVELOPMENT_SEEDS,
    HOST_BUILD_SHA256,
    HOST_CONTRACT_HEAD,
    RESOURCE_ENFORCEMENT_EVIDENCE_SHA256,
    SCENARIO_FAMILY,
    SCENARIO_MAX_TICKS,
    STAGE_ACCOUNTING_EVIDENCE_SHA256,
    TEACHER_FINAL_SUITE_SHA256,
    TUNING_SEEDS,
    FlightDeclaration,
    execution_gate,
    preparation_declaration,
)
from nca.train.flight_learning import (
    ACTION_SHA256,
    BC_SCHEDULE_SHA256,
    MAX_CHECKPOINT_BYTES,
    PREPROCESSING_SHA256,
)

ADMISSION_PACKET_SCHEMA = "nca-4ly004-admission-packet/v1"
SOURCE_BINDING_SCHEMA = "nca-4ly004-source-binding/v1"
SPLIT_MANIFEST_SCHEMA = "nca-4ly004-split-manifest/v1"
CUSTODY_SCHEMA = "nca-4ly004-final-custody/v1"
FINAL_SUITE_SCHEMA = "nca-4ly004-learner-final-suite/v1"
REVIEW_SCHEMA = "nca-4ly004-independent-execution-review/v1"
RESOURCE_POLICY_SCHEMA = "nca-4ly004-resource-policy/v1"

RUNTIME_PLATFORM = "github-actions-ubuntu-24.04-x86_64"
RUNTIME_PYTHON = "3.12.10"
RUNTIME_TORCH = "2.8.0+cpu"
RUNTIME_THREADS = 1

# Finite fail-stop allocation ceilings, not predictions of required use.
# Wall time is >2x the slowest retained qualified bridge-only projection for
# the complete physical-tick ceiling. Storage/publication cover the worst-case
# 63 x 32 MiB checkpoint bound plus retained evidence. RSS is a hard process-
# tree cap, not a claimed learner peak.
MAX_WALL_SECONDS = 18_000.0
MAX_RESIDENT_BYTES = 4 * 1024**3
MAX_STORAGE_BYTES = 8 * 1024**3
MAX_PUBLICATION_BYTES = 4 * 1024**3
RESOURCE_SAMPLE_INTERVAL_SECONDS = 0.25
QUALIFIED_SLOW_TICKS_PER_SECOND = 166.3225

SOURCE_CLOSURE_PATHS = (
    "src/nca/adapters/flight.py",
    "src/nca/adapters/flight_transport.py",
    "src/nca/adapters/flight_host_bridge.gd",
    "src/nca/train/flight.py",
    "src/nca/train/flight_accounting.py",
    "src/nca/train/flight_learning.py",
    "src/nca/train/flight_admission.py",
    "tools/flight_admission.py",
    ".github/workflows/flight-admission.yml",
    "configs/flight/4ly004-development-v1.json",
    "configs/flight/4ly004-tuning-v1.json",
)

FINAL_COMPOSITION = {
    "free_space": 12,
    "reversal": 12,
    "arrival": 12,
    "obstacles": 12,
    "recovery": 12,
}
FINAL_SCENARIO_COUNTS = {
    "horizontal_reach": 4,
    "vertical_reach": 4,
    "diagonal_reach": 4,
    "reverse": 12,
    "arrival_braking": 12,
    "wide_corridor": 3,
    "offset_gates": 3,
    "corner": 3,
    "slalom": 3,
    "wall_recovery": 12,
}
FINAL_EVALUATOR_CONTRACT = {
    "schema": "nca-4ly004-final-evaluator-contract/v1",
    "host_revision": HOST_CONTRACT_HEAD,
    "episode_schema": "4ly-episode/v1",
    "observation_schema": "4ly-observation/v1",
    "action_schema": "4ly-action/v1",
    "course_id": "4ly-flight-course/v1",
    "course_version": 1,
    "split_geometry_version": 3,
    "arrival": "accepted_host_low_speed_36_tick_dwell",
    "recovery": "accepted_host_contact_witness_then_low_speed_arrival",
    "family_weighting": "equal_macro_across_five_core_families",
    "comparators": (
        "qualified_scripted_teacher",
        "zero_velocity_neutral",
        "canonical_untrained_seed74011_epoch0",
        "trained_seed74011_selected_by_tuning",
        "trained_seed74013_selected_by_tuning",
        "trained_seed74017_selected_by_tuning",
    ),
}
FINAL_EVALUATOR_CONTRACT_SHA256 = sha256_json(FINAL_EVALUATOR_CONTRACT)

RESOURCE_POLICY = {
    "schema": RESOURCE_POLICY_SCHEMA,
    "maximum_wall_seconds": MAX_WALL_SECONDS,
    "maximum_resident_bytes": MAX_RESIDENT_BYTES,
    "maximum_storage_bytes": MAX_STORAGE_BYTES,
    "maximum_publication_bytes": MAX_PUBLICATION_BYTES,
    "sample_interval_seconds": RESOURCE_SAMPLE_INTERVAL_SECONDS,
    "whole_attempt": True,
    "process_tree_rss": "linux_procfs_root_plus_descendants",
    "storage_scope": "attempt_root_regular_files_no_symlink_follow",
    "publication_scope": "explicit_prepublication_byte_charge",
    "violation": "fail_stop_no_new_work",
    "checkpoint_max_bytes": MAX_CHECKPOINT_BYTES,
    "checkpoint_publication_ceiling": DESIGN_WORK_MAXIMA["checkpoint_publications"],
    "qualified_bridge_floor_ticks_per_second": QUALIFIED_SLOW_TICKS_PER_SECOND,
    "physical_tick_ceiling": DESIGN_WORK_MAXIMA["physical_ticks"],
    "physical_only_projection_seconds": (
        DESIGN_WORK_MAXIMA["physical_ticks"] / QUALIFIED_SLOW_TICKS_PER_SECOND
    ),
}
RESOURCE_POLICY_SHA256 = sha256_json(RESOURCE_POLICY)


def _fail(code: str, detail: str) -> None:
    raise ContractError(code, detail)


def _digest(value: Any, name: str) -> str:
    if type(value) is not str or not re.fullmatch(r"[a-f0-9]{64}", value):
        _fail("flight_admission_digest", f"{name} must be a SHA-256 digest")
    return value


def _revision(value: Any, name: str) -> str:
    if type(value) is not str or not re.fullmatch(r"[a-f0-9]{40}", value):
        _fail("flight_admission_revision", f"{name} must be a full Git revision")
    return value


def _exact(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        _fail("flight_admission_shape", f"{name} must contain exactly {sorted(keys)}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments], text=True, timeout=30
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError(
            "flight_source_git", "cannot establish clean learner source"
        ) from exc


def source_binding(root: str | Path) -> dict[str, Any]:
    """Bind the exact clean executable flight source without opening any data."""
    root = Path(root).resolve()
    revision = _revision(_git(root, "rev-parse", "HEAD"), "learner_revision")
    status = _git(root, "status", "--porcelain", "--", *SOURCE_CLOSURE_PATHS)
    if status:
        _fail("flight_source_dirty", "flight source closure is not clean")
    files: dict[str, str] = {}
    for relative in SOURCE_CLOSURE_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            _fail(
                "flight_source_file",
                f"source closure member is absent or unsafe: {relative}",
            )
        files[relative] = _sha256_file(path)
    identity = {
        "schema": SOURCE_BINDING_SCHEMA,
        "learner_revision": revision,
        "files": files,
        "preprocessing_sha256": PREPROCESSING_SHA256,
        "action_sha256": ACTION_SHA256,
        "bc_schedule_sha256": BC_SCHEDULE_SHA256,
    }
    return {**identity, "learner_source_sha256": sha256_json(identity)}


def build_split_manifest(split: str) -> dict[str, Any]:
    if split == "development":
        seeds = DEVELOPMENT_SEEDS
        scenarios = tuple(SCENARIO_MAX_TICKS)
    elif split == "tuning":
        seeds = TUNING_SEEDS
        scenarios = tuple(
            scenario
            for scenario in SCENARIO_MAX_TICKS
            if scenario != "unsupported_hover"
        )
    else:
        _fail(
            "flight_split_manifest",
            "only public development/tuning manifests may be built",
        )
    cases = []
    for scenario in sorted(scenarios):
        for seed in seeds:
            ticks = SCENARIO_MAX_TICKS[scenario]
            cases.append(
                {
                    "scenario_id": scenario,
                    "family": SCENARIO_FAMILY[scenario],
                    "seed": seed,
                    "max_ticks": ticks,
                    "max_decisions": ticks // 6,
                }
            )
    return {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "host_revision": HOST_CONTRACT_HEAD,
        "host_build_sha256": HOST_BUILD_SHA256,
        "course_id": "4ly-flight-course/v1",
        "course_version": 1,
        "split_geometry_version": 3,
        "split": split,
        "case_count": len(cases),
        "max_ticks": sum(item["max_ticks"] for item in cases),
        "max_decisions": sum(item["max_decisions"] for item in cases),
        "cases": cases,
    }


def validate_split_manifest(
    value: Mapping[str, Any], *, split: str
) -> dict[str, Any]:
    expected = build_split_manifest(split)
    candidate = strict_json_loads(canonical_json(value), max_bytes=1024 * 1024)
    if candidate != expected:
        _fail(
            "flight_split_manifest",
            f"{split} manifest differs from the frozen public case set",
        )
    return expected


def load_split_manifest(
    path: str | Path, *, split: str
) -> tuple[dict[str, Any], str]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        _fail(
            "flight_split_manifest",
            f"{split} manifest must be one regular file",
        )
    value = strict_json_loads(path.read_bytes(), max_bytes=1024 * 1024)
    manifest = validate_split_manifest(value, split=split)
    return manifest, sha256_json(manifest)


def _os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw or raw.startswith("#"):
            continue
        key, value = raw.split("=", 1)
        result[key] = value.strip().strip('"')
    return result


def _cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return "unavailable"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("model name") and ":" in line:
            value = line.split(":", 1)[1].strip()
            if value:
                return value[:256]
    return "unavailable"


def runtime_identity(*, enforce: bool = True) -> dict[str, Any]:
    """Return the declared CPU runtime; optionally refuse any runtime drift."""
    release = _os_release()
    observed = {
        "device": "cpu",
        "platform": RUNTIME_PLATFORM,
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": None,
        "threads": None,
        "os_id": release.get("ID"),
        "os_version_id": release.get("VERSION_ID"),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
    }
    try:
        import torch
    except ImportError as exc:
        raise ContractError(
            "flight_runtime", "pinned CPU Torch runtime is required"
        ) from exc
    observed["torch"] = str(torch.__version__)
    observed["threads"] = int(torch.get_num_threads())
    if enforce:
        if (
            not sys.platform.startswith("linux")
            or platform.machine().lower() not in {"x86_64", "amd64"}
            or observed["os_id"] != "ubuntu"
            or observed["os_version_id"] != "24.04"
        ):
            _fail(
                "flight_runtime",
                "4LY-004 admitted runtime is Ubuntu 24.04 Linux x86_64 only",
            )
        if (
            observed["python"] != RUNTIME_PYTHON
            or observed["torch"] != RUNTIME_TORCH
        ):
            _fail(
                "flight_runtime",
                "Python/Torch runtime differs from the frozen admission contract",
            )
        if observed["threads"] != RUNTIME_THREADS:
            _fail(
                "flight_runtime",
                "Torch thread count differs from the one-thread admission contract",
            )
        if bool(torch.cuda.is_available()):
            _fail("flight_runtime", "4LY-004 admission is CPU only")
    return observed


def validate_custody_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate metadata only; this function never accepts or opens suite bytes."""
    record = dict(
        _exact(
            value,
            {
                "schema",
                "suite_schema",
                "suite_sha256",
                "suite_bytes",
                "case_count",
                "family_composition",
                "scenario_composition",
                "host_revision",
                "host_build_sha256",
                "evaluator_contract_sha256",
                "teacher_final_suite_sha256",
                "teacher_final_reused",
                "custodian_identity",
                "reviewer_identity",
                "custodian_independent",
                "reviewer_independent",
                "repository_payload_present",
                "executor_opened_payload",
                "private_disjointness_verified",
                "private_feasibility_verified",
            },
            "custody",
        )
    )
    if (
        record["schema"] != CUSTODY_SCHEMA
        or record["suite_schema"] != FINAL_SUITE_SCHEMA
    ):
        _fail(
            "flight_final_custody",
            "unknown learner-final custody/suite schema",
        )
    _digest(record["suite_sha256"], "custody.suite_sha256")
    if type(record["suite_bytes"]) is not int or record["suite_bytes"] <= 0:
        _fail(
            "flight_final_custody",
            "custody suite_bytes must be positive",
        )
    if record["case_count"] != 60:
        _fail(
            "flight_final_custody",
            "learner final must contain exactly 60 cases",
        )
    if record["family_composition"] != FINAL_COMPOSITION:
        _fail(
            "flight_final_custody",
            "learner-final family composition changed",
        )
    if record["scenario_composition"] != FINAL_SCENARIO_COUNTS:
        _fail(
            "flight_final_custody",
            "learner-final scenario composition changed",
        )
    if (
        record["host_revision"] != HOST_CONTRACT_HEAD
        or record["host_build_sha256"] != HOST_BUILD_SHA256
    ):
        _fail(
            "flight_final_custody",
            "learner-final custody is bound to another host",
        )
    if (
        record["evaluator_contract_sha256"]
        != FINAL_EVALUATOR_CONTRACT_SHA256
    ):
        _fail(
            "flight_final_custody",
            "learner-final evaluator contract changed",
        )
    if (
        record["teacher_final_suite_sha256"] != TEACHER_FINAL_SUITE_SHA256
        or record["teacher_final_reused"] is not False
    ):
        _fail(
            "flight_final_custody",
            "teacher final cannot be reused as learner final",
        )
    for name in ("custodian_identity", "reviewer_identity"):
        if (
            type(record[name]) is not str
            or not record[name]
            or len(record[name]) > 256
        ):
            _fail(
                "flight_final_custody",
                f"{name} must identify the external role",
            )
    if record["custodian_identity"] == record["reviewer_identity"]:
        _fail(
            "flight_final_custody",
            "custodian and private reviewer must be distinct roles",
        )
    required_true = (
        "custodian_independent",
        "reviewer_independent",
        "private_disjointness_verified",
        "private_feasibility_verified",
    )
    if any(record[name] is not True for name in required_true):
        _fail(
            "flight_final_custody",
            "independent custody/disjointness/feasibility attestations are required",
        )
    if (
        record["repository_payload_present"] is not False
        or record["executor_opened_payload"] is not False
    ):
        _fail(
            "flight_final_custody",
            "learner-final payload must remain outside Git and unopened by executor",
        )
    return strict_json_loads(canonical_json(record))


def custody_record_sha256(record: Mapping[str, Any]) -> str:
    return sha256_json(validate_custody_record(record))


def resource_ceilings() -> dict[str, Any]:
    return {
        "maximum_wall_seconds": MAX_WALL_SECONDS,
        "maximum_resident_bytes": MAX_RESIDENT_BYTES,
        "maximum_storage_bytes": MAX_STORAGE_BYTES,
        "maximum_publication_bytes": MAX_PUBLICATION_BYTES,
    }


def _base_prospective_value(
    *,
    source: Mapping[str, Any],
    development_sha256: str,
    tuning_sha256: str,
    custody: Mapping[str, Any],
) -> dict[str, Any]:
    source = dict(source)
    _revision(source.get("learner_revision"), "source.learner_revision")
    _digest(
        source.get("learner_source_sha256"),
        "source.learner_source_sha256",
    )
    _digest(development_sha256, "development_sha256")
    _digest(tuning_sha256, "tuning_sha256")
    custody_value = validate_custody_record(custody)
    value = preparation_declaration().to_dict()
    value["status"] = "prospective"
    value["identities"]["learner_revision"] = source["learner_revision"]
    value["identities"]["learner_source_sha256"] = source[
        "learner_source_sha256"
    ]
    value["splits"].update(
        {
            "development_sha256": development_sha256,
            "tuning_sha256": tuning_sha256,
            "final_commitment_sha256": custody_value["suite_sha256"],
            "final_custody_sha256": sha256_json(custody_value),
            "opened_final": False,
        }
    )
    value["runtime"] = {
        "device": "cpu",
        "platform": RUNTIME_PLATFORM,
        "python": RUNTIME_PYTHON,
        "torch": RUNTIME_TORCH,
        "threads": RUNTIME_THREADS,
    }
    value["resource_ceilings"] = resource_ceilings()
    value["dependencies"]["resource_enforcement"] = {
        "status": "accepted",
        "evidence_sha256": RESOURCE_ENFORCEMENT_EVIDENCE_SHA256,
    }
    value["dependencies"]["stage_accounting_integration"] = {
        "status": "accepted",
        "evidence_sha256": STAGE_ACCOUNTING_EVIDENCE_SHA256,
    }
    value["dependencies"]["independent_execution_review"] = {
        "status": "pending",
        "evidence_sha256": None,
    }
    return value


def review_binding(
    *,
    source: Mapping[str, Any],
    development_sha256: str,
    tuning_sha256: str,
    custody: Mapping[str, Any],
) -> dict[str, Any]:
    value = _base_prospective_value(
        source=source,
        development_sha256=development_sha256,
        tuning_sha256=tuning_sha256,
        custody=custody,
    )
    declaration = FlightDeclaration.from_mapping(value)
    binding = {
        "schema": "nca-4ly004-execution-review-binding/v1",
        "declaration_without_review_sha256": declaration.sha256,
        "learner_revision": value["identities"]["learner_revision"],
        "learner_source_sha256": value["identities"][
            "learner_source_sha256"
        ],
        "development_sha256": value["splits"]["development_sha256"],
        "tuning_sha256": value["splits"]["tuning_sha256"],
        "final_commitment_sha256": value["splits"][
            "final_commitment_sha256"
        ],
        "final_custody_sha256": value["splits"]["final_custody_sha256"],
        "host_revision": HOST_CONTRACT_HEAD,
        "host_build_sha256": HOST_BUILD_SHA256,
        "runtime": deepcopy(value["runtime"]),
        "resource_policy_sha256": RESOURCE_POLICY_SHA256,
        "resource_enforcement_evidence_sha256": (
            RESOURCE_ENFORCEMENT_EVIDENCE_SHA256
        ),
        "stage_accounting_evidence_sha256": (
            STAGE_ACCOUNTING_EVIDENCE_SHA256
        ),
        "final_access": False,
    }
    return {**binding, "review_binding_sha256": sha256_json(binding)}


def validate_review_record(
    value: Mapping[str, Any], *, binding: Mapping[str, Any]
) -> dict[str, Any]:
    record = dict(
        _exact(
            value,
            {
                "schema",
                "review_binding_sha256",
                "accepted",
                "reviewer_identity",
                "independent",
                "source_clean",
                "runtime_accepted",
                "resource_policy_accepted",
                "stage_accounting_accepted",
                "custody_accepted",
                "final_access_unopened",
                "training_not_executed",
                "findings",
            },
            "independent_review",
        )
    )
    if record["schema"] != REVIEW_SCHEMA:
        _fail(
            "flight_execution_review",
            "unknown execution-review schema",
        )
    if (
        record["review_binding_sha256"]
        != binding["review_binding_sha256"]
    ):
        _fail(
            "flight_execution_review",
            "review is bound to another prospective package",
        )
    if (
        type(record["reviewer_identity"]) is not str
        or not record["reviewer_identity"]
    ):
        _fail(
            "flight_execution_review",
            "independent reviewer identity is required",
        )
    if (
        record["independent"] is not True
        or record["accepted"] is not True
    ):
        _fail(
            "flight_execution_review",
            "execution review must be independently accepted",
        )
    for name in (
        "source_clean",
        "runtime_accepted",
        "resource_policy_accepted",
        "stage_accounting_accepted",
        "custody_accepted",
        "final_access_unopened",
        "training_not_executed",
    ):
        if record[name] is not True:
            _fail(
                "flight_execution_review",
                f"review did not accept required gate: {name}",
            )
    if (
        not isinstance(record["findings"], list)
        or any(type(item) is not str for item in record["findings"])
    ):
        _fail(
            "flight_execution_review",
            "review findings must be a string array",
        )
    return strict_json_loads(canonical_json(record))


def prospective_declaration(
    *,
    source: Mapping[str, Any],
    development_sha256: str,
    tuning_sha256: str,
    custody: Mapping[str, Any],
    independent_review: Mapping[str, Any],
) -> tuple[FlightDeclaration, dict[str, str], dict[str, Any]]:
    binding = review_binding(
        source=source,
        development_sha256=development_sha256,
        tuning_sha256=tuning_sha256,
        custody=custody,
    )
    review = validate_review_record(
        independent_review,
        binding=binding,
    )
    value = _base_prospective_value(
        source=source,
        development_sha256=development_sha256,
        tuning_sha256=tuning_sha256,
        custody=custody,
    )
    review_digest = sha256_json(review)
    value["dependencies"]["independent_execution_review"] = {
        "status": "accepted",
        "evidence_sha256": review_digest,
    }
    declaration = FlightDeclaration.from_mapping(value)
    evidence = dict(ACCEPTED_DEPENDENCY_EVIDENCE)
    evidence.update(
        {
            "resource_enforcement": (
                RESOURCE_ENFORCEMENT_EVIDENCE_SHA256
            ),
            "stage_accounting_integration": (
                STAGE_ACCOUNTING_EVIDENCE_SHA256
            ),
            "independent_execution_review": review_digest,
        }
    )
    gate = execution_gate(
        declaration,
        current_evidence=evidence,
    )
    if not gate["admitted"]:
        _fail(
            "flight_execution_review",
            "reviewed prospective declaration still fails admission",
        )
    return declaration, evidence, binding


def admission_candidate(
    *,
    source: Mapping[str, Any],
    development_sha256: str,
    tuning_sha256: str,
    custody: Mapping[str, Any] | None = None,
    independent_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a machine-readable package without fabricating external evidence."""
    reasons: list[str] = []
    binding = None
    declaration = None
    evidence = dict(ACCEPTED_DEPENDENCY_EVIDENCE)
    if custody is None:
        reasons.append("learner_final_custody_missing")
    else:
        custody = validate_custody_record(custody)
        binding = review_binding(
            source=source,
            development_sha256=development_sha256,
            tuning_sha256=tuning_sha256,
            custody=custody,
        )
        if independent_review is None:
            reasons.append("independent_execution_review_missing")
        else:
            declaration, evidence, binding = prospective_declaration(
                source=source,
                development_sha256=development_sha256,
                tuning_sha256=tuning_sha256,
                custody=custody,
                independent_review=independent_review,
            )
    body = {
        "schema": ADMISSION_PACKET_SCHEMA,
        "status": (
            "READY_FOR_EXTERNAL_CUSTODY"
            if custody is None
            else (
                "READY_FOR_INDEPENDENT_EXECUTION_REVIEW"
                if independent_review is None
                else "ADMITTED"
            )
        ),
        "admitted": declaration is not None,
        "training_executed": False,
        "final_access": False,
        "source": deepcopy(dict(source)),
        "development_sha256": development_sha256,
        "tuning_sha256": tuning_sha256,
        "runtime": {
            "device": "cpu",
            "platform": RUNTIME_PLATFORM,
            "python": RUNTIME_PYTHON,
            "torch": RUNTIME_TORCH,
            "threads": RUNTIME_THREADS,
        },
        "resource_policy": deepcopy(RESOURCE_POLICY),
        "resource_policy_sha256": RESOURCE_POLICY_SHA256,
        "custody": deepcopy(custody),
        "review_binding": deepcopy(binding),
        "declaration": (
            None if declaration is None else declaration.to_dict()
        ),
        "declaration_sha256": (
            None if declaration is None else declaration.sha256
        ),
        "current_evidence": evidence,
        "reasons": sorted(reasons),
    }
    body["packet_sha256"] = sha256_json(body)
    return body


def _proc_table() -> dict[int, tuple[int, int]]:
    """Return pid -> (ppid, rss_bytes) from Linux procfs."""
    result: dict[int, tuple[int, int]] = {}
    proc = Path("/proc")
    if not proc.is_dir():
        _fail(
            "flight_resource_platform",
            "process-tree RSS enforcement requires Linux procfs",
        )
    page = os.sysconf("SC_PAGE_SIZE")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_fields = (entry / "stat").read_text().split()
            if len(stat_fields) < 24:
                continue
            pid = int(stat_fields[0])
            ppid = int(stat_fields[3])
            rss = int(stat_fields[23]) * page
            result[pid] = (ppid, max(0, rss))
        except (
            FileNotFoundError,
            ProcessLookupError,
            PermissionError,
            ValueError,
            OSError,
        ):
            continue
    return result


def process_tree_rss_bytes(root_pid: int | None = None) -> int:
    root_pid = os.getpid() if root_pid is None else root_pid
    if type(root_pid) is not int or root_pid <= 0:
        _fail(
            "flight_resource_pid",
            "positive process root PID required",
        )
    table = _proc_table()
    if root_pid not in table:
        _fail(
            "flight_resource_pid",
            "resource root process is absent from procfs",
        )
    owned = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _rss) in table.items():
            if pid not in owned and ppid in owned:
                owned.add(pid)
                changed = True
    return sum(table[pid][1] for pid in owned if pid in table)


def directory_bytes(root: str | Path) -> int:
    root = Path(root)
    if not root.exists():
        return 0
    if root.is_symlink() or not root.is_dir():
        _fail(
            "flight_resource_storage",
            "attempt storage root must be a real directory",
        )
    total = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        base_path = Path(base)
        for name in dirs:
            path = base_path / name
            if path.is_symlink():
                _fail(
                    "flight_resource_storage",
                    "attempt storage must not contain symlink directories",
                )
        for name in files:
            path = base_path / name
            if path.is_symlink() or not path.is_file():
                _fail(
                    "flight_resource_storage",
                    "attempt storage contains an unsafe file",
                )
            total += path.stat().st_size
    return total


@dataclass
class FlightResourceGuard:
    """Continuously observe whole-attempt finite resource ceilings."""

    attempt_root: Path
    limits: Mapping[str, Any]
    clock: Callable[[], float] = time.monotonic
    rss_reader: Callable[[], int] = process_tree_rss_bytes
    storage_reader: Callable[[str | Path], int] = directory_bytes
    sample_interval_seconds: float = RESOURCE_SAMPLE_INTERVAL_SECONDS
    abort_callback: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        self.attempt_root = Path(self.attempt_root)
        expected = resource_ceilings()
        if dict(self.limits) != expected:
            _fail(
                "flight_resource_policy",
                "resource guard limits differ from the admitted policy",
            )
        if not 0.05 <= self.sample_interval_seconds <= 5.0:
            _fail(
                "flight_resource_policy",
                "resource sampling interval is outside the bounded range",
            )
        self._started = self.clock()
        self._peak_rss = 0
        self._peak_storage = 0
        self._publication_bytes = 0
        self._violation: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stop.is_set()
        )

    def bind_declaration(self, declaration: FlightDeclaration) -> None:
        if not isinstance(declaration, FlightDeclaration):
            _fail(
                "flight_resource_policy",
                "resource guard requires a FlightDeclaration",
            )
        if declaration.to_dict()["resource_ceilings"] != dict(self.limits):
            _fail(
                "flight_resource_policy",
                "resource guard and declaration ceilings differ",
            )

    def start(self) -> "FlightResourceGuard":
        if self._thread is not None:
            _fail(
                "flight_resource_guard",
                "resource guard already started",
            )
        self.checkpoint("guard_start")
        self._thread = threading.Thread(
            target=self._monitor,
            name="4ly004-resource-guard",
            daemon=True,
        )
        self._thread.start()
        return self

    def _observe(self) -> None:
        elapsed = self.clock() - self._started
        rss = int(self.rss_reader())
        storage = int(self.storage_reader(self.attempt_root))
        self._peak_rss = max(self._peak_rss, rss)
        self._peak_storage = max(self._peak_storage, storage)
        if elapsed > float(self.limits["maximum_wall_seconds"]):
            self._set_violation("maximum_wall_seconds")
        elif rss > int(self.limits["maximum_resident_bytes"]):
            self._set_violation("maximum_resident_bytes")
        elif storage > int(self.limits["maximum_storage_bytes"]):
            self._set_violation("maximum_storage_bytes")
        elif self._publication_bytes > int(
            self.limits["maximum_publication_bytes"]
        ):
            self._set_violation("maximum_publication_bytes")

    def _set_violation(self, reason: str) -> None:
        if self._violation is not None:
            return
        self._violation = reason
        if self.abort_callback is not None:
            try:
                self.abort_callback(reason)
            except BaseException:
                pass

    def _monitor(self) -> None:
        while not self._stop.wait(self.sample_interval_seconds):
            try:
                self._observe()
            except BaseException as exc:
                self._set_violation(
                    f"resource_monitor_error:{type(exc).__name__}"
                )

    def charge_publication_bytes(self, count: int) -> None:
        if type(count) is not int or count < 0:
            _fail(
                "flight_resource_publication",
                "publication charge must be a nonnegative integer",
            )
        if (
            self._publication_bytes + count
            > int(self.limits["maximum_publication_bytes"])
        ):
            self._set_violation("maximum_publication_bytes")
            self.checkpoint("publication_reservation")
        self._publication_bytes += count
        self.checkpoint("publication_reservation")

    def checkpoint(self, scope: str) -> dict[str, Any]:
        if type(scope) is not str or not scope:
            _fail(
                "flight_resource_guard",
                "resource checkpoint scope is required",
            )
        self._observe()
        if self._violation is not None:
            _fail(
                "flight_resource_exhausted",
                f"{self._violation}:{scope}",
            )
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "nca-4ly004-resource-snapshot/v1",
            "elapsed_seconds": max(0.0, self.clock() - self._started),
            "peak_process_tree_rss_bytes": self._peak_rss,
            "peak_storage_bytes": self._peak_storage,
            "publication_bytes": self._publication_bytes,
            "violation": self._violation,
            "resource_policy_sha256": RESOURCE_POLICY_SHA256,
        }

    def close(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(
                timeout=max(
                    1.0,
                    self.sample_interval_seconds * 4.0,
                )
            )
        self._observe()
        return self.snapshot()

    def __enter__(self) -> "FlightResourceGuard":
        return self.start()

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()
