"""Durable accounting and fail-closed execution gates for 4LY-004.

This module is preparation infrastructure.  It imports no Torch or host runtime,
opens no final suite and cannot turn PR #85 into an experiment admission.  A
future execution must provide a complete source-bound declaration plus current
evidence whose digests match that declaration.

Reservations are durable before work and are never refunded.  A process that
dies after a reservation but before an outcome leaves unknown consumption; on
resume that uncertainty is persisted and all ordinary execution remains
blocked.  The journal is append-only and hash-linked so failures, costs and
checkpoint lineage survive interruption.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from nca.contracts import ContractError, canonical_json, sha256_json, strict_json_loads
from nca.evaluation.durable_journal import sync_directory


DECLARATION_SCHEMA = "nca-flight-experiment-declaration/v3"
LEDGER_RECORD_SCHEMA = "nca-flight-accounting-record/v2"
ZERO_HASH = "0" * 64
MAX_RECORD_BYTES = 64 * 1024
POLICY_SEEDS = (74011, 74013, 74017)

HOST_CONTRACT_HEAD = "9935d5696721d2c72a8273ea70454ab0c5b11361"
TEACHER_SOURCE_MERGE = "b5a8570e5123e1f633766e9390761a1c8134cfa2"
TRAINER_PROFILE_SHA256 = "f1f9286a27b929c96caca5aaf2be70cbd82e0fd933f29968c1aaaa43ac8f1704"
CONTROLLER_SHA256 = "fb86190c68c6a87e7d2ef7e845b1fb97305f188723a326f67f07caa6971cc8ae"
TEACHER_QUALIFICATION_CHECKOUT = "2bc2b6cfa9ad62512266df5804c5662b5ba8bd04"
TEACHER_SHA256 = "0fb571cfd314a53fc86b3517fb5a8258a77d7215fcbcb43fc7f53519e7e29761"
TEACHER_FINAL_SUITE_SHA256 = "4bd18d728581f64fa770c4dc29f765eff956f61999b96d03d8e378d54efc3401"
HOST_BUILD_SHA256 = "55e5ba334b6683adf2ba0d8fc93fb7defba109cd4a319d2f366a2a86e4dfdf84"
ACTUAL_HOST_CONFORMANCE_EVIDENCE_SHA256 = "193c150e83d7e83217692be75407ccc5acb4038e1a60183fbb489bd0b78bece9"
ACTUAL_HOST_CONFORMANCE_ARTIFACT_SHA256 = "6578a638cf3f70a59974b152474b908ab5836b1baf29dc51f80b7d20695059ea"
ACTUAL_HOST_CONFORMANCE_RUN_ID = 40211483325
QUALIFIED_HOST_MEASUREMENTS = {
    "startup_seconds": 0.899644376,
    "reset_latency_max_seconds": 0.047778348,
    "reset_latency_median_seconds": 0.039766714,
    "step_latency_max_seconds": 0.018060448,
    "step_latency_median_seconds": 0.010230353,
    "confirmed_physical_ticks": 20,
    "confirmed_ticks_per_second": 282.8249100698098,
    "owner_serialization_microseconds_max": 6164,
    "shutdown_seconds": 0.163780543,
    "cleanup_seconds": 0.000515674,
    "process_tree_peak_resident_bytes": 254550016,
    "owner_receipt_bytes": 36038,
    "event_payload_bytes": 63519,
    "evidence_file_bytes": 67363,
    "accounting_journal_bytes": 5363,
}
RESOURCE_ENFORCEMENT_CONTRACT = {
    "schema": "nca-4ly004-resource-enforcement/v1",
    "whole_attempt": True,
    "process_tree_rss": "linux_procfs_root_plus_descendants",
    "wall_clock": "monotonic",
    "storage": "attempt_root_regular_files_no_symlink_follow",
    "publication": "explicit_prepublication_byte_charge",
    "sampling_seconds": 0.25,
    "check_before_reserved_work": True,
    "fail_stop": True,
}
RESOURCE_ENFORCEMENT_EVIDENCE_SHA256 = sha256_json(RESOURCE_ENFORCEMENT_CONTRACT)
STAGE_ACCOUNTING_CONTRACT = {
    "schema": "nca-4ly004-stage-accounting/v1",
    "requires_stage_for_stage_sensitive_work": True,
    "optimizer_seed_isolation": list(POLICY_SEEDS),
    "final_stage_disabled_before_final_open_barrier": True,
}
HOST_QUALIFICATION_EVIDENCE_SHA256 = sha256_json({
    "accepted_merge": HOST_CONTRACT_HEAD,
    "teacher_source_merge": TEACHER_SOURCE_MERGE,
    "trainer_profile_sha256": TRAINER_PROFILE_SHA256,
    "controller_sha256": CONTROLLER_SHA256,
    "teacher_sha256": TEACHER_SHA256,
})
TEACHER_QUALIFICATION_EVIDENCE_SHA256 = sha256_json({
    "qualification_checkout": TEACHER_QUALIFICATION_CHECKOUT,
    "teacher_sha256": TEACHER_SHA256,
    "teacher_final_suite_sha256": TEACHER_FINAL_SUITE_SHA256,
    "teacher_final_may_be_reused_as_learner_final": False,
})
ACCEPTED_DEPENDENCY_EVIDENCE = {
    "host_4ly003": HOST_QUALIFICATION_EVIDENCE_SHA256,
    "teacher_qualification": TEACHER_QUALIFICATION_EVIDENCE_SHA256,
    "actual_host_conformance": ACTUAL_HOST_CONFORMANCE_EVIDENCE_SHA256,
    "throughput_measurement": ACTUAL_HOST_CONFORMANCE_EVIDENCE_SHA256,
    "resource_enforcement": RESOURCE_ENFORCEMENT_EVIDENCE_SHA256,
}
DEVELOPMENT_SEEDS = (101, 103, 107, 109)
TUNING_SEEDS = (211, 223, 227, 229)
SCENARIO_MAX_TICKS = {
    "unsupported_hover": 300,
    "horizontal_reach": 1080,
    "vertical_reach": 1080,
    "diagonal_reach": 1080,
    "reverse": 1560,
    "arrival_braking": 1500,
    "wide_corridor": 1800,
    "offset_gates": 2100,
    "corner": 2100,
    "slalom": 2400,
    "wall_recovery": 2100,
}
SCENARIO_FAMILY = {
    "unsupported_hover": "hover",
    "horizontal_reach": "free_space",
    "vertical_reach": "free_space",
    "diagonal_reach": "free_space",
    "reverse": "reversal",
    "arrival_braking": "arrival",
    "wide_corridor": "obstacles",
    "offset_gates": "obstacles",
    "corner": "obstacles",
    "slalom": "obstacles",
    "wall_recovery": "recovery",
}
CORE_SCENARIOS = tuple(
    scenario for scenario in SCENARIO_MAX_TICKS
    if scenario != "unsupported_hover"
)
FINAL_CASES_PER_SCENARIO = {
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
FINAL_COMPARATORS = 6

WORK_COUNTERS = (
    "host_starts",
    "resets",
    "host_calls",
    "physical_ticks",
    "decisions",
    "capture_labels",
    "optimizer_updates",
    "sample_presentations",
    "tuning_episodes",
    "checkpoint_publications",
    "startup_operations",
    "cleanup_operations",
)
DESIGN_WORK_MAXIMA = {
    "host_starts": 1,
    "resets": 884,
    "host_calls": 246_765,
    "physical_ticks": 1_475_280,
    "decisions": 245_880,
    "capture_labels": 11_400,
    "optimizer_updates": 5_400,
    "sample_presentations": 684_000,
    "tuning_episodes": 480,
    "checkpoint_publications": 63,
    "startup_operations": 1,
    "cleanup_operations": 1,
}
STAGE_CEILINGS = {
    "capture": {
        "resets": 44,
        "physical_ticks": 68_400,
        "decisions": 11_400,
        "capture_labels": 11_400,
    },
    "tuning": {
        "resets": 480,
        "physical_ticks": 806_400,
        "decisions": 134_400,
        "tuning_episodes": 480,
    },
    "final": {
        "resets": 360,
        "physical_ticks": 600_480,
        "decisions": 100_080,
    },
    **{
        f"clone:{seed}": {
            "optimizer_updates": 1_800,
            "sample_presentations": 228_000,
            "checkpoint_publications": 21,
        }
        for seed in POLICY_SEEDS
    },
}
STAGE_SENSITIVE_COUNTERS = frozenset({
    "resets", "physical_ticks", "decisions", "capture_labels",
    "optimizer_updates", "sample_presentations", "tuning_episodes",
    "checkpoint_publications",
})
STAGE_ACCOUNTING_CONTRACT = {
    **STAGE_ACCOUNTING_CONTRACT,
    "stage_ceilings": deepcopy(STAGE_CEILINGS),
}
STAGE_ACCOUNTING_EVIDENCE_SHA256 = sha256_json(STAGE_ACCOUNTING_CONTRACT)
ACCEPTED_DEPENDENCY_EVIDENCE["stage_accounting_integration"] = (
    STAGE_ACCOUNTING_EVIDENCE_SHA256
)
RESOURCE_CEILINGS = (
    "maximum_wall_seconds",
    "maximum_resident_bytes",
    "maximum_storage_bytes",
    "maximum_publication_bytes",
)
DEPENDENCIES = (
    "host_4ly003",
    "actual_host_conformance",
    "teacher_qualification",
    "throughput_measurement",
    "resource_enforcement",
    "stage_accounting_integration",
    "independent_execution_review",
)

# These arithmetic assertions are deliberately source-derived from #229's
# scenario deadlines and Stream E's frozen composition. They prevent a future
# refactor from reintroducing the superseded universal 960-tick assumption.
assert sum(SCENARIO_MAX_TICKS.values()) * len(DEVELOPMENT_SEEDS) == 68_400
assert sum(SCENARIO_MAX_TICKS[name] for name in CORE_SCENARIOS) * len(TUNING_SEEDS) * 4 * len(POLICY_SEEDS) == 806_400
assert sum(SCENARIO_MAX_TICKS[name] * count for name, count in FINAL_CASES_PER_SCENARIO.items()) * FINAL_COMPARATORS == 600_480
assert sum(FINAL_CASES_PER_SCENARIO.values()) == 60

MEASUREMENTS_REQUIRED = (
    "exact 54-channel observation/action/60Hz/six-tick/12-tick-lease conformance",
    "frozen development/tuning manifests and learner-final cryptographic commitment",
    "chosen CPU runtime identity: platform, Python, Torch and execution-thread settings",
    "source-bound actual-host ticks/s and decisions/s on the chosen runner",
    "reset/serialization and ordinary host-call latency, including tail behavior",
    "whole-process startup and cleanup duration on the chosen execution path",
    "process-tree peak resident memory for host, bridge, learner and evidence writer",
    "capture/journal/checkpoint/publication byte growth and durable storage throughput",
    "final host-call topology, stage propagation and finite startup/cleanup operation counts",
    "validated wall/RSS/storage enforcement bound to the chosen execution path",
    "independent source-bound execution review bound to the complete declaration",
)

def _fail(code: str, detail: str) -> None:
    raise ContractError(code, detail)


def _sha256_or_none(value: Any, name: str) -> None:
    if value is not None and (type(value) is not str or not re.fullmatch(r"[a-f0-9]{64}", value)):
        _fail("flight_declaration_digest", f"{name} must be null or a SHA-256 digest")


def _revision_or_none(value: Any, name: str) -> None:
    if value is not None and (type(value) is not str or not re.fullmatch(r"[a-f0-9]{40}", value)):
        _fail("flight_declaration_revision", f"{name} must be null or a full Git revision")


def _positive_int_or_none(value: Any, name: str) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        _fail("flight_declaration_ceiling", f"{name} must be null or a positive integer")


def _exact_keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        _fail("flight_declaration_shape", f"{name} must contain exactly {sorted(expected)}")
    return value


def _validate_declaration(value: Mapping[str, Any]) -> None:
    top = {
        "schema", "experiment_id", "status", "source_contract",
        "learner_configuration", "identities", "interfaces", "scenario",
        "splits", "runtime", "seeds", "work_ceilings", "stage_ceilings",
        "resource_ceilings", "dependencies", "attempt_policy", "final_access",
    }
    _exact_keys(value, top, "declaration")
    if value["schema"] != DECLARATION_SCHEMA:
        _fail("flight_declaration_schema", "unsupported declaration schema")
    if (type(value["experiment_id"]) is not str
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value["experiment_id"])):
        _fail("flight_declaration_id", "experiment_id is invalid")
    if value["status"] not in ("preparation_only", "prospective"):
        _fail("flight_declaration_status", "status must be preparation_only or prospective")
    if value["final_access"] is not False:
        _fail("flight_final_access", "4LY-004 preparation cannot grant final access")

    source = _exact_keys(value["source_contract"], {
        "repository", "pull_request", "revision", "qualification_status",
        "trainer_profile_sha256", "controller_sha256", "teacher_sha256",
        "teacher_qualification_checkout", "teacher_final_suite_sha256",
        "teacher_final_may_be_reused_as_learner_final",
    }, "source_contract")
    expected_source = {
        "repository": "techrote/cybersand",
        "pull_request": 249,
        "revision": HOST_CONTRACT_HEAD,
        "qualification_status": "accepted_lease_repair_host_with_qualified_teacher_lineage",
        "trainer_profile_sha256": TRAINER_PROFILE_SHA256,
        "controller_sha256": CONTROLLER_SHA256,
        "teacher_sha256": TEACHER_SHA256,
        "teacher_qualification_checkout": TEACHER_QUALIFICATION_CHECKOUT,
        "teacher_final_suite_sha256": TEACHER_FINAL_SUITE_SHA256,
        "teacher_final_may_be_reused_as_learner_final": False,
    }
    if dict(source) != expected_source:
        _fail("flight_source_contract", "accepted #249 host / #229 teacher-lineage binding changed")

    learner = _exact_keys(value["learner_configuration"], {
        "profile_id", "architecture", "algorithm", "optimizer",
        "learning_rate", "batch_capacity", "epochs", "recurrence",
        "ppo", "cuda", "domain_randomization", "parallel_physical_hosts",
    }, "learner_configuration")
    if dict(learner) != {
        "profile_id": "4ly_trainer",
        "architecture": "54-64-64-2-tanh",
        "algorithm": "behavior_cloning",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_capacity": 128,
        "epochs": 20,
        "recurrence": False,
        "ppo": False,
        "cuda": False,
        "domain_randomization": False,
        "parallel_physical_hosts": 1,
    }:
        _fail("flight_learner_configuration", "frozen Stream E learner configuration changed")

    identities = _exact_keys(value["identities"], {
        "learner_revision", "learner_source_sha256", "host_revision",
        "host_build_sha256", "profile_sha256", "controller_sha256", "teacher_sha256",
    }, "identities")
    _revision_or_none(identities["learner_revision"], "learner_revision")
    _sha256_or_none(identities["learner_source_sha256"], "learner_source_sha256")
    _revision_or_none(identities["host_revision"], "host_revision")
    for name in ("host_build_sha256", "profile_sha256", "controller_sha256", "teacher_sha256"):
        _sha256_or_none(identities[name], name)
    if identities["host_revision"] is not None and identities["host_revision"] != HOST_CONTRACT_HEAD:
        _fail("flight_host_identity", "host revision differs from the accepted Cybersand merge")
    if identities["profile_sha256"] is not None and identities["profile_sha256"] != TRAINER_PROFILE_SHA256:
        _fail("flight_profile_identity", "accepted Trainer profile differs from the preregistered contract")
    if identities["controller_sha256"] is not None and identities["controller_sha256"] != CONTROLLER_SHA256:
        _fail("flight_controller_identity", "accepted controller differs from the preregistered contract")
    if identities["teacher_sha256"] is not None and identities["teacher_sha256"] != TEACHER_SHA256:
        _fail("flight_teacher_identity", "teacher differs from the qualified frozen source")

    interfaces = _exact_keys(value["interfaces"], {
        "episode_schema", "observation_schema", "flat_schema", "action_schema",
        "observation_channels", "action_outputs", "physics_hz", "decision_ticks",
        "lease_ticks", "profile_id", "actor_id", "course_id", "course_version",
        "split_geometry_version",
    }, "interfaces")
    if dict(interfaces) != {
        "episode_schema": "4ly-episode/v1",
        "observation_schema": "4ly-observation/v1",
        "flat_schema": "4ly-flat-observation/v1",
        "action_schema": "4ly-action/v1",
        "observation_channels": 54,
        "action_outputs": 2,
        "physics_hz": 60,
        "decision_ticks": 6,
        "lease_ticks": 12,
        "profile_id": "4ly_trainer",
        "actor_id": 0,
        "course_id": "4ly-flight-course/v1",
        "course_version": 1,
        "split_geometry_version": 3,
    }:
        _fail("flight_interface_contract", "reviewed #229 flight interface/cadence contract changed")

    scenario = _exact_keys(value["scenario"], {
        "family_count", "scenario_max_ticks", "scenario_family",
        "development_seeds", "tuning_seeds", "final_cases_per_scenario",
        "final_comparators",
    }, "scenario")
    if (scenario["family_count"] != 6
            or scenario["scenario_max_ticks"] != SCENARIO_MAX_TICKS
            or scenario["scenario_family"] != SCENARIO_FAMILY
            or scenario["development_seeds"] != list(DEVELOPMENT_SEEDS)
            or scenario["tuning_seeds"] != list(TUNING_SEEDS)
            or scenario["final_cases_per_scenario"] != FINAL_CASES_PER_SCENARIO
            or scenario["final_comparators"] != FINAL_COMPARATORS):
        _fail("flight_scenario_contract", "scenario/deadline/split composition differs from Stream E")

    splits = _exact_keys(value["splits"], {
        "development_sha256", "tuning_sha256", "final_commitment_sha256",
        "final_custody_sha256", "opened_final",
    }, "splits")
    for name in (
        "development_sha256", "tuning_sha256", "final_commitment_sha256",
        "final_custody_sha256",
    ):
        _sha256_or_none(splits[name], f"splits.{name}")
    if splits["opened_final"] is not False:
        _fail("flight_final_access", "learner final payload must remain unopened")

    runtime = _exact_keys(value["runtime"], {
        "device", "platform", "python", "torch", "threads",
    }, "runtime")
    if runtime["device"] != "cpu":
        _fail("flight_runtime", "the preregistered learner path is CPU only")
    if runtime["threads"] is not None and (
            type(runtime["threads"]) is not int or runtime["threads"] <= 0):
        _fail("flight_runtime", "runtime thread count must be unresolved or a positive integer")
    for name in ("platform", "python", "torch"):
        if runtime[name] is not None and (type(runtime[name]) is not str or not runtime[name]):
            _fail("flight_runtime", f"runtime.{name} must be null or a nonempty identity")

    if value["seeds"] != list(POLICY_SEEDS):
        _fail("flight_seed_contract", "the three fixed policy seeds changed")

    ceilings = _exact_keys(value["work_ceilings"], set(WORK_COUNTERS), "work_ceilings")
    for name in WORK_COUNTERS:
        _positive_int_or_none(ceilings[name], f"work_ceilings.{name}")
    for name, maximum in DESIGN_WORK_MAXIMA.items():
        if ceilings[name] != maximum:
            _fail("flight_declaration_ceiling",
                  f"work_ceilings.{name} must equal the Stream E prospective maximum")
    for name in ("host_calls", "startup_operations", "cleanup_operations"):
        if ceilings[name] is not None:
            _positive_int_or_none(ceilings[name], f"work_ceilings.{name}")

    stages = _exact_keys(value["stage_ceilings"], set(STAGE_CEILINGS), "stage_ceilings")
    if dict(stages) != STAGE_CEILINGS:
        _fail("flight_stage_ceiling", "stage ceilings differ from the preregistered capture/tuning/final/seed bounds")

    resources = _exact_keys(value["resource_ceilings"], set(RESOURCE_CEILINGS), "resource_ceilings")
    if resources["maximum_wall_seconds"] is not None:
        wall = resources["maximum_wall_seconds"]
        if type(wall) not in (int, float) or type(wall) is bool or wall <= 0:
            _fail("flight_resource_ceiling", "maximum_wall_seconds must be null or positive")
    for name in RESOURCE_CEILINGS[1:]:
        _positive_int_or_none(resources[name], f"resource_ceilings.{name}")

    dependencies = _exact_keys(value["dependencies"], set(DEPENDENCIES), "dependencies")
    for name in DEPENDENCIES:
        item = _exact_keys(dependencies[name], {"status", "evidence_sha256"}, f"dependencies.{name}")
        if item["status"] not in ("pending", "accepted"):
            _fail("flight_dependency", f"{name} status must be pending or accepted")
        _sha256_or_none(item["evidence_sha256"], f"dependencies.{name}.evidence_sha256")
        if item["status"] == "accepted" and item["evidence_sha256"] is None:
            _fail("flight_dependency", f"{name} accepted without immutable evidence")

    policy = _exact_keys(value["attempt_policy"], {
        "maximum_attempts", "automatic_retry", "automatic_extension",
        "automatic_replay_unknown",
    }, "attempt_policy")
    if policy != {
        "maximum_attempts": 1,
        "automatic_retry": False,
        "automatic_extension": False,
        "automatic_replay_unknown": False,
    }:
        _fail("flight_attempt_policy", "one consumed attempt with no replay/extension is required")

@dataclass(frozen=True)
class FlightDeclaration:
    """Canonical immutable declaration value plus its content digest."""

    canonical: str
    sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FlightDeclaration":
        if not isinstance(value, Mapping):
            _fail("flight_declaration_shape", "declaration must be a mapping")
        candidate = deepcopy(dict(value))
        _validate_declaration(candidate)
        encoded = canonical_json(candidate)
        return cls(encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest())

    def to_dict(self) -> dict[str, Any]:
        return strict_json_loads(self.canonical)


def preparation_declaration() -> FlightDeclaration:
    """Incomplete source-bound template; deliberately not an execution admission."""
    pending = {name: {"status": "pending", "evidence_sha256": None} for name in DEPENDENCIES}
    for name, digest in ACCEPTED_DEPENDENCY_EVIDENCE.items():
        pending[name] = {"status": "accepted", "evidence_sha256": digest}
    return FlightDeclaration.from_mapping({
        "schema": DECLARATION_SCHEMA,
        "experiment_id": "4ly004-trainer-flight-v2",
        "status": "preparation_only",
        "source_contract": {
            "repository": "techrote/cybersand",
            "pull_request": 249,
            "revision": HOST_CONTRACT_HEAD,
            "qualification_status": "accepted_lease_repair_host_with_qualified_teacher_lineage",
            "trainer_profile_sha256": TRAINER_PROFILE_SHA256,
            "controller_sha256": CONTROLLER_SHA256,
            "teacher_sha256": TEACHER_SHA256,
            "teacher_qualification_checkout": TEACHER_QUALIFICATION_CHECKOUT,
            "teacher_final_suite_sha256": TEACHER_FINAL_SUITE_SHA256,
            "teacher_final_may_be_reused_as_learner_final": False,
        },
        "learner_configuration": {
            "profile_id": "4ly_trainer",
            "architecture": "54-64-64-2-tanh",
            "algorithm": "behavior_cloning",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_capacity": 128,
            "epochs": 20,
            "recurrence": False,
            "ppo": False,
            "cuda": False,
            "domain_randomization": False,
            "parallel_physical_hosts": 1,
        },
        "identities": {
            "learner_revision": None,
            "learner_source_sha256": None,
            "host_revision": HOST_CONTRACT_HEAD,
            "host_build_sha256": HOST_BUILD_SHA256,
            "profile_sha256": TRAINER_PROFILE_SHA256,
            "controller_sha256": CONTROLLER_SHA256,
            "teacher_sha256": TEACHER_SHA256,
        },
        "interfaces": {
            "episode_schema": "4ly-episode/v1",
            "observation_schema": "4ly-observation/v1",
            "flat_schema": "4ly-flat-observation/v1",
            "action_schema": "4ly-action/v1",
            "observation_channels": 54,
            "action_outputs": 2,
            "physics_hz": 60,
            "decision_ticks": 6,
            "lease_ticks": 12,
            "profile_id": "4ly_trainer",
            "actor_id": 0,
            "course_id": "4ly-flight-course/v1",
            "course_version": 1,
            "split_geometry_version": 3,
        },
        "scenario": {
            "family_count": 6,
            "scenario_max_ticks": dict(SCENARIO_MAX_TICKS),
            "scenario_family": dict(SCENARIO_FAMILY),
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "tuning_seeds": list(TUNING_SEEDS),
            "final_cases_per_scenario": dict(FINAL_CASES_PER_SCENARIO),
            "final_comparators": FINAL_COMPARATORS,
        },
        "splits": {
            "development_sha256": None,
            "tuning_sha256": None,
            "final_commitment_sha256": None,
            "final_custody_sha256": None,
            "opened_final": False,
        },
        "runtime": {
            "device": "cpu",
            "platform": None,
            "python": None,
            "torch": None,
            "threads": None,
        },
        "seeds": list(POLICY_SEEDS),
        "work_ceilings": {
            "host_starts": DESIGN_WORK_MAXIMA["host_starts"],
            "resets": DESIGN_WORK_MAXIMA["resets"],
            "host_calls": DESIGN_WORK_MAXIMA["host_calls"],
            "physical_ticks": DESIGN_WORK_MAXIMA["physical_ticks"],
            "decisions": DESIGN_WORK_MAXIMA["decisions"],
            "capture_labels": DESIGN_WORK_MAXIMA["capture_labels"],
            "optimizer_updates": DESIGN_WORK_MAXIMA["optimizer_updates"],
            "sample_presentations": DESIGN_WORK_MAXIMA["sample_presentations"],
            "tuning_episodes": DESIGN_WORK_MAXIMA["tuning_episodes"],
            "checkpoint_publications": DESIGN_WORK_MAXIMA["checkpoint_publications"],
            "startup_operations": DESIGN_WORK_MAXIMA["startup_operations"],
            "cleanup_operations": DESIGN_WORK_MAXIMA["cleanup_operations"],
        },
        "stage_ceilings": deepcopy(STAGE_CEILINGS),
        "resource_ceilings": {
            "maximum_wall_seconds": None,
            "maximum_resident_bytes": None,
            "maximum_storage_bytes": None,
            "maximum_publication_bytes": None,
        },
        "dependencies": pending,
        "attempt_policy": {
            "maximum_attempts": 1,
            "automatic_retry": False,
            "automatic_extension": False,
            "automatic_replay_unknown": False,
        },
        "final_access": False,
    })

def required_measurements() -> tuple[str, ...]:
    return MEASUREMENTS_REQUIRED


def execution_gate(declaration: FlightDeclaration, *, current_evidence: Mapping[str, str] | None,
                   ledger_snapshot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate prospective execution authority without performing any work."""
    value = declaration.to_dict()
    _validate_declaration(value)
    reasons: list[str] = []
    if value["status"] != "prospective":
        reasons.append("declaration_not_prospective")

    for name, identity in value["identities"].items():
        if identity is None:
            reasons.append(f"identity_unresolved:{name}")
    for name in (
        "development_sha256", "tuning_sha256", "final_commitment_sha256",
        "final_custody_sha256",
    ):
        if value["splits"][name] is None:
            reasons.append(f"split_unresolved:{name}")
    for name in ("platform", "python", "torch", "threads"):
        if value["runtime"][name] is None:
            reasons.append(f"runtime_unresolved:{name}")
    for name, ceiling in value["work_ceilings"].items():
        if ceiling is None:
            reasons.append(f"work_ceiling_unmeasured:{name}")
    for name, ceiling in value["resource_ceilings"].items():
        if ceiling is None:
            reasons.append(f"resource_ceiling_unmeasured:{name}")

    current = dict(current_evidence or {})
    for name, dependency in value["dependencies"].items():
        if dependency["status"] != "accepted" or dependency["evidence_sha256"] is None:
            reasons.append(f"dependency_unfinished:{name}")
        elif current.get(name) != dependency["evidence_sha256"]:
            reasons.append(f"evidence_stale_or_missing:{name}")

    if ledger_snapshot is not None:
        if ledger_snapshot.get("declaration_sha256") != declaration.sha256:
            reasons.append("ledger_declaration_mismatch")
        if ledger_snapshot.get("closed"):
            reasons.append("attempt_already_consumed")
        if ledger_snapshot.get("failed"):
            reasons.append("attempt_failed")
        if ledger_snapshot.get("unknown_consumption"):
            reasons.append("unknown_consumption")
        if ledger_snapshot.get("poisoned"):
            reasons.append("ledger_integrity_uncertain")
        if ledger_snapshot.get("pending_reservations"):
            reasons.append("pending_reservation")

    reasons = sorted(set(reasons))
    admitted = not reasons
    return {
        "schema": "nca-flight-execution-gate/v1",
        "admitted": admitted,
        "training_enabled": admitted,
        "final_access": False,
        "declaration_sha256": declaration.sha256,
        "reasons": reasons,
    }


def _regular_file(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        _fail("flight_ledger_file", "accounting journal must be one regular unlinked file")
    return info


def _record_bytes(kind: str, payload: Mapping[str, Any], sequence: int, previous: str) -> tuple[bytes, str]:
    record = {
        "schema": LEDGER_RECORD_SCHEMA,
        "kind": kind,
        "payload": deepcopy(dict(payload)),
        "_ledger": {"sequence": sequence, "previous_sha256": previous},
    }
    digest = sha256_json(record)
    record["_ledger"]["record_sha256"] = digest
    encoded = (canonical_json(record) + "\n").encode("utf-8")
    if len(encoded) > MAX_RECORD_BYTES:
        _fail("flight_ledger_record", "accounting record exceeds the bounded record size")
    return encoded, digest


def _empty_state(declaration: FlightDeclaration, attempt_id: str) -> dict[str, Any]:
    return {
        "declaration_sha256": declaration.sha256,
        "attempt_id": attempt_id,
        "sequence": 0,
        "chain_sha256": ZERO_HASH,
        "charged": {name: 0 for name in WORK_COUNTERS},
        "stage_counts": {
            stage: {name: 0 for name in limits}
            for stage, limits in STAGE_CEILINGS.items()
        },
        "current_stage": None,
        "confirmed_physical_ticks": 0,
        "pending": {},
        "failures": [],
        "checkpoints": {},
        "failed": False,
        "unknown_consumption": False,
        "interrupted": False,
        "closed": False,
        "poisoned": False,
        "reservations": 0,
    }


def _apply_record(state: dict[str, Any], kind: str, payload: Mapping[str, Any]) -> None:
    if kind == "attempt_open":
        if state["sequence"] != 1:
            _fail("flight_ledger_chain", "attempt_open must be the first record")
        if set(payload) != {"attempt_id", "declaration_sha256"}:
            _fail("flight_ledger_shape", "attempt_open has an invalid shape")
        if payload["attempt_id"] != state["attempt_id"] or payload["declaration_sha256"] != state["declaration_sha256"]:
            _fail("flight_ledger_identity", "attempt/declaration identity changed")
        return

    if state["closed"]:
        _fail("flight_ledger_chain", "records exist after attempt_close")

    if kind == "interruption":
        if set(payload) != {"reason"} or type(payload["reason"]) is not str or not payload["reason"]:
            _fail("flight_ledger_shape", "interruption has an invalid shape")
        state["interrupted"] = True
        return

    if kind == "attempt_failure":
        if set(payload) != {"reason"} or type(payload["reason"]) is not str or not payload["reason"]:
            _fail("flight_ledger_shape", "attempt failure has an invalid shape")
        if state["pending"]:
            _fail("flight_ledger_pending", "post-receipt failure cannot bypass a pending reservation")
        state["failed"] = True
        state["failures"].append({
            "reservation_id": None,
            "reason": payload["reason"],
            "confirmed_physical_ticks": 0,
        })
        return

    if kind == "reserve":
        if set(payload) != {"reservation_id", "increments", "scope", "cleanup", "stage"}:
            _fail("flight_ledger_shape", "reservation has an invalid shape")
        expected = f"r{state['reservations'] + 1:08d}"
        if payload["reservation_id"] != expected:
            _fail("flight_ledger_sequence", "reservation identity changed")
        stage = payload["stage"]
        if stage is not None and stage not in STAGE_CEILINGS:
            _fail("flight_ledger_stage", "reservation names an undeclared preregistered stage")
        if (type(payload["scope"]) is not str or not payload["scope"]
                or type(payload["cleanup"]) is not bool or not isinstance(payload["increments"], Mapping)
                or not payload["increments"]):
            _fail("flight_ledger_shape", "reservation metadata is invalid")
        if state["pending"]:
            _fail("flight_ledger_pending", "parallel reservations are not permitted")
        for name, amount in payload["increments"].items():
            if name not in WORK_COUNTERS or type(amount) is not int or amount <= 0:
                _fail("flight_ledger_counter", "reservation contains an undeclared counter")
            state["charged"][name] += amount
            if stage is not None and name in state["stage_counts"][stage]:
                state["stage_counts"][stage][name] += amount
        if payload["scope"] == "host_reset":
            state["current_stage"] = stage
        elif payload["scope"] == "host_step" and stage in ("capture", "tuning", "final"):
            state["current_stage"] = stage
        state["reservations"] += 1
        state["pending"][payload["reservation_id"]] = deepcopy(dict(payload))
        return

    if kind in ("complete", "failure", "unknown", "checkpoint"):
        reservation_id = payload.get("reservation_id")
        if reservation_id not in state["pending"]:
            _fail("flight_ledger_outcome", "outcome does not match one pending reservation")
        reservation = state["pending"][reservation_id]
        if kind == "complete":
            if set(payload) != {"reservation_id", "confirmed_physical_ticks"}:
                _fail("flight_ledger_shape", "completion has an invalid shape")
            ticks = payload["confirmed_physical_ticks"]
            if type(ticks) is not int or ticks < 0 or ticks > reservation["increments"].get("physical_ticks", 0):
                _fail("flight_ledger_ticks", "confirmed ticks exceed the reserved physical capacity")
            state["confirmed_physical_ticks"] += ticks
        elif kind == "failure":
            if (set(payload) != {"reservation_id", "reason", "confirmed_physical_ticks"}
                    or type(payload["reason"]) is not str or not payload["reason"]):
                _fail("flight_ledger_shape", "failure has an invalid shape")
            ticks = payload["confirmed_physical_ticks"]
            if type(ticks) is not int or ticks < 0 or ticks > reservation["increments"].get("physical_ticks", 0):
                _fail("flight_ledger_ticks", "failed work confirms ticks outside its reservation")
            state["confirmed_physical_ticks"] += ticks
            state["failed"] = True
            state["failures"].append({
                "reservation_id": reservation_id, "reason": payload["reason"],
                "confirmed_physical_ticks": ticks,
            })
        elif kind == "unknown":
            if (set(payload) != {"reservation_id", "reason", "confirmed_physical_ticks"}
                    or type(payload["reason"]) is not str or not payload["reason"]):
                _fail("flight_ledger_shape", "unknown outcome has an invalid shape")
            ticks = payload["confirmed_physical_ticks"]
            if type(ticks) is not int or ticks < 0 or ticks > reservation["increments"].get("physical_ticks", 0):
                _fail("flight_ledger_ticks", "ambiguous work confirms ticks outside its reservation")
            state["confirmed_physical_ticks"] += ticks
            state["unknown_consumption"] = True
            state["failures"].append({
                "reservation_id": reservation_id, "reason": payload["reason"],
                "confirmed_physical_ticks": ticks,
            })
        else:
            if set(payload) != {"reservation_id", "checkpoint_id", "sha256", "parent_id"}:
                _fail("flight_ledger_shape", "checkpoint publication has an invalid shape")
            if reservation["increments"] != {"checkpoint_publications": 1}:
                _fail("flight_checkpoint_reservation", "checkpoint publication requires its own reserved operation")
            checkpoint_id = payload["checkpoint_id"]
            if (type(checkpoint_id) is not str
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", checkpoint_id)):
                _fail("flight_checkpoint_id", "checkpoint identity is invalid")
            _sha256_or_none(payload["sha256"], "checkpoint.sha256")
            if payload["sha256"] is None:
                _fail("flight_checkpoint_digest", "published checkpoint needs an immutable digest")
            checkpoint_stage = reservation["stage"]
            if checkpoint_stage is not None and not str(checkpoint_stage).startswith("clone:"):
                _fail("flight_checkpoint_lineage", "checkpoint publication is not bound to a clone-seed stage")
            parent = payload["parent_id"]
            if parent is not None and parent not in state["checkpoints"]:
                _fail("flight_checkpoint_lineage", "checkpoint parent is absent")
            if (parent is not None
                    and state["checkpoints"][parent]["stage"] != checkpoint_stage):
                _fail("flight_checkpoint_lineage", "checkpoint parent belongs to another policy seed")
            if checkpoint_id in state["checkpoints"]:
                _fail("flight_checkpoint_lineage", "checkpoint identity was published twice")
            state["checkpoints"][checkpoint_id] = {
                "sha256": payload["sha256"],
                "parent_id": parent,
                "stage": checkpoint_stage,
            }
        del state["pending"][reservation_id]
        return

    if kind == "attempt_close":
        if set(payload) != {"reason"} or type(payload["reason"]) is not str or not payload["reason"]:
            _fail("flight_ledger_shape", "attempt_close has an invalid shape")
        if state["pending"]:
            _fail("flight_ledger_pending", "attempt closed with unresolved reservations")
        state["closed"] = True
        return

    _fail("flight_ledger_kind", f"unknown accounting record kind: {kind}")


def _replay(path: Path, declaration: FlightDeclaration, attempt_id: str) -> dict[str, Any]:
    _regular_file(path)
    state = _empty_state(declaration, attempt_id)
    previous = ZERO_HASH
    with path.open("rb") as source:
        while True:
            line = source.readline(MAX_RECORD_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_RECORD_BYTES or not line.endswith(b"\n"):
                _fail("flight_ledger_chain", "interrupted or oversized accounting record")
            record = strict_json_loads(line, max_bytes=MAX_RECORD_BYTES)
            if not isinstance(record, dict) or set(record) != {"schema", "kind", "payload", "_ledger"}:
                _fail("flight_ledger_shape", "accounting record has an invalid envelope")
            if record["schema"] != LEDGER_RECORD_SCHEMA or not isinstance(record["_ledger"], dict):
                _fail("flight_ledger_schema", "accounting record schema changed")
            meta = record["_ledger"]
            if set(meta) != {"sequence", "previous_sha256", "record_sha256"}:
                _fail("flight_ledger_chain", "accounting chain metadata changed")
            sequence = state["sequence"] + 1
            if meta["sequence"] != sequence or meta["previous_sha256"] != previous:
                _fail("flight_ledger_chain", "accounting sequence or predecessor changed")
            claimed = meta["record_sha256"]
            if type(claimed) is not str or not re.fullmatch(r"[a-f0-9]{64}", claimed):
                _fail("flight_ledger_chain", "accounting record digest is invalid")
            check = deepcopy(record)
            del check["_ledger"]["record_sha256"]
            if sha256_json(check) != claimed:
                _fail("flight_ledger_chain", "accounting record changed after publication")
            state["sequence"] = sequence
            state["chain_sha256"] = claimed
            previous = claimed
            _apply_record(state, record["kind"], record["payload"])
    if state["sequence"] == 0:
        _fail("flight_ledger_chain", "empty accounting journal")
    return state


def _snapshot(state: Mapping[str, Any], declaration: FlightDeclaration) -> dict[str, Any]:
    ceilings = declaration.to_dict()["work_ceilings"]
    exhausted = sorted(
        name for name, ceiling in ceilings.items()
        if ceiling is not None and state["charged"][name] >= ceiling
    )
    return {
        "schema": "nca-flight-accounting-snapshot/v1",
        "declaration_sha256": state["declaration_sha256"],
        "attempt_id": state["attempt_id"],
        "charged": deepcopy(state["charged"]),
        "stage_counts": deepcopy(state["stage_counts"]),
        "current_stage": state["current_stage"],
        "confirmed_physical_ticks": state["confirmed_physical_ticks"],
        "pending_reservations": sorted(state["pending"]),
        "failures": deepcopy(state["failures"]),
        "checkpoints": deepcopy(state["checkpoints"]),
        "failed": state["failed"],
        "unknown_consumption": state["unknown_consumption"],
        "interrupted": state["interrupted"],
        "closed": state["closed"],
        "poisoned": state["poisoned"],
        "reservations": state["reservations"],
        "records": state["sequence"],
        "chain_sha256": state["chain_sha256"],
        "exhausted_counters": exhausted,
        "refunded_work": False,
    }


class FlightAccountingLedger:
    """One durable attempt ledger.  It never starts a host or optimizer."""

    def __init__(self, path: Path, declaration: FlightDeclaration, attempt_id: str,
                 stream, state: dict[str, Any], current_evidence: Mapping[str, str],
                 resource_guard=None):
        self.path = path
        self.declaration = declaration
        self.attempt_id = attempt_id
        self._stream = stream
        self._state = state
        self._evidence = dict(current_evidence)
        self._resource_guard = resource_guard

    @classmethod
    def create(cls, path: str | Path, declaration: FlightDeclaration, *, attempt_id: str,
               current_evidence: Mapping[str, str],
               resource_guard=None) -> "FlightAccountingLedger":
        if type(attempt_id) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", attempt_id):
            _fail("flight_attempt_id", "attempt identity is invalid")
        gate = execution_gate(declaration, current_evidence=current_evidence)
        if not gate["admitted"]:
            _fail("flight_execution_refused", ",".join(gate["reasons"]))
        resource_dependency = declaration.to_dict()["dependencies"]["resource_enforcement"]
        if (
            resource_dependency["evidence_sha256"]
            == RESOURCE_ENFORCEMENT_EVIDENCE_SHA256
            and resource_guard is None
        ):
            _fail(
                "flight_resource_guard",
                "accepted production resource enforcement requires a live guard",
            )
        if resource_guard is not None:
            binder = getattr(resource_guard, "bind_declaration", None)
            if binder is None:
                _fail("flight_resource_guard", "resource guard lacks declaration binding")
            binder(declaration)
            resource_guard.checkpoint("ledger_create")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stream = path.open("xb")
        except FileExistsError as exc:
            raise ContractError("flight_attempt_exists", "attempt journal already exists") from exc
        try:
            sync_directory(path.parent)
            state = _empty_state(declaration, attempt_id)
            ledger = cls(
                path, declaration, attempt_id, stream, state, current_evidence,
                resource_guard=resource_guard,
            )
            ledger._append("attempt_open", {
                "attempt_id": attempt_id, "declaration_sha256": declaration.sha256,
            })
            _apply_record(state, "attempt_open", {
                "attempt_id": attempt_id, "declaration_sha256": declaration.sha256,
            })
            return ledger
        except BaseException:
            stream.close()
            raise

    @classmethod
    def resume(cls, path: str | Path, declaration: FlightDeclaration, *, attempt_id: str,
               current_evidence: Mapping[str, str],
               resource_guard=None) -> "FlightAccountingLedger":
        path = Path(path)
        state = _replay(path, declaration, attempt_id)
        if state["closed"]:
            _fail("flight_attempt_consumed", "closed attempt cannot be resumed or replayed")
        resource_dependency = declaration.to_dict()["dependencies"]["resource_enforcement"]
        if (
            resource_dependency["evidence_sha256"]
            == RESOURCE_ENFORCEMENT_EVIDENCE_SHA256
            and resource_guard is None
        ):
            _fail(
                "flight_resource_guard",
                "accepted production resource enforcement requires a live guard",
            )
        if resource_guard is not None:
            binder = getattr(resource_guard, "bind_declaration", None)
            if binder is None:
                _fail("flight_resource_guard", "resource guard lacks declaration binding")
            binder(declaration)
            resource_guard.checkpoint("ledger_resume")
        stream = path.open("ab")
        ledger = cls(
            path, declaration, attempt_id, stream, state, current_evidence,
            resource_guard=resource_guard,
        )
        for reservation_id in list(state["pending"]):
            ledger.mark_unknown(
                reservation_id, "interrupted_after_reservation_before_durable_outcome")
        ledger._commit("interruption", {"reason": "process_restarted_same_attempt"})
        return ledger

    @classmethod
    def read_snapshot(cls, path: str | Path, declaration: FlightDeclaration, *,
                      attempt_id: str) -> dict[str, Any]:
        return _snapshot(_replay(Path(path), declaration, attempt_id), declaration)

    def _append(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self._state["poisoned"]:
            _fail("flight_ledger_poisoned", "uncertain journal cannot accept another record")
        encoded, digest = _record_bytes(
            kind, payload, self._state["sequence"] + 1, self._state["chain_sha256"])
        try:
            if self._stream.write(encoded) != len(encoded):
                raise OSError("short accounting write")
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except BaseException:
            self._state["poisoned"] = True
            raise
        self._state["sequence"] += 1
        self._state["chain_sha256"] = digest

    def snapshot(self) -> dict[str, Any]:
        return _snapshot(self._state, self.declaration)

    def _commit(self, kind: str, payload: Mapping[str, Any]) -> None:
        # Validate against a copy before publishing.  Once a record is fsynced it
        # must never become an invalid tail merely because a caller was wrong.
        preview = deepcopy(self._state)
        _apply_record(preview, kind, payload)
        self._append(kind, payload)
        _apply_record(self._state, kind, payload)

    def resource_checkpoint(self, scope: str) -> None:
        if self._resource_guard is not None:
            self._resource_guard.checkpoint(scope)

    def _gate_before_work(self) -> None:
        self.resource_checkpoint("before_reserved_work")
        gate = execution_gate(
            self.declaration, current_evidence=self._evidence,
            ledger_snapshot=self.snapshot())
        if not gate["admitted"]:
            _fail("flight_execution_refused", ",".join(gate["reasons"]))

    def reserve_many(self, increments: Mapping[str, int], *, scope: str,
                     cleanup: bool = False, stage: str | None = None) -> str:
        if self._state["closed"] or self._state["poisoned"]:
            _fail("flight_ledger_closed", "closed or uncertain attempt cannot reserve work")
        if self._state["pending"]:
            _fail("flight_ledger_pending", "previous reservation lacks a durable outcome")
        if type(cleanup) is not bool or type(scope) is not str or not scope:
            _fail("flight_ledger_reservation", "reservation scope/cleanup marker is invalid")
        if stage is not None and stage not in STAGE_CEILINGS:
            _fail("flight_ledger_stage", f"unknown preregistered stage: {stage}")
        if stage == "final":
            _fail("flight_final_access", "learner final work is disabled in this preparation")
        if not isinstance(increments, Mapping) or not increments:
            _fail("flight_ledger_reservation", "nonempty increments are required")
        if (self._state["failed"] or self._state["unknown_consumption"]) and not cleanup:
            _fail("flight_execution_refused",
                  "failed or unknown attempt permits cleanup only")
        if cleanup and (stage is not None or not set(increments) <= {"cleanup_operations", "host_calls"}):
            _fail("flight_cleanup_scope", "cleanup cannot consume experiment stages or undeclared cleanup work")
        if not cleanup:
            self._gate_before_work()

        declaration = self.declaration.to_dict()
        ceilings = declaration["work_ceilings"]
        stage_ceilings = declaration["stage_ceilings"]
        stage_integration = declaration["dependencies"]["stage_accounting_integration"]["status"]
        if (stage_integration == "accepted" and stage is None
                and any(name in STAGE_SENSITIVE_COUNTERS for name in increments)):
            _fail(
                "flight_stage_required",
                "accepted stage-accounting integration requires a preregistered stage for this work",
            )
        normalized: dict[str, int] = {}
        for name, amount in increments.items():
            if name not in WORK_COUNTERS or type(amount) is not int or amount <= 0:
                _fail("flight_ledger_counter", f"invalid reservation counter: {name}")
            ceiling = ceilings[name]
            if ceiling is None:
                _fail("flight_execution_refused", f"work ceiling remains unmeasured: {name}")
            if self._state["charged"][name] + amount > ceiling:
                _fail("flight_budget_exhausted", f"prospective counter exhausted: {name}")
            if (stage is not None and name in STAGE_SENSITIVE_COUNTERS
                    and name not in stage_ceilings[stage]):
                _fail(
                    "flight_stage_counter",
                    f"{name} is not permitted in preregistered stage {stage}",
                )
            if stage is not None and name in stage_ceilings[stage]:
                if self._state["stage_counts"][stage][name] + amount > stage_ceilings[stage][name]:
                    _fail("flight_stage_exhausted", f"prospective stage counter exhausted: {stage}.{name}")
            normalized[name] = amount

        reservation_id = f"r{self._state['reservations'] + 1:08d}"
        payload = {
            "reservation_id": reservation_id,
            "increments": normalized,
            "scope": scope,
            "cleanup": cleanup,
            "stage": stage,
        }
        self._commit("reserve", payload)
        return reservation_id

    def reserve_host_start(self) -> str:
        return self.reserve_many(
            {"host_starts": 1, "startup_operations": 1}, scope="host_start")

    def reserve_reset(self, *, tuning: bool = False,
                      stage: str | None = None) -> str:
        if type(tuning) is not bool:
            _fail("flight_reset_scope", "tuning reset marker must be boolean")
        if tuning:
            if stage not in (None, "tuning"):
                _fail("flight_reset_scope", "tuning reset cannot be charged to another stage")
            stage = "tuning"
        increments = {"resets": 1, "host_calls": 1}
        if stage == "tuning":
            increments["tuning_episodes"] = 1
        return self.reserve_many(increments, scope="host_reset", stage=stage)

    def reserve_host_call(self, *, scope: str, stage: str | None = None) -> str:
        if type(scope) is not str or not scope:
            _fail("flight_host_call_scope", "host call scope is required")
        return self.reserve_many({"host_calls": 1}, scope=scope, stage=stage)

    def reserve_step(self, *, maximum_ticks: int = 6, capture_labels: int = 0,
                     stage: str | None = None, decision: bool = True) -> str:
        if type(maximum_ticks) is not int or not 1 <= maximum_ticks <= 6:
            _fail("flight_step_reservation", "step capacity must be one through six physical ticks")
        if type(capture_labels) is not int or capture_labels < 0:
            _fail("flight_step_reservation", "capture label reservation must be nonnegative")
        if type(decision) is not bool:
            _fail("flight_step_reservation", "decision marker must be boolean")
        if capture_labels and not decision:
            _fail("flight_capture_accounting", "null continuation cannot reserve a capture label")
        if capture_labels and stage is None:
            stage = "capture"
        if stage is None:
            stage = self._state["current_stage"]
        increments = {
            "host_calls": 1,
            "physical_ticks": maximum_ticks,
        }
        if decision:
            increments["decisions"] = 1
        if capture_labels:
            if stage != "capture":
                _fail("flight_capture_accounting", "capture labels may only be charged to capture")
            increments["capture_labels"] = capture_labels
        return self.reserve_many(increments, scope="host_step", stage=stage)

    def reserve_optimizer(self, *, updates: int = 1, presentations: int,
                          seed: int | None = None) -> str:
        stage = None
        if seed is not None:
            if seed not in POLICY_SEEDS:
                _fail("flight_clone_seed", "optimizer reservation names an undeclared policy seed")
            stage = f"clone:{seed}"
        return self.reserve_many(
            {"optimizer_updates": updates, "sample_presentations": presentations},
            scope="optimizer", stage=stage)

    def reserve_tuning_episode(self) -> str:
        return self.reserve_many(
            {"tuning_episodes": 1}, scope="tuning_episode", stage="tuning")

    def reserve_checkpoint_publication(self, *, seed: int | None = None) -> str:
        stage = None
        if seed is not None:
            if seed not in POLICY_SEEDS:
                _fail("flight_clone_seed", "checkpoint reservation names an undeclared policy seed")
            stage = f"clone:{seed}"
        return self.reserve_many(
            {"checkpoint_publications": 1}, scope="checkpoint_publication", stage=stage)

    def reserve_cleanup(self, *, host_call: bool = False) -> str:
        increments = {"cleanup_operations": 1}
        if host_call:
            increments["host_calls"] = 1
        return self.reserve_many(increments, scope="cleanup", cleanup=True)

    def complete(self, reservation_id: str, *, confirmed_physical_ticks: int = 0) -> None:
        self.resource_checkpoint("before_complete")
        payload = {
            "reservation_id": reservation_id,
            "confirmed_physical_ticks": confirmed_physical_ticks,
        }
        self._commit("complete", payload)

    def fail(self, reservation_id: str, reason: str, *,
             confirmed_physical_ticks: int = 0) -> None:
        payload = {
            "reservation_id": reservation_id,
            "reason": reason,
            "confirmed_physical_ticks": confirmed_physical_ticks,
        }
        self._commit("failure", payload)

    def mark_unknown(self, reservation_id: str, reason: str, *,
                     confirmed_physical_ticks: int = 0) -> None:
        payload = {
            "reservation_id": reservation_id,
            "reason": reason,
            "confirmed_physical_ticks": confirmed_physical_ticks,
        }
        self._commit("unknown", payload)

    def abort(self, reason: str) -> None:
        """Persist a sticky post-receipt integrity/semantic failure.

        Use this only after any associated reservation has already been settled.
        It preserves the confirmed work and adds the later validation failure
        rather than rewriting the original host/optimizer outcome.
        """
        if type(reason) is not str or not reason:
            _fail("flight_attempt_failure", "nonempty failure reason required")
        self._commit("attempt_failure", {"reason": reason})

    def publish_checkpoint(self, reservation_id: str, *, checkpoint_id: str,
                           sha256: str, parent_id: str | None) -> None:
        payload = {
            "reservation_id": reservation_id,
            "checkpoint_id": checkpoint_id,
            "sha256": sha256,
            "parent_id": parent_id,
        }
        self._commit("checkpoint", payload)

    def close(self, reason: str = "completed_accounting") -> dict[str, Any]:
        if self._state["closed"]:
            return self.snapshot()
        for reservation_id in list(self._state["pending"]):
            self.mark_unknown(reservation_id, "closed_with_unresolved_reserved_work")
        payload = {"reason": reason}
        self._commit("attempt_close", payload)
        try:
            self._stream.flush()
            os.fsync(self._stream.fileno())
        finally:
            self._stream.close()
        sync_directory(self.path.parent)
        return self.snapshot()
