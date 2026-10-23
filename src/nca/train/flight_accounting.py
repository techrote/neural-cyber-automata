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


DECLARATION_SCHEMA = "nca-flight-experiment-declaration/v2"
LEDGER_RECORD_SCHEMA = "nca-flight-accounting-record/v2"
ZERO_HASH = "0" * 64
MAX_RECORD_BYTES = 64 * 1024
POLICY_SEEDS = (74011, 74013, 74017)

HOST_CONTRACT_HEAD = "2e20731de0559ed59280d7a11f84f596d1cba985"
TRAINER_PROFILE_SHA256 = "f1f9286a27b929c96caca5aaf2be70cbd82e0fd933f29968c1aaaa43ac8f1704"
CONTROLLER_SHA256 = "fb86190c68c6a87e7d2ef7e845b1fb97305f188723a326f67f07caa6971cc8ae"
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
    "physical_ticks": 1_475_280,
    "decisions": 245_880,
    "capture_labels": 11_400,
    "optimizer_updates": 5_400,
    "sample_presentations": 684_000,
    "tuning_episodes": 480,
    "checkpoint_publications": 63,
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
    "accepted #229 exact host revision/build and accepted 4LY-003 evidence identity",
    "accepted Trainer profile, controller and scripted-teacher digests",
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
        "schema", "experiment_id", "status", "identities", "interfaces",
        "scenario", "splits", "runtime", "seeds", "work_ceilings",
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

    identities = _exact_keys(value["identities"], {
        "learner_revision", "learner_source_sha256", "host_revision",
        "host_build_sha256", "profile_sha256", "controller_sha256", "teacher_sha256",
    }, "identities")
    _revision_or_none(identities["learner_revision"], "learner_revision")
    _sha256_or_none(identities["learner_source_sha256"], "learner_source_sha256")
    _revision_or_none(identities["host_revision"], "host_revision")
    for name in ("host_build_sha256", "profile_sha256", "controller_sha256", "teacher_sha256"):
        _sha256_or_none(identities[name], name)

    interfaces = _exact_keys(value["interfaces"], {
        "observation_schema", "flat_schema", "action_schema",
        "observation_channels", "action_outputs", "decision_ticks",
    }, "interfaces")
    if interfaces != {
        "observation_schema": "4ly-observation/v1",
        "flat_schema": "4ly-flat-observation/v1",
        "action_schema": "4ly-action/v1",
        "observation_channels": 54,
        "action_outputs": 2,
        "decision_ticks": 6,
    }:
        _fail("flight_interface_contract", "54-channel/two-output/six-tick contract changed")

    scenario = _exact_keys(value["scenario"], {"manifest_sha256", "family_count"}, "scenario")
    _sha256_or_none(scenario["manifest_sha256"], "scenario.manifest_sha256")
    if scenario["family_count"] != 6:
        _fail("flight_scenario_contract", "the bounded design requires six frozen families")

    splits = _exact_keys(value["splits"], {
        "training_sha256", "tuning_sha256", "final_sha256", "opened_final",
    }, "splits")
    for name in ("training_sha256", "tuning_sha256", "final_sha256"):
        _sha256_or_none(splits[name], f"splits.{name}")
    if splits["opened_final"] is not False:
        _fail("flight_final_access", "final split contents must remain unopened")

    runtime = _exact_keys(value["runtime"], {
        "device", "platform", "python", "torch", "threads",
    }, "runtime")
    if runtime["device"] != "cpu":
        _fail("flight_runtime", "this preparation declares the small CPU learning path only")
    if type(runtime["threads"]) is not int or runtime["threads"] <= 0:
        _fail("flight_runtime", "runtime thread count must be a positive integer")
    for name in ("platform", "python", "torch"):
        if runtime[name] is not None and (type(runtime[name]) is not str or not runtime[name]):
            _fail("flight_runtime", f"runtime.{name} must be null or a nonempty identity")

    if value["seeds"] != list(POLICY_SEEDS):
        _fail("flight_seed_contract", "the three fixed policy seeds changed")

    ceilings = _exact_keys(value["work_ceilings"], set(WORK_COUNTERS), "work_ceilings")
    for name in WORK_COUNTERS:
        _positive_int_or_none(ceilings[name], f"work_ceilings.{name}")
    for name, maximum in DESIGN_WORK_MAXIMA.items():
        if ceilings[name] is not None and ceilings[name] > maximum:
            _fail("flight_declaration_ceiling",
                  f"work_ceilings.{name} exceeds the bounded PR #85 design")

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
    """Return the intentionally incomplete, non-admissible PR #85 template."""
    pending = {name: {"status": "pending", "evidence_sha256": None} for name in DEPENDENCIES}
    return FlightDeclaration.from_mapping({
        "schema": DECLARATION_SCHEMA,
        "experiment_id": "4ly004-trainer-flight-v1",
        "status": "preparation_only",
        "identities": {
            "learner_revision": None,
            "learner_source_sha256": None,
            "host_revision": None,
            "host_build_sha256": None,
            "profile_sha256": None,
            "controller_sha256": None,
            "teacher_sha256": None,
        },
        "interfaces": {
            "observation_schema": "4ly-observation/v1",
            "flat_schema": "4ly-flat-observation/v1",
            "action_schema": "4ly-action/v1",
            "observation_channels": 54,
            "action_outputs": 2,
            "decision_ticks": 6,
        },
        "scenario": {"manifest_sha256": None, "family_count": 6},
        "splits": {
            "training_sha256": None,
            "tuning_sha256": None,
            "final_sha256": None,
            "opened_final": False,
        },
        "runtime": {
            "device": "cpu",
            "platform": None,
            "python": None,
            "torch": None,
            "threads": 1,
        },
        "seeds": list(POLICY_SEEDS),
        "work_ceilings": {
            "host_starts": DESIGN_WORK_MAXIMA["host_starts"],
            "resets": DESIGN_WORK_MAXIMA["resets"],
            "host_calls": None,
            "physical_ticks": DESIGN_WORK_MAXIMA["physical_ticks"],
            "decisions": DESIGN_WORK_MAXIMA["decisions"],
            "capture_labels": DESIGN_WORK_MAXIMA["capture_labels"],
            "optimizer_updates": DESIGN_WORK_MAXIMA["optimizer_updates"],
            "sample_presentations": DESIGN_WORK_MAXIMA["sample_presentations"],
            "tuning_episodes": DESIGN_WORK_MAXIMA["tuning_episodes"],
            "checkpoint_publications": DESIGN_WORK_MAXIMA["checkpoint_publications"],
            "startup_operations": None,
            "cleanup_operations": None,
        },
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
    if value["scenario"]["manifest_sha256"] is None:
        reasons.append("scenario_manifest_unresolved")
    for name in ("training_sha256", "tuning_sha256", "final_sha256"):
        if value["splits"][name] is None:
            reasons.append(f"split_unresolved:{name}")
    for name in ("platform", "python", "torch"):
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

    if kind == "reserve":
        if set(payload) != {"reservation_id", "increments", "scope", "cleanup"}:
            _fail("flight_ledger_shape", "reservation has an invalid shape")
        expected = f"r{state['reservations'] + 1:08d}"
        if payload["reservation_id"] != expected:
            _fail("flight_ledger_sequence", "reservation identity changed")
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
            parent = payload["parent_id"]
            if parent is not None and parent not in state["checkpoints"]:
                _fail("flight_checkpoint_lineage", "checkpoint parent is absent")
            if checkpoint_id in state["checkpoints"]:
                _fail("flight_checkpoint_lineage", "checkpoint identity was published twice")
            state["checkpoints"][checkpoint_id] = {
                "sha256": payload["sha256"], "parent_id": parent,
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
                 stream, state: dict[str, Any], current_evidence: Mapping[str, str]):
        self.path = path
        self.declaration = declaration
        self.attempt_id = attempt_id
        self._stream = stream
        self._state = state
        self._evidence = dict(current_evidence)

    @classmethod
    def create(cls, path: str | Path, declaration: FlightDeclaration, *, attempt_id: str,
               current_evidence: Mapping[str, str]) -> "FlightAccountingLedger":
        if type(attempt_id) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", attempt_id):
            _fail("flight_attempt_id", "attempt identity is invalid")
        gate = execution_gate(declaration, current_evidence=current_evidence)
        if not gate["admitted"]:
            _fail("flight_execution_refused", ",".join(gate["reasons"]))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stream = path.open("xb")
        except FileExistsError as exc:
            raise ContractError("flight_attempt_exists", "attempt journal already exists") from exc
        try:
            sync_directory(path.parent)
            state = _empty_state(declaration, attempt_id)
            ledger = cls(path, declaration, attempt_id, stream, state, current_evidence)
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
               current_evidence: Mapping[str, str]) -> "FlightAccountingLedger":
        path = Path(path)
        state = _replay(path, declaration, attempt_id)
        if state["closed"]:
            _fail("flight_attempt_consumed", "closed attempt cannot be resumed or replayed")
        stream = path.open("ab")
        ledger = cls(path, declaration, attempt_id, stream, state, current_evidence)
        for reservation_id in list(state["pending"]):
            ledger.mark_unknown(
                reservation_id, "interrupted_after_reservation_before_durable_outcome")
        ledger._commit("interruption", {"reason": "process_restarted_for_readback_or_cleanup"})
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

    def _gate_before_work(self) -> None:
        gate = execution_gate(
            self.declaration, current_evidence=self._evidence,
            ledger_snapshot=self.snapshot())
        if not gate["admitted"]:
            _fail("flight_execution_refused", ",".join(gate["reasons"]))

    def reserve_many(self, increments: Mapping[str, int], *, scope: str,
                     cleanup: bool = False) -> str:
        if self._state["closed"] or self._state["poisoned"]:
            _fail("flight_ledger_closed", "closed or uncertain attempt cannot reserve work")
        if self._state["pending"]:
            _fail("flight_ledger_pending", "previous reservation lacks a durable outcome")
        if type(cleanup) is not bool or type(scope) is not str or not scope:
            _fail("flight_ledger_reservation", "reservation scope/cleanup marker is invalid")
        if not isinstance(increments, Mapping) or not increments:
            _fail("flight_ledger_reservation", "nonempty increments are required")
        if (self._state["failed"] or self._state["unknown_consumption"]) and not cleanup:
            _fail("flight_execution_refused",
                  "failed or unknown attempt permits cleanup only")
        if cleanup and not set(increments) <= {"cleanup_operations", "host_calls"}:
            _fail("flight_cleanup_scope", "cleanup can reserve only cleanup operations and host calls")
        if not cleanup:
            self._gate_before_work()

        ceilings = self.declaration.to_dict()["work_ceilings"]
        normalized: dict[str, int] = {}
        for name, amount in increments.items():
            if name not in WORK_COUNTERS or type(amount) is not int or amount <= 0:
                _fail("flight_ledger_counter", f"invalid reservation counter: {name}")
            ceiling = ceilings[name]
            if ceiling is None:
                _fail("flight_execution_refused", f"work ceiling remains unmeasured: {name}")
            if self._state["charged"][name] + amount > ceiling:
                _fail("flight_budget_exhausted", f"prospective counter exhausted: {name}")
            normalized[name] = amount

        reservation_id = f"r{self._state['reservations'] + 1:08d}"
        payload = {
            "reservation_id": reservation_id,
            "increments": normalized,
            "scope": scope,
            "cleanup": cleanup,
        }
        self._commit("reserve", payload)
        return reservation_id

    def reserve_host_start(self) -> str:
        return self.reserve_many(
            {"host_starts": 1, "startup_operations": 1}, scope="host_start")

    def reserve_reset(self, *, tuning: bool = False) -> str:
        if type(tuning) is not bool:
            _fail("flight_reset_scope", "tuning reset marker must be boolean")
        increments = {"resets": 1, "host_calls": 1}
        if tuning:
            increments["tuning_episodes"] = 1
        return self.reserve_many(increments, scope="host_reset")

    def reserve_host_call(self, *, scope: str) -> str:
        if type(scope) is not str or not scope:
            _fail("flight_host_call_scope", "host call scope is required")
        return self.reserve_many({"host_calls": 1}, scope=scope)

    def reserve_step(self, *, maximum_ticks: int = 6, capture_labels: int = 0) -> str:
        if type(maximum_ticks) is not int or not 1 <= maximum_ticks <= 6:
            _fail("flight_step_reservation", "step capacity must be one through six physical ticks")
        if type(capture_labels) is not int or capture_labels < 0:
            _fail("flight_step_reservation", "capture label reservation must be nonnegative")
        increments = {
            "host_calls": 1,
            "physical_ticks": maximum_ticks,
            "decisions": 1,
        }
        if capture_labels:
            increments["capture_labels"] = capture_labels
        return self.reserve_many(increments, scope="host_step")

    def reserve_optimizer(self, *, updates: int = 1, presentations: int) -> str:
        return self.reserve_many(
            {"optimizer_updates": updates, "sample_presentations": presentations},
            scope="optimizer")

    def reserve_tuning_episode(self) -> str:
        return self.reserve_many({"tuning_episodes": 1}, scope="tuning_episode")

    def reserve_checkpoint_publication(self) -> str:
        return self.reserve_many(
            {"checkpoint_publications": 1}, scope="checkpoint_publication")

    def reserve_cleanup(self, *, host_call: bool = False) -> str:
        increments = {"cleanup_operations": 1}
        if host_call:
            increments["host_calls"] = 1
        return self.reserve_many(increments, scope="cleanup", cleanup=True)

    def complete(self, reservation_id: str, *, confirmed_physical_ticks: int = 0) -> None:
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
