"""One declared authored-teacher diagnostic; no learner or checkpoint imports.

All cases are already-open development cases. This comparison cannot admit a
learned skill, select a checkpoint, or change the morphology prerequisite.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping
import zlib

from nca.adapters.godot import CybersandEnv, GodotHost, inspect_host, physical_config
from nca.contracts import ContractError, WireAction, canonical_json, sha256_json, strict_json_loads
from nca.controllers.baseline import AuthoredController, FeedbackObservation, SkillGoal
from nca.controllers.morphology import MorphologyAdapter
from nca.controllers.morphology.conformance import cancellation_receipt, validate_interval
from nca.data import ActionSource, EpisodeRecorder, PolicyContext, read_episode, write_episode
from nca.data.capture import source_identity
from nca.data.records import json_copy
from nca.runs import RunBudget
from nca.train.task import HopTask
from nca.evaluation.durable_journal import (DurableJournal, file_identity, publish_json,
    seal_owner_journal, sync_directory, verify_sealed_journal)
from nca.evaluation.durable_publication import (diagnostic_artifact_inventory,
    publish_json_v2, seal_owner_journal_v2, verify_sealed_journal_v2)
from nca.evaluation.jolt_character import character_metrics, summarize_character
from nca.evaluation.jolt_decisions import (SCHEMA as CONTROLLER_DECISION_SCHEMA,
    FAILURE_SCHEMA as CONTROLLER_FAILURE_SCHEMA, CLEANUP_SCHEMA,
    confirmed_cleanup, decision_accounting, replay_controller_frames, validate_decision_failure)


DECLARATION_SHA256 = "269cef7a852fe8d690bb95ab9187576a3babe0f1fa5d1e8ee28fafc2730c474a"
FAMILIES = ("authored-balanced-v1", "jolt-arrival-v1")
RECOVERY_FAMILIES = (*FAMILIES, "jolt-recovery-v1")
RECOVERY_V2_FAMILIES = (*FAMILIES, "jolt-recovery-v2")
RECOVERY_V3_FAMILIES = (*FAMILIES, "jolt-recovery-v3")
RECOVERY_V4_FAMILIES = (*FAMILIES, "jolt-recovery-v4")
V4_BUDGET = {
    "maximum_episodes": 96, "maximum_decisions": 4000,
    "maximum_requested_ticks": 24000, "maximum_host_requests": 4500,
    "maximum_wall_seconds": 900, "maximum_host_starts": 1,
    "maximum_episode_decisions": 40, "maximum_episode_ticks": 240,
    "optimizer_updates": 0, "checkpoint_executions": 0,
    "automatic_retry": False, "automatic_budget_extension": False,
    "maximum_attempts": 1, "learned_policy_calls": 0,
    "shutdown_reserve_seconds": 120, "maximum_owner_transport_seconds": 15,
    "independent_verification_reserve_seconds": 60,
    "supervisor_finalization_reserve_seconds": 6,
}
PRESERVED_SOURCES = {
    "src/nca/controllers/baseline.py": "19929806429698977cdba82b855c52d6362d1dd2bb4e2aa7161e41bb1bb3f210",
    "src/nca/controllers/shaping.py": "655719bb3e8faf6b3c4ff8c94d6db8624285828549adfd6de86b83f2c7597de8",
    "src/nca/train/task.py": "1e1a173b2b3228ec419a08cf9539ad285b65b36569e82d1e5d4e8f37ffa3d0ec",
    "src/nca/adapters/host_bridge.gd": "f58eb49369fff2b98f7d16da16c0531522aff589bf3008f6fff01972c9629938",
}


def write_json(path: Path, value: Mapping[str, Any], *, direct=False) -> str:
    """Exclusive, durable publication; an existing result is never replaced."""
    return (publish_json_v2 if direct else publish_json)(path, value)


def read_declaration(path: Path, *, expected_sha256=DECLARATION_SHA256):
    value = strict_json_loads(path.read_bytes())
    return _validate_declaration(value, expected_sha256=expected_sha256)


def _validate_declaration(value, *, expected_sha256):
    if (not isinstance(value, dict) or not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or sha256_json(value) != expected_sha256):
        raise ContractError("arrival_declaration", "the frozen prospective diagnostic declaration is required")
    if expected_sha256 != DECLARATION_SHA256:
        # A separately published continuation changes only named controllers
        # and the explicit allocation. It cannot repurpose opened cases or
        # silently loosen mechanics, task outcomes or execution semantics.
        declared_families = {"nca-jolt-recovery-diagnostic/v1": RECOVERY_FAMILIES,
                             "nca-jolt-recovery-diagnostic/v2": RECOVERY_V2_FAMILIES,
                             "nca-jolt-recovery-diagnostic/v3": RECOVERY_V3_FAMILIES,
                             "nca-jolt-recovery-diagnostic/v4": RECOVERY_V4_FAMILIES}
        if (type(value.get("schema")) is not str or value["schema"] not in declared_families
                or value.get("families") != list(declared_families[value["schema"]])):
            raise ContractError("arrival_declaration", "the separately declared three-family recovery diagnostic is required")
        if (value.get("case_source", {}).get("path") != "configs/first-skill/jolt-direction-frame-v1.json"
                or value["case_source"].get("sha256") != "af936549f71bc72bc7fabfc8c75b2d04b58a8348635b5f47ddec50ea904a7687"
                or any(value.get("preserved_sources", {}).get(path) != digest
                       for path, digest in PRESERVED_SOURCES.items())):
            raise ContractError("arrival_declaration", "the original cases, controller, task and host bridge must remain pinned")
        limits = value.get("budget", {})
        maximums = {"maximum_episodes": 96, "maximum_decisions": 3840,
                    "maximum_requested_ticks": 23040, "maximum_host_requests": 4226,
                    "maximum_wall_seconds": 900, "maximum_host_starts": 1,
                    "maximum_episode_decisions": 40, "maximum_episode_ticks": 240}
        is_v4 = value["schema"] == "nca-jolt-recovery-diagnostic/v4"
        if is_v4:
            maximums.update(maximum_decisions=4000, maximum_requested_ticks=24000,
                            maximum_host_requests=4500)
        if (any(type(limits.get(key)) is not int or not 0 < limits[key] <= cap
                for key, cap in maximums.items())
                or limits.get("automatic_retry") is not False
                or limits.get("automatic_budget_extension") is not False
                or any(type(limits.get(key)) is not int or limits[key] != 0
                       for key in ("optimizer_updates", "checkpoint_executions"))):
            raise ContractError("arrival_declaration", "bounded one-invocation authored-only allocation is required")
        physics = value.get("physics", {})
        if any(type(physics.get(key)) is not type(expected) or physics[key] != expected for key, expected in {
                "mode": "stepped", "base_hz": 60, "decision_ticks": 6,
                "helper_lease_ticks": 12, "host_mechanics_changed": False}.items()):
            raise ContractError("arrival_declaration", "the original physical cadence and assistance are required")
        if (value.get("goal") != {"pace": .8, "tolerance_cells": 4., "deadline_ticks": 240,
                "quiet_ticks": 12, "maximum_arrival_speed": 6.,
                "maximum_arrival_angle": .32, "require_support": True}
                or value["goal"].get("require_support") is not True):
            raise ContractError("arrival_declaration", "the original task success predicate is required")
        if value["schema"] in ("nca-jolt-recovery-diagnostic/v3", "nca-jolt-recovery-diagnostic/v4"):
            from nca.evaluation.jolt_recovery_acceptance import acceptance_specification
            # V3 is a new, complete allocation. Check its unchanged teacher
            # rule before starting a process, not only when auditing outcomes.
            if sha256_json(value.get("teacher_acceptance")) != sha256_json(
                    acceptance_specification(RECOVERY_V4_FAMILIES[2] if is_v4 else RECOVERY_V3_FAMILIES[2])):
                raise ContractError("arrival_declaration", "the new candidate must retain the prospectively fixed teacher acceptance")
            if (any(limits[key] != cap for key, cap in maximums.items())
                    or type(limits.get("maximum_attempts")) is not int or limits["maximum_attempts"] != 1
                    or type(limits.get("learned_policy_calls")) is not int or limits["learned_policy_calls"] != 0):
                raise ContractError("arrival_declaration", "v3 requires its complete new one-attempt numeric allocation")
            preserved = {
                "src/nca/controllers/jolt_arrival.py": "b9eb8304aa4c481c253ba1c39d39ae7a2bba03fe9e1d8bf867eef9c96f7a500b",
                "src/nca/controllers/jolt_recovery.py": "5bc64ff9e09092620ea74f7a6ab881e02acb0e984028674d4d4a5186ac4c9528",
                "src/nca/controllers/jolt_recovery_v2.py": "e919d10d78ba1d9dacecf135014f9e344a1ea7e43817f84d29e067577565fdae",
                "src/nca/controllers/recovery_supervisor.py": "e9a0b19aca661643c4a72acd0293ade72b5b569b8adb9abfca201de70205d09d",
            }
            if is_v4:
                preserved["src/nca/controllers/jolt_recovery_v3.py"] = "367447b4dd732a8e999f1d172795685fa4591028026ebbdd2697d12dc0c2aab3"
                if (set(limits) != set(V4_BUDGET)
                        or any(type(limits[key]) is not type(expected) or limits[key] != expected
                               for key, expected in V4_BUDGET.items())):
                    raise ContractError("arrival_declaration", "v4 requires its exact global, per-episode and reserve allocation")
                inventory_sha256 = "b48eeeffa3b07f0a12a8cc2736fd0ed80f500e95b00bb69f9e46fb2c3fabd06d"
                if (value.get("case_inventory_sha256") != inventory_sha256
                        or sha256_json(value.get("case_inventory")) != inventory_sha256
                        or sha256_json(value.get("prior_invocations")) != "0ae1d52a22c97e0e191fc2e4385b53850682d182caac4fa02cbdbac562167448"):
                    raise ContractError("arrival_declaration", "v4 must retain all original case/seed identities and previous invocation costs/failures")
                if (set(value.get("candidate_specifications", {})) != set(RECOVERY_V4_FAMILIES[1:])
                        or any(value["candidate_specifications"].get(family) != _controller(family).specification()
                               for family in RECOVERY_V4_FAMILIES[1:])):
                    raise ContractError("arrival_declaration", "v4 requires both exact prospective controller specifications")
            if any(value["preserved_sources"].get(path) != digest for path, digest in preserved.items()):
                raise ContractError("arrival_declaration", "v3 must preserve the arrival, recovery v1/v2 and supervisor sources")
            if (physics.get("host_revision") != "f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497"
                    or physics.get("profile_sha256") != "97559dcca7b1523ee622fd886615f05f5d7f85bacc0655ba9d642e909dc112bc"):
                raise ContractError("arrival_declaration", "v3 must preserve the accepted physical host and profile")
    return value


def _controller(family):
    if family == FAMILIES[0]:
        return AuthoredController("jev")
    if family == FAMILIES[1]:
        from nca.controllers.jolt_arrival import JoltArrivalController
        return JoltArrivalController()
    if family == RECOVERY_FAMILIES[2]:
        from nca.controllers.jolt_recovery import JoltRecoveryController
        return JoltRecoveryController()
    if family == RECOVERY_V2_FAMILIES[2]:
        from nca.controllers.jolt_recovery_v2 import JoltRecoveryControllerV2
        return JoltRecoveryControllerV2()
    if family == RECOVERY_V3_FAMILIES[2]:
        from nca.controllers.jolt_recovery_v3 import JoltRecoveryControllerV3
        return JoltRecoveryControllerV3()
    if family == RECOVERY_V4_FAMILIES[2]:
        from nca.controllers.jolt_recovery_v4 import JoltRecoveryControllerV4
        return JoltRecoveryControllerV4()
    raise ContractError("arrival_family", "undeclared authored controller")


def replay_task_metrics(episode):
    """Recompute the unchanged task from retained physical receipts only."""
    events = [event for event in episode.events if event.kind == "step"]
    frames = replay_controller_frames(episode)
    if not events or len(frames) != len(events) or events[0].context is None:
        raise ContractError("arrival_task_replay", "complete retained goal, decisions and physical receipts are required")
    context = events[0].context
    goal = SkillGoal(context.goal_id, context.skill, context.target, context.pace,
                     context.tolerance, context.timeout_seconds)
    task = HopTask(goal, episode.initial_observation, deadline_ticks=240, quiet_ticks_required=12)
    terms = []
    for event, frame in zip(events, frames):
        if (event.context != context or type(frame.get("inference_ns")) is not int
                or frame["inference_ns"] < 0):
            raise ContractError("arrival_task_replay", "retained goal or measured inference identity differs")
        raw = [{"tick_before": tick.tick_before, "failed": not tick.valid,
                "observation": tick.observation, "admission": tick.admission, "applied": tick.applied}
               for tick in event.ticks]
        terms.append(task.consume(WireAction.from_dict(frame["action"]), raw,
                                  inference_ns=frame["inference_ns"]))
    return task.metrics(administratively_capped=not task.done), terms


def diagnostic_cases(original):
    """Retain each of the original24 training and8 evaluation case/seed pairs."""
    cases = []
    for source in original["training_families"]:
        for index, seed in enumerate(source["demo_seeds"]):
            cases.append({"id": f"demo-{source['id']}-{index}", "family": source["id"],
                          "origin": "previously_open_training", "spawn": source["spawn"],
                          "target": source["target"], "host_seed": seed})
    for source in original["evaluation_families"]:
        cases.append({"id": f"reference-{source['id']}", "family": source["id"],
                      "origin": "previously_open_evaluation", "spawn": source["spawn"],
                      "target": source["target"], "host_seed": source["host_seed"]})
    if len(cases) != 32 or len({row["id"] for row in cases}) != 32:
        raise ContractError("arrival_cases", "exactly32 distinct previously open cases are required")
    return cases


class DiagnosticBudget:
    """Reserve all owner calls before transmission; uncertainty consumes budget."""
    def __init__(self, limits, journal, *, clock=time.monotonic, whole_process_started=None):
        self.limits, self.journal, self.clock = limits, journal, clock
        now = clock()
        self.v4 = "shutdown_reserve_seconds" in limits
        if self.v4:
            if (type(whole_process_started) not in (int, float)
                    or not math.isfinite(whole_process_started) or not 0 < whole_process_started <= now):
                raise ContractError("arrival_wall_origin", "v4 needs its original supervised process start")
            self.started = whole_process_started
        else:
            self.started = now
        self.counts = {"host_requests": 0, "step_requests": 0, "requested_step_ticks": 0,
                       "confirmed_step_ticks": 0, "episodes_started": 0,
                       "host_start_attempts": 0, "host_terminations": 0}
        self.pending = {}
        self._active_episode = None
        self._episode_decisions = self._episode_ticks = 0

    def note(self, event, **values):
        self.journal.write((canonical_json({"event": event,
            "elapsed_seconds": self.clock() - self.started, **values}) + "\n").encode())
        self.journal.flush()
        if self.v4 and event == "episode_completed":
            self._active_episode = None

    def check(self):
        if self.clock() - self.started >= self.limits["maximum_wall_seconds"]:
            raise ContractError("arrival_wall_budget", "declared complete invocation ceiling reached")

    def admit_transport(self, operation):
        """Recheck after durable reservation and before spending owner time."""
        if not self.v4:
            return
        self.check()
        active = self.limits["maximum_wall_seconds"] - self.limits["shutdown_reserve_seconds"]
        duration = self.limits["maximum_owner_transport_seconds"]
        if operation in ("cancel", "close"):
            return
        if self.clock() - self.started + duration > active:
            raise ContractError("arrival_shutdown_reserve", "the next bounded owner operation would endanger the reserved close/seal/verification time")

    def start_host(self):
        self.admit_transport("start_host")
        if self.counts["host_start_attempts"] >= self.limits["maximum_host_starts"]:
            raise ContractError("arrival_host_budget", "the sole host allocation is consumed")
        self.counts["host_start_attempts"] += 1
        self.note("host_start_reserved")
        self.admit_transport("start_host")

    def start_episode(self, case, family):
        self.check()
        self.admit_transport("start_episode")
        if self.v4 and (self._active_episode is not None or self.pending):
            raise ContractError("arrival_episode_budget", "the previous episode or request has not closed")
        if self.counts["episodes_started"] >= self.limits["maximum_episodes"]:
            raise ContractError("arrival_episode_budget", "declared episode allocation exhausted")
        self.counts["episodes_started"] += 1
        self._active_episode = (case, family)
        self._episode_decisions = self._episode_ticks = 0
        self.note("episode_started", case=case, family=family)

    def reserve(self, operation, fields):
        if self.v4:
            self.admit_transport(operation)
            if self.pending:
                raise ContractError("arrival_owner_budget", "an uncertain prior request cannot admit another owner operation")
            if operation not in ("cancel", "close") and self._active_episode is None:
                raise ContractError("arrival_episode_budget", "owner work requires a currently admitted episode")
        if operation not in ("cancel", "close"):
            self.check()
        if self.counts["host_requests"] >= self.limits["maximum_host_requests"]:
            raise ContractError("arrival_owner_budget", "declared owner-request allocation exhausted")
        ticks = fields.get("ticks", 0) if operation == "step" else 0
        if operation == "step":
            if (type(ticks) is not int or ticks != 6
                    or self.counts["step_requests"] >= self.limits["maximum_decisions"]
                    or self.counts["requested_step_ticks"] + ticks > self.limits["maximum_requested_ticks"]):
                raise ContractError("arrival_step_budget", "declared six-tick decision allocation exhausted")
            if self.v4 and (self._episode_decisions >= self.limits["maximum_episode_decisions"]
                    or self._episode_ticks + ticks > self.limits["maximum_episode_ticks"]):
                raise ContractError("arrival_step_budget", "the unchanged per-episode decision/tick boundary is exhausted")
            self.counts["step_requests"] += 1
            self.counts["requested_step_ticks"] += ticks
            self._episode_decisions += 1
            self._episode_ticks += ticks
        self.counts["host_requests"] += 1
        serial = self.counts["host_requests"]
        self.pending[serial] = {"serial": serial, "operation": operation, "requested_ticks": ticks}
        self.note("request_reserved", **self.pending[serial], fields=fields)
        return serial

    def confirm(self, serial, response):
        pending = self.pending[serial]
        self.note("response_received", serial=serial, response=response)
        if pending["operation"] == "step":
            actual = response.get("actual_ticks")
            if type(actual) is not int or not 0 <= actual <= pending["requested_ticks"]:
                raise ContractError("arrival_tick_receipt", "native actual tick receipt is malformed")
            self.counts["confirmed_step_ticks"] += actual
        del self.pending[serial]

    def snapshot(self):
        result = {**self.counts, "pending_requests": list(self.pending.values()),
                "unconfirmed_tail": bool(self.pending), "wall_seconds": self.clock() - self.started,
                "optimizer_updates": 0, "checkpoint_executions": 0, "learned_policy_calls": 0}
        if self.v4:
            result["whole_process_started_monotonic"] = self.started
        return result


class _AbsoluteDeadlineSocket:
    """Cap a complete diagnostic owner exchange across all send/recv calls."""

    def __init__(self, connection, deadline, clock):
        self.connection, self.deadline, self.clock = connection, deadline, clock
        self.timeout = connection.gettimeout()

    def __getattr__(self, key):
        return getattr(self.connection, key)

    def settimeout(self, timeout):
        self.timeout = timeout
        self.connection.settimeout(timeout)

    def _arm(self):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError("complete diagnostic owner exchange exceeded its absolute deadline")
        self.connection.settimeout(remaining if self.timeout is None else min(self.timeout, remaining))

    def sendall(self, data, *args, **kwargs):
        self._arm()
        result = self.connection.sendall(data, *args, **kwargs)
        self._arm()
        return result

    def recv(self, size, *args, **kwargs):
        self._arm()
        result = self.connection.recv(size, *args, **kwargs)
        self._arm()
        return result


@contextmanager
def _owner_exchange_deadline(host, budget, operation):
    if not budget.v4:
        yield
        return
    now = budget.clock()
    ceiling = budget.started + budget.limits["maximum_wall_seconds"]
    if operation not in ("cancel", "close"):
        ceiling -= budget.limits["shutdown_reserve_seconds"]
    deadline = min(now + budget.limits["maximum_owner_transport_seconds"], ceiling)
    channel = getattr(host, "_channel", None)
    connection = getattr(channel, "connection", None)
    previous_timeout = connection.gettimeout() if connection is not None else None
    if connection is not None:
        channel.connection = _AbsoluteDeadlineSocket(connection, deadline, budget.clock)
    try:
        yield
        if budget.clock() >= deadline:
            raise ContractError("arrival_owner_deadline", "the complete owner exchange exceeded its absolute deadline")
    finally:
        if connection is not None:
            channel.connection = connection
            try:
                connection.settimeout(previous_timeout)
            except OSError:
                pass  # Native protocol failure may already have closed it.


class MeteredHost:
    def __init__(self, host, budget):
        self.host, self.budget = host, budget

    def __getattr__(self, key):
        return getattr(self.host, key)

    def request(self, operation, **fields):
        serial = self.budget.reserve(operation, fields)
        self.budget.admit_transport(operation)
        with _owner_exchange_deadline(self.host, self.budget, operation):
            response = self.host.request(operation, **fields)
        self.budget.confirm(serial, response)
        return response


def validate_closed_owner_journal(path: Path, *, expected_counts, expected_rows,
                                  families=FAMILIES, artifact_root: Path | None = None):
    """Replay the closed gzip completely before trusting its in-memory summary.

    This is an offline accounting check, not a repair or a host invocation. A
    valid gzip footer alone is insufficient: every declared episode, request,
    native response and final close must reconcile with the retained results.
    """
    def require(condition, code, detail):
        if not condition:
            raise ContractError(f"arrival_journal_{code}", detail)

    counters = ("host_requests", "step_requests", "requested_step_ticks",
                "confirmed_step_ticks", "episodes_started", "host_start_attempts",
                "host_terminations")
    require(isinstance(expected_counts, Mapping) and all(
        type(expected_counts.get(key)) is int and expected_counts[key] >= 0 for key in counters),
        "counts", "complete nonnegative expected counters are required")
    require(expected_counts.get("pending_requests") == []
        and expected_counts.get("unconfirmed_tail") is False
        and all(type(expected_counts.get(key)) is int and expected_counts[key] == 0
                for key in ("optimizer_updates", "checkpoint_executions", "learned_policy_calls")),
        "counts", "pending or learned activity cannot support diagnostic completion")
    wall = expected_counts.get("wall_seconds")
    require(type(wall) in (int, float) and math.isfinite(wall) and wall >= 0,
            "counts", "a finite expected accounting interval is required")
    require(expected_rows and all(isinstance(row, Mapping) for row in expected_rows),
            "episodes", "the complete expected episode rows are required")
    keys = [(row.get("case_id"), row.get("family")) for row in expected_rows]
    require(all(type(case) is str and case and family in families for case, family in keys)
        and len(keys) == len(set(keys)) and len(keys) == expected_counts["episodes_started"],
        "episodes", "unique expected case/family rows must match episode accounting")
    counts = dict.fromkeys(counters, 0)
    active, pending, native_close, host_closed = None, None, False, False
    completed, records, last_elapsed, bound_ticks = 0, 0, 0., 0
    try:
        with gzip.open(path, "rb") as source:
            while True:
                # Read through EOF, including the gzip CRC/length trailer. A
                # streaming prefix must never be mistaken for a closed file.
                line = source.readline(1_048_577)
                if not line:
                    break
                records += 1
                require(len(line) <= 1_048_576 and line.endswith(b"\n"),
                        "record", "bounded complete JSONL records are required")
                try:
                    record = strict_json_loads(line)
                except ContractError as exc:
                    raise ContractError("arrival_journal_record", "invalid journal JSON") from exc
                require(isinstance(record, dict), "record", "journal records must be objects")
                elapsed, event = record.get("elapsed_seconds"), record.get("event")
                require(type(elapsed) in (int, float) and math.isfinite(elapsed)
                    and last_elapsed <= elapsed <= wall and not host_closed,
                    "order", "records must be chronological and precede final host closure")
                last_elapsed = elapsed
                if event == "host_start_reserved":
                    require(records == 1, "order", "exactly one initial host start is required")
                    counts["host_start_attempts"] += 1
                elif event == "episode_started":
                    key = (record.get("case"), record.get("family"))
                    require(counts["host_start_attempts"] == 1 and active is None
                        and pending is None and not native_close and completed < len(keys)
                        and key == keys[completed],
                        "episodes", "episodes must start once in the declared result order")
                    counts["episodes_started"] += 1
                    active = {"key": key, "stage": "reset", "decisions": 0, "ticks": 0,
                              "decision_failed": False}
                    if artifact_root is not None:
                        row = expected_rows[completed]
                        relative = row.get("artifact")
                        require(type(relative) is str and relative
                            and not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                            "artifact", "episode artifact must be a relative retained path")
                        episode_path = artifact_root / relative
                        require(file_identity(episode_path)["sha256"] == row.get("artifact_sha256"),
                                "artifact", "episode artifact bytes differ from their indexed hash")
                        episode = read_episode(episode_path)
                        require(episode.sha256 == row.get("record_sha256")
                            and sha256_json(episode.evaluation.get("metrics")) == sha256_json(row.get("metrics"))
                            and sha256_json(episode.evaluation.get("character_metrics")) == sha256_json(row.get("character_metrics")),
                            "artifact", "episode content and evaluation differ from the indexed row")
                        active["episode"] = episode
                        frames = replay_controller_frames(episode)
                        if episode.evaluation.get("controller_decision_schema") == CONTROLLER_DECISION_SCHEMA:
                            require(row.get("controller_decision_schema") == CONTROLLER_DECISION_SCHEMA
                                and all(sha256_json(row.get(name)) == sha256_json(episode.evaluation.get(name))
                                        for name in ("controller_decision_accounting", "error_cleanup")),
                                "artifact", "versioned decision accounting and cleanup differ from the retained episode")
                        active["step_events"] = [event for event in episode.events if event.kind == "step"]
                        first_step = next(iter(active["step_events"]), None)
                        require(first_step is not None and first_step.context is not None,
                                "artifact", "retained episode lacks its explicit caller goal")
                        recomputed = character_metrics(episode.initial_observation, first_step.context.target,
                            episode.events, frames)
                        require(sha256_json(recomputed) == sha256_json(row.get("character_metrics")),
                                "artifact", "retained character metrics differ from physical receipt replay")
                        task_metrics, task_terms = replay_task_metrics(episode)
                        require(episode.evaluation.get("task_reward_schema") == "jolt-arrival-best-progress/v1"
                            and sha256_json(task_metrics) == sha256_json(row.get("metrics"))
                            and sha256_json(task_terms) == sha256_json(episode.evaluation.get("reward_terms")),
                            "artifact", "retained task metrics or rewards differ from physical receipt replay")
                elif event == "request_reserved":
                    serial, operation = record.get("serial"), record.get("operation")
                    fields, ticks = record.get("fields"), record.get("requested_ticks")
                    require(counts["host_start_attempts"] == 1 and pending is None and not native_close
                        and type(serial) is int and serial == counts["host_requests"] + 1
                        and type(operation) is str and isinstance(fields, dict) and type(ticks) is int,
                        "order", "each synchronous request needs its next unique reservation")
                    if operation == "close":
                        require(active is None and completed == len(keys) and fields == {} and ticks == 0,
                                "close", "native close must follow every completed episode")
                    else:
                        stages = {"reset": ("reset",), "profile": ("profile",), "step": ("step",),
                                  "cancel": ("step", "terminal")}
                        require(operation in stages and active is not None
                            and active["stage"] in stages[operation],
                            "order", "episode owner operations are missing or out of order")
                        require(ticks == (6 if operation == "step" else 0)
                            and (operation != "step" or type(fields.get("ticks")) is int
                                 and fields["ticks"] == 6),
                            "counts", "each step must reserve the declared six physical ticks")
                        if operation == "step":
                            require(not active["decision_failed"], "decision",
                                    "a failed local decision must stop further physical requests")
                            counts["step_requests"] += 1
                            counts["requested_step_ticks"] += ticks
                        if operation == "cancel":
                            require(active["decisions"] > 0, "close", "episode closure needs a physical interval")
                    counts["host_requests"] += 1
                    pending = {"serial": serial, "operation": operation, "requested_ticks": ticks,
                               "fields": fields}
                elif event == "response_received":
                    response, serial = record.get("response"), record.get("serial")
                    require(pending is not None and type(serial) is int and serial == pending["serial"],
                            "order", "a native response must match the sole pending reservation")
                    require(isinstance(response, dict) and response.get("protocol") == "nca-host/1"
                        and type(response.get("id")) is int and response["id"] == serial
                        and response.get("ok") is True,
                        "response", "a successful correlated native response is required")
                    operation = pending["operation"]
                    if operation == "close":
                        require(response.get("closed") is True, "close", "native host close was not confirmed")
                        native_close = True
                    elif operation == "profile":
                        require(isinstance(response.get("profile"), dict), "response", "native profile response is missing")
                        active["stage"] = "step"
                    else:
                        view = response.get("observation")
                        require(isinstance(view, dict) and type(view.get("tick")) is int,
                                "response", "native observation and physical tick are required")
                        if operation == "reset":
                            require(view["tick"] == 0 and view.get("suspended") is False,
                                    "response", "an active native reset observation is required")
                            active["stage"] = "profile"
                            active["view"] = view
                            if artifact_root is not None:
                                require(sha256_json(view) == sha256_json(active["episode"].initial_observation),
                                        "artifact", "journal reset differs from the retained episode")
                        elif operation == "cancel":
                            require(response.get("cancelled") is True
                                and view["tick"] == active["ticks"],
                                "close", "native episode cancellation was not confirmed")
                            try:
                                cancellation_receipt(active["view"], view)
                            except (ContractError, KeyError, TypeError) as exc:
                                raise ContractError("arrival_journal_close", "native cancellation must revoke ownership without stepping") from exc
                            if (artifact_root is not None
                                    and active["episode"].evaluation.get("controller_decision_schema") == CONTROLLER_DECISION_SCHEMA):
                                cancellations = [item for item in active["episode"].events if item.kind == "cancel"]
                                require(len(cancellations) == 1
                                    and sha256_json(cancellations[0].issued_observation) == sha256_json(active["view"])
                                    and sha256_json(cancellations[0].next_observation) == sha256_json(view),
                                    "artifact", "native cancellation differs from its retained source and receipt")
                                cleanup = active["episode"].evaluation.get("error_cleanup")
                                if cleanup is not None and cleanup.get("confirmed") is True:
                                    require(cleanup.get("request_serial") == serial,
                                            "artifact", "error cleanup must identify this exact native cancellation")
                            active["stage"] = "closed"
                        else:
                            actual, start, end = (response.get(key) for key in ("actual_ticks", "start_tick", "tick"))
                            traces = response.get("traces")
                            require(type(actual) is int and actual == pending["requested_ticks"] == 6
                                and type(start) is int and start == active["ticks"]
                                and type(end) is int and end == start + actual and view["tick"] == end
                                and isinstance(traces, list) and len(traces) == actual
                                and all(isinstance(trace, dict)
                                    and type(trace.get("tick_before")) is int and trace["tick_before"] == start + index
                                    and type(trace.get("tick_after")) is int and trace["tick_after"] == start + index + 1
                                    for index, trace in enumerate(traces)),
                                "response", "complete contiguous native six-tick receipts are required")
                            require(all(type(response.get(key)) is bool for key in ("terminated", "truncated", "finished")),
                                    "response", "native terminal status is missing")
                            terminal = response["terminated"] or response["truncated"]
                            require(response["finished"] is terminal and view.get("suspended") is terminal,
                                    "close", "native terminal status and ownership closure must agree")
                            if artifact_root is not None:
                                require(active["decisions"] < len(active["step_events"]),
                                        "artifact", "journal has a step absent from the episode")
                                event = active["step_events"][active["decisions"]]
                                require(event.source_observation is not None
                                    and sha256_json(event.source_observation) == sha256_json(active["view"])
                                    and sha256_json(event.issued_observation) == sha256_json(active["view"]),
                                    "artifact", "synchronous controller source and issued views must match the preceding native observation")
                                require(sha256_json(pending["fields"].get("commands"))
                                    == sha256_json([event.requested_packet])
                                    and sha256_json(view) == sha256_json(event.next_observation)
                                    and len(event.ticks) == len(traces),
                                    "artifact", "journal request or returned view differs from the episode")
                                for raw, retained in zip(traces, event.ticks):
                                    shared = ("tick_before", "tick_after", "world_tick", "received_monotonic_usec",
                                              "completed_monotonic_usec", "requested", "admission", "applied",
                                              "passive", "actual_motion", "observation", "cellular_sample_tick")
                                    defaults = {"requested": [], "admission": [], "actual_motion": {}}
                                    require(all(sha256_json(raw.get(key, defaults.get(key))) == sha256_json(getattr(retained, key))
                                                for key in shared),
                                            "artifact", "journal physical receipt differs from the retained episode")
                                    bound_ticks += 1
                            counts["confirmed_step_ticks"] += actual
                            active["decisions"] += 1
                            active["ticks"] = end
                            active["view"] = view
                            if terminal:
                                active["stage"] = "terminal"
                    pending = None
                elif event == "controller_decision_failed":
                    require(active is not None and pending is None and not active["decision_failed"]
                        and active["stage"] in ("step", "terminal")
                        and (record.get("case"), record.get("family")) == active["key"],
                        "decision", "a refused local decision belongs to the active physical source")
                    failure = record.get("decision_failure")
                    validate_decision_failure(failure, active["view"], active["decisions"])
                    require(failure["owner_requests_after"] == counts["host_requests"]
                        and failure["step_requests_after"] == counts["step_requests"],
                        "decision", "refusal reservation counters differ from the native journal")
                    if artifact_root is not None:
                        require(active["episode"].evaluation.get("controller_decision_schema") == CONTROLLER_DECISION_SCHEMA
                            and sha256_json(active["episode"].evaluation.get("controller_decision_failures"))
                                == sha256_json([failure]),
                            "artifact", "the refused decision differs from its retained episode audit")
                    active["decision_failed"] = True
                elif event == "episode_completed":
                    row = record.get("row")
                    require(active is not None and pending is None and active["stage"] in ("terminal", "closed")
                        and (record.get("case"), record.get("family")) == active["key"]
                        and isinstance(row, dict) and sha256_json(row) == sha256_json(expected_rows[completed])
                        and not active["decision_failed"]
                        and row.get("owner_closure_confirmed") is True and row.get("error") is None
                        and type(row.get("decisions")) is int and row["decisions"] == active["decisions"]
                        and type(row.get("consumed_ticks")) is int and row["consumed_ticks"] == active["ticks"],
                        "episodes", "each completed row must match its native intervals and closure")
                    completed += 1
                    active = None
                elif event == "host_closed":
                    require(native_close and active is None and pending is None and completed == len(keys)
                        and record.get("termination_confirmed") is True,
                        "close", "a final confirmed host termination must follow its native close response")
                    counts["host_terminations"] += 1
                    host_closed = True
                else:
                    require(False, "record", "unknown or missing journal event")
    except (EOFError, OSError, zlib.error) as exc:
        raise ContractError("arrival_journal_stream", "the complete gzip stream and trailer are required") from exc
    require(host_closed and native_close and pending is None and active is None and completed == len(keys),
            "incomplete", "the journal has a missing episode, response or final native closure")
    require(counts == {key: expected_counts[key] for key in counters}
        and counts["host_start_attempts"] == counts["host_terminations"] == 1
        and counts["requested_step_ticks"] == counts["confirmed_step_ticks"],
        "counts", "replayed owner accounting differs from the expected complete counters")
    return {"verified": True, "records": records, "counts": counts,
            "episodes_completed": completed, "native_close_confirmed": True, "gzip_eof_verified": True,
            "episode_receipts_bound": artifact_root is not None, "bound_episode_ticks": bound_ticks}


def _error(exc):
    # Exception text and engine diagnostics can contain local paths. Preserve
    # class/code publicly and retain detailed execution logs only locally.
    return {"type": type(exc).__name__, "code": getattr(exc, "code", None)}


def validate_owner_interval(adapter, source, following, *, terminal):
    """Terminal suspension must carry exactly one native ownership revocation."""
    before = adapter.identity(source)
    after = adapter.terminal_identity(following) if terminal else adapter.identity(following)
    if (after.lifetime != before.lifetime
            or after.control_generation != before.control_generation + int(terminal)
            or after.topology_sha256 != before.topology_sha256
            or source.get("mount_generation") != following.get("mount_generation")
            or following.get("tick") != source.get("tick") + 6):
        raise ContractError("arrival_owner_interval", "exact source/physical identity and native terminal revocation are required")


def _capture(host, budget, case, family, output, original, declaration, root):
    budget.start_episode(case["id"], family)
    identifier = f"{family}-{case['id']}"
    settings = physical_config("jev", ticks=240, seed=case["host_seed"])
    settings["actors"][0]["position"] = list(case["spawn"])
    env = CybersandEnv(host, settings, decision_ticks=6, lease_ticks=12)
    controller = _controller(family)
    policy_hash = (original["baseline_source_sha256"] if family == FAMILIES[0]
                   else controller.specification_sha256)
    goal = SkillGoal(identifier, "directed_hop", tuple(case["target"]), .8, 4., 4.)
    source = ActionSource(family, "scripted", controller.controller_id, policy_hash)
    context = PolicyContext(goal.id, goal.skill, goal.target, goal.pace, goal.tolerance, goal.timeout_seconds)
    recorder = EpisodeRecorder(env, identifier, source, context=context, seed=case["host_seed"],
        scenario_family=case["family"], split="evaluation", budget=RunBudget(40, 600), source_root=root)
    if sha256_json(recorder.profile) != original["profile_sha256"]:
        raise ContractError("arrival_profile", "the declared accepted Jolt profile changed")
    adapter = MorphologyAdapter(recorder.profile, expected_sha256=original["profile_sha256"])
    task = HopTask(goal, recorder.initial, deadline_ticks=240, quiet_ticks_required=12)
    phases, terms, error, closure = [], [], None, False
    decision_failures, pending_decision, decision_journal_error, cleanup = [], None, None, None
    try:
        for _ in range(40):
            budget.check()
            budget.admit_transport("controller_decision")
            view = recorder.view
            started = time.perf_counter_ns()
            frame = controller.step(FeedbackObservation.from_host(view), goal)
            inference_ns = time.perf_counter_ns() - started
            frame_record = {"tick": view["tick"], "phase": frame.phase, "status": frame.status,
                            "reason": frame.reason, "action": frame.action.to_dict(),
                            "inference_ns": inference_ns,
                            "diagnostic_state": (controller.diagnostic_state()
                                if callable(getattr(controller, "diagnostic_state", None)) else None)}
            pending_decision = {"schema": CONTROLLER_FAILURE_SCHEMA,
                "frame": frame_record, "source_observation": json_copy(view),
                "after_physical_decisions": len(phases), "requested_packet": None,
                "owner_requests_before": budget.counts["host_requests"],
                "step_requests_before": budget.counts["step_requests"]}
            if family in (RECOVERY_FAMILIES[2], RECOVERY_V2_FAMILIES[2], RECOVERY_V3_FAMILIES[2], RECOVERY_V4_FAMILIES[2]):
                lease_ticks = frame_record["diagnostic_state"].get("action_lease_ticks")
                if (type(lease_ticks) is not int or not 6 <= lease_ticks <= 12
                        or view["tick"] + lease_ticks > 240):
                    raise ContractError("arrival_supervisor_lease", "recovery supervisor must admit a deadline-capped original-source lease")
                packet = env.command(frame.action.to_dict(), observation=view, lease_ticks=lease_ticks)
                pending_decision["requested_packet"] = json_copy(packet)
                after, _, terminated, truncated, info = recorder.step_packet(packet)
            else:
                after, _, terminated, truncated, info = recorder.step(frame.action.to_dict())
            # A returned native step event, not a local status/lease decision,
            # admits this frame to the physical receipt/character sequence.
            phases.append(frame_record)
            pending_decision = None
            event = recorder.events[-1]
            validate_interval(event.requested_packet, event.source_observation, after,
                              terminated, truncated, info, allow_terminal=True)
            validate_owner_interval(adapter, event.source_observation, after,
                                    terminal=terminated or truncated)
            if not event.imitation_eligible or not event.trace_complete:
                raise ContractError("arrival_consumption", "complete unchanged admitted action consumption is required")
            terms.append(task.consume(frame.action, info["trace"], inference_ns=inference_ns))
            if task.integrity_failure:
                raise ContractError("arrival_integrity", "a native refusal or physical integrity failure occurred")
            if task.done:
                break
            if terminated or truncated:
                raise ContractError("arrival_early_terminal", "host stopped before the unchanged task outcome")
        if not task.done:
            raise ContractError("arrival_episode_capped", "fixed episode ended without an intrinsic outcome")
        if task.deadline_missed:
            closure = bool(recorder.view.get("suspended"))
        else:
            before = recorder.view
            canceled = recorder.cancel("diagnostic_task_arrival")
            cancellation_receipt(before, canceled)
            closure = True
        if not closure:
            raise ContractError("arrival_owner_closure", "native owner cancellation or terminal closure was not confirmed")
    except Exception as exc:
        error = _error(exc)
        if pending_decision is not None:
            pending_decision.update(owner_requests_after=budget.counts["host_requests"],
                step_requests_after=budget.counts["step_requests"], error=error)
            pending_decision["disposition"] = ("refused_before_transport"
                if pending_decision["owner_requests_after"] == pending_decision["owner_requests_before"]
                and pending_decision["step_requests_after"] == pending_decision["step_requests_before"]
                else "owner_response_unrecorded")
            decision_failures.append(pending_decision)
            try:
                budget.note("controller_decision_failed", case=case["id"], family=family,
                            decision_failure=pending_decision)
            except Exception as journal_exc:
                # Keep the original failure and frame even when persistence
                # failed. The poisoned journal still prevents owner dispatch.
                decision_journal_error = _error(journal_exc)
        cleanup = {"schema": CLEANUP_SCHEMA, "attempted": True, "confirmed": False, "error": None}
        try:
            before = recorder.view
            canceled = recorder.cancel("diagnostic_error_cleanup")
            cleanup = confirmed_cleanup(before, canceled, budget.counts["host_requests"])
        except Exception as cleanup_exc:
            cleanup["error"] = _error(cleanup_exc)
    finally:
        controller.cancel()
    metrics = task.metrics(administratively_capped=not task.done)
    character = None
    try:
        character = character_metrics(recorder.initial, goal.target, recorder.events, phases)
    except Exception as exc:
        error = error or _error(exc)
    evaluation = {"diagnostic_only": True, "previous_case_origin": case["origin"],
        "task_reward_schema": "jolt-arrival-best-progress/v1", "metrics": metrics,
        "character_metrics": character,
        "reward_terms": terms, "controller_frames": phases, "error": error,
        "controller_decision_schema": CONTROLLER_DECISION_SCHEMA,
        "controller_decision_failures": decision_failures,
        "controller_decision_accounting": decision_accounting(phases, decision_failures),
        "decision_failure_journal_error": decision_journal_error, "error_cleanup": cleanup,
        "owner_closure_confirmed": closure, "declaration_sha256": sha256_json(declaration)}
    episode = recorder.finish("failed" if error else "completed", metrics["reason"], evaluation=evaluation)
    path = output / "episodes" / f"{identifier}.json.gz"
    artifact_sha = write_episode(path, episode)
    sync_directory(path.parent)
    row = {"case_id": case["id"], "family": family, "origin": case["origin"],
        "host_seed": case["host_seed"], "profile_sha256": episode.profile_sha256,
        "record_sha256": episode.sha256, "artifact": path.relative_to(output).as_posix(),
        "artifact_sha256": artifact_sha, "decisions": episode.decisions,
        "consumed_ticks": episode.consumed_ticks, "metrics": metrics, "character_metrics": character,
        "controller_decision_schema": CONTROLLER_DECISION_SCHEMA,
        "controller_decision_accounting": evaluation["controller_decision_accounting"],
        "error_cleanup": cleanup,
        "owner_closure_confirmed": closure, "error": error}
    write_json(output / "metrics" / f"{identifier}.json", row,
               direct=declaration.get("schema") == "nca-jolt-recovery-diagnostic/v4")
    budget.note("episode_completed", case=case["id"], family=family, row=row)
    return row


def summarize(rows, cases, *, families=FAMILIES):
    expected = {(case["id"], family) for case in cases for family in families}
    keys = [(row["case_id"], row["family"]) for row in rows]
    if len(keys) != len(set(keys)) or not set(keys).issubset(expected):
        raise ContractError("arrival_pairing", "duplicate or foreign diagnostic outcomes")
    valid = (set(keys) == expected and all(row["error"] is None and row["owner_closure_confirmed"]
        and not row["metrics"]["administratively_capped"] and not row["metrics"]["integrity_failure"]
        and row["metrics"]["failed_admissions"] == 0 for row in rows))
    metrics = ("overshoot_cells", "final_distance_cells", "capped_completion_seconds",
        "peak_toward_speed_cells_per_second", "mean_absolute_angle_radians",
        "ordinary_trim_reversals_per_second", "purposeful_launch", "primary_edges")
    family_results = {}
    for family in families:
        selected = [row for row in rows if row["family"] == family]
        n = len(selected)
        if not n:
            continue
        if any(type(row["metrics"].get(name)) not in (float, int, bool)
               or not math.isfinite(row["metrics"][name]) for row in selected for name in metrics):
            raise ContractError("arrival_metrics", "finite complete recorded metrics are required")
        family_results[family] = {"episodes": n, "successes": sum(row["metrics"]["success"] for row in selected),
            "mean": {name: sum(row["metrics"][name] for row in selected) / n for name in metrics},
            "failed_cases": [row["case_id"] for row in selected if not row["metrics"]["success"]]}
        for origin in ("previously_open_training", "previously_open_evaluation"):
            partition = [row for row in selected if row["origin"] == origin]
            family_results[family][origin] = {"episodes": len(partition),
                "successes": sum(row["metrics"]["success"] for row in partition)}
    return {"complete_valid_pairing": valid, "families": family_results,
            "interpretation": "descriptive paired development diagnostic; no sealed learned-policy inference",
            "morphology_admitted": False, "learned_skill_qualified": False, "owner_feel_accepted": False}


def run_diagnostic(*, root: Path, checkout: Path, godot: Path, declaration_path: Path,
                   output: Path, expected_revision: str, expected_declaration_sha256=DECLARATION_SHA256,
                   whole_process_started=None):
    """Single source-frozen invocation; stop on uncertain consumption or cost."""
    declaration = read_declaration(declaration_path, expected_sha256=expected_declaration_sha256)
    is_v4 = declaration.get("schema") == "nca-jolt-recovery-diagnostic/v4"
    if is_v4 and (type(whole_process_started) not in (int, float)
            or not math.isfinite(whole_process_started) or not 0 < whole_process_started <= time.monotonic()):
        raise ContractError("arrival_wall_origin", "v4 needs its original supervised process start before admission")
    families = tuple(declaration["families"])
    specifications = declaration.get("candidate_specifications", {FAMILIES[1]: declaration.get("candidate_specification")})
    if set(specifications) != set(families) - {FAMILIES[0]}:
        raise ContractError("arrival_candidate", "every declared candidate needs its complete frozen specification")
    for family, specification in specifications.items():
        if _controller(family).specification() != specification:
            raise ContractError("arrival_candidate", "candidate specification differs from its declaration")
    original_path = root / declaration["case_source"]["path"]
    original = strict_json_loads(original_path.read_bytes())
    if sha256_json(original) != declaration["case_source"]["sha256"]:
        raise ContractError("arrival_case_source", "original open case/configuration bytes changed")
    cases = diagnostic_cases(original)
    if is_v4 and cases != declaration["case_inventory"]:
        raise ContractError("arrival_case_source", "v4 case order or seeds differ from the prospective inventory")
    output.mkdir(parents=True, exist_ok=False)
    sync_directory(output.parent)
    rows, failure, identity, manifest, host = [], None, None, None, None
    with DurableJournal(output / "owner-journal.jsonl") as journal:
        budget = DiagnosticBudget(declaration["budget"], journal, whole_process_started=whole_process_started)
        try:
            revision, files, _ = source_identity(root)
            if revision != expected_revision or len(revision) != 40:
                raise ContractError("arrival_revision", "the explicitly published source revision is required")
            if subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=root).strip():
                raise ContractError("arrival_dirty_source", "the complete published checkout must be clean")
            execution_inputs = [declaration_path, original_path, root / "tools/jolt_arrival.py"]
            if is_v4:
                execution_inputs.append(root / "tools/execute_jolt_diagnostic.py")
            for path in execution_inputs:
                files[path.resolve().relative_to(root.resolve()).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            for relative, expected in files.items():
                committed = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=root,
                                                   stderr=subprocess.DEVNULL)
                if hashlib.sha256(committed).hexdigest() != expected:
                    raise ContractError("arrival_unpublished_source", "every input must match its published Git blob")
            for relative, expected in declaration["preserved_sources"].items():
                if files.get(relative) != expected:
                    raise ContractError("arrival_changed_dependency", "a preserved original dependency changed")
            identity = inspect_host(checkout, godot)
            if identity["host_revision"] != original["host_sha"] or identity["bridge_sha256"] != original["bridge_sha256"]:
                raise ContractError("arrival_host_identity", "the accepted physical host and unchanged bridge are required")
            manifest = {"schema": "nca-jolt-arrival-diagnostic/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_revision": revision, "source_files": files, "source_files_sha256": sha256_json(files),
                "declaration": declaration, "declaration_sha256": sha256_json(declaration),
                "host_identity": identity, "cases": cases, "python": platform.python_version(),
                "platform": platform.system(), "architecture": platform.machine(), "device": "cpu",
                "training": False, "evaluation_cases_previously_open": True}
            if is_v4:
                manifest["whole_process_started_monotonic"] = whole_process_started
            write_json(output / "manifest.json", manifest, direct=is_v4)
            budget.check()
            if is_v4:
                budget.start_host()
            else:
                budget.counts["host_start_attempts"] += 1
                budget.note("host_start_reserved")
            host = GodotHost(checkout, godot, lane="act004", mode="stepped", timeout=15.)
            metered = MeteredHost(host, budget)
            for case in cases:
                for family in families:
                    row = _capture(metered, budget, case, family, output, original, declaration, root)
                    rows.append(row)
                    print(canonical_json({"case": case["id"], "family": family,
                        "success": row["metrics"]["success"], "error": row["error"]}), flush=True)
                    if row["error"] is not None:
                        raise ContractError("arrival_episode_failed", "diagnostic retained an unsuccessful execution boundary")
            budget.check()
        except BaseException as exc:
            failure = _error(exc)
        finally:
            if host is not None:
                try:
                    if host._process is not None and host._process.poll() is None:
                        host._channel.connection.settimeout(2.)
                        MeteredHost(host, budget).request("close")
                except Exception as exc:
                    failure = failure or _error(exc)
                finally:
                    host.close(graceful=False)
                    closed = host._process is None or host._process.poll() is not None
                    budget.counts["host_terminations"] += int(closed)
                    budget.note("host_closed", termination_confirmed=closed)
            if manifest is not None:
                drift = any(hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected
                            for relative, expected in manifest["source_files"].items())
                if drift:
                    failure = failure or {"type": "SourceDrift", "code": "arrival_source_changed"}
                try:
                    if inspect_host(checkout, godot) != identity:
                        raise ContractError("arrival_host_drift", "verified host/runtime bytes changed during the diagnostic")
                except Exception as exc:
                    failure = failure or _error(exc)
            measured = budget.snapshot()
    try:
        journal_validation = (seal_owner_journal_v2 if is_v4 else seal_owner_journal)(output / "owner-journal.jsonl",
            expected_counts=measured, expected_rows=rows, durability=journal.durability_receipt(),
            validate_owner=validate_closed_owner_journal, families=families, artifact_root=output)
        budget.check()
    except Exception as exc:
        journal_validation = {"verified": False, "error": _error(exc)}
        failure = failure or _error(exc)
    comparison = summarize(rows, cases, families=families)
    character_comparison = summarize_character(
        [row for row in rows if row.get("character_metrics") is not None], families)
    completed = (failure is None and journal_validation["verified"]
        and comparison["complete_valid_pairing"] and not measured["unconfirmed_tail"]
        and measured["host_start_attempts"] == measured["host_terminations"] == 1
        and measured["requested_step_ticks"] == measured["confirmed_step_ticks"]
        and measured["wall_seconds"] <= declaration["budget"]["maximum_wall_seconds"])
    artifacts = {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(output.rglob("*")) if path.is_file()}
    if is_v4 and completed:
        try:
            artifacts = diagnostic_artifact_inventory(output, rows)
        except Exception as exc:
            completed = False
            failure = failure or _error(exc)
    result = {"schema": "nca-jolt-arrival-diagnostic-result/v1", "status": "completed" if completed else "incomplete",
        "source_revision": expected_revision, "declaration_sha256": sha256_json(declaration),
        "counts": measured, "comparison": comparison, "character_comparison": character_comparison,
        "rows": rows, "error": failure, "families": list(families),
        "worker_wall_seconds_through_seal": budget.clock() - budget.started,
        "artifact_sha256": artifacts, "host_identity": identity, "owner_journal_validation": journal_validation,
        "evidence_limit": "already-open authored development cases; unchanged learned prerequisite remains blocked"}
    write_json(output / "summary.json", result, direct=is_v4)
    return result


def verify_diagnostic_output(output: Path, *, expected_revision: str, expected_declaration_sha256: str):
    """Independently bind final files after worker exit, with zero owner calls."""
    summary_path = output / "summary.json"
    summary = strict_json_loads(summary_path.read_bytes(), max_bytes=8 * 1024 * 1024)
    manifest = strict_json_loads((output / "manifest.json").read_bytes())
    if (summary.get("status") != "completed" or summary.get("error") is not None
            or summary.get("source_revision") != expected_revision
            or manifest.get("source_revision") != expected_revision
            or summary.get("declaration_sha256") != expected_declaration_sha256
            or manifest.get("declaration_sha256") != expected_declaration_sha256
            or sha256_json(manifest.get("declaration")) != expected_declaration_sha256):
        raise ContractError("arrival_final_output", "complete source/declaration-bound worker evidence is required")
    families = tuple(manifest["declaration"]["families"])
    is_v4 = manifest["declaration"].get("schema") == "nca-jolt-recovery-diagnostic/v4"
    if summary.get("families") != list(families):
        raise ContractError("arrival_final_output", "summary families differ from the source-frozen declaration")
    identities = summary.get("artifact_sha256")
    current = {path.relative_to(output).as_posix() for path in output.rglob("*")
               if path.is_file() and path != summary_path}
    if not isinstance(identities, dict) or current != set(identities):
        raise ContractError("arrival_final_output", "final artifact inventory is incomplete or changed")
    for relative, expected in identities.items():
        if (Path(relative).is_absolute() or ".." in Path(relative).parts
                or file_identity(output / relative)["sha256"] != expected):
            raise ContractError("arrival_final_output", "final artifact bytes differ from the closed inventory")
    if is_v4:
        declaration = _validate_declaration(manifest["declaration"], expected_sha256=expected_declaration_sha256)
        if declaration.get("budget") != V4_BUDGET:
            raise ContractError("arrival_final_output", "v4 budget differs from its original exact allocation")
        strict_inventory = diagnostic_artifact_inventory(output, summary["rows"], summary=True)
        if {name: digest for name, digest in strict_inventory.items() if name != "summary.json"} != identities:
            raise ContractError("arrival_final_output", "v4 final inventory differs from its declared artifact paths")
        counts = summary["counts"]
        origin = manifest.get("whole_process_started_monotonic")
        if (type(origin) not in (int, float) or not math.isfinite(origin) or origin <= 0
                or counts.get("whole_process_started_monotonic") != origin
                or counts["episodes_started"] != 96 or counts["host_start_attempts"] != 1
                or counts["step_requests"] > 3840 or counts["requested_step_ticks"] > 23040
                or counts["host_requests"] > 4500 or counts["wall_seconds"] > 900
                or any(type(row.get("decisions")) is not int or not 1 <= row["decisions"] <= 40
                       or row.get("consumed_ticks") != 6 * row["decisions"] or row["consumed_ticks"] > 240
                       for row in summary["rows"])):
            raise ContractError("arrival_final_output", "v4 source-bound whole-process and per-episode counts exceed their allocation")
    validation = (verify_sealed_journal_v2 if is_v4 else verify_sealed_journal)(output,
        expected_counts=summary["counts"], expected_rows=summary["rows"],
        validate_owner=validate_closed_owner_journal, families=families)
    if validation != summary.get("owner_journal_validation"):
        raise ContractError("arrival_final_output", "independent sealed journal replay differs from the worker")
    comparison = summarize(summary["rows"], manifest["cases"], families=families)
    if not comparison["complete_valid_pairing"] or comparison != summary.get("comparison"):
        raise ContractError("arrival_final_output", "independent diagnostic pairing differs from the worker")
    if summarize_character(summary["rows"], families) != summary.get("character_comparison"):
        raise ContractError("arrival_final_output", "independent character aggregation differs from the worker")
    if is_v4 and diagnostic_artifact_inventory(output, summary["rows"], summary=True) != strict_inventory:
        raise ContractError("arrival_final_output", "v4 evidence changed during complete independent verification")
    return {"verified": True, "summary_sha256": file_identity(summary_path)["sha256"],
            "manifest_sha256": file_identity(output / "manifest.json")["sha256"],
            "artifact_count": len(identities), "owner_journal_validation": validation}
