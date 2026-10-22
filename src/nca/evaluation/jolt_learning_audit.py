"""Versioned offline readback of one immutable Jolt learning invocation.

This reader has no host handle, optimizer, or continuation path. Native
responses are bound to the original raw capture before task metrics, recurrent
policy outputs and pre-transform densities are reconstructed. Every neural
forward and checkpoint load is reserved by the caller before it happens.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, replace
import gzip
import hashlib
import math
from pathlib import Path
import random
import re
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import torch

from nca.adapters.godot import physical_config
from nca.contracts import ContractError, WireAction, canonical_json, sha256_json, strict_json_loads
from nca.controllers.baseline import FeedbackObservation, SkillGoal
from nca.controllers.jolt_learned import (
    ActionHistory, BASE_LOGIT_LIMIT, InvocationContext, JoltPolicy,
    combine_residual, emergency_request, features, joint_log_probability,
    load_policy, map_action, policy_state_sha256,
)
from nca.controllers.jolt_recovery_v4 import JoltRecoveryControllerV4
from nca.controllers.morphology.conformance import cancellation_receipt
from nca.controllers.recovery_supervisor import PhysicalRecoverySupervisor
from nca.data import read_episode
from nca.data.jolt_samples import JoltEpisode, native_packet_matches, validate_history
from nca.evaluation.durable_journal import MAX_RECORD_BYTES, file_identity, validate_record_chain
from nca.evaluation.jolt_character import character_metrics
from nca.evaluation.jolt_journal_v3 import journal_identity_v3
from nca.evaluation.jolt_learning import POLICY_SEEDS, controller_ids, qualify
from nca.train.task import HopTask, REWARD_SCHEMA


SCHEMA = "nca-learn-jolt-002-independent-readback/v1"
RUN_SCHEMA = "nca-learn-jolt-002-run/v1"
SCHEMA_V2 = "nca-learn-jolt-002-independent-readback/v2"
RUN_SCHEMA_V2 = "nca-learn-jolt-002-run/v2"
SCHEMA_V3 = "nca-learn-jolt-003-independent-readback/v3"
RUN_SCHEMA_V3 = "nca-learn-jolt-003-run/v1"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_RESERVATION_BYTES = 512 * 1024 * 1024
PPO_ID = re.compile(r"ppo-seed(190103|190207|190301)-u([0-7])-e([0-9]+)\Z")
CHECKPOINT_ID = re.compile(r"(bc|residual)-seed(190103|190207|190301)-(initial|final|u[1-8])\Z")


def _need(condition: bool, detail: str) -> None:
    if not condition:
        raise ContractError("jolt_audit", detail)


def _same(left: Any, right: Any, detail: str) -> None:
    _need(sha256_json(left) == sha256_json(right), detail)


def _near(left: float, right: float, detail: str) -> None:
    _need(type(left) in (int, float) and type(right) in (int, float)
          and math.isfinite(left) and math.isfinite(right)
          and math.isclose(left, right, rel_tol=1e-5, abs_tol=2e-5), detail)


def _execution_version(worker, manifest, declaration):
    """Require an explicitly matching version; never reinterpret v1 evidence."""
    version = {RUN_SCHEMA: 1, RUN_SCHEMA_V2: 2, RUN_SCHEMA_V3: 3}.get(worker.get("schema"))
    _need(version is not None and manifest.get("schema") == worker["schema"],
          "unknown or mismatched worker/manifest contract")
    schema = "nca-learn-jolt-003-execution/v3" if version == 3 else f"nca-learn-jolt-002-execution/v{version}"
    _need(declaration.get("schema") == schema,
          "execution declaration and result versions differ")
    return version


def _cancellation_error_v2(row, raw):
    """Bind the separate cleanup error without copying arbitrary exception text."""
    _need("cancellation_error" in row and "cancellation_error" in raw.evaluation,
          "v2 capture omitted its cancellation result")
    error = row["cancellation_error"]
    _same(raw.evaluation["cancellation_error"], error, "raw cancellation error differs")
    if error is not None:
        _need(isinstance(error, Mapping) and set(error) == {"type", "code"}
              and isinstance(error["type"], str)
              and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error["type"]) is not None
              and (error["code"] is None or isinstance(error["code"], str)
                   and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error["code"]) is not None),
              "cancellation error is not a bounded structured diagnostic")
        _need(row["error"] is not None, "cleanup error was omitted from episode failure")
    return error


def _episode_closure_v2(raw, row, *, deadline_missed):
    """Recompute native cancellation from retained events, not actor suspension."""
    cleanup_error = _cancellation_error_v2(row, raw)
    if not raw.events or cleanup_error is not None:
        return False
    final = raw.events[-1]
    if final.kind == "cancel":
        _need(final.valid is True and final.next_observation is not None
              and final.tick_before == final.tick_after
              and not final.ticks and final.requested_packet is None,
              "cancellation event cannot establish a native fence")
        cancellation_receipt(final.issued_observation, final.next_observation)
        return True
    return bool(deadline_missed and final.kind == "step" and final.valid
                and final.next_observation
                and final.next_observation.get("suspended") is True)


def _episode_closure_v3(raw, row, *, deadline_missed):
    """Retirement-proof failure and native cancellation failure stay distinct."""
    for field in ("retirement_validation_error", "cleanup_phase_error"):
        _need(field in row and field in raw.evaluation, "v3 capture omitted a separate cleanup/proof status")
        error = row[field]
        _same(error, raw.evaluation[field], "raw cleanup/proof error differs: " + field)
        if error is not None:
            _need(isinstance(error, Mapping) and set(error) == {"type", "code"}
                  and isinstance(error["type"], str)
                  and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error["type"]) is not None
                  and (error["code"] is None or isinstance(error["code"], str)
                       and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error["code"]) is not None)
                  and row["error"] is not None and row["outcome"]["complete"] is False,
                  "cleanup/proof failure was erased or accepted as a complete episode")
    closure = _episode_closure_v2(raw, row, deadline_missed=deadline_missed)
    if closure and raw.events[-1].kind == "cancel":
        from nca.evaluation.jolt_retirement import native_cancellation_v3
        native_cancellation_v3(raw.events[-1].issued_observation, raw.events[-1].next_observation)
    return closure


def _budget_exhaustion_v3(error):
    """Only original declared limit errors retain capped-invocation status."""
    if not isinstance(error, Mapping):
        return False
    code = error.get("code")
    return error.get("type") == "AdmissionError" or (error.get("type") == "ContractError"
        and isinstance(code, str) and (code == "jolt_v3_phase_deadline"
            or code.startswith("budget_counter_") or code.startswith("budget_stage_")))


def _durability_v2(durable, identity, *, system):
    """Verify the v2 writer's claim against this reader's complete raw stream."""
    fields = {"schema", "closed", "poisoned", "records", "record_fsyncs", "total_fsyncs",
              "directory_fsync", "chain_sha256", "readback_records", "closed_verified",
              "final_directory_fsync", "closed_identity"}
    _need(isinstance(durable, Mapping) and set(durable) == fields
          and durable["schema"] == "nca-owner-journal-durability/v2",
          "unknown or incomplete v2 journal durability contract")
    _need(all(type(durable[key]) is bool for key in ("closed", "poisoned", "directory_fsync",
                                                   "closed_verified", "final_directory_fsync"))
          and all(type(durable[key]) is int and durable[key] >= 0
                  for key in ("records", "record_fsyncs", "total_fsyncs", "readback_records")),
          "v2 journal durability counters or flags are malformed")
    _need(durable["closed"] and not durable["poisoned"] and durable["closed_verified"]
          and durable["records"] == durable["record_fsyncs"] == durable["readback_records"] == identity["records"]
          and durable["total_fsyncs"] == identity["records"] + 1
          and durable["chain_sha256"] == identity["chain_sha256"],
          "v2 owner journal did not independently verify closure")
    _same(durable["closed_identity"], identity, "v2 closed journal bytes differ from independent readback")
    _need(isinstance(system, str) and bool(system), "journal durability requires the frozen runtime platform")
    _need(system == "Windows" or durable["directory_fsync"] and durable["final_directory_fsync"],
          "v2 journal omitted required directory synchronization")


def _durability_v3(durable, identity, *, system):
    """V3 keeps independently verified closure and freezes its byte envelope."""
    from nca.train.jolt_storage_v3 import MAX_JOURNAL_RECORD_BYTES, MAX_JOURNAL_RECORDS, MAX_JOURNAL_BYTES
    limits = {"maximum_record_bytes": MAX_JOURNAL_RECORD_BYTES,
              "maximum_records": MAX_JOURNAL_RECORDS, "maximum_stream_bytes": MAX_JOURNAL_BYTES}
    _need(isinstance(durable, Mapping) and durable.get("schema") == "nca-owner-journal-durability/v3"
          and all(type(durable.get(key)) is int and durable[key] == value for key, value in limits.items())
          and identity["records"] <= MAX_JOURNAL_RECORDS and identity["bytes"] <= MAX_JOURNAL_BYTES,
          "v3 journal changed or exceeded its prospective aggregate envelope")
    original = {key: value for key, value in durable.items() if key not in limits}
    original["schema"] = "nca-owner-journal-durability/v2"
    _durability_v2(original, identity, system=system)


_journal_identity_v3 = journal_identity_v3  # Existing focused conformance callers.


def _path(root: Path, name: str) -> Path:
    _need(isinstance(name, str) and bool(name) and not Path(name).is_absolute()
          and ".." not in Path(name).parts, "artifact path escapes its immutable output")
    path = root / name
    _need(path.resolve().is_relative_to(root.resolve()) and path.is_file() and not path.is_symlink(),
          "artifact is absent, redirected or not a regular file")
    return path


def _json(path: Path, *, limit: int = MAX_JSON_BYTES) -> Any:
    _need(path.stat().st_size <= limit, "JSON evidence exceeds its bounded reader")
    return strict_json_loads(path.read_bytes(), max_bytes=limit)


def _episode_envelope_v3(path):
    """Bound decompression before constructing any episode object tree."""
    from nca.train.jolt_storage_v3 import episode_byte_limit, compressed_byte_limit
    maximum = episode_byte_limit(40)
    _need(path.stat().st_size <= compressed_byte_limit(maximum), "v3 episode exceeds its compressed envelope")
    with gzip.open(path, "rb") as stream:
        payload = stream.read(maximum + 1)
    _need(len(payload) <= maximum, "v3 episode expansion exceeds the original byte envelope")
    envelope = strict_json_loads(payload, max_bytes=maximum)
    _need(isinstance(envelope, dict) and set(envelope) == {"episode", "sha256"},
          "v3 episode envelope has undeclared fields")
    return envelope, len(payload)


def _raw_episode_v3(path):
    from nca.data.records import EpisodeRecord
    from nca.train.jolt_storage_v3 import episode_byte_limit
    envelope, size = _episode_envelope_v3(path)
    episode = EpisodeRecord.from_dict(envelope["episode"])
    _need(size <= episode_byte_limit(episode.decisions) and episode.sha256 == envelope["sha256"],
          "v3 raw episode differs from its original identity or per-decision byte bound")
    return episode


def _sample(path: Path, *, execution_version: int = 1) -> JoltEpisode:
    if execution_version == 3:
        from nca.data.jolt_samples import JoltEpisodeV3
        from nca.train.jolt_storage_v3 import episode_byte_limit
        envelope, size = _episode_envelope_v3(path)
        episode = JoltEpisodeV3.from_dict(envelope["episode"])
        _need(size <= episode_byte_limit(len(episode.decisions)), "v3 samples exceeded their original per-decision bound")
        _need(episode.sha256 == envelope["sha256"], "sample content identity differs")
        validate_history(episode)
        return episode
    _need(path.stat().st_size <= MAX_JSON_BYTES, "compressed samples exceed their bounded reader")
    with gzip.open(path, "rb") as stream:
        payload = stream.read(MAX_JSON_BYTES + 1)
    _need(len(payload) <= MAX_JSON_BYTES, "sample expansion exceeds its bounded reader")
    envelope = strict_json_loads(payload, max_bytes=MAX_JSON_BYTES)
    _need(set(envelope) == {"episode", "sha256"}, "sample envelope has an undeclared field")
    episode = JoltEpisode.from_dict(envelope["episode"])
    _need(episode.sha256 == envelope["sha256"], "sample content identity differs")
    validate_history(episode)
    return episode


class _Reservations:
    """Recompute every counter and snapshot from charged, ordered increments."""

    def __init__(self, budget, rows, snapshot, *, check=None):
        self.budget, self.rows = budget, rows
        self.check = check
        self.version = 3 if snapshot.get("schema") == "nca-jolt-learn-budget/v3" else 1
        self.counts = {key: 0 for key in budget["counters"]}
        self.stages = {stage: {key: 0 for key in limits} for stage, limits in budget["stages"].items()}
        self.prefixes = {0: (dict(self.counts), deepcopy(self.stages), sha256_json([]), None)}
        self._cursor = 0
        self._iterator = iter(rows)
        self._prefix_elapsed = None
        self._prefix_digest = hashlib.sha256(b"[")
        self._prefix_counts = dict(self.counts)
        self._prefix_stages = deepcopy(self.stages)
        self.used_owner_reservations: set[int] = set()
        self.owner_queues = defaultdict(deque)
        self.owner_elapsed = {}
        self.phase_history = deepcopy(snapshot.get("phase_history")) if self.version == 3 else None
        previous = -1.
        previous_phase = 0
        phase_names = ("active", "closure", "audit", "publication")
        phase_ends = {"active": 6000., "closure": 6180., "audit": 6780., "publication": 6888.}
        for index, row in enumerate(rows):
            if self.check is not None:
                self.check()
            fields = {"index", "elapsed_seconds", "stage", "cleanup", "increments"}
            if self.version == 3:
                fields.add("phase")
            _need(set(row) == fields
                  and row["index"] == index and type(row["cleanup"]) is bool,
                  "reservation order or schema differs")
            elapsed, stage = row["elapsed_seconds"], row["stage"]
            limit = budget["maximum_wall_seconds"] - (0 if row["cleanup"] else budget["cleanup_reserve_seconds"])
            if self.version == 3:
                phase = row["phase"]
                _need(phase in phase_names and phase_names.index(phase) >= previous_phase,
                      "reservation rolled back its original allocation phase")
                previous_phase = phase_names.index(phase)
                limit = phase_ends[phase]
                _need(phase == "active" or row["cleanup"] is True,
                      "ordinary work entered the reserved evidence tail")
                _need(isinstance(self.phase_history, list)
                      and any(entry.get("phase") == phase and entry.get("elapsed_seconds", float("inf")) <= elapsed
                              for entry in self.phase_history),
                      "reservation preceded its original phase entry")
                if phase == "audit":
                    _need(stage is None and set(row["increments"]) <= {"audit_forward", "checkpoint_loads"},
                          "non-audit work consumed the neural-readback interval")
                elif phase == "closure":
                    _need(set(row["increments"]) <= {"owner_requests", "owner_cancel", "owner_close",
                          "host_terminations", "uncertain_ticks", "episodes_aborted", "episodes_completed"},
                          "ordinary or neural work consumed native-closure time")
                _need(phase != "publication", "publication interval invented model or physical work")
            _need(type(elapsed) in (int, float) and math.isfinite(elapsed) and previous <= elapsed < limit,
                  "reservation exceeded the original wall deadline")
            _need(stage is None or stage in self.stages, "reservation used an undeclared stage")
            _need(bool(row["increments"]), "empty reservation")
            for key, amount in row["increments"].items():
                _need(key in self.counts and type(amount) is int and amount > 0,
                      "reservation has an undeclared, refunded or noninteger cost")
                _need(stage is not None or key not in {"decisions", "requested_ticks", "episodes_started"},
                      "physical cost lost its original stage")
                self.counts[key] += amount
                _need(self.counts[key] <= budget["counters"][key], "counter exceeded its prospective cap")
                if stage is not None and key in self.stages[stage]:
                    self.stages[stage][key] += amount
                    _need(self.stages[stage][key] <= budget["stages"][stage][key],
                          "stage exceeded its prospective cap")
            previous = elapsed
            if "owner_requests" in row["increments"]:
                self.owner_queues[sha256_json({"stage": stage, "increments": row["increments"]})].append(index)
                self.owner_elapsed[index] = elapsed
        self.snapshot(snapshot)

    def snapshot(self, snapshot):
        if self.check is not None:
            self.check()
        count = snapshot["reservation_count"]
        _need(type(count) is int and 0 <= count <= len(self.rows), "snapshot names an impossible reservation prefix")
        if count not in self.prefixes:
            # The worker's final snapshot is checked first, then journal
            # snapshots occur in original increasing order. Restarting once
            # permits two linear passes instead of hashing the whole growing
            # reservation array at each of the 57 fitting boundaries.
            if count < self._cursor:
                self._cursor = 0
                self._iterator = iter(self.rows)
                self._prefix_elapsed = None
                self._prefix_digest = hashlib.sha256(b"[")
                self._prefix_counts = {key: 0 for key in self.counts}
                self._prefix_stages = {stage: {key: 0 for key in values} for stage, values in self.stages.items()}
            while self._cursor < count:
                if self.check is not None:
                    self.check()
                row = next(self._iterator)
                if self._cursor:
                    self._prefix_digest.update(b",")
                self._prefix_digest.update(canonical_json(row).encode())
                for key, amount in row["increments"].items():
                    self._prefix_counts[key] += amount
                    if row["stage"] is not None and key in self._prefix_stages[row["stage"]]:
                        self._prefix_stages[row["stage"]][key] += amount
                self._cursor += 1
                self._prefix_elapsed = row["elapsed_seconds"]
            if self.version == 3 and count == len(self.rows):
                _need(next(self._iterator, None) is None, "reservation prefix omitted a trailing record")
            digest = self._prefix_digest.copy()
            digest.update(b"]")
            self.prefixes[count] = dict(self._prefix_counts), deepcopy(self._prefix_stages), digest.hexdigest(), self._prefix_elapsed
        counts, stages, digest, last_elapsed = self.prefixes[count]
        _need(snapshot["reservations_sha256"] == digest, "snapshot reservation digest differs")
        _same(snapshot["counts"], counts, "snapshot counters differ from original reservations")
        _same(snapshot["stage_counts"], stages, "snapshot stage counters differ")
        _need(snapshot["schema"] == f"nca-jolt-learn-budget/v{self.version}"
              and snapshot["maximum_wall_seconds"] == self.budget["maximum_wall_seconds"]
              and snapshot["cleanup_reserve_seconds"] == self.budget["cleanup_reserve_seconds"]
              and snapshot["refused_or_uncertain_work_refunded"] is False,
              "snapshot changed its original budget")
        elapsed = snapshot["wall_seconds"]
        _need(type(elapsed) in (int, float) and math.isfinite(elapsed)
              and (not count or last_elapsed <= elapsed)
              and elapsed < self.budget["maximum_wall_seconds"], "snapshot exceeds original wall limit")
        if self.version == 3:
            phase_names = ("active", "closure", "audit", "publication")
            ends = (6000., 6180., 6780., 6888.)
            history, phase = snapshot.get("phase_history"), snapshot.get("phase")
            _need(phase in phase_names and isinstance(history, list) and 1 <= len(history) <= 4,
                  "versioned budget lost original phase history")
            previous_phase, previous_time = -1, -1.
            for entry in history:
                _need(isinstance(entry, Mapping) and set(entry) == {"phase", "elapsed_seconds"}
                      and entry["phase"] in phase_names, "phase receipt shape changed")
                index, instant = phase_names.index(entry["phase"]), entry["elapsed_seconds"]
                _need(index == previous_phase + 1 and type(instant) in (int, float) and math.isfinite(instant)
                      and max(0., previous_time) <= instant <= elapsed and instant < ends[index],
                      "phase receipt reset or exceeded the original allocation clock")
                previous_phase, previous_time = index, instant
            _need(history[0]["phase"] == "active" and history[-1]["phase"] == phase
                  and elapsed < ends[phase_names.index(phase)], "snapshot exceeded its original phase deadline")
            _same(history, self.phase_history[:len(history)], "snapshot rewrote original phase history")
        return counts

    def owner(self, request):
        increments = {"owner_requests": 1, "owner_" + request["operation"]: 1}
        if request["operation"] == "step":
            increments.update(decisions=1, requested_ticks=6)
        queue = self.owner_queues[sha256_json({"stage": request["stage"], "increments": increments})]
        _need(bool(queue), "native owner work has no original atomic reservation")
        index = queue.popleft()
        _need(self.owner_elapsed[index] <= request["elapsed_seconds"],
              "native owner work preceded its reservation")
        self.used_owner_reservations.add(index)


def _native_step(event, request, response, *, schedule):
    """Bind raw receipt fields to the native response, including applied input."""
    _same(request["fields"], {"commands": [event.requested_packet], "ticks": 6},
          "native step changed the dispatched packet or repeat")
    _need(response.get("schedule") == schedule and response.get("failed") is not True,
          "native step schedule or integrity differs")
    _need(response.get("start_tick") == event.tick_before and response.get("tick") == event.tick_after
          and response.get("actual_ticks") == event.tick_after - event.tick_before
          and 1 <= response["actual_ticks"] <= 6, "native step interval differs from capture")
    _need(not response.get("idle_trace") and response.get("idle_trace_dropped", 0) == 0,
          "stepped capture contains unaccounted idle consumption")
    _same(response["observation"], event.next_observation, "native final observation differs from capture")
    _need(event.reward == 0. and event.terminated == response["terminated"]
          and event.truncated == response["truncated"] and event.reason == response["reason"],
          "raw capture rewrote native reward or terminal receipt")
    traces = response["traces"]
    _need(len(traces) == len(event.ticks), "native trace coverage differs")
    for index, (native, tick) in enumerate(zip(traces, event.ticks)):
        for key in ("tick_before", "tick_after", "world_tick", "received_monotonic_usec",
                    "completed_monotonic_usec", "requested", "admission", "applied", "passive",
                    "observation", "cellular_sample_tick"):
            _same(native[key], getattr(tick, key), "raw tick differs from its original native response: " + key)
        _same(native.get("actual_motion", {}), tick.actual_motion, "actual motion was rewritten")
        _need(tick.valid is (not native.get("failed", False)), "native failure was relabelled")
        _same(tick.consumed_action, tick.observation.get("action", {}), "post-step action view was rewritten")
        _need(tick.cellular_sample_tick == tick.tick_before, "cellular pre-sample schedule changed")
        _need(len(tick.requested) == (1 if index == 0 else 0)
              and (index != 0 or native_packet_matches(event.requested_packet, tick.requested[0])),
              "native trace contains an additional or altered command")
        if tick.consumed_request is not None:
            _need(native_packet_matches(event.requested_packet, tick.consumed_request)
                  and tick.source_id == event.source.id, "consumption attribution is not the dispatched source")


def _window_counts(records, stage):
    """Count actual model calls, prefix replay and losses without fitting."""
    total = Counter()
    advance = [row.observation_valid and row.proposal is not None for row in records]
    eligible = [row.imitation_eligible if stage == "bc" else
                bool(row.observation_valid and row.proposal is not None and row.event.valid
                     and row.event.trace_complete and row.actual_ticks > 0) for row in records]
    epochs = 40 if stage == "bc" else 4
    for start in range(0, len(records), 16):
        stop = min(start + 16, len(records))
        if not any(eligible[start:stop]):
            continue
        total[stage + "_optimizer"] += epochs
        total[stage + "_forward"] += epochs * sum(advance[start:stop])
        total[stage + "_burn_in"] += epochs * sum(advance[:start])
        if stage == "bc":
            total["bc_eligible_samples"] += epochs * sum(eligible[start:stop])
        else:
            total["ppo_critic_samples"] += epochs * sum(eligible[start:stop])
            total["ppo_actor_samples"] += epochs * sum(row.actor_eligible for row in records[start:stop])
    if stage == "ppo":
        total["ppo_base_forward"] = total["ppo_forward"]
        total["ppo_base_burn_in"] = total["ppo_burn_in"]
        total["ppo_replay_forward"] = sum(advance)
        total["ppo_replay_base_forward"] = sum(advance)
    return total


def _advantages(records, values, bootstrap, final_valid):
    """Independent duration-aware GAE reconstruction; no masked-row bridging."""
    valid = [bool(row.observation_valid and row.proposal is not None and row.event.valid
                  and row.event.trace_complete and row.actual_ticks > 0) for row in records]
    advantage = [0.] * len(records)
    following = 0.
    for index in range(len(records) - 1, -1, -1):
        row = records[index]
        if not valid[index]:
            following = 0.
            continue
        end = index + 1 == len(records)
        fresh = row.event.next_observation is not None and (final_valid or not end)
        joined = end or (valid[index + 1] and not records[index + 1].recurrent_reset
                          and not row.overridden and row.admitted)
        next_value = bootstrap if end else values[index + 1]
        gamma, lam = .997 ** row.actual_ticks, .99 ** row.actual_ticks
        delta = row.event.reward - values[index]
        if fresh and not row.event.terminated and joined:
            delta += gamma * next_value
        continuation = not (end or row.event.terminated or row.event.truncated or not fresh or not joined)
        following = delta + (gamma * lam * following if continuation else 0.)
        advantage[index] = following
    # The learner stores float32 advantages and adds float32 value predictions.
    advantages = torch.tensor(advantage, dtype=torch.float32)
    returns = advantages + torch.tensor(values, dtype=torch.float32)
    return {"advantages": advantages.tolist(), "returns": returns.tolist(), "valid": valid}


def _episode_counts_v3(sample):
    """Retain only bounded fitting-count facts after full independent replay.

    Keeping complete native/sample episodes for all demonstrations would multiply
    the raw-reader expansion by sixty-four. No later phase needs those views:
    their original bytes and replay digests are already independently bound.
    """
    return SimpleNamespace(id=sample.id, sha256=sample.sha256,
        decisions=tuple(SimpleNamespace(observation_valid=row.observation_valid,
            proposal=row.proposal, imitation_eligible=row.imitation_eligible,
            actor_eligible=row.actor_eligible, actual_ticks=row.actual_ticks,
            event=SimpleNamespace(valid=row.event.valid, trace_complete=row.event.trace_complete))
            for row in sample.decisions))


class _Audit:
    def __init__(self, output: Path, reserve: Callable, *, check=None):
        self.output, self.reserve = output, reserve
        self.check = check
        self._check()
        self.worker = _json(_path(output, "worker-result.json"))
        self._check()
        self.config = _json(_path(output, "config.json"))
        self.manifest = _json(_path(output, "manifest.json"))
        self.declaration = _json(_path(output, "declaration.json"))
        self.execution_version = {RUN_SCHEMA_V2: 2, RUN_SCHEMA_V3: 3}.get(self.worker.get("schema"), 1)
        self.models, self.checkpoints = {}, {}
        self.loaded = self.forwards = 0
        self.episodes, self.verified_rows, self.partial = {}, [], []
        self.teacher_rows, self.ppo_groups, self.fit_results = [], {}, []
        self.expected_counts = Counter()
        self.bootstrap_forwards = self.runtime_forwards = self.executions = 0
        self.dataset = None
        self.action_generators = {seed: torch.Generator().manual_seed(seed + 200000) for seed in POLICY_SEEDS}
        self.active_fit = None
        self.last_fit = None
        self.frozen = None
        self.opened = False
        self.closed = False
        self.started = self.terminated = False
        self.complete = self.worker["status"] == "completed"

    def _check(self):
        callback = getattr(self, "check", None)
        if callback is not None:
            callback()

    def identities(self):
        self._check()
        from nca.train.jolt_admission import validate_config
        from nca.train.jolt_learning import BC_SPEC, PPO_SPEC
        root = Path(__file__).resolve().parents[3]
        self.execution_version = _execution_version(self.worker, self.manifest, self.declaration)
        if self.execution_version in (2, 3):
            from nca.train.jolt_execution_contract import validate_v2_allocation_identity, validate_v3_allocation_identity
            (validate_v3_allocation_identity if self.execution_version == 3 else
             validate_v2_allocation_identity)(self.declaration)
            _need(type(self.declaration.get("attempts")) is int and self.declaration["attempts"] == 1,
                  "exactly one integer invocation is required")
        _need(self.worker["status"] in ("completed", "failed", "capped"), "unknown allocation completion status")
        _same(self.worker["manifest"], self.manifest, "worker manifest changed")
        _need(self.manifest["config_sha256"] == sha256_json(self.config)
              and self.declaration["config_sha256"] == sha256_json(self.config)
              and self.manifest["declaration_sha256"] == sha256_json(self.declaration),
              "configuration/declaration identities differ")
        _same(self.config["bc"], BC_SPEC, "BC objective differs from its declaration")
        _same(self.config["ppo"], PPO_SPEC, "PPO objective differs from its declaration")
        _same(self.declaration["budget"], self.config["budget"], "declaration changed its complete allocation")
        _need(self.declaration["automatic_retry"] is False and self.declaration["automatic_extension"] is False,
              "declaration allowed continuation or automatic extension")
        validate_config(root, self.config)
        _same(self.manifest["source_files"], self.declaration["source_files"], "source inventory differs")
        _need(self.manifest["source_sha256"] == sha256_json(self.manifest["source_files"])
              == self.declaration["source_sha256"], "source aggregate differs")
        paths = (set((root / "src/nca").rglob("*.py")) | set((root / "src/nca").rglob("*.gd"))
                 | set((root / "tools").glob("*.py")) | set((root / "tests").rglob("*.py"))
                 | set((root / "configs/learning").glob("*.json")) | {root / "pyproject.toml"})
        inventory = {path.relative_to(root).as_posix() for path in paths if not path.name.startswith("jolt-execution-")}
        _need(set(self.manifest["source_files"]) == inventory, "source inventory omitted an executing dependency")
        for relative, digest in self.manifest["source_files"].items():
            self._check()
            _need(file_identity(_path(root, relative))["sha256"] == digest,
                  "executing offline reader differs from frozen source")
        _need(self.manifest["source_revision"] == self.declaration["source_revision"]
              and self.declaration["attempts"] == 1, "source revision or invocation count changed")
        _same(self.manifest["host_identity"], self.declaration["host_identity"], "host declaration changed")
        _same(self.manifest["runtime"], self.declaration["runtime"], "runtime declaration changed")
        _need(all(self.manifest["runtime"].get(key) == value for key, value in
                  {"torch": "2.8.0+cpu", "numpy": "2.3.5", "device": "cpu", "threads": 1,
                   "interop_threads": 1, "deterministic_algorithms": True}.items()),
              "unfrozen runtime")
        _need(torch.__version__ == "2.8.0+cpu", "offline neural replay requires the pinned CPU Torch runtime")
        if self.execution_version == 3:
            from nca.train.jolt_admission_v3 import validate_declaration_reviews, validate_resource_admission
            _need(self.declaration.get("allocation_id") == self.config["allocation_id"]
                  and self.declaration.get("authorization_ref") == self.config["authorization_ref"]
                  and self.declaration.get("teacher_prerequisite") == self.config["prerequisite"]
                  and self.declaration.get("opened_inventory") == self.config["opened_inventory"]
                  and self.declaration.get("split") == self.config["split"]
                  and self.declaration.get("ppo_case_stream_sha256") == sha256_json(self.config["ppo_case_stream"])
                  and self.declaration.get("comparison_order_sha256") == sha256_json(self.config["comparison_order"]),
                  "v3 declaration changed prospective cases, comparisons, or allocation identity")
            _need(self.declaration.get("output_identity") == {"basename": self.config["allocation_id"],
                  "lifecycle_basename": self.config["allocation_id"] + "-lifecycle", "reuse_forbidden": True},
                  "v3 declaration changed its original output identity")
            validate_declaration_reviews(self.declaration)
            validate_resource_admission(self.declaration)
        reviews = [] if self.execution_version == 3 else self.declaration["independent_reviews"]
        _need(self.execution_version == 3 or len(reviews) >= 2 and len({row["path"] for row in reviews}) == len(reviews),
              "two distinct independent source reviews are required")
        reviewers = set()
        for review in reviews:
            path = _path(root, review["path"])
            _need(review["path"].startswith("docs/evidence/")
                  and file_identity(path)["sha256"] == review["sha256"], "review identity changed")
            record = _json(path)
            if self.execution_version == 2:
                from nca.train.jolt_execution_contract import require_prospective_review
                require_prospective_review(record)
            _need(record["accepted"] is True and record["source_sha256"] == self.manifest["source_sha256"],
                  "review did not accept the frozen source closure")
            reviewer = record.get("reviewer")
            _need(isinstance(reviewer, str) and bool(reviewer) and reviewer not in reviewers,
                  "source reviews do not have distinct independent reviewers")
            reviewers.add(reviewer)
        rows = self.worker["results"]
        if self.execution_version == 3:
            from nca.train.jolt_storage_v3 import validate_worker_result_v3
            validate_worker_result_v3(self.worker)
            self._check()
        self.episodes = {row["id"]: row for row in rows}
        _need(len(self.episodes) == len(rows), "duplicate episode result")
        if self.execution_version == 3:
            from nca.train.jolt_budget_v3 import reservation_records
            _need(self.worker["costs_before_readback"].get("schema") == "nca-jolt-learn-budget/v3",
                  "v3 allocation cannot reinterpret an older budget snapshot")
            rows = reservation_records(_path(self.output, "budget-reservations.json"))
        else:
            reservations = _json(_path(self.output, "budget-reservations.json"), limit=MAX_RESERVATION_BYTES)
            _need(reservations["schema"] == "jolt-budget-reservations/v1", "unknown reservation artifact")
            rows = reservations["reservations"]
        self.budget = _Reservations(self.config["budget"], rows, self.worker["costs_before_readback"], check=self._check)
        _need(self.worker["costs_before_readback"]["reservation_count"] == len(self.budget.rows),
              "worker omitted trailing charged reservations")

    def checkpoint(self, item):
        self._check()
        identifier = item["id"]
        match = CHECKPOINT_ID.fullmatch(identifier)
        _need(match is not None and identifier not in self.checkpoints and self.frozen is None,
              "duplicate, undeclared or post-freeze checkpoint")
        _need(self.dataset is not None, "checkpoint preceded the frozen original dataset")
        family, seed, phase = match[1], int(match[2]), match[3]
        _need(family == "residual" or not phase.startswith("u"), "BC has no update-selected checkpoint")
        path = _path(self.output, item["path"])
        if self.execution_version == 3:
            _need(path.stat().st_size <= 1024 * 1024, "v3 checkpoint exceeds the frozen 1 MiB envelope")
        _need(item["path"] == "checkpoints/" + identifier + ".pt"
              and file_identity(path)["sha256"] == item["file_sha256"], "checkpoint bytes differ")
        self.reserve("checkpoint_loads", 1)
        self.loaded += 1
        policy, base, metadata = load_policy(path, expected_config_sha256=sha256_json(self.config))
        _need(policy.kind == family and policy_state_sha256(policy) == item["policy_sha256"],
              "checkpoint model differs from its publication")
        _need((None if base is None else policy_state_sha256(base)) == item["base_sha256"],
              "embedded BC base differs")
        for key, value in {"controller_id": identifier, "seed": seed,
                           "source_sha256": self.manifest["source_sha256"],
                           "source_revision": self.manifest["source_revision"],
                           "profile_sha256": self.config["profile_sha256"],
                           "dataset_sha256": sha256_json(self.dataset)}.items():
            _need(metadata.get(key) == value, "checkpoint metadata changed: " + key)
        if family == "residual":
            own = self.checkpoints.get(f"bc-seed{seed}-final")
            _need(own is not None and item["base_sha256"] == own["policy_sha256"],
                  "residual does not retain its own frozen learned BC base")
        if phase == "initial":
            _need(metadata["phase"] == "initial", "initial checkpoint was relabelled")
            with torch.random.fork_rng():
                torch.manual_seed(seed + (100000 if family == "residual" else 0))
                expected = JoltPolicy(family)
            _need(policy_state_sha256(expected) == item["policy_sha256"], "initial checkpoint differs from its fixed seed")
        elif family == "bc":
            _need(metadata["phase"] == "final" and metadata["epoch"] == 40
                  and self.active_fit is not None and self.active_fit["event"] == "bc_fitting_started"
                  and self.active_fit["seed"] == seed,
                  "BC final is not the fixed epoch forty")
        elif phase == "final":
            _need(metadata["phase"] == "final" and metadata["completed_updates"] == 8
                  and self.active_fit is None
                  and item["policy_sha256"] == self.checkpoints[f"residual-seed{seed}-u8"]["policy_sha256"],
                  "residual final differs from its fixed eighth update")
        else:
            _need(metadata["phase"] == "update" and metadata["completed_updates"] == int(phase[1:])
                  and self.active_fit is not None and self.active_fit["event"] == "ppo_fitting_started"
                  and self.active_fit["seed"] == seed and self.active_fit["update"] + 1 == int(phase[1:]),
                  "residual update count changed")
        self.models[identifier] = policy, base
        self.checkpoints[identifier] = item

    def _forward(self, policy, encoded, hidden):
        self.reserve("audit_forward", 1)
        self.forwards += 1
        with torch.no_grad():
            return policy(torch.tensor([encoded], dtype=next(policy.parameters()).dtype), hidden)

    def _read_episode(self, row):
        self._check()
        version3 = getattr(self, "execution_version", 1) == 3
        raw_path = _path(self.output, row["artifact"])
        sample_path = _path(self.output, row["sample_artifact"])
        _need(file_identity(raw_path)["sha256"] == row["artifact_sha256"]
              and file_identity(sample_path)["sha256"] == row["sample_sha256"], "episode bytes differ from result")
        if version3:
            from nca.train.jolt_storage_v3 import metrics_byte_limit_v3
            metrics_limit = metrics_byte_limit_v3(row["kind"])
        else:
            metrics_limit = MAX_JSON_BYTES
        _same(_json(_path(self.output, "metrics/" + row["id"] + ".json"), limit=metrics_limit), row,
              "independent metrics artifact differs from worker result")
        raw = _raw_episode_v3(raw_path) if version3 else read_episode(raw_path)
        self._check()
        sample = _sample(sample_path, execution_version=getattr(self, "execution_version", 1))
        self._check()
        _need(raw.id == sample.id == row["id"] and row["decisions"] == len(sample.decisions)
              and raw.consumed_ticks == row["consumed_ticks"], "episode identity/count differs")
        _need(raw.learner_revision == self.manifest["source_revision"]
              and all(self.manifest["source_files"].get(key) == digest for key, digest in raw.source_files.items()),
              "capture used different learner source")
        package_files = {key: digest for key, digest in self.manifest["source_files"].items()
                         if key.startswith("src/nca/") or key in ("pyproject.toml", "tools/capture_replay.py")}
        _same(raw.source_files, package_files, "capture omitted a source dependency")
        _need(raw.profile_sha256 == row["profile_sha256"] == self.config["profile_sha256"],
              "physical capability changed")
        _same(raw.host_identity, self.host_identity, "raw episode host differs from the original process")
        stage = row["stage"]
        _need(stage in ("teacher", "ppo", "open", "sealed"), "undeclared episode stage")
        cases = self.config["cases"]["teacher" if stage == "ppo" else stage]
        case = next((case for case in cases if case["id"] == row["case_id"]), None)
        _need(case is not None and case["host_seed"] == row["host_seed"], "episode opened an undeclared case")
        expected_config = physical_config("jev", ticks=240, seed=case["host_seed"])
        expected_config["actors"][0]["position"] = case["spawn"]
        _same(raw.config, expected_config, "native reset configuration differs from declared physical case")
        split = "training" if stage in ("teacher", "ppo") else stage
        _need(row["split"] == sample.split == split and raw.split == ("training" if split == "training" else "evaluation"),
              "case changed split")
        _need(raw.scenario_family == case["id"], "capture changed scenario identity")
        goal = SkillGoal(row["id"], "directed_hop", tuple(case["target"]), .8, 4., 4.)
        context = InvocationContext.create(raw.initial_observation, goal)
        _need(raw.initial_observation["tick"] == 0, "reset renewed a partial invocation")
        if getattr(self, "execution_version", 1) == 3:
            _same(raw.evaluation.get("case_stream_entry"), row.get("case_stream_entry"),
                  "raw episode changed the frozen case-stream position")
            _need((stage == "ppo") == isinstance(row.get("case_stream_entry"), Mapping),
                  "case-stream provenance is required exactly for PPO episodes")
        return {"row": row, "raw": raw, "sample": sample, "goal": goal, "context": context,
                "event_cursor": 0, "proposals": [], "reset": False, "profile": False, "retirement_note": None}

    @torch.no_grad()
    def episode(self, bundle):
        row, raw, sample = bundle["row"], bundle["raw"], bundle["sample"]
        goal, context = bundle["goal"], bundle["context"]
        version3 = getattr(self, "execution_version", 1) == 3
        steps = [event for event in raw.events if event.kind == "step"]
        _need(len(sample.decisions) <= len(steps), "sample invented a physical decision")
        _need(row["error"] is not None or len(steps) == len(sample.decisions), "ordinary raw rows were censored")
        frames = raw.evaluation["controller_frames"]
        _need(len(frames) == len(sample.decisions), "task frames do not cover scored decisions")
        task = HopTask(goal, raw.initial_observation, deadline_ticks=240, quiet_ticks_required=12)
        supervisor = PhysicalRecoverySupervisor()
        teacher = JoltRecoveryControllerV4() if row["kind"] == "authored" else None
        learned = row["kind"] in ("bc", "residual")
        policy, base = self.models[row["controller_id"]] if learned else (None, None)
        _need(policy.kind == row["kind"] if learned else row["kind"] == row["controller_id"]
              and row["kind"] in ("authored", "idle"), "episode family differs from its executing controller")
        hidden = None if policy is None else policy.zero_hidden()
        base_hidden = None if base is None else base.zero_hidden()
        expected_policy = (policy_state_sha256(policy) if learned else teacher.specification_sha256
                           if teacher is not None else sha256_json({"idle": WireAction(effort=0.).to_dict()}))
        _need(row["policy_sha256"] == expected_policy, "episode policy identity changed")
        planned_maximum = 40
        ppo_match = PPO_ID.fullmatch(row["id"]) if row["stage"] == "ppo" else None
        if row["stage"] == "ppo":
            _need(ppo_match is not None, "undeclared PPO rollout identity")
            prior = self.ppo_groups.get((int(ppo_match[1]), int(ppo_match[2])), [])
            planned_maximum = min(40, 256 - sum(len(episode.decisions) for _, episode in prior))
        _need(0 < planned_maximum <= 40 and len(sample.decisions) <= planned_maximum,
              "episode exceeded its fixed original rollout allocation")
        values, expected_history, rewards = [], ActionHistory(), []
        for index, record in enumerate(sample.decisions):
            self._check()
            event = record.event
            _same(replace(event, reward=steps[index].reward, terminated=steps[index].terminated,
                          truncated=steps[index].truncated).to_dict(), steps[index].to_dict(),
                  "learning sample changed original native receipt fields")
            _same(record.invocation, context.to_dict(), "sample renewed original invocation/deadline")
            _same(record.history, expected_history.to_dict(), "proposal and dispatch histories were conflated")
            _need(record.observation_valid and record.recurrent_reset == (index == 0 or sample.decisions[index - 1].overridden),
                  "capture recurrence reset differs from physical boundary")
            _same(event.source_observation, event.issued_observation, "synchronous original source observation changed")
            source = event.source
            _need(source.id == source.policy_id == row["controller_id"] and source.policy_sha256 == expected_policy
                  and source.training_privilege == "none"
                  and source.kind == ("learned" if learned else "scripted" if teacher else "neutral"),
                  "capture action source was relabelled")
            _need(event.context is not None and (event.context.goal_id, event.context.skill,
                  event.context.target, event.context.pace, event.context.tolerance, event.context.timeout_seconds)
                  == (goal.id, goal.skill, goal.target, goal.pace, goal.tolerance, goal.timeout_seconds),
                  "capture caller context differs from the declared invocation")
            observation = event.source_observation
            feedback = context.validate_observation(observation, goal)
            if teacher is not None:
                frame = teacher.step(feedback, goal)
                supervision = teacher._latest
                expected_action = frame.action
                _need(frames[index]["phase"] == frame.phase and frames[index]["reason"] == frame.reason,
                      "teacher diagnostic does not replay from the accepted ordinary controller")
            else:
                supervision = supervisor.admit(feedback, goal, emergency=emergency_request(feedback))
                expected_action = supervision.override or WireAction(effort=0.)
            overridden = supervision.override is not None
            _need(record.supervisor_sha256 == sha256_json(supervisor.specification())
                  and record.overridden == overridden and record.supervision_reason == supervision.reason,
                  "shared emergency supervision differs from original observations")
            _need(supervision.status == "running" and 6 <= supervision.action_lease_ticks <= 12,
                  "supervisor did not admit the retained interval")
            if learned:
                self.executions += 1
                if record.recurrent_reset or overridden:
                    hidden = policy.zero_hidden()
                    base_hidden = None if base is None else base.zero_hidden()
                if overridden:
                    _need(record.proposal is None and record.draw is None, "override invented a policy draw")
                    values.append(0.)
                else:
                    encoded = features(observation, goal, context=context, history=expected_history)
                    mean, logits, value, hidden = self._forward(policy, encoded, hidden)
                    self.runtime_forwards += 1
                    draw = record.draw
                    _need(draw is not None and draw.policy_sha256 == expected_policy
                          and draw.base_sha256 == (None if base is None else policy_state_sha256(base))
                          and draw.source_sha256 == self.manifest["source_sha256"]
                          and draw.feature_sha256 == sha256_json(encoded), "draw identity or inputs differ")
                    latent = torch.tensor([draw.latent], dtype=mean.dtype)
                    buttons = torch.tensor([draw.buttons], dtype=mean.dtype)
                    if base is not None:
                        base_mean, base_logits, _, base_hidden = self._forward(base, encoded, base_hidden)
                        self.runtime_forwards += 1
                        combined, logits = combine_residual(base_mean, base_logits, latent, logits)
                    else:
                        combined, logits = latent, logits.clamp(-BASE_LOGIT_LIMIT, BASE_LOGIT_LIMIT)
                    stochastic = row["stage"] == "ppo"
                    _need(draw.sampling_mode == ("stochastic_draw" if stochastic else "deterministic_mean"),
                          "evaluation and fitting draw modes were mixed")
                    if stochastic:
                        generator = self.action_generators[int(ppo_match[1])]
                        expected_latent = mean + policy.log_std.clamp(-5., .5).exp() * torch.randn(
                            mean.shape, generator=generator, dtype=mean.dtype, device=mean.device)
                        expected_buttons = torch.bernoulli(logits.sigmoid(), generator=generator)
                        _need(torch.equal(latent, expected_latent) and torch.equal(buttons, expected_buttons),
                              "stochastic sample differs from the original independent action RNG")
                        score = float(joint_log_probability(latent, mean, policy.log_std, buttons, logits)[0])
                        _near(score, draw.log_probability, "pre-transform joint behaviour density does not replay")
                    else:
                        _need(torch.equal(latent, mean) and draw.log_probability is None
                              and tuple(bool(item) for item in (logits[0] >= 0).tolist()) == draw.buttons,
                              "deterministic action differs from frozen mean/threshold")
                    expected_action = map_action(combined[0].tolist(), draw.buttons)
                    values.append(float(value[0]))
            else:
                _need(record.draw is None, "scripted comparator invented a learned sample")
                values.append(0.)
            expected_proposal = None if overridden else expected_action
            _need(record.proposal == expected_proposal, "original ordinary proposal does not replay")
            _same(event.requested_packet["action"], expected_action.to_dict(), "dispatch differs from admitted proposal/override")
            lease = (teacher.diagnostic_state()["action_lease_ticks"]
                     if version3 and teacher is not None else supervision.action_lease_ticks)
            _need(event.requested_packet["lease_ticks"] == lease
                  and event.requested_packet["observed_tick"] == observation["tick"]
                  and observation["tick"] + supervision.action_lease_ticks <= context.deadline_tick,
                  "dispatcher renewed lease/deadline")
            proposed = bundle["proposals"][index]
            _same({key: proposed[key] for key in ("invocation", "history", "proposal", "dispatched_packet", "draw")},
                  {"invocation": record.invocation, "history": record.history,
                   "proposal": None if record.proposal is None else record.proposal.to_dict(),
                   "dispatched_packet": event.requested_packet,
                   "draw": None if record.draw is None else record.draw.to_dict()},
                  "pre-dispatch journal proposal changed in retained samples")
            _same(proposed["supervision"], asdict(supervision), "pre-dispatch supervisor receipt differs")
            _need(proposed["observation_sha256"] == sha256_json(observation), "pre-dispatch observation changed")
            frame = frames[index]
            _need(type(frame["inference_ns"]) is int and frame["inference_ns"] >= 0
                  and frame["tick"] == observation["tick"] and frame["status"] == supervision.status,
                  "controller frame clocks/status differ")
            _same(frame["action"], expected_action.to_dict(), "controller frame action changed")
            traces = [{**tick.to_dict(), "failed": not tick.valid} for tick in event.ticks]
            terms = task.consume(expected_action, traces, inference_ns=frame["inference_ns"])
            rewards.append(terms)
            _near(event.reward, terms["total"], "sample reward differs from per-tick physical receipt")
            _need(event.terminated == (task.succeeded or task.deadline_missed), "task terminal flag differs from receipt")
            _need(event.truncated == (index + 1 == planned_maximum and not task.done),
                  "administrative cutoff flag differs from original physical boundary")
            expected_history = ActionHistory(record.proposal, expected_action, overridden, record.admitted)
            if learned and (overridden or not record.admitted):
                hidden = policy.zero_hidden()
                base_hidden = None if base is None else base.zero_hidden()
        persisted = row["policy_values"]
        if row["stage"] == "ppo":
            _need(isinstance(persisted, list) and len(persisted) == len(values),
                  "value evidence does not cover every PPO decision")
            for index, (actual, expected) in enumerate(zip(persisted, values)):
                if version3 and sample.decisions[index].proposal is None:
                    _need(actual is None, "value-less emergency override invented a model value")
                else:
                    _near(actual, expected, "retained value does not replay from original weights/history")
        else:
            _need(persisted is None, "non-PPO result invented fitting values")
        bootstrap = row["bootstrap_value"]
        safe_retirement = False
        if version3:
            from nca.evaluation.jolt_retirement import OUTCOME_SCHEMA, validate_safe_retirement_v3
            from nca.evaluation.jolt_learning import _validate_outcome_v3
            _validate_outcome_v3(row)
            _same(row["outcome"], raw.evaluation.get("outcome"), "raw episode outcome changed")
            _same(row["outcome"], sample.outcome, "sample episode outcome changed")
            _same(row.get("retirement"), raw.evaluation.get("retirement"), "raw retirement evidence changed")
            _same(row.get("retirement"), sample.retirement, "sample retirement evidence changed")
            safe_retirement = row["outcome"]["kind"] == "safe_retirement"
            if safe_retirement:
                _need(not task.done and bool(raw.events) and raw.events[-1].kind == "cancel",
                      "safe retirement replaced a task terminal or omitted native cancellation")
                cancelled = raw.events[-1]
                path = row["retirement"]["path"]
                _need((path == "ordinary_attempt" and len(sample.decisions) < planned_maximum)
                      or (path == "cutoff_peek" and row["stage"] == "ppo"
                          and len(sample.decisions) == planned_maximum),
                      "safe-retirement attempt does not match the fixed physical boundary")
                expected = validate_safe_retirement_v3(sample.decisions, cancelled.issued_observation,
                    goal, context, expected_history, path, after_cancel=cancelled.next_observation)
                _same(row["retirement"], expected, "safe retirement does not replay from complete physical history")
                _need(cancelled.reason == "emergency_recovery_deadline",
                      "native cancellation reason differs from the verified retirement")
                _need(row["bootstrap_observation_sha256"] is None
                      and row["bootstrap_observation_valid"] is False
                      and row["bootstrap_reason"] == "safe_retirement"
                      and (bootstrap == 0. if row["stage"] == "ppo" else bootstrap is None),
                      "terminal safe failure invented a bootstrap observation or model value")
                if learned and path == "ordinary_attempt":
                    self.executions += 1
            else:
                kind = ("integrity_error" if row["error"] is not None else
                        "task_terminal" if task.done else "administrative_cutoff")
                _need(row["outcome"]["kind"] == kind,
                      "versioned episode outcome differs from original physical task/error evidence")
        if safe_retirement:
            pass  # No neural forward or physical proposal exists on this path.
        elif row["bootstrap_observation_sha256"] is not None:
            _need(learned and row["stage"] == "ppo" and bool(sample.decisions) and not task.done,
                  "bootstrap invented after a task terminal or outside PPO")
            observation = sample.decisions[-1].event.next_observation
            _need(row["bootstrap_observation_sha256"] == sha256_json(observation),
                  "bootstrap observation is not the actually received pre-cancel view")
            feedback = context.validate_observation(observation, goal)
            supervision = supervisor.admit(feedback, goal, emergency=emergency_request(feedback))
            _need(supervision.status == "running", "unavailable final observation cannot bootstrap")
            if supervision.override is None:
                encoded = features(observation, goal, context=context, history=expected_history)
                _, _, value, _ = self._forward(policy, encoded, hidden)
                self.bootstrap_forwards += 1
                if base is not None:
                    self._forward(base, encoded, base_hidden)
                    self.bootstrap_forwards += 1
                _near(bootstrap, float(value[0]), "administrative bootstrap value does not replay")
                _need(row["bootstrap_observation_valid"] is True and row["bootstrap_reason"] == "received_cutoff_observation",
                      "valid final value was silently discarded")
            else:
                _need(bootstrap == 0. and row["bootstrap_observation_valid"] is False
                      and row["bootstrap_reason"] == "supervisor_value_unavailable",
                      "supervisor override invented a critic value")
        elif getattr(self, "execution_version", 1) in (2, 3):
            absent = (type(bootstrap) in (int, float) and bootstrap == 0.
                      if row["stage"] == "ppo" else bootstrap is None)
            terminal = (task.done and row["bootstrap_observation_valid"] is True
                        and row["bootstrap_reason"] == "task_terminal")
            unattempted = (row["error"] is not None and row["bootstrap_observation_valid"] is False
                           and row["bootstrap_reason"] == "not_attempted")
            attempt_failed = (row["error"] is not None and learned and row["stage"] == "ppo"
                              and not task.done and len(sample.decisions) == planned_maximum
                              and bool(sample.decisions) and sample.decisions[-1].event.truncated is True
                              and row["bootstrap_observation_valid"] is False
                              and row["bootstrap_reason"] == "attempt_failed")
            _need(absent and (terminal or unattempted or attempt_failed),
                  "v2 absent bootstrap invented a terminal, attempted value or validity")
            if attempt_failed:
                self.partial.append({"episode": row["id"], "reason": "bootstrap_attempt_failed"})
        else:
            _need((bootstrap == 0. if row["stage"] == "ppo" else bootstrap is None)
                  and row["bootstrap_observation_valid"] is True and row["bootstrap_reason"] == "task_terminal",
                  "a missing final bootstrap view invented value or validity")
        _same(raw.evaluation["reward_terms"], rewards, "retained reward decomposition differs")
        metrics = task.metrics(administratively_capped=not task.done)
        if safe_retirement:
            metrics.update(success=False, reason="emergency_recovery_deadline",
                           administratively_capped=False, capped_completion_seconds=4.)
        capped_error = (_budget_exhaustion_v3(row["error"]) if version3 else
                        row["error"] is not None and row["error"]["type"] == "AdmissionError")
        if row["error"] is not None and not capped_error:
            metrics["integrity_failure"] = True
        if version3 and row["error"] is not None:
            metrics["success"] = False
        _same(row["metrics"], metrics, "task metrics do not replay from original per-tick receipts")
        _same(raw.evaluation["metrics"], metrics, "raw evaluation metrics changed")
        _need(raw.evaluation["task_reward_schema"] == REWARD_SCHEMA, "task reward schema changed")
        status = ("completed" if safe_retirement else
                  "capped" if capped_error else
                  "failed" if row["error"] else "completed" if task.done else "capped")
        _need(raw.status == sample.status == status and raw.reason == sample.reason == metrics["reason"],
              "episode completion status differs from its physical task/error evidence")
        _same(raw.evaluation["error"], row["error"], "raw failure differs from original episode result")
        if len(steps) == len(frames):
            character = character_metrics(raw.initial_observation, goal.target, raw.events, frames)
            _same(row["character_metrics"], character, "character metrics do not replay")
            _same(raw.evaluation["character_metrics"], character, "raw character metrics changed")
        else:
            _need(row["error"] is not None and row["character_metrics"] is None,
                  "unscored failed physical interval invented character evidence")
            self.partial.append({"episode": row["id"], "unscored_native_decisions": len(steps) - len(frames)})
        _need(row["supervisor_interventions"] == sum(record.overridden for record in sample.decisions),
              "supervisor intervention count changed")
        if version3:
            closure = _episode_closure_v3(raw, row, deadline_missed=task.deadline_missed)
        elif getattr(self, "execution_version", 1) == 2:
            closure = _episode_closure_v2(raw, row, deadline_missed=task.deadline_missed)
        else:
            closure = bool(raw.events and raw.events[-1].next_observation
                           and raw.events[-1].next_observation.get("suspended"))
        _need(row["owner_closure_confirmed"] == closure
              and raw.evaluation["owner_closure_confirmed"] == closure, "episode closure has no native fence")
        if row["stage"] == "teacher":
            self.teacher_rows.append((row, _episode_counts_v3(sample) if version3 else sample))
        elif row["stage"] == "ppo":
            match = PPO_ID.fullmatch(row["id"])
            _need(match is not None, "undeclared PPO rollout identity")
            seed, update, index = map(int, match.groups())
            group = self.ppo_groups.setdefault((seed, update), [])
            _need(index == len(group), "PPO physical episodes were reordered")
            expected_source = f"residual-seed{seed}-" + ("initial" if update == 0 else f"u{update}")
            _need(row["controller_id"] == expected_source, "PPO update used another old policy checkpoint")
            if version3:
                _need(bool(sample.decisions) and len(group) < 256,
                      "PPO capture made zero progress or exceeded its fixed start ceiling")
            group.append((row, _episode_counts_v3(sample) if version3 else sample))
        self.verified_rows.append({"id": row["id"], "raw_sha256": row["artifact_sha256"],
            "sample_sha256": sample.sha256, "decisions": len(sample.decisions),
            "consumed_ticks": raw.consumed_ticks, "metrics_sha256": sha256_json(metrics),
            "gae_sha256": sha256_json(_advantages(sample.decisions, values, bootstrap,
                row["bootstrap_observation_valid"])) if row["stage"] == "ppo" else None})

    def freeze_dataset(self, record):
        self._check()
        _need(self.dataset is None and self.frozen is None, "dataset was reopened after fitting")
        self.dataset = _json(_path(self.output, "dataset.json"))
        expected_cases = self.config["cases"]["teacher"]
        _need([row["case_id"] for row, _ in self.teacher_rows] == [case["id"] for case in expected_cases],
              "teacher dataset censored or reordered a declared case")
        eligible = [decision for _, episode in self.teacher_rows for decision in episode.decisions
                    if decision.imitation_eligible]
        version3 = self.execution_version == 3
        expected = {"schema": "jolt-demonstrations/v3" if version3 else "jolt-demonstrations/v2",
            "episodes": [{"id": episode.id, "sha256": episode.sha256} for _, episode in self.teacher_rows],
            "eligible_samples": len(eligible),
            "fixed_windows_per_epoch": sum(math.ceil(len(episode.decisions) / 16) for _, episode in self.teacher_rows),
            "unsuccessful_episodes_retained": sum(not row["metrics"]["success"] for row, _ in self.teacher_rows),
            "secondary_positive_labels": sum(decision.proposal.secondary for decision in eligible),
            "selection": "all eligible training rows, with no success filter"}
        if version3:
            expected["safe_retirements"] = sum(row["outcome"]["kind"] == "safe_retirement" for row, _ in self.teacher_rows)
        _same(self.dataset, expected, "BC dataset omitted failures, abstention or ordinary teacher rows")
        _need(bool(eligible) and expected["secondary_positive_labels"] == 0
              and record["dataset_sha256"] == sha256_json(expected)
              and record["bc_optimizer_bound"] == 3 * 40 * expected["fixed_windows_per_epoch"],
              "dataset shape/secondary labels differ from prospective fitting contract")

    def fit_start(self, record):
        self._check()
        _need(self.active_fit is None and self.frozen is None and self.dataset is not None,
              "fitting overlapped or occurred outside the fixed dataset boundary")
        self.budget.snapshot(record["counters"])
        family = "bc" if record["event"] == "bc_fitting_started" else "residual"
        seed = record["seed"]
        _need(seed in POLICY_SEEDS, "undeclared training seed")
        if family == "bc":
            previous = [item["seed"] for item in self.fit_results if item["family"] == "bc"]
            _need(seed == POLICY_SEEDS[len(previous)] and not any(item["family"] == "residual" for item in self.fit_results),
                  "BC seeds were refitted or reordered")
            _need(record["reserved_upper_bound"] == 40 * self.dataset["fixed_windows_per_epoch"],
                  "BC fitting shape changed after data")
        else:
            _need({item["seed"] for item in self.fit_results if item["family"] == "bc"} == set(POLICY_SEEDS),
                  "PPO started before all independent BC bases froze")
            update = record["update"]
            prior = [item for item in self.fit_results if item["family"] == "residual"]
            _need((seed, update) == (POLICY_SEEDS[len(prior) // 8], len(prior) % 8),
                  "PPO seeds/updates were refitted or reordered")
            group = self.ppo_groups.get((seed, update), [])
            _need(sum(len(episode.decisions) for _, episode in group) == 256,
                  "PPO fitting did not retain exactly 256 original decisions")
            if self.execution_version == 3:
                _need(1 <= len(group) <= 256 and all(len(episode.decisions) > 0 for _, episode in group),
                      "PPO fitting censored zero-progress captures or exceeded its frozen stream budget")
        self.active_fit = record

    def fit_finish(self, record):
        self._check()
        start = self.active_fit
        _need(start is not None and record["seed"] == start["seed"], "fitting completion has no original start")
        family = "bc" if record["event"] == "bc_fitting_finished" else "residual"
        result = record["result"]
        _need(result["family"] == family and result["seed"] == record["seed"], "fit result identity changed")
        before, after = self.budget.snapshot(start["counters"]), self.budget.snapshot(record["counters"])
        expected = Counter(checkpoint_writes=1)
        if family == "bc":
            _need(start["event"] == "bc_fitting_started", "fitting family changed")
            for _, sample in self.teacher_rows:
                expected.update(_window_counts(sample.decisions, "bc"))
            relative = f"learning/bc-seed{record['seed']}.json"
            _same(result["spec"], self.config["bc"], "BC fitting result changed fixed hyperparameters")
            eligible = [row for _, episode in self.teacher_rows for row in episode.decisions if row.imitation_eligible]
            positives = [sum(row.proposal.primary for row in eligible), sum(row.proposal.secondary for row in eligible)]
            weights = [max(1., min(10., (len(eligible) - count) / max(count, 1))) for count in positives]
            if positives[1] == 0:
                weights[1] = 1.
            _same(result["positive_button_labels"], [float(value) for value in positives], "BC button positives changed")
            for actual, wanted in zip(result["positive_weight"], weights):
                _near(actual, wanted, "BC class weighting opened another split or changed objective")
            _need(len(result["epoch_losses"]) == 40 and all(type(loss) in (int, float) and math.isfinite(loss)
                  for loss in result["epoch_losses"]), "BC did not finish the declared forty epochs")
            shape = {"episodes": len(self.teacher_rows), "physical_rows": sum(len(episode.decisions) for _, episode in self.teacher_rows),
                "eligible_rows": len(eligible), "windows_per_epoch": self.dataset["fixed_windows_per_epoch"],
                "maximum_optimizer_calls": 40 * self.dataset["fixed_windows_per_epoch"]}
            _same(result["resolved_shape"], shape, "BC resolved shape changed")
        else:
            _need(start["event"] == "ppo_fitting_started" and record["update"] == start["update"]
                  and result["update"] == record["update"] + 1, "PPO update changed at completion")
            group = self.ppo_groups[(record["seed"], record["update"])]
            for _, sample in group:
                expected.update(_window_counts(sample.decisions, "ppo"))
            relative = f"learning/residual-seed{record['seed']}-u{record['update']+1}.json"
            _same(result["spec"], self.config["ppo"], "PPO fitting changed fixed hyperparameters")
            eligible = sum(decision.actor_eligible for _, sample in group for decision in sample.decisions)
            _need(result["actor_samples"] == eligible and result["excluded_actor_samples"] == 256 - eligible
                  and result["frozen_base_sha256"] == self.checkpoints[f"bc-seed{record['seed']}-final"]["policy_sha256"],
                  "PPO losses or frozen base identity changed")
            if self.execution_version == 3:
                _need(result.get("terminal_schema") == "nca-jolt-learning-terminal/v3"
                      and result.get("confirmed_decisions") == 256
                      and result.get("episodes") == len(group)
                      and result.get("safe_retirements") == sum(row["outcome"]["kind"] == "safe_retirement" for row, _ in group)
                      and result.get("new_terminal_penalty") is False,
                      "PPO terminal semantics, retained failures or exact update budget changed")
            for key in ("mean_loss", "mean_ratio", "clip_fraction"):
                _need(result[key] is None or type(result[key]) in (int, float) and math.isfinite(result[key]),
                      "nonfinite PPO diagnostics")
        for key in before:
            _need(after[key] - before[key] == expected[key], "fitting reserved cost differs from ordered physical windows: " + key)
        _need(result["optimizer_calls"] == expected["bc_optimizer" if family == "bc" else "ppo_optimizer"],
              "fit result optimizer count differs from reservations")
        if self.execution_version == 3:
            from nca.train.jolt_storage_v3 import MAX_LEARNING_JSON_BYTES
            learning_limit = MAX_LEARNING_JSON_BYTES
        else:
            learning_limit = MAX_JSON_BYTES
        _same(_json(_path(self.output, relative), limit=learning_limit), result, "fitting artifact changed after publication")
        self.expected_counts.update({key: value for key, value in expected.items() if key != "checkpoint_writes"})
        self.fit_results.append(result)
        if family == "residual":
            del self.ppo_groups[(record["seed"], record["update"])]
        self.active_fit = None

    def journal(self):
        path = _path(self.output, "owner-journal.jsonl")
        version3 = getattr(self, "execution_version", 1) == 3
        identity = journal_identity_v3(path, check=self._check) if version3 else validate_record_chain(path)
        closure = self.worker["closure"]
        durable = closure["durability"]
        if version3:
            _durability_v3(durable, identity, system=self.manifest.get("runtime", {}).get("system"))
        elif getattr(self, "execution_version", 1) == 2:
            _durability_v2(durable, identity, system=self.manifest.get("runtime", {}).get("system"))
        else:
            _need(durable["closed"] is True and durable["poisoned"] is False
                  and durable["records"] == durable["record_fsyncs"] == identity["records"]
                  and durable["total_fsyncs"] == identity["records"] + 1
                  and durable["chain_sha256"] == identity["chain_sha256"], "original owner journal did not durably close")
        active, pending, serial, finished_ids, uncertain = None, None, 0, [], []
        requests_by_serial = {}
        refused = 0
        native_counts, confirmed, requested = Counter(), 0, 0
        stage_requests = Counter()
        stage_episodes = Counter()
        reset_ids = set()
        scenario_rngs = {seed: random.Random(seed + 300000) for seed in POLICY_SEEDS}
        scenario_positions = Counter()
        update_starts = Counter()
        stage_positions = Counter()
        freeze_seen = opened_seen = False
        ordinary_stopped = False
        replay_digest, replay_count, replay_bytes = hashlib.sha256(), 0, 0
        line_limit = 64 * 1024 if version3 else MAX_RECORD_BYTES
        with path.open("rb") as stream:
            for line in iter(lambda: stream.readline(line_limit + 1), b""):
                self._check()
                _need(len(line) <= line_limit, "owner journal record is oversized")
                if version3:
                    replay_digest.update(line)
                    replay_count += 1
                    replay_bytes += len(line)
                    _need(replay_count <= identity["records"] and replay_bytes <= identity["bytes"],
                          "journal grew after its independent chain verification")
                record = strict_json_loads(line, max_bytes=line_limit)
                event = record["event"]
                if event == "host_start_reserved":
                    _need(not self.started, "host start was repeated")
                    self.budget.snapshot(record["counts"])
                elif event == "host_started":
                    _need(not self.started, "host process was restarted")
                    self.started = True
                    self.host_identity = record["identity"]
                    for key, value in self.manifest["host_identity"].items():
                        _same(self.host_identity.get(key), value, "started host differs from prospective identity")
                    _need(self.host_identity["lane"] == "act004" and self.host_identity["mode"] == "stepped",
                          "owner used another physical lane or mode")
                elif event == "episode_started":
                    _need(self.started and active is None and self.active_fit is None and not self.closed,
                          "episode overlaps work or occurs outside original host lifetime")
                    _need(not version3 or not ordinary_stopped, "v3 invocation resumed after a fatal evidence failure")
                    _need(record["id"] not in finished_ids, "episode was retried")
                    stage = record["stage"]
                    if stage in ("open", "sealed"):
                        _need(self.frozen is not None, "outcomes opened before all six final policies froze")
                    if stage == "sealed":
                        _need(self.opened, "sealed reference or candidate outcome preceded the sealed barrier")
                    else:
                        _need(not self.opened, "ordinary work reopened after sealed evidence")
                    if stage in ("teacher", "ppo"):
                        _need(self.frozen is None, "training data reopened after policy freeze")
                    row = self.episodes.get(record["id"])
                    if self.execution_version == 3:
                        entry = record.get("case_stream_entry")
                        if stage == "ppo":
                            match = PPO_ID.fullmatch(record["id"])
                            _need(match is not None, "invalid PPO episode id")
                            seed, update, episode_index = map(int, match.groups())
                            position = scenario_positions[seed]
                            expected_entry = {"seed": seed, "index": position, "update": update}
                            _same(entry, expected_entry, "PPO start changed its fixed seeded stream position")
                            _need(position < 2048 and update_starts[(seed, update)] < 256
                                  and episode_index == update_starts[(seed, update)],
                                  "PPO stream exhausted or update episodes were replaced")
                            frozen = self.config["ppo_case_stream"]["seeds"][str(seed)][position]
                            case = scenario_rngs[seed].choice(self.config["cases"]["teacher"])
                            _need(record["case_id"] == frozen == case["id"],
                                  "PPO case differs from the complete precomputed seeded stream")
                            scenario_positions[seed] += 1
                            update_starts[(seed, update)] += 1
                        else:
                            _need(entry is None, "non-PPO start invented a case-stream position")
                            identifiers = (["authored"] if stage == "teacher" else
                                ["authored", "idle"] + [name for name in controller_ids()[2:]
                                    if stage == "sealed" or name.endswith("-final")])
                            position = stage_positions[stage]
                            cases = self.config["cases"][stage]
                            _need(position < len(cases) * len(identifiers),
                                  "stage started more than the frozen paired schedule")
                            _need((record["case_id"], record["controller_id"]) ==
                                  (cases[position // len(identifiers)]["id"], identifiers[position % len(identifiers)]),
                                  "case/controller order changed after outcomes")
                            if stage in ("open", "sealed"):
                                _same(self.config["comparison_order"][stage][position],
                                      {"case_id": record["case_id"], "controller_id": record["controller_id"]},
                                      "paired outcome order differs from the complete frozen comparison stream")
                            stage_positions[stage] += 1
                    if row is None:
                        active = {"row": None, "start": record, "proposals": [], "event_cursor": 0}
                    else:
                        _need(all(row[key] == record[key] for key in ("id", "case_id", "controller_id", "stage")),
                              "episode result differs from original start")
                        if self.execution_version == 3:
                            _same(row.get("case_stream_entry"), record.get("case_stream_entry"),
                                  "PPO result changed its prospective stream entry")
                        active = self._read_episode(row)
                        if stage == "teacher":
                            _need(row["kind"] == row["controller_id"] == "authored", "teacher dataset used another policy")
                        if stage == "ppo" and self.execution_version != 3:
                            match = PPO_ID.fullmatch(row["id"])
                            _need(match is not None, "invalid PPO episode id")
                            case = scenario_rngs[int(match[1])].choice(self.config["cases"]["teacher"])
                            _need(case["id"] == row["case_id"], "PPO case order differs from fixed independent seed")
                    self.expected_counts["episodes_started"] += 1
                    stage_episodes[stage] += 1
                elif event == "decision_proposed":
                    _need(active is not None and pending is None and record["index"] == len(active["proposals"]),
                          "decision was reordered or proposed with unresolved native work")
                    _need(not active.get("retirement_note"), "retired invocation proposed another action")
                    _need(not version3 or not ordinary_stopped, "v3 decision resumed after a fatal evidence failure")
                    identity_id = active["row"]["id"] if active["row"] else active["start"]["id"]
                    _need(record["episode_id"] == identity_id, "proposal belongs to another invocation")
                    active["proposals"].append(record)
                elif event == "owner_request":
                    _need(self.started and not self.closed and record["serial"] == serial + 1,
                          "owner request was duplicated or reordered")
                    operation = record["operation"]
                    _need(operation in ("reset", "profile", "step", "cancel", "close"),
                          "undeclared native operation")
                    _need(not version3 or not ordinary_stopped or operation in ("cancel", "close"),
                          "v3 physical work resumed after a fatal evidence failure")
                    _need(active is not None or operation == "close", "owner request lost its episode")
                    _need(not active or not active.get("retirement_note"),
                          "retired invocation performed another native operation")
                    if pending is not None:
                        _need(operation in ("cancel", "close") and not self.complete, "ordinary work resumed after uncertain exchange")
                        if pending["serial"] not in uncertain:
                            uncertain.append(pending["serial"])
                    self.budget.owner(record)
                    serial += 1
                    native_counts[operation] += 1
                    pending = record
                    requests_by_serial[serial] = record
                    if operation == "step":
                        _need(record["fields"]["ticks"] == 6 and len(record["fields"]["commands"]) == 1
                              and bool(active["proposals"]), "native work lacks one six-tick proposal")
                        _same(record["fields"]["commands"][0], active["proposals"][-1]["dispatched_packet"],
                              "native request differs from durably journaled proposal")
                        requested += 6
                        stage_requests[record["stage"]] += 1
                    if active and active["row"]:
                        _need(record["stage"] == active["row"]["stage"], "native work changed stage")
                        if operation == "reset":
                            _need(not active["reset"] and active["event_cursor"] == 0, "episode reset was repeated")
                            _same(record["fields"], {"config": active["raw"].config, "actor_id": 0},
                                  "native reset request changed retained configuration")
                        elif operation == "profile":
                            _need(active["reset"] and not active["profile"] and not record["fields"],
                                  "physical profile request changed")
                elif event == "owner_response":
                    _need(pending is not None and record["serial"] == pending["serial"],
                          "native response has no matching original request")
                    response, operation = record["response"], pending["operation"]
                    _need(response.get("id") == pending["serial"] and response.get("ok") is True,
                          "native response correlation or acceptance failed")
                    malformed = ((operation == "step" and (type(response.get("actual_ticks")) is not int
                                  or not 0 <= response["actual_ticks"] <= 6))
                                 or (operation == "close" and response.get("closed") is not True))
                    if malformed:
                        _need(not self.complete, "completed allocation contains malformed native consumption or closure")
                        if pending["serial"] not in uncertain:
                            uncertain.append(pending["serial"])
                        self.partial.append({"owner_serial": pending["serial"], "reason": "malformed_native_response"})
                        continue
                    if operation == "close":
                        _need(response.get("closed") is True, "close response did not confirm native owner closure")
                        self.closed = True
                    elif active and active["row"]:
                        raw = active["raw"]
                        if operation == "reset":
                            _need(response.get("actual_ticks") == 0 and response.get("tick") == 0
                                  and response.get("failed") is not True and response.get("schedule") == self.host_identity["schedule"],
                                  "native reset does not preserve accepted schedule")
                            _same(response["observation"], raw.initial_observation, "native reset observation differs")
                            run_id = raw.initial_observation["run_id"]
                            _need(run_id not in reset_ids, "reset reused an original physical lifetime")
                            reset_ids.add(run_id)
                            active["reset"] = True
                        elif operation == "profile":
                            _same(response["profile"], raw.profile, "native physical profile differs")
                            active["profile"] = True
                        else:
                            cursor = active["event_cursor"]
                            # A local failure record is evidence of a missing response,
                            # not another native request or a physical receipt.
                            while cursor < len(raw.events) and raw.events[cursor].kind == "failure":
                                _need(not self.complete, "completed experiment contains a transport failure")
                                cursor += 1
                            _need(cursor < len(raw.events), "native response has no retained raw event")
                            captured = raw.events[cursor]
                            _need(captured.kind == operation, "native operation differs from raw event")
                            if operation == "step":
                                _need(active["reset"] and active["profile"], "step preceded reset/profile evidence")
                                _native_step(captured, pending, response, schedule=self.host_identity["schedule"])
                            else:
                                _same(pending["fields"], {"run_id": captured.issued_observation["run_id"],
                                      "epoch": captured.issued_observation["epoch"]}, "cancellation targeted another owner lifetime")
                                _need(response.get("cancelled") is True, "native cancellation was refused")
                                _same(response["observation"], captured.next_observation, "native cancellation view differs")
                                cancellation_receipt(captured.issued_observation, captured.next_observation)
                            active["event_cursor"] = cursor + 1
                    if operation == "step":
                        amount = response.get("actual_ticks")
                        _need(type(amount) is int and 0 <= amount <= 6, "native response exceeds reserved ticks")
                        confirmed += amount
                        if any(admission.get("accepted") is not True for trace in response.get("traces", ())
                               for admission in trace.get("admission", ())):
                            refused += 6
                    pending = None
                elif event == "safe_retirement_verified":
                    _need(self.execution_version == 3 and active is not None and active["row"] is not None
                          and pending is None and not active.get("retirement_note")
                          and record.get("episode_id") == active["row"]["id"]
                          and active["event_cursor"] > 0
                          and active["raw"].events[active["event_cursor"] - 1].kind == "cancel",
                          "safe retirement lacks its original cancellation boundary")
                    _same(record.get("retirement"), active["row"].get("retirement"),
                          "retirement changed after its durable original note")
                    active["retirement_note"] = record["retirement"]
                elif event == "episode_finished":
                    _need(active is not None and active["row"] is not None, "episode completion has no retained start/artifact")
                    _same(record["row"], active["row"], "worker result changed original episode completion")
                    raw = active["raw"]
                    tail = raw.events[active["event_cursor"]:]
                    _need(not tail or not self.complete and all(item.kind == "failure" for item in tail),
                          "raw capture contains receipts absent from original owner journal")
                    _need(len(active["proposals"]) >= len(active["sample"].decisions), "sample invented an unjournaled proposal")
                    if self.execution_version == 3:
                        _same(active.get("retirement_note"), active["row"].get("retirement"),
                              "safe failure was continued without its durable retirement evidence")
                    self.episode(active)
                    finished_ids.append(active["row"]["id"])
                    self.expected_counts["episodes_aborted" if active["row"]["error"] else "episodes_completed"] += 1
                    ordinary_stopped = ordinary_stopped or active["row"]["error"] is not None
                    active = None
                elif event == "episode_failed_before_capture":
                    _need(active is not None and active["row"] is None and not self.complete,
                          "startup failure was disguised as a completed episode")
                    self.partial.append({"episode": record["id"], "reason": "failed_before_capture"})
                    ordinary_stopped = True
                    active = None
                elif event == "fitting_shape_frozen":
                    _need(not version3 or not ordinary_stopped, "v3 dataset froze after a fatal evidence failure")
                    self.freeze_dataset(record)
                elif event == "checkpoint_published":
                    _need(not version3 or not ordinary_stopped, "v3 checkpoint published after a fatal evidence failure")
                    self.checkpoint(record["checkpoint"])
                elif event in ("bc_fitting_started", "ppo_fitting_started"):
                    _need(not version3 or not ordinary_stopped, "v3 fitting resumed after a fatal evidence failure")
                    self.fit_start(record)
                elif event in ("bc_fitting_finished", "ppo_fitting_finished"):
                    self.fit_finish(record)
                elif event == "six_finals_frozen":
                    _need(not freeze_seen and self.active_fit is None and len(self.fit_results) == 27,
                          "final freeze omitted prescribed fitting or was repeated")
                    receipt = _json(_path(self.output, "frozen-finals.json"))
                    _same(record["receipt"], receipt, "final freeze receipt changed")
                    _need(record["sha256"] == sha256_json(receipt)
                          and receipt["schema"] == "jolt-six-final-freeze/v1"
                          and receipt["source_sha256"] == self.manifest["source_sha256"], "final freeze identity differs")
                    expected = {f"{family}-seed{seed}-final" for family in ("bc", "residual") for seed in POLICY_SEEDS}
                    _need(len(receipt["finals"]) == 6 and {item["id"] for item in receipt["finals"]} == expected,
                          "six-final barrier omitted a seed or family")
                    for item in receipt["finals"]:
                        _same(item, self.checkpoints[item["id"]], "final freeze changed a published checkpoint")
                    self.frozen, freeze_seen = receipt, True
                elif event == "sealed_outcomes_opened":
                    _need(self.frozen is not None and not opened_seen and active is None,
                          "sealed outcome opened before freeze or reopened")
                    digest = sha256_json(self.frozen)
                    _need(record["frozen_finals_sha256"] == digest, "sealed barrier used another set of finals")
                    _same(_json(_path(self.output, "sealed-opened.json")), {
                        "frozen_finals_sha256": digest, "source_sha256": self.manifest["source_sha256"],
                        "before_first_sealed_outcome": True}, "sealed opening receipt changed")
                    self.opened, opened_seen = True, True
                elif event == "allocation_stopped":
                    _need(not self.complete and record["status"] == self.worker["status"], "allocation stop status changed")
                    self.budget.snapshot(record["counters"])
                    _same(record["error"], self.worker["error"], "worker error differs from original stop")
                    ordinary_stopped = True
                elif event == "host_terminated":
                    _need(not self.terminated and record["process_exited"] is True
                          and record["native_close_confirmed"] == self.closed, "host termination evidence differs")
                    self.terminated = True
                elif event == "worker_closed":
                    _need(record["status"] == self.worker["status"], "worker close status changed")
                    self.budget.snapshot(record["counters"])
                else:
                    _need(False, "undeclared event in original owner journal: " + str(event))
        if version3:
            _need(replay_digest.hexdigest() == identity["sha256"]
                  and replay_count == identity["records"] and replay_bytes == identity["bytes"],
                  "journal original bytes changed between independent chain verification and replay")
        if pending is not None and pending["serial"] not in uncertain:
            _need(not self.complete, "completed allocation retains an unconfirmed owner request")
            uncertain.append(pending["serial"])
        uncertain_ticks = 6 * sum(requests_by_serial[number]["operation"] == "step" for number in uncertain)
        _need(finished_ids == [row["id"] for row in self.worker["results"]], "worker omitted or reordered a retained physical episode")
        _same(list(self.checkpoints.values()), self.worker["checkpoints"], "worker changed published checkpoints")
        _same(self.fit_results, self.worker["learning"], "worker omitted or changed fitting results")
        _need(closure["native_close_confirmed"] == self.closed
              and closure["pending_owner_request"] == (None if pending is None else pending["serial"])
              and closure["uncertain_owner_requests"] == uncertain
              and closure["requested_ticks"] == requested and closure["confirmed_ticks"] == confirmed
              and closure["refused_ticks"] == refused and closure["uncertain_ticks"] == uncertain_ticks,
              "worker native closure/costs differ from original requests/responses")
        _same(closure["owner_requests"], {key: native_counts[key] for key in ("reset", "profile", "configure", "cancel", "close", "step")},
              "owner operation totals changed")
        self.expected_counts.update(attempts=1, host_starts=1, host_start_attempts=1,
            host_terminations=int(self.terminated), owner_requests=sum(native_counts.values()),
            decisions=native_counts["step"], requested_ticks=requested, confirmed_ticks=confirmed,
            refused_ticks=refused, uncertain_ticks=uncertain_ticks,
            checkpoint_writes=len(self.checkpoints), checkpoint_executions=self.executions,
            inference_forward=self.runtime_forwards, bootstrap_forward=self.bootstrap_forwards)
        self.expected_counts.update({"owner_" + key: value for key, value in native_counts.items()})
        if self.complete:
            self.expected_counts["checkpoint_loads"] = 12
            _need(self.started and self.terminated and self.closed and pending is None and not uncertain
                  and active is None and self.active_fit is None and self.opened and len(self.checkpoints) == 36,
                  "completed allocation lacks original closure/fitting/freeze evidence")
            for key, count in self.budget.counts.items():
                _need(count == self.expected_counts[key], "completed cost differs from actual recorded work: " + key)
        else:
            for key, count in self.expected_counts.items():
                _need(self.budget.counts[key] >= count, "partial invocation undercounted retained work: " + key)
            self.partial.append({"reason": "incomplete_allocation", "status": self.worker["status"],
                                 "uncertain_owner_requests": uncertain,
                                 "pending_owner_request": closure["pending_owner_request"]})
        for stage, count in stage_requests.items():
            _need(self.budget.stages[stage]["decisions"] == count
                  and self.budget.stages[stage]["requested_ticks"] == count * 6,
                  "stage costs differ from actual native requests")
        for stage, costs in self.budget.stages.items():
            if "episodes_started" in costs:
                _need(costs["episodes_started"] == stage_episodes[stage] if self.complete
                      else costs["episodes_started"] >= stage_episodes[stage],
                      "episode stage reservations differ from original starts")
        return identity

    def finish(self):
        self.identities()
        journal = self.journal()
        evidence_complete = self.complete and not self.partial
        qualification = qualify(self.worker["results"], self.config["cases"]["sealed"],
                                evidence_complete=evidence_complete, execution_version=self.execution_version)
        _same(self.worker["qualification"], qualify(self.worker["results"], self.config["cases"]["sealed"],
              evidence_complete=False, execution_version=self.execution_version),
              "worker pre-readback qualification changed")
        if self.complete:
            open_pairs = {(row["case_id"], row["controller_id"]) for row in self.worker["results"] if row["stage"] == "open"}
            expected_open = {(case["id"], identifier) for case in self.config["cases"]["open"]
                             for identifier in ("authored", "idle", *(name for name in controller_ids() if name.endswith("-final")))}
            _need(open_pairs == expected_open and sum(row["stage"] == "open" for row in self.worker["results"]) == 32
                  and qualification["complete_pairs"], "completed experiment omitted paired open or sealed cases")
        self._check()
        return {"schema": {1: SCHEMA, 2: SCHEMA_V2, 3: SCHEMA_V3}[self.execution_version],
                "verified": True, "complete": evidence_complete,
                "qualified": bool(evidence_complete and qualification["qualified"]),
                "worker_status": self.worker["status"], "worker_result_sha256": file_identity(self.output / "worker-result.json")["sha256"],
                "source_sha256": self.manifest["source_sha256"], "config_sha256": sha256_json(self.config),
                "owner_journal": journal, "verified_episodes": self.verified_rows,
                "checkpoint_loads": self.loaded, "audit_forward": self.forwards,
                "costs_reconstructed": self.budget.counts, "partial_evidence": self.partial,
                "qualification": qualification,
                "evidence_limits": ["offline replay verifies original recorded timings, not a new latency measurement",
                    "native physical receipts are checked against the durable owner response stream",
                    "fitting is authenticated by frozen source, ordered data, optimizer reservations and retained checkpoints; no retraining occurs",
                    "owner feel and morphology transfer remain unaccepted"]}


def verify_output(output: Path, reserve: Callable[[str, int], Any], *, check=None) -> dict[str, Any]:
    """Read immutable worker artifacts once; never acquire a host or fit weights.

    Invalid evidence returns a retained negative audit, including the already
    charged readback costs. A complete or partial structurally valid allocation
    is reported separately from gameplay qualification.
    """
    if not callable(reserve):
        raise ValueError("offline replay requires prospective load/forward reservations")
    audit = None
    try:
        audit = _Audit(Path(output), reserve, check=check)
        return audit.finish()
    except Exception as exc:
        return {"schema": {1: SCHEMA, 2: SCHEMA_V2, 3: SCHEMA_V3}[audit.execution_version] if audit is not None else SCHEMA,
                "verified": False, "complete": False, "qualified": False,
                "error": {"type": type(exc).__name__, "code": getattr(exc, "code", None),
                          "reason": str(exc) if isinstance(exc, ContractError) and exc.code == "jolt_audit" else None},
                "verified_episodes": [] if audit is None else audit.verified_rows,
                "checkpoint_loads": 0 if audit is None else audit.loaded,
                "audit_forward": 0 if audit is None else audit.forwards,
                "partial_evidence": [] if audit is None else audit.partial,
                "owner_feel_accepted": False, "morphology_transfer_admitted": False}

