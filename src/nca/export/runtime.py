"""Small independent NumPy FP32 GRU and a strict external CPU owner boundary.

No Torch, optimizer, provider call or world mutation is part of numeric inference.
Fallback is a zero request plus an explicit requirement to cancel the real owner;
the zero request is never used as a replacement release input by dispatch().
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from nca.contracts import ContractError, WireAction, canonical_json, sha256_json, strict_json_loads
from nca.controllers.baseline import FeedbackObservation, SkillGoal
from nca.data.records import policy_view
from .spec import (ACTIONS, EXPECTED_SPECIFICATION, NORMALIZATION, RECURRENCE,
                   SENSORS, WEIGHT_SHAPES, RuntimeContext, source_files,
                   PINNED_HOST, PINNED_PROFILE_SHA256)


def _clip(value, low=-1.0, high=1.0):
    return max(low, min(high, value))


def encode(observation: Mapping[str, Any], goal: SkillGoal, previous: WireAction, start_tick: int) -> tuple[float, ...]:
    """Independent implementation of the bundled fixed normalization equations."""
    view = FeedbackObservation.from_host(policy_view(observation))
    if view.actor != "jev" or goal.skill != "directed_hop" or not view.valid or not 0 <= start_tick <= view.tick:
        raise ContractError("features", "valid Jolt directed-hop feedback and goal start required")
    def optional(value, scale=1.0):
        return (0.0, 0.0) if value is None else (_clip(float(value) / scale, 0, 2), 1.0)
    result = (
        _clip((goal.target[0]-view.position[0])/64, -2, 2),
        _clip((goal.target[1]-view.position[1])/64, -2, 2),
        _clip(view.velocity[0]/80, -2, 2), _clip(view.velocity[1]/100, -2, 2),
        math.sin(view.angle), math.cos(view.angle), _clip(view.angular_velocity/8, -2, 2),
        float(view.supported is True), float(view.supported is not None),
        *optional(view.energy), *optional(view.cooldown), *optional(view.ground_distance, 96),
        *optional(view.ceiling_distance, 96),
        max(0.0, 1.0-(view.tick-start_tick)/(60*goal.timeout_seconds)), goal.pace, goal.tolerance/8,
        previous.move[0], previous.trim, previous.effort, float(previous.primary),
        float(previous.secondary), float(previous.auxiliary))
    if not all(math.isfinite(value) for value in result):
        raise ContractError("features", "all normalized features must be finite")
    return result


def map_action(outputs, observation: Mapping[str, Any], goal: SkillGoal) -> WireAction:
    if len(outputs) != 6 or not np.isfinite(outputs).all():
        raise ContractError("output", "six finite BC outputs required")
    dx = goal.target[0] - observation["position"][0]
    dy = goal.target[1] - observation["position"][1]
    aim = (_clip(dx/30), min(-.55, dy/30-.55))
    norm = math.hypot(*aim)
    aim = tuple(value / norm for value in aim) if norm > 1e-12 else (0.0, 0.0)
    return WireAction(move=(_clip(float(outputs[0])), 0), aim=aim,
                      trim=_clip(float(outputs[1])), effort=_clip(float(outputs[2]), 0, 1),
                      primary=bool(outputs[3] >= .5), secondary=bool(outputs[4] >= .5),
                      auxiliary=bool(outputs[5] >= .5))


def _sigmoid(value):
    result = np.empty_like(value, dtype=np.float32)
    positive = value >= 0
    result[positive] = 1 / (1 + np.exp(-value[positive]))
    exponent = np.exp(value[~positive])
    result[~positive] = exponent / (1 + exponent)
    return result


class NumpyGRU:
    """Row-major reset/update/new gates with the declared reset-after candidate."""
    def __init__(self, weights: Mapping[str, Any]):
        if not isinstance(weights, Mapping) or set(weights) != set(WEIGHT_SHAPES):
            raise ContractError("weight_names", "exact BC v1 weights required")
        self.weights = {}
        for name, shape in WEIGHT_SHAPES.items():
            try:
                raw = np.asarray(weights[name])
            except (TypeError, ValueError) as exc:
                raise ContractError("weight_shape", name) from exc
            if raw.dtype.kind not in "fiu" or raw.shape != shape:
                raise ContractError("weight_shape", name)
            with np.errstate(over="ignore", invalid="ignore"):
                value = np.array(raw, dtype=np.float32, copy=True)
            if not np.isfinite(value).all():
                raise ContractError("nonfinite_weights", name)
            value.setflags(write=False)
            self.weights[name] = value
        self.weights = MappingProxyType(self.weights)

    @property
    def nbytes(self):
        return sum(value.nbytes for value in self.weights.values())

    @property
    def policy_sha256(self):
        # Preserve the learner v1 content algorithm, including specification
        # field order and exact FP32 bytes. It is not a JSON-file identity.
        result = hashlib.sha256(str(EXPECTED_SPECIFICATION).encode())
        for name, value in sorted(self.weights.items()):
            result.update(name.encode())
            result.update(str(tuple(value.shape)).encode())
            result.update(value.tobytes(order="C"))
        return result.hexdigest()

    def forward(self, inputs, hidden, *, reset=False):
        x, h = np.asarray(inputs, dtype=np.float32), np.asarray(hidden, dtype=np.float32)
        if x.shape != (26,) or h.shape != (32,) or not np.isfinite(x).all() or not np.isfinite(h).all():
            raise ContractError("network_input", "finite feature26 and hidden32 vectors required")
        if type(reset) is not bool:
            raise ContractError("network_reset", "explicit boolean reset required")
        if reset:
            h = np.zeros(32, dtype=np.float32)
        w = self.weights
        with np.errstate(over="raise", invalid="raise"):
            encoded = np.tanh(w["encoder.weight"] @ x + w["encoder.bias"])
            gi = w["recurrent.weight_ih"] @ encoded + w["recurrent.bias_ih"]
            gh = w["recurrent.weight_hh"] @ h + w["recurrent.bias_hh"]
            r = _sigmoid(gi[:32] + gh[:32])
            z = _sigmoid(gi[32:64] + gh[32:64])
            n = np.tanh(gi[64:] + r * gh[64:])
            following = (1-z)*n + z*h
            logits = w["actor.weight"] @ following + w["actor.bias"]
            value = float((w["critic.weight"] @ following + w["critic.bias"])[0])
            outputs = np.concatenate((np.tanh(logits[:2]), _sigmoid(logits[2:])))
        if not np.isfinite(following).all() or not np.isfinite(outputs).all() or not math.isfinite(value):
            raise ContractError("network_nonfinite", "nonfinite inference result")
        return outputs, value, following, logits


class RuntimeBundle:
    """Owned numeric artifact admitted only under its exact measured context."""
    def __init__(self, value: Mapping[str, Any], context: RuntimeContext):
        if not isinstance(value, Mapping) or not isinstance(context, RuntimeContext):
            raise ContractError("bundle_schema", "structured bundle and typed runtime context required")
        payload = strict_json_loads(canonical_json(value))
        expected = {"format", "specification", "context", "normalization", "sensors", "actions", "recurrence",
                    "profile", "weights", "weights_sha256", "policy_sha256", "training_evaluation",
                    "runtime_sources", "qualification", "sha256"}
        if set(payload) != expected or payload["format"] != "nca-cpu-policy/v1":
            raise ContractError("bundle_version", "exact numeric CPU bundle v1 required")
        declared = payload.pop("sha256")
        if sha256_json(payload) != declared:
            raise ContractError("bundle_hash", "bundle content differs from its identity")
        payload["sha256"] = declared
        for key, expected_value in (("specification", EXPECTED_SPECIFICATION), ("normalization", NORMALIZATION),
                                    ("sensors", SENSORS), ("actions", ACTIONS), ("recurrence", RECURRENCE)):
            if sha256_json(payload[key]) != sha256_json(expected_value):
                raise ContractError("contract_mismatch", key)
        expected_context = RuntimeContext.from_dict(payload["context"])
        if expected_context != context:
            fields = [key for key in context.to_dict() if context.to_dict()[key] != expected_context.to_dict()[key]]
            raise ContractError("runtime_context_mismatch", ",".join(fields))
        if (context.actor, context.profile_version, context.base_hz, context.decision_ticks,
                context.helper_lease_ticks, context.lane) != ("jev", 1, 60, 6, 12, "act004"):
            raise ContractError("unsupported_timing_profile", "Jolt v1 at60Hz/6decision/12lease only")
        if (context.profile_sha256 != PINNED_PROFILE_SHA256
                or any(getattr(context, key) != value for key, value in PINNED_HOST.items())
                or context.sensor_sha256 != sha256_json(SENSORS)
                or context.action_sha256 != sha256_json(ACTIONS)
                or context.normalization_sha256 != sha256_json(NORMALIZATION)):
            raise ContractError("unqualified_context", "pinned profile, host and control transforms required")
        if (context.python, context.numpy, context.platform, context.architecture) != (
                "3.12.14", "2.3.5", "Linux", "x86_64"):
            raise ContractError("unqualified_runtime", "only the declared Linux external CPU runtime")
        if payload["runtime_sources"] != source_files():
            raise ContractError("runtime_sources", "preprocessing/controller dependency bytes changed")
        if sha256_json(payload["weights"]) != payload["weights_sha256"]:
            raise ContractError("weights_hash", "numeric weight content changed")
        if not isinstance(payload["profile"], Mapping) or sha256_json(payload["profile"]) != context.profile_sha256:
            raise ContractError("profile_hash", "full authored profile differs")
        if sha256_json({"profile": payload["profile"]["profile"], "joints": payload["profile"]["joints"]}) != context.assist_sha256:
            raise ContractError("assist_hash", "authored assistance or connection changed")
        evidence = payload["training_evaluation"]
        if not isinstance(evidence, Mapping):
            raise ContractError("training_evidence", "evidence must be a structured manifest")
        for key in ("checkpoint_sha256", "numeric_source_sha256", "summary_sha256", "config_sha256",
                    "manifest_sha256", "learner_sources_sha256", "dataset_sha256"):
            from nca.contracts import digest
            if not isinstance(evidence.get(key), str):
                raise ContractError("training_evidence", key)
            digest(evidence.get(key), key)
        if (evidence.get("status") != "trained_and_evaluated" or evidence.get("kind") != "bc"
                or evidence.get("training_seed") != 1103 or evidence.get("training_privilege") != "none"
                or evidence.get("profile_sha256") != context.profile_sha256):
            raise ContractError("training_evidence", "fixed evaluated BC1103 evidence required")
        evidence_kind = evidence.get("evidence_kind")
        if evidence_kind not in (None, "staged_fixed_candidate/v1"):
            raise ContractError("training_evidence", "unknown evidence admission contract")
        if evidence_kind == "staged_fixed_candidate/v1":
            if (evidence.get("training_status") != "interrupted_no_summary"
                    or evidence.get("training_receipt_file") != "interruption-index.json"
                    or evidence.get("training_receipt_schema") != "nca-interrupted-training-artifact-index/v1"
                    or evidence.get("original_termination_confirmed") is not False
                    or evidence.get("selected_evaluation_rows") != 8
                    or not isinstance(evidence.get("staged_artifact_files"), Mapping)
                    or not isinstance(evidence.get("candidate_evaluation_root"), str)):
                raise ContractError("training_evidence", "explicit interrupted origin and separate completed evaluation required")
            for key in ("training_receipt_sha256", "candidate_evaluation_manifest_sha256",
                        "candidate_evaluation_summary_sha256"):
                digest(evidence.get(key), key)
        self._value, self.sha256, self.context = payload, declared, context
        self.part_ids = tuple(part["id"] for part in payload["profile"]["parts"])
        self.network = NumpyGRU(payload["weights"])
        if self.network.policy_sha256 != payload["policy_sha256"]:
            raise ContractError("policy_hash", "learner policy content differs from numeric weights")

    @property
    def value(self):
        return strict_json_loads(canonical_json(self._value))

    @classmethod
    def load(cls, path: str | Path, context: RuntimeContext, *, expected_sha256: str):
        location = Path(path)
        if location.stat().st_size > 2 * 1024 * 1024:
            raise ContractError("bundle_size", "numeric CPU bundle limit is2MiB")
        data = location.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ContractError("artifact_hash", "external bundle receipt differs")
        return cls(strict_json_loads(data), context)


@dataclass(frozen=True)
class SourceStamp:
    """Owned source identity carried through inference without refreshing age."""
    observation_id: str
    run_id: str
    epoch: int
    actor_id: int
    control_generation: int
    tick: int
    view_sha256: str


def _source_stamp(observation: Mapping[str, Any]) -> SourceStamp:
    fields = ("observation_id", "run_id", "epoch", "actor_id", "control_generation", "tick")
    local = {"policy": policy_view(observation), "scope": observation.get("observation_scope"),
             "mount_generation": observation.get("mount_generation"),
             "parts": [{key: part.get(key) for key in ("id", "index", "detached")}
                       for part in observation.get("parts", ())]}
    return SourceStamp(*(observation[key] for key in fields), sha256_json(local))


@dataclass(frozen=True)
class ExportDecision:
    action: WireAction
    status: str
    reason: str
    cancel_owner: bool
    features: tuple[float, ...] = ()
    hidden_before: tuple[float, ...] = ()
    hidden_after: tuple[float, ...] = ()
    logits: tuple[float, ...] = ()
    value: float | None = None
    source: SourceStamp | None = None
    helper_lease_ticks: int = 0
    goal_deadline_tick: int | None = None
    checked_tick: int | None = None


class ExportController:
    """Per-actor recurrence and explicit neutral/owner-cancel fallback."""
    def __init__(self, bundle: RuntimeBundle):
        self.bundle = bundle
        self.pending: ExportDecision | None = None
        self.owner_cleanup_unconfirmed = False
        self._owner_may_be_active = False
        self._owner_source: SourceStamp | None = None
        self._owner_boundary = None
        self._owner_goal: SkillGoal | None = None
        self._initialize_lifetime()

    def _initialize_lifetime(self):
        self.hidden = np.zeros(32, dtype=np.float32)
        self.previous = WireAction(effort=0)
        self.lifetime = None
        self.boundary = None
        self.goal = None
        self.last_tick = None
        self.highest_observed_tick = None
        self.highest_control_generation = None
        self.highest_source_generation = None
        self.start_tick = 0
        self.retired = set()
        self.retired_lifetimes = set()
        self._lifetime_key = None
        self._last_source = None

    def reset(self):
        """Reset local memory without claiming that the native owner was reset."""
        return self.cancel("controller_reset")

    def cancel(self, reason="cancelled"):
        if self.goal is not None:
            self.retired.add(self.goal.id)
        self.hidden.fill(0)
        self.previous = WireAction(effort=0)
        self.last_tick = None
        self.pending = None
        self.owner_cleanup_unconfirmed |= self._owner_may_be_active
        return ExportDecision(self.previous, "fallback", reason, True)

    def _view(self, observation):
        view = FeedbackObservation.from_host(policy_view(observation))
        expected_id = ":".join(str(observation[k]) for k in
                               ("run_id", "epoch", "actor_id", "control_generation", "tick"))
        if (view.actor != "jev" or observation.get("observation_scope") != SENSORS["scope"]
                or observation["observation_id"] != expected_id):
            raise ContractError("observation_identity", "source view identity or perception scope differs")
        parts = observation.get("parts", ())
        if (not isinstance(parts, (list, tuple)) or any(not isinstance(part, Mapping) for part in parts)
                or tuple(part.get("id") for part in parts) != self.bundle.part_ids
                or any(type(part.get("index")) is not int or part["index"] != i
                       or type(part.get("detached")) is not bool for i, part in enumerate(parts))):
            raise ContractError("part_identity", "part graph differs from bundle")
        return view

    def _clear_owner(self):
        self.owner_cleanup_unconfirmed = False
        self._owner_may_be_active = False
        self._owner_source = None
        self._owner_boundary = None
        self._owner_goal = None

    def confirm_owner_cancel(self, observation: Mapping[str, Any]):
        """Require the native revocation observation before clearing uncertainty."""
        try:
            view = self._view(observation)
            previous = self._owner_source
            if previous is not None and (
                    (observation["run_id"], observation["epoch"], observation["actor_id"])
                    != (previous.run_id, previous.epoch, previous.actor_id)
                    or view.tick < previous.tick or view.control_generation <= previous.control_generation):
                raise ContractError("owner_cancel_receipt", "the dispatched owner has not demonstrably been revoked")
            if (self.lifetime is not None and view.lifetime != self.lifetime
                    or self.highest_observed_tick is not None and view.tick < self.highest_observed_tick
                    or view.control_generation < 1
                    or self.highest_control_generation is not None
                    and view.control_generation <= self.highest_control_generation):
                raise ContractError("owner_cancel_receipt", "cancellation cannot rewind or retain observed authority")
        except (ValueError, TypeError, KeyError) as exc:
            raise ContractError("owner_cancel_receipt", "a complete native revocation observation is required") from exc
        self.cancel("owner_cancel_confirmed")
        self.lifetime = view.lifetime
        self._lifetime_key = (observation["run_id"], observation["epoch"], observation["actor_id"])
        self._last_source = _source_stamp(observation)
        self.highest_observed_tick = view.tick
        self.highest_control_generation = view.control_generation
        self.boundary = None
        self._clear_owner()

    def step(self, observation: Mapping[str, Any], goal: SkillGoal, *,
             source_generation: int = 1, current_tick: int | None = None,
             context: RuntimeContext | None = None) -> ExportDecision:
        try:
            if not isinstance(goal, SkillGoal):
                raise ContractError("goal_schema", "typed explicit goal required")
            if context is not None and context != self.bundle.context:
                raise ContractError("runtime_context_mismatch", "active deployment context changed")
            view = self._view(observation)
            if view.actor != "jev" or not view.valid or not view.controllable:
                raise ContractError("actor_unavailable", "invalid, suspended or foreign actor")
            now = view.tick if current_tick is None else current_tick
            if type(now) is not int or not 0 <= now-view.tick < 12:
                raise ContractError("source_expired", "original observation's twelve-tick lease expired")
            if type(source_generation) is not int or source_generation < 1:
                raise ContractError("source_generation", "positive explicit source grant required")
            previous_owner = self._owner_source or self._last_source
            new_world_reset = (previous_owner is not None and view.tick == 0 and view.control_generation == 0
                               and (observation["run_id"] != previous_owner.run_id
                                    or observation["epoch"] > previous_owner.epoch))
            if self.owner_cleanup_unconfirmed and not new_world_reset:
                raise ContractError("owner_cleanup_unconfirmed", "confirmed native cancellation or a new host reset is required")
            if self.lifetime != view.lifetime:
                if view.lifetime in self.retired_lifetimes:
                    raise ContractError("retired_actor", "old actor incarnation cannot resume")
                if (self._lifetime_key is not None and observation["run_id"] == self._lifetime_key[0]
                        and observation["epoch"] < self._lifetime_key[1]):
                    raise ContractError("retired_actor", "a run's actor epoch cannot move backward")
                retired = set(self.retired_lifetimes)
                if self.lifetime is not None:
                    retired.add(self.lifetime)
                if len(retired) > 1024:
                    raise ContractError("lifetime_capacity", "explicit controller reconstruction required")
                if self._owner_may_be_active and not new_world_reset:
                    raise ContractError("owner_boundary_requires_cancel", "a different actor cannot clear a live owner")
                self.pending = None
                self._initialize_lifetime()
                self.retired_lifetimes = retired
                self.lifetime = view.lifetime
                self._lifetime_key = (observation["run_id"], observation["epoch"], observation["actor_id"])
                if new_world_reset:
                    self._clear_owner()
            if self.highest_observed_tick is not None and view.tick < self.highest_observed_tick:
                raise ContractError("source_reordered", "older observation cannot start a fresh invocation")
            if self.highest_control_generation is not None and view.control_generation < self.highest_control_generation:
                raise ContractError("retired_control_generation", "owner authority cannot move backward")
            if self.highest_source_generation is not None and source_generation < self.highest_source_generation:
                raise ContractError("retired_source_generation", "source authority cannot move backward")
            # These actor-lifetime fences survive cancellation. Clearing the
            # recurrence/cadence cursor must not make an older grant or view
            # eligible merely because its caller supplies a new goal ID.
            self.highest_observed_tick = view.tick
            self.highest_control_generation = view.control_generation
            self.highest_source_generation = source_generation
            self._last_source = _source_stamp(observation)
            if self.goal is not None and goal.id == self.goal.id and goal != self.goal:
                raise ContractError("goal_collision", "same invocation cannot change parameters")
            if goal.id in self.retired:
                raise ContractError("retired_goal", "fresh goal identity required")
            parts = observation.get("parts", ())
            boundary = (view.control_generation, observation.get("mount_generation"), source_generation,
                        tuple((part["id"], part["index"], part["detached"]) for part in parts))
            if self._owner_may_be_active and (boundary != self._owner_boundary or goal != self._owner_goal):
                reason = "owner_or_mount_changed" if boundary != self._owner_boundary else "owner_boundary_requires_cancel"
                raise ContractError(reason, "revoke the live helper before changing invocation or source")
            if self.boundary is not None and boundary != self.boundary:
                old_goal = self.goal
                self.cancel("owner_or_mount_changed")
                self.boundary = boundary
                if old_goal is not None and old_goal.id == goal.id:
                    self.last_tick = view.tick
                    raise ContractError("owner_or_mount_changed", "old invocation retired")
            self.boundary = boundary
            if self.last_tick is not None and view.tick-self.last_tick != 6:
                raise ContractError("decision_cadence", "fresh completed six-tick observation required")
            if self.goal is None or self.goal.id != goal.id:
                if len(self.retired) >= 1024:
                    raise ContractError("goal_capacity", "fresh episode required")
                if self.goal is not None:
                    self.retired.add(self.goal.id)
                self.goal = goal
                self.hidden.fill(0)
                self.previous = WireAction(effort=0)
                self.start_tick = view.tick
            deadline = self.start_tick + round(goal.timeout_seconds*60)
            if now >= deadline:
                raise ContractError("goal_deadline", "skill duration elapsed")
            encoded = encode(observation, goal, self.previous, self.start_tick)
            before = tuple(float(v) for v in self.hidden)
            outputs, value, following, logits = self.bundle.network.forward(encoded, self.hidden)
            action = map_action(outputs, observation, goal)
            self.hidden, self.previous, self.last_tick = following, action, view.tick
            self.pending = ExportDecision(action, "accepted", "numeric_policy_request", False, encoded, before,
                                          tuple(float(v) for v in following), tuple(float(v) for v in logits), value,
                                          _source_stamp(observation), min(12, deadline - view.tick), deadline, now)
            return self.pending
        except (ContractError, ValueError, TypeError, KeyError, FloatingPointError) as exc:
            reason = exc.code if isinstance(exc, ContractError) else "invalid_runtime_input"
            return self.cancel(reason)


def _same_action(actual, expected, *, tolerance=2e-5):
    """Compare the native FP32 readback without re-normalizing its unit aim."""
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        return False
    for key, wanted in expected.items():
        value = actual[key]
        if type(wanted) is bool:
            if value is not wanted:
                return False
        else:
            if isinstance(wanted, (tuple, list)):
                if not isinstance(value, (tuple, list)) or len(value) != len(wanted):
                    return False
                pairs = zip(value, wanted)
            else:
                pairs = ((value, wanted),)
            if any(type(a) not in (int, float) or not math.isfinite(a) or abs(a - b) > tolerance for a, b in pairs):
                return False
    return True


def _dispatch_receipt(controller, decision, packet, transition, repeat_ticks):
    """A returned RPC must still prove the original request was consumed."""
    if not isinstance(transition, (tuple, list)) or len(transition) != 5:
        raise ContractError("dispatch_receipt", "complete native transition required")
    after, _, terminated, truncated, info = transition
    if not isinstance(after, Mapping) or not isinstance(info, Mapping):
        raise ContractError("dispatch_receipt", "native observation and trace required")
    view = controller._view(after)
    source = decision.source
    terminal = terminated or truncated
    actual = info.get("actual_ticks")
    traces = info.get("trace")
    if (type(terminated) is not bool or type(truncated) is not bool or type(actual) is not int
            or not 1 <= actual <= repeat_ticks or (not terminal and actual != repeat_ticks)
            or not isinstance(traces, list) or len(traces) != actual
            or info.get("consumed_trace_complete") is not True or info.get("idle_trace_dropped") != 0
            or not isinstance(info.get("idle_trace"), list) or info.get("unconsumed_terminal_commands") != []
            or not view.valid or view.controllable != (not terminal)
            or (after["run_id"], after["epoch"], after["actor_id"]) != (source.run_id, source.epoch, source.actor_id)
            or view.tick < source.tick + actual or view.tick > decision.goal_deadline_tick
            or view.tick > source.tick + decision.helper_lease_ticks
            or view.control_generation != source.control_generation + int(terminal)):
        raise ContractError("dispatch_receipt", "complete original-owner interval required")
    boundary = controller._owner_boundary
    if (after.get("mount_generation") != boundary[1]
            or tuple((part["id"], part["index"], part["detached"]) for part in after["parts"]) != boundary[3]):
        raise ContractError("dispatch_boundary", "native result changed the issued part or mount boundary")
    start = view.tick - actual
    if start >= source.tick + decision.helper_lease_ticks:
        raise ContractError("dispatch_receipt", "expired input cannot count as a consumed decision")
    for offset, row in enumerate(traces):
        if (not isinstance(row, Mapping) or row.get("tick_before") != start + offset
                or row.get("tick_after") != start + offset + 1 or row.get("failed") is not False):
            raise ContractError("dispatch_receipt", "causal native tick receipts required")
        if offset == 0:
            admissions, requests = row.get("admission"), row.get("requested")
            if (not isinstance(admissions, list) or len(admissions) != 1
                    or not isinstance(admissions[0], Mapping) or admissions[0].get("accepted") is not True
                    or not isinstance(requests, list) or len(requests) != 1
                    or not isinstance(requests[0], Mapping) or set(requests[0]) != set(packet)
                    or any(requests[0][key] != value for key, value in packet.items() if key != "action")
                    or not _same_action(requests[0]["action"], packet["action"], tolerance=1e-12)):
                raise ContractError("dispatch_refused", "the native owner must admit the original packet")
        elif row.get("admission") != [] or row.get("requested") != []:
            raise ContractError("dispatch_refused", "mixed or replacement input cannot confirm a decision")
        applied = row.get("applied", {})
        if not isinstance(applied, Mapping):
            raise ContractError("dispatch_receipt", "native input readback required")
        final_revocation = terminal and offset == actual - 1
        if final_revocation:
            valid = (applied.get("action_available") is False and applied.get("terminal_revocation") is True
                     and applied.get("action") is None)
        else:
            valid = (applied.get("action_available") is True and applied.get("terminal_revocation") is False
                     and _same_action(applied.get("action"), packet["action"]))
        if not valid:
            raise ContractError("dispatch_consumption", "native readback differs from the admitted request")
    controller.highest_observed_tick = view.tick
    controller.highest_control_generation = view.control_generation
    if terminal or view.tick >= decision.goal_deadline_tick:
        controller.cancel("native_terminal" if terminal else "goal_deadline")


def dispatch(env, controller: ExportController, decision: ExportDecision,
             source_observation: Mapping[str, Any], *, current_tick: int | None = None):
    """Consume one controller-owned decision, or confirm actual owner cancellation.

    The original source identity and absolute deadline are never refreshed.
    The native owner still decides admission if the host advances during an RPC.
    """
    if not isinstance(controller, ExportController) or not isinstance(decision, ExportDecision):
        raise ContractError("dispatch_controller", "typed controller and its decision are required")
    if decision.cancel_owner:
        controller.cancel(decision.reason)
        controller.owner_cleanup_unconfirmed = True
        receipt = env.cancel()
        controller.confirm_owner_cancel(receipt)
        return receipt
    try:
        if (controller.owner_cleanup_unconfirmed or controller.pending is not decision
                or decision.status != "accepted" or decision.source is None
                or decision.source != _source_stamp(source_observation)):
            raise ContractError("decision_source_mismatch", "only this controller's current unused decision may dispatch")
        now = decision.checked_tick if current_tick is None else current_tick
        if (type(now) is not int or now < decision.checked_tick
                or now >= decision.goal_deadline_tick or now >= decision.source.tick + decision.helper_lease_ticks):
            raise ContractError("dispatch_deadline", "the original goal or source lease has expired")
        repeat = min(6, decision.goal_deadline_tick - now)
        if now + repeat > decision.source.tick + decision.helper_lease_ticks:
            raise ContractError("dispatch_deadline", "the original source lease cannot cover this action repeat")
        controller.pending = None
        controller._owner_may_be_active = True
        controller._owner_source = decision.source
        controller._owner_boundary = controller.boundary
        controller._owner_goal = controller.goal
        packet = env.command(decision.action.to_dict(), observation=source_observation,
                             lease_ticks=decision.helper_lease_ticks)
        result = env.step_packet(packet, ticks=repeat)
        _dispatch_receipt(controller, decision, packet, result, repeat)
        return result
    except BaseException:
        controller.cancel("dispatch_failed")
        controller.owner_cleanup_unconfirmed = True
        try:
            receipt = env.cancel()
            controller.confirm_owner_cancel(receipt)
        except Exception:
            pass
        raise
