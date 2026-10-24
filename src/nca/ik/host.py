"""Strict NCA adapter for Cybersand physical_joint_target/v1.

This module does not solve IK and does not own physics.  It validates the
capability advertised by the Cybersand owner, binds one controller grant and
one issued observation to each joint-reference dispatch, and verifies the
per-base-tick receipts returned by the actual Godot/Rapier host.

The historical ACT003/ACT004 adapter and its accepted host pin remain separate.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from nca.adapters.protocol import (
    HostError, JsonChannel, ProtocolError, PROTOCOL, integer, owned_json,
)
from nca.contracts import sha256_json
from .profiles import runner_right_sole_profile

CAPABILITY = "physical_joint_target/v1"
PROFILE_ID = "runner-right-sole@1"
PROFILE_SHA256 = "6cb15832e240f82f4bcd582d2c26015ef986e44f335da683abaf9746e31503c4"
DESCRIPTOR_SHA256 = "76fd5e0a481e628f30fe2fb55d62eb5cc66014be07c13a5b7720d30e792d2c65"
DESCRIPTOR_SOURCE = "f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497"
TOPOLOGY_SHA256 = "41834423020c4dda28c425e695124dc743419b90b05426efc3f087238df71b32"
PART_IDS = ("chassis", "leg_-1", "foot_-1", "leg_1", "foot_1")
JOINT_IDS = ("hip_-1", "ankle_-1", "hip_1", "ankle_1")
ACTIVE_JOINT_IDS = ("hip_1", "ankle_1")
PART_PRESENT = (True, True, True, True, True, False, False, False)
JOINT_PRESENT = (True, True, True, True, False, False, False)
JOINT_ACTIVE = (False, False, True, True, False, False, False)

# Candidate accepted-host pin.  This is intentionally independent of
# nca.adapters.godot.PINNED_HOST_REVISION, which preserves historical evidence.
PINNED_JOINT_HOST_REVISION = "eb5051cc16f5efa20ba2e62514bf5ebd7edca968"
PINNED_JOINT_HOST_TREE = "d77f77a98e2c71dc76b4228cb27463430cd13291"
PINNED_GODOT_VERSION = "4.7.stable.official.5b4e0cb0f"
PINNED_GODOT_LINUX = "f85bbc6b15e22416c7d797cd60b63286dd67b9cb13498847056c18520ae55a75"
PINNED_NATIVE_LINUX = "3bfe1679c6d0201203fcf99b6f53d7f50875a530611776787131d7dc1d5098c2"
PINNED_RAPIER_LINUX = "566e68ae53bfb2eb7018c8e7d0dd3727121bf696462039b780a33d6b8369c66a"

_MAX_INTEGER = 9_007_199_254_740_991
_CAPABILITY_KEYS = {
    "protocol", "operation", "version", "capability", "mode",
    "host_source_commit", "host_runtime_sha256", "profile_id", "profile_sha256",
    "descriptor_file_sha256", "descriptor_source_commit", "topology_sha256",
    "effector", "part_ids", "joint_ids", "active_joint_ids", "part_capacity",
    "joint_capacity", "angle_unit", "length_unit", "target_mode",
    "target_limits_radians", "max_delta_radians", "base_hz", "decision_ticks",
    "max_observation_age_ticks", "max_lease_ticks", "helper_lease_ceiling_ticks",
    "deadline_exclusive", "source_grant_required", "non_commanded",
    "wire_buttons_mounts", "authored_gyro", "motor", "force_telemetry",
    "passive_force_decomposition", "receipts", "packet_kind",
    "observation_schema", "actor_id", "run_id", "epoch", "incarnation",
    "control_generation", "source_owner",
}
_GRANT_KEYS = {
    "protocol", "operation", "version", "run_id", "actor_id", "epoch",
    "incarnation", "control_generation", "source_id", "policy_sha256", "goal_id",
    "goal_deadline_tick", "grant_id", "grant_generation", "producer",
    "profile_sha256", "topology_sha256", "joint_ids",
}
_OBSERVATION_KEYS = {
    "protocol", "operation", "schema", "version", "run_id", "actor_id", "epoch",
    "incarnation", "control_generation", "grant_id", "source_id", "goal_id",
    "observed_tick", "observation_id", "observation_sha256", "profile_sha256",
    "topology_sha256", "root_pose", "part_ids", "joint_ids", "part_present",
    "joint_present", "joint_measured", "joint_active", "joint_positions_radians",
    "measured_chain_radians", "evidence_kind",
}
_SUBMIT_KEYS = {
    "protocol", "operation", "version", "run_id", "actor_id", "epoch",
    "incarnation", "control_generation", "source_id", "policy_sha256", "grant_id",
    "grant_generation", "goal_id", "goal_deadline_tick", "sequence",
    "observed_tick", "observation_id", "observation_sha256", "profile_sha256",
    "topology_sha256", "effector_id", "joint_ids", "joint_values",
    "target_mode", "lease_ticks", "producer",
}


class JointHostTransport(Protocol):
    """One ordered request/response transport owned outside this adapter."""

    def request(self, operation: str, **fields: Any) -> Mapping[str, Any]:
        ...


def _fail(message: str) -> None:
    raise ProtocolError(message)


def _exact(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object")
    actual = set(value)
    if actual != keys:
        _fail(f"{name} fields differ: missing={sorted(keys - actual)} extra={sorted(actual - keys)}")
    return value


def _digest(value: Any, length: int = 64, name: str = "digest") -> str:
    if (type(value) is not str or len(value) != length
            or any(ch not in "0123456789abcdef" for ch in value)):
        _fail(f"{name} is not a lowercase {length}-hex digest")
    return value


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or not value or len(value) > 128:
        _fail(f"{name} is not a bounded identifier")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:@")
    if any(ch not in allowed for ch in value):
        _fail(f"{name} contains an unsupported character")
    return value


def _int(value: Any, minimum: int = 0, maximum: int = _MAX_INTEGER, name: str = "integer") -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{name} is outside [{minimum},{maximum}]")
    return value


def _finite(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        _fail(f"{name} must be numeric, not boolean")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        _fail(f"{name} is nonfinite")
    return result


def _sequence(value: Any, expected: Sequence[Any], name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or tuple(value) != tuple(expected):
        _fail(f"{name} differs")
    return tuple(value)


def validate_capability(value: Mapping[str, Any], identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cap = _exact(value, _CAPABILITY_KEYS, "joint capability")
    expected_scalars = {
        "protocol": PROTOCOL, "operation": "ik_capabilities", "version": 1,
        "capability": CAPABILITY, "mode": "exclusive_joint_targets",
        "profile_id": PROFILE_ID, "profile_sha256": PROFILE_SHA256,
        "descriptor_file_sha256": DESCRIPTOR_SHA256,
        "descriptor_source_commit": DESCRIPTOR_SOURCE,
        "topology_sha256": TOPOLOGY_SHA256, "part_capacity": 8, "joint_capacity": 7,
        "angle_unit": "radians", "length_unit": "game_cells", "max_delta_radians": 0.25,
        "base_hz": 60, "decision_ticks": 6, "max_observation_age_ticks": 12,
        "max_lease_ticks": 18, "helper_lease_ceiling_ticks": 12,
        "deadline_exclusive": True, "source_grant_required": True,
        "non_commanded": "passive_constraints", "wire_buttons_mounts": "disabled_disarmed",
        "force_telemetry": "unavailable", "passive_force_decomposition": "unavailable",
        "packet_kind": "ik_submit", "observation_schema": "ik_host_observation/v1",
        "source_owner": "policy",
    }
    for key, expected in expected_scalars.items():
        if type(cap.get(key)) is not type(expected) or cap.get(key) != expected:
            _fail(f"joint capability {key} differs")
    _digest(cap["host_source_commit"], 40, "host_source_commit")
    _digest(cap["host_runtime_sha256"], 64, "host_runtime_sha256")
    if identity is not None:
        if cap["host_source_commit"] != identity.get("host_revision"):
            _fail("joint capability source revision differs from inspected checkout")
        if cap["host_runtime_sha256"] != identity.get("runtime_sha256"):
            _fail("joint capability runtime identity differs from inspected runtime")
    _identifier(cap["run_id"], "capability.run_id")
    _int(cap["actor_id"], 0, 7, "capability.actor_id")
    for key in ("epoch", "incarnation", "control_generation"):
        _int(cap[key], 0, _MAX_INTEGER, "capability." + key)
    _sequence(cap["part_ids"], PART_IDS, "capability.part_ids")
    _sequence(cap["joint_ids"], JOINT_IDS, "capability.joint_ids")
    _sequence(cap["active_joint_ids"], ACTIVE_JOINT_IDS, "capability.active_joint_ids")
    _sequence(cap["target_mode"], ("joint_position", "joint_delta"), "capability.target_mode")
    _sequence(cap["receipts"], ("requested", "admitted", "limited", "motor", "motion", "integrity"),
              "capability.receipts")
    if cap["target_limits_radians"] != [[-0.72, 0.72], [-0.27, 0.27]]:
        _fail("joint capability target limits differ")
    effector = _exact(cap["effector"], {"id", "part_id", "point_local_cells"}, "capability.effector")
    if effector != {"id": "right_sole", "part_id": "foot_1", "point_local_cells": [0.0, 1.0]}:
        _fail("joint capability effector differs")
    gyro = _exact(cap["authored_gyro"], {
        "neutral_target_radians", "gain_grounded", "gain_airborne", "gyro", "gyro_kp",
        "damping", "play_radians", "grounded_slope_lean", "button_boost"}, "capability.authored_gyro")
    expected_gyro = {"neutral_target_radians": 0.0, "gain_grounded": 1.0, "gain_airborne": 0.65,
                     "gyro": 26.0, "gyro_kp": 10.0, "damping": 3.0, "play_radians": 0.055,
                     "grounded_slope_lean": True, "button_boost": False}
    if dict(gyro) != expected_gyro:
        _fail("joint capability authored gyro differs")
    motor = _exact(cap["motor"], {"effort", "speed_cap_radians_per_second",
                                  "request_slew_radians_per_second_squared",
                                  "pin_joint_max_force_parameter",
                                  "max_force_is_measured_torque"}, "capability.motor")
    expected_motor = {"effort": 1.0, "speed_cap_radians_per_second": 7.0,
                      "request_slew_radians_per_second_squared": 20.0,
                      "pin_joint_max_force_parameter": 100.0,
                      "max_force_is_measured_torque": False}
    if dict(motor) != expected_motor:
        _fail("joint capability motor limits differ")
    # The immutable NCA profile must still describe the same accepted semantics.
    profile = runner_right_sole_profile()
    if profile.id + "@" + str(profile.version) != PROFILE_ID or profile.sha256 != PROFILE_SHA256:
        _fail("local immutable IK profile no longer matches the host capability")
    return owned_json(dict(cap))


def validate_grant(value: Mapping[str, Any], capability: Mapping[str, Any]) -> dict[str, Any]:
    grant = _exact(value, _GRANT_KEYS, "joint grant")
    if grant["protocol"] != PROTOCOL or grant["operation"] != "ik_grant" or grant["version"] != 1:
        _fail("joint grant protocol differs")
    for key in ("run_id", "source_id", "goal_id", "grant_id", "producer"):
        _identifier(grant[key], "grant." + key)
    _digest(grant["policy_sha256"], name="grant.policy_sha256")
    for key in ("actor_id", "epoch", "incarnation", "control_generation", "grant_generation"):
        _int(grant[key], 0, _MAX_INTEGER, "grant." + key)
    _int(grant["goal_deadline_tick"], 1, _MAX_INTEGER, "grant.goal_deadline_tick")
    for key in ("run_id", "actor_id", "epoch", "incarnation", "profile_sha256", "topology_sha256"):
        if grant[key] != capability[key]:
            _fail("joint grant differs from negotiated capability: " + key)
    if grant["control_generation"] <= capability["control_generation"]:
        _fail("joint grant did not establish a new owner generation")
    if grant["grant_generation"] != grant["control_generation"]:
        _fail("joint grant generation is not the owner generation")
    if grant["producer"] != capability["source_owner"]:
        _fail("joint grant producer differs")
    if tuple(grant["joint_ids"]) != ACTIVE_JOINT_IDS:
        _fail("joint grant active joints differ")
    return owned_json(dict(grant))


def validate_observation(value: Mapping[str, Any], grant: Mapping[str, Any]) -> dict[str, Any]:
    view = _exact(value, _OBSERVATION_KEYS, "joint observation")
    if (view["protocol"] != PROTOCOL or view["operation"] != "ik_observe"
            or view["schema"] != "ik_host_observation/v1" or view["version"] != 1
            or view["evidence_kind"] != "host_observation"):
        _fail("joint observation protocol differs")
    for key in ("run_id", "grant_id", "source_id", "goal_id", "observation_id"):
        _identifier(view[key], "observation." + key)
    _digest(view["observation_sha256"], name="observation.observation_sha256")
    for key in ("actor_id", "epoch", "incarnation", "control_generation", "observed_tick"):
        _int(view[key], 0, _MAX_INTEGER, "observation." + key)
    for key in ("run_id", "actor_id", "epoch", "incarnation", "control_generation",
                "grant_id", "source_id", "goal_id", "profile_sha256", "topology_sha256"):
        if view[key] != grant[key]:
            _fail("joint observation differs from current grant: " + key)
    _sequence(view["part_ids"], PART_IDS, "observation.part_ids")
    _sequence(view["joint_ids"], JOINT_IDS, "observation.joint_ids")
    _sequence(view["part_present"], PART_PRESENT, "observation.part_present")
    _sequence(view["joint_present"], JOINT_PRESENT, "observation.joint_present")
    _sequence(view["joint_measured"], JOINT_PRESENT, "observation.joint_measured")
    _sequence(view["joint_active"], JOINT_ACTIVE, "observation.joint_active")
    positions = view["joint_positions_radians"]
    if not isinstance(positions, (list, tuple)) or len(positions) != 7:
        _fail("joint observation position shape differs")
    for index, item in enumerate(positions):
        if index < 4:
            _finite(item, f"observation.joint_positions_radians[{index}]")
        elif item is not None:
            _fail("absent joint slot unexpectedly has a measurement")
    chain = view["measured_chain_radians"]
    if not isinstance(chain, (list, tuple)) or len(chain) != 2:
        _fail("joint observation measured chain differs")
    if any(abs(_finite(chain[i], f"observation.measured_chain_radians[{i}]") - float(positions[i + 2])) > 1e-12
           for i in range(2)):
        _fail("joint observation chain does not match active slots")
    root = _exact(view["root_pose"], {"position_cells", "angle_radians"}, "observation.root_pose")
    if not isinstance(root["position_cells"], (list, tuple)) or len(root["position_cells"]) != 2:
        _fail("joint observation root position shape differs")
    _finite(root["position_cells"][0], "root.x")
    _finite(root["position_cells"][1], "root.y")
    _finite(root["angle_radians"], "root.angle")
    return owned_json(dict(view))


def make_submit(grant: Mapping[str, Any], observation: Mapping[str, Any], sequence: int,
                values: Sequence[float], *, mode: str = "joint_position",
                lease_ticks: int = 12) -> dict[str, Any]:
    sequence = _int(sequence, 0, _MAX_INTEGER, "sequence")
    lease_ticks = _int(lease_ticks, 1, 18, "lease_ticks")
    if lease_ticks > 12:
        # NCA helper dispatch is intentionally tighter than the host's 1-18 envelope.
        _fail("NCA helper lease exceeds its 12-tick ceiling")
    if mode not in {"joint_position", "joint_delta"}:
        _fail("unsupported joint target mode")
    if len(values) != 2:
        _fail("joint reference requires exactly two values")
    limits = (0.72, 0.27) if mode == "joint_position" else (0.25, 0.25)
    normalized = []
    for index, value in enumerate(values):
        number = _finite(value, f"joint_values[{index}]")
        if abs(number) > limits[index]:
            _fail("joint reference exceeds negotiated limits")
        normalized.append(number)
    packet = {
        "protocol": PROTOCOL, "operation": "ik_submit", "version": 1,
        "run_id": observation["run_id"], "actor_id": observation["actor_id"],
        "epoch": observation["epoch"], "incarnation": observation["incarnation"],
        "control_generation": observation["control_generation"],
        "source_id": grant["source_id"], "policy_sha256": grant["policy_sha256"],
        "grant_id": grant["grant_id"], "grant_generation": grant["grant_generation"],
        "goal_id": grant["goal_id"], "goal_deadline_tick": grant["goal_deadline_tick"],
        "sequence": sequence, "observed_tick": observation["observed_tick"],
        "observation_id": observation["observation_id"],
        "observation_sha256": observation["observation_sha256"],
        "profile_sha256": observation["profile_sha256"],
        "topology_sha256": observation["topology_sha256"],
        "effector_id": "right_sole", "joint_ids": list(ACTIVE_JOINT_IDS),
        "joint_values": normalized, "target_mode": mode,
        "lease_ticks": lease_ticks, "producer": grant["producer"],
    }
    if set(packet) != _SUBMIT_KEYS:
        raise AssertionError("internal joint packet schema drift")
    return packet


def validate_admission(value: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("protocol") != PROTOCOL or value.get("operation") != "ik_submit":
        _fail("joint admission protocol differs")
    if value.get("accepted") is not True:
        _fail("joint target was not admitted: " + str(value.get("reason", "refused")))
    if value.get("requested") != packet:
        _fail("joint admission does not retain the exact request")
    admitted = value.get("admitted")
    if not isinstance(admitted, Mapping) or admitted.get("mode") != "exclusive_joint_targets":
        _fail("joint admission owner mode differs")
    if tuple(admitted.get("joint_ids", ())) != ACTIVE_JOINT_IDS:
        _fail("joint admission joint set differs")
    refs = admitted.get("references_radians")
    if not isinstance(refs, (list, tuple)) or len(refs) != 2:
        _fail("joint admission references are missing")
    for i, item in enumerate(refs):
        _finite(item, f"admission.references[{i}]")
    integrity = value.get("integrity")
    if not isinstance(integrity, Mapping):
        _fail("joint admission integrity is missing")
    if (integrity.get("original_observed_tick") != packet["observed_tick"]
            or integrity.get("terminal_boundary") != "not_consumed"
            or integrity.get("missing_receipt_count") != 0
            or integrity.get("unknown_receipt_count") != 0):
        _fail("joint admission integrity differs")
    return owned_json(dict(value))


class ReceiptTracker:
    """Reject gaps, reordered ticks, identity changes and unconfirmed terminals."""

    def __init__(self, packet: Mapping[str, Any]) -> None:
        self.packet = owned_json(dict(packet))
        self.last_completed_tick = int(packet["observed_tick"])
        self.seen_steps: set[int] = set()
        self.terminal: str | None = None

    def consume(self, values: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for raw in values:
            if not isinstance(raw, Mapping):
                _fail("joint receipt is not an object")
            row = owned_json(dict(raw))
            if row.get("protocol") != PROTOCOL or row.get("requested") != self.packet:
                _fail("joint receipt identity differs from the dispatched packet")
            if row.get("operation") == "ik_submit":
                # The admission record is not a consumed physics interval.
                if row.get("accepted") is not True:
                    _fail("stored joint admission is not accepted")
                rows.append(row)
                continue
            integrity = row.get("integrity")
            if not isinstance(integrity, Mapping):
                _fail("joint receipt integrity missing")
            if integrity.get("missing_receipt_count") != 0 or integrity.get("unknown_receipt_count") != 0:
                _fail("joint receipt reports missing or unknown physical work")
            if row.get("operation") == "ik_terminal":
                if integrity.get("consumed_base_tick") is not None or integrity.get("cancel_confirmed") is not True:
                    _fail("joint terminal is not an owner-confirmed no-step boundary")
                self.terminal = str(integrity.get("terminal_boundary", ""))
                rows.append(row)
                continue
            if row.get("operation") != "ik_receipt" or row.get("accepted") is not True:
                _fail("unexpected joint receipt operation")
            consumed = _int(integrity.get("consumed_base_tick"), 0, _MAX_INTEGER, "receipt.consumed_base_tick")
            completed = _int(integrity.get("completed_base_tick"), 0, _MAX_INTEGER, "receipt.completed_base_tick")
            if consumed in self.seen_steps:
                rows.append(row)
                continue
            if completed != consumed + 1 or consumed != self.last_completed_tick:
                _fail("joint receipt stream contains an untraced or reordered base tick")
            if integrity.get("terminal_boundary") != "completed":
                _fail("joint physical interval did not complete normally")
            motion, motor = row.get("motion"), row.get("motor")
            if not isinstance(motion, Mapping) or motion.get("available") is not True:
                _fail("joint receipt lacks completed measured motion")
            if motion.get("per_joint_force_telemetry") is not None or motion.get("passive_force_decomposition") is not None:
                _fail("joint receipt fabricated unavailable force telemetry")
            if not isinstance(motor, Mapping) or tuple(motor.get("commanded_joint_ids", ())) != ACTIVE_JOINT_IDS:
                _fail("joint receipt motor owner differs")
            self.seen_steps.add(consumed)
            self.last_completed_tick = completed
            rows.append(row)
        return tuple(rows)


class JointTargetClient:
    """Stateful source/grant/observation adapter over a strict request transport."""

    def __init__(self, transport: JointHostTransport, *,
                 identity: Mapping[str, Any] | None = None) -> None:
        self.transport = transport
        self.identity = None if identity is None else dict(identity)
        self.capability: dict[str, Any] | None = None
        self.grant_record: dict[str, Any] | None = None
        self.observation_record: dict[str, Any] | None = None
        self.next_sequence = 0
        self.trackers: dict[int, ReceiptTracker] = {}
        self.uncertain = False

    def negotiate(self) -> dict[str, Any]:
        response = self.transport.request("ik_capabilities")
        cap = validate_capability(response.get("capability", {}), self.identity)
        self.capability = cap
        return owned_json(cap)

    def grant(self, source_id: str, policy_sha256: str, goal_id: str,
              goal_deadline_tick: int) -> dict[str, Any]:
        if self.capability is None:
            self.negotiate()
        assert self.capability is not None
        response = self.transport.request("ik_grant", source_id=source_id,
                                          policy_sha256=policy_sha256, goal_id=goal_id,
                                          goal_deadline_tick=goal_deadline_tick)
        grant = validate_grant(response.get("grant", {}), self.capability)
        self.grant_record, self.observation_record = grant, None
        self.next_sequence = 0
        self.trackers.clear()
        self.uncertain = False
        return owned_json(grant)

    def observe(self) -> dict[str, Any]:
        if self.grant_record is None:
            raise HostError("joint grant required before observation")
        response = self.transport.request("ik_observe", grant_id=self.grant_record["grant_id"])
        view = validate_observation(response.get("observation", {}), self.grant_record)
        self.observation_record = view
        return owned_json(view)

    def submit(self, values: Sequence[float], *, mode: str = "joint_position",
               lease_ticks: int = 12, observation: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.uncertain:
            raise HostError("joint transport outcome is uncertain; confirmed cancellation required")
        if self.grant_record is None:
            raise HostError("joint grant required before dispatch")
        view = self.observation_record if observation is None else validate_observation(observation, self.grant_record)
        if view is None:
            view = self.observe()
        packet = make_submit(self.grant_record, view, self.next_sequence, values,
                             mode=mode, lease_ticks=lease_ticks)
        try:
            response = self.transport.request("ik_submit", packet=packet)
        except (HostError, OSError):
            self.uncertain = True
            raise
        admission = validate_admission(response.get("admission", {}), packet)
        self.trackers[self.next_sequence] = ReceiptTracker(packet)
        self.next_sequence += 1
        return admission

    def advance(self, ticks: int = 6) -> Mapping[str, Any]:
        integer(ticks, 1, 18, "ticks")
        return self.transport.request("advance", ticks=ticks)

    def receipts(self, sequence: int) -> tuple[dict[str, Any], ...]:
        if self.grant_record is None or sequence not in self.trackers:
            raise HostError("unknown joint dispatch sequence")
        response = self.transport.request("ik_receipts", grant_id=self.grant_record["grant_id"],
                                          sequence=sequence)
        values = response.get("receipts")
        if not isinstance(values, list):
            _fail("joint receipt response is not an array")
        return self.trackers[sequence].consume(values)

    def cancel(self) -> dict[str, Any]:
        if self.grant_record is None:
            raise HostError("no joint grant to cancel")
        response = self.transport.request("ik_cancel", grant_id=self.grant_record["grant_id"],
                                          source_id=self.grant_record["source_id"])
        value = response.get("cancellation")
        if not isinstance(value, Mapping) or value.get("confirmed") is not True:
            raise HostError("joint owner did not confirm cancellation")
        if value.get("momentum_reset") is not False or value.get("old_motor_references_revoked") is not True:
            _fail("joint cancellation semantics differ")
        self.grant_record = None
        self.observation_record = None
        self.uncertain = False
        return owned_json(dict(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_joint_host(checkout: str | Path, godot: str | Path,
                       *, timeout: float | None = 15) -> dict[str, Any]:
    """Authenticate the separately pinned joint-target source and runtime."""
    root = Path(checkout).resolve()

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    try:
        revision, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
        if revision != PINNED_JOINT_HOST_REVISION or tree != PINNED_JOINT_HOST_TREE:
            raise HostError("joint host revision/tree is not the reviewed candidate")
        if git("diff", "--name-only", "HEAD"):
            raise HostError("tracked joint host differs from the pinned revision")
        native_manifest = json.loads((root / "godot/addons/cybersand_native/runtime-provenance.linux.json").read_text())
        rapier_manifest = json.loads((root / "godot/addons/godot-rapier2d/runtime-provenance.linux.json").read_text())
        native_path = root / native_manifest["artifact"]
        rapier_path = root / rapier_manifest["artifact"]
        native_digest, rapier_digest = _sha256_file(native_path), _sha256_file(rapier_path)
        if native_digest != native_manifest["sha256"] or native_digest != PINNED_NATIVE_LINUX:
            raise HostError("joint host native artifact identity differs")
        if rapier_digest != rapier_manifest["sha256"] or rapier_digest != PINNED_RAPIER_LINUX:
            raise HostError("joint host Rapier artifact identity differs")
        executable = Path(shutil.which(str(godot)) or str(godot)).resolve()
        version = subprocess.check_output([str(executable), "--version"], text=True, timeout=timeout).strip()
        godot_digest = _sha256_file(executable)
        if version != PINNED_GODOT_VERSION or godot_digest != PINNED_GODOT_LINUX:
            raise HostError("joint host Godot runtime identity differs")
        scripts = {}
        for name in ("physical_actor_session", "physical_actor_rig", "physical_joint_target",
                     "physical_usable_equipment", "rapier_physics_bridge"):
            path = root / "godot/scripts" / (name + ".gd")
            scripts[str(path.relative_to(root)).replace("\\", "/")] = _sha256_file(path)
        runtime_sha256 = sha256_json({
            "godot_binary_sha256": godot_digest,
            "native_artifact_sha256": native_digest,
            "rapier_artifact_sha256": rapier_digest,
            "host_script_sha256": scripts,
        })
        return {
            "repository": "techrote/cybersand", "host_revision": revision, "host_tree": tree,
            "godot_version": version, "godot_binary_sha256": godot_digest,
            "native_artifact_sha256": native_digest, "rapier_artifact_sha256": rapier_digest,
            "host_script_sha256": scripts, "runtime_sha256": runtime_sha256,
        }
    except (OSError, KeyError, ValueError, subprocess.SubprocessError) as exc:
        raise HostError(f"cannot establish joint-target host identity: {exc}") from exc


class GodotJointHost:
    """Persistent loopback client for the NCA-owned joint-target bridge."""

    def __init__(self, checkout: str | Path, godot: str | Path, *, timeout: float = 30.0) -> None:
        if not 0 < timeout <= 120:
            raise ValueError("host deadline must be in (0,120] seconds")
        self.identity = inspect_joint_host(checkout, godot)
        self._request_id = 0
        self._closed = False
        self._channel: JsonChannel | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._diagnostics: collections.deque[str] = collections.deque(maxlen=100)
        self._temporary = tempfile.TemporaryDirectory(prefix="nca-ik-host-")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(timeout)
            token = os.urandom(32).hex()
            environment = os.environ.copy()
            if os.name != "nt":
                for key, folder in (("XDG_DATA_HOME", "data"), ("XDG_CONFIG_HOME", "config"),
                                    ("XDG_CACHE_HOME", "cache")):
                    path = Path(self._temporary.name) / folder
                    path.mkdir()
                    environment[key] = str(path)
            bridge = Path(__file__).with_name("joint_host_bridge.gd").resolve()
            command = [str(godot), "--headless", "--path", str(Path(checkout).resolve() / "godot"),
                       "--script", str(bridge), "--", f"--port={listener.getsockname()[1]}",
                       f"--token={token}", f"--source-commit={self.identity['host_revision']}",
                       f"--runtime-sha256={self.identity['runtime_sha256']}"]
            self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, env=environment)
            self._reader = threading.Thread(target=self._drain_diagnostics, daemon=True)
            self._reader.start()
            connection, address = listener.accept()
            if address[0] != "127.0.0.1":
                connection.close()
                raise ProtocolError("nonlocal joint host connection")
            connection.settimeout(timeout)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._channel = JsonChannel(connection)
            hello = self._channel.receive()
            if (hello.get("protocol") != PROTOCOL or hello.get("id") != 0 or hello.get("ok") is not True
                    or hello.get("token") != token or hello.get("lane") != "physical_joint_target/v1"
                    or hello.get("host_source_commit") != self.identity["host_revision"]
                    or hello.get("host_runtime_sha256") != self.identity["runtime_sha256"]
                    or hello.get("rapier_available") is not True):
                raise ProtocolError("joint host capability handshake mismatch")
        except BaseException as exc:
            self.close(graceful=False)
            diagnostics = "\n".join(self._diagnostics)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise HostError(f"persistent joint host did not start: {exc}\n{diagnostics}") from exc
        finally:
            listener.close()

    def _drain_diagnostics(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for data in iter(self._process.stdout.readline, b""):
            self._diagnostics.append(data.decode("utf-8", errors="replace").rstrip()[:2000])

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return tuple(self._diagnostics)

    def request(self, operation: str, **fields: Any) -> dict[str, Any]:
        if {"id", "protocol", "op"}.intersection(fields):
            raise ValueError("request fields cannot replace protocol correlation")
        if self._closed or self._channel is None:
            raise HostError("joint host is closed")
        with self._lock:
            self._request_id += 1
            request = {"protocol": PROTOCOL, "id": self._request_id, "op": operation, **fields}
            try:
                self._channel.send(request)
                result = self._channel.receive()
                if result.get("protocol") != PROTOCOL or result.get("id") != self._request_id:
                    raise ProtocolError("joint host response correlation mismatch")
                if type(result.get("ok")) is not bool:
                    raise ProtocolError("joint host response status missing")
            except ProtocolError:
                self.close(graceful=False)
                raise
            if not result["ok"]:
                raise HostError(str(result.get("error", "joint host refused request")))
            return result

    def close(self, *, graceful: bool = True) -> None:
        if self._closed:
            return
        if graceful and self._channel is not None and self._process is not None and self._process.poll() is None:
            try:
                self._channel.connection.settimeout(2.0)
                self.request("close")
            except HostError:
                pass
        self._closed = True
        if self._channel is not None:
            self._channel.connection.close()
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
            if hasattr(self, "_reader"):
                self._reader.join(timeout=2)
            if self._process.stdout is not None:
                self._process.stdout.close()
        self._temporary.cleanup()

    def __enter__(self) -> "GodotJointHost":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
