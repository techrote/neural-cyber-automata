"""Copied own-part sensors, explicit capability maps and recurrence boundaries.

This is an ACT004 adapter, not a plant or a claim of learned transfer. Profile
identity is evidence metadata only. Numeric policy inputs contain no rig names,
part IDs, actor embeddings, reward, evaluator state, scene search or cell patch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, Literal

from nca.contracts import ContractError, Record, WireAction, digest, identifier, sha256_json

MAX_PARTS = 8
MAX_JOINTS = 7
SCHEMA = "act004-morphology-local/v1"
OBSERVATION_SCOPE = "self, rig rays and bounded composed native cells"
ACTION_NAMES = ("drive_x", "drive_y", "trim", "effort", "impulse", "brake", "rebound")
ROLES = ("hull", "foot", "wheel", "thruster", "slider", "other")
NODE_NAMES = (
    "own_vx/120", "own_vy/120", "sin_angle", "cos_angle", "angular_velocity/12",
    "own_goal_dx/96", "own_goal_dy/96", "width/24", "height/24", "mass/5",
    "rest_offset_x/24", "rest_offset_y/24", "sin_rest_angle", "cos_rest_angle",
    *("role_" + role for role in ROLES),
)
EDGE_NAMES = (
    "rotary", "spring", "slide", "anchor_x/24", "anchor_y/24", "axis_x", "axis_y",
    "lower_limit", "upper_limit", "stiffness/100", "damping/10", "max_force/500",
    "motor_speed/20", "rest_length/24", "sin_phase", "cos_phase", "target",
    "sin_relative_angle", "cos_relative_angle", "angular_speed/12", "travel/12", "anchor_error/24",
)
ROOT_SENSORS = (
    ("grounded", "bool", 1.), ("blocked", "bool", 1.),
    ("movement_limited", "bool", 1.), ("power_enabled", "bool", 1.),
    ("flight_mode", "bool", 1.), ("energy", "number", 1.),
    ("cooldown", "number", 1.), ("charge", "number", 1.),
    ("stuck_time", "number", 4.), ("ground_distance", "number", 192.),
    ("ceiling_distance", "number", 192.), ("wall_left", "number", 30.),
    ("wall_right", "number", 30.),
)
MECHANICS = (
    ("speed", 100.), ("accel", 500.), ("gyro", 50.), ("gyro_kp", 20.),
    ("damping", 10.), ("play", .2), ("gravity", 200.), ("thrust", 300.),
    ("jump", 120.), ("slew", 12.), ("air_control", 1.),
)
ROOT_NAMES = (
    *(name for name, _, _ in ROOT_SENSORS), "contact_normal_x", "contact_normal_y",
    *("profile_" + name for name, _ in MECHANICS), *("previous_" + name for name in ACTION_NAMES),
)
COMMAND_NAMES = ("pace", "tolerance/32", "remaining_deadline_fraction", "source_age/12")
CAPABILITIES = {
    "jev": {"movement": "hop", "primary": "pulse", "secondary": "rebound",
            "impulse": "primary", "brake": None, "rebound": "secondary"},
    "astra": {"movement": "flight", "primary": "pulse", "secondary": "brake",
              "impulse": "primary", "brake": "secondary", "rebound": None},
    "monowheel": {"movement": "roll", "primary": "spin", "secondary": "hop",
                  "impulse": "secondary", "brake": None, "rebound": None},
    "ripper": {"movement": "ground", "primary": "cut", "secondary": "brake",
               "impulse": None, "brake": "secondary", "rebound": None},
}


def _number(value: Any, field: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ContractError("morphology_number", f"{field} must be finite, not a boolean")
    return float(value)


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError("morphology_identity", f"{field} must be an integer >= {minimum}")
    return value


def _pair(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ContractError("morphology_shape", f"{field} must have two components")
    return tuple(_number(item, field) for item in value)


def _boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ContractError("morphology_boolean", f"{field} must be a boolean")
    return value


def _clip(value: float) -> float:
    return max(-2., min(2., value))


def _local(vector: tuple[float, float], angle: float) -> tuple[float, float]:
    x, y = vector
    return (math.cos(angle) * x + math.sin(angle) * y,
            -math.sin(angle) * x + math.cos(angle) * y)


@dataclass(frozen=True)
class MorphologyGoal(Record):
    id: str
    target: tuple[float, float]
    pace: float = .8
    tolerance: float = 4.
    deadline_ticks: int = 240

    def validate(self) -> None:
        identifier(self.id, "goal.id")
        if not 0 < self.pace <= 1 or not .1 <= self.tolerance <= 32 or not 6 <= self.deadline_ticks <= 600:
            raise ContractError("morphology_goal", "goal pace, tolerance or observable deadline outside declaration")


@dataclass(frozen=True)
class SemanticAction(Record):
    drive_x: float = 0.
    drive_y: float = 0.
    trim: float = 0.
    effort: float = 0.
    impulse: bool = False
    brake: bool = False
    rebound: bool = False

    def validate(self) -> None:
        if any(abs(getattr(self, key)) > 1 for key in ACTION_NAMES[:3]) or not 0 <= self.effort <= 1:
            raise ContractError("morphology_action", "already bounded semantic requests required")

    @property
    def vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in ACTION_NAMES)


@dataclass(frozen=True)
class _OwnedProfile(Record):
    value: Mapping[str, Any]


@dataclass(frozen=True)
class MorphologyIdentity(Record):
    run_id: str
    epoch: int
    actor_id: int
    model: str
    profile_version: int
    profile_sha256: str
    control_generation: int
    source_generation: int
    observation_id: str
    tick: int
    topology_sha256: str
    mount_generation: int | None = None

    def validate(self) -> None:
        for name in ("run_id", "model", "observation_id"):
            if not getattr(self, name) or len(getattr(self, name)) > 256:
                raise ContractError("morphology_identity", f"invalid {name}")
        for name in ("profile_sha256", "topology_sha256"):
            digest(getattr(self, name), name)
        if min(self.actor_id, self.tick, self.control_generation) < 0 or min(self.epoch, self.source_generation) < 1:
            raise ContractError("morphology_identity", "invalid lifetime counter")
        if self.profile_version != 1 or (self.mount_generation is not None and self.mount_generation < 0):
            raise ContractError("morphology_identity", "unsupported profile or invalid mount generation")
        expected = f"{self.run_id}:{self.epoch}:{self.actor_id}:{self.control_generation}:{self.tick}"
        if self.observation_id != expected:
            raise ContractError("morphology_identity", "source observation ID differs from its counters")

    @property
    def lifetime(self) -> tuple[str, int, int]:
        return self.run_id, self.epoch, self.actor_id


@dataclass(frozen=True)
class EdgeSample(Record):
    source: int
    destination: int
    present: bool
    active: bool
    values: tuple[float, ...]
    valid: tuple[bool, ...]

    def validate(self) -> None:
        if not 0 <= self.source < MAX_PARTS or not 0 <= self.destination < MAX_PARTS:
            raise ContractError("morphology_edge", "edge endpoint outside padding")
        if self.active and (not self.present or self.source == self.destination):
            raise ContractError("morphology_edge", "active edges need two present endpoints")
        if len(self.values) != len(EDGE_NAMES) or len(self.valid) != len(EDGE_NAMES):
            raise ContractError("morphology_shape", "edge feature schema differs")
        if any(value != 0 for value, valid in zip(self.values, self.valid) if not valid):
            raise ContractError("morphology_mask", "unavailable edge values must be zero with false validity")
        if not self.active and (any(self.values) or any(self.valid)):
            raise ContractError("morphology_mask", "inactive edges cannot carry hidden sensor payloads")


@dataclass(frozen=True)
class MorphologyFrame(Record):
    identity: MorphologyIdentity
    goal: MorphologyGoal
    nodes: tuple[tuple[float, ...], ...]
    node_valid: tuple[tuple[bool, ...], ...]
    present: tuple[bool, ...]
    active: tuple[bool, ...]
    edges: tuple[EdgeSample, ...]
    root_values: tuple[float, ...]
    root_valid: tuple[bool, ...]
    command: tuple[float, ...]
    action_mask: tuple[bool, ...]
    root_index: int = 0
    schema: Literal["act004-morphology-local/v1"] = SCHEMA

    def validate(self) -> None:
        if not all(len(value) == MAX_PARTS for value in (self.nodes, self.node_valid, self.present, self.active)):
            raise ContractError("morphology_shape", "exactly eight padded part slots required")
        if not 0 <= self.root_index < MAX_PARTS or not self.active[self.root_index]:
            raise ContractError("morphology_root", "controllable root part required")
        for row, valid, present, active in zip(self.nodes, self.node_valid, self.present, self.active):
            if len(row) != len(NODE_NAMES) or len(valid) != len(NODE_NAMES) or (active and not present):
                raise ContractError("morphology_shape", "node feature or presence schema differs")
            if any(value != 0 for value, available in zip(row, valid) if not available):
                raise ContractError("morphology_mask", "unavailable node features must be masked zeros")
            if not active and (any(row) or any(valid)):
                raise ContractError("morphology_mask", "inactive parts cannot carry current policy sensors")
        if len(self.edges) != MAX_JOINTS:
            raise ContractError("morphology_shape", "exactly seven padded connection slots required")
        endpoints = set()
        for edge in self.edges:
            if edge.present:
                pair = tuple(sorted((edge.source, edge.destination)))
                if pair in endpoints or edge.source == edge.destination:
                    raise ContractError("morphology_edge", "duplicate or self connection")
                endpoints.add(pair)
            if edge.active and not (self.active[edge.source] and self.active[edge.destination]):
                raise ContractError("morphology_mask", "active edge has an inactive endpoint")
        if (len(self.root_values) != len(ROOT_NAMES) or len(self.root_valid) != len(ROOT_NAMES)
                or len(self.command) != len(COMMAND_NAMES) or len(self.action_mask) != len(ACTION_NAMES)):
            raise ContractError("morphology_shape", "root, command or action schema differs")
        if any(value != 0 for value, valid in zip(self.root_values, self.root_valid) if not valid):
            raise ContractError("morphology_mask", "unavailable root features must be masked zeros")

    @property
    def policy_sha256(self) -> str:
        # Source identities remain receipt metadata, never a learned embedding.
        value = self.to_dict()
        del value["identity"]
        del value["goal"]
        return sha256_json(value)

    def padded(self) -> tuple[float, ...]:
        result = []
        for values, valid, present, active in zip(self.nodes, self.node_valid, self.present, self.active):
            result.extend(values)
            result.extend(float(value) for value in valid)
            result.extend((float(present), float(active)))
        for edge in self.edges:
            result.extend(edge.values)
            result.extend(float(value) for value in edge.valid)
            result.extend(float(edge.present and edge.source == i) for i in range(MAX_PARTS))
            result.extend(float(edge.present and edge.destination == i) for i in range(MAX_PARTS))
            result.extend((float(edge.present), float(edge.active)))
        result.extend(self.root_values)
        result.extend(float(value) for value in self.root_valid)
        result.extend(float(value) for value in self.action_mask)
        result.extend(self.command)
        result.extend(float(i == self.root_index) for i in range(MAX_PARTS))
        return tuple(result)


def information_reach(frame: MorphologyFrame, rounds: int, root: int | None = None) -> tuple[int, ...]:
    """Current input reach with synchronous messages, excluding prior memory."""
    _integer(rounds, "message rounds")
    if rounds > 8:
        raise ContractError("message_budget", "at most eight declared rounds")
    start = frame.root_index if root is None else _integer(root, "root")
    if start >= MAX_PARTS or not frame.active[start]:
        raise ContractError("morphology_root", "reach starts at an active node")
    reached = {start}
    for _ in range(rounds):
        previous = set(reached)
        for edge in frame.edges:
            if edge.active:
                if edge.source in previous:
                    reached.add(edge.destination)
                if edge.destination in previous:
                    reached.add(edge.source)
    return tuple(sorted(reached))


class MorphologyAdapter:
    """A pinned profile projected into one declared local observation schema."""
    def __init__(self, profile: Mapping[str, Any], *, expected_sha256: str):
        from nca.data.records import json_copy
        digest(expected_sha256, "profile receipt")
        if sha256_json(profile) != expected_sha256:
            raise ContractError("morphology_profile_hash", "full selected host profile differs from receipt")
        self.profile = json_copy(profile)
        self.profile_sha256 = expected_sha256
        self.model = profile.get("id")
        if self.model not in CAPABILITIES or profile.get("representation") != "articulated-rapier" or profile.get("version") != 1:
            raise ContractError("morphology_profile", "only the four declared physical profiles are mapped")
        self.capability = _OwnedProfile(CAPABILITIES[self.model]).value
        for name in ("movement", "primary", "secondary"):
            if profile.get("profile", {}).get(name) != self.capability[name]:
                raise ContractError("morphology_capability", "wire names cannot silently change actuator meaning")
        self.parts = self.profile.get("parts")
        self.joints = self.profile.get("joints")
        if (not isinstance(self.parts, list) or not 1 <= len(self.parts) <= MAX_PARTS
                or not isinstance(self.joints, list) or len(self.joints) > MAX_JOINTS
                or self.profile.get("part_count") != len(self.parts)):
            raise ContractError("morphology_capacity", "one to eight parts and at most seven authored joints")
        if len({part.get("id") for part in self.parts}) != len(self.parts):
            raise ContractError("morphology_profile", "distinct authored part identities required")
        for part in self.parts:
            identifier(part["id"], "part.id")
            if part.get("shape") != "rectangle" or min(_pair(part.get("size"), "part.size")) <= 0:
                raise ContractError("morphology_geometry", "positive authored rectangles required")
            if _number(part.get("mass"), "part.mass") <= 0:
                raise ContractError("morphology_geometry", "positive part mass required")
            _pair(part.get("offset"), "part.offset")
            _number(part.get("angle"), "part.angle")
        pairs, names = set(), set()
        for joint in self.joints:
            a, b = (_integer(joint.get(key), "joint." + key) for key in ("a", "b"))
            pair = tuple(sorted((a, b)))
            if a == b or max(a, b) >= len(self.parts) or pair in pairs or joint.get("id") in names:
                raise ContractError("morphology_profile", "invalid or duplicate joint")
            if joint.get("type") not in ("rotary", "spring", "slide"):
                raise ContractError("morphology_profile", "unsupported connection semantics")
            pairs.add(pair)
            names.add(joint["id"])
        self.action_mask = (True, self.capability["movement"] == "flight", True, True,
                            *(self.capability[name] is not None for name in ACTION_NAMES[4:]))
        self.profile = _OwnedProfile(self.profile).value
        self.parts, self.joints = self.profile["parts"], self.profile["joints"]

    def source_sha256(self, view: Mapping[str, Any]) -> str:
        """Stamp only original local inputs, so another view cannot renew inference."""
        keys = ("version", "representation", "observation_scope", "observation_id", "run_id", "epoch",
                "actor_id", "model", "rig_id", "profile_version", "control_generation", "tick",
                "part_count", "fault", "suspended", "contact_normal", *(name for name, _, _ in ROOT_SENSORS))
        value = {key: view.get(key) for key in keys}
        value["parts"] = [{key: part.get(key) for key in
                           ("id", "index", "position", "rotation", "velocity", "angular_velocity", "size", "mass", "detached")}
                          for part in view.get("parts", ())]
        value["joints"] = [{key: joint.get(key) for key in
                            ("id", "a", "b", "type", "enabled", "limits", "target", "angle", "speed", "travel", "anchor_error")}
                           for joint in view.get("joints", ())]
        return sha256_json(value)

    def identity(self, view: Mapping[str, Any], *, source_generation: int = 1,
                 mount_generation: int | None = None) -> MorphologyIdentity:
        """Validate an available original observation that may authorize inference."""
        return self._identity(view, source_generation=source_generation,
                              mount_generation=mount_generation, expected_suspended=False)

    def terminal_identity(self, view: Mapping[str, Any], *, source_generation: int = 1,
                          mount_generation: int | None = None) -> MorphologyIdentity:
        """Validate a revoked terminal observation without admitting inference."""
        return self._identity(view, source_generation=source_generation,
                              mount_generation=mount_generation, expected_suspended=True)

    def _identity(self, view: Mapping[str, Any], *, source_generation: int,
                  mount_generation: int | None, expected_suspended: bool) -> MorphologyIdentity:
        if (type(view.get("version")) is not int or view["version"] != 2
                or view.get("representation") != "articulated-rapier"
                or view.get("observation_scope") != OBSERVATION_SCOPE
                or view.get("model") != self.model or view.get("rig_id") != self.model):
            raise ContractError("morphology_observation", "original own-actor ACT004 observation required")
        if not isinstance(view.get("fault"), str) or type(view.get("suspended")) is not bool:
            raise ContractError("morphology_observation", "explicit fault and suspension state required")
        if view["fault"] or view["suspended"] is not expected_suspended:
            raise ContractError("morphology_unavailable", "fault or suspension differs from the required observation state")
        for key in ("epoch", "actor_id", "profile_version", "control_generation", "tick"):
            _integer(view.get(key), key, 1 if key in ("epoch", "profile_version") else 0)
        _integer(source_generation, "source_generation", 1)
        if mount_generation is not None:
            _integer(mount_generation, "mount_generation")
        expected_id = ":".join(str(view[key]) for key in ("run_id", "epoch", "actor_id", "control_generation", "tick"))
        if view.get("observation_id") != expected_id:
            raise ContractError("morphology_identity", "observation identity was refreshed or malformed")
        parts, joints = view.get("parts"), view.get("joints")
        _integer(view.get("part_count"), "part_count", 1)
        if (not isinstance(parts, (list, tuple)) or len(parts) != len(self.parts)
                or not isinstance(joints, (list, tuple)) or len(joints) != len(self.joints)
                or view.get("part_count") != len(parts)):
            raise ContractError("morphology_topology", "host part or joint count changed")
        for i, (sample, definition) in enumerate(zip(parts, self.parts)):
            if (not isinstance(sample, Mapping) or sample.get("id") != definition["id"]
                    or _integer(sample.get("index"), "part.index") != i):
                raise ContractError("morphology_topology", "part ordering is part of this adapter contract")
            _boolean(sample.get("detached"), "part.detached")
            for x, y in zip(_pair(sample.get("size"), "part.size"), definition["size"]):
                if not math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-6):
                    raise ContractError("morphology_geometry", "observed geometry differs from the pinned profile")
            if not math.isclose(_number(sample.get("mass"), "part.mass"), definition["mass"], rel_tol=1e-6, abs_tol=1e-6):
                raise ContractError("morphology_geometry", "observed mass differs from the pinned profile")
        for sample, definition in zip(joints, self.joints):
            if not isinstance(sample, Mapping):
                raise ContractError("morphology_topology", "original joint records required")
            for key in ("id", "type", "a", "b"):
                if key in ("a", "b"):
                    _integer(sample.get(key), "joint." + key)
                if sample.get(key) != definition[key]:
                    raise ContractError("morphology_topology", "joint identity or endpoint changed")
            _boolean(sample.get("enabled"), "joint.enabled")
        topology = {"parts": [(part["id"], part["index"], part["detached"]) for part in parts],
                    "joints": [(joint["id"], joint["a"], joint["b"], joint["enabled"]) for joint in joints]}
        return MorphologyIdentity(
            view["run_id"], view["epoch"], view["actor_id"], self.model, view["profile_version"],
            self.profile_sha256, view["control_generation"], source_generation, view["observation_id"],
            view["tick"], sha256_json(topology), mount_generation,
        )

    def frame(self, view: Mapping[str, Any], goal: MorphologyGoal, *, previous: SemanticAction | None = None,
              started_tick: int | None = None, source_generation: int = 1,
              mount_generation: int | None = None, current_tick: int | None = None) -> MorphologyFrame:
        identity = self.identity(view, source_generation=source_generation, mount_generation=mount_generation)
        if not isinstance(goal, MorphologyGoal):
            raise ContractError("morphology_goal", "explicit typed goal required")
        now = identity.tick if current_tick is None else _integer(current_tick, "current_tick")
        if not 0 <= now - identity.tick < 12:
            raise ContractError("morphology_expired", "original observation lease expired")
        start = identity.tick if started_tick is None else _integer(started_tick, "started_tick")
        if not 0 <= identity.tick - start or now - start >= goal.deadline_ticks:
            raise ContractError("morphology_deadline", "goal deadline elapsed or start lies in the future")
        previous = SemanticAction() if previous is None else previous
        if not isinstance(previous, SemanticAction):
            raise ContractError("morphology_history", "typed previous semantic request required")
        nodes, validity, present, active = [], [], [], []
        for part, definition in zip(view["parts"], self.parts):
            available = not part["detached"]
            present.append(True)
            active.append(available)
            if not available:
                nodes.append((0.,) * len(NODE_NAMES))
                validity.append((False,) * len(NODE_NAMES))
                continue
            angle = _number(part.get("rotation"), "part.rotation")
            pos = _pair(part.get("position"), "part.position")
            velocity = _local(_pair(part.get("velocity"), "part.velocity"), angle)
            displacement = _local((goal.target[0] - pos[0], goal.target[1] - pos[1]), angle)
            role = definition.get("role")
            role = role if role in ROLES else "other"
            values = (*[v / 120. for v in velocity], math.sin(angle), math.cos(angle),
                      _number(part.get("angular_velocity"), "part.angular_velocity") / 12.,
                      *[v / 96. for v in displacement], *[v / 24. for v in definition["size"]],
                      definition["mass"] / 5., *[v / 24. for v in definition["offset"]],
                      math.sin(definition["angle"]), math.cos(definition["angle"]),
                      *[float(role == value) for value in ROLES])
            nodes.append(tuple(_clip(value) for value in values))
            validity.append((True,) * len(NODE_NAMES))
        while len(nodes) < MAX_PARTS:
            nodes.append((0.,) * len(NODE_NAMES))
            validity.append((False,) * len(NODE_NAMES))
            present.append(False)
            active.append(False)
        edges = []
        for sample, definition in zip(view["joints"], self.joints):
            a, b = definition["a"], definition["b"]
            usable = sample["enabled"] and active[a] and active[b]
            if not usable:
                edges.append(EdgeSample(a, b, True, False, (0.,) * len(EDGE_NAMES), (False,) * len(EDGE_NAMES)))
                continue
            kind = definition["type"]
            scale = math.pi if kind == "rotary" else 12.
            limits = _pair(sample.get("limits"), "joint.limits")
            angle = _number(sample.get("angle"), "joint.angle")
            phase = _number(definition.get("phase", 0.), "joint.phase")
            travel = sample.get("travel")
            values = (*[float(kind == name) for name in ("rotary", "spring", "slide")],
                      *[v / 24. for v in _pair(definition.get("anchor"), "joint.anchor")],
                      *_pair(definition.get("axis"), "joint.axis"), *[v / scale for v in limits],
                      _number(definition.get("stiffness"), "joint.stiffness") / 100.,
                      _number(definition.get("damping"), "joint.damping") / 10.,
                      _number(definition.get("max_force"), "joint.max_force") / 500.,
                      _number(definition.get("motor_speed"), "joint.motor_speed") / 20.,
                      _number(definition.get("rest_length"), "joint.rest_length") / 24.,
                      math.sin(phase), math.cos(phase), _number(sample.get("target"), "joint.target") / scale,
                      math.sin(angle), math.cos(angle), _number(sample.get("speed"), "joint.speed") / 12.,
                      0. if travel is None else _number(travel, "joint.travel") / 12.,
                      _number(sample.get("anchor_error"), "joint.anchor_error") / 24.)
            valid = tuple(name != "travel/12" or travel is not None for name in EDGE_NAMES)
            edges.append(EdgeSample(a, b, True, True, tuple(_clip(v) for v in values), valid))
        while len(edges) < MAX_JOINTS:
            edges.append(EdgeSample(0, 0, False, False, (0.,) * len(EDGE_NAMES), (False,) * len(EDGE_NAMES)))
        root, root_valid = [], []
        for name, kind, scale in ROOT_SENSORS:
            value = view.get(name)
            root_valid.append(value is not None)
            value = 0. if value is None else float(_boolean(value, name) if kind == "bool" else _number(value, name)) / scale
            root.append(_clip(value))
        normal = view.get("contact_normal")
        root.extend((0., 0.) if normal is None else _pair(normal, "contact_normal"))
        root_valid.extend((normal is not None,) * 2)
        for name, scale in MECHANICS:
            root.append(_clip(_number(self.profile["profile"].get(name), "profile." + name) / scale))
            root_valid.append(True)
        root.extend(previous.vector)
        root_valid.extend((True,) * len(ACTION_NAMES))
        command = (goal.pace, goal.tolerance / 32., 1. - (now - start) / goal.deadline_ticks,
                   (now - identity.tick) / 12.)
        return MorphologyFrame(identity, goal, tuple(nodes), tuple(validity), tuple(present), tuple(active),
                               tuple(edges), tuple(root), tuple(root_valid), command, self.action_mask)


def map_action(adapter: MorphologyAdapter, action: SemanticAction, view: Mapping[str, Any],
               goal: MorphologyGoal) -> tuple[WireAction, tuple[str, ...]]:
    """One capability map for both candidate families; unavailable requests label masks."""
    if not isinstance(action, SemanticAction) or not isinstance(goal, MorphologyGoal):
        raise ContractError("morphology_action", "typed action and explicit goal required")
    adapter.identity(view)
    masked = tuple(name for name, allowed in zip(ACTION_NAMES, adapter.action_mask)
                   if not allowed and getattr(action, name))
    dx, dy = (action.drive_x, action.drive_y if adapter.action_mask[1] else 0.)
    norm = max(1., math.hypot(dx, dy))
    pos = _pair(view["parts"][0]["position"], "root.position")
    aim = (goal.target[0] - pos[0], goal.target[1] - pos[1])
    aim_norm = math.hypot(*aim)
    aim = (0., 0.) if aim_norm == 0 else (aim[0] / aim_norm, aim[1] / aim_norm)
    buttons = {"primary": False, "secondary": False}
    for semantic in ACTION_NAMES[4:]:
        channel = adapter.capability[semantic]
        if channel is not None:
            buttons[channel] |= getattr(action, semantic)
    return WireAction(move=(dx / norm, dy / norm), aim=aim, trim=action.trim,
                      effort=action.effort, **buttons), masked


class StateFence:
    """Per-instance invocation history; reset all memory on any physical boundary.

    No speculative hidden-state remapping is offered. Current four profiles have
    no exposed independently replaceable mount identity; a supplied mount count
    is an explicit observed connection generation, never an assumed zero.
    """
    def __init__(self):
        self.identity: MorphologyIdentity | None = None
        self.goal: MorphologyGoal | None = None
        self.start_tick = 0
        self.retired_goals: set[str] = set()
        self.retired_lifetimes: set[tuple[str, int, int]] = set()
        self.last_tick: int | None = None

    def cancel(self) -> None:
        if self.goal is not None:
            self.retired_goals.add(self.goal.id)
        self.last_tick = None

    def accept(self, identity: MorphologyIdentity, goal: MorphologyGoal) -> tuple[str, ...]:
        if not isinstance(identity, MorphologyIdentity) or not isinstance(goal, MorphologyGoal):
            raise ContractError("morphology_identity", "typed identity and goal required")
        reasons = []
        previous = self.identity
        if previous is None or identity.lifetime != previous.lifetime:
            if identity.lifetime in self.retired_lifetimes:
                raise ContractError("retired_morphology", "old actor incarnation cannot resume")
            if previous is not None:
                self.retired_lifetimes.add(previous.lifetime)
            if len(self.retired_lifetimes) > 1024:
                raise ContractError("morphology_capacity", "reconstruct the controller explicitly")
            self.retired_goals.clear()
            self.goal, self.last_tick = None, None
            reasons.append("actor_lifetime")
        else:
            if identity.tick < previous.tick:
                self.cancel()
                raise ContractError("morphology_reordered", "authority changes do not rewind the same actor's clock")
            if (identity.model, identity.profile_version, identity.profile_sha256) != (
                    previous.model, previous.profile_version, previous.profile_sha256):
                self.cancel()
                raise ContractError("profile_requires_reset", "new physical profile requires a new host incarnation")
            if identity.control_generation < previous.control_generation or identity.source_generation < previous.source_generation:
                self.cancel()
                raise ContractError("retired_owner", "old authority generation cannot resume")
            if previous.mount_generation is not None:
                if identity.mount_generation is None:
                    self.cancel()
                    raise ContractError("morphology_mount_unavailable", "a missing sample cannot erase a known mount generation")
                if identity.mount_generation < previous.mount_generation:
                    self.cancel()
                    raise ContractError("retired_mount", "old observed attachment generation cannot resume")
            for field, reason in (("control_generation", "control_generation"),
                                  ("source_generation", "source_generation"),
                                  ("topology_sha256", "part_or_connection_change"),
                                  ("mount_generation", "mount_generation")):
                if getattr(identity, field) != getattr(previous, field):
                    reasons.append(reason)
            if reasons:
                self.cancel()
        self.identity = identity
        if goal.id in self.retired_goals:
            raise ContractError("retired_invocation", "a fresh goal is required after cancellation or physical changes")
        if self.goal is not None and goal.id == self.goal.id and goal != self.goal:
            self.cancel()
            raise ContractError("morphology_goal_collision", "an invocation cannot change its target or deadline")
        # Check boundaries before time order: an actor reset legitimately resets
        # the host tick. A source/part change still requires a fresh invocation.
        if not reasons and self.last_tick is not None and identity.tick - self.last_tick != 6:
            self.cancel()
            raise ContractError("morphology_cadence", "one fresh completed six-tick observation required")
        if self.goal is None or goal.id != self.goal.id:
            if self.goal is not None:
                self.retired_goals.add(self.goal.id)
            if len(self.retired_goals) > 1024:
                raise ContractError("morphology_capacity", "a fresh episode is required")
            self.goal = goal
            self.start_tick = identity.tick
            reasons.append("goal_invocation")
        self.last_tick = identity.tick
        return tuple(reasons)
