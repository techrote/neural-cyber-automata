"""Exact local sensor assignment for the fixed Jolt three-part topology."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nca.contracts import ContractError, Record, WireAction, identifier
from nca.controllers.baseline import FeedbackObservation, SkillGoal
from nca.controllers.shaping import clamp, unit
from nca.data import PolicyContext, policy_view

PART_IDS = ("impact_core", "rebound_pad_-1", "rebound_pad_1")
EDGES = ((0, 1), (1, 0), (0, 2), (2, 0))
JOLT_PROFILE_SHA256 = "97559dcca7b1523ee622fd886615f05f5d7f85bacc0655ba9d642e909dc112bc"
FEATURE_SCHEMA = "jolt-per-part-proprioception-goal/v1"
FEATURE_NAMES = (
    "target_dx_div64", "target_dy_div64", "own_vx_div80", "own_vy_div100",
    "own_sin_rotation", "own_cos_rotation", "own_angular_velocity_div8",
    "root_grounded", "root_grounded_valid", "root_energy", "root_energy_valid",
    "root_cooldown", "root_cooldown_valid", "root_ground_distance_div96", "root_ground_distance_valid",
    "root_ceiling_distance_div96", "root_ceiling_distance_valid", "time_remaining_fraction",
    "goal_pace", "goal_tolerance_div8", "previous_move_x", "previous_trim", "previous_effort",
    "previous_primary", "previous_secondary", "previous_auxiliary",
    "own_mass_div2", "own_size_x_div12", "own_size_y_div12", "is_root",
)
ACTION_NAMES = ("move_x", "trim", "effort", "primary", "secondary", "auxiliary")


def finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ContractError("graph_sensor", f"{name} requires a finite number")
    return float(value)


def pair(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ContractError("graph_sensor", f"{name} requires two numbers")
    return tuple(finite(item, name) for item in value)


def checked_parts(view):
    parts = view.get("parts", ())
    if not isinstance(parts, (tuple, list)) or len(parts) != 3:
        raise ContractError("graph_topology", "the commissioned Jolt graph requires exactly three parts")
    for index, (part, expected) in enumerate(zip(parts, PART_IDS)):
        if (not isinstance(part, Mapping) or part.get("id") != expected or part.get("index") != index
                or type(part.get("index")) is not int
                or part.get("detached") is not False):
            raise ContractError("graph_topology", "unknown, reordered or detached physical part")
    joints = view.get("joints", ())
    if not isinstance(joints, (tuple, list)) or len(joints) != 2 or any(
            not isinstance(joint, Mapping) or joint.get("a") != 0 or joint.get("b") != index + 1 or joint.get("enabled") is not True
            or type(joint.get("a")) is not int or type(joint.get("b")) is not int
            or joint.get("id") != ("pad_flex_-1", "pad_flex_1")[index]
            or joint.get("type") != "rotary" for index, joint in enumerate(joints)):
        raise ContractError("graph_topology", "physical joints differ from the declared star")
    return parts


@dataclass(frozen=True)
class GraphIdentity(Record):
    run_id: str
    actor_id: int
    epoch: int
    control_generation: int
    source_generation: int
    mount_generation: int
    goal_id: str
    model: str = "jev"
    profile_version: int = 1

    def validate(self):
        identifier(self.run_id, "graph.run_id")
        identifier(self.goal_id, "graph.goal_id")
        if (not 0 <= self.actor_id < 8 or self.epoch < 1 or self.model != "jev" or self.profile_version != 1
                or min(self.control_generation, self.source_generation, self.mount_generation) < 0):
            raise ContractError("graph_identity", "Jolt owner, profile and generation fields differ")

    @classmethod
    def from_view(cls, view, goal, *, source_generation=0, mount_generation=None):
        return cls(view["run_id"], view["actor_id"], view["epoch"], view["control_generation"],
                   source_generation, view.get("mount_generation", 0) if mount_generation is None else mount_generation,
                   goal.id, view["model"], view["profile_version"])


@dataclass(frozen=True)
class GraphFrame(Record):
    identity: GraphIdentity
    tick: int
    observation_id: str
    nodes: tuple[tuple[float, ...], ...]
    root_view: Mapping[str, Any]
    goal: PolicyContext
    feature_schema: str = FEATURE_SCHEMA

    def validate(self):
        identifier(self.observation_id, "graph.observation_id")
        if self.tick < 0 or self.feature_schema != FEATURE_SCHEMA:
            raise ContractError("graph_frame", "graph tick/schema mismatch")
        if len(self.nodes) != 3 or any(len(node) != len(FEATURE_NAMES) for node in self.nodes):
            raise ContractError("graph_features", "exactly three 30-feature local nodes are required")
        if self.goal.goal_id != self.identity.goal_id:
            raise ContractError("graph_goal", "feature context changed goal identity")


def action_vector(action: WireAction) -> tuple[float, ...]:
    return (action.move[0], action.trim, action.effort, float(action.primary),
            float(action.secondary), float(action.auxiliary))


def make_frame(view: Mapping[str, Any], goal: SkillGoal, *, previous: WireAction | None = None,
               start_tick: int = 0, source_generation: int = 0, mount_generation: int | None = None) -> GraphFrame:
    """Read each node's own motion; root sensors never enter pad feature slots."""
    root = FeedbackObservation.from_host(policy_view(view))
    if root.actor != "jev" or goal.skill != "directed_hop" or not root.valid:
        raise ContractError("graph_capability", "this candidate requires a valid Jolt directed-hop view")
    if type(start_tick) is not int or not 0 <= start_tick <= root.tick:
        raise ContractError("graph_clock", "goal origin must be a completed earlier tick")
    parts = checked_parts(view)
    identity = GraphIdentity.from_view(view, goal, source_generation=source_generation, mount_generation=mount_generation)
    previous = WireAction(effort=0) if previous is None else previous
    context = (max(0.0, 1 - (root.tick - start_tick) / (60 * goal.timeout_seconds)),
               goal.pace, goal.tolerance / 8, *action_vector(previous))

    def optional(value, scale=1.0):
        return (0.0, 0.0) if value is None else (clamp(finite(value, "root sensor") / scale, 0, 2), 1.0)

    root_sensors = (float(root.supported is True), float(root.supported is not None),
                    *optional(root.energy), *optional(root.cooldown),
                    *optional(root.ground_distance, 96), *optional(root.ceiling_distance, 96))
    nodes = []
    for index, part in enumerate(parts):
        position, velocity = pair(part["position"], "part.position"), pair(part["velocity"], "part.velocity")
        angle = finite(part["rotation"], "part.rotation")
        size = pair(part["size"], "part.size")
        mass = finite(part["mass"], "part.mass")
        if mass <= 0 or min(size) <= 0:
            raise ContractError("graph_sensor", "body mass and dimensions must be positive")
        values = (clamp((goal.target[0] - position[0]) / 64, -2, 2),
                  clamp((goal.target[1] - position[1]) / 64, -2, 2),
                  clamp(velocity[0] / 80, -2, 2), clamp(velocity[1] / 100, -2, 2),
                  math.sin(angle), math.cos(angle), clamp(finite(part["angular_velocity"], "part.angular_velocity") / 8, -2, 2),
                  *(root_sensors if index == 0 else (0.0,) * 10), *context,
                  clamp(mass / 2, 0, 2), clamp(size[0] / 12, 0, 2), clamp(size[1] / 12, 0, 2), float(index == 0))
        nodes.append(values)
    return GraphFrame(identity, root.tick, view["observation_id"], tuple(nodes), policy_view(view),
                      PolicyContext(goal.id, goal.skill, goal.target, goal.pace, goal.tolerance, goal.timeout_seconds))


def gate_primary(action: WireAction, view: Mapping[str, Any], previous: WireAction) -> WireAction:
    """Common request-side support gate; no hidden force or host-state mutation."""
    feedback = FeedbackObservation.from_host(policy_view(view))
    allowed = (feedback.supported is True and feedback.cooldown is not None
               and feedback.cooldown <= 0 and not previous.primary)
    if action.primary and not allowed:
        values = action.to_dict()
        values["primary"] = False
        return WireAction.from_dict(values)
    return action


def mapped_request(outputs: Sequence[float], view: Mapping[str, Any], goal: SkillGoal, previous: WireAction) -> WireAction:
    if len(outputs) != 6:
        raise ContractError("graph_action", "the graph root readout has six channels")
    outputs = tuple(finite(value, "graph readout") for value in outputs)
    feedback = FeedbackObservation.from_host(policy_view(view))
    dx, dy = goal.target[0] - feedback.position[0], goal.target[1] - feedback.position[1]
    aim = unit((clamp(dx / 30), min(-.55, dy / 30 - .55)))
    action = WireAction(move=(clamp(outputs[0]), 0), aim=aim, trim=clamp(outputs[1]),
                        effort=clamp(outputs[2], 0, 1), primary=outputs[3] >= .5,
                        secondary=outputs[4] >= .5, auxiliary=outputs[5] >= .5)
    return gate_primary(action, view, previous)
