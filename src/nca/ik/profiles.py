"""Pinned nominal Runner geometry; no solver, host actuation or ML dependency.

The two active joints are solver variables. Passive/disabled participation says
nothing about whether the physical host motors or moves the other joints.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

from nca.contracts import ContractError, Record, digest, identifier, sha256_json, unique


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ContractError("ik_numeric", f"{name} must be a finite number, not a boolean")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ContractError("ik_numeric", f"{name} is nonfinite") from exc
    if not math.isfinite(result):
        raise ContractError("ik_numeric", f"{name} is nonfinite")
    return result


def rotate(point: tuple[float, float], angle: float) -> tuple[float, float]:
    """Rotate in the down-Y frame: positive angles look clockwise on screen."""
    if not isinstance(point, (tuple, list)) or len(point) != 2:
        raise ContractError("ik_shape", "point must have exactly two coordinates")
    x, y = (_number(value, "point") for value in point)
    angle = _number(angle, "angle")
    c, s = math.cos(angle), math.sin(angle)
    result = c * x - s * y, s * x + c * y
    if not all(math.isfinite(value) for value in result):
        raise ContractError("ik_numeric", "rotated point overflowed")
    return result


def relative_rotary_angle(child_angle: float, parent_angle: float,
                          zero_radians: float) -> float:
    """Return child minus parent minus authored zero in canonical [-pi, pi).

    Source body angles may straddle the wrap boundary. Finite inputs whose
    subtraction overflows are refused rather than producing an invented angle.
    """
    child = _number(child_angle, "child_angle")
    parent = _number(parent_angle, "parent_angle")
    zero = _number(zero_radians, "zero_radians")
    relative = child - parent - zero
    if not math.isfinite(relative):
        raise ContractError("ik_numeric", "relative angle subtraction overflowed")
    wrapped = math.remainder(relative, math.tau)
    return wrapped - math.tau if wrapped >= math.pi else wrapped


def _add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]


@dataclass(frozen=True)
class IKSourceFile(Record):
    path: str
    git_blob_sha1: str
    content_sha256: str
    size_bytes: int

    def validate(self) -> None:
        if self.path not in ("godot/scripts/physical_actor_catalog.gd",
                             "godot/scripts/physical_actor_rig.gd"):
            raise ContractError("ik_source", "only the two audited geometry sources are supported")
        if not re.fullmatch(r"[a-f0-9]{40}", self.git_blob_sha1) or self.size_bytes < 1:
            raise ContractError("ik_source", "invalid source blob identity or byte count")
        digest(self.content_sha256, "source.content_sha256")


@dataclass(frozen=True)
class IKSource(Record):
    repository: Literal["techrote/cybersand"]
    commit: str
    files: tuple[IKSourceFile, ...]
    derivation: Literal["audited-source-nominal-not-live-host-export"]

    def validate(self) -> None:
        if not re.fullmatch(r"[a-f0-9]{40}", self.commit):
            raise ContractError("ik_source", "full host source commit required")
        paths = tuple(item.path for item in self.files)
        unique(paths, "source.files")
        if set(paths) != {"godot/scripts/physical_actor_catalog.gd",
                          "godot/scripts/physical_actor_rig.gd"}:
            raise ContractError("ik_source", "both audited geometry sources are required")


@dataclass(frozen=True)
class IKConvention(Record):
    frame: Literal["root_local"]
    length_unit: Literal["game_cells"]
    angle_unit: Literal["radians"]
    time_unit: Literal["base_ticks_at_60_hz"]
    axes: Literal["x_right_y_down"]
    positive_rotation: Literal["clockwise_on_screen"]
    rotation_rule: Literal["(cos(a)*x-sin(a)*y,sin(a)*x+cos(a)*y)"]
    anchor_rule: Literal["R(-initial_angle)*(authored_pivot-initial_position)"]
    precision: Literal["nominal_binary64_host_rounding_unqualified"]
    root_motion: Literal["observed_frame_only_no_root_output"]


@dataclass(frozen=True)
class IKPart(Record):
    id: str
    index: int
    initial_position: tuple[float, float]
    initial_angle_radians: float
    size: tuple[float, float]
    mass: float
    role: Literal["hull", "leg", "foot"]
    shape: Literal["centred_rectangle"] = "centred_rectangle"

    def validate(self) -> None:
        identifier(self.id, "part.id")
        if not 0 <= self.index < 8 or min(self.size) <= 0 or self.mass <= 0:
            raise ContractError("ik_part", "invalid part slot, rectangle or mass")
        if abs(self.initial_angle_radians) > math.pi:
            raise ContractError("ik_part", "authored initial angle must be in [-pi, pi]")


@dataclass(frozen=True)
class IKJoint(Record):
    id: str
    index: int
    parent_part_id: str
    child_part_id: str
    authored_pivot: tuple[float, float]
    parent_anchor_local: tuple[float, float]
    child_anchor_local: tuple[float, float]
    zero_radians: float
    lower: float
    upper: float
    target_lower: float
    target_upper: float
    participation: Literal["active", "passive", "disabled"]
    kind: Literal["rotary"] = "rotary"
    coordinate_unit: Literal["radians"] = "radians"
    coordinate_semantics: Literal["child_minus_parent_minus_authored_zero"] = "child_minus_parent_minus_authored_zero"
    motor_speed_radians_per_second: float = 7.0

    def validate(self) -> None:
        for name in ("id", "parent_part_id", "child_part_id"):
            identifier(getattr(self, name), "joint." + name)
        if not 0 <= self.index < 7 or self.parent_part_id == self.child_part_id:
            raise ContractError("ik_topology", "invalid joint slot or self-loop")
        if not (-math.pi <= self.lower < self.upper <= math.pi
                and self.lower <= self.target_lower < self.target_upper <= self.upper
                and self.target_lower <= 0 <= self.target_upper):
            raise ContractError("ik_joint_bounds", "hard/target limits must include authored neutral")
        if abs(self.zero_radians) > math.pi or self.motor_speed_radians_per_second <= 0:
            raise ContractError("ik_joint", "invalid rotary reference or motor speed")


@dataclass(frozen=True)
class IKEffector(Record):
    id: str
    part_id: str
    point_local: tuple[float, float]
    outward_normal_local: tuple[float, float]
    task_components: tuple[Literal["position_x"], Literal["position_y"]] = ("position_x", "position_y")
    orientation: Literal["diagnostic_only_not_independent_control"] = "diagnostic_only_not_independent_control"

    def validate(self) -> None:
        identifier(self.id, "effector.id")
        identifier(self.part_id, "effector.part_id")
        if not math.isclose(math.hypot(*self.outward_normal_local), 1.0, abs_tol=1e-12):
            raise ContractError("ik_effector", "effector normal must be a unit vector")


@dataclass(frozen=True)
class IKWorkspace(Record):
    joint_ids: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    parameterization: Literal["joint_rectangle_pushforward_by_nominal_fk"]
    claim: Literal["nominal_geometric_reachability_only"]

    def validate(self) -> None:
        unique(self.joint_ids, "workspace.joint_ids")
        if len(self.joint_ids) != 2 or len(self.lower) != 2 or len(self.upper) != 2:
            raise ContractError("ik_workspace", "the point pilot requires two joint intervals")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ContractError("ik_workspace", "workspace intervals must be nonempty")


@dataclass(frozen=True)
class IKProfile(Record):
    id: Literal["runner-right-sole"]
    version: Literal[1]
    source: IKSource
    convention: IKConvention
    root_part_id: str
    parts: tuple[IKPart, ...]
    joints: tuple[IKJoint, ...]
    solver_joint_ids: tuple[str, ...]
    effector: IKEffector
    workspace: IKWorkspace
    authored_play_radians: float = 0.055
    max_delta_radians: float = 0.25
    part_capacity: Literal[8] = 8
    joint_capacity: Literal[7] = 7
    decision_ticks: Literal[6] = 6
    max_observation_age_ticks: Literal[12] = 12
    max_target_lease_ticks: Literal[18] = 18
    schema_version: Literal[1] = 1
    topology_kind: Literal["rooted_tree"] = "rooted_tree"
    host_target_lane: Literal["absent_requires_separate_host_integration"] = "absent_requires_separate_host_integration"
    participation_semantics: Literal["solver_only_passive_does_not_mean_physically_frozen"] = "solver_only_passive_does_not_mean_physically_frozen"
    observation_target_semantics: Literal["rotary_host_target_is_motor_velocity_not_position"] = "rotary_host_target_is_motor_velocity_not_position"

    def validate(self) -> None:
        if not 1 <= len(self.parts) <= self.part_capacity or len(self.joints) > self.joint_capacity:
            raise ContractError("ik_capacity", "profile exceeds fixed part/joint capacity")
        for values, name in ((self.parts, "parts"), (self.joints, "joints")):
            unique(tuple(item.id for item in values), name)
            if tuple(item.index for item in values) != tuple(range(len(values))):
                raise ContractError("ik_slots", "present items need unique, ordered contiguous slots")
        parts = {part.id: part for part in self.parts}
        if self.root_part_id not in parts or self.effector.part_id not in parts:
            raise ContractError("ik_topology", "root/effector refers to an absent part")
        root = parts[self.root_part_id]
        if root.index != 0 or root.initial_position != (0., 0.) or root.initial_angle_radians != 0:
            raise ContractError("ik_frame", "root-local initial transform must be identity in slot zero")
        parents: dict[str, IKJoint] = {}
        for joint in self.joints:
            if (joint.parent_part_id not in parts or joint.child_part_id not in parts
                    or joint.child_part_id == self.root_part_id or joint.child_part_id in parents):
                raise ContractError("ik_topology", "missing endpoint, multiple parent or root cycle")
            parents[joint.child_part_id] = joint
        reached = {self.root_part_id}
        for _ in self.parts:
            reached.update(j.child_part_id for j in self.joints if j.parent_part_id in reached)
        if reached != set(parts) or len(self.joints) != len(self.parts) - 1:
            raise ContractError("ik_topology", "only a connected acyclic rooted tree is supported")
        for joint in self.joints:
            parent, child = parts[joint.parent_part_id], parts[joint.child_part_id]
            if not math.isclose(child.initial_angle_radians - parent.initial_angle_radians,
                                joint.zero_radians, rel_tol=0, abs_tol=1e-12):
                raise ContractError("ik_zero", "rotary zero differs from authored initial transforms")
            for part, anchor in ((parent, joint.parent_anchor_local), (child, joint.child_anchor_local)):
                pivot = _add(part.initial_position, rotate(anchor, part.initial_angle_radians))
                if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-10)
                       for a, b in zip(pivot, joint.authored_pivot)):
                    raise ContractError("ik_anchor", "local anchor does not reconstruct its authored pivot")
        unique(self.solver_joint_ids, "solver_joint_ids")
        chain = self.effector_chain
        if (len(chain) != 2 or tuple(j.id for j in chain) != self.solver_joint_ids
                or tuple(j.id for j in self.joints if j.participation == "active") != self.solver_joint_ids):
            raise ContractError("ik_solver_chain", "only the two active root-to-effector rotary joints are supported")
        if not math.isfinite(self.nominal_reach_cells) or self.nominal_reach_cells <= 0:
            raise ContractError("ik_geometry", "finite positive nominal chain reach required")
        if self.workspace.joint_ids != self.solver_joint_ids:
            raise ContractError("ik_workspace", "workspace coordinate order differs from solver")
        for joint, lo, hi in zip(chain, self.workspace.lower, self.workspace.upper):
            if not joint.target_lower < lo < hi < joint.target_upper:
                raise ContractError("ik_workspace", "workspace must retain an inner target-limit margin")
        effector_part = parts[self.effector.part_id]
        if any(abs(value) > size / 2 for value, size in zip(self.effector.point_local, effector_part.size)):
            raise ContractError("ik_effector", "declared point must belong to the effector rectangle")
        if not 0 <= self.authored_play_radians < math.pi or not 0 < self.max_delta_radians <= 0.25:
            raise ContractError("ik_bounds", "invalid authored play or bounded-delta ceiling")

    @property
    def effector_chain(self) -> tuple[IKJoint, ...]:
        """Profile validation establishes one acyclic path; no off-chain state is invented."""
        parents = {joint.child_part_id: joint for joint in self.joints}
        chain: list[IKJoint] = []
        part = self.effector.part_id
        while part != self.root_part_id:
            joint = parents[part]
            chain.append(joint)
            part = joint.parent_part_id
        return tuple(reversed(chain))

    @property
    def nominal_reach_cells(self) -> float:
        first, second = self.effector_chain
        return (math.dist(first.child_anchor_local, second.parent_anchor_local)
                + math.dist(second.child_anchor_local, self.effector.point_local))

    @property
    def source_sha256(self) -> str:
        return self.source.sha256

    @property
    def topology_sha256(self) -> str:
        return sha256_json({"root": self.root_part_id,
                            "capacity": [self.part_capacity, self.joint_capacity],
                            "parts": [(p.id, p.index) for p in self.parts],
                            "joints": [(j.id, j.index, j.parent_part_id, j.child_part_id,
                                        j.kind, j.participation) for j in self.joints],
                            "solver_joint_ids": self.solver_joint_ids,
                            "effector": [self.effector.id, self.effector.part_id]})


@dataclass(frozen=True)
class NominalPose(Record):
    point: tuple[float, float]
    orientation_radians: float


def runner_right_sole_profile() -> IKProfile:
    """Load the packaged source-derived profile without importing a host or learner."""
    return IKProfile.from_json(files("nca").joinpath("ik/runner_right_sole_v1.json").read_bytes())


def nominal_outer_reach(profile: IKProfile) -> tuple[tuple[float, float], float]:
    """Unrestricted radial upper bound; being inside does not prove reachability."""
    if not isinstance(profile, IKProfile):
        raise ContractError("ik_profile", "validated IKProfile required")
    return profile.effector_chain[0].parent_anchor_local, profile.nominal_reach_cells


def nominal_pose(profile: IKProfile, joint_positions: Mapping[str, float]) -> NominalPose:
    """Tiny nominal fixture FK for exactly the active two-joint endpoint path.

    This is neither an inverse solver nor an estimate of compliant physical pose.
    Off-chain passive coordinates are legitimately absent, not assumed to be zero.
    """
    if not isinstance(profile, IKProfile):
        raise ContractError("ik_profile", "validated IKProfile required")
    if not isinstance(joint_positions, Mapping) or set(joint_positions) != set(profile.solver_joint_ids):
        raise ContractError("ik_joint_positions", "exactly the declared solver joint IDs are required")
    point, angle = (0., 0.), 0.
    for joint in profile.effector_chain:
        q = _number(joint_positions[joint.id], joint.id)
        if not joint.lower <= q <= joint.upper:
            raise ContractError("ik_joint_bounds", "coordinate exceeds its hard geometric limits")
        pivot = _add(point, rotate(joint.parent_anchor_local, angle))
        angle += joint.zero_radians + q
        point = _sub(pivot, rotate(joint.child_anchor_local, angle))
    return NominalPose(_add(point, rotate(profile.effector.point_local, angle)), angle)
