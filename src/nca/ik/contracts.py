"""Strict nominal IK records; these are not ACT003 packets or host admissions.

The caller supplies trusted issued observations and current authority. Validation
does not send a command, allocate authority, or establish physical capability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from nca.contracts import (ContractError, LifecycleStamp, Record, digest,
                           identifier, nonnegative, unique)
from .profiles import (IKProfile, nominal_outer_reach, nominal_pose, rotate)


def _fail(code: str, detail: str) -> None:
    raise ContractError(code, detail)


def _identities(record: Record) -> None:
    for name in ("profile_sha256", "source_sha256", "topology_sha256"):
        digest(getattr(record, name), name)


def _profile_identity(record: Record, profile: IKProfile) -> None:
    for name, expected in (("profile_sha256", profile.sha256),
                           ("source_sha256", profile.source_sha256),
                           ("topology_sha256", profile.topology_sha256)):
        if getattr(record, name) != expected:
            _fail(name.removesuffix("_sha256") + "_mismatch", name)


@dataclass(frozen=True)
class RootPose(Record):
    """Issued chassis pose in world cells/radians, with x right and y down."""

    position: tuple[float, float]
    angle_radians: float


@dataclass(frozen=True)
class IKObservation(Record):
    """Copied observations use fixed slots; unknown measurements are null."""

    id: str
    tick: int
    lifecycle: LifecycleStamp
    profile_sha256: str
    source_sha256: str
    topology_sha256: str
    root_pose: RootPose | None
    part_present: tuple[bool, ...]
    joint_present: tuple[bool, ...]
    joint_measured: tuple[bool, ...]
    joint_active: tuple[bool, ...]
    joint_positions: tuple[float | None, ...]
    evidence_kind: Literal["nominal_fixture", "host_observation"] = "nominal_fixture"
    schema_version: Literal[1] = 1

    def validate(self) -> None:
        identifier(self.id, "observation.id")
        nonnegative(self.tick, "observation.tick")
        _identities(self)
        if len(self.part_present) != 8 or any(len(values) != 7 for values in
                (self.joint_present, self.joint_measured, self.joint_active,
                 self.joint_positions)):
            _fail("shape_mismatch", "IK v1 uses eight part and seven joint slots")
        for present, measured, active, value in zip(
                self.joint_present, self.joint_measured, self.joint_active,
                self.joint_positions):
            if (value is not None) != measured or (measured or active) and not present:
                _fail("invalid_mask", "present, measured and active are distinct masks")
            if value is not None and not -math.pi <= value < math.pi:
                _fail("out_of_range", "rotary measurement must be zero-relative and wrapped")

    def validate_profile(self, profile: IKProfile) -> None:
        _profile_identity(self, profile)
        expected_parts = tuple(i in {part.index for part in profile.parts}
                               for i in range(profile.part_capacity))
        expected_joints = tuple(i in {joint.index for joint in profile.joints}
                                for i in range(profile.joint_capacity))
        expected_active = tuple(i in {joint.index for joint in profile.joints
                                      if joint.id in profile.solver_joint_ids}
                                for i in range(profile.joint_capacity))
        if (self.part_present != expected_parts or
                self.joint_present != expected_joints):
            _fail("topology_mismatch", "fixed pilot parts/joints differ from the descriptor")
        if self.joint_active != expected_active:
            _fail("activity_mismatch", "only the declared right chain participates in IK")
        if self.root_pose is None or any(active and not measured for active, measured
                                         in zip(self.joint_active, self.joint_measured)):
            _fail("missing_required_measurement", "issued root and active joint poses required")

    def configuration(self, profile: IKProfile) -> dict[str, float]:
        self.validate_profile(profile)
        return {joint.id: self.joint_positions[joint.index]
                for joint in profile.joints if joint.id in profile.solver_joint_ids}


@dataclass(frozen=True)
class IKAuthority(Record):
    """Trusted caller state; recheck it at consumption, not only at solving."""

    current_tick: int
    lifecycle: LifecycleStamp
    source_id: str
    goal_id: str
    policy_sha256: str
    grant_id: str
    grant_generation: int
    last_consumed_sequence: int = -1
    revoked: bool = False

    def validate(self) -> None:
        nonnegative(self.current_tick, "authority.current_tick")
        identifier(self.source_id, "authority.source_id")
        identifier(self.goal_id, "authority.goal_id")
        digest(self.policy_sha256, "authority.policy_sha256")
        identifier(self.grant_id, "authority.grant_id")
        nonnegative(self.grant_generation, "authority.grant_generation")
        if self.last_consumed_sequence < -1:
            _fail("out_of_range", "last consumed sequence must be -1 or nonnegative")


@dataclass(frozen=True)
class IKTarget(Record):
    id: str
    lifecycle: LifecycleStamp
    source_id: str
    goal_id: str
    policy_sha256: str
    grant_id: str
    grant_generation: int
    sequence: int
    observation_id: str
    observed_tick: int
    observation_sha256: str
    profile_sha256: str
    source_sha256: str
    topology_sha256: str
    effector_id: str
    joint_ids: tuple[str, ...]
    requested_point: tuple[float, float]
    point_root: tuple[float, float]
    input_frame: Literal["root_local", "world"]
    lease_ticks: int
    goal_deadline_tick: int
    posture_preference: tuple[float, ...] | None = None
    output_kind: Literal["joint_position", "joint_delta"] = "joint_position"
    tolerance_cells: float = 0.000001
    schema_version: Literal[1] = 1

    def validate(self) -> None:
        for name in ("id", "source_id", "goal_id", "grant_id", "observation_id", "effector_id"):
            identifier(getattr(self, name), "target." + name)
        for name in ("sequence", "grant_generation", "observed_tick", "goal_deadline_tick"):
            nonnegative(getattr(self, name), "target." + name)
        _identities(self)
        digest(self.observation_sha256, "observation_sha256")
        digest(self.policy_sha256, "policy_sha256")
        unique(self.joint_ids, "target.joint_ids")
        if not self.joint_ids or len(self.joint_ids) > 7:
            _fail("shape_mismatch", "target joint IDs")
        for joint_id in self.joint_ids:
            identifier(joint_id, "target.joint_ids")
        if self.posture_preference is not None and len(self.posture_preference) != len(self.joint_ids):
            _fail("shape_mismatch", "posture preference follows ordered joint IDs")
        if not 1 <= self.lease_ticks <= 18 or self.goal_deadline_tick <= self.observed_tick:
            _fail("out_of_range", "exclusive target lifetime")
        if self.tolerance_cells <= 0:
            _fail("out_of_range", "target tolerance must be positive")

    def expires_at(self, profile: IKProfile) -> int:
        return min(self.observed_tick + self.lease_ticks,
                   self.observed_tick + profile.max_observation_age_ticks,
                   self.goal_deadline_tick)

    def validate_against(self, profile: IKProfile, observation: IKObservation,
                         authority: IKAuthority) -> None:
        _profile_identity(self, profile)
        observation.validate_profile(profile)
        if self.lifecycle != observation.lifecycle or self.lifecycle != authority.lifecycle:
            _fail("lifecycle_mismatch", "run, actor, epoch and every generation must match")
        if self.source_id != authority.source_id or self.goal_id != authority.goal_id:
            _fail("authority_mismatch", "source or goal ownership changed")
        if self.grant_id != authority.grant_id or self.grant_generation != authority.grant_generation:
            _fail("authority_mismatch", "owner-issued source grant changed")
        if self.policy_sha256 != authority.policy_sha256:
            _fail("policy_mismatch", "producing policy changed under this source")
        if authority.revoked:
            _fail("cancelled", "owner or goal revoked this target")
        if self.sequence <= authority.last_consumed_sequence:
            _fail("already_consumed", "target sequence does not advance this owner")
        if (self.observation_id != observation.id or self.observed_tick != observation.tick or
                self.observation_sha256 != observation.sha256):
            _fail("observation_mismatch", "use the exact original issued observation")
        if authority.current_tick < self.observed_tick:
            _fail("future_observation", "observation is newer than trusted current tick")
        if authority.current_tick >= self.observed_tick + profile.max_observation_age_ticks:
            _fail("stale_observation", "source observation age is exclusive")
        if self.lease_ticks > profile.max_target_lease_ticks:
            _fail("out_of_range", "lease exceeds profile limit")
        if authority.current_tick >= self.expires_at(profile):
            _fail("expired_target", "exclusive source-bound lease or goal deadline")
        if self.effector_id != profile.effector.id or self.joint_ids != profile.solver_joint_ids:
            _fail("unsupported_profile", "target must use the declared point effector and ordered joints")
        if self.posture_preference is not None:
            joints = {joint.id: joint for joint in profile.joints}
            for joint_id, preference in zip(self.joint_ids, self.posture_preference):
                joint = joints[joint_id]
                if not joint.target_lower <= preference <= joint.target_upper:
                    _fail("out_of_range", "posture preference exceeds inner target bounds")
        expected = point_to_root(self.requested_point, self.input_frame, observation)
        if math.dist(expected, self.point_root) > 1e-10:
            _fail("frame_mismatch", "world conversion must use the original issued root pose")


def point_to_root(point: tuple[float, float], frame: str,
                  observation: IKObservation) -> tuple[float, float]:
    """Convert intent once; a moving live root cannot refresh an old target."""
    # RootPose supplies strict tuple/finite validation for this public helper.
    point = RootPose(point, 0.0).position
    if frame == "root_local":
        return point
    if frame != "world":
        _fail("unsupported_frame", "IK v1 supports root_local or world input")
    if observation.root_pose is None:
        _fail("missing_required_measurement", "world conversion needs the issued root pose")
    root = observation.root_pose
    relative = (point[0] - root.position[0], point[1] - root.position[1])
    return RootPose(rotate(relative, -root.angle_radians), 0.0).position


def target_from_observation(profile: IKProfile, observation: IKObservation, *,
                            id: str, source_id: str, goal_id: str, sequence: int,
                            policy_sha256: str, grant_id: str, grant_generation: int,
                            point: tuple[float, float], goal_deadline_tick: int,
                            input_frame: Literal["root_local", "world"] = "root_local",
                            lease_ticks: int = 12,
                            posture_preference: tuple[float, ...] | None = None,
                            output_kind: Literal["joint_position", "joint_delta"] = "joint_position",
                            tolerance_cells: float = 0.000001) -> IKTarget:
    observation.validate_profile(profile)
    target = IKTarget(
        id=id, lifecycle=observation.lifecycle, source_id=source_id, goal_id=goal_id,
        policy_sha256=policy_sha256, grant_id=grant_id, grant_generation=grant_generation,
        sequence=sequence, observation_id=observation.id, observed_tick=observation.tick,
        observation_sha256=observation.sha256, profile_sha256=profile.sha256,
        source_sha256=profile.source_sha256, topology_sha256=profile.topology_sha256,
        effector_id=profile.effector.id, joint_ids=profile.solver_joint_ids,
        requested_point=point, point_root=point_to_root(point, input_frame, observation),
        input_frame=input_frame, lease_ticks=lease_ticks, goal_deadline_tick=goal_deadline_tick,
        posture_preference=posture_preference, output_kind=output_kind,
        tolerance_cells=tolerance_cells)
    target.validate_against(profile, observation,
                            IKAuthority(observation.tick, observation.lifecycle, source_id, goal_id,
                                        policy_sha256, grant_id, grant_generation))
    return target


@dataclass(frozen=True)
class JointReference(Record):
    """Desired angle or bounded angle increment; never velocity or effort."""

    joint_ids: tuple[str, ...]
    values: tuple[float, ...]
    kind: Literal["joint_position", "joint_delta"] = "joint_position"
    unit: Literal["radians"] = "radians"

    def validate(self) -> None:
        unique(self.joint_ids, "reference.joint_ids")
        if not self.joint_ids or len(self.joint_ids) != len(self.values):
            _fail("shape_mismatch", "reference values follow ordered joint IDs")
        for joint_id in self.joint_ids:
            identifier(joint_id, "reference.joint_ids")


@dataclass(frozen=True)
class InfeasibilityProof(Record):
    """A sufficient unrestricted radial bound, not an iterative failure label."""

    center_root: tuple[float, float]
    outer_radius_cells: float
    target_distance_cells: float
    kind: Literal["outside_outer_radius"] = "outside_outer_radius"

    def validate(self) -> None:
        if self.outer_radius_cells <= 0 or self.target_distance_cells <= self.outer_radius_cells:
            _fail("invalid_proof", "target must lie strictly outside a positive outer bound")


@dataclass(frozen=True)
class IKResult(Record):
    target_sha256: str
    solver_id: str
    status: Literal["solved", "best_effort", "geometrically_infeasible",
                    "iteration_budget_exhausted", "invalid_input", "unsupported_profile",
                    "missing_required_measurement", "nonfinite_result"]
    reason: str
    reference: JointReference | None = None
    achieved_point_root: tuple[float, float] | None = None
    residual_cells: float | None = None
    orientation_radians: float | None = None
    proof: InfeasibilityProof | None = None
    iterations: int = 0
    correction_iterations: int = 0
    used_fallback: bool = False
    evidence_kind: Literal["nominal_geometry"] = "nominal_geometry"
    schema_version: Literal[1] = 1

    def validate(self) -> None:
        digest(self.target_sha256, "result.target_sha256")
        identifier(self.solver_id, "result.solver_id")
        if not self.reason.strip() or len(self.reason) > 512:
            _fail("invalid_reason", "result requires a bounded explanation")
        nonnegative(self.iterations, "result.iterations")
        nonnegative(self.correction_iterations, "result.correction_iterations")
        actionable = self.status in ("solved", "best_effort")
        payload = (self.reference, self.achieved_point_root,
                   self.residual_cells, self.orientation_radians)
        if actionable and any(value is None for value in payload):
            _fail("missing_result", "a candidate must carry reference, nominal pose and residual")
        if not actionable and any(value is not None for value in payload):
            _fail("rejected_payload", "failure results cannot carry an action payload")
        if (self.status == "geometrically_infeasible") != (self.proof is not None):
            _fail("invalid_proof", "geometric infeasibility requires a checkable proof")
        if self.residual_cells is not None and self.residual_cells < 0:
            _fail("out_of_range", "negative residual")

    def validate_against(self, profile: IKProfile, target: IKTarget,
                         observation: IKObservation, authority: IKAuthority) -> None:
        target.validate_against(profile, observation, authority)
        if self.target_sha256 != target.sha256:
            _fail("target_mismatch", "result belongs to a different immutable target")
        if self.proof is not None:
            center, radius = nominal_outer_reach(profile)
            distance = math.dist(center, target.point_root)
            if (math.dist(center, self.proof.center_root) > 1e-9 or
                    abs(radius - self.proof.outer_radius_cells) > 1e-9 or
                    not math.isclose(distance, self.proof.target_distance_cells, rel_tol=1e-12, abs_tol=1e-9) or
                    distance <= radius + 1e-9):
                _fail("invalid_proof", "the profile's independent radial bound does not prove this claim")
            return
        if self.reference is None:
            return
        reference = self.reference
        if reference.joint_ids != target.joint_ids or reference.kind != target.output_kind:
            _fail("reference_mismatch", "result joint order or output mode differs from the request")
        current = observation.configuration(profile)
        positions = dict(zip(reference.joint_ids, reference.values))
        if reference.kind == "joint_delta":
            if any(abs(value) > profile.max_delta_radians for value in reference.values):
                _fail("out_of_range", "joint delta exceeds the declared per-decision bound")
            positions = {joint_id: current[joint_id] + value for joint_id, value in positions.items()}
        for joint in profile.joints:
            if joint.id in positions and not joint.target_lower <= positions[joint.id] <= joint.target_upper:
                _fail("out_of_range", "joint reference exceeds the inner target limits")
        pose = nominal_pose(profile, positions)
        residual = math.dist(pose.point, target.point_root)
        if (math.dist(pose.point, self.achieved_point_root) > 1e-8 or
                abs(pose.orientation_radians - self.orientation_radians) > 1e-9 or
                not math.isclose(residual, self.residual_cells, rel_tol=1e-10, abs_tol=1e-8)):
            _fail("result_mismatch", "nominal pose and residual must match the emitted references")
        if (self.status == "solved") != (residual <= target.tolerance_cells):
            _fail("status_mismatch", "solved and best_effort are distinguished by the requested tolerance")
