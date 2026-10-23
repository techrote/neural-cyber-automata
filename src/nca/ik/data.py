"""Reproducible nominal IK tasks, isolated families, and checked teacher labels.

These are geometric tasks, never ACT004 episodes. Generator inputs replay
independently of iteration/wall-limited teacher outcomes. Final scoring access
and immutable storage are owned by tools/ik_dataset.py.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import platform
from typing import Literal

from nca.contracts import ContractError, Record, canonical_json, digest, identifier, sha256_json, strict_json_loads
from nca.randomness import ScenarioSplit
from .kinematics import PlanarChain, chain_from_profile, cpu_fk
from .profiles import runner_right_sole_profile
from .reference import (AnalyticSolution, NumericSolution, SolverConfig,
                        analytic_two_link, solve_numeric)

PARTITIONS = ("training", "tuning", "final")
STRESS_CLASSES = ("inner_limit_boundary", "hard_limit_boundary", "certified_geometric_infeasible",
                  "unsupported_profile_or_effector", "missing_measurement_or_inactive_dof",
                  "nonfinite_or_malformed_input", "iteration_exhaustion")
BOUNDARY_CLASSES = STRESS_CLASSES[:2]
CURRICULUM = ("interior", "limits", "moving_and_reversing", "explicit_failures")
GENERATOR_VERSION = "runner-ik-families/v1"
FEATURE_SCHEMA = "runner-ik-permitted-inputs/v1"
FEATURE_NAMES = ("current_hip", "current_ankle", "target_x", "target_y",
                 "posture_hip", "posture_ankle", "present_hip", "present_ankle",
                 "measured_hip", "measured_ankle", "active_hip", "active_ankle",
                 "task_x", "task_y")
Partition = Literal["training", "tuning", "final"]


@dataclass(frozen=True)
class GenerationBudget(Record):
    max_unique_tasks: int = 10220
    max_teacher_attempts: int = 10220
    max_teacher_iterations: int = 654080
    max_wall_seconds: float = 900.0
    max_memory_mib: int = 512
    publication_reserve_seconds: float = 120.0

    def validate(self):
        if not 1 <= self.max_unique_tasks <= 1_000_000:
            raise ContractError("ik_dataset_budget", "one to one million unique tasks")
        if not self.max_unique_tasks <= self.max_teacher_attempts <= 1_000_000:
            raise ContractError("ik_dataset_budget", "attempt ceiling must cover unique tasks")
        if not 0 <= self.max_teacher_iterations <= 256_000_000:
            raise ContractError("ik_dataset_budget", "invalid aggregate iteration ceiling")
        if not 0 < self.publication_reserve_seconds < self.max_wall_seconds <= 3600:
            raise ContractError("ik_dataset_budget", "finite wall time and publication reserve required")
        if not 64 <= self.max_memory_mib <= 4096:
            raise ContractError("ik_dataset_budget", "bounded memory required")


@dataclass(frozen=True)
class DatasetConfig(Record):
    id: str = "runner-ik-corpus-v1"
    mode: Literal["fixture", "prospective"] = "prospective"
    generation_seed: int = 53021
    split_seed: int = 61003
    points_per_bin: tuple[int, int, int] = (60, 15, 25)
    trajectories_per_partition: int = 40
    trajectory_steps: int = 20
    stress_per_class: int = 20
    batch_size: int = 64
    withheld_bins: tuple[int, ...] = (6, 15, 24, 33, 46, 55, 64, 73)
    # FK images of these q rectangles are the exact workspace regions. This
    # mapping is one-to-one in the accepted Runner range; boxes in x/y are not.
    hip_lanes: tuple[tuple[float, float], ...] = ((.06, .50), (.62, .70), (.82, .94))
    joint_guard_radians: float = .002
    point_guard_cells: float = .0005
    teacher: SolverConfig = field(default_factory=lambda: SolverConfig(tolerance_cells=1e-7))
    budget: GenerationBudget = field(default_factory=GenerationBudget)
    schema: Literal["nca-ik-dataset-config/v1"] = "nca-ik-dataset-config/v1"

    def validate(self):
        identifier(self.id, "dataset.id")
        if self.split_seed != 61003 or not 0 <= self.generation_seed < 2**53:
            raise ContractError("ik_dataset_seed", "preserve split seed 61003 and finite generator seed")
        if any(not 1 <= n <= 10000 for n in self.points_per_bin):
            raise ContractError("ik_dataset_count", "positive bounded point populations required")
        if not 1 <= self.trajectories_per_partition <= 1000 or not 3 <= self.trajectory_steps <= 1000:
            raise ContractError("ik_dataset_count", "bounded complete trajectories required")
        if not 1 <= self.stress_per_class <= 1000 or not 1 <= self.batch_size <= 1024:
            raise ContractError("ik_dataset_count", "bounded stress and batch populations required")
        if len(set(self.withheld_bins)) != 8 or any(not 0 <= b < 80 for b in self.withheld_bins):
            raise ContractError("ik_dataset_regions", "exactly eight distinct grid regions required")
        if len(self.hip_lanes) != 3 or any(not .02 <= a < b <= .98 for a, b in self.hip_lanes):
            raise ContractError("ik_dataset_regions", "three interior partition lanes required")
        if not 0 < self.joint_guard_radians <= .004 or not 0 < self.point_guard_cells <= .001:
            raise ContractError("ik_dataset_guard", "positive conservative geometric guards required")
        if any((self.hip_lanes[i+1][0] - self.hip_lanes[i][1]) * .12 <= 2*self.joint_guard_radians
               for i in range(2)):
            raise ContractError("ik_dataset_guard", "partition lanes overlap their exclusion bands")
        if self.mode == "prospective" and (self.points_per_bin[2] < 25 or
                self.trajectories_per_partition < 40 or self.trajectory_steps < 20 or
                self.stress_per_class < 20):
            raise ContractError("ik_dataset_acceptance", "preserve accepted final population minima")
        if self.expected_tasks > self.budget.max_unique_tasks:
            raise ContractError("ik_dataset_budget", "scheduled population exceeds unique-task ceiling")
        if self.expected_tasks * self.teacher.max_iterations > self.budget.max_teacher_iterations:
            raise ContractError("ik_dataset_budget", "reserve worst-case iterations before generation")

    @property
    def expected_tasks(self):
        return (72*(self.points_per_bin[0]+self.points_per_bin[1]) + 80*self.points_per_bin[2]
                + 3*self.trajectories_per_partition*self.trajectory_steps + 3*len(STRESS_CLASSES)*self.stress_per_class)


@lru_cache(maxsize=16)
def _config_hash(config):
    return config.sha256


@lru_cache(maxsize=1)
def _profile():
    return runner_right_sole_profile()


@lru_cache(maxsize=1)
def _profile_hash():
    return _profile().sha256


@lru_cache(maxsize=1)
def _chain():
    return chain_from_profile(_profile())


@lru_cache(maxsize=16)
def _families(seed):
    ordered = sorted((f"ik-family-{i:03d}" for i in range(100)),
                     key=lambda v: sha256_json([GENERATOR_VERSION, seed, v]))
    split = ScenarioSplit(tuple(ordered[:70]), tuple(ordered[70:85]), tuple(ordered[85:]))
    return split


def _partition_families(config, partition):
    split = _families(config.split_seed)
    families = getattr(split, "evaluation" if partition == "final" else partition)
    from .fresh_final import FreshFinalConfig
    if type(config) is FreshFinalConfig:
        if partition != "final":
            raise ContractError("ik_fresh_partition", "fresh recipe is final-only")
        return tuple(config.family_namespace + ":" + f for f in families)
    return families


def _bounds_for(config, partition, bin_index):
    from .fresh_final import FreshFinalConfig
    if type(config) is FreshFinalConfig:
        return config.bounds(partition, bin_index)
    if partition not in PARTITIONS or not 0 <= bin_index < 80:
        raise ContractError("ik_dataset_region", "unknown partition or bin")
    if partition != "final" and bin_index in config.withheld_bins:
        raise ContractError("ik_dataset_leakage", "withheld region cannot enter training or tuning")
    h, a = divmod(bin_index, 8)
    lane = (.08, .92) if bin_index in config.withheld_bins else config.hip_lanes[PARTITIONS.index(partition)]
    return ((-.6 + .12*(h+lane[0]), -.6 + .12*(h+lane[1])),
            (-.2 + .05*(a+.08), -.2 + .05*(a+.92)))


def build_split_plan(config: DatasetConfig):
    """Freeze all families and FK-image regions before any variant is generated."""
    from .fresh_final import FreshFinalConfig
    partitions = ("final",) if type(config) is FreshFinalConfig else PARTITIONS
    families = {p: list(_partition_families(config, p)) for p in partitions}
    regions = []
    for p in partitions:
        for b in range(80):
            if p != "final" and b in config.withheld_bins:
                continue
            regions.append({"partition": p, "bin": b, "q_bounds": _bounds_for(config, p, b),
                            "kind": "withheld_region" if b in config.withheld_bins else "interpolation",
                            "mapping": "accepted Runner FK image; closed bounds"})
    return {"schema": "nca-ik-split-plan/v1", "generator": GENERATOR_VERSION,
            "config_sha256": _config_hash(config), "split_seed": config.split_seed,
            "family_assignments": families, "regions": regions,
            "withheld_bins": config.withheld_bins, "grid_shape": [10, 8],
            "nominal_q_bounds": [[-.6, .6], [-.2, .2]],
            "joint_guard_radians": config.joint_guard_radians,
            "point_guard_cells": config.point_guard_cells,
            "curriculum": CURRICULUM,
            "extrapolation_definition": "withheld FK-image holes inside nominal workspace; near-limit stress outside it",
            "assignment_order": "family and complete trajectory before sampling or augmentation",
            "final_access": "structural QA only until durable candidate-bound opening"}


@dataclass(frozen=True)
class RuntimeInputs(Record):
    profile_sha256: str
    current: tuple[float | None, float | None]
    target_root: tuple[float, float]
    posture: tuple[float, float]
    present: tuple[bool, bool] = (True, True)
    measured: tuple[bool, bool] = (True, True)
    active: tuple[bool, bool] = (True, True)
    task_mask: tuple[bool, bool] = (True, True)
    schema: Literal["runner-ik-permitted-inputs/v1"] = FEATURE_SCHEMA

    def validate(self):
        digest(self.profile_sha256, "profile_sha256")
        for q, p, m, a in zip(self.current, self.present, self.measured, self.active):
            if m and (not p or q is None) or not m and q is not None or a and not p:
                raise ContractError("ik_dataset_mask", "missing measurements stay null and masks remain distinct")
        if any(q is not None and abs(q) > limit for q, limit in zip(self.current, (.75, .30))):
            raise ContractError("ik_dataset_bounds", "current coordinates exceed hard limits")
        if any(abs(q) > limit for q, limit in zip(self.posture, (.72, .27))):
            raise ContractError("ik_dataset_bounds", "declared preference exceeds command limits")


def runtime_features(inputs: RuntimeInputs):
    """Only explicit deployable fields; unknown fields fail in the strict codec.

    Null coordinates become masked numeric placeholders, never measurements.
    This API accepts no Sample/teacher dictionary and no evaluator feature names.
    """
    if not isinstance(inputs, RuntimeInputs) or inputs.profile_sha256 != _profile_hash():
        raise ContractError("ik_dataset_features", "exact permitted-input record and profile required")
    return tuple(float(v) for v in (*(q if q is not None else 0. for q in inputs.current),
                 *inputs.target_root, *inputs.posture, *inputs.present, *inputs.measured,
                 *inputs.active, *inputs.task_mask))


@dataclass(frozen=True)
class TaskState(Record):
    kind: Literal["point", "trajectory", "stress"]
    partition: Partition
    bin_index: int
    ordinal: int
    step: int = 0
    stress_class: str = ""
    augmentation: int = 0

    def validate(self):
        if not 0 <= self.bin_index < 80 or self.ordinal < 0 or self.step < 0 or self.augmentation not in (0, 1, 2):
            raise ContractError("ik_dataset_state", "invalid finite generator address")
        if (self.kind == "stress") != (self.stress_class in STRESS_CLASSES):
            raise ContractError("ik_dataset_state", "stress address and class disagree")
        if self.kind != "stress" and self.stress_class:
            raise ContractError("ik_dataset_state", "ordinary address cannot name a stress class")


@dataclass(frozen=True)
class GeometricTask(Record):
    id: str
    config_sha256: str
    state: TaskState
    partition: Partition
    family_id: str
    trajectory_id: str
    inputs: RuntimeInputs
    target_configuration: tuple[float, float] | None
    original_point: tuple[float, float]
    root_pose: tuple[float, float, float]
    region_kind: Literal["interpolation", "withheld_region", "outside_nominal", "infeasible"]
    stage: int
    invalid_specimen: str | None = None
    evidence_kind: Literal["nominal_geometry_not_act004"] = "nominal_geometry_not_act004"

    def validate(self):
        digest(self.id, "task.id")
        digest(self.config_sha256, "task.config_sha256")
        identifier(self.family_id, "task.family")
        if self.partition != self.state.partition or not 0 <= self.stage < 4:
            raise ContractError("ik_dataset_state", "task address and partition differ")

    @property
    def input_sha256(self):
        # Arbitrary IDs, frames, provenance and partition cannot hide a copy.
        return (sha256_json({"original_malformed_input":self.invalid_specimen})
                if self.invalid_specimen is not None else self.inputs.sha256)


def _unit(config, state, channel):
    # Hash counter RNG: independent of iteration/batch/worker ordering. Fifty-two
    # bits convert exactly to binary64; trigonometry is rounded to 12 decimals.
    key = [GENERATOR_VERSION, config.generation_seed, replace(state, augmentation=0).to_dict(), channel]
    return int(sha256_json(key)[:13], 16) / 2**52


def _rounded(values):
    return tuple(round(float(v), 12) for v in values)


def _target_q(config, state):
    bounds = _bounds_for(config, state.partition, state.bin_index)
    if state.kind == "trajectory":
        anchor = replace(state, step=0)
        t = state.step / max(1, config.trajectory_steps-1)
        # Continuous complete q-space curves confined to one assigned patch;
        # alternating sweep/reversal families are fixed before labels exist.
        phase = math.pi*t if state.ordinal % 2 else 2*math.pi*t
        return _rounded((sum(bounds[0])/2 + .36*(bounds[0][1]-bounds[0][0])*math.sin(phase),
                         sum(bounds[1])/2 + .32*(bounds[1][1]-bounds[1][0])*math.cos(phase + .2*_unit(config, anchor, "phase"))))
    q = [a + (b-a)*(.08 + .84*_unit(config, state, f"q{i}")) for i, (a, b) in enumerate(bounds)]
    if state.stress_class in BOUNDARY_CLASSES:
        # Exact hip boundary targets, with separated ankle strips per partition.
        # Hard-bound targets are outside command limits and retain that outcome.
        lane = config.hip_lanes[PARTITIONS.index(state.partition)]
        if state.ordinal%4 < 2:
            q[0] = (1 if state.ordinal % 2 else -1) * (.72 if state.stress_class == "inner_limit_boundary" else .75)
            q[1] = -.2 + .05*(state.bin_index%8 + lane[0] + (lane[1]-lane[0])*(.08+.84*_unit(config,state,"limit")))
        else:
            q[1] = (1 if state.ordinal%2 else -1) * (.27 if state.stress_class == "inner_limit_boundary" else .30)
    return _rounded(q)


def _allowed_bins(config, partition):
    return [b for b in range(80) if partition == "final" or b not in config.withheld_bins]


def replay_task(config: DatasetConfig, state: TaskState | dict):
    """Replay exact task identity/inputs, not elapsed time or teacher decisions."""
    state = TaskState.from_dict(state) if isinstance(state, dict) else state
    if not isinstance(state, TaskState):
        raise ContractError("ik_dataset_state", "typed generator address required")
    if state.kind == "point" and (state.ordinal >= config.points_per_bin[PARTITIONS.index(state.partition)] or state.step):
        raise ContractError("ik_dataset_state", "point address outside allocation")
    if state.kind == "trajectory" and (state.ordinal >= config.trajectories_per_partition or state.step >= config.trajectory_steps):
        raise ContractError("ik_dataset_state", "trajectory address outside allocation")
    if state.kind == "stress" and (state.ordinal >= config.stress_per_class or state.step):
        raise ContractError("ik_dataset_state", "stress address outside allocation")
    allowed = _allowed_bins(config, state.partition)
    if state.kind != "point" and state.bin_index != allowed[state.ordinal % len(allowed)]:
        raise ContractError("ik_dataset_state", "trajectory/stress region must precede sample generation")
    q = _target_q(config, state)
    pose = cpu_fk(_chain(), q)
    independent = independent_two_link_point(_chain(), q)
    if not pose.valid or math.dist(pose.point, independent) > 1e-8:
        raise ContractError("ik_dataset_endpoint", "generator disagrees with independent endpoint")
    # Preserve exact boundary FK bytes: decimal rounding can move a certified
    # boundary point just outside the independent inverse's legal interval.
    target = pose.point if state.stress_class in BOUNDARY_CLASSES else _rounded(pose.point)
    region = "withheld_region" if state.bin_index in config.withheld_bins else "interpolation"
    if state.stress_class in BOUNDARY_CLASSES:
        region = "outside_nominal"
    if state.stress_class == "certified_geometric_infeasible":
        target = _rounded((15 + 2*PARTITIONS.index(state.partition) + .05*state.ordinal,
                           12 + _unit(config, state, "unreachable")))
        q, region = None, "infeasible"
    # Independently generated measured starts and declared continuation posture;
    # target-generating q is never copied into a runtime input.
    if state.kind == "trajectory" and state.step:
        previous = _target_q(config, replace(state, step=state.step-1))
        current = _rounded((previous[0]+.007*math.sin(state.step), previous[1]+.004*math.cos(state.step)))
    else:
        current = _rounded((1.2*_unit(config, state, "current0")-.6,
                            .4*_unit(config, state, "current1")-.2))
    posture = current
    measured = (False, True) if state.stress_class == "missing_measurement_or_inactive_dof" else (True, True)
    observed = (None, current[1]) if not measured[0] else current
    # A different requested descriptor is an explicit unsupported-input case,
    # not a claim to have generated a valid closed-loop mechanism.
    profile_hash = (sha256_json({"unsupported_fixture": "closed-loop"})
                    if state.stress_class == "unsupported_profile_or_effector" else _profile_hash())
    malformed = state.stress_class == "nonfinite_or_malformed_input"
    inputs = RuntimeInputs(profile_hash, observed, target, posture, measured=measured,
                           task_mask=(False,False) if malformed else (True,True))
    invalid_specimen = None
    if malformed:
        # Preserve the actual original request, including isolated target/start
        # values. Four identical parser strings shared by all partitions would
        # be leaked duplicates regardless of their finite diagnostic envelope.
        specimen = inputs.to_dict()
        variant = state.ordinal%4
        if variant == 0:
            specimen["target_root"][0] = "__NONFINITE__"
            invalid_specimen = canonical_json(specimen).replace('"__NONFINITE__"','NaN')
        elif variant == 1:
            specimen["current"][0] = True
            invalid_specimen = canonical_json(specimen)
        elif variant == 2:
            invalid_specimen = '{"target_root":'+canonical_json(specimen["target_root"])+','+canonical_json(specimen)[1:]
        else:
            specimen["target_root"] = specimen["target_root"][:1]
            invalid_specimen = canonical_json(specimen)
    base = replace(state, step=0, augmentation=0)
    family_list = _partition_families(config, state.partition)
    # Disjoint seed families have many complete trajectories/point groups.
    family = family_list[int(sha256_json(base)[:8],16) % len(family_list)]
    trajectory = sha256_json([_config_hash(config), base])
    frame = (0., 0., 0.) if state.augmentation == 0 else ((10., 20., math.pi/2) if state.augmentation == 1 else (-7., 3., -.4))
    c, s = math.cos(frame[2]), math.sin(frame[2])
    original = _rounded((frame[0]+c*target[0]-s*target[1], frame[1]+s*target[0]+c*target[1]))
    stage = (0 if state.kind == "point" else 2 if state.kind == "trajectory" else
             1 if state.stress_class in BOUNDARY_CLASSES else 3)
    return GeometricTask(sha256_json([GENERATOR_VERSION,_config_hash(config),state]),
                         _config_hash(config), state, state.partition, family, trajectory,
                         inputs, q, original, frame, region, stage, invalid_specimen)


def iter_tasks(config: DatasetConfig, partition: str | None = None):
    if partition is not None and partition not in PARTITIONS:
        raise ContractError("ik_dataset_partition", "unknown partition")
    from .fresh_final import FreshFinalConfig
    if type(config) is FreshFinalConfig:
        if partition not in (None, "final"):
            raise ContractError("ik_fresh_partition", "fresh recipe is final-only")
        partition = "final"
    for stage in range(4):
        for p in PARTITIONS if partition is None else (partition,):
            allowed = _allowed_bins(config, p)
            if stage == 0:
                for b in allowed:
                    for n in range(config.points_per_bin[PARTITIONS.index(p)]):
                        yield replay_task(config, TaskState("point", p, b, n,augmentation=n%3))
            elif stage == 2:
                for n in range(config.trajectories_per_partition):
                    for step in range(config.trajectory_steps):
                        yield replay_task(config, TaskState("trajectory",p,allowed[n % len(allowed)],n,step,augmentation=n%3))
            else:
                for cls in STRESS_CLASSES:
                    if (cls in BOUNDARY_CLASSES) != (stage == 1):
                        continue
                    for n in range(config.stress_per_class):
                        yield replay_task(config, TaskState("stress",p,allowed[n % len(allowed)],n,stress_class=cls,augmentation=n%3))


def iter_task_batches(config: DatasetConfig, partition: str | None = None):
    batch = []
    for task in iter_tasks(config, partition):
        batch.append(task)
        if len(batch) == config.batch_size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def augment_task(task: GeometricTask, config: DatasetConfig, variant: int = 1):
    validate_task(task, config)
    return replay_task(config, replace(task.state, augmentation=variant))


def _task_matches_replay(task, replayed, config, *, _accepted_archive=False):
    if task == replayed:
        return True
    if (not _accepted_archive or not isinstance(task, GeometricTask)
            or not isinstance(replayed, GeometricTask)
            or platform.system() != "Windows" or platform.python_version() != "3.12.10"
            or config.sha256 != "fb8614412a62f7495519f463e3052ab060b0701701a3dfdb7265f0ad64488b51"):
        return False
    # Preserve identity, state, partition, order, groups, masks and all other
    # task fields. Only the inventoried target/original-point numeric replay
    # differences can match; neither the retained task nor its inputs change.
    if replace(task, inputs=replace(task.inputs, target_root=replayed.inputs.target_root),
               original_point=replayed.original_point) != replayed:
        return False
    entry = _windows_task_replay_inventory().get(task.id)
    return (entry is not None and sha256_json(task.to_dict()) == entry["retained_sha256"]
            and sha256_json(replayed.to_dict()) == entry["windows_replay_sha256"])


def validate_task(task: GeometricTask, config: DatasetConfig, *, _accepted_archive=False):
    if not isinstance(task, GeometricTask) or not _task_matches_replay(
            task, replay_task(config, task.state), config, _accepted_archive=_accepted_archive):
        raise ContractError("ik_dataset_replay", "task values, group, split or augmentation differ from frozen generator state")


def _two_vectors(chain):
    if not isinstance(chain, PlanarChain) or len(chain.joints) != 2 or any(j.kind != "rotary" for j in chain.joints):
        raise ContractError("ik_dataset_checker", "independent checker requires two rotary links")
    first, second = chain.joints
    def rotate(v, angle):
        c, s = math.cos(angle), math.sin(angle)
        return c*v[0]-s*v[1], s*v[0]+c*v[1]
    a = rotate((second.parent_anchor[0]-first.child_anchor[0], second.parent_anchor[1]-first.child_anchor[1]), first.zero_radians)
    b = rotate((chain.effector_local[0]-second.child_anchor[0], chain.effector_local[1]-second.child_anchor[1]), first.zero_radians+second.zero_radians)
    return first.parent_anchor, a, b


def independent_two_link_point(chain: PlanarChain, q):
    center, a, b = _two_vectors(chain)
    c, s = math.cos(q[0]), math.sin(q[0])
    cb, sb = math.cos(q[0]+q[1]), math.sin(q[0]+q[1])
    return (center[0]+c*a[0]-s*a[1]+cb*b[0]-sb*b[1],
            center[1]+s*a[0]+c*a[1]+sb*b[0]+cb*b[1])


def branch_identity(chain: PlanarChain, q):
    _, a, b = _two_vectors(chain)
    bend = q[1] + math.atan2(b[1],b[0]) - math.atan2(a[1],a[0])
    sine = math.sin(bend)
    return "singular" if abs(sine) <= 1e-8 else "positive" if sine > 0 else "negative"


@dataclass(frozen=True)
class TeacherLabel(Record):
    teacher: NumericSolution
    analytic: AnalyticSolution
    independent_endpoint: tuple[float, float] | None
    endpoint_error_cells: float | None
    branch_id: Literal["positive", "negative", "singular", "unavailable"]
    verified: bool
    pose_loss: bool
    endpoint_loss: bool
    status_loss: bool = True


def label_configuration(chain, target, current, *, solver_config=None, lower=None,
                        upper=None, posture=None, present=None, measured=None,
                        active=None, task_mask=(True, True)):
    """One teacher attempt, with independent status/geometry kept alongside it.

    Analytic answers never replace a failed numerical label. Timing and all
    projection/correction diagnostics remain in the unmodified teacher record.
    """
    solver_config = SolverConfig(tolerance_cells=1e-7) if solver_config is None else solver_config
    teacher = solve_numeric(chain,target,current,config=solver_config,lower=lower,upper=upper,
                            posture=posture,present=present,measured=measured,active=active,task_mask=task_mask)
    return _checked_label(chain,target,current,teacher,solver_config,lower=lower,upper=upper,
                          present=present,measured=measured,task_mask=task_mask)


def _checked_label(chain,target,current,teacher,solver_config,*,lower=None,upper=None,
                   present=None,measured=None,task_mask=(True,True)):
    analytic = analytic_two_link(chain,target,current,lower=lower,upper=upper)
    point, error, branch, verified = None, None, "unavailable", False
    if teacher.configuration is not None:
        try:
            point = independent_two_link_point(chain,teacher.configuration)
            error = math.hypot(*(t-p for t,p,enabled in zip(target,point,task_mask) if enabled))
            branch = branch_identity(chain,teacher.configuration)
            limits = list(zip(lower or tuple(j.lower for j in chain.joints),upper or tuple(j.upper for j in chain.joints)))
            verified = (all(a <= q <= b for q,(a,b) in zip(teacher.configuration,limits))
                        and teacher.point is not None and math.dist(point,teacher.point) <= 1e-8
                        and teacher.residual_cells is not None and abs(error-teacher.residual_cells) <= 1e-8)
        except (ContractError, TypeError, ValueError):
            verified = False
    pose_loss = (teacher.status == "solved" and verified and analytic.status == "solved"
                 and error <= solver_config.tolerance_cells + 1e-10
                 and any(branch == branch_identity(chain,q) and math.dist(q,teacher.configuration) <= 1e-4
                         for q in analytic.configurations))
    endpoint_loss = (analytic.status == "solved" and all(task_mask)
                     and (present is None or all(present)) and (measured is None or all(measured))
                     and teacher.status not in ("invalid_input","unsupported_profile","missing_required_measurement","nonfinite_result"))
    return TeacherLabel(teacher,analytic,point,error,branch,bool(verified),bool(pose_loss),bool(endpoint_loss))


@dataclass(frozen=True)
class GeometricSample(Record):
    task: GeometricTask
    label: TeacherLabel
    schema: Literal["nca-ik-geometric-sample/v1"] = "nca-ik-geometric-sample/v1"


def generate_sample(task: GeometricTask, config: DatasetConfig):
    validate_task(task,config)
    if task.invalid_specimen is not None:
        return GeometricSample(task,_malformed_label(task.invalid_specimen))
    inputs = task.inputs
    solver = replace(config.teacher,max_iterations=0) if task.state.stress_class == "iteration_exhaustion" else config.teacher
    # Missing coordinates remain missing at the validation boundary; zero is
    # passed only behind the false measurement mask, which the solver refuses.
    current = tuple(0. if q is None else q for q in inputs.current)
    chain = _chain() if inputs.profile_sha256 == _profile_hash() else None
    label = label_configuration(chain,inputs.target_root,current,solver_config=solver,
                                lower=(-.72,-.27),upper=(.72,.27),posture=inputs.posture,
                                present=inputs.present,measured=inputs.measured,active=inputs.active,
                                task_mask=inputs.task_mask)
    return GeometricSample(task,label)


def _malformed_label(specimen):
    try:
        RuntimeInputs.from_dict(strict_json_loads(specimen))
    except ContractError as exc:
        return TeacherLabel(NumericSolution("invalid_input",exc.code),
                            AnalyticSolution("invalid_input",reason="parser_refused_original_specimen"),
                            None,None,"unavailable",False,False,False)
    raise ContractError("ik_dataset_specimen", "malformed specimen unexpectedly admitted")


_WINDOWS_REPLAY_SHA256 = "ba42f53d140543713e68653f22540cd1cd4ed552252d7b41ce967c0c3f04a39c"
_WINDOWS_TASK_REPLAY_SHA256 = "68682fd6222865f08278c09d3e5e67da5d09b57eeb92d3c5ded87109daa2612b"


@lru_cache(maxsize=1)
def _windows_task_replay_inventory():
    raw = Path(__file__).with_name("accepted_windows_task_replay_v1.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != _WINDOWS_TASK_REPLAY_SHA256:
        raise ContractError("ik_dataset_compatibility", "Windows task replay inventory bytes changed")
    return strict_json_loads(raw, max_bytes=16384)["entries"]


@lru_cache(maxsize=1)
def _windows_replay_inventory():
    """Hash-only compatibility metadata; never contains replacement labels."""
    raw = Path(__file__).with_name("accepted_windows_replay_v1.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != _WINDOWS_REPLAY_SHA256:
        raise ContractError("ik_dataset_compatibility", "Windows replay inventory bytes changed")
    return strict_json_loads(raw, max_bytes=262144)["entries"]


def _authenticated_windows_label_replay(sample_id, checked, retained):
    """Accept only enumerated exact original/Windows label digest pairs.

    Original manifest, registry and shard bytes must already be authenticated.
    No numeric tolerance is granted. Teacher records, branch/status decisions,
    proof metadata, configuration counts and supervision masks remain exact.
    Task replay, splits and normalization retain their independent strict gates.
    """
    if platform.system() != "Windows" or platform.python_version() != "3.12.10":
        return False
    current, original = checked.analytic, retained.analytic
    if (len(current.configurations) != len(original.configurations)
            or len(current.points) != len(original.points)
            or replace(checked, analytic=replace(current,
                       configurations=original.configurations, points=original.points),
                       independent_endpoint=retained.independent_endpoint,
                       endpoint_error_cells=retained.endpoint_error_cells) != retained):
        return False
    entry = _windows_replay_inventory().get(sample_id)
    return (entry is not None and sha256_json(retained.to_dict()) == entry["retained_sha256"]
            and sha256_json(checked.to_dict()) == entry["windows_replay_sha256"])


def validate_sample(sample: GeometricSample, config: DatasetConfig, *, _accepted_archive=False):
    """Check actual retained outcomes without spending another teacher attempt."""
    if not isinstance(sample, GeometricSample):
        raise ContractError("ik_dataset_sample", "typed sample required")
    task, label = sample.task, sample.label
    validate_task(task,config, _accepted_archive=_accepted_archive)
    if task.invalid_specimen is not None:
        if label != _malformed_label(task.invalid_specimen):
            raise ContractError("ik_dataset_label", "original malformed specimen refusal changed")
        return
    inputs, teacher = task.inputs, label.teacher
    solver = replace(config.teacher,max_iterations=0) if task.state.stress_class == "iteration_exhaustion" else config.teacher
    if (not 0 <= teacher.iterations <= solver.max_iterations or teacher.evaluations < 0
            or teacher.elapsed_seconds < 0 or teacher.projection_events < 0
            or not 0 <= teacher.maximum_step_norm <= solver.max_step+1e-10):
        raise ContractError("ik_dataset_teacher", "teacher accounting exceeds declared bounds")
    if teacher.status == "iteration_budget_exhausted" and teacher.iterations != solver.max_iterations:
        raise ContractError("ik_dataset_teacher", "iteration exhaustion must consume the exact per-call ceiling")
    if teacher.status in ("best_effort","iteration_budget_exhausted") and (
            teacher.residual_cells is None or teacher.residual_cells <= solver.tolerance_cells):
        raise ContractError("ik_dataset_teacher", "unsolved status contradicts a converged or absent residual")
    if teacher.status == "time_budget_exhausted" and teacher.elapsed_seconds < solver.max_seconds:
        raise ContractError("ik_dataset_teacher", "time exhaustion precedes the per-call wall ceiling")
    if teacher.status in ("invalid_input","unsupported_profile","missing_required_measurement") and (
            teacher.configuration is not None or teacher.point is not None or teacher.residual_cells is not None
            or teacher.iterations or teacher.evaluations or teacher.projection_events):
        raise ContractError("ik_dataset_teacher", "rejected input cannot acquire a candidate or numerical work")
    if task.state.stress_class == "missing_measurement_or_inactive_dof" and teacher.status != "missing_required_measurement":
        raise ContractError("ik_dataset_teacher", "missing measurement must remain a refusal")
    if task.state.stress_class == "unsupported_profile_or_effector" and teacher.status != "unsupported_profile":
        raise ContractError("ik_dataset_teacher", "unsupported descriptor must remain a refusal")
    current = tuple(0. if q is None else q for q in inputs.current)
    chain = _chain() if inputs.profile_sha256 == _profile_hash() else None
    checked = _checked_label(chain,inputs.target_root,current,teacher,solver,
                              lower=(-.72,-.27),upper=(.72,.27),present=inputs.present,
                              measured=inputs.measured,task_mask=inputs.task_mask)
    if checked != label and not (_accepted_archive
                                 and _authenticated_windows_label_replay(task.id, checked, label)):
        raise ContractError("ik_dataset_label", "label differs from independent retained-result checks")
    if teacher.status == "solved" and not label.pose_loss:
        raise ContractError("ik_dataset_label", "solved label failed independent branch/endpoint/limit checks")
    if teacher.configuration is not None and not label.verified:
        raise ContractError("ik_dataset_label", "retained candidate has invalid endpoint/limit accounting")
    # Independently undo the original root frame; canonical-target identity must
    # survive all declared frame augmentations before any split validation.
    x,y,angle = task.root_pose
    dx,dy = task.original_point[0]-x, task.original_point[1]-y
    restored = (math.cos(angle)*dx+math.sin(angle)*dy, -math.sin(angle)*dx+math.cos(angle)*dy)
    if math.dist(restored,inputs.target_root) > 1e-9:
        raise ContractError("ik_dataset_frame", "original and canonical target disagree")
    if task.target_configuration is not None and math.dist(
            independent_two_link_point(_chain(),task.target_configuration),inputs.target_root) > 1e-8:
        raise ContractError("ik_dataset_endpoint", "generated legal endpoint failed independent check")


def _point_segment(p,a,b):
    dx,dy = b[0]-a[0], b[1]-a[1]
    norm = dx*dx+dy*dy
    u = 0. if norm == 0 else max(0.,min(1.,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/norm))
    return math.hypot(p[0]-a[0]-u*dx,p[1]-a[1]-u*dy)


def _segment_distance(a,b,c,d):
    u,v,w = (b[0]-a[0],b[1]-a[1]),(d[0]-c[0],d[1]-c[1]),(c[0]-a[0],c[1]-a[1])
    cross = u[0]*v[1]-u[1]*v[0]
    if cross:
        s,t = (w[0]*v[1]-w[1]*v[0])/cross,(w[0]*u[1]-w[1]*u[0])/cross
        if 0 <= s <= 1 and 0 <= t <= 1:
            return 0.
    return min(_point_segment(a,c,d),_point_segment(b,c,d),_point_segment(c,a,b),_point_segment(d,a,b))


def validate_dataset(samples, config: DatasetConfig, *, require_complete=False, _accepted_archive=False):
    """Reject renamed copies, family/trajectory splits and near path leakage.

    Segment checks cover the continuous piecewise-linear root-local target path,
    including a trajectory that crosses another partition between its samples.
    A 0.05-cell broad-phase grid limits comparisons without weakening guards.
    """
    samples = tuple(samples)
    ids, content, groups, trajectories = set(), {}, {}, defaultdict(dict)
    segments = []
    for sample in samples:
        validate_sample(sample,config, _accepted_archive=_accepted_archive)
        task = sample.task
        if task.id in ids:
            raise ContractError("ik_dataset_duplicate", "duplicate sample ID")
        ids.add(task.id)
        previous = content.get(task.input_sha256)
        if previous is not None:
            if previous.task.partition != task.partition:
                raise ContractError("ik_dataset_leakage", "identical permitted inputs occur in different splits")
            if previous.label.pose_loss and sample.label.pose_loss and math.dist(
                    previous.label.teacher.configuration,sample.label.teacher.configuration) > 1e-5:
                raise ContractError("ik_dataset_ambiguous_label", "one permitted input has contradictory pose labels")
        content[task.input_sha256] = sample
        for key in (task.family_id,task.trajectory_id):
            if key in groups and groups[key] != task.partition:
                raise ContractError("ik_dataset_leakage", "family or complete trajectory appears in two splits")
            groups[key] = task.partition
        point = task.inputs.target_root
        segments.append((point,point,task.partition))
        if task.state.kind == "trajectory":
            steps = trajectories[task.trajectory_id]
            if task.state.step in steps and steps[task.state.step].inputs != task.inputs:
                raise ContractError("ik_dataset_trajectory", "one trajectory step has contradictory inputs")
            steps[task.state.step] = task
    for steps in trajectories.values():
        for index,task in steps.items():
            if index+1 in steps:
                segments.append((task.inputs.target_root,steps[index+1].inputs.target_root,task.partition))
        if require_complete and set(steps) != set(range(config.trajectory_steps)):
            raise ContractError("ik_dataset_incomplete", "complete trajectories must retain every decision")
    grid,stored = defaultdict(list),[]
    guard,cell = config.point_guard_cells,.05
    for a,b,partition in segments:
        cells = [(x,y) for x in range(math.floor((min(a[0],b[0])-guard)/cell),math.floor((max(a[0],b[0])+guard)/cell)+1)
                 for y in range(math.floor((min(a[1],b[1])-guard)/cell),math.floor((max(a[1],b[1])+guard)/cell)+1)]
        visited = set()
        for key in cells:
            for index in grid[key]:
                if index in visited:
                    continue
                visited.add(index)
                c,d,other = stored[index]
                if partition != other and _segment_distance(a,b,c,d) <= guard:
                    raise ContractError("ik_dataset_leakage", "cross-partition point or trajectory enters the spatial guard")
        index = len(stored)
        stored.append((a,b,partition))
        for key in cells:
            grid[key].append(index)
    if require_complete:
        expected = {task.id for task in iter_tasks(config)}
        if ids != expected:
            raise ContractError("ik_dataset_incomplete", "sample set differs from the complete declared generator schedule")


@dataclass(frozen=True)
class Normalization(Record):
    count: int
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    training_inputs_sha256: str
    feature_names: tuple[str, ...] = FEATURE_NAMES
    partition: Literal["training"] = "training"
    schema: Literal["nca-ik-train-normalization/v1"] = "nca-ik-train-normalization/v1"

    def validate(self):
        digest(self.training_inputs_sha256,"normalization.training_inputs")
        if self.count <= 0 or self.feature_names != FEATURE_NAMES or len(self.mean) != len(FEATURE_NAMES) or len(self.scale) != len(FEATURE_NAMES):
            raise ContractError("ik_dataset_normalization", "exact feature schema and nonempty training fit required")
        if any(v <= 0 for v in self.scale):
            raise ContractError("ik_dataset_normalization", "positive finite scales required")

    def apply(self,inputs: RuntimeInputs):
        return tuple((v-m)/s for v,m,s in zip(runtime_features(inputs),self.mean,self.scale))


def fit_normalization(samples, config: DatasetConfig):
    """Fit only scheduled valid training inputs, regardless of teacher success.

    Inner-limit inputs participate; explicit invalid/infeasible/exhaustion stress
    classes do not. This rule is fixed by task class, never by teacher outcome.
    """
    selected,seen = [],set()
    for sample in samples:
        if not isinstance(sample,GeometricSample) or sample.task.partition != "training":
            raise ContractError("ik_dataset_normalization_leakage", "only training samples may fit normalization")
        validate_sample(sample,config)
        if sample.task.id in seen:
            raise ContractError("ik_dataset_duplicate", "normalizer cannot reweight duplicate sample IDs")
        seen.add(sample.task.id)
        if sample.task.state.kind != "stress" or sample.task.state.stress_class == "inner_limit_boundary":
            selected.append(sample.task)
    if not selected:
        raise ContractError("ik_dataset_normalization", "no eligible training inputs")
    selected.sort(key=lambda t:t.id)
    rows = [runtime_features(t.inputs) for t in selected]
    mean = tuple(math.fsum(row[i] for row in rows)/len(rows) for i in range(len(FEATURE_NAMES)))
    variance = tuple(math.fsum((row[i]-mean[i])**2 for row in rows)/len(rows) for i in range(len(FEATURE_NAMES)))
    scale = tuple(math.sqrt(v) if v > 1e-24 else 1. for v in variance)
    return Normalization(len(rows),mean,scale,sha256_json([[t.id,t.input_sha256] for t in selected]))


def coverage_report(samples, config: DatasetConfig):
    samples = tuple(samples)
    partitions = {}
    for partition in PARTITIONS:
        rows = [s for s in samples if s.task.partition == partition]
        points = [s for s in rows if s.task.state.kind == "point"]
        nominal = [s for s in rows if s.task.state.kind != "stress"]
        trajectories = defaultdict(set)
        for row in rows:
            if row.task.state.kind == "trajectory":
                trajectories[row.task.trajectory_id].add(row.task.state.step)
        partitions[partition] = {
            "scheduled_samples": len(rows), "nominal_denominator": len(nominal),
            "point_queries":len(points), "point_grid_counts":[sum(s.task.state.bin_index == b for s in points) for b in range(80)],
            "withheld_point_queries":sum(s.task.region_kind == "withheld_region" for s in points),
            "complete_trajectories":sum(steps == set(range(config.trajectory_steps)) for steps in trajectories.values()),
            "trajectory_steps":sum(len(steps) for steps in trajectories.values()),
            "families_used":len({s.task.family_id for s in rows}),
            "stress_populations":dict(Counter(s.task.state.stress_class for s in rows if s.task.state.kind == "stress")),
            "teacher_status_counts":dict(Counter(s.label.teacher.status for s in rows)),
            "analytic_status_counts":dict(Counter(s.label.analytic.status for s in rows)),
            "region_counts":dict(Counter(s.task.region_kind for s in rows)),
            "strata":{
                "interior_points":len(points),
                "rest_neighbourhood":sum(s.task.target_configuration is not None and math.hypot(*s.task.target_configuration) <= .05 for s in nominal),
                "sealed_target_regions":sum(s.task.region_kind == "withheld_region" for s in nominal),
                "smooth_sweeps":len({s.task.trajectory_id for s in nominal if s.task.state.kind == "trajectory" and s.task.state.ordinal%2}),
                "reversals":len({s.task.trajectory_id for s in nominal if s.task.state.kind == "trajectory" and not s.task.state.ordinal%2}),
                "root_frame_transforms":sum(s.task.state.augmentation != 0 for s in nominal)},
            "stage_counts":dict(Counter(CURRICULUM[s.task.stage] for s in rows)),
            "branch_counts":dict(Counter(s.label.branch_id for s in rows)),
            "pose_loss_samples":sum(s.label.pose_loss for s in rows),
            "endpoint_loss_samples":sum(s.label.endpoint_loss for s in rows),
            "status_loss_samples":sum(s.label.status_loss for s in rows),
            "teacher_failure_denominator":len(rows),
            "retained_teacher_failures":sum(s.label.teacher.status != "solved" for s in rows),
            "projection_events":sum(s.label.teacher.projection_events for s in rows),
            "teacher_iterations":sum(s.label.teacher.iterations for s in rows),
            "max_independently_checked_solved_error_cells":max((s.label.endpoint_error_cells for s in rows if s.label.pose_loss),default=None),
        }
    final = partitions["final"]
    sufficient = (config.mode == "prospective" and len(samples) == config.expected_tasks
                  and len({s.task.family_id for s in samples}) == 100
                  and final["point_queries"] >= 2000 and min(final["point_grid_counts"]) >= 25
                  and final["withheld_point_queries"] >= 200 and final["complete_trajectories"] >= 40
                  and config.trajectory_steps >= 20 and all(final["stress_populations"].get(c,0) >= 20 for c in STRESS_CLASSES)
                  and all(final["strata"].values()))
    return {"schema":"nca-ik-dataset-coverage/v1","evidence_kind":"nominal_geometric_data_quality_only",
            "config_sha256":_config_hash(config),"expected_samples":config.expected_tasks,"observed_samples":len(samples),
            "generation_families_used":len({s.task.family_id for s in samples}),
            "partitions":partitions,"accepted_final_population_minima_met":sufficient,
            "runner_singular_cases":0,"runner_multiple_branch_cases":0,
            "singular_branch_reason":"no singularity or second branch in accepted Runner domain; separate synthetic development fixtures",
            "failures_retained":True,"training_performed":False,"cuda_qualified":False,"physical_qualified":False,
            "final_scoring_opened":False}
