"""Small feedback skills for the accepted Vector launcher and Hammerhead cutter.

Only wire requests leave this module. Body integration, mount attachment,
clearance, resource spending, recoil and material effects remain in Cybersand.
Pouncer's existing charge-jump is a separate cancellation conformance example.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Literal

from nca.contracts import (ContractError, Record, WireAction, digest, identifier,
                           sha256_json)

PROFILE_HASHES = {
    "astra": "28b3665b5ecb695aa58e10a557af79f499f927152fe233c8abe74d4867ab71b0",
    "hammerhead": "692cd3d13e988f8a22daf10fb375f18406aa059c5a9fba574bcffa2442dfaa3c",
    "pouncer": "3b53852093330e41a3015d35045627bbf3542f4db4b477ff9daa08cbc5b7fd45",
}
HOST_REVISION = "f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497"
PLAN_AGE_TICKS = 120
REQUEST_CAPACITY = 512
TRACE_CAPACITY = 2048
TERMINAL = {"succeeded", "failed", "timed_out", "cancelled", "invalidated"}


def _bounded(value: float, lo: float, hi: float, name: str):
    if not lo <= value <= hi:
        raise ContractError("out_of_range", name)


def _timing(timeout: int, no_progress: int):
    _bounded(timeout, 6, 600, "timeout_ticks")
    _bounded(no_progress, 6, min(120, timeout), "no_progress_ticks")


@dataclass(frozen=True)
class MoveParameters(Record):
    target: tuple[float, float]
    tolerance: float = 3.0
    effort: float = 0.7
    timeout_ticks: int = 180
    no_progress_ticks: int = 60

    def validate(self):
        if not 128 <= self.target[0] < 640 or not 128 <= self.target[1] < 384:
            raise ContractError("target_bounds", "target must be a point in the authored arena")
        _bounded(self.tolerance, 0.5, 8, "tolerance")
        _bounded(self.effort, 0.15, 1, "effort")
        _timing(self.timeout_ticks, self.no_progress_ticks)


@dataclass(frozen=True)
class GripParameters(Record):
    effort: float = 0.6
    timeout_ticks: int = 120
    no_progress_ticks: int = 36

    def validate(self):
        _bounded(self.effort, 0.15, 1, "effort")
        _timing(self.timeout_ticks, self.no_progress_ticks)


@dataclass(frozen=True)
class UseParameters(Record):
    aim: tuple[float, float] = (0, -1)
    effort: float = 0.7
    settle_ticks: int = 6
    timeout_ticks: int = 90
    no_progress_ticks: int = 36
    require_change: bool = False

    def validate(self):
        _bounded(math.hypot(*self.aim), 0.001, 1.000000001, "aim norm")
        _bounded(self.effort, 0.15, 1, "effort")
        _timing(self.timeout_ticks, self.no_progress_ticks)
        _bounded(self.settle_ticks, 1, min(60, self.timeout_ticks), "settle_ticks")


@dataclass(frozen=True)
class RecoverParameters(Record):
    effort: float = 0.7
    angle_tolerance: float = 0.3
    angular_speed_tolerance: float = 0.8
    settle_ticks: int = 12
    timeout_ticks: int = 120
    no_progress_ticks: int = 60

    def validate(self):
        _bounded(self.effort, 0.15, 1, "effort")
        _bounded(self.angle_tolerance, 0.05, 0.8, "angle_tolerance")
        _bounded(self.angular_speed_tolerance, 0.05, 3, "angular_speed_tolerance")
        _timing(self.timeout_ticks, self.no_progress_ticks)
        _bounded(self.settle_ticks, 1, min(60, self.timeout_ticks), "settle_ticks")


@dataclass(frozen=True)
class ReleaseParameters(Record):
    timeout_ticks: int = 30
    no_progress_ticks: int = 12

    def validate(self):
        _timing(self.timeout_ticks, self.no_progress_ticks)


@dataclass(frozen=True)
class ChargeParameters(Record):
    target_charge: float = 0.4
    effort: float = 0.7
    timeout_ticks: int = 120
    no_progress_ticks: int = 60

    def validate(self):
        _bounded(self.target_charge, 0.05, 0.9, "target_charge")
        _bounded(self.effort, 0.15, 1, "effort")
        _timing(self.timeout_ticks, self.no_progress_ticks)


PARAMETERS = {"approach": MoveParameters, "carry": MoveParameters,
              "grip": GripParameters, "aim_use": UseParameters,
              "recover": RecoverParameters, "release": ReleaseParameters,
              "charge_hold": ChargeParameters}


def validate_parameters(skill: str, parameters: Mapping[str, Any]) -> None:
    """Shared strict callback for human shortcuts, scripts and decision providers."""
    if skill not in PARAMETERS:
        raise ContractError("unknown_skill", str(skill))
    PARAMETERS[skill].from_dict(parameters)


def available_skills(model: str) -> tuple[str, ...]:
    if model == "astra":
        return "approach", "aim_use", "recover"
    if model == "hammerhead":
        return "approach", "grip", "aim_use", "carry", "recover", "release"
    if model == "pouncer":
        return ("charge_hold",)
    return ()


@dataclass(frozen=True)
class UsableCapability(Record):
    model: Literal["astra", "hammerhead", "pouncer"]
    profile_sha256: str
    part_ids: tuple[str, ...]
    mount_id: str
    mount_part: int
    held_part: int | None
    offset: tuple[float, float]
    limits: tuple[float, float]
    usable: str
    source_revision: Literal[HOST_REVISION] = HOST_REVISION
    schema_version: Literal[1] = 1

    def validate(self):
        digest(self.profile_sha256, "profile_sha256")
        if self.profile_sha256 != PROFILE_HASHES[self.model]:
            raise ContractError("profile_mismatch", "only the pinned authored capability is qualified")
        expected = {
            "astra": (("chassis", "rotor_1", "rotor_2"), None, (3.5, 1.0), "launcher"),
            "hammerhead": (("chassis", "shoe_-1", "shoe_1", "hammer_arm", "hammer"), 4, (0.0, -1.0), "cutter"),
            "pouncer": (("chassis", "shoe_-1", "shoe_1", "glide_fin_-1", "glide_fin_1"), None, (4.0, 0.0), "depositor"),
        }[self.model]
        if ((self.part_ids, self.held_part, self.offset, self.usable) != expected
                or self.mount_id != "usable" or self.mount_part != 0 or self.limits != (-2.5, 2.5)):
            raise ContractError("capability_mismatch", "mount/body mapping differs from the audited host")

    @classmethod
    def from_profile(cls, profile: Mapping[str, Any]):
        model = profile.get("id")
        if model not in PROFILE_HASHES or profile.get("version") != 1:
            raise ContractError("unsupported_profile", "selected usable profiles only")
        actual = sha256_json(profile)
        if actual != PROFILE_HASHES[model]:
            raise ContractError("profile_mismatch", "authored source data changed")
        mounts = profile["mounts"]
        if len(mounts) != 1:
            raise ContractError("mount_count", "only the selected slot-zero mount is supported")
        mount = mounts[0]
        return cls(model, actual, tuple(part["id"] for part in profile["parts"]),
                   mount["id"], mount["part"], mount.get("held_part"), tuple(mount["offset"]),
                   tuple(mount["limits"]), mount["usable"])


@dataclass(frozen=True)
class ActorIdentity(Record):
    run_id: str
    actor_id: int
    epoch: int
    model: str
    profile_version: Literal[1]
    control_generation: int

    def validate(self):
        identifier(self.run_id, "run_id")
        if self.model not in PROFILE_HASHES or not 0 <= self.actor_id < 8 or self.epoch < 1 or self.control_generation < 0:
            raise ContractError("actor_identity", "invalid selected actor lifetime")

    @classmethod
    def from_view(cls, view: Mapping[str, Any]):
        if (type(view.get("version")) is not int or view["version"] != 2 or view.get("representation") != "articulated-rapier"
                or view.get("observation_scope") != "self, rig rays and bounded composed native cells"):
            raise ContractError("observation_scope", "requires an ACT004 issued local observation")
        value = cls(*(view[key] for key in ("run_id", "actor_id", "epoch", "model", "profile_version", "control_generation")))
        tick = view.get("tick")
        if type(tick) is not int or tick < 0 or view.get("observation_id") != value.observation_id(tick):
            raise ContractError("observation_identity", "local view tick and issued identity disagree")
        return value

    def observation_id(self, tick: int) -> str:
        return f"{self.run_id}:{self.epoch}:{self.actor_id}:{self.control_generation}:{tick}"


@dataclass(frozen=True)
class MountReference(Record):
    actor: ActorIdentity
    profile_sha256: str
    mount_id: str
    attachment_generation: int
    object_part: str
    slot: Literal[0] = 0
    schema_version: Literal[1] = 1

    def validate(self):
        digest(self.profile_sha256, "profile_sha256")
        identifier(self.mount_id, "mount_id")
        identifier(self.object_part, "object_part")
        if self.attachment_generation < 0:
            raise ContractError("mount_generation", "attachment generation cannot be negative")


@dataclass(frozen=True)
class SkillRequest(Record):
    id: str
    skill: str
    parameters: Mapping[str, Any]
    actor: ActorIdentity
    mount: MountReference
    source: Literal["human", "script", "model"]
    source_generation: int
    source_observation_id: str
    source_tick: int
    schema_version: Literal[1] = 1

    def validate(self):
        identifier(self.id, "skill_request.id")
        validate_parameters(self.skill, self.parameters)
        if self.source_generation < 1 or self.source_tick < 0:
            raise ContractError("skill_source", "source generation and tick must be valid")
        if self.source_observation_id != self.actor.observation_id(self.source_tick):
            raise ContractError("skill_source", "source observation must belong to its declared actor lifetime")


@dataclass(frozen=True)
class SkillInitiation(Record):
    request: SkillRequest
    accepted: bool
    reason: str
    state: str
    schema_version: Literal[1] = 1


@dataclass(frozen=True)
class EffectResult(Record):
    tick: int
    actor_slot: int
    epoch: int
    sequence: int
    consumed: bool
    status: Literal["applied", "no_change", "refused"]
    reason: str
    changed_cells: int
    attempted_writes: int
    reads: int

    def validate(self):
        if (self.tick < 0 or not 0 <= self.actor_slot < 8 or self.epoch < 1 or self.sequence < 1
                or min(self.changed_cells, self.attempted_writes, self.reads) < 0):
            raise ContractError("effect_receipt", "invalid effect identity or counters")
        if self.status == "applied" and (not self.consumed or self.changed_cells <= 0):
            raise ContractError("effect_receipt", "applied effect requires actual changed cells")
        if self.status == "no_change" and (not self.consumed or self.changed_cells or self.reason):
            raise ContractError("effect_receipt", "no-change receipt semantics disagree")


@dataclass(frozen=True)
class SkillProgress(Record):
    request_id: str | None
    tick: int
    status: Literal["idle", "running", "succeeded", "failed", "timed_out", "cancelled", "invalidated"]
    phase: str
    reason: str
    action: WireAction = field(default_factory=WireAction)
    local_success: bool | None = None
    evaluator_success: None = None
    effects: tuple[Mapping[str, Any], ...] = ()
    distance: float | None = None
    cancel_host: bool = False
    schema_version: Literal[1] = 1

    def validate(self):
        if self.tick < 0 or (self.local_success is not None and self.status not in TERMINAL):
            raise ContractError("skill_progress", "success requires a terminal local predicate")
        if self.status == "succeeded" and self.local_success is not True:
            raise ContractError("skill_progress", "success requires positive local feedback")

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ContractError("local_observation", name)
    return float(value[0]), float(value[1])


def _clamp(value: float, limit: float = 1.0) -> float:
    return max(-limit, min(limit, value))


def _angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def interval_observations(info: Mapping[str, Any]) -> tuple[dict, ...]:
    """Extract only already-permitted local observations from completed receipts.

    A missing interval is a data gap, not permission to infer a successful effect.
    No reward, global snapshot, target oracle or privileged evaluator enters a skill.
    """
    if info.get("consumed_trace_complete") is False:
        raise ContractError("feedback_gap", "the completed local feedback interval is incomplete")
    result = []
    for trace in (*info.get("idle_trace", ()), *info.get("trace", ())):
        view = trace["observation"]
        ActorIdentity.from_view(view)
        from nca.contracts import canonical_json, strict_json_loads
        result.append(strict_json_loads(canonical_json(view)))
    return tuple(result)


class SkillRuntime:
    """One actor, one resource owner and one bounded skill at a time.

    `start` admits a plan; only `progress` proposes wire input. The outer owner
    must call its real cancellation API whenever `cancel_host` is true, including
    successful completion. That revokes pending input/charge without resetting
    bodies, detaching equipment or undoing already-applied native effects.
    """

    def __init__(self, profile: Mapping[str, Any] | UsableCapability):
        self.capability = profile if isinstance(profile, UsableCapability) else UsableCapability.from_profile(profile)
        self.allowed_skills = available_skills(self.capability.model)
        self.mount_generation = 0
        self._identity: ActorIdentity | None = None
        self._detached: bool | None = None
        self._serial = 0
        self._request: SkillRequest | None = None
        self._receipts: dict[str, SkillInitiation] = {}
        self._latest = SkillProgress(None, 0, "idle", "idle", "no_skill")
        self._started = 0
        self._phase = "idle"
        self._phase_tick = 0
        self._best = math.inf
        self._best_tick = 0
        self._quiet_since: int | None = None
        self._trigger_tick: int | None = None
        self._effect_sequence = -1
        self._effects: dict[tuple[int, int], Mapping[str, Any]] = {}
        self._last_feedback_tick = -1
        self._last_progress_tick = -1
        self._baseline = None
        self.events: list[dict] = []
        self.dropped_events = 0

    def _event(self, event: str, **value):
        if len(self.events) >= TRACE_CAPACITY:
            del self.events[0]
            self.dropped_events += 1
        self.events.append({"schema_version": 1, "event": event, **value})

    def _validate_view(self, view: Mapping[str, Any]) -> ActorIdentity:
        identity = ActorIdentity.from_view(view)
        if identity.model != self.capability.model:
            raise ContractError("profile_mismatch", "runtime is bound to a selected authored model")
        parts = view.get("parts")
        if not isinstance(parts, (tuple, list)) or tuple(part.get("id") for part in parts) != self.capability.part_ids:
            raise ContractError("morphology_changed", "part identity differs from the pinned body graph")
        for index, part in enumerate(parts):
            if type(part.get("index")) is not int or part["index"] != index or type(part.get("detached")) is not bool:
                raise ContractError("part_identity", "invalid part index or attachment state")
            _pair(part.get("position"), "part.position")
            if type(part.get("rotation")) not in (int, float) or not math.isfinite(part["rotation"]):
                raise ContractError("local_observation", "part.rotation")
        _pair(view.get("position"), "position")
        _pair(view.get("velocity"), "velocity")
        for key in ("rotation", "angular_velocity", "energy", "charge"):
            if type(view.get(key)) not in (int, float) or not math.isfinite(view[key]):
                raise ContractError("local_observation", key)
        return identity

    def mount_ref(self, view: Mapping[str, Any]) -> MountReference:
        identity = self._validate_view(view)
        held = self.capability.held_part
        detached = bool(view["parts"][held]["detached"]) if held is not None else False
        # Attachment is an observable connection generation, not an invented
        # independently replaceable host object ID. The existing body survives it.
        same_object = self._identity is not None and all(
            getattr(self._identity, key) == getattr(identity, key)
            for key in ("run_id", "actor_id", "epoch", "model", "profile_version"))
        if not same_object:
            self._detached = detached
            self.mount_generation = 0
        elif detached != self._detached:
            self._detached = detached
            self.mount_generation += 1
        # Owner cancellation changes control authority, not the held body's
        # connection history. Retain its generation across that change while
        # the complete actor identity still fences every old source request.
        self._identity = identity
        part = held if held is not None else self.capability.mount_part
        return MountReference(identity, self.capability.profile_sha256, self.capability.mount_id,
                              self.mount_generation, self.capability.part_ids[part])

    def start(self, skill: str, parameters: Mapping[str, Any], view: Mapping[str, Any], *,
              source: Literal["human", "script", "model"] = "script", request_id: str | None = None,
              source_generation: int = 1, source_observation_id: str | None = None,
              source_tick: int | None = None, mount_ref: MountReference | None = None) -> SkillInitiation:
        validate_parameters(skill, parameters)
        if request_id in self._receipts:
            old = self._receipts[request_id]
            original = old.request
            if (skill != original.skill or sha256_json(parameters) != sha256_json(original.parameters)
                    or source != original.source or source_generation != original.source_generation
                    or (source_observation_id is not None and source_observation_id != original.source_observation_id)
                    or (source_tick is not None and source_tick != original.source_tick)
                    or (mount_ref is not None and mount_ref != original.mount)):
                raise ContractError("request_collision", "same invocation ID cannot mean a new payload")
            return old
        current = self.mount_ref(view)
        tick = view["tick"]
        self._serial += 1
        request = SkillRequest(request_id or f"skill-{self._serial}", skill, parameters, current.actor,
                               current if mount_ref is None else mount_ref, source, source_generation,
                               view["observation_id"] if source_observation_id is None else source_observation_id,
                               tick if source_tick is None else source_tick)
        reason = ""
        if len(self._receipts) >= REQUEST_CAPACITY:
            reason = "request_capacity"
        elif skill not in self.allowed_skills:
            reason = "unsupported_capability"
        elif self._request is not None and not self._latest.terminal:
            reason = "resource_busy"
        elif view.get("suspended") or view.get("fault"):
            reason = "actor_not_controllable"
        elif request.mount != current:
            reason = "stale_mount_reference"
        elif not 0 <= tick - request.source_tick < PLAN_AGE_TICKS:
            reason = "source_observation_expired"
        elif skill in ("carry", "aim_use") and self.capability.held_part is not None and self._detached:
            reason = "usable_detached"
        receipt = SkillInitiation(request, not reason, reason or "accepted_without_execution", "refused" if reason else "accepted")
        if len(self._receipts) < REQUEST_CAPACITY:
            self._receipts[request.id] = receipt
        self._event("initiation", tick=tick, receipt=receipt.to_dict())
        if reason:
            return receipt
        self._request = request
        self._started = self._phase_tick = self._best_tick = tick
        self._phase = "initiated"
        self._best = math.inf
        self._quiet_since = self._trigger_tick = None
        self._effects = {}
        self._effect_sequence = max((r.get("sequence", -1) for r in view.get("last_effects", ())), default=-1)
        self._last_feedback_tick = tick
        self._last_progress_tick = -1
        self._baseline = None
        self._latest = SkillProgress(request.id, tick, "running", "initiated", "accepted_without_execution")
        return receipt

    def _result(self, view: Mapping[str, Any], *, status: str = "running", reason: str = "feedback_control",
                action: WireAction | None = None, distance: float | None = None) -> SkillProgress:
        terminal = status in TERMINAL
        value = SkillProgress(self._request.id if self._request else None, view["tick"], status,
                              self._phase, reason, action or WireAction(),
                              status == "succeeded" if terminal else None, None,
                              tuple(self._effects.values()), distance, terminal)
        self._latest = value
        self._event("progress", result=value.to_dict())
        return value

    def cancel(self, reason: str = "cancelled") -> SkillProgress:
        """Revoke proposed input only; the caller must also invoke env.cancel()."""
        if self._baseline is not None:
            self._baseline.cancel(reason)
        self._phase = "cancelled"
        if self._request is None:
            self._latest = SkillProgress(None, self._latest.tick, "idle", "idle", reason, cancel_host=True)
        elif not self._latest.terminal:
            self._latest = SkillProgress(self._request.id, self._latest.tick, "cancelled", "cancelled", reason,
                                         effects=tuple(self._effects.values()), local_success=False, cancel_host=True)
        self._event("cancellation", result=self._latest.to_dict())
        return self._latest

    def _movement(self, view: Mapping[str, Any], target: tuple[float, float], effort: float) -> WireAction:
        position, velocity = view["position"], view["velocity"]
        if self.capability.model == "astra":
            # Reuse T103's declared controller, including its bounded profile-
            # shared motor requests. It receives no evaluator state.
            from nca.controllers import AuthoredController, FeedbackObservation, SkillGoal
            if self._baseline is None:
                self._baseline = AuthoredController("astra")
            goal = SkillGoal(self._request.id, "burst_traverse", target=target, pace=effort,
                             tolerance=2.0, timeout_seconds=10.0)
            return self._baseline.step(FeedbackObservation.from_host(view), goal).action
        # Hammerhead drives its authored gait. This feedback is a requested
        # horizontal velocity, not a private force, terrain edit or path solver.
        move = _clamp((target[0] - position[0]) * 0.10 - velocity[0] * 0.025)
        trim = _clamp(-0.55 * _angle(view["rotation"]) - 0.18 * view["angular_velocity"])
        return WireAction(move=(move, 0), trim=trim, effort=effort)

    def _observe_effects(self, observations: Sequence[Mapping[str, Any]], view: Mapping[str, Any]):
        if len(observations) > 54:
            raise ContractError("feedback_capacity", "at most 36 idle and 18 repeat observations")
        identity = self._request.actor
        previous_tick = self._last_feedback_tick
        for sample in (*observations, view):
            observed = self._validate_view(sample)
            if observed != identity or sample["tick"] > view["tick"]:
                raise ContractError("feedback_identity", "effect feedback belongs to another lifetime or future tick")
            if sample["tick"] <= previous_tick:
                continue
            if sample["tick"] < self._last_feedback_tick:
                raise ContractError("feedback_order", "completed feedback must be chronological")
            self._last_feedback_tick = sample["tick"]
            if self._trigger_tick is None or sample["tick"] <= self._trigger_tick:
                continue
            for receipt in sample.get("last_effects", ()):
                if (receipt.get("actor_slot") != identity.actor_id or receipt.get("epoch") != identity.epoch
                        or type(receipt.get("sequence")) is not int or receipt["sequence"] <= self._effect_sequence):
                    raise ContractError("effect_identity", "effect must be newly observed for this actor")
                key = (identity.epoch, receipt["sequence"])
                outcome = EffectResult.from_dict({"tick": sample["tick"], **dict(receipt)}).to_dict()
                # Re-reading a copied receipt retains its first observed event
                # time; it never becomes a new use or renews a skill deadline.
                self._effects.setdefault(key, outcome)

    def progress(self, view: Mapping[str, Any], *, observations: Sequence[Mapping[str, Any]] = ()) -> SkillProgress:
        try:
            current = self.mount_ref(view)
        except (ContractError, KeyError, TypeError):
            if self._request is None:
                raise
            return self._result({"tick": self._latest.tick}, status="invalidated", reason="invalid_local_observation")
        if self._request is None:
            self._latest = SkillProgress(None, view["tick"], "idle", "idle", "no_skill")
            return self._latest
        if self._latest.terminal:
            return self._latest
        if current.actor != self._request.actor:
            return self._result(view, status="invalidated", reason="actor_lifetime_changed")
        if current != self._request.mount:
            expected = ((self._request.skill == "release" and self._phase == "release_requested" and self._detached)
                        or (self._request.skill == "grip" and self._phase == "attach_requested" and not self._detached))
            if not expected:
                return self._result(view, status="invalidated", reason="mount_attachment_changed")
        if view.get("suspended") or view.get("fault"):
            return self._result(view, status="invalidated", reason="actor_not_controllable")
        tick = view["tick"]
        if tick < self._last_progress_tick:
            return self._result(view, status="invalidated", reason="observation_reordered")
        if tick == self._last_progress_tick:
            return self._latest
        self._last_progress_tick = tick
        try:
            self._observe_effects(observations, view)
        except (ContractError, KeyError, TypeError):
            return self._result(view, status="invalidated", reason="invalid_effect_feedback")
        params = PARAMETERS[self._request.skill].from_dict(self._request.parameters)
        skill = self._request.skill
        if skill == "aim_use" and self._effects:
            if len(self._effects) != 1:
                return self._result(view, status="invalidated", reason="ambiguous_effect_feedback")
            first = next(iter(self._effects.values()))
            # A six-tick repeat may report a valid earlier effect at the skill
            # deadline. Judge its completed event tick, preserving report time
            # separately in SkillProgress.tick. Future/late effects stay logged.
            if first["tick"] <= self._started + params.timeout_ticks:
                if first["status"] == "refused" or not first.get("consumed"):
                    return self._result(view, status="failed", reason="effect_" + str(first.get("reason", "refused")))
                if params.require_change and first.get("changed_cells", 0) == 0:
                    return self._result(view, status="failed", reason="effect_no_change")
                return self._result(view, status="succeeded", reason="local_effect_applied" if first.get("changed_cells", 0) else "local_effect_consumed_no_change")
        if tick - self._started >= params.timeout_ticks:
            return self._result(view, status="timed_out", reason="skill_deadline")
        if skill == "aim_use":
            if self._trigger_tick is not None:
                if tick - self._trigger_tick >= params.no_progress_ticks:
                    return self._result(view, status="failed", reason="no_effect_feedback")
                self._phase = "await_effect"
                return self._result(view, action=WireAction(aim=params.aim, effort=params.effort))
            part = self.capability.held_part if self.capability.held_part is not None else self.capability.mount_part
            local_angle = _angle(math.atan2(params.aim[1], params.aim[0]) - view["parts"][part]["rotation"])
            if not self.capability.limits[0] <= local_angle <= self.capability.limits[1]:
                return self._result(view, status="failed", reason="aim_outside_mount_limits")
            self._phase = "aiming"
            if tick - self._started < params.settle_ticks or not view.get("tool_ready"):
                return self._result(view, action=WireAction(aim=params.aim, effort=params.effort))
            if view["energy"] <= 0:
                return self._result(view, status="failed", reason="resource_unavailable")
            self._trigger_tick = tick
            self._phase = "trigger_requested"
            return self._result(view, action=WireAction(aim=params.aim, effort=params.effort, tool=True))
        if skill == "release":
            if self._detached:
                return self._result(view, status="succeeded", reason="already_released" if self._phase == "initiated" else "local_detachment_confirmed")
            if self._phase == "release_requested":
                return self._result(view, status="failed", reason="release_not_observed")
            self._phase = "release_requested"
            return self._result(view, action=WireAction(release_usable=True))
        if skill == "grip":
            if not self._detached:
                return self._result(view, status="succeeded", reason="already_attached" if self._phase == "initiated" else "local_attachment_confirmed")
            if self._phase == "attach_requested":
                return self._result(view, status="failed", reason="grip_refused_or_clearance")
            held = view["parts"][self.capability.held_part]["position"]
            angle = view["parts"][self.capability.mount_part]["rotation"]
            ox, oy = self.capability.offset
            offset = (ox * math.cos(angle) - oy * math.sin(angle), ox * math.sin(angle) + oy * math.cos(angle))
            target = (held[0] - offset[0], held[1] - offset[1])
            distance = math.dist(view["position"], target)
            if distance <= 7.5:
                self._phase = "attach_requested"
                return self._result(view, action=WireAction(attach_usable=True, effort=params.effort), distance=distance)
            self._phase = "approach_own_part"
        elif skill in ("approach", "carry"):
            target = params.target
            distance = math.dist(view["position"], target)
            if distance <= params.tolerance:
                return self._result(view, status="succeeded", reason="local_target_reached", distance=distance)
            self._phase = "carry_attached" if skill == "carry" else "approaching"
        elif skill == "recover":
            rotation, omega = abs(_angle(view["rotation"])), abs(view["angular_velocity"])
            if rotation <= params.angle_tolerance and omega <= params.angular_speed_tolerance:
                if self._quiet_since is None:
                    self._quiet_since = tick
                if tick - self._quiet_since >= params.settle_ticks:
                    return self._result(view, status="succeeded", reason="local_attitude_recovered")
            else:
                self._quiet_since = None
            distance = rotation + omega * 0.1
            self._phase = "recovering"
            action = WireAction(trim=_clamp(-0.55 * _angle(view["rotation"]) - 0.18 * view["angular_velocity"]),
                                secondary=self.capability.model == "astra",
                                auxiliary=self.capability.model == "hammerhead", effort=params.effort)
            if self._no_progress(distance, tick, params.no_progress_ticks):
                return self._result(view, status="failed", reason="no_progress", distance=distance)
            return self._result(view, action=action, distance=distance)
        else:  # charge_hold: cancellation disarms; a neutral/release is never emitted.
            distance = max(0.0, params.target_charge - view["charge"])
            self._phase = "charge_ready_hold" if distance == 0 else "charging"
            if distance and self._no_progress(distance, tick, params.no_progress_ticks):
                return self._result(view, status="failed", reason="no_charge_progress", distance=distance)
            return self._result(view, action=WireAction(primary=True, effort=params.effort), distance=distance)
        if self._no_progress(distance, tick, params.no_progress_ticks):
            return self._result(view, status="failed", reason="no_progress", distance=distance)
        return self._result(view, action=self._movement(view, target, params.effort), distance=distance)

    def _no_progress(self, distance: float, tick: int, budget: int) -> bool:
        if distance <= self._best - 0.05:
            self._best, self._best_tick = distance, tick
        return tick - self._best_tick >= budget


def register_skills(registry) -> None:
    """Register one versioned implementation with a declared capability set."""
    registry.register("skill", "authored-usables", 1, SkillRuntime,
                      ("act004-wire", "local-feedback", "exclusive-actor", "cancellation"))
