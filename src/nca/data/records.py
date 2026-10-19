"""Immutable demonstrations with action, authority and observation provenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from nca.contracts import ContractError, Record, digest, identifier, sha256_json
from nca.randomness import ScenarioSplit


# Deliberately excludes reward, success, evaluator, global scene and host info.
# Parts/joints and the bounded patch are retained in issued_observation, not in
# this first compact policy view. A different policy view needs a new schema.
POLICY_FIELDS = (
    "version", "representation", "observation_id", "run_id", "epoch", "actor_id",
    "profile_version", "control_generation", "model", "rig_id", "tick",
    "position", "velocity", "rotation", "angular_velocity", "grounded", "energy",
    "cooldown", "ground_distance", "ceiling_distance", "blocked", "fault", "suspended",
    "charge", "gripped", "rope_attached", "rope_length", "tension", "stuck_time",
    "wall_left", "wall_right", "tool_ready", "power_enabled", "movement_limited",
)
POLICY_SCHEMA = "act004-local-proprioception-rays/v1"
IDENTITY_KEYS = ("run_id", "epoch", "actor_id", "model", "profile_version", "control_generation")


def json_copy(value: Any) -> Any:
    """Thaw immutable record values into independent JSON transport values."""
    from nca.contracts import canonical_json, strict_json_loads
    return strict_json_loads(canonical_json(value), max_bytes=32 * 1024 * 1024)


def policy_view(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only declared fields from an issued ACT004 observation."""
    if observation.get("version") != 2 or observation.get("representation") != "articulated-rapier":
        raise ContractError("observation_schema", "data policy view requires ACT004 v2")
    if any(key not in observation for key in ("run_id", "epoch", "actor_id", "tick", "observation_id")):
        raise ContractError("observation_identity", "an issued observation is required")
    return json_copy({key: observation[key] for key in POLICY_FIELDS if key in observation})


def boundary_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[str, ...]:
    """Lifecycle changes split recurrence; dynamic pose does not."""
    changes = []
    for key, label in (("run_id", "reset"), ("epoch", "epoch"), ("actor_id", "actor"),
                       ("model", "morphology"), ("profile_version", "profile"),
                       ("control_generation", "control_generation"), ("mount_generation", "mount")):
        if before.get(key) != after.get(key):
            changes.append(label)

    def topology(view):
        return tuple((part.get("id"), part.get("index"), part.get("detached", False))
                     for part in view.get("parts", ()))

    if topology(before) != topology(after):
        changes.append("part_topology")
    return tuple(changes)


@dataclass(frozen=True)
class PolicyContext(Record):
    """Explicit caller-supplied goal, not an evaluator-derived hidden target."""

    goal_id: str
    skill: str
    target: tuple[float, float]
    pace: float = .8
    tolerance: float = 4.0
    timeout_seconds: float = 12.0

    def validate(self) -> None:
        identifier(self.goal_id, "goal_id")
        identifier(self.skill, "skill")
        if not 0 < self.pace <= 1 or self.tolerance <= 0 or self.timeout_seconds <= 0:
            raise ContractError("goal_bounds", "goal pace, tolerance and deadline")


@dataclass(frozen=True)
class ActionSource(Record):
    id: str
    kind: Literal["scripted", "human", "learned", "model", "neutral"]
    policy_id: str
    policy_sha256: str
    training_privilege: Literal["none", "privileged_teacher", "privileged_critic"] = "none"

    def validate(self) -> None:
        identifier(self.id, "source.id")
        if not self.policy_id or len(self.policy_id) > 128:
            raise ContractError("policy_identity", "policy ID must be nonempty and bounded")
        digest(self.policy_sha256, "source.policy_sha256")


@dataclass(frozen=True)
class PolicySample(Record):
    """Likelihood is for these sampled values, never relabelled applied forces."""

    space: Literal["wire_request", "latent_residual"]
    values: Mapping[str, Any]
    policy_sha256: str
    log_probability: float | None = None
    distribution: str | None = None

    def validate(self) -> None:
        digest(self.policy_sha256, "sample.policy_sha256")
        if not self.values:
            raise ContractError("empty_sample", "sampled policy output is required")
        if self.log_probability is not None and not self.distribution:
            raise ContractError("likelihood_identity", "a likelihood requires its distribution identity")


@dataclass(frozen=True)
class TickRecord(Record):
    tick_before: int
    tick_after: int
    world_tick: int
    received_monotonic_usec: int
    completed_monotonic_usec: int
    requested: tuple[Mapping[str, Any], ...]
    admission: tuple[Mapping[str, Any], ...]
    consumed_request: Mapping[str, Any] | None
    source_id: str | None
    consumed_action: Mapping[str, Any]
    applied: Mapping[str, Any]
    passive: Mapping[str, Any]
    actual_motion: Mapping[str, Any]
    observation: Mapping[str, Any]
    cellular_sample_tick: int | None
    valid: bool = True

    def validate(self) -> None:
        if self.tick_before < 0 or self.tick_after < self.tick_before:
            raise ContractError("tick_order", "data tick cannot move backward")
        if self.valid and self.tick_after != self.tick_before + 1:
            raise ContractError("tick_gap", "a valid base receipt must advance exactly one tick")
        if self.valid and self.world_tick != self.tick_after:
            raise ContractError("schedule_mismatch", "physical and cellular owner tick disagree")
        if self.completed_monotonic_usec < self.received_monotonic_usec:
            raise ContractError("wall_clock", "host receipt completion predates receipt")
        if self.observation.get("tick") != self.tick_after:
            raise ContractError("observation_tick", "receipt observation belongs to another tick")
        if len(self.requested) != len(self.admission):
            raise ContractError("admission_count", "every submitted packet needs an admission result")
        for result in self.admission:
            if type(result.get("accepted")) is not bool or not isinstance(result.get("reason"), str):
                raise ContractError("admission_schema", "admission must retain accepted/reason")
        if self.consumed_request is not None:
            source_tick = self.consumed_request.get("observed_tick")
            lease = self.consumed_request.get("lease_ticks")
            if type(source_tick) not in (int, float) or type(lease) not in (int, float):
                raise ContractError("consumption_identity", "consumed request requires source timing")
            if not source_tick <= self.tick_before < source_tick + lease:
                raise ContractError("consumed_stale", "consumption exceeded source lease")
            if self.consumed_request.get("run_id") != self.observation.get("run_id"):
                raise ContractError("consumption_identity", "consumed request belongs to another run")
            if self.source_id is None:
                raise ContractError("consumption_source", "consumed input requires an identified source")
        elif self.source_id is not None:
            raise ContractError("consumption_source", "passive interval cannot invent an action source")
        if self.passive.get("available") is False and not self.passive.get("reason"):
            raise ContractError("unavailable_reason", "unavailable passive wrench requires a reason")


@dataclass(frozen=True)
class CaptureEvent(Record):
    kind: Literal["step", "cancel", "takeover", "boundary", "failure"]
    index: int
    segment: int
    tick_before: int
    tick_after: int
    wall_started_ns: int
    wall_completed_ns: int
    source: ActionSource
    issued_observation: Mapping[str, Any]
    policy_observation: Mapping[str, Any]
    requested_packet: Mapping[str, Any] | None = None
    sample: PolicySample | None = None
    context: PolicyContext | None = None
    ticks: tuple[TickRecord, ...] = ()
    next_observation: Mapping[str, Any] | None = None
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    reason: str = ""
    intervention: Literal["none", "human_correction", "model_override", "source_takeover"] = "none"
    replaced: bool = False
    discontinuities: tuple[str, ...] = ()
    valid: bool = True
    trace_complete: bool = True
    source_before: ActionSource | None = None
    source_observation: Mapping[str, Any] | None = None

    def validate(self) -> None:
        if min(self.index, self.segment, self.tick_before, self.tick_after, self.wall_started_ns) < 0:
            raise ContractError("negative_counter", "event counters must be nonnegative")
        if self.tick_after < self.tick_before or self.wall_completed_ns < self.wall_started_ns:
            raise ContractError("event_order", "event clocks cannot move backward")
        policy_source = self.issued_observation if self.source_observation is None else self.source_observation
        if sha256_json(self.policy_observation) != sha256_json(policy_view(policy_source)):
            raise ContractError("policy_privilege", "policy observation differs from declared whitelist")
        if self.tick_before != self.issued_observation.get("tick"):
            raise ContractError("observation_tick", "event must retain its issued observation tick")
        if self.kind == "step" and self.valid:
            if not self.ticks or not self.trace_complete:
                raise ContractError("incomplete_trace", "a valid decision requires every consumed tick")
            expected = self.tick_before
            for tick in self.ticks:
                if tick.tick_before != expected or not tick.valid:
                    raise ContractError("tick_gap", "valid decision receipts are not contiguous")
                expected = tick.tick_after
            if expected != self.tick_after:
                raise ContractError("tick_gap", "decision receipt coverage differs from elapsed ticks")
        if self.next_observation is not None and self.next_observation.get("tick") != self.tick_after:
            raise ContractError("observation_tick", "next observation tick differs from event end")
        if self.source_observation is not None:
            if self.requested_packet is None:
                raise ContractError("request_observation", "a source observation requires a submitted packet")
            for observation_key, request_key in (("observation_id", "observation_id"), ("tick", "observed_tick"),
                                                  ("run_id", "run_id"), ("epoch", "epoch"), ("actor_id", "actor_id")):
                if self.source_observation.get(observation_key) != self.requested_packet.get(request_key):
                    raise ContractError("request_observation", "source observation does not match the original request")
        if self.sample is not None:
            if self.sample.policy_sha256 != self.source.policy_sha256:
                raise ContractError("policy_mismatch", "sample and action source weights/config differ")
            if self.sample.space == "wire_request" and self.requested_packet is not None:
                if self.sample.values != self.requested_packet.get("action") and not self.replaced:
                    raise ContractError("sample_mismatch", "sampled wire request changed without replacement label")
        if self.kind in ("cancel", "takeover", "failure") and not self.reason:
            raise ContractError("event_reason", "control/failure events need a reason")

    @property
    def imitation_eligible(self) -> bool:
        if (self.kind != "step" or not self.valid or not self.trace_complete
                or self.requested_packet is None or self.source_observation is None):
            return False
        if self.replaced or any(reason not in ("initial", "terminal_control_fence") for reason in self.discontinuities):
            return False
        sequence = self.requested_packet.get("sequence")
        admitted_current = any(
            requested.get("sequence") == sequence
            and requested.get("observation_id") == self.requested_packet.get("observation_id")
            and admission.get("accepted") is True
            for tick in self.ticks for requested, admission in zip(tick.requested, tick.admission))
        return admitted_current and bool(self.ticks) and all(
            tick.source_id == self.source.id and tick.consumed_request is not None
            and tick.consumed_request.get("sequence") == sequence for tick in self.ticks)

    @property
    def on_policy_eligible(self) -> bool:
        return (self.imitation_eligible and self.source.kind == "learned" and self.intervention == "none"
                and self.sample is not None and self.sample.log_probability is not None)

    def require_on_policy(self) -> PolicySample:
        if not self.on_policy_eligible:
            raise ContractError("off_policy_interval", "corrected, replaced, mixed, refused or unscored samples cannot enter PPO actor loss")
        return self.sample


@dataclass(frozen=True)
class EpisodeRecord(Record):
    id: str
    host_identity: Mapping[str, Any]
    learner_revision: str
    learner_sources_sha256: str
    config: Mapping[str, Any]
    profile: Mapping[str, Any]
    initial_observation: Mapping[str, Any]
    events: tuple[CaptureEvent, ...]
    status: Literal["completed", "capped", "canceled", "failed", "inconclusive"]
    reason: str
    scenario_family: str
    split: Literal["training", "tuning", "evaluation"]
    source_files: Mapping[str, str] = field(default_factory=dict)
    schema_version: Literal[1] = 1
    policy_schema: Literal["act004-local-proprioception-rays/v1"] = POLICY_SCHEMA
    reward_kind: str = "no_task_reward"
    final_observation_valid: bool = True
    evaluation: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        identifier(self.id, "episode.id")
        identifier(self.scenario_family, "scenario_family")
        digest(self.learner_sources_sha256, "learner_sources_sha256")
        if not self.learner_revision or not self.reason:
            raise ContractError("episode_provenance", "source revision and completion reason required")
        if self.source_files and sha256_json(self.source_files) != self.learner_sources_sha256:
            raise ContractError("source_hash", "source file manifest differs from recorded aggregate")
        if self.host_identity.get("lane") != "act004":
            raise ContractError("environment_lane", "this dataset schema records actual ACT004 owners")
        policy_view(self.initial_observation)
        expected_tick = int(self.initial_observation["tick"])
        segment = 0
        for index, event in enumerate(self.events):
            if event.index != index or event.tick_before != expected_tick or event.segment < segment:
                raise ContractError("episode_order", "events must preserve the original consumed order")
            expected_tick = event.tick_after
            segment = event.segment
        if not self.final_observation_valid and self.status == "completed":
            raise ContractError("invalid_completion", "a missing final state cannot count as completed")

    @property
    def decisions(self) -> int:
        return sum(event.kind == "step" for event in self.events)

    @property
    def consumed_ticks(self) -> int:
        return sum(len(event.ticks) for event in self.events)

    @property
    def profile_sha256(self) -> str:
        return sha256_json(self.profile)

    @property
    def config_sha256(self) -> str:
        return sha256_json(self.config)

    def imitation_samples(self):
        """No info/evaluator dictionaries are exposed by the learning iterator."""
        previous_segment = None
        final_step = max((event.index for event in self.events if event.kind == "step"), default=-1)
        for event in self.events:
            if not event.imitation_eligible:
                previous_segment = None
                continue
            reset = previous_segment != event.segment
            previous_segment = event.segment
            yield {"observation": event.policy_observation, "context": event.context,
                   "action": event.requested_packet["action"], "reset": reset,
                   "intervention": event.intervention, "source": event.source,
                   "episode_id": self.id, "event_index": event.index,
                   "terminal": event.terminated or event.truncated,
                   "episode_end": event.index == final_step}


@dataclass(frozen=True)
class Dataset(Record):
    """A checked view over independently stored episodes and scenario-family splits."""

    id: str
    episodes: tuple[EpisodeRecord, ...]
    splits: ScenarioSplit

    def validate(self) -> None:
        identifier(self.id, "dataset.id")
        ids = [episode.id for episode in self.episodes]
        if len(set(ids)) != len(ids):
            raise ContractError("duplicate_episode", "dataset episodes must have distinct IDs")
        for episode in self.episodes:
            if self.splits.partition(episode.scenario_family) != episode.split:
                raise ContractError("split_leakage", "episode family assigned to another declared partition")

    def imitation_samples(self, split: str = "training"):
        if split not in ("training", "tuning", "evaluation"):
            raise ContractError("unknown_split", split)
        for episode in self.episodes:
            if episode.split == split:
                yield from episode.imitation_samples()
