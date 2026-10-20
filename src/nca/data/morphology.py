"""Explicit morphology inputs and semantic samples over unchanged owner receipts.

The legacy root-only policy projection is never serialized as a graph input.
Its general receipt validators are reused internally, after reconstructing that
projection only for structural validation. Old episode readers reject this new
policy schema, and existing episode bytes and content hashes remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import math
from pathlib import Path
from typing import Any, Literal, Mapping

from nca.contracts import ContractError, Record, digest, sha256_json, strict_json_loads
from nca.controllers.morphology.core import (
    MorphologyAdapter, MorphologyFrame, MorphologyGoal, SemanticAction, map_action,
)
from nca.controllers.morphology.conformance import _same_action
from .records import CaptureEvent, EpisodeRecord, POLICY_SCHEMA, policy_view

MORPHOLOGY_POLICY_SCHEMA = "act004-morphology-local/v1"
MORPHOLOGY_EPISODE_SCHEMA = "nca-morphology-episode/v1"
SAMPLER_SCHEMA = "masked-latent-normal-bernoulli-semantic/v1"
FIXED_LATENT_STD = .35


@dataclass(frozen=True)
class MorphologySample(Record):
    """Density belongs to the original latent draw, before the wire mapping."""

    policy_sha256: str
    frame_sha256: str
    action: SemanticAction
    action_mask: tuple[bool, ...]
    value: float
    sampling_mode: Literal["deterministic_mean", "stochastic_draw"]
    latent: tuple[float, ...] | None = None
    log_probability: float | None = None
    distribution: Literal["masked-latent-normal-bernoulli-semantic/v1"] = SAMPLER_SCHEMA
    latent_standard_deviation: float = FIXED_LATENT_STD

    def validate(self) -> None:
        digest(self.policy_sha256, "morphology_sample.policy_sha256")
        digest(self.frame_sha256, "morphology_sample.frame_sha256")
        if len(self.action_mask) != 7 or not all(self.action_mask[i] for i in (0, 2, 3)):
            raise ContractError("morphology_sample", "complete accepted capability mask required")
        if self.latent_standard_deviation != FIXED_LATENT_STD:
            raise ContractError("morphology_sample", "fixed nontrainable sampling variance required")
        if any(value != 0 for value, active in zip(self.action.vector, self.action_mask) if not active):
            raise ContractError("morphology_sample", "unavailable channels cannot enter requested-action history")
        if self.sampling_mode == "deterministic_mean":
            if self.latent is not None or self.log_probability is not None:
                raise ContractError("morphology_sample", "deterministic means have no sampled likelihood")
        else:
            if self.latent is None or len(self.latent) != 4 or self.log_probability is None:
                raise ContractError("morphology_sample", "original four-channel latent and density required")
            for index, (value, active) in enumerate(zip(self.latent, self.action_mask[:4])):
                if not active and value != 0:
                    raise ContractError("morphology_sample", "unavailable latent dimensions must be fixed zero")
                if index < 3:
                    expected = math.tanh(value) if active else 0.
                else:
                    expected = (1 / (1 + math.exp(-value)) if value >= 0 else
                                math.exp(value) / (1 + math.exp(value))) if active else 0.
                if abs(self.action.vector[index] - expected) > 2e-7:
                    raise ContractError("morphology_sample", "semantic action differs from its recorded latent draw")


@dataclass(frozen=True)
class MorphologyEvent(CaptureEvent):
    """Actual graph/padded input plus source-preserving native tick receipts."""

    policy_observation: MorphologyFrame | None
    sample: MorphologySample | None = None
    context: MorphologyGoal | None = None
    semantic_request: SemanticAction | None = None
    reward_terms: Mapping[str, Any] = field(default_factory=dict)

    def owner_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        del payload["semantic_request"], payload["reward_terms"]
        source = self.source_observation if self.source_observation is not None else self.issued_observation
        payload["policy_observation"] = policy_view(source)
        payload["sample"] = payload["context"] = None
        return payload

    def validate(self) -> None:
        CaptureEvent.from_dict(self.owner_payload())
        if self.kind == "step":
            if self.policy_observation is None or self.context is None or self.semantic_request is None:
                raise ContractError("morphology_capture", "each step needs its full local frame, goal and semantic request")
            if self.source.kind == "learned" and self.sample is None:
                raise ContractError("morphology_capture", "a learned request requires its exact policy sample")
            if self.policy_observation.goal != self.context:
                raise ContractError("morphology_capture", "recorded frame belongs to another explicit goal")
            if self.sample is not None and (
                    self.sample.frame_sha256 != self.policy_observation.policy_sha256
                    or self.sample.policy_sha256 != self.source.policy_sha256
                    or self.sample.action != self.semantic_request
                    or self.sample.action_mask != self.policy_observation.action_mask):
                raise ContractError("morphology_capture", "sample differs from its original policy, frame or request")
        elif self.sample is not None or self.semantic_request is not None:
            raise ContractError("morphology_capture", "non-step events cannot fabricate an executed policy sample")

    @property
    def on_policy_eligible(self) -> bool:
        return (self.imitation_eligible and self.source.kind == "learned" and self.intervention == "none"
                and self.sample is not None and self.sample.sampling_mode == "stochastic_draw"
                and self.sample.log_probability is not None)

    def require_on_policy(self) -> MorphologySample:
        if not self.on_policy_eligible:
            raise ContractError("off_policy_interval", "only original admitted stochastic morphology samples enter PPO")
        return self.sample


@dataclass(frozen=True)
class MorphologyEpisode(EpisodeRecord):
    events: tuple[MorphologyEvent, ...]
    policy_schema: Literal["act004-morphology-local/v1"] = MORPHOLOGY_POLICY_SCHEMA
    schema: Literal["nca-morphology-episode/v1"] = MORPHOLOGY_EPISODE_SCHEMA
    goal: MorphologyGoal | None = None

    def owner_record(self) -> EpisodeRecord:
        payload = self.to_dict()
        del payload["schema"], payload["goal"]
        payload["policy_schema"] = POLICY_SCHEMA
        payload["events"] = [event.owner_payload() for event in self.events]
        return EpisodeRecord.from_dict(payload)

    def validate(self) -> None:
        self.owner_record()
        if self.goal is None:
            raise ContractError("morphology_capture", "an explicit caller goal is required")
        adapter = MorphologyAdapter(self.profile, expected_sha256=self.profile_sha256)
        previous = SemanticAction()
        initial_tick = self.initial_observation["tick"]
        for event in self.events:
            if event.kind != "step":
                previous = SemanticAction()
                continue
            if event.source_observation is None or event.context != self.goal:
                raise ContractError("morphology_capture", "actual issued source and unchanged caller goal required")
            if sha256_json(event.source_observation) != sha256_json(event.issued_observation):
                raise ContractError("morphology_capture", "delayed or refreshed sources are excluded from this comparison")
            expected = adapter.frame(event.source_observation, self.goal, previous=previous,
                                     started_tick=initial_tick)
            if expected != event.policy_observation:
                raise ContractError("morphology_capture", "recorded model input differs from its permitted actual local source")
            wire, _ = map_action(adapter, event.semantic_request, event.source_observation, self.goal)
            if not _same_action(wire.to_dict(), event.requested_packet["action"]):
                raise ContractError("morphology_capture", "semantic request differs from the original submitted wire action")
            previous = event.semantic_request
        bootstrap = self.evaluation.get("bootstrap")
        if bootstrap is not None:
            steps = [event for event in self.events if event.kind == "step"]
            if (self.status != "capped" or not steps or steps[-1].terminated or steps[-1].truncated
                    or set(bootstrap) != {"frame", "value", "source"}
                    or type(bootstrap["value"]) not in (int, float) or not math.isfinite(bootstrap["value"])
                    or bootstrap["source"] != "actual final continuing observation before owner cancellation"):
                raise ContractError("morphology_capture", "only a valid continuing allocation cutoff can retain a bootstrap")
            last = steps[-1]
            expected = adapter.frame(last.next_observation, self.goal, previous=last.semantic_request,
                                     started_tick=initial_tick)
            if MorphologyFrame.from_dict(bootstrap["frame"]) != expected:
                raise ContractError("morphology_capture", "the bootstrap input differs from the final actual receipt")

    def imitation_samples(self):
        for event in self.events:
            if event.imitation_eligible and event.intervention == "none":
                yield event.policy_observation, event.semantic_request


def capture_from_owner(record: EpisodeRecord, *, goal: MorphologyGoal,
                       bindings: Mapping[int, tuple[MorphologyFrame, SemanticAction,
                                                    MorphologySample | None, Mapping[str, Any]]]) -> MorphologyEpisode:
    """Convert a generic in-memory transport receipt to the distinct policy schema."""
    payload = record.to_dict()
    payload.update(policy_schema=MORPHOLOGY_POLICY_SCHEMA, schema=MORPHOLOGY_EPISODE_SCHEMA,
                   goal=goal.to_dict())
    for event in payload["events"]:
        event["context"] = goal.to_dict()
        event["sample"] = None
        event["policy_observation"] = None
        event["semantic_request"] = None
        event["reward_terms"] = {}
        if event["kind"] == "step":
            try:
                frame, semantic, sample, terms = bindings[event["index"]]
            except KeyError as exc:
                raise ContractError("morphology_capture", "a completed owner step has no original policy input binding") from exc
            event["policy_observation"] = frame.to_dict()
            event["semantic_request"] = semantic.to_dict()
            event["sample"] = None if sample is None else sample.to_dict()
            event["reward_terms"] = dict(terms)
    return MorphologyEpisode.from_dict(payload)


def read_morphology_episode(path: str | Path) -> MorphologyEpisode:
    """Read the distinct graph-input schema with the existing bounded envelope."""
    from .storage import MAX_EPISODE_BYTES
    target = Path(path)
    if target.stat().st_size > MAX_EPISODE_BYTES:
        raise ContractError("dataset_size", "morphology episode file exceeds its bounded size")
    opener = gzip.open if target.suffix == ".gz" else open
    try:
        with opener(target, "rb") as stream:
            payload = stream.read(MAX_EPISODE_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise ContractError("dataset_encoding", "morphology episode compression is invalid") from exc
    value = strict_json_loads(payload, max_bytes=MAX_EPISODE_BYTES)
    if not isinstance(value, dict) or set(value) != {"episode", "sha256"}:
        raise ContractError("dataset_schema", "morphology episode envelope fields differ")
    episode = MorphologyEpisode.from_dict(value["episode"])
    if value["sha256"] != episode.sha256:
        raise ContractError("dataset_hash", "morphology episode content differs from its receipt")
    return episode
