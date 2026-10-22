"""Versioned LEARN-JOLT-002 evidence, independent of optional Torch imports.

The v1 capture event remains the native receipt container. This separate v2
record retains the ordinary proposal before supervision and binds it to the
immutable invocation and the exact two-channel action history used by policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

from nca.contracts import ContractError, Record, WireAction, digest, identifier, sha256_json
from .records import CaptureEvent, policy_view


SAMPLE_SCHEMA = "jolt-pretransform-normal-bernoulli/v2"
DECISION_SCHEMA = "jolt-proposal-supervision-consumption/v2"
EPISODE_SCHEMA = "jolt-invocation-episode/v2"
EPISODE_SCHEMA_V3 = "jolt-invocation-episode/v3"
OUTCOME_SCHEMA_V3 = "nca-jolt-episode-outcome/v3"
NATIVE_ACTION_TOLERANCE = 1e-6


def native_action_matches(expected: Mapping[str, Any], actual: Any) -> bool:
    """Native Vector2/JSON readback uses the established absolute 1e-6 bound.

    Original policy draws/proposals remain exact. Only original-input native
    wire receipt comparisons use this bound; buttons never use a tolerance.
    """
    if not isinstance(actual, Mapping) or set(expected) != set(actual):
        return False
    for key, wanted in expected.items():
        got = actual[key]
        if type(wanted) is bool:
            if got is not wanted:
                return False
        elif isinstance(wanted, (tuple, list)):
            if not isinstance(got, (tuple, list)) or len(got) != len(wanted):
                return False
            if any(type(value) not in (int, float) or not math.isfinite(value)
                   or abs(value - target) > NATIVE_ACTION_TOLERANCE for value, target in zip(got, wanted)):
                return False
        elif (type(got) not in (int, float) or not math.isfinite(got)
              or abs(got - wanted) > NATIVE_ACTION_TOLERANCE):
            return False
    return True


def native_packet_matches(expected: Mapping[str, Any], actual: Any) -> bool:
    if not isinstance(actual, Mapping) or set(expected) != set(actual):
        return False
    if any(actual[key] != value or ((type(value) is bool or type(actual[key]) is bool)
                                   and actual[key] is not value)
           for key, value in expected.items() if key != "action"):
        return False
    return native_action_matches(expected.get("action", {}), actual.get("action"))


@dataclass(frozen=True)
class JoltDraw(Record):
    """Density belongs to the original raw Normal/Bernoulli draw only."""

    latent: tuple[float, float, float, float, float]
    buttons: tuple[bool, bool]
    sampling_mode: Literal["stochastic_draw", "deterministic_mean"]
    log_probability: float | None
    policy_sha256: str
    base_sha256: str | None
    source_sha256: str
    feature_sha256: str
    history_sha256: str
    invocation_sha256: str
    proposal_sha256: str
    schema: Literal["jolt-pretransform-normal-bernoulli/v2"] = SAMPLE_SCHEMA

    def validate(self) -> None:
        for name in ("policy_sha256", "source_sha256", "feature_sha256", "history_sha256",
                     "invocation_sha256", "proposal_sha256"):
            digest(getattr(self, name), name)
        if self.base_sha256 is not None:
            digest(self.base_sha256, "base_sha256")
        if (self.sampling_mode == "stochastic_draw") != (self.log_probability is not None):
            raise ContractError("jolt_density", "only an original stochastic draw has a behaviour density")


def draw_from_decision(decision: Any, *, source_sha256: str) -> JoltDraw | None:
    """Bind an executing v2 decision without inventing an overridden proposal."""
    if decision.proposal is None:
        if any(value is not None for value in (decision.latent, decision.buttons, decision.log_probability)):
            raise ContractError("jolt_draw", "an absent proposal cannot retain an ordinary policy sample")
        return None
    return JoltDraw(decision.latent, decision.buttons, decision.sampling_mode,
                    decision.log_probability, decision.policy_sha256, decision.base_sha256,
                    source_sha256, sha256_json(decision.features), sha256_json(decision.history.to_dict()),
                    sha256_json(decision.context.to_dict()), decision.proposal.sha256)


@dataclass(frozen=True)
class JoltDecisionRecord(Record):
    """Proposal, supervisor, dispatch, admission and consumption stay separate.

``event`` stores complete source/dispatch observations, the transmitted packet,
all native admissions, every consumed base tick and its actual physical receipt.
Neither an impulse nor a supervisor replacement can become an ordinary label.
"""

    event: CaptureEvent
    invocation: Mapping[str, Any]
    history: Mapping[str, Any]
    proposal: WireAction | None
    supervisor_sha256: str
    supervision_reason: str = ""
    overridden: bool = False
    draw: JoltDraw | None = None
    observation_valid: bool = True
    recurrent_reset: bool = False
    schema: Literal["jolt-proposal-supervision-consumption/v2"] = DECISION_SCHEMA

    def validate(self) -> None:
        digest(self.supervisor_sha256, "supervisor_sha256")
        if not self.invocation or not self.history:
            raise ContractError("jolt_context", "original invocation and explicit history masks are required")
        if self.overridden != self.event.replaced:
            raise ContractError("jolt_supervision", "capture replacement and supervision must agree")
        if self.overridden and not self.supervision_reason:
            raise ContractError("jolt_supervision", "a replacement needs its observed reason")
        if self.event.source_observation is None and self.observation_valid:
            raise ContractError("jolt_observation", "a valid policy record needs its full source observation")
        if self.proposal is not None:
            if (self.proposal.move[1] != 0 or self.proposal.reel != 0 or self.proposal.auxiliary
                    or self.proposal.tool or self.proposal.attach_usable or self.proposal.release_usable):
                raise ContractError("jolt_action", "ordinary Jolt policy owns only five continuous and two button channels")
            if not self.overridden and self.event.requested_packet is not None:
                dispatched = WireAction.from_dict(self.event.requested_packet["action"])
                if dispatched != self.proposal:
                    raise ContractError("jolt_dispatch", "changed ordinary proposal requires explicit supervision")
        if self.draw is not None:
            if self.proposal is None or self.event.source.kind != "learned":
                raise ContractError("jolt_draw", "a policy draw needs its original learned proposal")
            if self.draw.policy_sha256 != self.event.source.policy_sha256:
                raise ContractError("jolt_draw", "sample and action source identify different weights")
            if (self.draw.history_sha256 != sha256_json(self.history)
                    or self.draw.invocation_sha256 != sha256_json(self.invocation)
                    or self.draw.proposal_sha256 != self.proposal.sha256):
                raise ContractError("jolt_draw", "proposal, invocation or recurrent history changed after sampling")
            if self.event.sample is not None:
                if (self.event.sample.distribution != SAMPLE_SCHEMA
                        or sha256_json(self.event.sample.values) != self.draw.sha256
                        or self.event.sample.log_probability != self.draw.log_probability):
                    raise ContractError("jolt_draw", "legacy transport sample disagrees with the versioned draw")

    @property
    def admitted(self) -> bool:
        packet = self.event.requested_packet
        if packet is None:
            return False
        return any(native_packet_matches(packet, request) and admission.get("accepted") is True
                   for tick in self.event.ticks
                   for request, admission in zip(tick.requested, tick.admission))

    @property
    def imitation_eligible(self) -> bool:
        return bool(self.observation_valid and self.proposal is not None
                    and not self.overridden and self.event.intervention == "none"
                    and self.event.imitation_eligible and self.exact_consumption)

    @property
    def exact_consumption(self) -> bool:
        """Sequence numbers alone do not prove which command was consumed."""
        packet = self.event.requested_packet
        if packet is None or self.proposal is None or not self.event.ticks:
            return False
        for tick in self.event.ticks:
            if not native_packet_matches(packet, tick.consumed_request):
                return False
            # This is the per-tick original-input receipt. A later neutral
            # suspended view is never invented as the action actually applied.
            if (tick.applied.get("action_available") is not True
                    or not native_action_matches(self.proposal.to_dict(), tick.applied.get("action"))):
                return False
            terminal_fence = ("terminal_control_fence" in self.event.discontinuities
                              and tick.tick_after == self.event.tick_after
                              and tick.observation.get("suspended") is True)
            if not terminal_fence and not native_action_matches(self.proposal.to_dict(), tick.consumed_action):
                return False
        return True

    @property
    def actor_eligible(self) -> bool:
        return bool(self.imitation_eligible and self.event.source.kind == "learned"
                    and self.draw is not None and self.draw.sampling_mode == "stochastic_draw")

    @property
    def actual_ticks(self) -> int:
        return self.event.tick_after - self.event.tick_before

    @property
    def policy_observation(self) -> Mapping[str, Any] | None:
        return None if not self.observation_valid else policy_view(self.event.source_observation)


@dataclass(frozen=True)
class JoltEpisode(Record):
    id: str
    split: Literal["training", "open", "sealed"]
    decisions: tuple[JoltDecisionRecord, ...]
    status: Literal["completed", "capped", "failed", "canceled"]
    reason: str = ""
    schema: Literal["jolt-invocation-episode/v2"] = EPISODE_SCHEMA

    def validate(self) -> None:
        identifier(self.id, "episode.id")
        if len(self.decisions) > 40:
            raise ContractError("jolt_episode", "the declared 240-tick episode permits at most forty decisions")
        if not self.decisions:
            if self.status == "completed":
                raise ContractError("jolt_episode", "an empty startup failure is evidence, never a completed episode")
            return
        invocation = sha256_json(self.decisions[0].invocation)
        for index, row in enumerate(self.decisions):
            if sha256_json(row.invocation) != invocation:
                raise ContractError("jolt_invocation", "sequence boundaries cannot renew an invocation or its deadline")
            if row.event.kind != "step":
                raise ContractError("jolt_episode", "episode rows represent physical decision attempts; control events stay in capture evidence")
            if index and row.event.tick_before < self.decisions[index - 1].event.tick_after:
                raise ContractError("jolt_sequence", "decision attempts cannot reorder or overlap consumed time")
            if index and row.event.index <= self.decisions[index - 1].event.index:
                raise ContractError("jolt_sequence", "capture indices must remain strictly ordered")
            if index + 1 < len(self.decisions) and (row.event.terminated or row.event.truncated):
                raise ContractError("jolt_sequence", "no decisions may follow task termination or administrative cutoff")


@dataclass(frozen=True)
class JoltEpisodeV3(JoltEpisode):
    """A separately versioned outcome boundary over unchanged native decisions.

    A verified safe retirement is a complete unsuccessful episode. It does not
    rewrite a consumed tick as a task deadline, invent a last transition, or
    supply a new reward. ``retirement`` is authenticated against the complete
    raw observations/cancellation by the independent retirement verifier.
    """

    outcome: Mapping[str, Any] | None = None
    retirement: Mapping[str, Any] | None = None
    schema: Literal["jolt-invocation-episode/v3"] = EPISODE_SCHEMA_V3

    def validate(self) -> None:
        super().validate()
        outcome = self.outcome
        if (not isinstance(outcome, Mapping)
                or set(outcome) != {"schema", "kind", "complete", "success", "reason"}
                or outcome["schema"] != OUTCOME_SCHEMA_V3
                or outcome["kind"] not in {"safe_retirement", "task_terminal", "administrative_cutoff", "integrity_error"}
                or type(outcome["complete"]) is not bool or type(outcome["success"]) is not bool
                or not isinstance(outcome["reason"], str) or not outcome["reason"]):
            raise ContractError("jolt_outcome", "v3 requires an explicit typed episode outcome")
        if outcome["complete"] != (outcome["kind"] != "integrity_error"):
            raise ContractError("jolt_outcome", "integrity failures cannot acquire a complete outcome")
        if outcome["complete"] and (not self.decisions or any(row.actual_ticks <= 0 for row in self.decisions)):
            raise ContractError("jolt_outcome", "receipt-complete captures require positive confirmed progress")
        if outcome["success"] and (outcome["kind"] != "task_terminal" or outcome["reason"] != "arrival"):
            raise ContractError("jolt_outcome", "only a native-receipt task arrival may succeed")
        if self.reason != outcome["reason"]:
            raise ContractError("jolt_outcome", "episode and versioned outcome reasons differ")
        if outcome["kind"] == "safe_retirement":
            if (self.status != "completed" or outcome["success"]
                    or outcome["reason"] != "emergency_recovery_deadline"
                    or not isinstance(self.retirement, Mapping)
                    or self.retirement.get("schema") != "nca-jolt-safe-retirement/v3"
                    or self.retirement.get("reason") != outcome["reason"]
                    or self.retirement.get("path") not in ("ordinary_attempt", "cutoff_peek")
                    or any(row.event.terminated for row in self.decisions)):
                raise ContractError("jolt_outcome", "safe retirement needs its separate failed-terminal proof")
        elif self.retirement is not None:
            raise ContractError("jolt_outcome", "an ordinary or erroneous capture cannot invent safe retirement")
        if outcome["kind"] == "task_terminal" and (self.status != "completed" or not self.decisions[-1].event.terminated):
            raise ContractError("jolt_outcome", "task completion must retain its original terminal tick flags")
        if outcome["kind"] == "administrative_cutoff" and (
                self.status != "capped" or not self.decisions[-1].event.truncated or outcome["success"]):
            raise ContractError("jolt_outcome", "an administrative cutoff is a complete capped capture")
        if outcome["kind"] == "integrity_error" and self.status not in ("failed", "capped", "canceled"):
            raise ContractError("jolt_outcome", "integrity errors remain incomplete capture failures")


def reconstructed_history(previous: JoltDecisionRecord | None) -> dict[str, Any]:
    """The next feature uses the prior ordinary proposal and actual dispatch."""
    from nca.controllers.jolt_learned import ActionHistory
    if previous is None:
        return ActionHistory().to_dict()
    packet = previous.event.requested_packet
    dispatched = None if packet is None else WireAction.from_dict(packet["action"])
    return ActionHistory(proposal=previous.proposal, dispatched=dispatched,
                         intervention=previous.overridden or previous.event.intervention != "none",
                         admitted=previous.admitted).to_dict()


def validate_history(episode: JoltEpisode) -> None:
    """Check recorded history against receipts without censoring missing rows."""
    previous = None
    for row in episode.decisions:
        expected = reconstructed_history(previous)
        if sha256_json(row.history) != sha256_json(expected):
            raise ContractError("jolt_history", "recorded previous action differs from proposal/dispatch receipt history")
        missing = previous is not None and (not previous.observation_valid
                    or previous.event.next_observation is None
                    or row.event.tick_before != previous.event.tick_after)
        if missing and not row.recurrent_reset:
            raise ContractError("jolt_history", "missing physical observations require an explicit recurrent reset")
        if (previous is not None and not missing and row.observation_valid
                and sha256_json(row.event.issued_observation) != sha256_json(previous.event.next_observation)):
            raise ContractError("jolt_history", "the next decision does not use the previous complete issued observation")
        previous = row
