"""Independent replay of the sole LEARN-JOLT-003 safe failure boundary.

This module performs no model inference, fitting or physical work. A reason
string never establishes a terminal: the complete original observation history
must reproduce the unchanged supervisor's emergency deadline, and native
cancellation must revoke the same incarnation without changing physical state.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Mapping, Sequence

from nca.contracts import ContractError, WireAction, canonical_json, sha256_json
from nca.controllers.baseline import FeedbackObservation, SkillGoal
from nca.controllers.morphology.conformance import cancellation_receipt
from nca.controllers.recovery_supervisor import PhysicalRecoverySupervisor
from nca.controllers.shaping import wrapped_angle


RETIREMENT_SCHEMA = "nca-jolt-safe-retirement/v3"
OUTCOME_SCHEMA = "nca-jolt-episode-outcome/v3"
CANCELLATION_SCHEMA = "nca-jolt-native-cancellation/v3"
SAFE_REASON = "emergency_recovery_deadline"
MAX_RETIREMENT_BYTES = 4 * 1024
MAX_OBSERVATION_BYTES = 64 * 1024


def _need(condition: bool, detail: str) -> None:
    if not condition:
        raise ContractError("jolt_safe_retirement", detail)


def _same(left: Any, right: Any, detail: str) -> None:
    _need(sha256_json(left) == sha256_json(right), detail)


def _asdict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else value.to_dict()


def _boundary(observation: Mapping[str, Any]) -> str:
    parts = observation.get("parts")
    _need(isinstance(parts, (list, tuple)) and all(isinstance(part, Mapping) for part in parts),
          "original actor topology is unavailable")
    return sha256_json({"mount_generation": observation.get("mount_generation"),
        "parts": [(part.get("id"), part.get("index"), part.get("detached", False)) for part in parts]})


def _observation(value: Any, invocation: Mapping[str, Any]) -> FeedbackObservation:
    _need(isinstance(value, Mapping), "the complete original observation is required")
    # canonical_json rejects non-finite nested values, including fields that the
    # feedback projection does not consume. The final view cannot hide damage.
    _need(len(canonical_json(value).encode()) <= MAX_OBSERVATION_BYTES,
          "observation exceeds the declared retirement reader")
    feedback = FeedbackObservation.from_host(value)
    _need(feedback.actor == "jev" and feedback.valid and feedback.controllable
          and feedback.lifetime == invocation["lifetime"]
          and feedback.control_generation == invocation["control_generation"]
          and _boundary(value) == invocation["boundary_sha256"]
          and feedback.tick >= invocation["start_tick"],
          "observation lost the original actor authority or topology")
    stamped = ":".join(str(value.get(key)) for key in
                       ("run_id", "epoch", "actor_id", "control_generation", "tick"))
    _need(value.get("observation_id") == stamped, "observation does not bind its original generation and tick")
    return feedback


def _emergency(feedback: FeedbackObservation) -> str | None:
    return ("unstable_attitude" if
            abs(wrapped_angle(feedback.angle)) >= PhysicalRecoverySupervisor.EMERGENCY_ANGLE
            or abs(feedback.angular_velocity) >= PhysicalRecoverySupervisor.EMERGENCY_ANGULAR_SPEED else None)


def native_cancellation_v3(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Strengthen the existing native fence with action, edge and lease proof."""
    _need(isinstance(before, Mapping) and isinstance(after, Mapping),
          "both original native cancellation observations are required")
    _need(len(canonical_json(before).encode()) <= MAX_OBSERVATION_BYTES
          and len(canonical_json(after).encode()) <= MAX_OBSERVATION_BYTES,
          "cancellation observation exceeds its bounded reader")
    fence = cancellation_receipt(before, after)
    _need(type(after.get("pending_input_edges")) is int and after["pending_input_edges"] == 0,
          "native cancellation did not prove cleared input edges")
    _need(type(after.get("lease_remaining")) is int and after["lease_remaining"] == 0,
          "native cancellation did not prove the expired owner lease")
    _need(after.get("fault", "") == "", "native cancellation returned a faulted actor")
    _same(after.get("action"), {}, "native cancellation did not clear the requested action")
    _same(_boundary(before), _boundary(after), "native cancellation changed actor topology")
    # Suspension is deliberately absent from this proof: a valid cancel may
    # clear control while leaving the host's physical actor unsuspended.
    return {"schema": CANCELLATION_SCHEMA,
            "before_sha256": sha256_json(before), "after_sha256": sha256_json(after),
            "same_tick_pose_velocity_parts_and_joint_physics": True,
            "same_actor_incarnation": True, "joint_targets_revoked": True,
            "action_cleared": True, "pending_input_edges": 0, "lease_remaining": 0,
            "control_generation_before": fence["control_generation_before"],
            "control_generation_after": fence["control_generation_after"],
            "neutral_step_used": False}


def validate_safe_retirement_v3(decisions: Sequence[Any], final_observation: Mapping[str, Any],
        goal: SkillGoal, context: Any, history: Any, path: str, *,
        after_cancel: Mapping[str, Any]) -> dict[str, Any]:
    """Replay all consumed intervals and the final non-dispatched attempt.

    ``ordinary_attempt`` is charged once as a learned controller invocation by
    the caller; ``cutoff_peek`` is an administrative bootstrap probe. Neither
    produces a decision record, a new action, a value or a model forward.
    """
    _need(path in ("ordinary_attempt", "cutoff_peek"), "unknown retirement discovery path")
    _need(isinstance(goal, SkillGoal) and goal.skill == "directed_hop"
          and goal.timeout_seconds == 4., "the original four-second Jolt goal is required")
    _need(0 < len(decisions) <= 40, "safe retirement requires positive bounded physical progress")
    invocation = _asdict(context)
    _need(set(invocation) == {"schema", "goal", "start_tick", "deadline_tick", "lifetime",
                              "control_generation", "boundary_sha256"}
          and invocation["schema"] == "jolt-invocation/v2", "original invocation contract changed")
    first = decisions[0].event.source_observation
    _need(isinstance(first, Mapping), "first source observation is unavailable")
    first_feedback = FeedbackObservation.from_host(first)
    expected_invocation = {"schema": "jolt-invocation/v2", "goal": asdict(goal),
        "start_tick": first_feedback.tick,
        "deadline_tick": first_feedback.tick + math.ceil(goal.timeout_seconds * 60 - 1e-12),
        "lifetime": first_feedback.lifetime, "control_generation": first_feedback.control_generation,
        "boundary_sha256": _boundary(first)}
    _same(invocation, expected_invocation, "retirement renewed or changed the original invocation")
    supervisor = PhysicalRecoverySupervisor()
    expected_history = {"schema": "jolt-dual-action-history/v2", "proposal": None,
                        "dispatched": None, "intervention": False, "admitted": False}
    previous = None
    for index, decision in enumerate(decisions):
        event = decision.event
        _need(event.kind == "step" and event.valid and event.trace_complete
              and decision.observation_valid and decision.actual_ticks == 6 and decision.admitted,
              "a safe failure requires complete confirmed six-tick decisions")
        _need(event.source_observation is not None and event.next_observation is not None
              and event.requested_packet is not None, "physical interval omitted original evidence")
        _same(event.source_observation, event.issued_observation, "retirement history changed its source observation")
        if previous is not None:
            _same(event.source_observation, previous, "retirement history omitted or reordered an interval")
        feedback = _observation(event.source_observation, invocation)
        _need(event.tick_before == feedback.tick and event.tick_after == feedback.tick + 6,
              "retirement history contains uncertain physical time")
        _same(decision.invocation, invocation, "retirement history renewed its deadline")
        _same(decision.history, expected_history, "retirement history changed proposal or dispatch state")
        supervision = supervisor.admit(feedback, goal, emergency=_emergency(feedback))
        _need(supervision.status == "running" and 6 <= supervision.action_lease_ticks <= 12,
              "physical action followed an already retired invocation")
        _need(decision.supervisor_sha256 == sha256_json(supervisor.specification())
              and decision.supervision_reason == supervision.reason
              and decision.overridden is (supervision.override is not None),
              "emergency history does not replay under the shared supervisor")
        action = WireAction.from_dict(event.requested_packet["action"])
        if supervision.override is not None:
            _need(decision.proposal is None and decision.draw is None
                  and action == supervision.override, "emergency interval invented ordinary learning evidence")
        else:
            _need(decision.proposal == action, "ordinary dispatch differs from its retained proposal")
        lease = event.requested_packet["lease_ticks"]
        _need(type(lease) is int and 6 <= lease <= supervision.action_lease_ticks
              and (event.source.kind == "scripted" or lease == supervision.action_lease_ticks)
              and event.tick_before + lease <= invocation["deadline_tick"],
              "retirement history renewed or enlarged physical authority")
        _need(not event.terminated, "safe retirement cannot replace an earlier task terminal")
        _need(not event.truncated or path == "cutoff_peek" and index == len(decisions) - 1,
              "safe retirement cannot follow an earlier administrative cutoff")
        expected_history = {"schema": "jolt-dual-action-history/v2",
            "proposal": None if decision.proposal is None else decision.proposal.to_dict(),
            "dispatched": action.to_dict(), "intervention": decision.overridden, "admitted": decision.admitted}
        previous = event.next_observation
    _same(final_observation, previous, "retirement did not use the final received pre-cancel observation")
    _same(_asdict(history), expected_history, "retirement attempt changed recurrent history")
    _need(decisions[-1].event.truncated is (path == "cutoff_peek"),
          "retirement path differs from its physical cutoff boundary")
    final_feedback = _observation(final_observation, invocation)
    # The unchanged supervisor orders original deadline, quiet exit and
    # emergency expiry. Replaying it rejects forged reason-only terminals.
    final_supervision = supervisor.admit(final_feedback, goal, emergency=_emergency(final_feedback))
    _need(final_supervision.status == "canceled" and final_supervision.reason == SAFE_REASON,
          "complete emergency history did not independently reach its 48-tick bound")
    cancellation = native_cancellation_v3(final_observation, after_cancel)
    result = {"schema": RETIREMENT_SCHEMA, "reason": SAFE_REASON, "path": path,
        "observation_sha256": sha256_json(final_observation), "invocation": invocation,
        "history": expected_history, "supervisor_sha256": sha256_json(supervisor.specification()),
        "final_supervision": asdict(final_supervision), "cancellation": cancellation}
    _need(len(canonical_json(result).encode()) <= MAX_RETIREMENT_BYTES,
          "retirement metadata exceeds its declared bound")
    return result
