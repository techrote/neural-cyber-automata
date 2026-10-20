"""Versioned separation of physical steps and unrecorded local decisions.

This audit never turns a refused or uncertain command into a physical receipt.
Legacy controller-frame arrays retain their original strict interpretation.
"""

from __future__ import annotations

from collections.abc import Mapping

from nca.contracts import ContractError, WireAction, sha256_json
from nca.controllers.morphology.conformance import cancellation_receipt


SCHEMA = "nca-jolt-controller-decisions/v2"
FAILURE_SCHEMA = "nca-jolt-controller-decision-failure/v1"
CLEANUP_SCHEMA = "nca-jolt-error-cleanup/v1"


def _require(condition, detail):
    if not condition:
        raise ContractError("arrival_decision_audit", detail)


def decision_accounting(frames, failures):
    return {"controller_calls_with_frames": len(frames) + len(failures),
            "physical_step_frames": len(frames),
            "refused_before_transport": sum(item["disposition"] == "refused_before_transport" for item in failures),
            "owner_response_unrecorded": sum(item["disposition"] == "owner_response_unrecorded" for item in failures)}


def validate_decision_failure(failure, source, physical_decisions):
    """Bind the unrecorded decision to its last retained physical source."""
    _require(isinstance(failure, Mapping) and failure.get("schema") == FAILURE_SCHEMA,
             "a versioned unrecorded controller decision is required")
    frame = failure.get("frame")
    _require(isinstance(frame, Mapping) and isinstance(failure.get("source_observation"), Mapping)
        and sha256_json(failure["source_observation"]) == sha256_json(source)
        and type(frame.get("tick")) is int and frame["tick"] == source.get("tick")
        and type(failure.get("after_physical_decisions")) is int
        and failure["after_physical_decisions"] == physical_decisions,
        "the unrecorded decision must retain the exact last physical observation")
    _require(type(frame.get("inference_ns")) is int and frame["inference_ns"] >= 0,
             "the unrecorded decision needs its measured inference interval")
    _require(isinstance(frame.get("action"), Mapping), "the complete local wire action is required")
    WireAction.from_dict(frame["action"])
    error = failure.get("error")
    _require(isinstance(error, Mapping) and type(error.get("type")) is str and error["type"],
             "an unrecorded decision must retain its original failure")
    names = ("owner_requests_before", "owner_requests_after", "step_requests_before", "step_requests_after")
    _require(all(type(failure.get(name)) is int and failure[name] >= 0 for name in names),
             "the before/after owner-reservation counters are required")
    deltas = (failure["owner_requests_after"] - failure["owner_requests_before"],
              failure["step_requests_after"] - failure["step_requests_before"])
    expected = {"refused_before_transport": (0, 0), "owner_response_unrecorded": (1, 1)}
    _require(failure.get("disposition") in expected and deltas == expected[failure["disposition"]],
             "a reserved or uncertain step must never be labelled refused before transport")
    packet = failure.get("requested_packet")
    if packet is not None:
        _require(isinstance(packet, Mapping)
            and sha256_json(packet.get("action")) == sha256_json(frame["action"])
            and packet.get("observation_id") == source.get("observation_id")
            and packet.get("observed_tick") == source.get("tick"),
            "an unrecorded packet must preserve its local action and source")


def confirmed_cleanup(before, after, request_serial):
    """Return compact receipt links after independently checking cancellation."""
    proof = cancellation_receipt(before, after)
    return {"schema": CLEANUP_SCHEMA, "attempted": True, "confirmed": True, "error": None,
            "request_serial": request_serial, "tick": after["tick"],
            "source_observation_sha256": sha256_json(before),
            "after_observation_sha256": sha256_json(after),
            "control_generation_before": proof["control_generation_before"],
            "control_generation_after": proof["control_generation_after"],
            "same_tick_physical_state": proof["same_tick_pose_velocity_parts_and_joint_physics"],
            "joint_targets_revoked": proof["joint_targets_revoked"],
            "neutral_step_used": proof["neutral_step_used"]}


def replay_controller_frames(episode):
    """Read v2 without normalizing or discarding any legacy failure frame."""
    evaluation = episode.evaluation
    frames = evaluation.get("controller_frames", ())
    schema = evaluation.get("controller_decision_schema")
    if schema is None:
        _require(not any(name in evaluation for name in
            ("controller_decision_failures", "controller_decision_accounting", "error_cleanup")),
            "additive decision fields require their explicit schema")
        return frames
    _require(schema == SCHEMA, "unknown controller-decision schema")
    failures = evaluation.get("controller_decision_failures")
    _require(isinstance(failures, (list, tuple)) and len(failures) <= 1,
             "this stop-on-failure runner permits at most one unrecorded decision")
    steps = [event for event in episode.events if event.kind == "step"]
    _require(isinstance(frames, (list, tuple)) and len(frames) == len(steps),
             "only frames with retained physical step events belong in controller_frames")
    for event, frame in zip(steps, frames):
        _require(isinstance(frame, Mapping) and frame.get("tick") == event.tick_before
            and event.requested_packet is not None
            and sha256_json(frame.get("action")) == sha256_json(event.requested_packet.get("action")),
            "each physical frame must match its actual step request")
    source = steps[-1].next_observation if steps else episode.initial_observation
    for failure in failures:
        validate_decision_failure(failure, source, len(steps))
        _require(episode.status != "completed" and evaluation.get("error") is not None
            and sha256_json(failure["error"]) == sha256_json(evaluation["error"]),
            "an unrecorded decision cannot be relabelled as a completed episode")
    _require(sha256_json(evaluation.get("controller_decision_accounting"))
        == sha256_json(decision_accounting(frames, failures)),
        "controller call, physical step and refusal counters differ")
    cleanup = evaluation.get("error_cleanup")
    if cleanup is not None:
        _require(isinstance(cleanup, Mapping) and cleanup.get("schema") == CLEANUP_SCHEMA
            and cleanup.get("attempted") is True and type(cleanup.get("confirmed")) is bool
            and evaluation.get("error") is not None and episode.status != "completed",
            "error cleanup is a separate failure-bound native operation")
        if cleanup["confirmed"]:
            cancellations = [event for event in episode.events if event.kind == "cancel"]
            _require(len(cancellations) == 1 and type(cleanup.get("request_serial")) is int,
                     "confirmed error cleanup requires its retained native cancellation")
            event = cancellations[0]
            expected = confirmed_cleanup(event.issued_observation, event.next_observation,
                                         cleanup["request_serial"])
            _require(sha256_json(cleanup) == sha256_json(expected),
                     "error cleanup differs from its native cancellation receipt")
        else:
            _require(isinstance(cleanup.get("error"), Mapping),
                     "unconfirmed native cleanup must retain the cleanup failure")
    return frames
