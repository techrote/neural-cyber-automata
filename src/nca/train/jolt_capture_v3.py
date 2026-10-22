"""LEARN-JOLT-003 receipt-complete episodes with narrowly proved retirement.

Execution v1/v2 still use their original capture. This module changes only the
episode outcome/terminal boundary: the same supervisor, teacher, policy, native
host, rewards, received physical ticks and per-tick flags remain unchanged.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import gzip
import math
from pathlib import Path
import time

from nca.adapters.godot import CybersandEnv, physical_config
from nca.contracts import ContractError, WireAction, sha256_json, strict_json_loads
from nca.controllers.baseline import FeedbackObservation, SkillGoal
from nca.controllers.jolt_learned import (
    ActionHistory, InvocationContext, JoltController, SupervisorRetirement,
    emergency_request, policy_state_sha256,
)
from nca.controllers.jolt_recovery_v4 import JoltRecoveryControllerV4
from nca.controllers.recovery_supervisor import PhysicalRecoverySupervisor
from nca.data import ActionSource, EpisodeRecorder, PolicyContext
from nca.data.jolt_samples import JoltDecisionRecord, JoltEpisodeV3, OUTCOME_SCHEMA_V3, validate_history
from nca.evaluation.jolt_character import character_metrics
from nca.evaluation.jolt_retirement import native_cancellation_v3, validate_safe_retirement_v3
from nca.runs import RunBudget
from nca.train.task import HopTask, REWARD_SCHEMA
from nca.train.jolt_storage_v3 import (
    compressed_byte_limit, episode_byte_limit, metrics_byte_limit_v3, publish_json_v3,
    write_episode_v3, write_sample_episode_v3 as _write_sample_episode_v3,
)


MAX_SAMPLE_BYTES_V3 = 64 * 1024 * 1024
# Historical bounded declarations retain their nested capture wall guard.
# The explicitly unlimited declaration uses None at both levels.
PER_EPISODE_WALL_SECONDS_V3 = 1800.


@dataclass(frozen=True)
class CaptureBudgetV3(RunBudget):
    """Preserve the decision cap while allowing an explicit absent wall cap."""

    max_seconds: float | None = None

    def validate(self) -> None:
        if type(self.max_decisions) is not int or self.max_decisions < 1:
            raise ContractError("invalid_budget", "the capture decision ceiling must be positive")
        if self.max_seconds is not None and (type(self.max_seconds) not in (int, float)
                or not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ContractError("invalid_budget", "a capture wall ceiling is positive or explicitly absent")

    def exhausted(self, decisions: int, seconds: float) -> bool:
        if type(decisions) is not int or decisions < 0 or type(seconds) not in (int, float):
            raise ContractError("invalid_measurement", "budget counters must be nonnegative numbers")
        if not math.isfinite(seconds) or seconds < 0:
            raise ContractError("invalid_measurement", "elapsed time must be finite and nonnegative")
        return decisions >= self.max_decisions or (self.max_seconds is not None and seconds >= self.max_seconds)


def write_sample_episode_v3(path, episode, *, check=None):
    """Publish the versioned boundary under its coupled decision-byte bound."""
    if not isinstance(episode, JoltEpisodeV3):
        raise ContractError("jolt_sample_version", "v3 writes require the new episode boundary")
    return _write_sample_episode_v3(path, episode, check=check)


def read_sample_episode_v3(path):
    path = Path(path)
    maximum = min(MAX_SAMPLE_BYTES_V3, episode_byte_limit(40))
    if path.stat().st_size > min(MAX_SAMPLE_BYTES_V3, compressed_byte_limit(maximum)):
        raise ContractError("jolt_sample_size", "v3 compressed sample exceeds its bounded reader")
    with gzip.open(path, "rb") as source:
        data = source.read(maximum + 1)
    value = strict_json_loads(data, max_bytes=maximum)
    if not isinstance(value, dict) or set(value) != {"episode", "sha256"}:
        raise ContractError("jolt_sample_schema", "v3 sample envelope fields differ")
    episode = JoltEpisodeV3.from_dict(value["episode"])
    coupled_maximum = episode_byte_limit(len(episode.decisions))
    if len(data) > coupled_maximum or path.stat().st_size > compressed_byte_limit(coupled_maximum):
        raise ContractError("jolt_sample_size", "sample exceeds its coupled per-decision byte envelope")
    if episode.sha256 != value["sha256"]:
        raise ContractError("jolt_sample_hash", "v3 sample bytes differ from their identity")
    validate_history(episode)
    return episode


def capture_v3(owner, ledger, journal, case, *, identifier, controller_id, kind, stage,
               output, root, config, source_sha256, policy=None, base=None, generator=None,
               maximum_decisions=40, case_stream_entry=None):
    """Keep a safe failure only after independent replay and native revocation."""
    from nca.train.jolt_learning import rollout_sequence_v3
    from nca.train.jolt_budget_v3 import budget_exhaustion_v3
    from nca.train.jolt_screen import _draw, error_record, note
    if type(maximum_decisions) is not int or not 1 <= maximum_decisions <= 40:
        raise ContractError("jolt_capture_window", "v3 episodes have one to forty physical decisions")
    if stage != "ppo" and maximum_decisions != 40:
        raise ContractError("jolt_capture_window", "only fixed PPO update boundaries may shorten an invocation")
    split = {"teacher": "training", "ppo": "training", "open": "open", "sealed": "sealed"}[stage]
    owner.stage = stage
    ledger.reserve("episodes_started", stage=stage)
    note(journal, "episode_started", id=identifier, case_id=case["id"], controller_id=controller_id,
         stage=stage, case_stream_entry=case_stream_entry)
    settings = physical_config("jev", ticks=240, seed=case["host_seed"])
    settings["actors"][0]["position"] = list(case["spawn"])
    env = CybersandEnv(owner, settings, decision_ticks=6, lease_ticks=12)
    goal = SkillGoal(identifier, "directed_hop", tuple(case["target"]), .8, 4., 4.)
    controller = (JoltRecoveryControllerV4() if kind == "authored" else
                  JoltController(policy, base_policy=base, generator=generator,
                      reserve=lambda name, amount=1: ledger.reserve(name, amount, stage=stage),
                      structured_retirement=True) if policy is not None else None)
    supervisor = PhysicalRecoverySupervisor() if kind == "idle" else None
    policy_sha = (controller.specification_sha256 if kind == "authored" else
                  sha256_json({"idle": WireAction(effort=0.).to_dict()}) if kind == "idle"
                  else policy_state_sha256(policy))
    source = ActionSource(controller_id, "scripted" if kind == "authored" else
                          "neutral" if kind == "idle" else "learned", controller_id, policy_sha)
    recorder = task = context = None
    decisions, values, frames, rewards = [], [], [], []
    history = ActionHistory()
    closure, error, caught, cancellation_error = False, None, None, None
    cleanup_phase_error = None
    retirement_validation_error = None
    retirement = None
    retirement_path = "ordinary_attempt"
    bootstrap, bootstrap_valid = 0., False
    bootstrap_reason, bootstrap_observation_sha256 = "not_attempted", None

    def enter_failed_closure():
        """A failed capture cannot dispatch again; its cleanup uses the tail."""
        nonlocal cleanup_phase_error
        if getattr(ledger, "phase", None) == "active":
            try:
                ledger.enter_phase("closure")
            except Exception as exc:
                cleanup_phase_error = error_record(exc)

    try:
        remaining = ledger.remaining_seconds()
        if remaining is not None and remaining <= 0:
            ledger.check()
            raise ContractError("jolt_capture_wall", "no original active allocation remains")
        capture_seconds = None if remaining is None else min(PER_EPISODE_WALL_SECONDS_V3, remaining)
        recorder = EpisodeRecorder(env, identifier, source,
            context=PolicyContext(goal.id, goal.skill, goal.target, goal.pace, goal.tolerance, goal.timeout_seconds),
            seed=case["host_seed"], scenario_family=case["id"],
            split="training" if split == "training" else "evaluation",
            budget=CaptureBudgetV3(40, capture_seconds), source_root=root)
        if sha256_json(recorder.profile) != config["profile_sha256"]:
            raise ContractError("jolt_profile", "the accepted physical profile changed")
        context = InvocationContext.create(recorder.initial, goal)
        if policy is not None:
            controller.start(recorder.initial, goal)
        task = HopTask(goal, recorder.initial, deadline_ticks=240, quiet_ticks_required=12)
        for index in range(maximum_decisions):
            ledger.check()
            view = recorder.view
            begin = time.perf_counter_ns()
            decision = None
            # All comparators share the original authority/time validation.
            feedback = context.validate_observation(view, goal) if kind in ("authored", "idle") else None
            if kind == "authored":
                frame = controller.step(feedback, goal)
                supervision = controller._latest
                action = frame.action
                proposal = None if supervision.override is not None else action
                diagnostic = controller.diagnostic_state()
                lease = diagnostic["action_lease_ticks"]
                phase, reason = frame.phase, frame.reason
            elif kind == "idle":
                supervision = supervisor.admit(feedback, goal, emergency=emergency_request(feedback))
                action = supervision.override if supervision.override is not None else WireAction(effort=0.)
                proposal = None if supervision.override is not None else action
                lease = supervision.action_lease_ticks
                phase, reason, diagnostic = "idle", supervision.reason, {}
            else:
                # The attempted call is charged before supervision. A retiring
                # attempt consumes no forward, proposal, action or native tick.
                ledger.reserve("checkpoint_executions", stage=stage)
                decision = controller.step(view, goal, deterministic=stage != "ppo")
                supervision = decision.supervision
                action, proposal, lease = decision.action, decision.proposal, supervision.action_lease_ticks
                if decision.repeat_ticks != 6:
                    raise ContractError("jolt_repeat", "the declared decision consumes exactly six ticks")
                history = decision.history
                phase, reason, diagnostic = "learned", supervision.reason, {}
            inference_ns = time.perf_counter_ns() - begin
            if supervision.status != "running":
                raise SupervisorRetirement(supervision.reason)
            if not 6 <= lease <= 12 or view["tick"] + lease > context.deadline_tick:
                raise ContractError("jolt_supervisor_lease", "original authority and deadline must remain intact")
            overridden = supervision.override is not None
            packet = env.command(action.to_dict(), observation=view, lease_ticks=lease)
            note(journal, "decision_proposed", episode_id=identifier, index=index,
                 observation_sha256=sha256_json(view), invocation=context.to_dict(), history=history.to_dict(),
                 proposal=None if proposal is None else proposal.to_dict(), supervision=asdict(supervision),
                 dispatched_packet=packet, draw=None if decision is None or decision.proposal is None
                 else _draw(decision, source_sha256).to_dict())
            _, _, terminated, truncated, info = recorder.step_packet(packet, replaced=overridden,
                intervention="model_override" if overridden else "none")
            event = recorder.events[-1]
            terms = task.consume(action, info["trace"], inference_ns=inference_ns)
            rewards.append(terms)
            cutoff = index + 1 == maximum_decisions and not task.done
            sample_event = replace(event, reward=terms["total"],
                terminated=task.succeeded or task.deadline_missed, truncated=cutoff)
            record = JoltDecisionRecord(sample_event, context.to_dict(), history.to_dict(), proposal,
                sha256_json(PhysicalRecoverySupervisor().specification()), supervision.reason, overridden,
                None if decision is None else _draw(decision, source_sha256),
                recurrent_reset=index == 0 or bool(decisions and decisions[-1].overridden))
            decisions.append(record)
            values.append(None if decision is None or decision.value is None else decision.value)
            diagnostic.update(supervisor_intervention=overridden, supervisor_reason=supervision.reason)
            frames.append({"tick": view["tick"], "action": action.to_dict(), "phase": phase, "reason": reason,
                "status": supervision.status, "inference_ns": inference_ns, "diagnostic_state": diagnostic})
            if decision is not None:
                controller.record_dispatch(decision, action, admitted=record.admitted, intervention=overridden)
            history = ActionHistory(proposal, action, overridden, record.admitted)
            if record.actual_ticks != 6 or not record.admitted or task.integrity_failure:
                raise ContractError("jolt_native_admission", "native refusal or uncertain tick consumption aborts")
            if task.done:
                break
            if terminated or truncated:
                raise ContractError("jolt_early_terminal", "native host ended before the fixed task deadline")
        if not task.done and stage != "ppo":
            raise ContractError("jolt_evaluation_cutoff", "an unfinished evaluation cannot become a complete outcome")
        if task.done:
            bootstrap_valid, bootstrap_reason = True, "task_terminal"
        else:
            retirement_path = "cutoff_peek"
            original_callback = controller.reserve
            peek = copy.deepcopy(controller, memo={id(original_callback): original_callback})
            peek.reserve = lambda name, amount=1: ledger.reserve("bootstrap_forward", amount, stage=stage)
            bootstrap_reason = "attempt_failed"
            bootstrap_decision = peek.step(recorder.view, goal, deterministic=True)
            bootstrap_observation_sha256 = sha256_json(recorder.view)
            bootstrap_valid = bootstrap_decision.proposal is not None and bootstrap_decision.value is not None
            bootstrap = float(bootstrap_decision.value) if bootstrap_valid else 0.
            bootstrap_reason = "received_cutoff_observation" if bootstrap_valid else "supervisor_value_unavailable"
        if task.deadline_missed:
            closure = bool(recorder.view.get("suspended"))
        else:
            before = recorder.view
            after = recorder.cancel("task_arrival" if task.done else "administrative_rollout_cutoff")
            native_cancellation_v3(before, after)
            closure = True
        if not closure:
            raise ContractError("jolt_owner_closure", "native closure is required")
    except Exception as exc:
        caught, error = exc, error_record(exc)
        if recorder is not None:
            candidate = (isinstance(exc, SupervisorRetirement)
                         and exc.reason == "emergency_recovery_deadline"
                         and task is not None and not task.done and bool(decisions))
            if not candidate:
                enter_failed_closure()
            try:
                before = recorder.view
                after = recorder.cancel("emergency_recovery_deadline" if candidate else "failed_episode_cleanup")
                native_cancellation_v3(before, after)
                closure = True
            except Exception as cleanup_exc:
                closure = False
                cancellation_error = error_record(cleanup_exc)
            if candidate and closure:
                try:
                    proof = validate_safe_retirement_v3(decisions, before, goal, context, history,
                                                       retirement_path, after_cancel=after)
                    note(journal, "safe_retirement_verified", episode_id=identifier, retirement=proof)
                    retirement, error, caught = proof, None, None
                    bootstrap, bootstrap_valid = 0., False
                    bootstrap_reason, bootstrap_observation_sha256 = "safe_retirement", None
                except Exception as proof_exc:
                    # Native revocation may be valid even when the retirement
                    # proof or its durable note fails. Both remain separate
                    # evidence; neither failure permits another invocation.
                    retirement_validation_error = error_record(proof_exc)
        if error is not None:
            enter_failed_closure()
    finally:
        if controller is not None:
            controller.cancel()
    if recorder is None or task is None:
        note(journal, "episode_failed_before_capture", id=identifier, error=error)
        raise ContractError("jolt_capture_start", "startup failure retained in original journal")
    metrics = task.metrics(administratively_capped=not task.done and retirement is None)
    if retirement is not None:
        metrics.update(success=False, reason="emergency_recovery_deadline", administratively_capped=False,
                       integrity_failure=False, success_seconds=None, capped_completion_seconds=4.)
    capped = caught is not None and (type(caught).__name__ == "AdmissionError" or budget_exhaustion_v3(caught))
    if error is not None and not capped:
        metrics["integrity_failure"] = True
    if error is not None:
        metrics["success"] = False
    try:
        character = character_metrics(recorder.initial, goal.target, recorder.events, frames)
    except Exception as exc:
        character = None
        error, caught = error or error_record(exc), caught or exc
        retirement = None
        metrics["integrity_failure"] = True
        metrics["success"] = False
        enter_failed_closure()
    status = "capped" if capped else "failed" if error else "completed" if task.done or retirement else "capped"
    outcome = {"schema": OUTCOME_SCHEMA_V3,
        "kind": "integrity_error" if error else "safe_retirement" if retirement else
                "task_terminal" if task.done else "administrative_cutoff",
        "complete": error is None and closure,
        "success": bool(metrics["success"] and error is None), "reason": metrics["reason"]}
    sample_episode = JoltEpisodeV3(identifier, split, tuple(decisions), status, metrics["reason"],
                                  outcome=outcome, retirement=retirement)
    validate_history(sample_episode)
    evaluation = {"metrics": metrics, "character_metrics": character, "reward_terms": rewards,
        "controller_frames": frames, "task_reward_schema": REWARD_SCHEMA,
        "owner_closure_confirmed": closure, "error": error, "cancellation_error": cancellation_error,
        "cleanup_phase_error": cleanup_phase_error,
        "retirement_validation_error": retirement_validation_error,
        "outcome": outcome, "retirement": retirement, "case_stream_entry": case_stream_entry}
    raw_episode = recorder.finish(status, metrics["reason"], evaluation=evaluation)
    raw_path = output / "episodes" / f"{identifier}.json.gz"
    publication_check = lambda: ledger.check(cleanup=True)
    raw_sha = write_episode_v3(raw_path, raw_episode, check=publication_check)
    sample_path = output / "samples" / f"{identifier}.json.gz"
    sample_identity = write_sample_episode_v3(sample_path, sample_episode, check=publication_check)
    row = {"id": identifier, "case_id": case["id"], "controller_id": controller_id,
        "kind": kind, "stage": stage, "split": split, "host_seed": case["host_seed"],
        "profile_sha256": raw_episode.profile_sha256, "policy_sha256": policy_sha,
        "artifact": raw_path.relative_to(output).as_posix(), "artifact_sha256": raw_sha,
        "sample_artifact": sample_path.relative_to(output).as_posix(), "sample_sha256": sample_identity["sha256"],
        "decisions": len(decisions), "consumed_ticks": raw_episode.consumed_ticks,
        "policy_values": values if stage == "ppo" else None,
        "bootstrap_value": bootstrap if stage == "ppo" else None,
        "bootstrap_observation_sha256": bootstrap_observation_sha256,
        "bootstrap_observation_valid": bootstrap_valid, "bootstrap_reason": bootstrap_reason,
        "metrics": metrics, "character_metrics": character,
        "supervisor_interventions": sum(record.overridden for record in decisions),
        "owner_closure_confirmed": closure, "error": error, "cancellation_error": cancellation_error,
        "cleanup_phase_error": cleanup_phase_error,
        "retirement_validation_error": retirement_validation_error,
        "outcome": outcome, "retirement": retirement, "case_stream_entry": case_stream_entry}
    publish_json_v3(output / "metrics" / f"{identifier}.json", row,
                    maximum_bytes=metrics_byte_limit_v3(kind), check=publication_check)
    ledger.reserve("episodes_aborted" if error else "episodes_completed", stage=stage, cleanup=True)
    note(journal, "episode_finished", row=row)
    if error:
        exc = caught or ContractError("jolt_capture_failed", "failed episode and receipts retained")
        exc.row = row
        raise exc
    try:
        sequence = (rollout_sequence_v3(sample_episode, values, bootstrap_value=bootstrap,
                                       final_observation_valid=bootstrap_valid) if stage == "ppo" else None)
        # Include durable episode_finished and terminal conversion in the
        # original active interval, including the last comparison episode.
        ledger.check()
    except Exception as exc:
        exc.row = row
        raise
    return sample_episode, row, sequence
