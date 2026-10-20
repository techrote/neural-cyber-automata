"""One actual waypoint episode with full local input and native owner receipts.

This module supplies no experiment admission. The enclosing continuation must
validate its immutable prerequisite and phase allocations before calling it.
"""

from __future__ import annotations

from dataclasses import replace
import gzip
import time
from pathlib import Path

import torch

from nca.adapters.godot import CybersandEnv
from nca.contracts import ContractError, canonical_json, sha256_json
from nca.controllers.morphology.core import MorphologyAdapter, MorphologyGoal, SemanticAction, map_action
from nca.controllers.morphology.models import dispatch
from nca.controllers.morphology.conformance import cancellation_receipt, validate_interval
from nca.data import ActionSource, EpisodeRecorder
from nca.data.capture import CaptureBudgetExceeded
from nca.data.morphology import capture_from_owner
from nca.data.storage import write_episode
from nca.envs import physical_config
from nca.runs import RunBudget
from .artifacts import immutable_bytes, immutable_json
from .goal_frame import _portable_error, audit_episode_boundaries
from .morphology_learning import MorphologyActor, MorphologyRollout, MorphologyTransition, policy_sha256
from .morphology_config import MorphologyBudgetExceeded
from .morphology_task import REFERENCE_SPEC, TASK_SPEC, WaypointReference, WaypointTask
from .screen import ScreenBudgetExceeded


class MorphologyCaptureError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__("a failed morphology episode was preserved")


class _RecordedOwner:
    """Let the unchanged morphology dispatch validate a recorded actual RPC."""
    def __init__(self, recorder):
        self.recorder = recorder

    def command(self, *args, **kwargs):
        return self.recorder.env.command(*args, **kwargs)

    def step_packet(self, packet):
        return self.recorder.step_packet(packet)

    def cancel(self):
        return self.recorder.cancel("morphology_dispatch_owner_cleanup")


def capture_waypoint(host, scenario, *, episode_id: str, split: str, phase: str,
                     policy_id: str, kind: str, model, seed: int, sample_seed: int,
                     budget, output: Path, profiles, root: Path,
                     maximum_decisions: int = 40, training: bool = False,
                     training_seed: int | None = None):
    """Record one real reset and every original issued/consumed six-tick action."""
    budget.check()
    if type(maximum_decisions) is not int or not 1 <= maximum_decisions <= 40 or type(training) is not bool:
        raise ContractError("morphology_capture", "bounded exact episode allocation and explicit training mode required")
    if kind not in ("reference", "idle", "padded_recurrent", "shared_graph"):
        raise ContractError("morphology_capture", "only declared reference, idle or candidate controllers are available")
    if kind in ("reference", "idle") and (model is not None or training):
        raise ContractError("morphology_capture", "a scripted reference cannot be relabelled as an on-policy learner")
    if kind in ("padded_recurrent", "shared_graph") and (model is None or model.family != kind):
        raise ContractError("morphology_capture", "the declared candidate family must match the actual model")
    if ((model is None and training_seed is not None) or (model is not None
            and (type(training_seed) is not int or training_seed not in (6113, 6229, 6337)))):
        raise ContractError("morphology_capture", "candidate identity requires its original declared training seed")
    profile = scenario["profile"]
    expected_profile = profiles["profiles"][profile]["profile_sha256"]
    goal = MorphologyGoal(episode_id, tuple(scenario["target"]))
    host_config = physical_config(profile, ticks=goal.deadline_ticks, seed=seed)
    host_config["actors"][0]["position"] = list(scenario["spawn"])
    env = CybersandEnv(host, host_config, decision_ticks=6, lease_ticks=12)
    policy_hash = (sha256_json(REFERENCE_SPEC) if kind == "reference" else
                   sha256_json({"kind": "morphology-idle/v1", "semantic_request": SemanticAction().to_dict()})
                   if kind == "idle" else policy_sha256(model))
    source = ActionSource(policy_id, "scripted" if kind == "reference" else "neutral" if kind == "idle" else "learned",
                          policy_id, policy_hash)
    recorder = EpisodeRecorder(env, episode_id, source, seed=seed, scenario_family=scenario["id"], split=split,
        budget=RunBudget(max_decisions=40, max_seconds=min(120., budget.remaining_seconds())), source_root=root)
    adapter = MorphologyAdapter(recorder.profile, expected_sha256=expected_profile)
    task = WaypointTask(adapter, goal, recorder.initial)
    reference = WaypointReference(adapter) if kind == "reference" else None
    actor = MorphologyActor(model, adapter, seed=sample_seed, stochastic=training) if model is not None else None
    bindings, transitions = {}, []
    previous, error, capped, bootstrap, bootstrap_frame = SemanticAction(), None, False, 0., None
    native_terminal, owner_closed = False, False
    try:
        for index in range(maximum_decisions):
            budget.check()
            view = recorder.view
            started = time.perf_counter_ns()
            sample = None
            if actor is not None:
                decision = actor.step(view, goal)
                if decision.cancel_owner or actor.last_sample is None or actor.last_frame is None:
                    raise ContractError("morphology_refusal", "the candidate refused its source before dispatch")
                frame, semantic, sample = actor.last_frame, actor.last_sample.action, actor.last_sample
                wire = decision.action
            else:
                frame = adapter.frame(view, goal, previous=previous, started_tick=recorder.initial["tick"])
                semantic = reference.step(view, goal) if reference is not None else SemanticAction()
                wire, _ = map_action(adapter, semantic, view, goal)
            inference_ns = time.perf_counter_ns() - started
            event_index = len(recorder.events)
            bindings[event_index] = (frame, semantic, sample, {})
            # Reserve before dispatch. An ambiguous transport tail cannot be
            # silently retried or reported as zero attempted game work.
            budget.consume(phase)
            if actor is not None:
                result = dispatch(_RecordedOwner(recorder), actor, decision, view)
            else:
                packet = env.command(wire.to_dict(), observation=view, lease_ticks=min(12, 240 - view["tick"]))
                result = recorder.step_packet(packet)
                after, _, terminated, truncated, info = result
                validate_interval(packet, view, after, terminated, truncated, info, allow_terminal=True)
                original = adapter.identity(view)
                following = adapter.terminal_identity(after) if terminated or truncated else adapter.identity(after)
                if (following.lifetime != original.lifetime
                        or following.control_generation != original.control_generation + int(terminated or truncated)
                        or not (terminated or truncated) and following.topology_sha256 != original.topology_sha256):
                    raise ContractError("morphology_boundary", "the reference result changed owner or physical source")
            after, _, terminated, truncated, info = result
            native_terminal = terminated or truncated
            terms = task.consume(wire, info["trace"], inference_ns=inference_ns)
            bindings[event_index] = (frame, semantic, sample, terms)
            previous = semantic
            if task.integrity_failure:
                raise ContractError("morphology_boundary", "invalid or refused owner receipts cannot supply a learning reward")
            if training:
                event = recorder.events[event_index]
                transitions.append(MorphologyTransition(frame, sample, terms["total"], info["actual_ticks"],
                    episode_id, index, actor_eligible=event.imitation_eligible and event.intervention == "none",
                    task_terminal=task.done, final_observation_valid=True))
            if task.done:
                break
            if native_terminal:
                raise ContractError("morphology_terminal", "the host stopped before the declared task outcome")
        else:
            capped = not task.done
        if capped and training:
            # Peek on the actual final observation without consuming a new
            # action or changing the live invocation, sampler or hidden state.
            bootstrap_frame = adapter.frame(recorder.view, goal, previous=actor.previous, started_tick=actor.fence.start_tick)
            with torch.no_grad():
                bootstrap = float(model.forward_frame(bootstrap_frame, actor.hidden.detach().clone()).value)
            transitions[-1] = replace(transitions[-1], administrative_cutoff=True)
        if native_terminal:
            owner_closed = True
        else:
            before = recorder.view
            after = recorder.cancel("waypoint_arrival" if task.succeeded else "administrative_batch_cutoff")
            cancellation_receipt(before, after)
            if actor is not None:
                actor.confirm_owner_cancel(after)
            owner_closed = True
        if reference is not None:
            reference.cancel()
    except BaseException as exc:
        error = _portable_error(exc)
        capped = isinstance(exc, (ScreenBudgetExceeded, MorphologyBudgetExceeded, CaptureBudgetExceeded))
        task.integrity_failure = not capped
        transitions, bootstrap, bootstrap_frame = [], 0., None
        try:
            if recorder.events and recorder.events[-1].kind == "cancel":
                cancellation_receipt(recorder.events[-1].issued_observation, recorder.events[-1].next_observation)
            elif not native_terminal:
                before = recorder.view
                after = recorder.cancel("morphology_episode_error_cleanup")
                cancellation_receipt(before, after)
            owner_closed = True
        except BaseException:
            owner_closed = False
        if actor is not None:
            actor.cancel()
        if reference is not None:
            reference.cancel()
    if model is not None:
        try:
            if policy_sha256(model) != policy_hash:
                raise ContractError("morphology_policy_changed", "the original policy changed during the actual capture")
        except BaseException as exc:
            if error is None:
                error = _portable_error(exc)
            task.integrity_failure = True
            transitions, bootstrap, bootstrap_frame = [], 0., None
    metrics = task.metrics(administratively_capped=capped)
    # Dispatch can reject an actual refusal before the task consumes that
    # interval. Its known native admission count still belongs in the receipt.
    metrics["failed_admissions"] = sum(admission.get("accepted") is not True
        for event in recorder.events for tick in event.ticks for admission in tick.admission)
    evaluation = {"task_spec_sha256": sha256_json(TASK_SPEC), "metrics": metrics, "policy_id": policy_id,
                  "sample_policy_sha256": policy_hash, "error": error, "phase": phase,
                  "actual_owner_closed": owner_closed, "model_input_schema": "act004-morphology-local/v1",
                  "bootstrap": None if bootstrap_frame is None else
                      {"frame": bootstrap_frame.to_dict(), "value": bootstrap,
                       "source": "actual final continuing observation before owner cancellation"}}
    status = "capped" if capped else "failed" if error is not None or not owner_closed else "completed"
    owner = recorder.finish(status, metrics["reason"], evaluation=evaluation)
    boundary = audit_episode_boundaries(owner, expected_profile)
    artifact = output / "episodes" / f"{episode_id}.json.gz"
    try:
        episode = capture_from_owner(owner, goal=goal, bindings=bindings)
        artifact_sha = write_episode(artifact, episode)
        record_hash, record_schema = episode.sha256, episode.schema
    except BaseException as exc:
        # Preserve every completed native receipt even if a malformed source
        # prevents construction of an admitted morphology episode. This distinct
        # failure envelope cannot be read as an eligible learner trajectory.
        error = _portable_error(exc)
        failed = {"schema": "nca-morphology-incomplete-owner-receipts/v1", "admitted": False,
                  "owner_receipts": owner.to_dict(), "owner_record_sha256": owner.sha256,
                  "actual_input_bindings": {str(index): {"frame": values[0].to_dict(),
                      "semantic_request": values[1].to_dict(), "sample": None if values[2] is None else values[2].to_dict(),
                      "reward_terms": dict(values[3])} for index, values in bindings.items()},
                  "error": error, "policy_input_claim": "only separately retained full local frames; legacy root projection is transport diagnostic only"}
        record_hash = sha256_json(failed)
        artifact = output / "incomplete-owner-receipts" / f"{episode_id}.json.gz"
        artifact_sha = immutable_bytes(artifact, gzip.compress(canonical_json({**failed, "sha256": record_hash}).encode(), mtime=0))
        record_schema, episode = failed["schema"], None
        boundary["passed"] = False
        boundary["source_boundary_regressions"] += 1
        boundary["errors"].append({"event_index": None, **error})
    boundary.update(record_sha256=record_hash, record_schema=record_schema)
    boundary["sha256"] = sha256_json(boundary)
    immutable_json(output / "boundary-audit" / f"{episode_id}.json", boundary)
    result = {"episode_id": episode_id, "scenario_family": scenario["id"], "profile": profile, "split": split,
              "phase": phase, "host_seed": seed, "sample_seed": sample_seed, "kind": kind, "policy_id": policy_id,
              "training_seed": training_seed,
              "policy_sha256": policy_hash, "artifact": artifact.relative_to(output).as_posix(),
              "artifact_sha256": artifact_sha, "record_sha256": record_hash, "record_schema": record_schema,
              "profile_sha256": expected_profile, "decisions": owner.decisions,
              "task_spec_sha256": sha256_json(TASK_SPEC),
              "completed_ticks": owner.consumed_ticks, "metrics": metrics,
              "error": error, "boundary_audit_sha256": boundary["sha256"],
              "source_boundary_regressions": boundary["source_boundary_regressions"], "actual_owner_closed": owner_closed}
    immutable_json(output / "metrics" / f"{episode_id}.json", result)
    if not boundary["passed"] or error is not None or not owner_closed:
        if capped:
            exc = MorphologyBudgetExceeded("the declared morphology episode budget elapsed")
            exc.result = result
            raise exc
        raise MorphologyCaptureError(result)
    return episode, result, MorphologyRollout(tuple(transitions), bootstrap, bootstrap_frame) if transitions else None
