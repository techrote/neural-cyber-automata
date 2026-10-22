"""Single-invocation LEARN-JOLT-002 orchestration over the accepted host.

The worker admits only an independently frozen declaration. Every owner call
has a durable reservation, uncertain work is charged, and a capped allocation
cannot be resumed. Large original receipts and checkpoints remain outside Git.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy
import torch

from nca.adapters.godot import CybersandEnv, GodotHost, inspect_host, physical_config
from nca.contracts import ContractError, WireAction, canonical_json, sha256_json
from nca.controllers.baseline import FeedbackObservation, SkillGoal
from nca.controllers.jolt_recovery_v4 import JoltRecoveryControllerV4
from nca.controllers.recovery_supervisor import PhysicalRecoverySupervisor
from nca.controllers.jolt_learned import (
    ActionHistory, InvocationContext, JoltController, JoltPolicy, SupervisorRetirement, emergency_request,
    features, load_policy, policy_state_sha256, save_policy,
)
from nca.data import ActionSource, EpisodeRecorder, PolicyContext, read_episode, write_episode
from nca.data.jolt_samples import JoltDecisionRecord, JoltDraw, JoltEpisode, validate_history
from nca.data.records import json_copy
from nca.evaluation.durable_journal import (
    DurableJournal, file_identity, publish_json, sync_directory, validate_record_chain,
)
from nca.evaluation.jolt_arrival import _AbsoluteDeadlineSocket
from nca.evaluation.jolt_character import character_metrics
from nca.evaluation.jolt_learning import POLICY_SEEDS, qualify
from nca.runs import RunBudget
from nca.train.task import HopTask, REWARD_SCHEMA


DECLARATION_SCHEMA = "nca-learn-jolt-002-execution/v1"
RUN_SCHEMA = "nca-learn-jolt-002-run/v1"
DECLARATION_SCHEMA_V2 = "nca-learn-jolt-002-execution/v2"
RUN_SCHEMA_V2 = "nca-learn-jolt-002-run/v2"
DECLARATION_SCHEMA_V3 = "nca-learn-jolt-003-execution/v3"
RUN_SCHEMA_V3 = "nca-learn-jolt-003-run/v1"


def runtime_identity():
    """Public runtime/hardware facts only; no node names or local paths."""
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {"torch": str(torch.__version__), "numpy": numpy.__version__, "device": "cpu", "threads": 1,
            "interop_threads": 1, "python": platform.python_version(), "system": platform.system(),
            "kernel": platform.release(), "machine": platform.machine(), "libc": list(platform.libc_ver()),
            "cpu_model": cpu_model, "logical_cpus_available": os.cpu_count(),
            "deterministic_algorithms": True}


def error_record(exc):
    # Exception text may include an absolute path or private process details.
    record = {"type": type(exc).__name__, "code": getattr(exc, "code", None)}
    if isinstance(exc, SupervisorRetirement):
        record["reason"] = exc.reason
    return record


def note(journal, event, **fields):
    journal.write((canonical_json({"event": event, **fields}) + "\n").encode())


def source_inventory(root):
    root = Path(root)
    paths = sorted(set((root / "src/nca").rglob("*.py")) | set((root / "src/nca").rglob("*.gd"))
                   | set((root / "tools").glob("*.py")) | set((root / "tests").rglob("*.py"))
                   | set((root / "configs/learning").glob("*.json")))
    paths.append(root / "pyproject.toml")
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
            if not path.name.startswith("jolt-execution-")}


def validate_execution(root, declaration, *, config, host_identity, execution_version=1):
    if type(execution_version) is int and execution_version == 3:
        from nca.train.jolt_admission_v3 import validate_execution_v3
        return validate_execution_v3(root, declaration, config=config, host_identity=host_identity)
    if type(execution_version) is not int or execution_version not in (1, 2):
        raise ContractError("jolt_execution_version", "unknown execution version")
    schema = DECLARATION_SCHEMA if execution_version == 1 else DECLARATION_SCHEMA_V2
    if (declaration.get("schema") != schema or declaration.get("attempts") != 1
            or (execution_version == 2 and type(declaration.get("attempts")) is not int)):
        raise ContractError("jolt_execution", "exactly one versioned prospective invocation is required")
    if execution_version == 2:
        from nca.train.jolt_execution_contract import validate_v2_allocation_identity
        validate_v2_allocation_identity(declaration)
    inventory = source_inventory(root)
    if declaration.get("source_files") != inventory or declaration.get("source_sha256") != sha256_json(inventory):
        raise ContractError("jolt_execution_sources", "executing source differs from the prospective declaration")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if declaration.get("source_revision") != revision:
        raise ContractError("jolt_execution_revision", "execution must use the frozen clean source commit")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise ContractError("jolt_execution_dirty", "publish all source, configuration and reviews before execution")
    if declaration.get("config_sha256") != sha256_json(config):
        raise ContractError("jolt_execution_config", "the experiment configuration changed after admission")
    if (declaration.get("budget") != config["budget"] or declaration.get("automatic_retry") is not False
            or declaration.get("automatic_extension") is not False):
        raise ContractError("jolt_execution_budget", "the declaration must preserve the complete one-attempt allocation")
    if declaration.get("host_identity") != host_identity:
        raise ContractError("jolt_execution_host", "host/runtime identity differs from the prospective declaration")
    runtime = runtime_identity()
    if declaration.get("runtime") != runtime or torch.__version__ != "2.8.0+cpu" or numpy.__version__ != "2.3.5":
        raise ContractError("jolt_execution_runtime", "the pinned CPU runtime is required")
    reviews = declaration.get("independent_reviews", [])
    if len(reviews) < 2 or len({review.get("path") for review in reviews}) != len(reviews):
        raise ContractError("jolt_execution_review", "independent implementation and admission reviews are required")
    reviewers = set()
    for review in reviews:
        relative = review.get("path", "")
        if not relative.startswith("docs/evidence/") or ".." in Path(relative).parts:
            raise ContractError("jolt_execution_review", "review must be a repository evidence artifact")
        if file_identity(Path(root) / relative)["sha256"] != review.get("sha256"):
            raise ContractError("jolt_execution_review", "review bytes changed after prospective admission")
        record = json.loads((Path(root) / relative).read_text())
        if execution_version == 2:
            from nca.train.jolt_execution_contract import require_prospective_review
            require_prospective_review(record)
        if record.get("accepted") is not True or record.get("source_sha256") != declaration["source_sha256"]:
            raise ContractError("jolt_execution_review", "reviews must accept exactly the executing source closure")
        reviewer = record.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer or reviewer in reviewers:
            raise ContractError("jolt_execution_review", "each source review requires a distinct independent reviewer")
        reviewers.add(reviewer)


class MeteredOwner:
    """One pending exchange, exact source packets and a whole-exchange deadline."""

    def __init__(self, host, ledger, journal, *, transport_seconds=15, execution_version=1):
        self.host, self.ledger, self.journal = host, ledger, journal
        self.transport_seconds = transport_seconds
        self.execution_version = execution_version
        self.stage = "teacher"
        self.serial = 0
        self.pending = None
        self.pending_operation = None
        self.uncertain = []
        self.requests = {name: 0 for name in ("reset", "profile", "configure", "cancel", "close", "step")}
        self.requested_ticks = self.confirmed_ticks = self.refused_ticks = self.uncertain_ticks = 0
        self.closed_native = False

    def __getattr__(self, key):
        return getattr(self.host, key)

    def _mark_uncertain(self):
        if self.pending is not None and self.pending not in self.uncertain:
            self.uncertain.append(self.pending)
            if self.pending_operation == "step":
                self.ledger.reserve("uncertain_ticks", 6, stage=self.stage, cleanup=True)
                self.uncertain_ticks += 6

    def request(self, operation, **fields):
        if operation not in self.requests:
            raise ContractError("jolt_owner_operation", "operation is absent from the prospective owner inventory")
        cleanup = operation in ("cancel", "close")
        if self.pending is not None and not cleanup:
            raise ContractError("jolt_owner_uncertain", "uncertain prior work prevents further ordinary dispatch")
        self.ledger.check(cleanup=cleanup, allowance_seconds=self.transport_seconds)
        reservations = {"owner_requests": 1, f"owner_{operation}": 1}
        if operation == "step":
            if fields.get("ticks") != 6:
                raise ContractError("jolt_step_cadence", "the declared experiment consumes six-tick decisions")
            reservations.update(decisions=1, requested_ticks=6)
        self.ledger.reserve_many(reservations, stage=self.stage, cleanup=cleanup,
                                 allowance_seconds=self.transport_seconds)
        self.serial += 1
        serial = self.serial
        self.requests[operation] += 1
        if operation == "step":
            self.requested_ticks += 6
        if self.pending is not None:
            self._mark_uncertain()
        self.pending = serial
        self.pending_operation = operation
        deadline = time.monotonic() + min(self.transport_seconds, self.ledger.remaining_seconds(cleanup=cleanup))
        note(self.journal, "owner_request", serial=serial, stage=self.stage, operation=operation,
             fields=fields, elapsed_seconds=time.monotonic() - self.ledger.started)
        # Durability is inside the same operation deadline and cannot renew it.
        remaining = self.ledger.remaining_seconds(cleanup=cleanup)
        if remaining <= 0:
            raise ContractError("jolt_wall_budget", "the original invocation deadline is exhausted")
        channel = self.host._channel
        original = channel.connection
        timeout = original.gettimeout()
        channel.connection = _AbsoluteDeadlineSocket(original, deadline, time.monotonic)
        try:
            response = self.host.request(operation, **fields)
            if self.execution_version == 3 and time.monotonic() >= deadline:
                raise ContractError("jolt_owner_exchange_deadline", "complete native response exceeded its original exchange deadline")
        except Exception:
            self._mark_uncertain()
            raise
        finally:
            channel.connection = original
            try:
                original.settimeout(timeout)
            except OSError:
                pass
        try:
            note(self.journal, "owner_response", serial=serial, response=response)
            if self.execution_version == 3 and time.monotonic() >= deadline:
                raise ContractError("jolt_owner_exchange_deadline", "native response durability exceeded its original exchange deadline")
            if operation == "step":
                ticks = response.get("actual_ticks")
                if type(ticks) is not int or not 0 <= ticks <= 6:
                    raise ContractError("jolt_tick_receipt", "actual consumption is outside the reserved interval")
                self.confirmed_ticks += ticks
                if ticks:
                    self.ledger.reserve("confirmed_ticks", ticks, stage=self.stage, cleanup=True)
                rejected = any(admission.get("accepted") is not True for trace in response.get("traces", ())
                               for admission in trace.get("admission", ()))
                if rejected:
                    self.ledger.reserve("refused_ticks", 6, stage=self.stage, cleanup=True)
                    self.refused_ticks += 6
            if operation == "close" and response.get("closed") is not True:
                raise ContractError("jolt_close_receipt", "native close requires its explicit confirmed receipt")
        except Exception:
            self._mark_uncertain()
            raise
        if operation == "close":
            self.closed_native = True
        self.pending = None
        self.pending_operation = None
        return response


def write_sample_episode(path, episode):
    payload = (canonical_json({"episode": episode.to_dict(), "sha256": episode.sha256}) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(gzip.compress(payload, compresslevel=1, mtime=0))
        target.flush()
        os.fsync(target.fileno())
    sync_directory(path.parent)
    return file_identity(path)


def read_sample_episode(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        value = json.load(source)
    episode = JoltEpisode.from_dict(value["episode"])
    if episode.sha256 != value["sha256"]:
        raise ContractError("jolt_sample_hash", "retained sample bytes differ from their identity")
    validate_history(episode)
    return episode


def _draw(decision, source_sha256):
    if decision.proposal is None:
        return None
    return JoltDraw(latent=decision.latent, buttons=decision.buttons,
        sampling_mode=decision.sampling_mode, log_probability=decision.log_probability,
        policy_sha256=decision.policy_sha256, base_sha256=decision.base_sha256,
        source_sha256=source_sha256, feature_sha256=sha256_json(decision.features),
        history_sha256=sha256_json(decision.history.to_dict()),
        invocation_sha256=sha256_json(decision.context.to_dict()), proposal_sha256=decision.proposal.sha256)


def capture(owner, ledger, journal, case, *, identifier, controller_id, kind, stage,
            output, root, config, source_sha256, policy=None, base=None, generator=None,
            maximum_decisions=40, execution_version=1, case_stream_entry=None):
    """One invocation; preserve every unsuccessful label and partial tail."""
    if type(execution_version) is int and execution_version == 3:
        from nca.train.jolt_capture_v3 import capture_v3
        return capture_v3(owner, ledger, journal, case, identifier=identifier, controller_id=controller_id,
            kind=kind, stage=stage, output=output, root=root, config=config, source_sha256=source_sha256,
            policy=policy, base=base, generator=generator, maximum_decisions=maximum_decisions,
            case_stream_entry=case_stream_entry)
    from nca.train.jolt_learning import rollout_sequence
    split = {"teacher": "training", "ppo": "training", "open": "open", "sealed": "sealed"}[stage]
    owner.stage = stage
    ledger.reserve("episodes_started", stage=stage)
    note(journal, "episode_started", id=identifier, case_id=case["id"], controller_id=controller_id, stage=stage)
    settings = physical_config("jev", ticks=240, seed=case["host_seed"])
    settings["actors"][0]["position"] = list(case["spawn"])
    env = CybersandEnv(owner, settings, decision_ticks=6, lease_ticks=12)
    goal = SkillGoal(identifier, "directed_hop", tuple(case["target"]), .8, 4., 4.)
    controller = (JoltRecoveryControllerV4() if kind == "authored" else
                  JoltController(policy, base_policy=base, generator=generator,
                                 reserve=lambda counter, amount=1: ledger.reserve(counter, amount, stage=stage),
                                 structured_retirement=execution_version == 2)
                  if policy is not None else None)
    supervisor = PhysicalRecoverySupervisor() if kind == "idle" else None
    policy_sha = (controller.specification_sha256 if kind == "authored" else
                  sha256_json({"idle": WireAction(effort=0.).to_dict()}) if kind == "idle" else policy_state_sha256(policy))
    source = ActionSource(controller_id, "scripted" if kind == "authored" else "neutral" if kind == "idle" else "learned",
                          controller_id, policy_sha)
    recorder = None
    decisions, values, frames, rewards = [], [], [], []
    history = ActionHistory()
    closure, error, bootstrap, task, caught = False, None, 0., None, None
    bootstrap_valid, bootstrap_observation_sha256, bootstrap_reason = True, None, "task_terminal"
    cancellation_error = None
    if execution_version == 2:
        bootstrap_valid, bootstrap_reason = False, "not_attempted"
    try:
        recorder = EpisodeRecorder(env, identifier, source,
            context=PolicyContext(goal.id, goal.skill, goal.target, goal.pace, goal.tolerance, goal.timeout_seconds),
            seed=case["host_seed"], scenario_family=case["id"], split="training" if split == "training" else "evaluation",
            budget=RunBudget(40, 1800), source_root=root)
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
            if kind == "authored":
                frame = controller.step(FeedbackObservation.from_host(view), goal)
                supervision = controller._latest
                action = frame.action
                proposal = None if supervision.override is not None else action
                diagnostic = controller.diagnostic_state()
                lease = diagnostic["action_lease_ticks"]
                phase, reason = frame.phase, frame.reason
            elif kind == "idle":
                feedback = context.validate_observation(view, goal)
                supervision = supervisor.admit(feedback, goal, emergency=emergency_request(feedback))
                action = supervision.override if supervision.override is not None else WireAction(effort=0.)
                proposal = None if supervision.override is not None else action
                lease = supervision.action_lease_ticks
                phase, reason, diagnostic = "idle", supervision.reason, {}
            else:
                ledger.reserve("checkpoint_executions", stage=stage)
                decision = controller.step(view, goal, deterministic=stage != "ppo")
                supervision = decision.supervision
                action, proposal, lease = decision.action, decision.proposal, supervision.action_lease_ticks
                if decision.repeat_ticks != 6:
                    raise ContractError("jolt_repeat", "runtime repeat cannot expand or shorten the declared decision")
                history = decision.history
                phase, reason, diagnostic = "learned", supervision.reason, {}
            inference_ns = time.perf_counter_ns() - begin
            overridden = supervision.override is not None
            if execution_version == 2 and supervision.status != "running":
                raise SupervisorRetirement(supervision.reason)
            if supervision.status != "running" or not 6 <= lease <= 12 or view["tick"] + lease > context.deadline_tick:
                raise ContractError("jolt_supervisor_lease", "shared supervision must preserve original authority and deadline")
            packet = env.command(action.to_dict(), observation=view, lease_ticks=lease)
            # Preserve a proposal even when transport later fails; no fabricated receipt follows.
            note(journal, "decision_proposed", episode_id=identifier, index=index,
                 observation_sha256=sha256_json(view), invocation=context.to_dict(), history=history.to_dict(),
                 proposal=None if proposal is None else proposal.to_dict(), supervision=asdict(supervision),
                 dispatched_packet=packet, draw=None if decision is None else (_draw(decision, source_sha256).to_dict()
                     if decision.proposal is not None else None))
            after, _, terminated, truncated, info = recorder.step_packet(packet, replaced=overridden,
                intervention="model_override" if overridden else "none")
            event = recorder.events[-1]
            terms = task.consume(action, info["trace"], inference_ns=inference_ns)
            rewards.append(terms)
            cutoff = index + 1 == maximum_decisions and not task.done
            sample_event = replace(event, reward=terms["total"], terminated=task.succeeded or task.deadline_missed,
                                   truncated=cutoff)
            record = JoltDecisionRecord(sample_event, context.to_dict(), history.to_dict(), proposal,
                sha256_json(PhysicalRecoverySupervisor().specification()), supervision.reason, overridden,
                None if decision is None else _draw(decision, source_sha256),
                recurrent_reset=index == 0 or bool(decisions and decisions[-1].overridden))
            decisions.append(record)
            values.append(0. if decision is None or decision.value is None else decision.value)
            diagnostic.update(supervisor_intervention=overridden, supervisor_reason=supervision.reason)
            frames.append({"tick": view["tick"], "action": action.to_dict(), "phase": phase, "reason": reason,
                           "status": supervision.status, "inference_ns": inference_ns, "diagnostic_state": diagnostic})
            if decision is not None:
                controller.record_dispatch(decision, action, admitted=record.admitted, intervention=overridden)
            history = ActionHistory(proposal, action, overridden, record.admitted)
            if not record.admitted or task.integrity_failure:
                raise ContractError("jolt_native_admission", "native refusal or integrity failure ends the allocation")
            if task.done:
                break
            if terminated or truncated:
                raise ContractError("jolt_early_terminal", "native host ended before the fixed task deadline")
        if not task.done and stage != "ppo":
            raise ContractError("jolt_evaluation_cutoff", "uncompleted diagnostic cases are retained as incomplete")
        if execution_version == 2 and task.done:
            bootstrap_valid, bootstrap_reason = True, "task_terminal"
        if not task.done:
            # This observation was physically received before cancellation. Bootstrap
            # inference is charged and cannot advance the original controller/history.
            original_callback = controller.reserve
            peek = copy.deepcopy(controller, memo={id(original_callback): original_callback})
            peek.reserve = lambda counter, amount=1: ledger.reserve("bootstrap_forward", amount, stage=stage)
            if execution_version == 2:
                bootstrap_reason = "attempt_failed"
            bootstrap_decision = peek.step(recorder.view, goal, deterministic=True)
            bootstrap_observation_sha256 = sha256_json(recorder.view)
            bootstrap_valid = bootstrap_decision.proposal is not None and bootstrap_decision.value is not None
            bootstrap = float(bootstrap_decision.value) if bootstrap_valid else 0.
            bootstrap_reason = "received_cutoff_observation" if bootstrap_valid else "supervisor_value_unavailable"
        if task.deadline_missed:
            closure = bool(recorder.view.get("suspended"))
        else:
            from nca.controllers.morphology.conformance import cancellation_receipt
            before = recorder.view
            after = recorder.cancel("task_arrival" if task.done else "administrative_rollout_cutoff")
            cancellation_receipt(before, after)
            closure = True
        if not closure:
            raise ContractError("jolt_owner_closure", "native closure is required")
    except Exception as exc:
        caught = exc
        error = error_record(exc)
        if recorder is not None:
            try:
                before = recorder.view
                after = recorder.cancel("failed_episode_cleanup")
                if execution_version == 2:
                    from nca.controllers.morphology.conformance import cancellation_receipt
                    cancellation_receipt(before, after)
                    closure = True
                else:
                    closure = bool(recorder.view.get("suspended"))
            except Exception as cleanup_exc:
                if execution_version == 2:
                    closure = False
                    recorded = error_record(cleanup_exc)
                    cancellation_error = {key: recorded[key] for key in ("type", "code")}
    finally:
        if controller is not None:
            controller.cancel()
    if recorder is None or task is None:
        note(journal, "episode_failed_before_capture", id=identifier, error=error)
        raise ContractError("jolt_capture_start", "startup failure retained in original journal")
    metrics = task.metrics(administratively_capped=not task.done)
    capped = caught is not None and type(caught).__name__ == "AdmissionError"
    if error is not None and not capped:
        metrics["integrity_failure"] = True
    try:
        character = character_metrics(recorder.initial, goal.target, recorder.events, frames)
    except Exception as exc:
        character = None
        error = error or error_record(exc)
        caught = caught or exc
    status = "capped" if capped else "failed" if error else "completed" if task.done else "capped"
    sample_episode = JoltEpisode(identifier, split, tuple(decisions), status, metrics["reason"])
    validate_history(sample_episode)
    evaluation = {"metrics": metrics,
        "character_metrics": character, "reward_terms": rewards, "controller_frames": frames,
        "task_reward_schema": REWARD_SCHEMA, "owner_closure_confirmed": closure, "error": error}
    if execution_version == 2:
        evaluation["cancellation_error"] = cancellation_error
    raw_episode = recorder.finish(status, metrics["reason"], evaluation=evaluation)
    raw_path = output / "episodes" / f"{identifier}.json.gz"
    raw_sha = write_episode(raw_path, raw_episode)
    sync_directory(raw_path.parent)
    sample_path = output / "samples" / f"{identifier}.json.gz"
    sample_identity = write_sample_episode(sample_path, sample_episode)
    row = {"id": identifier, "case_id": case["id"], "controller_id": controller_id,
        "kind": kind, "stage": stage, "split": split, "host_seed": case["host_seed"],
        "profile_sha256": raw_episode.profile_sha256, "policy_sha256": policy_sha,
        "artifact": raw_path.relative_to(output).as_posix(), "artifact_sha256": raw_sha,
        "sample_artifact": sample_path.relative_to(output).as_posix(), "sample_sha256": sample_identity["sha256"],
        "decisions": len(decisions), "consumed_ticks": raw_episode.consumed_ticks,
        "policy_values": values if stage == "ppo" else None,
        "bootstrap_value": bootstrap if stage == "ppo" else None,
        "bootstrap_observation_sha256": bootstrap_observation_sha256,
        "bootstrap_observation_valid": bootstrap_valid,
        "bootstrap_reason": bootstrap_reason,
        "metrics": metrics, "character_metrics": character, "supervisor_interventions": sum(record.overridden for record in decisions),
        "owner_closure_confirmed": closure, "error": error}
    if execution_version == 2:
        row["cancellation_error"] = cancellation_error
    publish_json(output / "metrics" / f"{identifier}.json", row)
    ledger.reserve("episodes_aborted" if error else "episodes_completed", stage=stage, cleanup=True)
    note(journal, "episode_finished", row=row)
    if error:
        exc = caught or ContractError("jolt_capture_failed", "failed physical episode and original receipts retained")
        exc.row = row
        raise exc
    sequence = (rollout_sequence(sample_episode, values, bootstrap_value=bootstrap,
                                final_observation_valid=bootstrap_valid) if stage == "ppo" else None)
    return sample_episode, row, sequence


def _checkpoint(output, ledger, journal, checkpoints, policy, identifier, metadata, *, base=None,
                execution_version=1):
    ledger.reserve("checkpoint_writes")
    path = output / "checkpoints" / f"{identifier}.pt"
    if execution_version == 3:
        from nca.train.jolt_storage_v3 import save_policy_v3
        receipt = save_policy_v3(path, policy, base_policy=base,
            metadata={**metadata, "controller_id": identifier}, check=ledger.check)
    else:
        receipt = save_policy(path, policy, base_policy=base, metadata={**metadata, "controller_id": identifier})
    sync_directory(path.parent)
    row = {"id": identifier, "path": path.relative_to(output).as_posix(), **receipt}
    checkpoints.append(row)
    note(journal, "checkpoint_published", checkpoint=row)
    return row


def freeze_candidates(checkpoints, *, destination, source_sha256, journal, publisher=publish_json):
    """All six finals must exist before opening *any* sealed outcome."""
    expected = {f"{family}-seed{seed}-final" for family in ("bc", "residual") for seed in POLICY_SEEDS}
    finals = [row for row in checkpoints if row["id"] in expected]
    if len(finals) != 6 or {row["id"] for row in finals} != expected:
        raise ContractError("jolt_final_freeze", "six distinct, fixed final checkpoints are required")
    for row in finals:
        if file_identity(destination / row["path"])["sha256"] != row["file_sha256"]:
            raise ContractError("jolt_final_freeze", "a final checkpoint changed before the sealed boundary")
    receipt = {"schema": "jolt-six-final-freeze/v1", "source_sha256": source_sha256,
               "finals": finals, "selection": "fixed final epochs/updates; no seed or checkpoint selection"}
    publisher(destination / "frozen-finals.json", receipt)
    note(journal, "six_finals_frozen", receipt=receipt, sha256=sha256_json(receipt))
    return receipt


def close_worker_v2(native, owner, ledger, journal, *, status, failure, recording_errors):
    """Revoke the worker even when its journal can no longer record a note.

    Each cleanup action is independent. Failures remain explicit in the worker
    result and cannot become a verified journal or successful allocation.
    """
    def attempt(stage, action):
        nonlocal status, failure
        try:
            return action()
        except BaseException as exc:
            error = error_record(exc)
            recording_errors.append({"stage": stage, "error": error})
            status, failure = "failed", failure or error
            return None

    if native is not None:
        if owner is not None:
            attempt("owner_close", lambda: owner.request("close"))
        attempt("host_termination_reservation", lambda: ledger.reserve("host_terminations", cleanup=True))
        attempt("host_termination", lambda: native.close(graceful=False))
        exited = attempt("host_exit_readback", lambda: native._process.poll() is not None)
        attempt("host_terminated_note", lambda: note(journal, "host_terminated",
            native_close_confirmed=bool(owner and owner.closed_native), process_exited=exited is True))
    attempt("worker_closed_note", lambda: note(journal, "worker_closed", status=status, counters=ledger.snapshot()))
    attempt("journal_close", journal.close)
    durable = journal.durability_receipt()
    if not (durable.get("closed_verified") is True and durable.get("poisoned") is False):
        status = "failed"
        failure = failure or {"type": "ContractError", "code": "journal_closure_unverified"}
    return status, failure


def run(*, root, checkout, godot, output, config_path, declaration_path, whole_started,
        execution_version=1):
    from nca.train.jolt_admission import BudgetLedger, validate_config
    from nca.train.jolt_learning import BC_SPEC, PPO_SPEC, clone_sequences, train_clone, ppo_update
    root, output = Path(root).resolve(), Path(output).resolve()
    config = json.loads(Path(config_path).read_text())
    declaration = json.loads(Path(declaration_path).read_text())
    # Strict read-only admission precedes host construction and any model draw.
    validate_config(root, config)
    identity = inspect_host(checkout, godot)
    validate_execution(root, declaration, config=config, host_identity=identity,
                       execution_version=execution_version)
    if config["bc"] != BC_SPEC or config["ppo"] != PPO_SPEC:
        raise ContractError("jolt_learning_spec", "fitting/loss/sampling rules differ from their prospective configuration")
    if output.exists():
        raise ContractError("jolt_resume_forbidden", "an immutable new evidence directory is required; no retries")
    output.mkdir(parents=True)
    sync_directory(output.parent)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if execution_version == 3:
        from nca.train.jolt_budget_v3 import BudgetLedgerV3
        ledger = BudgetLedgerV3(config["budget"], started=whole_started,
                               spool_path=output / "reservation-ledger.jsonl")
    else:
        ledger = BudgetLedger(config["budget"], started=whole_started)
    publication_closed = False

    def publication_check():
        if publication_closed:
            ledger.original_clock.check("publication")
        else:
            ledger.check(cleanup=True)

    def publish(path, value, *, maximum_bytes=64 * 1024 * 1024):
        if execution_version == 3:
            from nca.train.jolt_storage_v3 import publish_json_v3
            return publish_json_v3(path, value, maximum_bytes=maximum_bytes, check=publication_check)
        return publish_json(path, value)

    ledger.reserve("attempts")
    publish(output / "declaration.json", declaration)
    publish(output / "config.json", config)
    run_schema = RUN_SCHEMA_V3 if execution_version == 3 else RUN_SCHEMA if execution_version == 1 else RUN_SCHEMA_V2
    manifest = {"schema": run_schema, "source_revision": declaration["source_revision"],
        "source_sha256": declaration["source_sha256"], "source_files": declaration["source_files"],
        "declaration_sha256": sha256_json(declaration), "config_sha256": sha256_json(config),
        "host_identity": identity, "runtime": declaration["runtime"],
        "whole_process_started_monotonic": whole_started, "created_utc": datetime.now(timezone.utc).isoformat()}
    publish(output / "manifest.json", manifest)
    results, checkpoints, training_results, demos = [], [], [], []
    compact_demos, demo_identities, secondary_positive_labels = [], [], 0
    status, failure, native, owner = "incomplete", None, None, None
    recording_errors = []
    if execution_version == 3:
        from nca.train.jolt_storage_v3 import BoundedVerifiedJournalV3
        journal = BoundedVerifiedJournalV3(output / "owner-journal.jsonl")
    elif execution_version == 2:
        from nca.evaluation.verified_journal import VerifiedDurableJournal
        journal = VerifiedDurableJournal(output / "owner-journal.jsonl")
    else:
        journal = DurableJournal(output / "owner-journal.jsonl")
    try:
        ledger.reserve_many({"host_starts": 1, "host_start_attempts": 1}, allowance_seconds=15)
        note(journal, "host_start_reserved", counts=ledger.snapshot())
        if execution_version == 3:
            from nca.train.jolt_native_v3 import BoundedGodotHostV3
            native = BoundedGodotHostV3(checkout, godot, lane="act004", mode="stepped", timeout=15,
                original_clock=ledger.original_clock, verified_identity=identity)
        else:
            native = GodotHost(checkout, godot, lane="act004", mode="stepped", timeout=15)
        owner = MeteredOwner(native, ledger, journal, execution_version=execution_version)
        note(journal, "host_started", identity=native.identity)
        for case in config["cases"]["teacher"]:
            episode, row, _ = capture(owner, ledger, journal, case, identifier=f"teacher-{case['id']}",
                controller_id="authored", kind="authored", stage="teacher", output=output, root=root,
                config=config, source_sha256=declaration["source_sha256"], execution_version=execution_version)
            if execution_version == 3:
                from nca.train.jolt_learning import convert_clone_episode_v3
                compact_demos.append(convert_clone_episode_v3(episode,
                    expected_teacher_sha256=JoltRecoveryControllerV4().specification_sha256,
                    expected_supervisor_sha256=sha256_json(PhysicalRecoverySupervisor().specification())))
                demo_identities.append({"id": episode.id, "sha256": episode.sha256})
                secondary_positive_labels += sum(record.imitation_eligible and record.proposal.secondary
                                                  for record in episode.decisions)
                del episode
            else:
                demos.append(episode)
            results.append(row)
            print(f"teacher episodes={len(demo_identities) if execution_version == 3 else len(demos)}/{len(config['cases']['teacher'])} decisions={ledger.counts['decisions']}", flush=True)
        converted = (tuple(compact_demos) if execution_version == 3 else clone_sequences(demos,
            expected_teacher_sha256=JoltRecoveryControllerV4().specification_sha256,
            expected_supervisor_sha256=sha256_json(PhysicalRecoverySupervisor().specification())))
        if execution_version == 3 and not any(item.eligible for episode in converted for item in episode.rows):
            raise ContractError("jolt_teacher_labels", "no eligible ordinary teacher data; no fitting is admitted")
        dataset = {"schema": "jolt-demonstrations/v3" if execution_version == 3 else "jolt-demonstrations/v2",
                   "episodes": demo_identities if execution_version == 3 else [{"id": episode.id, "sha256": episode.sha256} for episode in demos],
                   "eligible_samples": sum(row.eligible for episode in converted for row in episode.rows),
                   "fixed_windows_per_epoch": sum(math.ceil(len(episode.rows) / 16) for episode in converted),
                   "unsuccessful_episodes_retained": sum(not row["metrics"]["success"] for row in results),
                   "secondary_positive_labels": secondary_positive_labels if execution_version == 3 else
                       sum(record.imitation_eligible and record.proposal.secondary
                       for episode in demos for record in episode.decisions), "selection": "all eligible training rows, with no success filter"}
        if execution_version == 3:
            dataset["safe_retirements"] = sum(row["outcome"]["kind"] == "safe_retirement" for row in results)
        if dataset["secondary_positive_labels"] != 0:
            raise ContractError("jolt_teacher_labels", "qualified v4 provides no positive secondary demonstrations")
        publish(output / "dataset.json", dataset)
        note(journal, "fitting_shape_frozen", dataset_sha256=sha256_json(dataset),
             bc_optimizer_bound=3 * 40 * dataset["fixed_windows_per_epoch"])
        metadata = {"config_sha256": sha256_json(config), "source_sha256": declaration["source_sha256"],
                    "dataset_sha256": sha256_json(dataset), "profile_sha256": config["profile_sha256"],
                    "source_revision": declaration["source_revision"]}
        # Fit the fixed BC family first. All three initializations and optimizers
        # remain independent; shared data supplies no seed selection.
        bc_final = {}
        for seed in POLICY_SEEDS:
            ledger.check()
            torch.manual_seed(seed)
            policy = JoltPolicy("bc")
            _checkpoint(output, ledger, journal, checkpoints, policy, f"bc-seed{seed}-initial",
                {**metadata, "seed": seed, "phase": "initial"}, execution_version=execution_version)
            note(journal, "bc_fitting_started", seed=seed, counters=ledger.snapshot(),
                 reserved_upper_bound=40 * dataset["fixed_windows_per_epoch"])
            fit = train_clone(policy, converted, seed=seed, reserve=ledger.reserve)
            fit.pop("optimizer", None)
            fit.update(seed=seed, family="bc")
            training_results.append(fit)
            publish(output / "learning" / f"bc-seed{seed}.json", fit,
                    maximum_bytes=16 * 1024 if execution_version == 3 else 1024 * 1024)
            _checkpoint(output, ledger, journal, checkpoints, policy, f"bc-seed{seed}-final",
                {**metadata, "seed": seed, "phase": "final", "epoch": 40}, execution_version=execution_version)
            bc_final[seed] = copy.deepcopy(policy).eval().requires_grad_(False)
            note(journal, "bc_fitting_finished", seed=seed, counters=ledger.snapshot(), result=fit)
            print(f"BC seed={seed} epoch=40 complete", flush=True)
        for seed in POLICY_SEEDS:
            ledger.check()
            torch.manual_seed(seed + 100000)
            residual = JoltPolicy("residual")
            base = bc_final[seed]
            generator = torch.Generator().manual_seed(seed + 200000)
            order_rng = random.Random(seed + 300000)
            case_cursor = 0
            case_lookup = {case["id"]: case for case in config["cases"]["teacher"]}
            optimizer = torch.optim.Adam(residual.parameters(), lr=PPO_SPEC["learning_rate"])
            _checkpoint(output, ledger, journal, checkpoints, residual, f"residual-seed{seed}-initial",
                {**metadata, "seed": seed, "phase": "initial"}, base=base, execution_version=execution_version)
            for update in range(8):
                sequences, decision_count, episode_count = [], 0, 0
                while decision_count < 256:
                    if execution_version == 3:
                        if episode_count >= 256 or case_cursor >= 2048:
                            raise ContractError("jolt_ppo_starts", "fixed PPO stream/start ceiling exhausted")
                        case = case_lookup[config["ppo_case_stream"]["seeds"][str(seed)][case_cursor]]
                        stream_entry = {"seed": seed, "index": case_cursor, "update": update}
                        case_cursor += 1
                    else:
                        case = order_rng.choice(config["cases"]["teacher"])
                        stream_entry = None
                    identifier = f"ppo-seed{seed}-u{update}-e{episode_count}"
                    _, row, sequence = capture(owner, ledger, journal, case, identifier=identifier,
                        controller_id=f"residual-seed{seed}-initial" if update == 0 else f"residual-seed{seed}-u{update}",
                        kind="residual", stage="ppo", output=output, root=root, config=config,
                        source_sha256=declaration["source_sha256"], policy=residual, base=base, generator=generator,
                        maximum_decisions=min(40, 256 - decision_count), execution_version=execution_version,
                        case_stream_entry=stream_entry)
                    if row["decisions"] <= 0 or sequence is None:
                        raise ContractError("jolt_ppo_empty", "a counted rollout must contain retained physical decisions")
                    results.append(row)
                    sequences.append(sequence)
                    decision_count += row["decisions"]
                    episode_count += 1
                note(journal, "ppo_fitting_started", seed=seed, update=update, counters=ledger.snapshot())
                if execution_version == 3:
                    from nca.train.jolt_learning import ppo_update_v3
                    if decision_count != 256:
                        raise ContractError("jolt_ppo_decisions", "fixed update must contain exactly 256 confirmed decisions")
                    fit_function = ppo_update_v3
                else:
                    fit_function = ppo_update
                fit = fit_function(residual, base, optimizer, sequences, seed=seed + update,
                                 reserve=ledger.reserve, source_sha256=declaration["source_sha256"])
                fit.update(seed=seed, family="residual", update=update + 1)
                training_results.append(fit)
                publish(output / "learning" / f"residual-seed{seed}-u{update+1}.json", fit,
                        maximum_bytes=16 * 1024 if execution_version == 3 else 1024 * 1024)
                _checkpoint(output, ledger, journal, checkpoints, residual, f"residual-seed{seed}-u{update+1}",
                    {**metadata, "seed": seed, "phase": "update", "completed_updates": update + 1},
                    base=base, execution_version=execution_version)
                note(journal, "ppo_fitting_finished", seed=seed, update=update, counters=ledger.snapshot(), result=fit)
                print(f"PPO seed={seed} update={update+1}/8 decisions={(update+1)*256}", flush=True)
            _checkpoint(output, ledger, journal, checkpoints, residual, f"residual-seed{seed}-final",
                {**metadata, "seed": seed, "phase": "final", "completed_updates": 8},
                base=base, execution_version=execution_version)
        freeze = freeze_candidates(checkpoints, destination=output,
            source_sha256=declaration["source_sha256"], journal=journal, publisher=publish)
        policies = {}
        for item in checkpoints:
            if not item["id"].endswith(("-initial", "-final")):
                continue
            ledger.reserve("checkpoint_loads")
            policy, base, _ = load_policy(output / item["path"], expected_config_sha256=sha256_json(config))
            policies[item["id"]] = (policy, base)
        for stage in ("open", "sealed"):
            candidates = [("authored", "authored", None, None), ("idle", "idle", None, None)]
            candidates.extend((name, policy.kind, policy, base) for name, (policy, base) in policies.items()
                              if stage == "sealed" or name.endswith("-final"))
            if execution_version == 3:
                planned = [{"case_id": case["id"], "controller_id": name}
                           for case in config["cases"][stage] for name, _, _, _ in candidates]
                if planned != config["comparison_order"][stage]:
                    raise ContractError("jolt_comparison_order", "case/controller order changed before opening outcomes")
            if stage == "sealed":
                note(journal, "sealed_outcomes_opened", frozen_finals_sha256=sha256_json(freeze))
                publish(output / "sealed-opened.json", {"frozen_finals_sha256": sha256_json(freeze),
                    "source_sha256": declaration["source_sha256"], "before_first_sealed_outcome": True})
            for case in config["cases"][stage]:
                for name, kind, policy, base in candidates:
                    _, row, _ = capture(owner, ledger, journal, case, identifier=f"{stage}-{case['id']}-{name}",
                        controller_id=name, kind=kind, stage=stage, output=output, root=root, config=config,
                        source_sha256=declaration["source_sha256"], policy=policy, base=base,
                        execution_version=execution_version)
                    results.append(row)
                print(f"{stage} case={case['id']} complete decisions={ledger.counts['decisions']}", flush=True)
        status = "completed"
    except BaseException as exc:
        if execution_version == 1 and not isinstance(exc, Exception):
            raise
        failure = error_record(exc)
        if getattr(exc, "row", None) is not None:
            results.append(exc.row)
        exhausted = type(exc).__name__ == "AdmissionError"
        if execution_version == 3:
            from nca.train.jolt_budget_v3 import budget_exhaustion_v3
            exhausted = exhausted or budget_exhaustion_v3(exc)
        status = "capped" if exhausted else "failed"
        try:
            note(journal, "allocation_stopped", status=status, error=failure, counters=ledger.snapshot())
        except BaseException as recording_exc:
            if execution_version == 1:
                raise
            recording_errors.append({"stage": "allocation_stopped_note", "error": error_record(recording_exc)})
            status = "failed"
    finally:
        if execution_version == 3:
            try:
                if ledger.phase == "active":
                    ledger.enter_phase("closure")
                elif ledger.phase != "closure":
                    raise ContractError("jolt_v3_phase", "worker cleanup must occur in its reserved closure interval")
            except BaseException as exc:
                recording_errors.append({"stage": "worker_closure_deadline", "error": error_record(exc)})
                status, failure = "failed", failure or error_record(exc)
        if execution_version in (2, 3):
            status, failure = close_worker_v2(native, owner, ledger, journal, status=status,
                                             failure=failure, recording_errors=recording_errors)
        elif native is not None:
            try:
                if owner is not None:
                    owner.request("close")
            except Exception as exc:
                failure = failure or error_record(exc)
                status = "failed"
            try:
                ledger.reserve("host_terminations", cleanup=True)
            except Exception as exc:
                failure = failure or error_record(exc)
                status = "failed"
            finally:
                # Revocation is mandatory even if a cleanup deadline itself
                # fails; such a late close can never qualify this allocation.
                native.close(graceful=False)
            note(journal, "host_terminated", native_close_confirmed=bool(owner and owner.closed_native),
                 process_exited=native._process.poll() is not None)
        if execution_version == 1:
            note(journal, "worker_closed", status=status, counters=ledger.snapshot())
            journal.close()
    closure = {"native_close_confirmed": bool(owner and owner.closed_native),
               "pending_owner_request": None if owner is None else owner.pending,
               "uncertain_owner_requests": [] if owner is None else owner.uncertain,
               "owner_requests": {} if owner is None else owner.requests,
               "requested_ticks": 0 if owner is None else owner.requested_ticks,
               "confirmed_ticks": 0 if owner is None else owner.confirmed_ticks,
               "refused_ticks": 0 if owner is None else owner.refused_ticks,
               "uncertain_ticks": 0 if owner is None else owner.uncertain_ticks,
               "durability": journal.durability_receipt()}
    if execution_version in (2, 3):
        closure["recording_errors"] = recording_errors
    if execution_version == 3:
        ledger.publish_reservations(output / "budget-reservations.json")
        from nca.evaluation.jolt_learning import qualify_v3
        qualification_function = qualify_v3
    else:
        publish(output / "budget-reservations.json", {"schema": "jolt-budget-reservations/v1",
                                                           "reservations": ledger.reservations})
        qualification_function = qualify
    candidate = {"schema": run_schema, "status": status, "error": failure, "manifest": manifest,
        "results": results, "checkpoints": checkpoints, "learning": training_results,
        "costs_before_readback": ledger.snapshot(), "closure": closure,
        "qualification": qualification_function(results, config["cases"]["sealed"], evidence_complete=False)}
    if execution_version == 3:
        from nca.train.jolt_storage_v3 import validate_worker_result_v3
        validate_worker_result_v3(candidate, check=publication_check)
    publish(output / "worker-result.json", candidate)
    # An independently implemented verifier receives only immutable bytes. It
    # cannot obtain the live host, fit weights, or alter the declared outcomes.
    from nca.evaluation.jolt_learning_audit import verify_output
    if execution_version == 3:
        ledger.enter_phase("audit")
    audit_options = {"check": lambda: ledger.check(cleanup=True)} if execution_version == 3 else {}
    audit = verify_output(output, reserve=lambda name, amount=1: ledger.reserve(name, amount, cleanup=True), **audit_options)
    if execution_version == 3:
        ledger.check(cleanup=True)
        ledger.enter_phase("publication")
    publish(output / "independent-readback.json", audit)
    if execution_version == 3:
        ledger.publish_reservations(output / "closed-budget-reservations.json")
    else:
        publish(output / "closed-budget-reservations.json", {"schema": "jolt-budget-reservations/v1",
                                                                  "reservations": ledger.reservations})
    closed_costs = ledger.close()
    publication_closed = True
    final = {"schema": run_schema, "status": status, "error": failure,
             "worker_result_sha256": file_identity(output / "worker-result.json")["sha256"],
             "audit_sha256": file_identity(output / "independent-readback.json")["sha256"],
             "costs": closed_costs, "closure": closure,
             "qualification": qualification_function(results, config["cases"]["sealed"], evidence_complete=audit["verified"] and audit["complete"] and status == "completed")}
    publish(output / "result.json", final)
    # Exact inventory is sealed only after all publications exist. The external
    # supervisor independently hashes this inventory inside the same deadline.
    if execution_version == 3:
        from nca.train.jolt_storage_v3 import file_identity_v3
        inventory = {path.relative_to(output).as_posix(): file_identity_v3(path, check=publication_check)
                     for path in sorted(output.rglob("*")) if path.is_file()}
    else:
        inventory = {path.relative_to(output).as_posix(): file_identity(path)
                     for path in sorted(output.rglob("*")) if path.is_file()}
    publish(output / "inventory.json", {"schema": "jolt-closed-inventory/v1", "files": inventory})
    if execution_version == 3:
        ledger.original_clock.check("publication")
    if time.monotonic() >= ledger.deadline:
        raise ContractError("jolt_finalization_deadline", "publication exceeded the original whole-process deadline")
    print(canonical_json({"status": status, "qualified": final["qualification"]["qualified"],
                          "decisions": ledger.counts["decisions"]}), flush=True)
    return final
