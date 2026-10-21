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
    ActionHistory, InvocationContext, JoltController, JoltPolicy, emergency_request,
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


def runtime_identity():
    """Public runtime/hardware facts only; no node names or local paths."""
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {"torch": torch.__version__, "numpy": numpy.__version__, "device": "cpu", "threads": 1,
            "interop_threads": 1, "python": platform.python_version(), "system": platform.system(),
            "kernel": platform.release(), "machine": platform.machine(), "libc": list(platform.libc_ver()),
            "cpu_model": cpu_model, "logical_cpus_available": os.cpu_count(),
            "deterministic_algorithms": True}


def error_record(exc):
    # Exception text may include an absolute path or private process details.
    return {"type": type(exc).__name__, "code": getattr(exc, "code", None)}


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


def validate_execution(root, declaration, *, config, host_identity):
    if declaration.get("schema") != DECLARATION_SCHEMA or declaration.get("attempts") != 1:
        raise ContractError("jolt_execution", "exactly one versioned prospective invocation is required")
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
        if record.get("accepted") is not True or record.get("source_sha256") != declaration["source_sha256"]:
            raise ContractError("jolt_execution_review", "reviews must accept exactly the executing source closure")
        reviewer = record.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer or reviewer in reviewers:
            raise ContractError("jolt_execution_review", "each source review requires a distinct independent reviewer")
        reviewers.add(reviewer)


class MeteredOwner:
    """One pending exchange, exact source packets and a whole-exchange deadline."""

    def __init__(self, host, ledger, journal, *, transport_seconds=15):
        self.host, self.ledger, self.journal = host, ledger, journal
        self.transport_seconds = transport_seconds
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
            maximum_decisions=40):
    """One invocation; preserve every unsuccessful label and partial tail."""
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
                                 reserve=lambda counter, amount=1: ledger.reserve(counter, amount, stage=stage))
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
        if not task.done:
            # This observation was physically received before cancellation. Bootstrap
            # inference is charged and cannot advance the original controller/history.
            original_callback = controller.reserve
            peek = copy.deepcopy(controller, memo={id(original_callback): original_callback})
            peek.reserve = lambda counter, amount=1: ledger.reserve("bootstrap_forward", amount, stage=stage)
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
                recorder.cancel("failed_episode_cleanup")
                closure = bool(recorder.view.get("suspended"))
            except Exception:
                pass
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
    raw_episode = recorder.finish(status, metrics["reason"], evaluation={"metrics": metrics,
        "character_metrics": character, "reward_terms": rewards, "controller_frames": frames,
        "task_reward_schema": REWARD_SCHEMA, "owner_closure_confirmed": closure, "error": error})
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


def _checkpoint(output, ledger, journal, checkpoints, policy, identifier, metadata, *, base=None):
    ledger.reserve("checkpoint_writes")
    path = output / "checkpoints" / f"{identifier}.pt"
    receipt = save_policy(path, policy, base_policy=base, metadata={**metadata, "controller_id": identifier})
    sync_directory(path.parent)
    row = {"id": identifier, "path": path.relative_to(output).as_posix(), **receipt}
    checkpoints.append(row)
    note(journal, "checkpoint_published", checkpoint=row)
    return row


def freeze_candidates(checkpoints, *, destination, source_sha256, journal):
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
    publish_json(destination / "frozen-finals.json", receipt)
    note(journal, "six_finals_frozen", receipt=receipt, sha256=sha256_json(receipt))
    return receipt


def run(*, root, checkout, godot, output, config_path, declaration_path, whole_started):
    from nca.train.jolt_admission import BudgetLedger, validate_config
    from nca.train.jolt_learning import BC_SPEC, PPO_SPEC, clone_sequences, train_clone, ppo_update
    root, output = Path(root).resolve(), Path(output).resolve()
    config = json.loads(Path(config_path).read_text())
    declaration = json.loads(Path(declaration_path).read_text())
    # Strict read-only admission precedes host construction and any model draw.
    validate_config(root, config)
    identity = inspect_host(checkout, godot)
    validate_execution(root, declaration, config=config, host_identity=identity)
    if config["bc"] != BC_SPEC or config["ppo"] != PPO_SPEC:
        raise ContractError("jolt_learning_spec", "fitting/loss/sampling rules differ from their prospective configuration")
    if output.exists():
        raise ContractError("jolt_resume_forbidden", "an immutable new evidence directory is required; no retries")
    output.mkdir(parents=True)
    sync_directory(output.parent)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    ledger = BudgetLedger(config["budget"], started=whole_started)
    ledger.reserve("attempts")
    publish_json(output / "declaration.json", declaration)
    publish_json(output / "config.json", config)
    manifest = {"schema": RUN_SCHEMA, "source_revision": declaration["source_revision"],
        "source_sha256": declaration["source_sha256"], "source_files": declaration["source_files"],
        "declaration_sha256": sha256_json(declaration), "config_sha256": sha256_json(config),
        "host_identity": identity, "runtime": declaration["runtime"],
        "whole_process_started_monotonic": whole_started, "created_utc": datetime.now(timezone.utc).isoformat()}
    publish_json(output / "manifest.json", manifest)
    results, checkpoints, training_results, demos = [], [], [], []
    status, failure, native, owner = "incomplete", None, None, None
    journal = DurableJournal(output / "owner-journal.jsonl")
    try:
        ledger.reserve_many({"host_starts": 1, "host_start_attempts": 1}, allowance_seconds=15)
        note(journal, "host_start_reserved", counts=ledger.snapshot())
        native = GodotHost(checkout, godot, lane="act004", mode="stepped", timeout=15)
        owner = MeteredOwner(native, ledger, journal)
        note(journal, "host_started", identity=native.identity)
        for case in config["cases"]["teacher"]:
            episode, row, _ = capture(owner, ledger, journal, case, identifier=f"teacher-{case['id']}",
                controller_id="authored", kind="authored", stage="teacher", output=output, root=root,
                config=config, source_sha256=declaration["source_sha256"])
            demos.append(episode)
            results.append(row)
            print(f"teacher episodes={len(demos)}/{len(config['cases']['teacher'])} decisions={ledger.counts['decisions']}", flush=True)
        converted = clone_sequences(demos, expected_teacher_sha256=JoltRecoveryControllerV4().specification_sha256,
            expected_supervisor_sha256=sha256_json(PhysicalRecoverySupervisor().specification()))
        dataset = {"schema": "jolt-demonstrations/v2", "episodes": [{"id": episode.id, "sha256": episode.sha256} for episode in demos],
                   "eligible_samples": sum(row.eligible for episode in converted for row in episode.rows),
                   "fixed_windows_per_epoch": sum(math.ceil(len(episode.rows) / 16) for episode in converted),
                   "unsuccessful_episodes_retained": sum(not row["metrics"]["success"] for row in results),
                   "secondary_positive_labels": sum(record.imitation_eligible and record.proposal.secondary
                       for episode in demos for record in episode.decisions), "selection": "all eligible training rows, with no success filter"}
        if dataset["secondary_positive_labels"] != 0:
            raise ContractError("jolt_teacher_labels", "qualified v4 provides no positive secondary demonstrations")
        publish_json(output / "dataset.json", dataset)
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
            _checkpoint(output, ledger, journal, checkpoints, policy, f"bc-seed{seed}-initial", {**metadata, "seed": seed, "phase": "initial"})
            note(journal, "bc_fitting_started", seed=seed, counters=ledger.snapshot(),
                 reserved_upper_bound=40 * dataset["fixed_windows_per_epoch"])
            fit = train_clone(policy, converted, seed=seed, reserve=ledger.reserve)
            fit.pop("optimizer", None)
            fit.update(seed=seed, family="bc")
            training_results.append(fit)
            publish_json(output / "learning" / f"bc-seed{seed}.json", fit)
            _checkpoint(output, ledger, journal, checkpoints, policy, f"bc-seed{seed}-final", {**metadata, "seed": seed, "phase": "final", "epoch": 40})
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
            optimizer = torch.optim.Adam(residual.parameters(), lr=PPO_SPEC["learning_rate"])
            _checkpoint(output, ledger, journal, checkpoints, residual, f"residual-seed{seed}-initial",
                {**metadata, "seed": seed, "phase": "initial"}, base=base)
            for update in range(8):
                sequences, decision_count, episode_count = [], 0, 0
                while decision_count < 256:
                    case = order_rng.choice(config["cases"]["teacher"])
                    identifier = f"ppo-seed{seed}-u{update}-e{episode_count}"
                    _, row, sequence = capture(owner, ledger, journal, case, identifier=identifier,
                        controller_id=f"residual-seed{seed}-initial" if update == 0 else f"residual-seed{seed}-u{update}",
                        kind="residual", stage="ppo", output=output, root=root, config=config,
                        source_sha256=declaration["source_sha256"], policy=residual, base=base, generator=generator,
                        maximum_decisions=min(40, 256 - decision_count))
                    if row["decisions"] <= 0 or sequence is None:
                        raise ContractError("jolt_ppo_empty", "a counted rollout must contain retained physical decisions")
                    results.append(row)
                    sequences.append(sequence)
                    decision_count += row["decisions"]
                    episode_count += 1
                note(journal, "ppo_fitting_started", seed=seed, update=update, counters=ledger.snapshot())
                fit = ppo_update(residual, base, optimizer, sequences, seed=seed + update,
                                 reserve=ledger.reserve, source_sha256=declaration["source_sha256"])
                fit.update(seed=seed, family="residual", update=update + 1)
                training_results.append(fit)
                publish_json(output / "learning" / f"residual-seed{seed}-u{update+1}.json", fit)
                _checkpoint(output, ledger, journal, checkpoints, residual, f"residual-seed{seed}-u{update+1}",
                    {**metadata, "seed": seed, "phase": "update", "completed_updates": update + 1}, base=base)
                note(journal, "ppo_fitting_finished", seed=seed, update=update, counters=ledger.snapshot(), result=fit)
                print(f"PPO seed={seed} update={update+1}/8 decisions={(update+1)*256}", flush=True)
            _checkpoint(output, ledger, journal, checkpoints, residual, f"residual-seed{seed}-final",
                {**metadata, "seed": seed, "phase": "final", "completed_updates": 8}, base=base)
        freeze = freeze_candidates(checkpoints, destination=output, source_sha256=declaration["source_sha256"], journal=journal)
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
            if stage == "sealed":
                note(journal, "sealed_outcomes_opened", frozen_finals_sha256=sha256_json(freeze))
                publish_json(output / "sealed-opened.json", {"frozen_finals_sha256": sha256_json(freeze),
                    "source_sha256": declaration["source_sha256"], "before_first_sealed_outcome": True})
            for case in config["cases"][stage]:
                for name, kind, policy, base in candidates:
                    _, row, _ = capture(owner, ledger, journal, case, identifier=f"{stage}-{case['id']}-{name}",
                        controller_id=name, kind=kind, stage=stage, output=output, root=root, config=config,
                        source_sha256=declaration["source_sha256"], policy=policy, base=base)
                    results.append(row)
                print(f"{stage} case={case['id']} complete decisions={ledger.counts['decisions']}", flush=True)
        status = "completed"
    except Exception as exc:
        failure = error_record(exc)
        if getattr(exc, "row", None) is not None:
            results.append(exc.row)
        status = "capped" if type(exc).__name__ == "AdmissionError" else "failed"
        note(journal, "allocation_stopped", status=status, error=failure, counters=ledger.snapshot())
    finally:
        if native is not None:
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
    publish_json(output / "budget-reservations.json", {"schema": "jolt-budget-reservations/v1",
                                                       "reservations": ledger.reservations})
    candidate = {"schema": RUN_SCHEMA, "status": status, "error": failure, "manifest": manifest,
        "results": results, "checkpoints": checkpoints, "learning": training_results,
        "costs_before_readback": ledger.snapshot(), "closure": closure,
        "qualification": qualify(results, config["cases"]["sealed"], evidence_complete=False)}
    publish_json(output / "worker-result.json", candidate)
    # An independently implemented verifier receives only immutable bytes. It
    # cannot obtain the live host, fit weights, or alter the declared outcomes.
    from nca.evaluation.jolt_learning_audit import verify_output
    audit = verify_output(output, reserve=lambda name, amount=1: ledger.reserve(name, amount, cleanup=True))
    publish_json(output / "independent-readback.json", audit)
    publish_json(output / "closed-budget-reservations.json", {"schema": "jolt-budget-reservations/v1",
                                                              "reservations": ledger.reservations})
    closed_costs = ledger.close()
    final = {"schema": RUN_SCHEMA, "status": status, "error": failure,
             "worker_result_sha256": file_identity(output / "worker-result.json")["sha256"],
             "audit_sha256": file_identity(output / "independent-readback.json")["sha256"],
             "costs": closed_costs, "closure": closure,
             "qualification": qualify(results, config["cases"]["sealed"], evidence_complete=audit["verified"] and audit["complete"] and status == "completed")}
    publish_json(output / "result.json", final)
    # Exact inventory is sealed only after all publications exist. The external
    # supervisor independently hashes this inventory inside the same deadline.
    inventory = {path.relative_to(output).as_posix(): file_identity(path) for path in sorted(output.rglob("*")) if path.is_file()}
    publish_json(output / "inventory.json", {"schema": "jolt-closed-inventory/v1", "files": inventory})
    if time.monotonic() >= ledger.deadline:
        raise ContractError("jolt_finalization_deadline", "publication exceeded the original whole-process deadline")
    print(canonical_json({"status": status, "qualified": final["qualification"]["qualified"],
                          "decisions": ledger.counts["decisions"]}), flush=True)
    return final
