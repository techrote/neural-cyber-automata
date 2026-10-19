"""One fixed actual-host Jolt BC follow-up; no automatic training extension."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import torch

from nca.adapters.godot import GodotHost
from nca.contracts import ContractError, sha256_json
from nca.controllers.goal_frame import (GoalFrameController, GoalFramePolicy, load_goal_frame_policy,
                                       transform_sequences)
from nca.controllers.learned import policy_state_sha256
from nca.controllers.morphology import MorphologyAdapter
from nca.controllers.morphology.conformance import cancellation_receipt, validate_interval, _error, _same_action
from nca.data import Dataset
from nca.data.capture import source_identity
from nca.data.storage import read_episode
from nca.randomness import ScenarioSplit
from .artifacts import freeze_manifest, immutable_json, write_checkpoint
from .goal_frame_config import (FOLLOWUP_SHA256, read_followup_config, summarize_prerequisite,
                                verify_original_boundary)
from .imitation import clone_sequences
from .recurrent import train_clone
from .screen import ScreenBudget, ScreenBudgetExceeded, capture_task


def _portable_error(exc: BaseException) -> dict[str, str]:
    value = _error(exc)
    value["message"] = value.pop("detail")
    return value


def audit_episode_boundaries(episode, expected_profile_sha256: str) -> dict[str, Any]:
    """Validate original source, consumed requests and actual owner revocation."""
    adapter = MorphologyAdapter(episode.profile, expected_sha256=expected_profile_sha256)
    errors, decisions, cancellations = [], 0, 0
    previous = episode.initial_observation
    for event in episode.events:
        try:
            if sha256_json(previous) != sha256_json(event.issued_observation):
                raise ContractError("goal_frame_source", "every decision must continue from the preceding actual owner receipt")
            if event.kind == "step":
                decisions += 1
                if (not event.imitation_eligible or event.intervention != "none"
                        or sha256_json(event.source_observation) != sha256_json(event.issued_observation)):
                    raise ContractError("goal_frame_source", "complete original unintervened request attribution required")
                source = adapter.identity(event.source_observation)
                terminal = event.terminated or event.truncated
                following = (adapter.terminal_identity(event.next_observation) if terminal
                             else adapter.identity(event.next_observation))
                if (source.lifetime != following.lifetime
                        or following.control_generation != source.control_generation + int(terminal)
                        or (not terminal and source.topology_sha256 != following.topology_sha256)):
                    raise ContractError("goal_frame_source", "an unexpected owner or physical lifetime changed")
                for tick in event.ticks:
                    consumed = tick.consumed_request
                    if (consumed is None or set(consumed) != set(event.requested_packet)
                            or any(consumed[key] != event.requested_packet[key] for key in consumed if key != "action")
                            or not _same_action(event.requested_packet["action"], consumed.get("action"))
                            or (tick.applied.get("action_available") is True
                                and not _same_action(event.requested_packet["action"], tick.consumed_action))):
                        raise ContractError("goal_frame_consumption", "every consumed source must retain the original request identity")
                trace = [{"tick_before": tick.tick_before, "tick_after": tick.tick_after,
                          "requested": tick.requested, "admission": tick.admission,
                          "failed": not tick.valid, "applied": tick.applied} for tick in event.ticks]
                validate_interval(event.requested_packet, event.source_observation, event.next_observation,
                                  event.terminated, event.truncated,
                                  {"actual_ticks": len(trace), "elapsed_ticks": event.tick_after - event.tick_before,
                                   "delay_ticks_before_repeat": 0, "trace": trace,
                                   "consumed_trace_complete": event.trace_complete}, allow_terminal=True)
            elif event.kind == "cancel":
                cancellations += 1
                cancellation_receipt(event.issued_observation, event.next_observation)
            else:
                raise ContractError("goal_frame_boundary", "unplanned capture boundary in the fixed autonomous episode")
        except (ContractError, KeyError, TypeError, ValueError) as exc:
            errors.append({"event_index": event.index, **_portable_error(exc)})
        previous = event.next_observation
    if episode.events:
        last = episode.events[-1]
        closed = (cancellations == 1 and last.kind == "cancel") or (
            cancellations == 0 and last.kind == "step" and (last.terminated or last.truncated))
        if not closed:
            errors.append({"event_index": last.index, "type": "ContractError", "code": "goal_frame_owner_closure",
                           "message": "one final owner cancellation or the intrinsic terminal must close the invocation"})
    return {"episode_id": episode.id, "record_sha256": episode.sha256,
            "decisions": decisions, "completed_ticks": episode.consumed_ticks,
            "cancellations": cancellations, "source_boundary_regressions": len(errors),
            "passed": not errors and decisions > 0, "errors": errors}


def _verify_episode_files(output: Path, results) -> None:
    for row in results:
        path = output / row["artifact"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["artifact_sha256"]:
            raise ContractError("goal_frame_artifact", "a preserved episode artifact changed")


def verify_checkpoint_file(output: Path, receipt) -> Path:
    path = output / "checkpoints" / receipt["file"]
    data = path.read_bytes()
    if len(data) != receipt["bytes"] or hashlib.sha256(data).hexdigest() != receipt["sha256"]:
        raise ContractError("goal_frame_checkpoint", "saved checkpoint bytes differ from their frozen receipt")
    return path


def verify_published_files(root: Path, revision: str, files) -> None:
    """Every executed input must be a tracked, byte-identical published blob."""
    try:
        subprocess.run(["git", "ls-files", "--error-unmatch", "--", *files], cwd=root,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        for relative, expected in files.items():
            blob = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=root,
                                           stderr=subprocess.DEVNULL)
            if (hashlib.sha256(blob).hexdigest() != expected
                    or hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected):
                raise ContractError("goal_frame_source", "executed source differs from its published Git blob")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("goal_frame_source", "all executed sources and declarations must be tracked at the freeze") from exc


def run_followup(*, root: Path, checkout: Path, godot: Path, config_path: Path, output: Path):
    """Freeze inputs, collect training data, fit six policies, then open new cases."""
    config = read_followup_config(config_path)
    verify_original_boundary(config, root)
    output.mkdir(parents=True, exist_ok=False)
    budget = ScreenBudget(config["budget"]["maximum_total_decisions"], config["budget"]["maximum_wall_seconds"])
    results, checkpoints, attempts, learning, fitting_attempts = [], [], [], [], []
    manifest, error = None, None
    status, phase, host_starts = "incomplete", "source_admission", 0
    try:
        if subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=root, text=True).strip():
            raise ContractError("goal_frame_source", "the published source freeze must be clean before fitting")
        revision, package_sources, _ = source_identity(root)
        published_inputs = dict(package_sources)
        for path in (root / "tools/direction_frame.py", config_path,
                     root / "configs/first-skill/jolt-screen-v1.json",
                     root / "configs/morphology/entry-gate-v1.json",
                     root / "docs/evidence/t106-prerequisite-result.json"):
            published_inputs[path.resolve().relative_to(root.resolve()).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        verify_published_files(root, revision, published_inputs)
        immutable_json(output / "published-inputs.json", {"revision": revision, "files": published_inputs,
                                                        "sha256": sha256_json(published_inputs)})
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        manifest = freeze_manifest(output, config, config["accepted_dependency_commits"], root=root)
        budget.check()
        metadata = {"config_sha256": FOLLOWUP_SHA256, "manifest_sha256": manifest["sha256"],
                    "learner_sources_sha256": manifest["learner_sources_sha256"],
                    "baseline_source_sha256": config["baseline_source_sha256"], "training_privilege": "none",
                    "profile_sha256": config["profile_sha256"]}
        splits = ScenarioSplit(tuple(row["id"] for row in config["training_families"]), (),
                               tuple(row["id"] for row in config["evaluation_families"]))
        host_starts = 1
        phase = "host_startup"
        with GodotHost(checkout, godot, lane="act004", mode="stepped", timeout=30) as host:
            if (host.identity["host_revision"] != config["host_sha"]
                    or host.identity["bridge_sha256"] != config["bridge_sha256"]):
                raise ContractError("goal_frame_host", "the accepted host and frozen bridge are required")
            immutable_json(output / "environment.json", {"host_identity": host.identity,
                                                           "manifest_sha256": manifest["sha256"]})

            def collect(scenario, *, episode_id, split, policy_id, kind, policy, seed):
                budget.check()
                attempt = {"episode_id": episode_id, "split": split, "status": "started"}
                attempts.append(attempt)
                episode, original, _ = capture_task(host, scenario, episode_id=episode_id, split=split,
                    policy_id=policy_id, kind=kind, policy=policy, seed=seed, budget=budget,
                    output=output, config=config, controller_factory=GoalFrameController,
                    error_serializer=_portable_error)
                result = dict(original)
                results.append(result)
                if (episode.learner_revision != manifest["learner_revision"]
                        or episode.learner_sources_sha256 != manifest["learner_sources_sha256"]
                        or dict(episode.source_files) != manifest["source_files"]):
                    raise ContractError("goal_frame_source", "captured episode differs from the frozen learner sources")
                if episode.profile_sha256 != config["profile_sha256"]:
                    raise ContractError("goal_frame_profile", "the accepted Jolt profile changed")
                audit = audit_episode_boundaries(episode, config["profile_sha256"])
                audit["sha256"] = sha256_json(audit)
                immutable_json(output / "boundary-audit" / f"{episode_id}.json", audit)
                result.update(source_boundary_regressions=audit["source_boundary_regressions"],
                              boundary_audit_sha256=audit["sha256"], completed_ticks=episode.consumed_ticks)
                attempt["status"] = "completed" if audit["passed"] else "failed"
                if not audit["passed"]:
                    raise ContractError("goal_frame_boundary", "preserved episode failed its source/owner audit")
                return episode

            phase, demos = "authored_training_collection", []
            for scenario in config["training_families"]:
                for repeat, seed in enumerate(scenario["demo_seeds"]):
                    demos.append(collect(scenario, episode_id=f"demo-{scenario['id']}-{repeat}",
                        split="training", policy_id="authored-balanced", kind="baseline", policy=None, seed=seed))
            _verify_episode_files(output, results)
            samples = clone_sequences(Dataset("jolt-direction-frame-authored-v1", tuple(demos), splits))
            dataset = {"id": "jolt-direction-frame-authored-v1", "splits": splits.to_dict(),
                       "episodes": [{"id": episode.id, "record_sha256": episode.sha256,
                                     "profile_sha256": episode.profile_sha256} for episode in demos],
                       "imitation_samples": sum(len(rows) for rows, _ in samples),
                       "successful_teacher_episodes": sum(episode.evaluation["metrics"]["success"] for episode in demos),
                       "data_selection": "all eligible training samples including unsuccessful trajectories; identical sample count for both families"}
            dataset["sha256"] = sha256_json(dataset)
            immutable_json(output / "dataset.json", dataset)
            metadata["dataset_sha256"] = dataset["sha256"]
            phase = "fixed_bc_fitting"
            policies = []
            for seed in config["training_seeds"]:
                initial_weights = None
                for family in config["candidates_in_priority_order"]:
                    budget.check()
                    torch.manual_seed(seed)
                    policy = GoalFramePolicy(family)
                    if initial_weights is None:
                        initial_weights = {name: value.detach().clone() for name, value in policy.state_dict().items()}
                    elif any(not torch.equal(value, initial_weights[name]) for name, value in policy.state_dict().items()):
                        raise ContractError("goal_frame_initialization", "paired candidates must start with identical numeric weights")
                    initial_meta = {**metadata, "training_seed": seed, "phase": "initial", "kind": "bc",
                                    "family": family, "feature_frame": policy.frame}
                    prefix = f"{family}-seed{seed}"
                    initial_path = output / "checkpoints" / f"{prefix}-initial.pt"
                    initial_receipt = write_checkpoint(initial_path, policy, initial_meta)
                    checkpoints.append(initial_receipt)
                    started = time.monotonic()
                    fitting = {"family": family, "seed": seed, "status": "started", "completed_optimizer_updates": 0}
                    fitting_attempts.append(fitting)
                    trained = train_clone(policy, transform_sequences(samples, family), seed=seed,
                        epochs=config["bc"]["epochs"], unroll=config["bc"]["unroll"],
                        learning_rate=config["bc"]["learning_rate"], check_budget=budget.check,
                        on_update=lambda count: fitting.update(completed_optimizer_updates=count))
                    optimizer = torch.optim.Adam(policy.parameters(), lr=config["bc"]["learning_rate"])
                    optimizer.load_state_dict(trained.pop("optimizer"))
                    step_counts = {int(state["step"]) for state in optimizer.state.values()}
                    if len(step_counts) != 1:
                        raise ContractError("goal_frame_optimizer", "each fitted parameter must have the same update count")
                    if next(iter(step_counts)) != fitting["completed_optimizer_updates"]:
                        raise ContractError("goal_frame_optimizer", "optimizer state and completed update receipts differ")
                    trained.update(family=family, seed=seed, wall_seconds=time.monotonic() - started,
                                   optimizer_steps=step_counts.pop())
                    learning.append(trained)
                    immutable_json(output / "learning" / f"{prefix}.json", trained)
                    final_meta = {**initial_meta, "phase": "final", "epochs": config["bc"]["epochs"]}
                    final_path = output / "checkpoints" / f"{prefix}-final.pt"
                    final_receipt = write_checkpoint(final_path, policy, final_meta, optimizer=optimizer,
                        training_state={"completed_epochs": config["bc"]["epochs"], "next_episode_boundary": True,
                                        "torch_global_rng_state": torch.get_rng_state(), "further_training_authorized": False})
                    checkpoints.append(final_receipt)
                    fitting["status"] = "completed"
                    for label, receipt in (("initial", initial_receipt), ("final", final_receipt)):
                        path = verify_checkpoint_file(output, receipt)
                        loaded, loaded_metadata = load_goal_frame_policy(path, expected_config_sha256=FOLLOWUP_SHA256)
                        if loaded_metadata != receipt["metadata"] or policy_state_sha256(loaded) != receipt["policy_sha256"]:
                            raise ContractError("goal_frame_checkpoint", "loaded seed, phase, source or dataset differs from its receipt")
                        policies.append((family, seed, label, loaded, receipt))
                    print(f"{family} seed={seed}: fixed40 epochs completed", flush=True)
            phase = "freeze_before_holdout"
            for receipt in checkpoints:
                verify_checkpoint_file(output, receipt)
            frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "checkpoints": checkpoints,
                      "selection": "all six fixed final epoch40 checkpoints; no case-dependent selection"}
            frozen["sha256"] = sha256_json(frozen)
            immutable_json(output / "frozen-candidates.json", frozen)
            budget.check()
            immutable_json(output / "evaluation-opened.json", {"opened_utc": datetime.now(timezone.utc).isoformat(),
                "manifest_sha256": manifest["sha256"], "frozen_candidates_sha256": frozen["sha256"],
                "all_six_final_checkpoint_hashes_frozen": True})
            phase = "new_sealed_evaluation"
            for scenario in config["evaluation_families"]:
                for kind in ("baseline", "idle"):
                    collect(scenario, episode_id=f"evaluation-{scenario['id']}-{kind}", split="evaluation",
                            policy_id=kind, kind=kind, policy=None, seed=scenario["host_seed"])
                for family, seed, label, policy, receipt in policies:
                    policy_id = f"{family}-seed{seed}-{label}"
                    verify_checkpoint_file(output, receipt)
                    expected_hash = receipt["policy_sha256"]
                    if policy_state_sha256(policy) != expected_hash:
                        raise ContractError("goal_frame_checkpoint", "a frozen evaluation policy changed")
                    collect(scenario, episode_id=f"evaluation-{scenario['id']}-{policy_id}", split="evaluation",
                            policy_id=policy_id, kind=family, policy=policy, seed=scenario["host_seed"])
                    if policy_state_sha256(policy) != expected_hash:
                        raise ContractError("goal_frame_checkpoint", "evaluation may not fit or mutate a policy")
                print(f"{scenario['id']}: complete paired evaluation; requests={budget.decisions}", flush=True)
            status, phase = "completed", "complete"
        budget.check()
        _verify_episode_files(output, results)
        for receipt in checkpoints:
            verify_checkpoint_file(output, receipt)
    except BaseException as exc:
        if getattr(exc, "result", None) is not None:
            results.append(dict(exc.result))
        if attempts and attempts[-1]["status"] == "started":
            attempts[-1]["status"] = "failed"
        if fitting_attempts and fitting_attempts[-1]["status"] == "started":
            fitting_attempts[-1]["status"] = "interrupted"
        status = "capped" if isinstance(exc, ScreenBudgetExceeded) else "failed"
        error = _portable_error(exc)
    if manifest is not None:
        try:
            revision, sources, digest = source_identity(root)
            if revision != manifest["learner_revision"] or sources != manifest["source_files"] or digest != manifest["learner_sources_sha256"]:
                raise ContractError("goal_frame_source_drift", "published source changed during the fixed experiment")
            read_followup_config(config_path)
            verify_original_boundary(config, root)
            verify_published_files(root, manifest["learner_revision"], published_inputs)
        except BaseException as exc:
            status, error = "failed", _portable_error(exc)
    for row in results:
        if "completed_ticks" not in row:
            try:
                row["completed_ticks"] = read_episode(output / row["artifact"]).consumed_ticks
            except Exception:
                row["completed_ticks"] = None
    elapsed = time.monotonic() - budget.started
    if elapsed > budget.maximum_seconds:
        status = "capped"
        error = {"type": "ScreenBudgetExceeded", "message": "the aggregate declared wall ceiling elapsed"}
    evaluation = summarize_prerequisite(results, config, run_completed=status == "completed")
    files = [{"file": path.relative_to(output).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "bytes": path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file()]
    summary = {"schema": "nca-jolt-direction-frame-followup-result/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "status": status, "phase": phase, "error": error, "config_sha256": FOLLOWUP_SHA256,
               "manifest_sha256": None if manifest is None else manifest["sha256"],
               "learner_revision": None if manifest is None else manifest["learner_revision"],
               "learner_sources_sha256": None if manifest is None else manifest["learner_sources_sha256"],
               "runtime": {"python": platform.python_version(), "platform": platform.system(),
                           "architecture": platform.machine(), "torch": str(torch.__version__),
                           "torch_threads": torch.get_num_threads(), "device": "cpu",
                           "numpy": None if manifest is None else manifest["runtime"]["numpy"]},
               "host_start_attempts": host_starts, "attempted_game_requests": budget.decisions,
               "recorded_game_requests": sum(row["decisions"] for row in results),
               "unconfirmed_game_requests": budget.decisions - sum(row["decisions"] for row in results),
               "confirmed_physical_ticks": sum(row["completed_ticks"] or 0 for row in results),
               "unknown_tick_counts": sum(row["completed_ticks"] is None for row in results),
               "wall_seconds": elapsed, "budget": config["budget"], "capture_attempts": attempts,
               "checkpoints": checkpoints, "learning": learning, "fitting_attempts": fitting_attempts,
               "completed_optimizer_updates": sum(row["completed_optimizer_updates"] for row in fitting_attempts),
               "results": results, "artifacts": files,
               "evaluation": evaluation, "morphology_transfer_evaluations": 0, "owner_feel_accepted": False,
               "original_results_rewritten": False, "automatic_budget_extension": False}
    summary["sha256"] = sha256_json(summary)
    immutable_json(output / "summary.json", summary)
    return summary
