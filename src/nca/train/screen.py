"""Bounded actual-host first-skill screen; no surrogate physics or hidden goals."""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import itertools
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

import torch

from nca.adapters.godot import CybersandEnv, GodotHost, physical_config
from nca.contracts import WireAction, sha256_json
from nca.controllers.baseline import AuthoredController, FeedbackObservation, SkillGoal
from nca.controllers.learned import LearnedController, RecurrentPolicy, load_policy, policy_state_sha256
from nca.data import ActionSource, Dataset, EpisodeRecorder, PolicyContext, PolicySample, write_episode
from nca.randomness import ScenarioSplit
from nca.runs import RunBudget
from .artifacts import export_weights, freeze_manifest, immutable_json, write_checkpoint
from .imitation import clone_sequences
from .recurrent import RolloutSequence, Transition, attributed_actor_sample, ppo_update, train_clone
from .task import HopTask, REWARD_SCHEMA


class ScreenBudgetExceeded(RuntimeError):
    pass


class CapturedTaskError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(f"retained failed episode {result['episode_id']}: {result['error']}")


class ScreenBudget:
    def __init__(self, decisions=13000, seconds=1800):
        self.maximum_decisions, self.maximum_seconds = decisions, seconds
        self.decisions = 0
        self.started = time.monotonic()

    def check(self):
        if self.decisions >= self.maximum_decisions or time.monotonic() - self.started >= self.maximum_seconds:
            raise ScreenBudgetExceeded("predeclared screen decision or wall-time budget exhausted")

    def consume(self):
        self.check()
        self.decisions += 1


def _source(kind: str, policy_id: str, policy_sha: str) -> ActionSource:
    return ActionSource(policy_id, "neutral" if kind == "idle" else "scripted" if kind == "baseline" else "learned",
                        policy_id, policy_sha)


def recorded_policy_sample(decision, policy_sha: str) -> PolicySample | None:
    """Preserve sampling identity without giving deterministic requests a score."""
    if decision is None or decision.replaced:
        return None
    if decision.latent is None:
        return PolicySample("wire_request", decision.action.to_dict(), policy_sha)
    stochastic = decision.sampling_mode == "stochastic_draw"
    if decision.sampling_mode not in ("stochastic_draw", "deterministic_mean") or (
            stochastic != (decision.log_probability is not None)):
        raise ValueError("residual sampling mode and behaviour likelihood disagree")
    return PolicySample(
        "latent_residual", {"latent": list(decision.latent), "bounded_residual": list(decision.residual),
                            "baseline_request": decision.baseline.to_dict(),
                            "combined_request": decision.action.to_dict(),
                            "sampling_mode": decision.sampling_mode,
                            "log_probability_space": "bounded_residual" if stochastic else None},
        policy_sha, decision.log_probability,
        "scaled-tanh-diagonal-normal/v1" if stochastic else "deterministic-scaled-tanh-mean/v1")


def capture_task(host: GodotHost, scenario: Mapping[str, Any], *, episode_id: str, split: str,
                 policy_id: str, kind: str, policy: RecurrentPolicy | None, seed: int,
                 budget: ScreenBudget, output: Path, config: Mapping[str, Any],
                 maximum_decisions: int = 40, generator: torch.Generator | None = None,
                 training: bool = False):
    """One reset, one explicit goal, every actual base receipt and its outcome."""
    budget.check()
    host_config = physical_config("jev", ticks=config["goal"]["deadline_ticks"], seed=seed)
    host_config["actors"][0]["position"] = list(scenario["spawn"])
    env = CybersandEnv(host, host_config, decision_ticks=6, lease_ticks=12)
    goal = SkillGoal(episode_id, "directed_hop", tuple(scenario["target"]),
                     config["goal"]["pace"], config["goal"]["tolerance_cells"],
                     config["goal"]["deadline_ticks"] / 60)
    context = PolicyContext(goal.id, goal.skill, goal.target, goal.pace, goal.tolerance, goal.timeout_seconds)
    policy_sha = (sha256_json({"kind": "idle", "wire": WireAction(effort=0.0).to_dict()}) if kind == "idle"
                  else config["baseline_source_sha256"] if kind == "baseline" else policy_state_sha256(policy))
    recorder = EpisodeRecorder(env, episode_id, _source(kind, policy_id, policy_sha), context=context,
                               seed=seed, scenario_family=scenario["id"], split=split,
                               budget=RunBudget(max_decisions=40, max_seconds=120))
    task = HopTask(goal, recorder.initial, deadline_ticks=config["goal"]["deadline_ticks"],
                   quiet_ticks_required=config["goal"]["quiet_ticks"])
    controller = AuthoredController("jev") if kind == "baseline" else (
        LearnedController(policy, generator=generator) if policy is not None else None)
    transitions, reward_terms = [], []
    capped, error, bootstrap = False, None, 0.0
    try:
        for index in range(maximum_decisions):
            budget.check()
            begin = time.perf_counter_ns()
            decision = None
            if kind == "baseline":
                action = controller.step(FeedbackObservation.from_host(recorder.view), goal).action
            elif kind == "idle":
                action = WireAction(effort=0.0)
            else:
                decision = controller.step(recorder.view, goal, deterministic=not training)
                action = decision.action
            inference_ns = time.perf_counter_ns() - begin
            sample = recorded_policy_sample(decision, policy_sha)
            budget.consume()
            after, _, terminated, truncated, info = recorder.step(action.to_dict(), sample=sample,
                                                                  replaced=decision.replaced if decision else False)
            terms = task.consume(action, info["trace"], inference_ns=inference_ns)
            reward_terms.append(terms)
            if task.integrity_failure:
                raise ValueError("policy rollout contains a refused request or host integrity failure")
            if training:
                event = recorder.events[-1]
                eligible = event.on_policy_eligible
                latent, score = attributed_actor_sample(event) if eligible else (None, None)
                transitions.append(Transition(
                    decision.features, latent, score, decision.value, terms["total"], info["actual_ticks"], eligible,
                    task_terminal=task.succeeded or task.deadline_missed,
                    final_observation_valid=FeedbackObservation.from_host(after).valid,
                    episode_id=episode_id, event_index=index))
            if task.done:
                break
            if terminated or truncated:
                # This wrapper's intrinsic deadline exactly matches 240 ticks.
                # Any other host stop is not ordinary task completion.
                raise ValueError("host ended before the predeclared task outcome")
        else:
            capped = not task.done
        if capped and training:
            # PPO updates start a fresh episode. Evaluate this valid final
            # observation once on a copy of the recurrent/reference state.
            peek = copy.deepcopy(controller)
            bootstrap = peek.step(recorder.view, goal, deterministic=True).value
            transitions[-1] = replace(transitions[-1], administrative_cutoff=True)
        if not task.deadline_missed:
            recorder.cancel("task_arrival" if task.succeeded else "administrative_batch_cutoff")
            if controller is not None:
                controller.cancel()
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        capped = isinstance(exc, ScreenBudgetExceeded)
        if not capped:
            task.integrity_failure = True
        try:
            recorder.cancel("screen_error_cleanup")
        except Exception:
            pass
        if controller is not None:
            controller.cancel()
        # An unknown next state may not borrow a stale/reset bootstrap. Keep
        # the entire failed sequence out of learner updates and retain it in
        # evaluation coverage and failure totals.
        transitions, bootstrap = [], 0.0
    metrics = task.metrics(administratively_capped=capped)
    evaluation = {"task_reward_schema": REWARD_SCHEMA, "metrics": metrics,
                  "reward_terms": reward_terms, "policy_id": policy_id,
                  "sample_policy_sha256": policy_sha, "error": error}
    status = "failed" if task.integrity_failure else "capped" if capped else "completed"
    reason = metrics["reason"]
    episode = recorder.finish(status, reason, evaluation=evaluation)
    path = output / "episodes" / f"{episode_id}.json.gz"
    artifact_sha = write_episode(path, episode)
    result = {"episode_id": episode_id, "scenario_family": scenario["id"], "split": split,
              "host_seed": seed, "kind": kind, "policy_id": policy_id, "policy_sha256": policy_sha,
              "artifact": path.relative_to(output).as_posix(), "artifact_sha256": artifact_sha,
              "record_sha256": episode.sha256, "profile_sha256": episode.profile_sha256,
              "decisions": episode.decisions, "metrics": metrics, "error": error}
    immutable_json(output / "metrics" / f"{episode_id}.json", result)
    if error is not None:
        if capped:
            exception = ScreenBudgetExceeded(error["message"])
            exception.result = result
            raise exception
        raise CapturedTaskError(result)
    return episode, result, RolloutSequence(tuple(transitions), bootstrap) if transitions else None


def _quantile(values, fraction):
    ordered = sorted(values)
    cursor = (len(ordered) - 1) * fraction
    left, right = math.floor(cursor), math.ceil(cursor)
    return ordered[left] + (ordered[right] - ordered[left]) * (cursor - left)


def policy_seed_interval(values):
    """Exactly 27 seed-level resamples for the declared three-policy screen."""
    if len(values) != 3:
        raise ValueError("uncertainty unit is exactly three independently trained policies")
    means = [sum(draw) / 3 for draw in itertools.product(values, repeat=3)]
    return {"mean": sum(values) / 3, "lower95": _quantile(means, .025),
            "upper95": _quantile(means, .975), "individual_seeds": list(values),
            "method": "three-policy-seed percentile bootstrap; all27 draws; exploratory"}


def summarize_screen(results, config):
    rows = [row for row in results if row["split"] == "evaluation"]
    scenarios = [item["id"] for item in config["evaluation_families"]]
    reference = {row["scenario_family"]: row for row in rows if row["kind"] == "baseline"}
    idle = [row for row in rows if row["kind"] == "idle"]
    summary = {"complete": False, "conclusion": "inconclusive_incomplete_screen", "candidates": {},
               "evaluation_rows": len(rows), "owner_feel_accepted": False,
               "uncertainty_limit": "three policy seeds and eight fixed scenarios are an exploratory screen"}
    if len(reference) != len(scenarios) or len(idle) != len(scenarios):
        return summary
    metric_keys = ("success", "capped_completion_seconds", "ordinary_trim_reversals_per_second",
                   "mean_absolute_angle_radians", "peak_toward_speed_cells_per_second", "purposeful_launch")

    def aggregate(items):
        return {key: sum(float(row["metrics"][key]) for row in items) / len(items) for key in metric_keys}

    baseline = aggregate(list(reference.values()))
    summary["baseline"] = baseline
    summary["idle"] = aggregate(idle)
    summary["negative_control_audit"] = {"idle_successes": sum(row["metrics"]["success"] for row in idle),
                                         "idle_positive_returns": sum(row["metrics"]["return"] > 0 for row in idle),
                                         "idle_return_range": [min(row["metrics"]["return"] for row in idle),
                                                               max(row["metrics"]["return"] for row in idle)]}
    complete = True
    for kind in ("bc", "residual"):
        per_seed, success_gains, time_gains, gates, initial_comparisons = [], [], [], [], []
        for seed in config["training_seeds"]:
            policy_id = f"{kind}-seed{seed}-final"
            selected = [row for row in rows if row["policy_id"] == policy_id]
            initial = [row for row in rows if row["policy_id"] == f"{kind}-seed{seed}-initial"]
            if sorted(row["scenario_family"] for row in selected) != sorted(scenarios) or len(initial) != len(scenarios):
                complete = False
                continue
            means = aggregate(selected)
            initial_means = aggregate(initial)
            per_seed.append({"seed": seed, **means})
            initial_comparisons.append({"seed": seed, "initial": initial_means,
                                        "success_gain": means["success"] - initial_means["success"],
                                        "completion_gain_seconds": initial_means["capped_completion_seconds"] - means["capped_completion_seconds"]})
            success_gains.append(means["success"] - baseline["success"])
            time_gains.append(baseline["capped_completion_seconds"] - means["capped_completion_seconds"])
            gate = config["screen_gate"]
            speed_ratio = means["peak_toward_speed_cells_per_second"] / max(1e-9, baseline["peak_toward_speed_cells_per_second"])
            gate_result = {
                "integrity": all(not row["metrics"]["integrity_failure"] and row["metrics"]["failed_admissions"] == 0 for row in selected),
                "uncensored": all(not row["metrics"]["administratively_capped"] for row in selected),
                "ordinary_trim": means["ordinary_trim_reversals_per_second"] <= baseline["ordinary_trim_reversals_per_second"] + gate["maximum_ordinary_trim_reversals_per_second_increase"],
                "orientation": means["mean_absolute_angle_radians"] <= baseline["mean_absolute_angle_radians"] + gate["maximum_mean_absolute_angle_increase_radians"],
                "vigor": gate["minimum_peak_toward_speed_ratio"] <= speed_ratio <= gate["maximum_peak_toward_speed_ratio"],
                "launch_character": means["purposeful_launch"] >= baseline["purposeful_launch"] - gate["maximum_purposeful_launch_fraction_loss"],
                "latency": max(row["metrics"]["inference_p95_ms"] for row in selected) <= gate["maximum_inference_p95_ms"],
            }
            gates.append({"seed": seed, **gate_result, "passed": all(gate_result.values())})
        if len(per_seed) != 3:
            summary["candidates"][kind] = {"conclusion": "inconclusive_missing_seeds", "per_seed": per_seed}
            continue
        success, completion = policy_seed_interval(success_gains), policy_seed_interval(time_gains)
        gate = config["screen_gate"]
        success_threshold = success["mean"] >= gate["minimum_success_rate_gain"]
        time_threshold = (baseline["success"] >= gate["alternative_reference_success_minimum"]
                          and completion["mean"] >= gate["alternative_capped_completion_gain_seconds"]
                          and completion["mean"] / baseline["capped_completion_seconds"] >= gate["alternative_capped_completion_gain_fraction"]
                          and min(success_gains) >= 0)
        useful_mean = success_threshold or time_threshold
        positive_lower = ((success_threshold and success["lower95"] > 0)
                          or (time_threshold and completion["lower95"] > 0))
        character_ok = all(item["passed"] for item in gates)
        conclusion = ("numerically_useful_owner_feel_pending" if useful_mean and positive_lower and character_ok
                      else "inconclusive_at_budget" if useful_mean and character_ok
                      else "no_go_at_budget")
        summary["candidates"][kind] = {"conclusion": conclusion, "per_seed": per_seed,
                                      "paired_success_gain": success, "paired_completion_gain_seconds": completion,
                                      "compared_with_initial_weights": initial_comparisons,
                                      "character_and_runtime_gates": gates, "useful_mean": useful_mean,
                                      "positive_bootstrap_lower95": positive_lower}
    summary["complete"] = complete
    if complete:
        summary["conclusion"] = "screen_complete_see_candidate_decisions"
    return summary


def run_screen(*, checkout: str | Path, godot: str | Path, output: str | Path,
               config: Mapping[str, Any], dependency_commits: Mapping[str, str]):
    destination = Path(output)
    if destination.exists():
        raise ValueError("choose a new immutable run directory; previous evidence is never overwritten")
    destination.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = freeze_manifest(destination, config, dependency_commits)
    budget = ScreenBudget(config["budget"]["maximum_total_decisions"], config["budget"]["maximum_wall_seconds"])
    results, checkpoints, demos, updates, episode_counter = [], [], [], [], 0
    splits = ScenarioSplit(tuple(item["id"] for item in config["training_families"]),
                           tuple(item["id"] for item in config["tuning_families"]),
                           tuple(item["id"] for item in config["evaluation_families"]))
    metadata = {"config_sha256": manifest["config_sha256"],
                "learner_sources_sha256": manifest["learner_sources_sha256"],
                "baseline_source_sha256": config["baseline_source_sha256"],
                "manifest_sha256": manifest["sha256"], "training_privilege": "none"}
    status, error, all_policies = "incomplete", None, []
    try:
        with GodotHost(checkout, godot, lane="act004", mode="stepped") as host:
            if host.identity.get("host_revision") != config["host_sha"] or host.identity.get("bridge_sha256") != config["bridge_sha256"]:
                raise ValueError("actual host/bridge revision differs from the predeclared screen")
            immutable_json(destination / "environment.json", {"host_identity": host.identity,
                                                               "manifest_sha256": manifest["sha256"]})
            for scenario in config["training_families"]:
                for repeat, seed in enumerate(scenario["demo_seeds"]):
                    episode, result, _ = capture_task(host, scenario, episode_id=f"demo-{scenario['id']}-{repeat}",
                                                     split="training", policy_id="authored-balanced", kind="baseline",
                                                     policy=None, seed=seed, budget=budget, output=destination, config=config)
                    demos.append(episode)
                    results.append(result)
            dataset = Dataset("jolt-hop-authored-v1", tuple(demos), splits)
            samples = clone_sequences(dataset)
            dataset_receipt = {"id": dataset.id, "splits": splits.to_dict(),
                               "episodes": [{"id": episode.id, "record_sha256": episode.sha256,
                                             "profile_sha256": episode.profile_sha256} for episode in demos],
                               "imitation_samples": sum(len(rows) for rows, _ in samples),
                               "successful_teacher_episodes": sum(episode.evaluation["metrics"]["success"] for episode in demos),
                               "data_selection": "all eligible training samples; no success filtering"}
            dataset_receipt["sha256"] = sha256_json(dataset_receipt)
            immutable_json(destination / "dataset.json", dataset_receipt)
            metadata["dataset_sha256"] = dataset_receipt["sha256"]
            metadata["profile_sha256"] = demos[0].profile_sha256
            if any(episode.profile_sha256 != metadata["profile_sha256"] for episode in demos):
                raise ValueError("demonstrations changed the accepted actor capability profile")
            for seed in config["training_seeds"]:
                seed_started = time.monotonic()
                torch.manual_seed(seed)
                clone = RecurrentPolicy("bc")
                initial_clone = copy.deepcopy(clone)
                initial_meta = {**metadata, "training_seed": seed, "phase": "initial", "kind": "bc"}
                checkpoints.append(write_checkpoint(destination / "checkpoints" / f"bc-seed{seed}-initial.pt", clone, initial_meta))
                bc_start = time.monotonic()
                bc_result = train_clone(clone, samples, seed=seed, epochs=config["bc"]["epochs"],
                                        unroll=config["bc"]["unroll"], learning_rate=config["bc"]["learning_rate"])
                bc_result.pop("optimizer")
                bc_result.update(seed=seed, wall_seconds=time.monotonic() - bc_start)
                immutable_json(destination / "learning" / f"bc-seed{seed}.json", bc_result)
                final_meta = {**initial_meta, "phase": "final", "epochs": config["bc"]["epochs"]}
                checkpoints.append(write_checkpoint(destination / "checkpoints" / f"bc-seed{seed}-final.pt", clone, final_meta))
                export_weights(destination / "exports" / f"bc-seed{seed}-final.weights.json", clone, final_meta)
                all_policies.extend((("bc", seed, "initial", initial_clone), ("bc", seed, "final", clone)))
                torch.manual_seed(seed + 100000)
                residual = RecurrentPolicy("residual")
                initial_residual = copy.deepcopy(residual)
                rng = random.Random(seed + 300000)
                generator = torch.Generator().manual_seed(seed + 200000)
                optimizer = torch.optim.Adam(residual.parameters(), lr=config["ppo"]["learning_rate"])
                initial_meta = {**metadata, "training_seed": seed, "phase": "initial", "kind": "residual"}
                checkpoints.append(write_checkpoint(destination / "checkpoints" / f"residual-seed{seed}-initial.pt", residual, initial_meta))
                consumed = 0
                local_episode_counter = 0
                for update in range(config["ppo"]["updates"]):
                    budget.check()
                    if time.monotonic() - seed_started >= config["budget"]["maximum_seed_wall_seconds"]:
                        raise ScreenBudgetExceeded("predeclared per-seed wall budget exhausted")
                    sequences, batch_count = [], 0
                    while batch_count < config["ppo"]["rollout_decisions"]:
                        scenario = rng.choice(config["training_families"])
                        host_seed = 100000 + seed * 100 + local_episode_counter
                        episode_id = f"ppo-seed{seed}-u{update:02d}-e{local_episode_counter:04d}"
                        _, result, sequence = capture_task(
                            host, scenario, episode_id=episode_id, split="training", policy_id=f"residual-seed{seed}-u{update:02d}",
                            kind="residual", policy=residual, seed=host_seed, budget=budget, output=destination, config=config,
                            maximum_decisions=min(40, config["ppo"]["rollout_decisions"] - batch_count), generator=generator, training=True)
                        results.append(result)
                        sequences.append(sequence)
                        batch_count += result["decisions"]
                        consumed += result["decisions"]
                        local_episode_counter += 1
                    update_start = time.monotonic()
                    update_result = ppo_update(residual, optimizer, sequences, seed=seed + update,
                                               epochs=config["ppo"]["epochs_per_update"], unroll=config["ppo"]["unroll"],
                                               clip_ratio=config["ppo"]["clip_ratio"], value_weight=config["ppo"]["value_weight"],
                                               entropy_weight=config["ppo"]["latent_entropy_weight"])
                    update_result.update(seed=seed, update=update, consumed_decisions=consumed,
                                         learner_seconds=time.monotonic() - update_start)
                    updates.append(update_result)
                    immutable_json(destination / "learning" / f"residual-seed{seed}-u{update:02d}.json", update_result)
                    checkpoint_meta = {**initial_meta, "phase": "training_boundary", "completed_updates": update + 1}
                    checkpoints.append(write_checkpoint(
                        destination / "checkpoints" / f"residual-seed{seed}-u{update:02d}.pt", residual, checkpoint_meta,
                        optimizer=optimizer, training_state={"next_episode_boundary": True, "completed_updates": update + 1,
                                                            "consumed_decisions": consumed, "episode_counter": local_episode_counter,
                                                            "scenario_rng_state": rng.getstate(), "torch_generator_state": generator.get_state(),
                                                            "torch_global_rng_state": torch.get_rng_state()}))
                    print(f"seed={seed} PPO update={update+1}/8 decisions={consumed} actor_samples={update_result['actor_samples']}", flush=True)
                final_meta = {**initial_meta, "phase": "final", "decisions": consumed, "completed_updates": config["ppo"]["updates"]}
                checkpoints.append(write_checkpoint(destination / "checkpoints" / f"residual-seed{seed}-final.pt", residual, final_meta))
                export_weights(destination / "exports" / f"residual-seed{seed}-final.weights.json", residual, final_meta)
                all_policies.extend((("residual", seed, "initial", initial_residual), ("residual", seed, "final", residual)))
            immutable_json(destination / "frozen-candidates.json", {"frozen_utc": datetime.now(timezone.utc).isoformat(),
                                                                      "checkpoints": checkpoints, "selection": "all fixed final checkpoints; no tuning selection"})
            for split in ("tuning", "evaluation"):
                if split == "evaluation":
                    immutable_json(destination / "evaluation-opened.json", {"opened_utc": datetime.now(timezone.utc).isoformat(),
                                                                            "manifest_sha256": manifest["sha256"],
                                                                            "all_final_checkpoint_hashes_frozen": True})
                candidates = [("baseline", None, "reference", None), ("idle", None, "reference", None)]
                candidates.extend(item for item in all_policies if split == "evaluation" or item[2] == "final")
                for scenario in config[f"{split}_families"]:
                    for kind, seed, phase, policy in candidates:
                        policy_id = kind if seed is None else f"{kind}-seed{seed}-{phase}"
                        episode_id = f"{split}-{scenario['id']}-{policy_id}"
                        _, result, _ = capture_task(host, scenario, episode_id=episode_id, split=split,
                                                    policy_id=policy_id, kind=kind, policy=policy, seed=scenario["host_seed"],
                                                    budget=budget, output=destination, config=config)
                        results.append(result)
                print(f"{split} completed; aggregate game decisions={budget.decisions}", flush=True)
            status = "completed"
    except Exception as exc:
        if getattr(exc, "result", None) is not None:
            results.append(exc.result)
        status = "capped" if isinstance(exc, ScreenBudgetExceeded) else "failed"
        error = {"type": type(exc).__name__, "message": str(exc)}
    summary = {"status": status, "error": error, "manifest_sha256": manifest["sha256"],
               "config_sha256": manifest["config_sha256"], "learner_sources_sha256": manifest["learner_sources_sha256"],
               "actual_decisions": budget.decisions, "wall_seconds": time.monotonic() - budget.started,
               "checkpoints": checkpoints, "ppo_updates": updates, "results": results,
               "evaluation": summarize_screen(results, config)}
    summary["sha256"] = sha256_json(summary)
    immutable_json(destination / "summary.json", summary)
    return summary
