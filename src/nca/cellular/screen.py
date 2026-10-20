"""Execute the single frozen CPU screen against the pinned actual ACT004 host."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from nca.adapters.godot import CybersandEnv, GodotHost, PINNED_HOST_REVISION, physical_config
from nca.contracts import (ContractError, Record, WireAction, canonical_json, digest,
                           identifier, sha256_json, strict_json_loads)
from nca.controllers.baseline import AuthoredController, FeedbackObservation, SkillGoal
from nca.data import ActionSource, Dataset, EpisodeRecorder, PolicyContext, write_episode
from nca.data.capture import CaptureBudgetExceeded, source_identity
from nca.randomness import ScenarioSplit, derive_seed
from nca.runs import ArtifactRef, RunBudget, RunManifest, RunStore, sha256_file, utc_now
from .core import FEATURE_SCHEMA, JOLT_PROFILE_SHA256, action_vector, checked_parts, mapped_request
from .evaluation import ArrivalTask, utility_gate
from .messages import FaultSpec
from .model import GraphController, GraphPolicy, load_policy, save_policy, weight_sha256
from .training import train_bc, training_sequences

CONFIG_SHA256 = "e5a66f5a830b9e097a5bbb9647ade4bea42c82e26fb5718273b5bae2b992de5d"
RUN_OBSERVATION_SCHEMA = "jolt-per-part-proprioception-goal-v1"


@dataclass(frozen=True)
class BudgetReceipt(Record):
    """Measured prior work, linked to a separate immutable evidence receipt."""
    id: str
    content_sha256: str
    game_decisions: int
    wall_seconds: float

    def validate(self):
        identifier(self.id, "budget_receipt.id")
        digest(self.content_sha256, "budget_receipt.content_sha256")
        if not 0 <= self.game_decisions <= 4000 or not 0 <= self.wall_seconds <= 900:
            raise ContractError("graph_budget", "prior usage must fit the commissioned ceiling")


@dataclass(frozen=True)
class PriorBudget(Record):
    """Prior attempts consume the original ceiling; they never extend it."""
    receipts: tuple[BudgetReceipt, ...] = ()
    schema: Literal["nca-cellular-prior-budget/v1"] = "nca-cellular-prior-budget/v1"

    def validate(self):
        if len(self.receipts) > 32 or len({item.id for item in self.receipts}) != len(self.receipts):
            raise ContractError("graph_budget", "prior receipts must be distinct and bounded")
        if self.game_decisions > 4000 or self.wall_seconds > 900:
            raise ContractError("graph_budget", "prior work already exceeds the commissioned ceiling")

    @property
    def game_decisions(self):
        return sum(item.game_decisions for item in self.receipts)

    @property
    def wall_seconds(self):
        return math.fsum(item.wall_seconds for item in self.receipts)


def load_declaration(path):
    config = strict_json_loads(Path(path).read_bytes())
    if sha256_json(config) != CONFIG_SHA256:
        raise ContractError("graph_preregistration", "this runner requires the exact predeclared screen")
    return config


def frozen_sources(source_root, config_path):
    """Refuse uncommitted executed sources before opening any game outcomes."""
    root = Path(source_root).resolve()
    revision, files, _ = source_identity(root)
    for relative in ("tools/cellular_screen.py", "configs/cellular/jolt-three-part-hop-v1.json"):
        files[relative] = sha256_file(root / relative)
    if sha256_file(config_path) != files["configs/cellular/jolt-three-part-hop-v1.json"]:
        raise ContractError("graph_preregistration", "configuration bytes differ from the committed declaration")
    paths = sorted(files)
    try:
        subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", "--", *paths],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dirty = subprocess.check_output(["git", "-C", str(root), "diff", "--name-only", "HEAD", "--", *paths], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("graph_source_freeze", "every executed source and declaration must be committed") from exc
    if dirty:
        raise ContractError("graph_source_freeze", "publish an isolated clean source candidate before this screen")
    return revision, files, sha256_json(files)


class ScreenBudget:
    """One ceiling covers demonstrations, training, diagnostics and evaluation."""
    def __init__(self, *, clock=time.monotonic, max_decisions=4000, max_seconds=900, prior=None):
        if max_decisions != 4000 or max_seconds != 900:
            raise ContractError("graph_budget", "the commissioned screen ceilings cannot be extended")
        self.prior = PriorBudget() if prior is None else prior
        if not isinstance(self.prior, PriorBudget):
            raise ContractError("graph_budget", "prior usage requires a validated immutable receipt")
        self.clock, self.started = clock, clock()
        self.decisions = self.prior.game_decisions
        self.limit = RunBudget(max_decisions, max_seconds)

    @property
    def elapsed(self):
        return self.prior.wall_seconds + self.attempt_elapsed

    @property
    def attempt_elapsed(self):
        return self.clock() - self.started

    @property
    def attempt_decisions(self):
        return self.decisions - self.prior.game_decisions

    def check(self):
        if self.limit.exhausted(self.decisions, self.elapsed):
            raise CaptureBudgetExceeded()

    def claim_decision(self):
        self.check()
        # Reserve before dispatch: an uncertain failed owner call still costs.
        self.decisions += 1


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(value) + "\n")


def artifact(root, path, kind):
    path = Path(path)
    return ArtifactRef(path.relative_to(root).as_posix(), sha256_file(path), path.stat().st_size, kind)


class Screen:
    """One serialized experiment; policy inputs never include task metrics."""
    def __init__(self, config, source_root, output, revision, files, source_hash, budget):
        self.config, self.root, self.output = config, Path(source_root), Path(output)
        self.revision, self.files, self.source_hash = revision, files, source_hash
        self.budget = budget
        self.split = ScenarioSplit(*(tuple(item["id"] for item in config[name])
                                     for name in ("training_families", "tuning_families", "evaluation_families")))
        self.rows, self.artifacts, self.training_reports = [], [], []
        self.checkpoints = []
        self.host_identity = {}
        self.environment = {"python": platform.python_version(), "system": platform.system(),
                            "machine": platform.machine(), "torch": str(torch.__version__),
                            "numpy": np.__version__, "torch_threads": 1,
                            "device": "cpu", "gpu_used": False,
                            "prior_budget": budget.prior.to_dict(),
                            "prior_budget_sha256": budget.prior.sha256}

    def prepare(self):
        """Publish a validated ledger before a host can consume any game work."""
        path = self.write_manifest("t110-jolt-three-part-preflight", "prepared", self.budget.elapsed,
                                   self.budget.decisions,
                                   {"phase": "before_actual_host", "attempt_game_decisions": 0,
                                    "prior_game_decisions": self.budget.prior.game_decisions},
                                   tuple(self.artifacts), ())
        self.artifacts.append(artifact(self.output, path, "prepared_run_manifest"))
        return path

    def save_json(self, relative, value, kind):
        path = self.output / relative
        write_json(path, value)
        ref = artifact(self.output, path, kind)
        self.artifacts.append(ref)
        return ref

    def run_episode(self, host, family, seed, policy_seed, method, stage, fault="normal", policy=None,
                    family_index=0):
        self.budget.check()
        started = time.monotonic()
        episode_id = f"t110-{stage}-{family['id']}-{policy_seed}-{seed}-{method}-{fault}"
        config = physical_config("jev", ticks=180, seed=seed)
        config["actors"][0]["position"] = list(family["spawn"])
        goal = SkillGoal("t110-hop", "directed_hop", tuple(family["target"]), .8, 4.0, 3.0)
        context = PolicyContext(goal.id, goal.skill, goal.target, goal.pace, goal.tolerance, goal.timeout_seconds)
        reference = AuthoredController("jev") if method == "authored" else None
        communication_seed = derive_seed(policy_seed, "cellular.communication", family_index)
        learned = (GraphController(policy, profile_sha256=JOLT_PROFILE_SHA256,
                                  fault=FaultSpec(fault), communication_seed=communication_seed)
                   if reference is None else None)
        policy_hash = (weight_sha256(policy) if reference is None else
                       sha256_json({"authored_tuning": reference.tuning.digest,
                                    "mapper": "shared-six-channel-support-gate/v1",
                                    "sources_sha256": self.source_hash}))
        source = ActionSource(method, "scripted" if reference else "learned", f"cellular-{method}/v1", policy_hash)
        env = CybersandEnv(host, config, decision_ticks=6, lease_ticks=12)
        recorder, task = None, None
        traces, local_artifacts, failure = [], [], None
        status = "completed"
        try:
            recorder = EpisodeRecorder(env, episode_id, source, context=context, seed=seed,
                        scenario_family=family["id"], split=self.split.partition(family["id"]),
                        budget=RunBudget(30, 900), source_root=self.root)
            if sha256_json(recorder.profile) != JOLT_PROFILE_SHA256:
                raise ContractError("graph_profile", "actual host profile differs from the frozen candidate")
            checked_parts(recorder.view)
            if tuple(recorder.view["position"]) != tuple(family["spawn"]):
                raise ContractError("graph_spawn", "the host did not admit the explicit scenario spawn")
            task = ArrivalTask(goal, recorder.view)
            previous = WireAction(effort=0)
            while not task.done:
                self.budget.check()
                before = time.perf_counter_ns()
                if reference is not None:
                    memory_reset = fault == "memory_reset" and recorder.view["tick"] - task.start_tick == 60
                    if memory_reset:
                        reference.reset()
                    proposed = reference.step(FeedbackObservation.from_host(recorder.view), goal)
                    request = mapped_request(action_vector(proposed.action), recorder.view, goal, previous)
                    trace = {"tick": recorder.view["tick"], "observation_id": recorder.view["observation_id"],
                             "phase": proposed.phase, "private_memory_reset": memory_reset,
                             "communication": "not-applicable-authored-reference"}
                else:
                    result = learned.step(recorder.view, goal)
                    request = result.action
                    trace = {"tick": result.frame.tick, "observation_id": result.frame.observation_id,
                             "identity": result.frame.identity.to_dict(), "nodes": result.frame.nodes,
                             "hidden_before": result.hidden_before, "communication": result.communication,
                             "reason": result.reason}
                inference_ns = time.perf_counter_ns() - before
                if (request.move[1] != 0 or request.reel != 0 or request.tool
                        or request.attach_usable or request.release_usable):
                    raise ContractError("graph_actuation", "request exceeds the six declared actor channels")
                self.budget.claim_decision()
                _, _, terminated, truncated, info = recorder.step(request.to_dict())
                task.consume(request, info["trace"], inference_ns)
                trace.update(request=request.to_dict(), inference_ns=inference_ns)
                traces.append(trace)
                previous = request
                if (terminated or truncated) and not task.done:
                    raise ContractError("graph_terminal", "the actual owner ended before the declared outcome")
            if not env._finished:
                # Quiet arrival ends the controller invocation before the
                # native deadline. Record the actual owner's revocation now;
                # finalizing a data record alone does not revoke its request.
                arrived = task.success_tick is not None and not task.integrity_failure
                recorder.cancel("cellular_task_arrival" if arrived else "cellular_task_stop")
        except BaseException as exc:
            failure = type(exc).__name__ + (":" + exc.code if isinstance(exc, ContractError) else "")
            status = "capped" if isinstance(exc, CaptureBudgetExceeded) else "canceled" if isinstance(exc, KeyboardInterrupt) else "failed"
            if task is not None and status == "failed":
                task.integrity_failure = True
            if recorder is not None and not env._finished:
                try:
                    recorder.cancel("cellular_screen_abort")
                except Exception:
                    pass
            raise
        finally:
            if learned is not None:
                learned.cancel()
            if reference is not None:
                reference.cancel("cellular_screen_closed")
            metrics = task.metrics() if task is not None else {"success": False, "integrity_failure": True,
                                                              "reason": "missing_initial_state"}
            episode = (recorder.finish(status=status, reason=failure or metrics["reason"], evaluation=metrics)
                       if recorder is not None else None)
            if episode is not None:
                path = self.output / "episodes" / (episode_id + ".json.gz")
                write_episode(path, episode)
                ref = artifact(self.output, path, "episode")
                self.artifacts.append(ref)
                local_artifacts.append(ref)
            trace_ref = self.save_json("communications/" + episode_id + ".json", traces, "communication_trace")
            local_artifacts.append(trace_ref)
            row = {"id": episode_id, "stage": stage, "seed": policy_seed, "host_seed": seed,
                   "method": method, "family": family["id"], "fault": fault,
                   "communication_seed": communication_seed, "policy_sha256": policy_hash,
                   "episode_sha256": episode.sha256 if episode else None, "status": status,
                   "decisions": episode.decisions if episode else 0,
                   "consumed_ticks": episode.consumed_ticks if episode else 0, "metrics": metrics}
            self.rows.append(row)
            self.write_manifest(episode_id, status, time.monotonic() - started, row["decisions"],
                                row, tuple(local_artifacts), () if failure is None else (failure,),
                                resolved={"declaration_sha256": CONFIG_SHA256, "episode": config,
                                          "source_files_sha256": self.source_hash, "goal": context.to_dict(),
                                          "policy_feature_schema": FEATURE_SCHEMA})
        return episode

    def write_manifest(self, run_id, status, elapsed, decisions, metrics, refs, failures, resolved=None):
        manifest = RunManifest(run_id=run_id, experiment="conditional-jolt-per-part-graph",
            learner_revision=self.revision, host_revision=PINNED_HOST_REVISION,
            profile_sha256=JOLT_PROFILE_SHA256, resolved_config=self.config if resolved is None else resolved,
            environment={**self.environment, "host": self.host_identity,
                         "learner_source_files_sha256": self.source_hash},
            seeds=tuple(self.config["training_seeds"]), scenario_split=self.split,
            status=status, gameplay_lane="physical_ACT004", execution_lane="stepped",
            observation_schema=RUN_OBSERVATION_SCHEMA, action_schema="act004-wire-v2",
            suite="jolt-three-part-hop-v1", budget=self.budget.limit, decisions=decisions,
            runtime_seconds=elapsed, metrics=metrics, artifacts=refs, failures=failures)
        return RunStore(self.output / "runs").write(manifest)

    def train_candidates(self, dataset):
        sequences = training_sequences(dataset)
        dataset_hash = dataset.sha256
        self.save_json("training-dataset.json", {"id": dataset.id, "sha256": dataset_hash,
                       "splits": dataset.splits.to_dict(),
                       "episodes": [{"id": ep.id, "sha256": ep.sha256} for ep in dataset.episodes],
                       "eligible_rows": sum(len(values[0]) for values in sequences),
                       "policy_schema": FEATURE_SCHEMA, "source": "autonomous-authored-only",
                       "corrections": 0, "privileged_teachers": 0, "ppo_likelihood_rows": 0}, "dataset_manifest")
        for seed in self.config["training_seeds"]:
            for method in ("graph", "no_message"):
                self.budget.check()
                torch.manual_seed(seed)
                policy = GraphPolicy(method)
                if policy.specification()["parameter_count"] != 2878:
                    raise ContractError("graph_parameter_budget", "matched network size changed")
                metadata = {"config_sha256": CONFIG_SHA256, "dataset_sha256": dataset_hash,
                            "seed": seed, "learner_revision": self.revision,
                            "learner_sources_sha256": self.source_hash, "training_privilege": "none"}
                initial_path = self.output / "checkpoints" / f"{method}-{seed}-initial.pt"
                save_policy(policy, initial_path, {**metadata, "training_updates": 0})
                self.artifacts.append(artifact(self.output, initial_path, "initial_checkpoint"))
                report = train_bc(policy, sequences, seed, self.budget.check)
                final_path = self.output / "checkpoints" / f"{method}-{seed}-final.pt"
                save_policy(policy, final_path, {**metadata, "training_updates": report["updates"]})
                self.artifacts.append(artifact(self.output, final_path, "trained_checkpoint"))
                self.checkpoints.append({"method": method, "seed": seed,
                         "initial": artifact(self.output, initial_path, "initial_checkpoint").to_dict(),
                         "final": artifact(self.output, final_path, "trained_checkpoint").to_dict(),
                         "policy_sha256": weight_sha256(policy)})
                self.training_reports.append({"method": method, "seed": seed, **report})
                print(json.dumps({"stage": "checkpoint_complete", "method": method, "seed": seed,
                                  "game_decisions": self.budget.decisions}), flush=True)
        if len(self.checkpoints) != 6:
            raise ContractError("graph_holdout_seal", "all six final checkpoints must precede holdout access")
        self.save_json("training-reports.json", self.training_reports, "training_metrics")
        self.save_json("frozen-candidates.json", {"created_utc": utc_now(), "config_sha256": CONFIG_SHA256,
                       "dataset_sha256": dataset_hash, "checkpoints": self.checkpoints,
                       "heldout_opened": False, "checkpoint_selection": "fixed-final-epoch"}, "candidate_freeze")

    def open_holdout(self):
        path = self.output / "frozen-candidates.json"
        frozen = strict_json_loads(path.read_bytes())
        if len(frozen["checkpoints"]) != 6 or frozen["heldout_opened"] is not False:
            raise ContractError("graph_holdout_seal", "incomplete candidate freeze")
        for item in frozen["checkpoints"]:
            for phase in ("initial", "final"):
                if sha256_file(self.output / item[phase]["uri"]) != item[phase]["content_sha256"]:
                    raise ContractError("graph_holdout_seal", "a frozen checkpoint changed")
        self.save_json("holdout-opened.json", {"created_utc": utc_now(), "candidate_freeze_sha256": sha256_file(path),
                       "config_sha256": CONFIG_SHA256, "no_subsequent_training_or_selection": True}, "holdout_opening")

    def evaluate(self, host, stage, families, faults, methods, *, initial=False):
        for index, seed in enumerate(self.config["training_seeds"]):
            for family_index, family in enumerate(families):
                for fault in faults:
                    for method in methods:
                        policy = None
                        if method != "authored":
                            path = self.output / "checkpoints" / f"{method}-{seed}-{'initial' if initial else 'final'}.pt"
                            policy, _ = load_policy(path, expected_config_sha256=CONFIG_SHA256)
                        self.run_episode(host, family, family["host_seed"] + 100 * index,
                                         seed, method, stage, fault, policy, family_index)
                print(json.dumps({"stage": stage, "seed": seed, "family": family["id"],
                                  "game_decisions": self.budget.decisions}), flush=True)


def run_screen(*, config_path, source_root, output, host_checkout, godot, prior_budget=None):
    prior = PriorBudget() if prior_budget is None else PriorBudget.from_json(Path(prior_budget).read_bytes())
    budget = ScreenBudget(prior=prior)
    config = load_declaration(config_path)
    revision, files, source_hash = frozen_sources(source_root, config_path)
    if str(torch.__version__) != "2.8.0+cpu" or np.__version__ != "2.3.5":
        raise ContractError("graph_runtime", "screen evidence requires the declared Torch CPU and NumPy versions")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    screen = Screen(config, source_root, output, revision, files, source_hash, budget)
    screen.save_json("source-manifest.json", {"revision": revision, "files": files, "sha256": source_hash}, "source_manifest")
    screen.save_json("declaration.json", config, "declaration")
    screen.save_json("prior-budget.json", prior, "prior_budget_receipt")
    screen.prepare()
    failure, status, gate = None, "completed", {"decision": "inconclusive", "reason": "screen_not_completed"}
    try:
        budget.check()
        with GodotHost(host_checkout, godot, lane="act004", mode="stepped") as host:
            screen.host_identity = host.identity
            episodes = []
            for family in config["training_families"]:
                for seed in family["demo_seeds"]:
                    episodes.append(screen.run_episode(host, family, seed, 0, "authored", "demonstration"))
            dataset = Dataset("t110-authored-training", tuple(episodes), screen.split)
            screen.train_candidates(dataset)
            screen.evaluate(host, "tuning", config["tuning_families"], ("normal",), config["evaluation"]["methods"])
            screen.open_holdout()
            screen.evaluate(host, "evaluation", config["evaluation_families"],
                            tuple(item["id"] for item in config["faults"]), config["evaluation"]["methods"])
            screen.evaluate(host, "untrained", config["evaluation_families"], ("normal",),
                            ("graph", "no_message"), initial=True)
            budget.check()
            gate = utility_gate(screen.rows, config)
            if gate["decision"] == "inconclusive":
                status, failure = "inconclusive", gate["reason"]
    except BaseException as exc:
        failure = type(exc).__name__ + (":" + exc.code if isinstance(exc, ContractError) else "")
        status = "capped" if isinstance(exc, CaptureBudgetExceeded) else "canceled" if isinstance(exc, KeyboardInterrupt) else "failed"
    summary = {"status": status, "failure": failure, "config_sha256": CONFIG_SHA256,
               "learner_revision": revision, "learner_sources_sha256": source_hash,
               "game_decisions": budget.decisions, "runtime_seconds": budget.elapsed,
               "attempt_game_decisions": budget.attempt_decisions,
               "attempt_runtime_seconds": budget.attempt_elapsed,
               "prior_budget": prior.to_dict(), "prior_budget_sha256": prior.sha256,
               "rows": screen.rows, "utility_gate": gate, "frozen_checkpoints": len(screen.checkpoints),
               "training": screen.training_reports, "owner_feel_accepted": False,
               "direct_cell_write_requests": 0, "physical_topology_mutation_measured": False}
    screen.save_json("screen-summary.json", summary, "screen_summary")
    screen.write_manifest("t110-jolt-three-part-screen", status, budget.elapsed, budget.decisions,
                          summary, tuple(screen.artifacts), () if failure is None else (failure,))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new external artifact directory, outside Git")
    parser.add_argument("--prior-budget", type=Path,
                        help="immutable prior-attempt receipts, charged against the same 4000/900 ceiling")
    parser.add_argument("--host", default=os.environ.get("NCA_HOST_CHECKOUT"))
    parser.add_argument("--godot", default=os.environ.get("NCA_GODOT"))
    args = parser.parse_args(argv)
    if not args.host or not args.godot:
        parser.error("set NCA_HOST_CHECKOUT and NCA_GODOT, or pass --host and --godot")
    try:
        summary = run_screen(config_path=args.config, source_root=args.source_root, output=args.output,
                             host_checkout=args.host, godot=args.godot, prior_budget=args.prior_budget)
    except (ContractError, FileExistsError) as exc:
        # An escaped recording failure may follow actual host work. Its phase is
        # unknown here; only the immutable attempt artifacts establish progress.
        print(json.dumps({"status": "failed", "reason": getattr(exc, "code", type(exc).__name__),
                          "phase": "unknown_consult_attempt_artifacts"}))
        return 1
    print(json.dumps({"status": summary["status"], "failure": summary["failure"],
                      "game_decisions": summary["game_decisions"],
                      "utility_decision": summary["utility_gate"]["decision"]}, sort_keys=True), flush=True)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
