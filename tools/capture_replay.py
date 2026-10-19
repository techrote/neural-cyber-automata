#!/usr/bin/env python3
"""Bounded ACT004 demonstration capture, consumed-request replay and run receipt."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
import uuid
from pathlib import Path

from nca.adapters.godot import CybersandEnv, GodotHost, PINNED_HOST_REVISION, physical_config
from nca.contracts import sha256_json
from nca.controllers import AuthoredController, FeedbackObservation, SkillGoal
from nca.data import (ActionSource, CaptureBudgetExceeded, EpisodeRecorder, PolicyContext,
                      replay_episode, source_identity, write_episode)
from nca.randomness import ScenarioSplit
from nca.runs import ArtifactRef, RunBudget, RunManifest, RunStore, sha256_file


def control_boundaries(recorder: EpisodeRecorder) -> None:
    """Actual scripted source changes; this is not a claim of recorded human input."""
    recorder.step({"move": [.7, 0], "effort": .7})
    original = recorder.events[-1].requested_packet
    recorder.step_packet(original, ticks=1)  # duplicate attempt, no new consumption
    recorder.cancel("scripted_cancel_fixture")
    recorder.step({"move": [-.4, 0], "effort": .5})
    queued = recorder.env.command({"move": [1, 0], "primary": True})
    correction = ActionSource("script-b", "scripted", "control-boundaries-script-b/v1",
                              sha256_json({"fixture": "control-boundaries", "source": "b"}))
    recorder.takeover(correction, "scripted_source_transfer_fixture")
    recorder.step_packet(queued, ticks=1)  # old observation generation must fail
    while recorder.view["tick"] < recorder.env.config["max_ticks"]:
        recorder.step({"move": [.1, 0], "effort": .4})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("NCA_HOST_CHECKOUT"))
    parser.add_argument("--godot", default=os.environ.get("NCA_GODOT"))
    parser.add_argument("--output", type=Path, required=True, help="new .json.gz episode artifact")
    parser.add_argument("--run-id", default="capture-" + uuid.uuid4().hex[:12])
    parser.add_argument("--suite", choices=("control-boundaries", "jolt-demo"), default="control-boundaries")
    parser.add_argument("--ticks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-offset", type=float, default=48.0)
    parser.add_argument("--pace", type=float, default=.8)
    args = parser.parse_args()
    if not args.host or not args.godot:
        parser.error("set NCA_HOST_CHECKOUT and NCA_GODOT, or pass --host/--godot")
    if not 26 <= args.ticks <= 600:
        parser.error("--ticks must be between 26 and 600 for this bounded capture")
    if not math.isfinite(args.target_offset) or abs(args.target_offset) > 100 or not 0 < args.pace <= 1:
        parser.error("target offset must be finite within +/-100 cells; pace must be in (0,1]")
    if args.output.exists():
        parser.error("--output already exists; immutable captures require a new destination")
    started = time.monotonic()
    revision, files, tree_hash = source_identity()
    recorder = None
    episode = None
    report = None
    failure = None
    artifact_refs = []
    host_identity = {}
    budget = RunBudget(100, 120)
    config = physical_config("jev", ticks=args.ticks, seed=args.seed)
    try:
        with GodotHost(args.host, args.godot) as host:
            host_identity = host.identity
            env = CybersandEnv(host, config)
            if args.suite == "jolt-demo":
                controller = AuthoredController("jev")
                source = ActionSource("jolt-script", "scripted", "authored-directed-hop/v1",
                                      sha256_json({"controller": "authored-directed-hop/v1",
                                                   "tuning": controller.tuning.digest,
                                                   "source_sha256": files.get("src/nca/controllers/baseline.py")}))
            else:
                controller = None
                source = ActionSource("script-a", "scripted", "control-boundaries-script-a/v1",
                                      sha256_json({"fixture": "control-boundaries", "source": "a"}))
            recorder = EpisodeRecorder(env, args.run_id, source, seed=args.seed,
                                       scenario_family=args.suite, split="evaluation", budget=budget)
            if controller is not None:
                target = (recorder.view["position"][0] + args.target_offset, recorder.view["position"][1])
                goal = SkillGoal("jolt-demo", "directed_hop", target, pace=args.pace)
                recorder.context = PolicyContext(goal.id, goal.skill, goal.target, goal.pace,
                                                 goal.tolerance, goal.timeout_seconds)
                while recorder.view["tick"] < args.ticks:
                    frame = controller.step(FeedbackObservation.from_host(recorder.view), goal)
                    recorder.step(frame.action.to_dict())
            else:
                control_boundaries(recorder)
            evaluation = ({"goal_id": goal.id, "controller_status_estimate": frame.status,
                           "distance_to_goal": math.dist(recorder.view["position"], goal.target),
                           "speed": math.hypot(*recorder.view["velocity"]), "owner_feel_accepted": False}
                          if controller is not None else {"task_quality": None, "owner_feel_accepted": False})
            episode = recorder.finish(evaluation=evaluation)
            content_hash = write_episode(args.output, episode)
            artifact_refs.append(ArtifactRef(args.output.name, content_hash, args.output.stat().st_size, "episode"))
            report = replay_episode(episode, host)
            report_path = args.output.with_name(args.output.name + ".replay.json")
            with report_path.open("x", encoding="utf-8") as stream:
                stream.write(report.to_json() + "\n")
            artifact_refs.append(ArtifactRef(report_path.name, sha256_file(report_path), report_path.stat().st_size, "replay"))
            if not report.passed:
                failure = "replay_tolerance_exceeded"
    except Exception as exc:
        failure = type(exc).__name__
        if recorder is not None and not recorder._done:
            episode = recorder.finish(status="capped" if isinstance(exc, CaptureBudgetExceeded) else "failed",
                                      reason=failure)
            content_hash = write_episode(args.output, episode)
            artifact_refs.append(ArtifactRef(args.output.name, content_hash, args.output.stat().st_size, "episode"))
    elapsed = time.monotonic() - started
    manifest = RunManifest(
        run_id=args.run_id, experiment="demonstration-capture-replay", learner_revision=revision,
        host_revision=PINNED_HOST_REVISION,
        profile_sha256=episode.profile_sha256 if episode else sha256_json({"profile_unavailable": failure}),
        resolved_config={"config": config, "suite": args.suite, "target_offset": args.target_offset,
                         "pace": args.pace, "learner_sources_sha256": tree_hash},
        environment={"python": platform.python_version(), "system": platform.system(),
                     "machine": platform.machine(), "host": host_identity},
        seeds=(args.seed,), scenario_split=ScenarioSplit(("future-training",), ("future-tuning",), (args.suite,)),
        status="completed" if failure is None else "capped" if failure == "CaptureBudgetExceeded" else "failed",
        gameplay_lane="physical_ACT004", execution_lane="stepped", budget=budget,
        observation_schema="act004-local-v1", action_schema="act004-wire-v2", suite="capture-replay-v1",
        decisions=episode.decisions if episode else 0, runtime_seconds=elapsed,
        metrics={"consumed_ticks": episode.consumed_ticks if episode else None,
                 "replay_passed": report.passed if report else None,
                 "replay_maximum_errors": dict(report.maximum_errors) if report else None,
                 "trained_policy": False, "owner_feel_accepted": False},
        artifacts=tuple(artifact_refs), failures=() if failure is None else (failure,))
    RunStore(args.output.parent / "runs").write(manifest)
    print(json.dumps({"run_id": args.run_id, "status": manifest.status,
                      "decisions": manifest.decisions,
                      "consumed_ticks": episode.consumed_ticks if episode else None,
                      "replay_passed": report.passed if report else None,
                      "failure": failure}, sort_keys=True))
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
