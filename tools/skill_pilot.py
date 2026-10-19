#!/usr/bin/env python3
"""Record short actual-owner usable fixtures; no model calls or weight training."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nca.adapters.godot import sha256_file  # noqa: E402
from nca.contracts import sha256_json  # noqa: E402
from nca.envs import CybersandEnv, GodotHost, physical_config  # noqa: E402
from nca.skills import SkillRuntime, interval_observations  # noqa: E402
from nca.skills.tasks import SkillSpec, TaskPlan, TaskStep, run_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, default=os.environ.get("NCA_HOST_CHECKOUT"))
    parser.add_argument("--godot", type=Path, default=os.environ.get("NCA_GODOT"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--max-seconds", type=float, default=120)
    args = parser.parse_args()
    if not args.host or not args.godot or not 0 <= args.seed <= 2147483647 or not 1 <= args.max_seconds <= 600:
        parser.error("real runtime, bounded seed and a 1..600-second budget are required")
    if args.output.exists():
        parser.error("immutable evidence already exists; choose a new output path")
    root = Path(__file__).resolve().parents[1]
    names = ["src/nca/skills/__init__.py", "src/nca/skills/usables.py", "src/nca/skills/tasks.py",
             "src/nca/adapters/protocol.py", "src/nca/adapters/godot.py", "src/nca/adapters/host_bridge.gd",
             "src/nca/controllers/baseline.py", "src/nca/controllers/shaping.py", "src/nca/contracts.py",
             "tests/test_skills.py", "tests/test_skills_host.py", "tools/skill_pilot.py"]
    hashes = {name: sha256_file(root / name) for name in names}
    result = {"format": "nca-usable-pilot/1", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source_sha256": hashes, "scoped_source_sha256": sha256_json(hashes),
              "python": platform.python_version(), "platform": platform.system(),
              "architecture": platform.machine(), "device": "CPU", "logical_cpu_count": os.cpu_count(),
              "seed": args.seed, "max_seconds": args.max_seconds,
              "base_hz": 60, "decision_ticks": 6, "helper_lease_ticks": 12,
              "observation_schema": "ACT004-local/v2", "action_schema": "physical_wire_action/v2",
              "training": False, "model_calls": 0, "owner_feel_accepted": False,
              "cases": [], "errors": []}
    started = time.perf_counter()

    def budget():
        if time.perf_counter() - started >= args.max_seconds:
            raise TimeoutError("bounded fixture budget exhausted")

    try:
        with GodotHost(args.host, args.godot) as host:
            result["host_identity"] = host.identity

            def environment(model, position, *, ticks=600):
                budget()
                config = physical_config(model, ticks=ticks, seed=args.seed)
                if position is not None:
                    config["actors"][0]["position"] = list(position)
                config["local_patch_radius"] = 8
                env = CybersandEnv(host, config)
                view, _ = env.reset()
                runtime = SkillRuntime(host.request("profile")["profile"])
                return env, view, runtime, config

            def task_case(name, model, position, plan, expected="completed", settle=0):
                env, view, runtime, config = environment(model, position)

                class BudgetedEnv:
                    def step(self, action):
                        budget()
                        return env.step(action)

                    def cancel(self):
                        return env.cancel()

                preparation = []
                for _ in range(settle // 6):
                    view, _, _, _, info = env.step({"move": [0, 0]})
                    preparation.append({"actual_ticks": info["actual_ticks"], "command": {"move": [0, 0]}})
                initial = view
                outcome, final, trace = run_task(BudgetedEnv(), runtime, plan, view)
                budget()
                case = {"name": name, "model": model, "resolved_config": config,
                        "profile_sha256": runtime.capability.profile_sha256, "source": "script",
                        "preparation": preparation, "plan": plan.to_dict(), "expected_status": expected,
                        "expectation_satisfied": outcome.status == expected,
                        "outcome": outcome.to_dict(), "initial_position": initial["position"],
                        "final_position": final["position"], "final_tick": final["tick"],
                        "final_detached": [part["detached"] for part in final["parts"]],
                        "trace": list(trace), "trace_sha256": sha256_json(trace),
                        "runtime_events": runtime.events, "dropped_runtime_events": runtime.dropped_events}
                result["cases"].append(case)

            task_case("launcher-clearance-recovery", "astra", (300, 300), TaskPlan("launcher-recovery", (
                TaskStep(SkillSpec("approach", {"target": [304, 300], "tolerance": 2.5})),
                TaskStep(SkillSpec("aim_use", {"aim": [1, 0]}),
                         (SkillSpec("recover"), SkillSpec("aim_use", {"aim": [0, -1], "require_change": True}))),
            )))
            task_case("articulated-pickup-carry-use-release", "hammerhead", (300, 300), TaskPlan("held-carry", (
                TaskStep(SkillSpec("release")), TaskStep(SkillSpec("grip")),
                TaskStep(SkillSpec("carry", {"target": [306, 344], "tolerance": 4,
                                            "timeout_ticks": 240, "no_progress_ticks": 90})),
                TaskStep(SkillSpec("aim_use", {"aim": [1, 0], "require_change": True})),
                TaskStep(SkillSpec("release")),
            )))
            task_case("late-pickup-no-progress", "hammerhead", (300, 344), TaskPlan("late-pickup", (
                TaskStep(SkillSpec("release")), TaskStep(SkillSpec("grip")),
            )), expected="failed", settle=24)

            env, view, runtime, config = environment("astra", (320, 243))
            receipt = runtime.start("aim_use", {"aim": [0.4, -math.sqrt(0.84)], "require_change": True}, view)
            feedback = ()
            for _ in range(16):
                budget()
                output = runtime.progress(view, observations=feedback)
                if output.terminal:
                    break
                view, _, _, _, info = env.step(output.action.to_dict())
                feedback = interval_observations(info)
            before = view
            cancelled = env.cancel()
            after, _, _, _, info = env.step_packet(None, ticks=6)
            preserved = (output.local_success is True and 6 in before["cell_patch"]["materials"]
                         and before["cell_patch"] == cancelled["cell_patch"]
                         and before["velocity"] == cancelled["velocity"]
                         and all(not tick["observation"]["last_effects"] for tick in info["trace"]))
            result["cases"].append({"name": "native-effect-survives-cancellation", "model": "astra",
                                    "resolved_config": config, "profile_sha256": runtime.capability.profile_sha256,
                                    "expectation_satisfied": preserved, "initiation": receipt.to_dict(),
                                    "outcome": output.to_dict(), "cancel_tick": cancelled["tick"],
                                    "tick_after_passive_interval": after["tick"],
                                    "cell_patch_before_cancel": before["cell_patch"],
                                    "cell_patch_after_cancel": cancelled["cell_patch"],
                                    "cell_patch_after_native_interval": after["cell_patch"],
                                    "velocity_before_cancel": before["velocity"],
                                    "velocity_after_cancel": cancelled["velocity"],
                                    "runtime_events": runtime.events})

            env, view, runtime, config = environment("pouncer", None)
            receipt = runtime.start("charge_hold", {}, view)
            feedback = ()
            for _ in range(12):
                budget()
                output = runtime.progress(view, observations=feedback)
                if output.terminal:
                    break
                view, _, _, _, info = env.step(output.action.to_dict())
                feedback = interval_observations(info)
                if view["charge"] >= 0.35:
                    break
            before = view
            output = runtime.cancel("human_takeover")
            cancelled = env.cancel()
            after, _, _, _, info = env.step_packet(None, ticks=6)
            disarmed = (before["charge"] >= 0.35 and cancelled["charge"] == 0
                        and before["velocity"] == cancelled["velocity"]
                        and all(tick["observation"]["charge"] == 0
                                and math.hypot(*tick["observation"]["applied_impulse"]) < 0.001
                                for tick in info["trace"]))
            result["cases"].append({"name": "charge-cancellation-without-launch", "model": "pouncer",
                                    "resolved_config": config, "profile_sha256": runtime.capability.profile_sha256,
                                    "expectation_satisfied": disarmed, "initiation": receipt.to_dict(),
                                    "outcome": output.to_dict(), "charge_before_cancel": before["charge"],
                                    "charge_after_cancel": cancelled["charge"],
                                    "charge_after_passive_interval": after["charge"],
                                    "velocity_before_cancel": before["velocity"],
                                    "velocity_after_cancel": cancelled["velocity"],
                                    "post_cancel_impulses": [tick["observation"]["applied_impulse"] for tick in info["trace"]],
                                    "runtime_events": runtime.events})
            result["diagnostics"] = list(host.diagnostics)
    except Exception as error:
        result["errors"].append({"type": type(error).__name__, "reason": "fixture stopped; inspect local diagnostics"})
    if any(sha256_file(root / name) != digest for name, digest in hashes.items()):
        result["errors"].append({"type": "SourceDrift", "reason": "a scoped implementation changed during the run"})
    result["elapsed_seconds"] = time.perf_counter() - started
    result["status"] = "passed" if (not result["errors"] and len(result["cases"]) == 5
                                     and all(case["expectation_satisfied"] for case in result["cases"])) else "failed"
    result["evidence_limits"] = ["five declared fixtures, not an exhaustive usable catalogue",
                                  "local success only; no privileged task evaluator",
                                  "model-intent labels have no live model calls",
                                  "no arbitrary independent mount/object replacement API",
                                  "native Fire/material continuation only; no separately qualified projectile timer",
                                  "one Linux CPU runtime; Windows and owner feel remain unmeasured"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                      "cases": [{"name": case["name"], "expectation_satisfied": case["expectation_satisfied"]}
                                for case in result["cases"]], "errors": result["errors"]}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
