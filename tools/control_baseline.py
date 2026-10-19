#!/usr/bin/env python3
"""Reproduce the bounded matched Jolt/Vector fixture screen; no training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nca.adapters.godot import CybersandEnv, GodotHost, physical_config  # noqa: E402
from nca.controllers import BaselineTuning  # noqa: E402
from nca.contracts import WireAction  # noqa: E402
from nca.evaluation.feel import (  # noqa: E402
    CybersandFixtureHost, MotionSample, fixture_suite, measure_response,
    render_pose_comparison, run_fixture,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("NCA_HOST_CHECKOUT"))
    parser.add_argument("--godot", default=os.environ.get("NCA_GODOT"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--skip-clips", action="store_true")
    args = parser.parse_args()
    if not args.host or not args.godot:
        parser.error("provide --host/--godot or NCA_HOST_CHECKOUT/NCA_GODOT")
    if not 0 < args.max_seconds <= 600:
        parser.error("the bounded fixture screen permits at most 600 seconds")
    if args.output.exists():
        parser.error("output already exists; select a new immutable run directory")
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src/nca/controllers").glob("*.py")) + [
        root / "src/nca/evaluation/feel.py", root / "src/nca/adapters/godot.py",
        root / "src/nca/adapters/protocol.py", root / "src/nca/adapters/host_bridge.gd",
        Path(__file__).resolve(),
    ]
    # Optional learned modules are unrelated to this authored screen.
    sources = [path for path in sources if path.name != "learned.py"]
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sources}
    report = {
        "schema": "nca.t103-pilots/v2", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "learner_base_revision": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashes, "python": platform.python_version(), "device": "CPU",
        "runtime_os": platform.system(), "architecture": platform.machine(),
        "seed": 1703, "runs": [], "clips": [], "errors": [], "maximum_decisions": 1080,
        "maximum_wall_seconds": args.max_seconds, "evidence_class": "actual-game",
        "owner_feel_judgment": "not_recorded", "training": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        with GodotHost(args.host, args.godot) as host:
            report["host_identity"] = host.identity
            for actor in ("jev", "astra"):
                for variant in ("firm", "balanced", "compliant"):
                    for fixture in fixture_suite(actor)[:3]:
                        if time.monotonic() - started >= args.max_seconds:
                            raise TimeoutError("bounded screen wall budget reached")
                        run_started = time.monotonic()
                        env = CybersandEnv(host, physical_config(actor, ticks=fixture.duration_ticks, seed=fixture.seed))
                        env.reset(seed=fixture.seed)
                        wrapper = CybersandFixtureHost(env)
                        trace = run_fixture(wrapper, fixture, tuning=BaselineTuning.named(variant))
                        trace["consumed_receipts"] = wrapper.receipts
                        trace["resolved_host_config"] = env.config
                        trace["source_sha256"] = hashes
                        filename = f"{actor}-{variant}-{fixture.id.split('/')[0]}.json.gz"
                        artifact = gzip.compress(json.dumps(trace, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(), mtime=0)
                        with (args.output / filename).open("xb") as stream:
                            stream.write(artifact)
                        points = []
                        for row in trace["samples"]:
                            point = {key: row[key] for key in MotionSample.__dataclass_fields__}
                            if point["requested"] is not None:
                                point["requested"] = WireAction.from_dict(point["requested"])
                            points.append(MotionSample(**point))
                        goals = [event for event in fixture.events if event.kind == "goal"]
                        boundaries = [event.tick for event in goals] + [fixture.duration_ticks]
                        segments = []
                        for index, event in enumerate(goals):
                            segment = tuple(row for row in points if event.tick <= row.tick <= boundaries[index + 1])
                            target = tuple(points[0].position[i] + event.parameters[i] for i in range(2))
                            segments.append(asdict(measure_response(segment, origin=segment[0].position,
                                                                   target=target, command_tick=event.tick)))
                        goal_ids = {row["goal_id"] for row in trace["samples"] if row.get("goal_id")}
                        result = {
                            "actor": actor, "tuning": variant, "fixture": fixture.id,
                            "wall_seconds": time.monotonic() - run_started, "decisions": len(wrapper.receipts),
                            "final_position": trace["samples"][-1]["position"],
                            "terminal_reason": trace["terminal_reason"], "segments": segments,
                            "artifact": filename, "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
                            "artifact_bytes": len(artifact), "fixture_sha256": fixture.digest,
                            "tuning_sha256": BaselineTuning.named(variant).digest,
                            "local_success_by_goal": {goal: next((row["tick"] for row in trace["samples"]
                                if row.get("goal_id") == goal and row.get("controller_status") == "succeeded"), None)
                                for goal in sorted(goal_ids)},
                        }
                        report["runs"].append(result)
                        print(json.dumps({key: result[key] for key in ("actor", "tuning", "fixture", "local_success_by_goal")}), flush=True)
        report["fixture_wall_seconds"] = time.monotonic() - started
        if not args.skip_clips:
            for actor in ("jev", "astra"):
                for fixture_name in ("step", "pulse", "reversal"):
                    traces = {variant: json.loads(gzip.decompress((args.output / f"{actor}-{variant}-{fixture_name}.json.gz").read_bytes()))
                              for variant in ("firm", "balanced", "compliant")}
                    filename = f"{actor}-{fixture_name}-matched-poses.gif"
                    render_pose_comparison(traces, str(args.output / filename))
                    artifact = (args.output / filename).read_bytes()
                    report["clips"].append({"artifact": filename, "sha256": hashlib.sha256(artifact).hexdigest(),
                                            "bytes": len(artifact), "kind": "matched recorded body pose replay; not game-rendered footage"})
    except Exception as exc:
        report["errors"].append({"type": type(exc).__name__, "reason": "screen did not complete; inspect local diagnostics"})
        # A portable failure record still retains every completed run and artifact.
        print(f"Screen failed: {type(exc).__name__}", file=sys.stderr)
    if any(hashlib.sha256((root / path).read_bytes()).hexdigest() != digest for path, digest in hashes.items()):
        report["errors"].append({"type": "SourceDrift", "reason": "scoped source changed during screen"})
    report["total_wall_seconds"] = time.monotonic() - started
    report["total_decisions"] = sum(row["decisions"] for row in report["runs"])
    report["total_host_ticks"] = report["total_decisions"] * 6
    report["status"] = "passed" if not report["errors"] and len(report["runs"]) == 18 else "failed"
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "decisions": report["total_decisions"],
                      "summary_sha256": hashlib.sha256((args.output / "summary.json").read_bytes()).hexdigest()}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
