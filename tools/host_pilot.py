#!/usr/bin/env python3
"""Bounded CPU host throughput/conformance pilot; no weight training."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nca.adapters.godot import CybersandEnv, GodotHost, physical_config, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, default=os.environ.get("NCA_HOST_CHECKOUT"))
    parser.add_argument("--godot", type=Path, default=os.environ.get("NCA_GODOT"))
    parser.add_argument("--model", default="ripper")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--decisions", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.host or not args.godot:
        parser.error("--host and --godot or matching environment variables are required")
    if not 1 <= args.decisions <= 2000 or not 0 < args.max_seconds <= 600:
        parser.error("budget: 1..2000 decisions and >0..600 seconds")
    if args.output.exists():
        parser.error("evidence output already exists; use a new immutable run path")
    root = Path(__file__).resolve().parents[1]
    source_files = ["src/nca/adapters/protocol.py", "src/nca/adapters/godot.py", "src/nca/adapters/host_bridge.gd",
                    "tools/host_pilot.py"]
    code_hashes = {name: sha256_file(root / name) for name in source_files}
    record = {"format": "nca-host-pilot/1", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "adapter_source_sha256": code_hashes, "python": platform.python_version(),
              "platform": platform.system(), "architecture": platform.machine(), "device": "CPU",
              "logical_cpu_count": os.cpu_count(), "max_seconds": args.max_seconds,
              "maximum_decisions": args.decisions, "seed": args.seed, "model": args.model,
              "training": False, "owner_feel_accepted": False, "errors": []}
    times: list[float] = []
    ticks = 0
    trace_hash = hashlib.sha256()
    started = time.perf_counter()
    try:
        with GodotHost(args.host, args.godot) as host:
            record["host_identity"] = host.identity
            config = physical_config(args.model, ticks=args.decisions * 6, seed=args.seed)
            record["resolved_config"] = config
            env = CybersandEnv(host, config)
            reset_start = time.perf_counter()
            observation, _ = env.reset()
            record["reset_seconds"] = time.perf_counter() - reset_start
            profile = host.request("profile")["profile"]
            record["profile_sha256"] = hashlib.sha256(
                json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            record["observation_version"] = observation.get("version")
            record["action_schema"] = "physical_wire_action/v2"
            record["decision_ticks"] = env.decision_ticks
            record["helper_lease_ticks"] = env.lease_ticks
            record["part_count"] = observation["part_count"]
            record["initial_position"] = observation["position"]
            step_start = time.perf_counter()
            for index in range(args.decisions):
                if time.perf_counter() - started >= args.max_seconds:
                    break
                before = time.perf_counter()
                # A fixed bounded input measures the owner path, not skill quality.
                action = {"move": [1.0 if index % 40 < 20 else -1.0, -0.1], "effort": 0.7}
                observation, reward, terminated, truncated, info = env.step(action)
                times.append(time.perf_counter() - before)
                ticks += info["actual_ticks"]
                trace_hash.update(json.dumps(info["trace"], sort_keys=True, separators=(",", ":")).encode())
                if terminated or truncated:
                    break
            elapsed = time.perf_counter() - step_start
            record.update(actual_ticks=ticks, decisions=len(times), step_wall_seconds=elapsed,
                          ticks_per_second=ticks / elapsed, decisions_per_second=len(times) / elapsed,
                          mean_decision_seconds=statistics.mean(times) if times else None,
                          p95_decision_seconds=sorted(times)[max(0, int(len(times) * 0.95) - 1)] if times else None,
                          final_position=observation["position"], trace_sha256=trace_hash.hexdigest(),
                          terminal_tick=observation["tick"], diagnostics=list(host.diagnostics),
                          completed_decision_budget=len(times) == args.decisions)
    except Exception as exc:
        # No local paths, host names or private stack traces in public records.
        record["errors"].append({"type": type(exc).__name__, "reason": "host pilot failed; inspect local diagnostics"})
    if any(sha256_file(root / name) != digest for name, digest in code_hashes.items()):
        record["errors"].append({"type": "SourceDrift", "reason": "adapter source changed during pilot"})
    record["elapsed_seconds"] = time.perf_counter() - started
    record["status"] = "passed" if not record["errors"] and ticks else "failed"
    record["evidence_limits"] = ["one process on this CPU environment", "no learned control or score",
                                  "no desktop asynchronous schedule parity", "owner feel not judged",
                                  "per-part passive wrench not supplied by accepted host"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: record.get(key) for key in ("status", "model", "actual_ticks", "decisions", "ticks_per_second", "errors")}))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
