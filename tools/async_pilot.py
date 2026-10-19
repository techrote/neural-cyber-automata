#!/usr/bin/env python3
"""Record bounded actual realtime conformance, including injected provider faults.

The fixed conformance cases live beside their assertions in test_decisions_host.
This command adds immutable source/runtime and complete owner-operation receipts;
it does not substitute a simulator or a live model service for those cases.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time
import unittest

from nca.contracts import canonical_json, sha256_json
from nca.data.capture import source_identity
from nca.envs import GodotHost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--godot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".index.json").exists():
        parser.error("choose a new immutable output destination")
    root = Path(__file__).resolve().parents[1]
    revision, sources, source_hash = source_identity(root)
    scoped = ("src/nca/adapters/decisions.py", "src/nca/adapters/skill_control.py",
              "src/nca/adapters/godot.py", "src/nca/adapters/host_bridge.gd",
              "src/nca/skills/usables.py", "tests/test_decisions.py",
              "tests/test_decisions_host.py", "tools/async_pilot.py")
    scoped_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in scoped}
    os.environ["NCA_HOST_CHECKOUT"] = str(args.checkout)
    os.environ["NCA_GODOT"] = str(args.godot)
    specification = importlib.util.spec_from_file_location("actual_decision_cases", root / "tests/test_decisions_host.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    operations, cases = [], []
    started = time.monotonic()
    host_identity = None

    class RecordedHost:
        def __init__(self):
            nonlocal host_identity
            self.actual = GodotHost(args.checkout, args.godot, mode="realtime")
            host_identity = self.actual.identity
            self.steps = 0

        def __getattr__(self, name):
            return getattr(self.actual, name)

        def request(self, operation, **fields):
            if time.monotonic() - started > 120 or self.steps >= 2000:
                raise RuntimeError("predeclared pilot budget exhausted")
            before = time.monotonic()
            result = self.actual.request(operation, **fields)
            self.steps += operation == "step"
            operations.append({"case": cases[-1]["name"] if cases else "setup",
                               "operation": operation, "request": fields, "response": result,
                               "wall_seconds": time.monotonic() - before})
            return result

        def close(self):
            self.actual.close()

    class RecordedCases(module.ContinuousDecisionTests):
        @classmethod
        def setUpClass(cls):
            cls.host = RecordedHost()

        def setUp(self):
            cases.append({"name": self._testMethodName, "operation_start": len(operations)})

        def tearDown(self):
            cases[-1]["operation_end"] = len(operations)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecordedCases)
    result = unittest.TestResult()
    suite.run(result)
    failed = {test._testMethodName: details.splitlines()[-1] for test, details in (*result.failures, *result.errors)}
    for case in cases:
        case["passed"] = case["name"] not in failed
        case["failure"] = failed.get(case["name"])
    drift = [name for name, digest in scoped_hashes.items()
             if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    record = {"schema": "nca-async-realtime-pilot/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "learner_revision": revision, "learner_sources_sha256": source_hash, "source_files": sources,
              "scoped_source_sha256": sha256_json(scoped_hashes), "scoped_source_files": scoped_hashes,
              "source_drift": drift, "host_identity": host_identity, "python": platform.python_version(),
              "platform": platform.system(), "architecture": platform.machine(),
              "budget": {"wall_seconds": 120, "step_calls": 2000}, "mode": "realtime",
              "provider": "deterministic mocks and injected failures", "live_calls": 0, "tokens": None,
              "cost": 0, "owner_feel_accepted": False, "evaluator_success": None,
              "tests_run": result.testsRun, "skipped": len(result.skipped), "cases": cases,
              "operations": operations, "wall_seconds": time.monotonic() - started,
              "passed": result.wasSuccessful() and not result.skipped and not drift,
              "limits": ["scripted human-priority injection, no live human session",
                         "actual detach/reattach and full reset, no independent object replacement",
                         "Linux realtime headless owner, no desktop or other-platform timing qualification"]}
    payload = gzip.compress(canonical_json(record).encode(), mtime=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
    index = {key: value for key, value in record.items() if key != "operations"}
    index["artifact"] = {"path": args.output.name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    index["owner_operations"] = len(operations)
    index["step_calls"] = sum(item["operation"] == "step" for item in operations)
    index["actual_ticks"] = sum(item["response"].get("actual_ticks", 0) for item in operations if item["operation"] == "step")
    with args.output.with_suffix(".index.json").open("x") as stream:
        json.dump(index, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: index[key] for key in ("passed", "tests_run", "skipped", "wall_seconds",
                                                "owner_operations", "step_calls", "actual_ticks", "artifact")}, indent=2))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
