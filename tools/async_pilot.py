#!/usr/bin/env python3
"""Record bounded stepped and realtime conformance, including provider faults.

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
    hosts = []
    attempted_step_calls = 0

    class RecordedHost:
        def __init__(self, mode):
            if time.monotonic() - started > 120:
                raise RuntimeError("predeclared pilot budget exhausted before host start")
            self.actual = GodotHost(args.checkout, args.godot, mode=mode)
            self.id = f"host-{len(hosts) + 1}"
            self.receipt = {"id": self.id, "mode": mode, "identity": self.actual.identity,
                            "cleanup": {"status": "not_called", "internal_close_rpc_recorded": False}}
            hosts.append(self.receipt)

        def __getattr__(self, name):
            return getattr(self.actual, name)

        def request(self, operation, **fields):
            nonlocal attempted_step_calls
            if time.monotonic() - started > 120 or attempted_step_calls >= 2000:
                raise RuntimeError("predeclared pilot budget exhausted")
            before = time.monotonic()
            attempted_step_calls += operation == "step"
            record = {"case": cases[-1]["name"] if cases else "setup", "host_id": self.id,
                      "mode": self.actual.mode, "operation": operation, "request": fields,
                      "response": None, "status": "incomplete", "error_type": None}
            operations.append(record)
            try:
                result = self.actual.request(operation, **fields)
                record.update(response=result, status="completed")
                return result
            except BaseException as exc:
                record["error_type"] = type(exc).__name__
                raise
            finally:
                record["wall_seconds"] = time.monotonic() - before

        def close(self):
            # Cleanup remains permitted after the execution budget is exhausted.
            # GodotHost.close owns its graceful RPC and process termination; that
            # internal exchange is distinct from recorded test-driven requests.
            before = time.monotonic()
            try:
                self.actual.close()
                self.receipt["cleanup"]["status"] = "completed"
            except BaseException as exc:
                self.receipt["cleanup"].update(status="failed", error_type=type(exc).__name__)
                raise
            finally:
                self.receipt["cleanup"]["wall_seconds"] = time.monotonic() - before

    def recorded_cases(base):
        class RecordedCases(base):
            @classmethod
            def setUpClass(cls):
                cls.host = RecordedHost(cls.mode)

            def setUp(self):
                cases.append({"id": self.id(), "name": self._testMethodName, "mode": self.mode, "host_id": self.host.id,
                              "operation_start": len(operations)})

            def tearDown(self):
                cases[-1]["operation_end"] = len(operations)

        RecordedCases.__name__ = "Recorded" + base.__name__
        RecordedCases.__qualname__ = RecordedCases.__name__
        return RecordedCases

    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(recorded_cases(base))
                               for base in (module.SteppedDecisionTests, module.ContinuousDecisionTests))
    result = unittest.TestResult()
    suite.run(result)
    failure_details = [{"id": test.id(), "case_id": getattr(test, "test_case", test).id(),
                        "detail": details.splitlines()[-1]}
                       for test, details in (*result.failures, *result.errors)]
    failed = {failure["case_id"]: failure["detail"] for failure in failure_details}
    for case in cases:
        case["passed"] = case["id"] not in failed
        case["failure"] = failed.get(case["id"])
    drift = [name for name, digest in scoped_hashes.items()
             if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    elapsed = time.monotonic() - started
    within_budget = elapsed <= 120 and attempted_step_calls <= 2000
    record = {"schema": "nca-async-host-pilot/v2", "created_utc": datetime.now(timezone.utc).isoformat(),
              "learner_revision": revision, "learner_sources_sha256": source_hash, "source_files": sources,
              "scoped_source_sha256": sha256_json(scoped_hashes), "scoped_source_files": scoped_hashes,
              "source_drift": drift, "hosts": hosts, "python": platform.python_version(),
              "platform": platform.system(), "architecture": platform.machine(),
              "budget": {"wall_seconds": 120, "step_calls": 2000}, "mode": "mixed-stepped-realtime",
              "attempted_step_calls": attempted_step_calls, "within_budget": within_budget,
              "provider": "deterministic mocks and injected failures", "live_calls": 0, "tokens": None,
              "cost": 0, "owner_feel_accepted": False, "evaluator_success": None,
              "tests_run": result.testsRun, "skipped": len(result.skipped), "cases": cases,
              "suite_failures": failure_details,
              "operations": operations, "wall_seconds": elapsed,
              "passed": result.wasSuccessful() and not result.skipped and not drift and within_budget,
              "limits": ["scripted human-priority injection, no live human session",
                         "actual detach/reattach and full reset, no independent object replacement",
                         "exact contracts use the stepped owner; independent advance uses realtime",
                         "test-driven requests are recorded; host cleanup is separate and its internal close RPC is not captured",
                         "Linux headless owner, no desktop or other-platform timing qualification"]}
    payload = gzip.compress(canonical_json(record).encode(), mtime=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
    index = {key: value for key, value in record.items() if key != "operations"}
    index["artifact"] = {"path": args.output.name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    index["owner_operations"] = len(operations)
    index["step_calls"] = sum(item["operation"] == "step" and item["status"] == "completed" for item in operations)
    index["actual_ticks"] = sum((item["response"] or {}).get("actual_ticks", 0)
                                for item in operations if item["operation"] == "step")
    with args.output.with_suffix(".index.json").open("x") as stream:
        json.dump(index, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: index[key] for key in ("passed", "tests_run", "skipped", "wall_seconds",
                                                "owner_operations", "step_calls", "actual_ticks", "artifact")}, indent=2))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
