"""Actual non-physical observer watchdog tests; never start Godot or training.

The original one-use learner invocation is replaced by disposable Python
children, and the independent lifecycle verdict is explicitly mocked.
© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from nca.contracts import ContractError
from nca.train.jolt_memory_watchdog_v3 import (
    ProcessTreeWatchdogV3, watchdog_policy_v3,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("observe_jolt_memory_v3", ROOT / "tools/observe_jolt_v3.py")
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


@unittest.skipUnless(sys.platform.startswith("linux"), "the actual observer requires Linux subreaping")
class ObservedMemoryWatchdogTests(unittest.TestCase):
    def fixture(self, root, *, policy=None):
        budget = json.loads((ROOT / "configs/learning/jolt-learn-003-v2.json").read_bytes())["budget"]
        document = {
            "budget": budget,
            # The deliberately synthetic resource record is *never admitted*.
            "resource_admission": {
                "schema": "nca-jolt-003-resource-admission/v2",
                "memory_watchdog": watchdog_policy_v3() if policy is None else policy,
            },
        }
        declaration = root / "declaration.json"
        declaration.write_text(json.dumps(document))
        return SimpleNamespace(
            output=root / "learn-jolt-003-unopened-synthetic",
            declaration=declaration, config=root / "unused",
            checkout=root / "unused", godot=root / "unused",
        )

    def invoke(self, args, script, *, guard_type=None):
        with redirect_stdout(io.StringIO()) as out, \
                mock.patch("nca.train.jolt_admission_v3.validate_resource_admission"), \
                mock.patch("nca.train.jolt_executor_v3.verify_lifecycle",
                           return_value={"completed": True, "verified": True}) as lifecycle:
            if guard_type is None:
                result = observer.observe(args, started=time.monotonic(),
                    command=[sys.executable, "-c", script])
            else:
                with mock.patch.object(observer, "ProcessTreeWatchdogV3", guard_type):
                    result = observer.observe(args, started=time.monotonic(),
                        command=[sys.executable, "-c", script])
        receipt = json.loads(args.output.with_name(args.output.name + "-observed-exit.json").read_bytes())
        watchdog = json.loads(args.output.with_name(args.output.name + "-memory-watchdog.json").read_bytes())
        return result, receipt, watchdog, lifecycle, json.loads(out.getvalue())

    def test_normal_cpu_child_retains_separate_memory_receipt_and_observed_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            code, observed, sample, _life, terminal = self.invoke(args, "print('synthetic child')")
            self.assertEqual(code, 0)
            self.assertTrue(terminal["completed"])
            self.assertTrue(observed["memory_watchdog_enforced"])
            self.assertFalse(observed["memory_watchdog_triggered"])
            self.assertEqual(sample["policy"]["soft_threshold_bytes"], 8 * 1024**3)
            self.assertGreaterEqual(sample["sample_count"], 2)
            self.assertGreater(sample["peak_sampled_process_tree_rss_bytes"], 0)
            self.assertTrue(sample["owned_tree_cleanup_confirmed"])
            self.assertFalse(sample["kernel_hard_limit_enforced"])
            self.assertEqual(observed["memory_watchdog"]["sha256"],
                __import__("hashlib").sha256(args.output.with_name(args.output.name + "-memory-watchdog.json").read_bytes()).hexdigest())

    def test_overlimit_retains_original_failure_and_kills_synthetic_child(self):
        class ForcedOverage(ProcessTreeWatchdogV3):
            def check(self, *, force=False):
                super().check(force=force)
                if self.sample_count >= 2:
                    self._fail("jolt_v3_memory_limit")
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            code, observed, sample, lifecycle, terminal = self.invoke(
                args, "import time; time.sleep(30)", guard_type=ForcedOverage)
            self.assertEqual(code, 2)
            self.assertFalse(terminal["completed"])
            self.assertTrue(observed["memory_watchdog_triggered"])
            self.assertTrue(sample["limit_reached"])
            self.assertTrue(sample["owned_tree_cleanup_confirmed"])
            self.assertTrue(observed["exit_observed"])
            self.assertFalse(any(record.get("completed") is True for record in observed["errors"]))
            self.assertTrue(any(err.get("code") == "jolt_v3_memory_limit" for err in observed["errors"]))
            self.assertEqual(lifecycle.call_count, 1)
            self.assertTrue(args.output.with_name(args.output.name + "-outer-stdout.log").exists())

    def test_modified_or_missing_watchdog_policy_refuses_before_spawn(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.fixture(root, policy={**watchdog_policy_v3(), "soft_threshold_bytes": 16 * 1024**3})
            with mock.patch.object(observer.subprocess, "Popen") as spawn, self.assertRaises(ContractError):
                observer.observe(args, started=time.monotonic(), command=["unused"])
            spawn.assert_not_called()
            self.assertFalse(args.output.with_name(args.output.name + "-outer-stdout.log").exists())


if __name__ == "__main__":
    unittest.main()
