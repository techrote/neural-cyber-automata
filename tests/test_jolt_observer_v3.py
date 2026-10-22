"""Actual subprocess exit/stream conformance without an experiment invocation."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from nca.contracts import ContractError
from nca.train.jolt_execution_v3 import LEGACY_TIMING_SCALARS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("observe_jolt_v3", ROOT / "tools/observe_jolt_v3.py")
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


@unittest.skipUnless(sys.platform.startswith("linux"), "prospective executor requires Linux subreaping")
class ObserverV3Tests(unittest.TestCase):
    def fixture(self, root, *, unlimited=False):
        declaration = root / "declaration.json"
        budget = json.loads((ROOT / "docs/evidence/jolt-learn-003-allocation-plan-20261022.json").read_bytes())["proposed_budget"]
        if unlimited:
            budget.update(dict.fromkeys(LEGACY_TIMING_SCALARS))
        declaration.write_text(json.dumps({"budget": budget}) + "\n")
        return SimpleNamespace(output=root / "learn-jolt-003-synthetic", declaration=declaration,
                               config=root / "unused", checkout=root / "unused", godot=root / "unused")

    def invoke(self, root, script, *, verdict=None, unlimited=False, elapsed_before=0.):
        args = self.fixture(root, unlimited=unlimited)
        terminal = io.StringIO()
        with redirect_stdout(terminal), mock.patch("nca.train.jolt_executor_v3.verify_lifecycle",
                return_value=verdict or {"completed": False, "verified": False}) as check:
            code = observer.observe(args, started=time.monotonic() - elapsed_before,
                                    command=[sys.executable, "-c", script])
        receipt = json.loads(args.output.with_name(args.output.name + "-observed-exit.json").read_bytes())
        return code, args, receipt, json.loads(terminal.getvalue()), check

    def test_unlimited_observer_retains_actual_exit_after_historical_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            code, args, receipt, terminal, check = self.invoke(Path(temp), "print('observed')",
                verdict={"completed": True, "verified": True}, unlimited=True, elapsed_before=100_000.)
            self.assertEqual(code, 0)
            self.assertTrue(terminal["completed"])
            self.assertIsNone(receipt["maximum_wall_seconds"])
            self.assertEqual(receipt["observed_exit_code"], 0)
            self.assertGreaterEqual(receipt["observed_exit_elapsed_seconds"], 100_000.)
            self.assertTrue(receipt["cleanup_confirmed"])
            self.assertEqual(check.call_args.kwargs["expected_budget"],
                             json.loads(args.declaration.read_bytes())["budget"])
            self.assertEqual(args.output.with_name(args.output.name + "-outer-stdout.log").read_bytes(), b"observed\n")

    def test_historical_observer_limit_and_mixed_budget_still_refuse_before_spawn(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(observer.subprocess, "Popen") as spawn:
            code, _, receipt, terminal, _ = self.invoke(Path(temp), "unused", elapsed_before=7201.)
            self.assertEqual(code, 2)
            self.assertFalse(terminal["completed"])
            self.assertEqual(receipt["maximum_wall_seconds"], 7200)
            self.assertFalse(receipt["exit_observed"])
            spawn.assert_not_called()
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp), unlimited=True)
            declaration = json.loads(args.declaration.read_bytes())
            declaration["budget"]["maximum_owner_transport_seconds"] = 15
            args.declaration.write_text(json.dumps(declaration))
            with mock.patch.object(observer.subprocess, "Popen") as spawn, self.assertRaises(ContractError):
                observer.observe(args, started=time.monotonic(), command=["unused"])
            spawn.assert_not_called()
            self.assertFalse(args.output.with_name(args.output.name + "-outer-stdout.log").exists())

    def test_original_nonzero_exit_and_exact_streams_survive_a_positive_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            code, args, receipt, terminal, check = self.invoke(Path(temp),
                "import sys; print('original stdout'); print('original stderr',file=sys.stderr); sys.exit(13)",
                verdict={"completed": True, "verified": True})
            self.assertEqual(code, 2)
            self.assertEqual(receipt["observed_exit_code"], 13)
            self.assertTrue(receipt["exit_observed"])
            self.assertTrue(receipt["cleanup_confirmed"])
            self.assertFalse(terminal["completed"])
            self.assertGreater(receipt["observed_exit_elapsed_seconds"], 0.)
            self.assertEqual(args.output.with_name(args.output.name + "-outer-stdout.log").read_bytes(), b"original stdout\n")
            self.assertEqual(args.output.with_name(args.output.name + "-outer-stderr.log").read_bytes(), b"original stderr\n")
            self.assertEqual(check.call_args.kwargs["observed_exit_code"], 13)

    def test_failed_spawn_has_no_invented_exit_and_no_second_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            with mock.patch.object(observer.subprocess, "Popen", side_effect=OSError("synthetic spawn")) as spawn:
                with redirect_stdout(io.StringIO()):
                    code = observer.observe(args, started=time.monotonic(), command=["unused"])
            self.assertEqual(spawn.call_count, 1)
            self.assertEqual(code, 2)
            record = json.loads(args.output.with_name(args.output.name + "-observed-exit.json").read_bytes())
            self.assertFalse(record["exit_observed"])
            self.assertIsNone(record["observed_exit_code"])
            self.assertIsNone(record["observed_exit_elapsed_seconds"])
            with mock.patch.object(observer.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(ValueError, "retry is forbidden"):
                    observer.observe(args, started=time.monotonic(), command=["unused"])
                spawn.assert_not_called()

    def test_excess_stdout_is_bounded_and_never_counts_as_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            code, args, receipt, terminal, _ = self.invoke(Path(temp),
                "import sys; sys.stdout.write('x'*100000); sys.stdout.flush()")
            self.assertEqual(code, 2)
            self.assertFalse(terminal["completed"])
            self.assertLessEqual(receipt["stdout"]["bytes"], 65536)
            self.assertTrue(receipt["cleanup_confirmed"])
            self.assertTrue(receipt["errors"])


if __name__ == "__main__":
    unittest.main()
