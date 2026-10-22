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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("observe_jolt_v3", ROOT / "tools/observe_jolt_v3.py")
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


@unittest.skipUnless(sys.platform.startswith("linux"), "prospective executor requires Linux subreaping")
class ObserverV3Tests(unittest.TestCase):
    def fixture(self, root):
        declaration = root / "declaration.json"
        declaration.write_text("{}\n")
        return SimpleNamespace(output=root / "learn-jolt-003-synthetic", declaration=declaration,
                               config=root / "unused", checkout=root / "unused", godot=root / "unused")

    def invoke(self, root, script, *, verdict=None):
        args = self.fixture(root)
        terminal = io.StringIO()
        with redirect_stdout(terminal), mock.patch("nca.train.jolt_executor_v3.verify_lifecycle",
                return_value=verdict or {"completed": False, "verified": False}) as check:
            code = observer.observe(args, started=time.monotonic(),
                                    command=[sys.executable, "-c", script])
        receipt = json.loads(args.output.with_name(args.output.name + "-observed-exit.json").read_bytes())
        return code, args, receipt, json.loads(terminal.getvalue()), check

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
