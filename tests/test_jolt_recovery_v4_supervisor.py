"""Whole-process v4 supervision with fake workers and no physical runtime."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class RecoveryV4SupervisorTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("v4_supervisor", ROOT / "tools/jolt_arrival.py")
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)
        self.declaration = {"schema": "nca-jolt-recovery-diagnostic/v4", "budget": {
            "maximum_wall_seconds": 900, "shutdown_reserve_seconds": 120,
            "independent_verification_reserve_seconds": 60,
            "supervisor_finalization_reserve_seconds": 6}}

    def args(self, folder):
        return SimpleNamespace(output=Path(folder) / "one-attempt",
            declaration=Path(folder) / "synthetic-declaration.json",
            expected_declaration_sha256="b" * 64, expected_revision="a" * 40,
            host=Path("unused-synthetic-host"), godot=Path("unused-synthetic-godot"))

    def test_startup_worker_sealing_and_verifier_share_original_900_seconds(self):
        clock = [100.0]
        worker = mock.Mock(pid=123456, returncode=0)
        worker.poll.return_value = 0

        def read_declaration(*args, **kwargs):
            clock[0] += 10.0  # Source/declaration setup cannot renew the clock.
            return self.declaration

        def worker_wait(*, timeout):
            self.assertEqual(timeout, 810.0)  # Twenty seconds of Python startup plus ten of setup.
            clock[0] = 920.0  # Worker has used its entire shared 840-second allowance.
            return 0

        def verify(command, **kwargs):
            self.assertIn("--verify-only", command)
            self.assertEqual(kwargs["timeout"], 54.0)
            kwargs["stdout"].write(b'{"verified":true}\n')
            clock[0] = 925.0
            return SimpleNamespace(returncode=0)

        worker.wait.side_effect = worker_wait
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(folder)
            args.whole_process_started = 80.0  # The external executor started before this process.
            with mock.patch("nca.evaluation.jolt_arrival.read_declaration", side_effect=read_declaration), \
                    mock.patch.object(self.cli.time, "monotonic", side_effect=lambda: clock[0]), \
                    mock.patch.object(self.cli.subprocess, "Popen", return_value=worker) as spawn, \
                    mock.patch.object(self.cli.subprocess, "run", side_effect=verify) as verifier, \
                    mock.patch.object(self.cli, "process_group_exists", return_value=False):
                self.assertEqual(self.cli.supervise(args), 0)
            command = spawn.call_args.args[0]
            self.assertEqual(command[command.index("--whole-process-started") + 1], "80.0")
            self.assertEqual(spawn.call_count, 1)
            self.assertEqual(verifier.call_count, 1)
            receipt = json.loads(args.output.with_name(args.output.name + "-supervisor.json").read_bytes())
            self.assertTrue(receipt["completed"])
            self.assertEqual(receipt["whole_process_started_monotonic"], 80.0)
            self.assertEqual(receipt["wall_seconds"], 845.0)
            self.assertEqual(receipt["cleanup_reserve_seconds"], 120)
            self.assertEqual(receipt["active_work_maximum_seconds"], 780)
            self.assertEqual(receipt["worker_maximum_seconds"], 840)
            terminal = json.loads(args.output.with_name(args.output.name + "-finalization.json").read_bytes())
            self.assertTrue(terminal["completed"])
            self.assertEqual(terminal["whole_process_started_monotonic"], 80.0)
            self.assertEqual(terminal["wall_seconds"], 845.0)
            self.assertFalse(list(Path(folder).glob("*.pending")))

    def test_worker_timeout_consumes_attempt_and_does_not_start_verifier_or_retry(self):
        worker = mock.Mock(pid=123456, returncode=-15)
        worker.poll.return_value = -15
        worker.wait.side_effect = subprocess.TimeoutExpired("synthetic-worker", 840)
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(folder)
            with mock.patch("nca.evaluation.jolt_arrival.read_declaration", return_value=self.declaration), \
                    mock.patch.object(self.cli.subprocess, "Popen", return_value=worker) as spawn, \
                    mock.patch.object(self.cli.subprocess, "run") as verifier, \
                    mock.patch.object(self.cli, "terminate_owned_tree", return_value=True) as cleanup, \
                    mock.patch.object(self.cli, "process_group_exists", return_value=False):
                self.assertEqual(self.cli.supervise(args), 2)
                self.assertEqual(spawn.call_count, 1)
                verifier.assert_not_called()
                cleanup.assert_called_once_with(worker)
                with self.assertRaisesRegex(ValueError, "automatic retry"):
                    self.cli.supervise(args)
                self.assertEqual(spawn.call_count, 1)
            receipt = json.loads(args.output.with_name(args.output.name + "-supervisor.json").read_bytes())
            self.assertFalse(receipt["completed"])
            self.assertTrue(receipt["timeout"])
            self.assertEqual(receipt["attempts"], 1)
            self.assertFalse(receipt["final_output_verification"]["verified"])

    def test_verifier_failure_preserves_failed_original_verdict(self):
        worker = mock.Mock(pid=123456, returncode=0)
        worker.poll.return_value = 0
        worker.wait.return_value = 0
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(folder)
            with mock.patch("nca.evaluation.jolt_arrival.read_declaration", return_value=self.declaration), \
                    mock.patch.object(self.cli.subprocess, "Popen", return_value=worker), \
                    mock.patch.object(self.cli.subprocess, "run", return_value=SimpleNamespace(returncode=2)), \
                    mock.patch.object(self.cli, "process_group_exists", return_value=False):
                self.assertEqual(self.cli.supervise(args), 2)
            receipt = json.loads(args.output.with_name(args.output.name + "-supervisor.json").read_bytes())
            self.assertFalse(receipt["completed"])
            self.assertEqual(receipt["error_type"], "ValueError")
            self.assertTrue(receipt["worker_terminated"])
            self.assertTrue(receipt["owned_process_group_terminated"])

    def test_slow_primary_receipt_or_terminal_marker_cannot_return_success(self):
        from nca.evaluation.durable_publication import publish_json_v2
        for slow_suffix in ("-supervisor.json", "-finalization.json"):
            with self.subTest(slow_suffix=slow_suffix), tempfile.TemporaryDirectory() as folder:
                clock = [100.0]
                worker = mock.Mock(pid=123456, returncode=0)
                worker.poll.return_value = 0
                worker.wait.return_value = 0
                args = self.args(folder)

                def verify(command, **kwargs):
                    kwargs["stdout"].write(b'{"verified":true}\n')
                    return SimpleNamespace(returncode=0)

                def publish(path, value):
                    result = publish_json_v2(path, value)
                    if path.name.endswith(slow_suffix):
                        clock[0] = 1001.0
                    return result

                with mock.patch("nca.evaluation.jolt_arrival.read_declaration", return_value=self.declaration), \
                        mock.patch("nca.evaluation.durable_publication.publish_json_v2", side_effect=publish), \
                        mock.patch.object(self.cli.time, "monotonic", side_effect=lambda: clock[0]), \
                        mock.patch.object(self.cli.subprocess, "Popen", return_value=worker), \
                        mock.patch.object(self.cli.subprocess, "run", side_effect=verify), \
                        mock.patch.object(self.cli, "process_group_exists", return_value=False):
                    self.assertEqual(self.cli.supervise(args), 2)
                receipt = json.loads(args.output.with_name(args.output.name + "-supervisor.json").read_bytes())
                terminal = json.loads(args.output.with_name(args.output.name + "-finalization.json").read_bytes())
                self.assertTrue(receipt["completed"])  # Preserve the original pre-publication receipt.
                if slow_suffix == "-supervisor.json":
                    self.assertFalse(terminal["completed"])
                    self.assertEqual(terminal["wall_seconds"], 901.0)


if __name__ == "__main__":
    unittest.main()
