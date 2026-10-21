"""External exit accounting with fake children/clocks; never starts a host."""

from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class RecoveryV4ExecutorTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("v4_executor", ROOT / "tools/execute_jolt_diagnostic.py")
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)
        self.declaration = {"schema": "nca-jolt-recovery-diagnostic/v4", "budget": {"maximum_wall_seconds": 900}}

    def args(self, folder):
        return SimpleNamespace(output=Path(folder) / "one-attempt", declaration=Path(folder) / "declaration.json",
            expected_revision="a" * 40, expected_declaration_sha256="b" * 64)

    def write_completion(self, args, origin, stream, defect=None):
        args.output.mkdir()
        primary = {"schema": "nca-jolt-arrival-supervisor/v1", "source_revision": args.expected_revision,
            "declaration_sha256": args.expected_declaration_sha256, "whole_process_started_monotonic": origin,
            "attempts": 1, "returncode": 0, "timeout": False, "completed": True, "worker_terminated": True,
            "owned_process_group_terminated": True, "final_output_verification": {"verified": True}, "wall_seconds": 3.0}
        if defect == "primary-source":
            primary["source_revision"] = "c" * 40
        primary_path = args.output.with_name(args.output.name + "-supervisor.json")
        primary_path.write_text(json.dumps(primary))
        final = {"schema": "nca-jolt-arrival-finalization/v1", "source_revision": args.expected_revision,
            "declaration_sha256": args.expected_declaration_sha256, "whole_process_started_monotonic": origin,
            "supervisor_sha256": hashlib.sha256(primary_path.read_bytes()).hexdigest(),
            "completed": True, "wall_seconds": 3.5}
        if defect == "origin":
            final["whole_process_started_monotonic"] = origin + 1
        if defect == "primary-hash":
            final["supervisor_sha256"] = "0" * 64
        final_path = args.output.with_name(args.output.name + "-finalization.json")
        final_path.write_text(json.dumps(final))
        terminal = {"completed": True, "returncode": 0, "timeout": False, "wall_seconds": 3.75,
            "finalization_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest()}
        if defect == "terminal-verdict":
            terminal["completed"] = False
        if defect == "marker-hash":
            terminal["finalization_sha256"] = "0" * 64
        if defect == "time-order":
            terminal["wall_seconds"] = 3.25
        stream.write((json.dumps(terminal) + "\n").encode())

    def run_fake(self, args, *, defect=None, exit_code=0, timeout=False,
                 spawn_seconds=2.0, execution_seconds=4.0, closed=True, proc_error=False):
        clock = [100.0]
        child = mock.Mock(pid=12345, returncode=None)
        child.poll.side_effect = lambda: child.returncode
        forwarded = ["--output", str(args.output), "--host", "literal $() host"]

        def spawn(command, **kwargs):
            self.assertEqual(command[2:-2], forwarded)
            self.assertEqual(command[-2:], ["--whole-process-started", "100.0"])
            self.assertTrue(kwargs["start_new_session"])
            self.assertNotIn("shell", kwargs)
            clock[0] += spawn_seconds

            def wait(*, timeout):
                self.assertEqual(timeout, 900.0 - spawn_seconds)
                if timeout_requested:
                    clock[0] = 1000.0
                    raise subprocess.TimeoutExpired("fake-supervisor", timeout)
                clock[0] = 100.0 + execution_seconds
                self.write_completion(args, 100.0, kwargs["stdout"], defect)
                child.returncode = exit_code
                return exit_code

            child.wait.side_effect = wait
            return child

        def terminate(process, owner_start):
            self.assertIs(process, child)
            self.assertEqual(owner_start, 17)
            child.returncode = -9
            clock[0] += 2.0
            return closed

        def publish(path, value):
            # Publication is deliberately slower than the measured child exit.
            clock[0] += 37.0
            with path.open("x") as target:
                json.dump(value, target)

        timeout_requested = timeout
        with ExitStack() as stack:
            stack.enter_context(mock.patch("nca.evaluation.jolt_arrival.read_declaration", return_value=self.declaration))
            stack.enter_context(mock.patch("nca.evaluation.durable_publication.publish_json_v2", side_effect=publish))
            stack.enter_context(mock.patch.object(self.cli, "prepare_process_tree", return_value=17))
            stack.enter_context(mock.patch.object(self.cli, "owned_processes",
                side_effect=PermissionError if proc_error else None, return_value={}))
            stack.enter_context(mock.patch.object(self.cli, "direct_children", return_value=()))
            spawned = stack.enter_context(mock.patch.object(self.cli.subprocess, "Popen", side_effect=spawn))
            cleanup = stack.enter_context(mock.patch.object(self.cli, "terminate_owned_tree", side_effect=terminate))
            stack.enter_context(mock.patch.object(self.cli.time, "monotonic", side_effect=lambda: clock[0]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            result = self.cli.execute(args, forwarded)
        receipt = json.loads(args.output.with_name(args.output.name + "-executor.json").read_bytes())
        return result, receipt, spawned, cleanup, child

    def test_spawn_and_final_exit_are_measured_before_offline_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(folder)
            result, receipt, spawn, cleanup, child = self.run_fake(args)
            self.assertEqual(result, 0)
            self.assertTrue(receipt["completed"])
            self.assertEqual(receipt["schema"], "nca-jolt-arrival-executor/v1")
            self.assertEqual(receipt["whole_process_started_monotonic"], 100.0)
            self.assertEqual(receipt["process_exited_monotonic"], 104.0)
            self.assertEqual(receipt["wall_seconds"], 4.0)
            self.assertEqual(receipt["maximum_wall_seconds"], 900)
            self.assertEqual(receipt["attempts"], 1)
            self.assertEqual(receipt["executor_source_sha256"],
                hashlib.sha256((ROOT / "tools/execute_jolt_diagnostic.py").read_bytes()).hexdigest())
            self.assertEqual(set(receipt["streams"]), {"one-attempt-executor-stdout.log", "one-attempt-executor-stderr.log"})
            for name, identity in receipt["streams"].items():
                payload = (Path(folder) / name).read_bytes()
                self.assertEqual(identity, {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
            self.assertIn("post-execution offline accounting", receipt["receipt_publication"])
            spawn.assert_called_once()
            child.wait.assert_called_once_with(timeout=898.0)
            cleanup.assert_not_called()

    def test_failed_original_verdict_or_binding_is_never_repaired(self):
        for defect in ("primary-source", "origin", "primary-hash", "terminal-verdict", "marker-hash", "time-order"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                args = self.args(folder)
                result, receipt, spawn, cleanup, _ = self.run_fake(args, defect=defect)
                self.assertEqual(result, 2)
                self.assertFalse(receipt["completed"])
                self.assertEqual(receipt["error_type"], "ValueError")
                self.assertTrue(args.output.with_name(args.output.name + "-finalization.json").is_file())
                spawn.assert_called_once()
                cleanup.assert_not_called()

    def test_child_nonzero_or_observed_exit_over_900_fails(self):
        for options in ({"exit_code": 2}, {"execution_seconds": 901.0}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as folder:
                result, receipt, spawn, _, _ = self.run_fake(self.args(folder), **options)
                self.assertEqual(result, 2)
                self.assertFalse(receipt["completed"])
                self.assertEqual(receipt["timeout"], "execution_seconds" in options)
                spawn.assert_called_once()

    def test_timeout_revokes_tree_once_and_preserves_cost_without_retry(self):
        for closed in (True, False):
            with self.subTest(closed=closed), tempfile.TemporaryDirectory() as folder:
                args = self.args(folder)
                result, receipt, spawn, cleanup, _ = self.run_fake(args, timeout=True, closed=closed)
                self.assertEqual(result, 2)
                self.assertTrue(receipt["timeout"])
                self.assertFalse(receipt["completed"])
                self.assertEqual(receipt["wall_seconds"], 902.0)
                self.assertEqual(receipt["returncode"], -9)
                self.assertEqual(receipt["owned_process_tree_terminated"], closed)
                cleanup.assert_called_once()
                spawn.assert_called_once()
                with mock.patch.object(self.cli.subprocess, "Popen") as retry:
                    with self.assertRaisesRegex(ValueError, "no retry"):
                        self.cli.execute(args, [])
                    retry.assert_not_called()

    def test_spawn_overhead_exhaustion_does_not_grant_a_fresh_wait(self):
        with tempfile.TemporaryDirectory() as folder:
            result, receipt, spawn, cleanup, child = self.run_fake(self.args(folder), spawn_seconds=901)
            self.assertEqual(result, 2)
            self.assertTrue(receipt["timeout"])
            child.wait.assert_not_called()
            spawn.assert_called_once()
            cleanup.assert_called_once()

    def test_missing_proc_visibility_cannot_claim_tree_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            result, receipt, _, _, _ = self.run_fake(self.args(folder), proc_error=True)
            self.assertEqual(result, 2)
            self.assertFalse(receipt["owned_process_tree_terminated"])
            self.assertEqual(receipt["cleanup_error_type"], "PermissionError")

    def test_any_existing_output_or_sibling_blocks_before_launch(self):
        for suffix in ("", "-unexpected", "-executor.json", "-finalization.json", "-executor-stdout.log"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as folder:
                args = self.args(folder)
                previous = args.output.with_name(args.output.name + suffix)
                previous.write_bytes(b"original bytes")
                with mock.patch.object(self.cli.subprocess, "Popen") as spawn:
                    with self.assertRaisesRegex(ValueError, "already exists"):
                        self.cli.execute(args, [])
                    spawn.assert_not_called()
                self.assertEqual(previous.read_bytes(), b"original bytes")

    def test_unsupported_platform_fails_before_creating_streams_or_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(folder)
            with mock.patch("nca.evaluation.jolt_arrival.read_declaration", return_value=self.declaration), \
                    mock.patch.object(self.cli.sys, "platform", "unsupported"), \
                    mock.patch.object(self.cli.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "Linux"):
                    self.cli.execute(args, [])
                spawn.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_cli_preserves_unknown_tokens_and_rejects_private_mode_abbreviations(self):
        argv = ["--output", "unused", "--expected-revision", "a" * 40, "--declaration", "unused.json",
            "--expected-declaration-sha256", "b" * 64, "--host", "literal $(no shell)", "--godot", "space in path"]
        with mock.patch.object(self.cli, "execute", return_value=0) as execute:
            self.assertEqual(self.cli.main(argv), 0)
            self.assertEqual(execute.call_args.args[1], argv)
        for flag in ("--worker", "--work", "--verify-only", "--veri", "--whole-process-started=100", "--whole=100"):
            with self.subTest(flag=flag), mock.patch.object(self.cli, "execute") as execute, \
                    mock.patch.object(self.cli.sys, "stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.cli.main([*argv, flag])
                execute.assert_not_called()

    def test_tree_closure_includes_reparented_descendants_from_other_sessions(self):
        clock = [0.0]
        parent = {"parent": 10, "start": 20, "state": "S", "session": 20}
        worker = {"parent": 20, "start": 21, "state": "S", "session": 21}
        adopted = {"parent": 10, "start": 21, "state": "Z", "session": 21}
        child = mock.Mock(pid=20)
        child.poll.return_value = -9
        with mock.patch.object(self.cli.os, "getpid", return_value=10), \
                mock.patch.object(self.cli, "owned_processes", side_effect=[{20: parent, 21: worker}, {21: adopted}, {21: adopted}, {}]), \
                mock.patch.object(self.cli, "direct_children", return_value=()), \
                mock.patch.object(self.cli, "_kill_identity") as kill, \
                mock.patch.object(self.cli.os, "waitpid", return_value=(21, 0)) as reap, \
                mock.patch.object(self.cli.os, "WNOHANG", 1, create=True), \
                mock.patch.object(self.cli.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(self.cli.time, "sleep", side_effect=lambda value: clock.__setitem__(0, clock[0] + value)):
            self.assertTrue(self.cli.terminate_owned_tree(child, 1))
        self.assertEqual([call.args[0] for call in kill.call_args_list], [20, 21, 21])
        reap.assert_called_once_with(21, 1)

    def test_empty_proc_scan_cannot_hide_a_newly_reparented_kernel_child(self):
        clock = [0.0]
        child = mock.Mock(pid=20)
        child.poll.return_value = -9
        with mock.patch.object(self.cli, "owned_processes", return_value={}), \
                mock.patch.object(self.cli, "direct_children", return_value=(21,)), \
                mock.patch.object(self.cli.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(self.cli.time, "sleep", side_effect=lambda value: clock.__setitem__(0, clock[0] + value)):
            self.assertFalse(self.cli.terminate_owned_tree(child, 1, grace_seconds=.02))

    def test_non_reaping_kernel_query_distinguishes_absence_from_live_children(self):
        for result, expected in ((None, True), (SimpleNamespace(si_pid=21), True), (ChildProcessError(), False)):
            with self.subTest(result=result), ExitStack() as stack:
                for name, value in (("P_ALL", 0), ("WEXITED", 4), ("WNOHANG", 1), ("WNOWAIT", 16)):
                    stack.enter_context(mock.patch.object(self.cli.os, name, value, create=True))
                wait = stack.enter_context(mock.patch.object(self.cli.os, "waitid", create=True,
                    side_effect=result if isinstance(result, Exception) else None, return_value=result))
                self.assertIs(self.cli.direct_children(), expected)
                wait.assert_called_once_with(0, 0, 21)

    @unittest.skipUnless(sys.platform.startswith("linux") and hasattr(__import__("os"), "pidfd_open"),
        "Linux subreaper integration; all deadline and receipt tests use portable fake children")
    def test_isolated_python_tree_closes_a_separately_sessioned_descendant(self):
        # Only this disposable harness enables subreaping; its descendants are
        # harmless Python sleepers with their own finite ten-second lifetimes.
        script = f"""
import importlib.util, os, subprocess, sys
spec = importlib.util.spec_from_file_location('executor', {str(ROOT / 'tools/execute_jolt_diagnostic.py')!r})
executor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(executor)
owner = executor.prepare_process_tree()
program = "import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'], start_new_session=True); print(p.pid, flush=True); time.sleep(10)"
process = subprocess.Popen([sys.executable, '-c', program], stdout=subprocess.PIPE, text=True, start_new_session=True)
try:
    descendant = int(process.stdout.readline())
    assert os.getpgid(descendant) != os.getpgid(process.pid)
    assert executor.terminate_owned_tree(process, owner)
    assert process.poll() is not None
    assert not executor.direct_children()
    assert not executor.owned_processes(owner)
finally:
    executor.terminate_owned_tree(process, owner)
    process.stdout.close()
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=12)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pid_identity_change_cannot_signal_an_unrelated_replacement(self):
        with mock.patch.object(self.cli.os, "pidfd_open", create=True, return_value=99), \
                mock.patch.object(self.cli, "_process", return_value={"start": 2, "state": "S"}), \
                mock.patch.object(self.cli.signal, "pidfd_send_signal", create=True) as kill, \
                mock.patch.object(self.cli.os, "close") as close:
            self.cli._kill_identity(123, {"start": 1})
        kill.assert_not_called()
        close.assert_called_once_with(99)


if __name__ == "__main__":
    unittest.main()
