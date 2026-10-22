"""Synthetic original-clock v3 lifecycle and independently observed exit tests."""
from contextlib import ExitStack, redirect_stdout
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

from nca.contracts import ContractError, sha256_json
from nca.train import jolt_executor_v3 as executor
from nca.train.jolt_execution_v3 import PHASE_OFFSETS

ROOT = Path(__file__).resolve().parents[1]


class ExecutorV3Tests(unittest.TestCase):
    def fixture(self, root):
        budget = json.loads((ROOT / "docs/evidence/jolt-learn-003-allocation-plan-20261022.json").read_bytes())["proposed_budget"]
        allocation = "learn-jolt-003-synthetic"
        config = {"allocation_id": allocation, "budget": budget}
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config))
        declaration = {"schema": executor.DECLARATION_SCHEMA, "attempts": 1,
            "automatic_retry": False, "automatic_extension": False,
            "allocation_id": allocation,
            "authorization_ref": "https://github.com/techrote/neural-cyber-automata/issues/62#issuecomment-1",
            "config_sha256": sha256_json(config), "budget": budget,
            "output_identity": {"basename": allocation, "lifecycle_basename": allocation + "-lifecycle", "reuse_forbidden": True}}
        declaration_path = root / "declaration.json"
        declaration_path.write_text(json.dumps(declaration))
        return SimpleNamespace(output=root / allocation, declaration=declaration_path,
                               config=config_path, checkout=root / "unused", godot=root / "unused")

    def invoke(self, root, *, origin=0., clock_value=1., worker_code=0, verifier_code=0,
               advance_after=None, advance_to=None, publication_failure=None,
               worker_wait_error=None, cleanup_error=None, stdout_error=False,
               mutation=None):
        args = self.fixture(root)
        if mutation:
            declaration = json.loads(args.declaration.read_bytes())
            mutation(declaration)
            args.declaration.write_text(json.dumps(declaration))
        current = [clock_value]
        commands, waits, children = [], [], []
        helper = SimpleNamespace(prepare_process_tree=lambda: 123,
                                 owned_processes=lambda identity: {})
        def cleanup(process, identity, **kwargs):
            if cleanup_error:
                raise cleanup_error
            if process is not None and process.poll() is None:
                process.kill()
            return True
        helper.terminate_owned_tree = mock.Mock(side_effect=cleanup)
        real_publish = executor.publish_json_v3
        def publish(path, value):
            if path.name == publication_failure:
                raise OSError("synthetic publication failure")
            answer = real_publish(path, value)
            if path.name == advance_after:
                current[0] = advance_to
            return answer
        def spawn(command, **kwargs):
            commands.append(command)
            phase = "worker" if command[2] == "worker-v3" else "verification"
            child = mock.Mock(returncode=None)
            child.poll.side_effect = lambda: child.returncode
            def kill():
                child.returncode = -9
            child.kill.side_effect = kill
            def wait(timeout):
                waits.append((phase, timeout))
                if phase == "worker" and worker_wait_error and child.returncode is None:
                    raise worker_wait_error
                child.returncode = worker_code if phase == "worker" else verifier_code
                if phase == "verification":
                    child.synthetic_stdout.write(json.dumps({"inventory_verified": True, "readback_verified": True,
                        "readback_complete": True, "costs_verified": True, "status": "completed", "qualified": False}).encode())
                return child.returncode
            child.wait.side_effect = wait
            children.append(child)
            return child
        def bounded_wait(process, stdout, stderr, *, deadline, monotonic):
            process.synthetic_stdout = stdout
            return process.wait(timeout=deadline - monotonic())
        stdout = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {"execute_jolt_diagnostic": helper}))
            stack.enter_context(mock.patch.object(executor, "time", SimpleNamespace(monotonic=lambda: current[0])))
            stack.enter_context(mock.patch.object(executor, "publish_json_v3", side_effect=publish))
            stack.enter_context(mock.patch.object(executor.subprocess, "Popen", side_effect=spawn))
            stack.enter_context(mock.patch.object(executor, "wait_bounded_child", side_effect=bounded_wait))
            stack.enter_context(mock.patch("nca.train.jolt_admission_v3.validate_resource_admission", return_value=None))
            stack.enter_context(redirect_stdout(stdout))
            if stdout_error:
                stack.enter_context(mock.patch("builtins.print", side_effect=BrokenPipeError()))
            code = executor.execute(args, whole_started=origin, script=Path("synthetic.py"))
        return code, stdout.getvalue().encode(), args, commands, waits, helper

    def lifecycle(self, args):
        return args.output.with_name(args.output.name + "-lifecycle")

    def test_complete_negative_comparison_retains_original_clock_and_observed_exit(self):
        for origin in (0., .00001, 10.):
            with self.subTest(origin=origin), tempfile.TemporaryDirectory() as folder:
                code, stdout, args, commands, waits, _ = self.invoke(Path(folder), origin=origin, clock_value=origin + 1.)
                self.assertEqual(code, 0)
                self.assertEqual([c[2] for c in commands], ["worker-v3", "inventory-v3"])
                self.assertEqual(waits[0][1], 6887.)
                self.assertEqual(waits[1][1], 7127.)
                terminal = json.loads(stdout)
                self.assertTrue(terminal["completed"])
                result = executor.verify_lifecycle(self.lifecycle(args), terminal_stdout=stdout,
                    observed_exit_code=0, observed_exit_elapsed_seconds=2.,
                    expected_declaration=executor.file_identity(args.declaration))
                self.assertTrue(result["verified"])
                self.assertFalse(result["worker_inventory"]["qualified"])
                self.assertEqual(result["whole_process_started_monotonic"], origin)
                self.assertEqual(result["observed_exit_elapsed_seconds"], 2.)

    def test_missing_or_late_outer_exit_never_upgrades_terminal_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            _, stdout, args, _, _, _ = self.invoke(Path(folder))
            with self.assertRaises(ContractError):
                executor.verify_lifecycle(self.lifecycle(args), terminal_stdout=stdout, observed_exit_code=0)
            verdict = executor.verify_lifecycle(self.lifecycle(args), terminal_stdout=stdout,
                observed_exit_code=0, observed_exit_elapsed_seconds=7201.)
            self.assertFalse(verdict["verified"])
            with self.assertRaises(ContractError):
                executor.verify_lifecycle(self.lifecycle(args), terminal_stdout=stdout,
                    observed_exit_code=0, observed_exit_elapsed_seconds=.5)

    def test_child_exit_cleanup_publication_and_final_phase_boundaries_fail_closed(self):
        for artifact, when in (("worker-started.json", 6888.), ("worker-cleanup.json", 6900.),
                               ("verification-started.json", 7128.), ("verification-exit.json", 7140.),
                               ("terminal.json", 7201.)):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as folder:
                code, stdout, args, _, _, _ = self.invoke(Path(folder), advance_after=artifact, advance_to=when)
                self.assertEqual(code, 2)
                self.assertFalse(json.loads(stdout)["completed"])
                self.assertTrue((self.lifecycle(args) / "executor.json").is_file())

    def test_source_output_and_budget_mismatch_refuse_before_worker(self):
        changes = [lambda d: d.update(schema="nca-learn-jolt-002-execution/v2"),
                   lambda d: d.update(config_sha256="0" * 64),
                   lambda d: d["output_identity"].update(basename="another-allocation"),
                   lambda d: d["budget"].update(maximum_wall_seconds=7201),
                   lambda d: d.update(attempts=2)]
        for mutation in changes:
            with tempfile.TemporaryDirectory() as folder:
                code, stdout, _, commands, _, _ = self.invoke(Path(folder), mutation=mutation)
                self.assertEqual(code, 2)
                self.assertFalse(commands)
                self.assertFalse(json.loads(stdout)["completed"])

    def test_timeout_and_cleanup_failure_do_not_launch_another_worker_or_verifier(self):
        for kwargs in ({"worker_wait_error": subprocess.TimeoutExpired("synthetic", 1)},
                       {"cleanup_error": OSError("synthetic cleanup")}, {"worker_code": 7}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as folder:
                code, stdout, _, commands, _, helper = self.invoke(Path(folder), **kwargs)
                self.assertEqual(code, 2)
                self.assertEqual(len(commands), 1)
                self.assertFalse(json.loads(stdout)["completed"])
                self.assertGreaterEqual(helper.terminate_owned_tree.call_count, 1)

    def test_publication_and_broken_stdout_failure_preserve_original_negative_receipts(self):
        for name in ("invocation.json", "worker-exit.json", "verification.json", "finalization.json", "terminal.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                code, stdout, args, _, _, _ = self.invoke(Path(folder), publication_failure=name)
                self.assertEqual(code, 2)
                self.assertFalse(json.loads(stdout)["completed"])
                self.assertTrue((self.lifecycle(args) / "executor.json").is_file())
        with tempfile.TemporaryDirectory() as folder:
            code, stdout, args, _, _, _ = self.invoke(Path(folder), stdout_error=True)
            self.assertEqual(code, 2)
            self.assertEqual(stdout, b"")
            self.assertTrue((self.lifecycle(args) / "terminal-output-error.json").is_file())

    def test_exact_lifecycle_inventory_and_raw_stream_identity_are_required(self):
        for kind in ("extra", "stream", "stage"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                _, stdout, args, _, _, _ = self.invoke(Path(folder))
                root = self.lifecycle(args)
                if kind == "extra":
                    (root / "extra.json").write_text("{}")
                else:
                    path = root / ("worker-stdout.log" if kind == "stream" else "worker-exit.json")
                    path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaises(ContractError):
                    executor.verify_lifecycle(root, terminal_stdout=stdout, observed_exit_code=0,
                                              observed_exit_elapsed_seconds=2.)


if __name__ == "__main__":
    unittest.main()
