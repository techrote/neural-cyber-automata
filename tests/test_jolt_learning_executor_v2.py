"""Synthetic lifecycle failures, including real disposable Python children.

No physical host, checkpoint, neural model or fitting procedure is constructed.
"""

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

from nca.contracts import ContractError
from nca.train import jolt_executor
from nca.train.jolt_execution_contract import validate_v2_allocation_identity


class LearningExecutorV2Tests(unittest.TestCase):
    def fixture(self, root):
        declaration = root / "declaration.json"
        declaration.write_text(json.dumps({"schema": jolt_executor.DECLARATION_SCHEMA, "attempts": 1,
            "automatic_retry": False, "automatic_extension": False,
            "allocation_id": "synthetic-executor-fixture",
            "authorization_ref": "https://github.com/techrote/neural-cyber-automata/issues/62#issuecomment-1",
            "budget": {"maximum_wall_seconds": 1800}}))
        return SimpleNamespace(output=root / "output", declaration=declaration,
            config=root / "config.json", checkout=root / "unused-host", godot=root / "unused-godot")

    def invoke(self, root, *, ownership_error=None, worker_error=None, worker_returncode=0,
               verifier_error=None, verifier_returncode=0, verified=True, complete=True,
               publication_failure=None, identity_failure=None, cleanup_error=None,
               cleanup_confirmed=True, pending=False, clock_at=101., advance_after=None,
               advance_to=1901., stdout_error=False, script=None, whole_started=100.):
        args = self.fixture(root)
        clock = [clock_at]
        helper = SimpleNamespace(prepare_process_tree=mock.Mock(return_value=987, side_effect=ownership_error),
            terminate_owned_tree=mock.Mock(return_value=cleanup_confirmed, side_effect=cleanup_error),
            owned_processes=mock.Mock(return_value={1: {}} if pending else {}))
        real_publish = jolt_executor.publish_json_v2
        real_identity = jolt_executor.file_identity
        children, commands = [], []

        def publish(path, value):
            if path.name == publication_failure:
                raise OSError("synthetic publication path contains /private/not-for-publication")
            result = real_publish(path, value)
            if path.name == advance_after:
                clock[0] = advance_to
            return result

        def identity(path):
            if path.name == identity_failure:
                raise OSError("synthetic hash failure")
            return real_identity(path)

        def spawn(command, **kwargs):
            commands.append(command)
            phase = "worker" if command[2] == "worker-v2" else "verification"
            child = mock.Mock(returncode=None)
            child.poll.side_effect = lambda: child.returncode
            error = worker_error if phase == "worker" else verifier_error
            code = worker_returncode if phase == "worker" else verifier_returncode

            def wait(timeout):
                if error is not None:
                    raise error
                child.returncode = code
                if phase == "verification":
                    kwargs["stdout"].write(json.dumps({"inventory_verified": True,
                        "readback_verified": verified, "readback_complete": complete,
                        "costs_verified": True, "status": "completed", "qualified": False}).encode())
                return code

            def kill():
                child.returncode = -9

            child.wait.side_effect = wait
            child.kill.side_effect = kill
            children.append(child)
            return child

        def cleanup(process, identity, **kwargs):
            if cleanup_error is not None:
                raise cleanup_error
            self.assertEqual(identity, 987)
            if process is not None and process.poll() is None:
                process.kill()
            return cleanup_confirmed

        helper.terminate_owned_tree.side_effect = cleanup
        output = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {"execute_jolt_diagnostic": helper}))
            stack.enter_context(mock.patch.object(jolt_executor.time, "monotonic", side_effect=lambda: clock[0]))
            stack.enter_context(mock.patch.object(jolt_executor, "publish_json_v2", side_effect=publish))
            stack.enter_context(mock.patch.object(jolt_executor, "file_identity", side_effect=identity))
            stack.enter_context(mock.patch.object(jolt_executor.subprocess, "Popen", side_effect=spawn))
            stack.enter_context(redirect_stdout(output))
            if stdout_error:
                stack.enter_context(mock.patch("builtins.print", side_effect=BrokenPipeError()))
            code = jolt_executor.execute(args, whole_started=whole_started, script=script or Path("synthetic.py"))
        terminal = None if stdout_error else json.loads(output.getvalue())
        return code, terminal, args, helper, children, commands

    def read(self, args, name):
        return json.loads(args.output.with_name("output-lifecycle").joinpath(name).read_bytes())

    def test_complete_negative_result_has_durable_stages_and_post_publication_verdict(self):
        with tempfile.TemporaryDirectory() as folder:
            code, terminal, args, helper, children, commands = self.invoke(Path(folder))
            self.assertEqual(code, 0)
            self.assertTrue(terminal["completed"])
            self.assertEqual([command[2] for command in commands], ["worker-v2", "inventory-v2"])
            self.assertEqual(commands[0][-2:], ["--whole-started", "100.0"])
            helper.prepare_process_tree.assert_called_once()
            self.assertEqual(helper.terminate_owned_tree.call_count, 2)
            self.assertEqual(self.read(args, "worker-exit.json")["returncode"], 0)
            receipt = self.read(args, "executor.json")
            self.assertTrue(receipt["completion_candidate"])
            self.assertFalse(receipt["inventory"]["qualified"])
            self.assertEqual(len(receipt["stage_receipts"]), 9)
            durable = self.read(args, "terminal.json")
            self.assertTrue(durable["completion_candidate"])
            self.assertNotIn("completed", durable)
            self.assertTrue(durable["requires_observed_zero_exit"])
            self.assertTrue(durable["requires_post_publication_stdout_verdict"])
            self.assertEqual(terminal["terminal_receipt"], jolt_executor.file_identity(
                args.output.with_name("output-lifecycle") / "terminal.json"))

    def test_pre_worker_failure_retains_negative_receipts_without_spawn(self):
        for kwargs in ({"ownership_error": RuntimeError("synthetic /private/namespace")},
                       {"publication_failure": "invocation.json"},
                       {"publication_failure": "ownership.json"}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as folder:
                code, terminal, args, _, children, _ = self.invoke(Path(folder), **kwargs)
                self.assertEqual(code, 2)
                self.assertFalse(terminal["completed"])
                self.assertEqual(children, [])
                self.assertFalse(self.read(args, "terminal.json")["completion_candidate"])
                self.assertTrue(self.read(args, "executor.json")["errors"])
                self.assertNotIn("/private", json.dumps(self.read(args, "executor.json")))

    def test_original_deadline_includes_startup_and_does_not_spawn_late_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            code, terminal, args, _, children, _ = self.invoke(Path(folder), clock_at=1871.)
            self.assertEqual(code, 2)
            self.assertFalse(terminal["completed"])
            self.assertEqual(children, [])
            self.assertTrue(self.read(args, "executor.json")["timeout"])

    def test_worker_timeout_or_interrupt_revokes_worker_and_preserves_exit(self):
        for error in (subprocess.TimeoutExpired("synthetic", 1), KeyboardInterrupt(), SystemExit(7)):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as folder:
                code, terminal, args, helper, children, commands = self.invoke(Path(folder), worker_error=error)
                self.assertEqual(code, 2)
                self.assertFalse(terminal["completed"])
                self.assertEqual(len(commands), 1)
                children[0].kill.assert_called_once()
                helper.terminate_owned_tree.assert_called_once()
                self.assertEqual(self.read(args, "worker-exit.json")["returncode"], -9)
                self.assertTrue(self.read(args, "executor.json")["errors"])

    def test_nonzero_worker_cannot_start_verifier(self):
        with tempfile.TemporaryDirectory() as folder:
            code, terminal, args, _, _, commands = self.invoke(Path(folder), worker_returncode=7)
            self.assertEqual(code, 2)
            self.assertFalse(terminal["completed"])
            self.assertEqual(len(commands), 1)
            self.assertEqual(self.read(args, "executor.json")["returncode"], 7)

    def test_verifier_failure_and_incomplete_evidence_never_qualify(self):
        cases = ({"verifier_returncode": 2}, {"verifier_error": KeyboardInterrupt()},
                 {"verified": False}, {"complete": False})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as folder:
                code, terminal, args, helper, children, commands = self.invoke(Path(folder), **kwargs)
                self.assertEqual(code, 2)
                self.assertFalse(terminal["completed"])
                self.assertEqual(len(commands), 2)
                self.assertEqual(helper.terminate_owned_tree.call_count, 2)
                self.assertFalse(self.read(args, "executor.json")["completion_candidate"])

    def test_terminal_publication_or_hash_failure_cannot_erase_earlier_stages(self):
        for kwargs in ({"publication_failure": "executor.json"},
                       {"publication_failure": "finalization.json"},
                       {"publication_failure": "terminal.json"},
                       {"identity_failure": "worker-stdout.log"}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as folder:
                code, terminal, args, _, _, _ = self.invoke(Path(folder), **kwargs)
                self.assertEqual(code, 2)
                self.assertFalse(terminal["completed"])
                self.assertEqual(self.read(args, "worker-exit.json")["returncode"], 0)
                self.assertTrue(self.read(args, "verification.json")["inventory"]["readback_verified"])
                self.assertTrue(terminal["errors"])

    def test_post_spawn_publication_failure_immediately_revokes_without_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            code, terminal, args, helper, children, commands = self.invoke(
                Path(folder), publication_failure="worker-started.json")
            self.assertEqual(code, 2)
            self.assertEqual(len(commands), 1)
            children[0].kill.assert_called_once()
            self.assertEqual(self.read(args, "worker-exit.json")["returncode"], -9)
            self.assertFalse(terminal["completed"])

    def test_unconfirmed_process_tree_can_never_be_complete(self):
        for kwargs in ({"cleanup_confirmed": False}, {"pending": True},
                       {"cleanup_error": OSError("synthetic namespace unavailable")}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as folder:
                code, terminal, args, _, _, _ = self.invoke(Path(folder), **kwargs)
                self.assertEqual(code, 2)
                self.assertFalse(terminal["completed"])
                self.assertFalse(self.read(args, "executor.json")["cleanup_confirmed"])

    def test_slow_publication_preserves_candidate_but_rejects_final_success(self):
        for name in ("executor.json", "finalization.json", "terminal.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                code, terminal, args, _, _, _ = self.invoke(Path(folder), advance_after=name)
                self.assertEqual(code, 2)
                self.assertFalse(terminal["completed"])
                self.assertEqual(terminal["wall_seconds"], 1801.)
                self.assertFalse(self.read(args, "late-exit.json")["completed"])

    def test_slow_stage_closure_preserves_worker_and_verifier_reserves(self):
        for phase, end in (("worker", 1871.), ("verification", 1896.)):
            for suffix in ("cleanup", "exit"):
                with self.subTest(phase=phase, suffix=suffix), tempfile.TemporaryDirectory() as folder:
                    code, terminal, args, _, _, commands = self.invoke(Path(folder),
                        advance_after=f"{phase}-{suffix}.json", advance_to=end)
                    self.assertEqual(code, 2)
                    self.assertFalse(terminal["completed"])
                    self.assertTrue(self.read(args, "executor.json")["timeout"])
                    self.assertFalse(self.read(args, "executor.json")["completion_candidate"])
                    if phase == "worker":
                        self.assertEqual(len(commands), 1)

    def test_broken_stdout_retains_terminal_and_explicit_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            code, _, args, _, _, _ = self.invoke(Path(folder), stdout_error=True)
            self.assertEqual(code, 2)
            self.assertTrue(self.read(args, "terminal.json")["completion_candidate"])
            failure = self.read(args, "terminal-output-error.json")
            self.assertFalse(failure["completed"])
            self.assertEqual(failure["errors"][-1]["type"], "BrokenPipeError")

    def test_invalid_declaration_still_retains_startup_failure_and_blocks_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.fixture(root)
            args.declaration.write_text("{incomplete")
            with mock.patch.object(jolt_executor.subprocess, "Popen") as spawn, redirect_stdout(io.StringIO()):
                self.assertEqual(jolt_executor.execute(args, whole_started=time.monotonic(), script=Path("unused.py")), 2)
                spawn.assert_not_called()
                with self.assertRaisesRegex(ValueError, "already been used"):
                    jolt_executor.execute(args, whole_started=time.monotonic(), script=Path("unused.py"))
            self.assertFalse(self.read(args, "terminal.json")["completion_candidate"])

    def test_original_invocation_receipts_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, args, _, _, _ = self.invoke(Path(folder))
            original = args.output.with_name("output-lifecycle").joinpath("terminal.json").read_bytes()
            with mock.patch.object(jolt_executor.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(ValueError, "already been used"):
                    jolt_executor.execute(args, whole_started=time.monotonic(), script=Path("unused.py"))
                spawn.assert_not_called()
            self.assertEqual(args.output.with_name("output-lifecycle").joinpath("terminal.json").read_bytes(), original)

    def test_v1_consumed_or_retry_declaration_refuses_before_worker(self):
        mutations = ({"schema": "nca-learn-jolt-002-execution/v1"}, {"attempts": 2},
                     {"automatic_retry": True}, {"automatic_extension": True},
                     {"budget": {"maximum_wall_seconds": 1801}}, {"allocation_id": None},
                     {"allocation_id": "bad slug"}, {"authorization_ref": None},
                     {"authorization_ref": "https://github.com/techrote/neural-cyber-automata/issues/62#issuecomment-"})
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                args = self.fixture(Path(folder))
                declaration = json.loads(args.declaration.read_bytes())
                declaration.update(mutation)
                args.declaration.write_text(json.dumps(declaration))
                with mock.patch.object(jolt_executor.subprocess, "Popen") as spawn, redirect_stdout(io.StringIO()):
                    self.assertEqual(jolt_executor.execute(args, whole_started=time.monotonic(),
                                                          script=Path("unused.py")), 2)
                    spawn.assert_not_called()
                self.assertFalse(self.read(args, "terminal.json")["completion_candidate"])

    def test_allocation_identity_syntax_is_bounded_without_claiming_authorization(self):
        original = {"allocation_id": "synthetic-executor-fixture",
            "authorization_ref": "https://github.com/techrote/neural-cyber-automata/issues/62#issuecomment-1"}
        self.assertEqual(validate_v2_allocation_identity(original), original)
        for slug in ("", "a" * 81, "UPPERCASE", "../path", "-leading", "trailing-", "two--hyphens"):
            with self.subTest(slug=slug), self.assertRaises(ContractError):
                validate_v2_allocation_identity({**original, "allocation_id": slug})
        prefix = "https://github.com/techrote/neural-cyber-automata/issues/62#issuecomment-"
        for reference in (prefix, prefix + "0", prefix + "01", prefix + "1?query", prefix + "1/extra",
                          prefix + "9" * 21, prefix.replace("/62#", "/64#") + "1"):
            with self.subTest(reference=reference), self.assertRaises(ContractError):
                validate_v2_allocation_identity({**original, "authorization_ref": reference})

    def rebound(self, args, terminal, *, mutation=None, stream_mutation=None):
        """Rehash deliberate corruptions to test semantic gates beyond SHA."""
        lifecycle = args.output.with_name("output-lifecycle")
        docs = {path.name: json.loads(path.read_bytes()) for path in lifecycle.glob("*.json")}
        if mutation:
            mutation(docs)
        if stream_mutation:
            stream_mutation(lifecycle)
        for name, value in docs.items():
            if name not in {"executor.json", "finalization.json", "terminal.json"}:
                (lifecycle / name).write_text(json.dumps(value))
        executor = docs["executor.json"]
        executor["stage_receipts"] = {name: jolt_executor.file_identity(lifecycle / name)
                                     for name in executor["stage_receipts"]}
        executor["streams"] = {name: jolt_executor.file_identity(lifecycle / name) for name in executor["streams"]}
        (lifecycle / "executor.json").write_text(json.dumps(executor))
        docs["finalization.json"]["executor"] = jolt_executor.file_identity(lifecycle / "executor.json")
        (lifecycle / "finalization.json").write_text(json.dumps(docs["finalization.json"]))
        docs["terminal.json"].update(executor=jolt_executor.file_identity(lifecycle / "executor.json"),
            finalization=jolt_executor.file_identity(lifecycle / "finalization.json"))
        (lifecycle / "terminal.json").write_text(json.dumps(docs["terminal.json"]))
        terminal.update(executor=jolt_executor.file_identity(lifecycle / "executor.json"),
            finalization=jolt_executor.file_identity(lifecycle / "finalization.json"),
            terminal_receipt=jolt_executor.file_identity(lifecycle / "terminal.json"))
        return json.dumps(terminal).encode()

    def test_independent_lifecycle_readback_binds_declaration_exit_and_all_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            code, terminal, args, _, _, _ = self.invoke(Path(folder))
            lifecycle = args.output.with_name("output-lifecycle")
            verified = jolt_executor.verify_lifecycle(lifecycle, terminal_stdout=json.dumps(terminal).encode(),
                observed_exit_code=code, expected_declaration=jolt_executor.file_identity(args.declaration))
            self.assertTrue(verified["verified"])
            self.assertTrue(verified["completed"])
            self.assertEqual(verified["scope"], "executor_lifecycle_only")
            self.assertFalse(verified["worker_inventory"]["qualified"])
            self.assertEqual(verified["artifacts"], 16)
            self.assertEqual(verified["whole_process_started_monotonic"], 100.)
            self.assertEqual(verified["declaration"], jolt_executor.file_identity(args.declaration))
            with self.assertRaisesRegex(ContractError, "another independently bound declaration"):
                jolt_executor.verify_lifecycle(lifecycle, terminal_stdout=json.dumps(terminal).encode(),
                    observed_exit_code=0, expected_declaration={"bytes": 1, "sha256": "0" * 64})

    def test_candidate_or_stdout_alone_cannot_supply_observed_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            _, terminal, args, _, _, _ = self.invoke(Path(folder))
            lifecycle = args.output.with_name("output-lifecycle")
            result = jolt_executor.verify_lifecycle(lifecycle, terminal_stdout=json.dumps(terminal).encode(),
                                                   observed_exit_code=2)
            self.assertFalse(result["completed"])
            self.assertFalse(result["verified"])
            for exit_code in (None, False):
                with self.subTest(exit_code=exit_code), self.assertRaisesRegex(ContractError, "observed integer"):
                    jolt_executor.verify_lifecycle(lifecycle, terminal_stdout=json.dumps(terminal).encode(),
                                                   observed_exit_code=exit_code)
            with self.assertRaisesRegex(ContractError, "post-publication verdict"):
                jolt_executor.verify_lifecycle(lifecycle, terminal_stdout=lifecycle / "terminal.json",
                                               observed_exit_code=0)

    def test_complete_negative_executor_is_never_accepted_as_success(self):
        with tempfile.TemporaryDirectory() as folder:
            code, terminal, args, _, _, _ = self.invoke(Path(folder), worker_returncode=7)
            result = jolt_executor.verify_lifecycle(args.output.with_name("output-lifecycle"),
                terminal_stdout=json.dumps(terminal).encode(), observed_exit_code=code)
            self.assertFalse(result["verified"])
            self.assertFalse(result["completed"])
            with self.assertRaisesRegex(ContractError, "another independently bound declaration"):
                jolt_executor.verify_lifecycle(args.output.with_name("output-lifecycle"),
                    terminal_stdout=json.dumps(terminal).encode(), observed_exit_code=code,
                    expected_declaration={"bytes": 1, "sha256": "0" * 64})

    def test_unrehashed_stage_and_stream_corruption_is_detected(self):
        for name in ("worker-exit.json", "worker-stdout.log", "verification-stdout.log", "executor.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                _, terminal, args, _, _, _ = self.invoke(Path(folder))
                lifecycle = args.output.with_name("output-lifecycle")
                with (lifecycle / name).open("ab") as stream:
                    stream.write(b" ")
                with self.assertRaises(ContractError):
                    jolt_executor.verify_lifecycle(lifecycle, terminal_stdout=json.dumps(terminal).encode(),
                                                   observed_exit_code=0)

    def test_rehashed_semantic_corruptions_do_not_upgrade_failed_lifecycle(self):
        mutations = [
            lambda docs: docs["invocation.json"].update(declaration_schema="nca-learn-jolt-002-execution/v1"),
            lambda docs: docs["worker-started.json"].update(whole_process_started_monotonic=101.),
            lambda docs: docs["worker-exit.json"].update(returncode=7),
            lambda docs: docs["worker-exit.json"].update(process_exited=False),
            lambda docs: docs["worker-cleanup.json"].update(cleanup_confirmed=False),
            lambda docs: docs["verification-cleanup.json"].update(owned_processes_remaining=1),
            lambda docs: docs["verification-exit.json"].update(timeout=True),
            lambda docs: docs["worker-exit.json"].update(wall_seconds=100.),
            lambda docs: docs["finalization.json"].update(wall_seconds_after_executor_publication=1801.),
            lambda docs: docs["terminal.json"].update(completion_candidate=False),
            lambda docs: docs["executor.json"].update(errors=[{"stage": "failure", "type": "OSError", "code": None}]),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as folder:
                _, terminal, args, _, _, _ = self.invoke(Path(folder))
                stdout = self.rebound(args, terminal, mutation=mutation)
                with self.assertRaises(ContractError):
                    jolt_executor.verify_lifecycle(args.output.with_name("output-lifecycle"),
                        terminal_stdout=stdout, observed_exit_code=0)

    def test_verifier_output_must_equal_original_resealed_inventory_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            _, terminal, args, _, _, _ = self.invoke(Path(folder))

            def mutate_stream(lifecycle):
                path = lifecycle / "verification-stdout.log"
                value = json.loads(path.read_bytes())
                value["readback_complete"] = False
                path.write_text(json.dumps(value))

            stdout = self.rebound(args, terminal, stream_mutation=mutate_stream)
            with self.assertRaisesRegex(ContractError, "verification did not complete"):
                jolt_executor.verify_lifecycle(args.output.with_name("output-lifecycle"),
                    terminal_stdout=stdout, observed_exit_code=0)

    def test_surplus_failure_marker_or_missing_stage_blocks_acceptance(self):
        for name in ("late-exit.json", "terminal-output-error.json", "unexplained-extra"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                _, terminal, args, _, _, _ = self.invoke(Path(folder))
                lifecycle = args.output.with_name("output-lifecycle")
                (lifecycle / name).write_text("{}")
                with self.assertRaisesRegex(ContractError, "missing or surplus"):
                    jolt_executor.verify_lifecycle(lifecycle, terminal_stdout=json.dumps(terminal).encode(),
                                                   observed_exit_code=0)
        with tempfile.TemporaryDirectory() as folder:
            _, terminal, args, _, _, _ = self.invoke(Path(folder))
            lifecycle = args.output.with_name("output-lifecycle")
            (lifecycle / "worker-cleanup.json").unlink()
            with self.assertRaisesRegex(ContractError, "missing or surplus"):
                jolt_executor.verify_lifecycle(lifecycle, terminal_stdout=json.dumps(terminal).encode(),
                                               observed_exit_code=0)


class RealSyntheticChildTests(unittest.TestCase):
    """Exercise actual spawn, exit, stream sync and timeout; no host or learner."""

    def run_synthetic(self, root, *, worker_exit=0, worker_sleep=0, verifier_text=None, timeout=False):
        script = root / "synthetic-worker.py"
        verdict = {"inventory_verified": True, "readback_verified": True, "readback_complete": True,
                   "costs_verified": True, "status": "completed", "qualified": False}
        verifier_text = json.dumps(verdict) if verifier_text is None else verifier_text
        script.write_text("import sys, time\n"
            "if sys.argv[1] == 'worker-v2':\n"
            f"    time.sleep({worker_sleep!r})\n"
            "    print('synthetic child only', flush=True)\n"
            f"    raise SystemExit({worker_exit!r})\n"
            f"print({verifier_text!r}, flush=True)\n")
        args = LearningExecutorV2Tests().fixture(root)

        def terminate(process, identity, **kwargs):
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=3)
            return True

        helper = SimpleNamespace(prepare_process_tree=lambda: 987, owned_processes=lambda identity: {},
                                 terminate_owned_tree=terminate)
        started = time.monotonic() - (1769.8 if timeout else 0)
        output = io.StringIO()
        with mock.patch.dict(sys.modules, {"execute_jolt_diagnostic": helper}), redirect_stdout(output):
            result = jolt_executor.execute(args, whole_started=started, script=script)
        lifecycle = args.output.with_name("output-lifecycle")
        return result, json.loads(output.getvalue()), lifecycle

    def test_real_child_exit_and_complete_negative_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            result, terminal, lifecycle = self.run_synthetic(Path(folder))
            self.assertEqual(result, 0)
            self.assertTrue(terminal["completed"])
            self.assertEqual(json.loads((lifecycle / "worker-exit.json").read_bytes())["returncode"], 0)
            self.assertEqual((lifecycle / "worker-stdout.log").read_text().strip(), "synthetic child only")
            readback = jolt_executor.verify_lifecycle(lifecycle, terminal_stdout=json.dumps(terminal).encode(),
                observed_exit_code=result, expected_declaration=jolt_executor.file_identity(Path(folder) / "declaration.json"))
            self.assertTrue(readback["verified"])
            self.assertTrue(readback["completed"])

    def test_real_child_nonzero_and_malformed_inventory_are_negative(self):
        for kwargs in ({"worker_exit": 7}, {"verifier_text": "{truncated"}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as folder:
                result, terminal, lifecycle = self.run_synthetic(Path(folder), **kwargs)
                self.assertEqual(result, 2)
                self.assertFalse(terminal["completed"])
                self.assertTrue((lifecycle / "executor.json").is_file())
                self.assertTrue((lifecycle / "terminal.json").is_file())

    def test_real_child_timeout_is_killed_and_has_no_verification_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            result, terminal, lifecycle = self.run_synthetic(Path(folder), timeout=True, worker_sleep=3)
            self.assertEqual(result, 2)
            self.assertFalse(terminal["completed"])
            receipt = json.loads((lifecycle / "worker-exit.json").read_bytes())
            self.assertTrue(receipt["timeout"])
            self.assertTrue(receipt["process_exited"])
            self.assertNotEqual(receipt["returncode"], 0)
            self.assertFalse((lifecycle / "verification-started.json").exists())


if __name__ == "__main__":
    unittest.main()
