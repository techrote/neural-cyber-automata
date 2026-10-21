"""Bounded supervisor/worker conformance with synthetic source admission.

Source verification and process creation are patched only within these tests.
No production bypass exists. Optional cases execute the actual CPU worker in
process for at most three tiny steps; these fixtures do not qualify hardware.
"""

from collections.abc import Mapping
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from nca.contracts import sha256_json
from nca.ik import declaration
from nca.ik.checkpoint import load_checkpoint
from nca.ik.device import DeviceError
from nca.ik.runtime import (FINALIZATION_RESERVE, cancel_run, execute, inspect_run,
                            read_json)
from nca.ik.runtime_worker import worker
from nca.ik.workload import SmokeWorkload
from nca.runs import immutable_json, sha256_file


OPTIONAL = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None
REAL_POPEN = subprocess.Popen


class FinishedProcess:
    def __init__(self, code):
        self.returncode = code

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


def direct_worker_process(command, **options):
    """Run the real worker under the supervisor's real active request."""
    if not isinstance(command, list) or command[1:4] != ["-m", "nca.ik.runtime", "_worker"]:
        # platform.processor() may invoke a harmless OS inventory command while
        # building the device report.  The test intercepts only the IK worker.
        return REAL_POPEN(command, **options)
    root = Path(command[command.index("--run") + 1])
    name = command[command.index("--attempt") + 1]
    captured = io.StringIO()
    # An in-process fixture must not replace the unittest process's handlers.
    with patch("nca.ik.runtime_worker.signal.signal"), redirect_stdout(captured):
        code = worker(root, name)
    options["stdout"].write(captured.getvalue().encode())
    options["stdout"].flush()
    return FinishedProcess(code)


class Clock:
    def __init__(self):
        self.now = 1000.

    def __call__(self):
        return self.now


class RuntimeFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.count = 0

    def prepare(self, *, device="cpu", **config_changes):
        config_path = Path(__file__).resolve().parents[1] / "configs/ik/runtime-smoke-v1.json"
        config = json.loads(config_path.read_text())
        config.update(batch_size=3, steps=3, max_optimizer_steps=3, max_samples=9)
        config.update(config_changes)
        self.count += 1
        fixture = self.root / f"config-{self.count}.json"
        fixture.write_text(json.dumps(config))
        source_files = {"src/nca/ik/workload.py": "b" * 64}
        snapshot = {"revision": "a" * 40, "source_sha256": sha256_json(source_files), "files": source_files}
        path = self.root / f"run-{self.count}"
        with patch("nca.ik.declaration.source_snapshot", return_value=snapshot):
            declaration.prepare(fixture, device, path)
        return path, sha256_file(path / "declaration.json")

    def execute_direct(self, path, digest, **options):
        with patch("nca.ik.declaration.verify_source"), \
                patch("nca.ik.runtime.subprocess.Popen", side_effect=direct_worker_process):
            return execute(path, digest, **options)

    def assertNestedEqual(self, left, right, torch):
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor)
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.device, right.device)
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, Mapping):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assertNestedEqual(left[key], right[key], torch)
        elif isinstance(left, (list, tuple)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for actual, expected in zip(left, right):
                self.assertNestedEqual(actual, expected, torch)
        else:
            self.assertEqual(left, right)


class IKRuntimeSupervisorTests(RuntimeFixture):
    def test_failed_attempts_retain_aggregate_wall_step_and_sample_cost(self):
        path, digest = self.prepare(steps=2, max_optimizer_steps=2, max_samples=6)
        clock = Clock()
        delays = iter((2., 3.))

        def failed_child(command, **options):
            name = command[command.index("--attempt") + 1]
            attempt = path / "attempts" / name
            request = read_json(attempt / "request.json")
            immutable_json(attempt / "reserve-000001.json",
                           {"attempted_step": 1, "samples": 3,
                            "next_committed_step": request["prior_committed_steps"] + 1})
            immutable_json(attempt / "worker.json", {"status": "failed", "reason": "out_of_memory",
                           "qualification": "unmeasured", "attempted_steps": 1})
            clock.now += next(delays)
            return FinishedProcess(1)

        with patch("nca.ik.runtime.time.monotonic", side_effect=clock), \
                patch("nca.ik.declaration.verify_source"), \
                patch("nca.ik.runtime.subprocess.Popen", side_effect=failed_child) as spawn:
            first = execute(path, digest)
            first_record = (path / "records/attempt-0001.json").read_bytes()
            second = execute(path, digest, resume=True)
            capped = execute(path, digest, resume=True)
        self.assertEqual(spawn.call_count, 2)
        self.assertEqual(first["charged_seconds"], 2. + FINALIZATION_RESERVE)
        self.assertEqual(second["charged_seconds"], 5. + 2 * FINALIZATION_RESERVE)
        self.assertEqual(second["charged_steps"], 2)
        self.assertEqual(second["charged_samples"], 6)
        self.assertEqual(second["committed_steps"], 0)
        self.assertEqual(second["remaining_steps"], 0)
        self.assertEqual(second["qualification"], "unmeasured")
        self.assertEqual([item["status"] for item in capped["attempts"]], ["failed", "failed"])
        self.assertEqual(capped["admission_refused"], "completed_or_exhausted_allocation")
        self.assertEqual(capped["charged_seconds"], second["charged_seconds"])
        self.assertEqual(len(list((path / "attempts").iterdir())), 2)
        self.assertEqual((path / "records/attempt-0001.json").read_bytes(), first_record)
        second_request = read_json(path / "attempts/attempt-0002/request.json")
        self.assertEqual(second_request["prior_charged_steps"], 1)
        self.assertEqual(second_request["prior_charged_seconds"], first["charged_seconds"])
        self.assertTrue(all(item["resolved_config"]["batch_size"] == 3 for item in capped["attempts"]))

    def test_source_mismatch_records_immutable_failure_without_starting_worker(self):
        path, digest = self.prepare()
        with patch("nca.ik.declaration.verify_source", side_effect=ValueError("synthetic source mismatch")), \
                patch("nca.ik.runtime.subprocess.Popen") as spawn:
            result = execute(path, digest)
        spawn.assert_not_called()
        self.assertEqual(result["attempts"][0]["status"], "failed")
        self.assertEqual(result["charged_steps"], 0)
        self.assertGreaterEqual(result["charged_seconds"], FINALIZATION_RESERVE)
        self.assertFalse(result["active"])
        self.assertFalse(result["unresolved_attempt"])
        self.assertEqual(result["qualification"], "unmeasured")
        original = (path / "records/attempt-0001.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "resume"):
            execute(path, digest)
        self.assertEqual((path / "records/attempt-0001.json").read_bytes(), original)

    def test_exhausted_wall_allocation_refuses_new_attempt_without_reset(self):
        path, digest = self.prepare(max_wall_seconds=7.)
        clock = Clock()

        def failed_startup(command, **options):
            name = command[command.index("--attempt") + 1]
            immutable_json(path / "attempts" / name / "worker.json",
                           {"status": "failed", "reason": "synthetic startup failure",
                            "qualification": "unmeasured"})
            clock.now += 2.
            return FinishedProcess(1)

        with patch("nca.ik.runtime.time.monotonic", side_effect=clock), \
                patch("nca.ik.declaration.verify_source"), \
                patch("nca.ik.runtime.subprocess.Popen", side_effect=failed_startup) as spawn:
            first = execute(path, digest)
            second = execute(path, digest, resume=True)
        self.assertEqual(spawn.call_count, 1)
        self.assertEqual(first["charged_seconds"], 7.)
        self.assertEqual(first["remaining_seconds"], 0.)
        self.assertEqual(second["admission_refused"], "completed_or_exhausted_allocation")
        self.assertEqual(second["attempt_count"], 1)
        self.assertEqual(second["charged_seconds"], 7.)

    def test_runtime_mismatch_is_retained_before_optional_device_or_optimizer_work(self):
        path, digest = self.prepare()
        changed = copy.deepcopy(declaration.runtime_expectation())
        changed["python"] = "0.0.0-fixture"
        with patch("nca.ik.declaration.runtime_expectation", return_value=changed), \
                patch("nca.ik.device.configure_device") as configure:
            result = self.execute_direct(path, digest)
        configure.assert_not_called()
        self.assertEqual(result["attempts"][0]["status"], "failed")
        self.assertEqual(result["charged_steps"], 0)
        self.assertEqual(result["qualification"], "unmeasured")
        self.assertFalse(list((path / "attempts/attempt-0001").glob("reserve-*.json")))

    def test_missing_explicit_cuda_is_failed_without_fallback(self):
        path, digest = self.prepare(device="cuda:0")
        with patch("nca.ik.device.configure_device", side_effect=DeviceError(
                "cuda_unavailable", "synthetic unavailable device", {"requested_device": "cuda:0"})) as configure:
            result = self.execute_direct(path, digest)
        self.assertEqual(configure.call_count, 1)
        self.assertEqual(configure.call_args.args[0], "cuda:0")
        record = result["attempts"][0]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["device"], "cuda:0")
        self.assertEqual(record["failures"], ["cuda_unavailable"])
        self.assertEqual(result["charged_steps"], 0)
        self.assertEqual(result["qualification"], "unmeasured")

    def test_unsealed_or_active_history_cannot_reset_its_allocation(self):
        path, digest = self.prepare()
        immutable_json(path / "active.json", {"attempt": "attempt-0001", "declaration_sha256": digest})
        (path / "attempts/attempt-0001").mkdir(parents=True)
        state = inspect_run(path)
        self.assertTrue(state["active"])
        self.assertTrue(state["unresolved_attempt"])
        self.assertEqual(state["remaining_seconds"], 0.)
        with patch("nca.ik.runtime.subprocess.Popen") as spawn, self.assertRaisesRegex(ValueError, "active or unsealed"):
            execute(path, digest, resume=True)
        spawn.assert_not_called()
        receipt = cancel_run(path)
        self.assertEqual(receipt["cancellation_requested"], "attempt-0001")
        original = (path / "cancel-attempt-0001.json").read_bytes()
        cancel_run(path)
        self.assertEqual((path / "cancel-attempt-0001.json").read_bytes(), original)

    def test_worker_success_flag_without_step_evidence_cannot_qualify(self):
        path, digest = self.prepare()

        def false_success(command, **options):
            name = command[command.index("--attempt") + 1]
            immutable_json(path / "attempts" / name / "worker.json",
                           {"status": "completed", "reason": "synthetic false success",
                            "qualification": "cpu_smoke_passed", "committed_steps": 3})
            return FinishedProcess(0)

        with patch("nca.ik.declaration.verify_source"), \
                patch("nca.ik.runtime.subprocess.Popen", side_effect=false_success):
            result = execute(path, digest)
        self.assertEqual(result["attempts"][0]["status"], "failed")
        self.assertEqual(result["qualification"], "unmeasured")
        self.assertEqual(result["committed_steps"], 0)

    def test_cancellation_just_after_child_exit_is_retained_by_supervisor(self):
        path, digest = self.prepare()

        def exiting_child(command, **options):
            name = command[command.index("--attempt") + 1]
            immutable_json(path / "attempts" / name / "worker.json",
                           {"status": "failed", "reason": "synthetic stopped child",
                            "qualification": "unmeasured"})
            cancel_run(path)
            return FinishedProcess(1)

        with patch("nca.ik.declaration.verify_source"), \
                patch("nca.ik.runtime.subprocess.Popen", side_effect=exiting_child):
            result = execute(path, digest)
        self.assertEqual(result["attempts"][0]["status"], "canceled")
        self.assertEqual(result["attempts"][0]["failures"], ["operator_cancellation"])
        self.assertEqual(result["qualification"], "unmeasured")
        self.assertEqual(result["charged_steps"], 0)


@unittest.skipUnless(OPTIONAL, "optional pinned IK worker runtime is not installed")
class IKRuntimeCPUWorkerTests(RuntimeFixture):
    def test_stop_then_resume_matches_uninterrupted_samples_model_and_optimizer(self):
        import torch
        baseline, baseline_digest = self.prepare()
        expected = self.execute_direct(baseline, baseline_digest)
        resumed, digest = self.prepare()
        first = self.execute_direct(resumed, digest, stop_after=1)
        self.assertEqual(first["committed_steps"], 1)
        self.assertEqual(first["qualification"], "unmeasured")
        first_bytes = (resumed / "records/attempt-0001.json").read_bytes()
        actual = self.execute_direct(resumed, digest, resume=True)
        self.assertEqual(actual["committed_steps"], 3)
        self.assertEqual(actual["charged_steps"], 3)
        self.assertEqual(actual["charged_samples"], 9)
        self.assertEqual(actual["qualification"], "cpu_smoke_passed")
        self.assertGreater(actual["charged_seconds"], first["charged_seconds"])
        self.assertEqual((resumed / "records/attempt-0001.json").read_bytes(), first_bytes)
        expected_steps = [read_json(p) for p in sorted((baseline / "attempts").glob("*/step-*.json"))]
        actual_steps = [read_json(p) for p in sorted((resumed / "attempts").glob("*/step-*.json"))]
        self.assertEqual([row["sample_ids"] for row in actual_steps], [row["sample_ids"] for row in expected_steps])
        self.assertEqual([row["result"]["loss"] for row in actual_steps], [row["result"]["loss"] for row in expected_steps])
        payloads = []
        for root, result in ((baseline, expected), (resumed, actual)):
            reference = result["checkpoint"]["reference"]
            payloads.append(load_checkpoint(root / result["checkpoint"]["path"],
                                           expected_sha256=reference["sha256"], expected_metadata=reference["metadata"]))
        for key in ("model", "optimizer", "rng"):
            self.assertNestedEqual(payloads[0][key], payloads[1][key], torch)
        for key in ("committed_steps", "cursor", "sampler_position", "permutation", "epoch"):
            self.assertEqual(payloads[0]["training_state"][key], payloads[1]["training_state"][key])

    def test_oom_reservation_is_charged_and_same_batch_resume_remains_bounded(self):
        path, digest = self.prepare(steps=2, max_optimizer_steps=3, max_samples=9)
        with patch.object(SmokeWorkload, "step", side_effect=MemoryError("synthetic OOM")):
            failed = self.execute_direct(path, digest)
        self.assertEqual(failed["attempts"][0]["status"], "failed")
        self.assertEqual(failed["attempts"][0]["failures"], ["out_of_memory"])
        self.assertEqual((failed["charged_steps"], failed["committed_steps"]), (1, 0))
        self.assertIsNotNone(failed["checkpoint"])
        actual = self.execute_direct(path, digest, resume=True)
        self.assertEqual((actual["charged_steps"], actual["committed_steps"], actual["charged_samples"]), (3, 2, 9))
        self.assertEqual(actual["remaining_steps"], 0)
        self.assertEqual([row["resolved_config"]["batch_size"] for row in actual["attempts"]], [3, 3])
        self.assertEqual(actual["attempts"][0]["failures"], ["out_of_memory"])

    def test_corrupted_checkpoint_refuses_resume_before_new_optimizer_work(self):
        path, digest = self.prepare()
        state = self.execute_direct(path, digest, stop_after=1)
        checkpoint = path / state["checkpoint"]["path"]
        contents = checkpoint.read_bytes()
        checkpoint.write_bytes(contents[:-1] + bytes([contents[-1] ^ 1]))
        with patch.object(SmokeWorkload, "step") as step, self.assertRaisesRegex(ValueError, "artifact bytes differ"):
            self.execute_direct(path, digest, resume=True)
        step.assert_not_called()
        self.assertEqual(len(list((path / "attempts").iterdir())), 1)
        self.assertFalse((path / "active.json").exists())

    def test_cancellation_arriving_during_final_step_cannot_be_qualified(self):
        path, digest = self.prepare(steps=1, max_optimizer_steps=1, max_samples=3)
        ordinary_step = SmokeWorkload.step

        def cancel_after_step(workload, batch):
            result = ordinary_step(workload, batch)
            cancel_run(path)
            return result

        with patch.object(SmokeWorkload, "step", new=cancel_after_step):
            result = self.execute_direct(path, digest)
        self.assertEqual(result["committed_steps"], 1)
        self.assertEqual(result["attempts"][0]["status"], "canceled")
        self.assertEqual(result["qualification"], "unmeasured")

    def test_environment_continuation_limit_is_sticky_across_later_resumes(self):
        path, digest = self.prepare(steps=2, max_optimizer_steps=2, max_samples=6)
        changed = copy.deepcopy(declaration.runtime_expectation())
        changed["python"] = "0.0.0-fixture"
        with patch("nca.ik.declaration.runtime_expectation", return_value=changed):
            first = self.execute_direct(path, digest, stop_after=1, allow_environment_change=True)
        self.assertEqual(first["qualification"], "unmeasured")
        self.assertTrue(first["attempts"][0]["metrics"]["worker"]["reproducibility_limits"])
        # Returning to the declared runtime cannot erase the earlier attempt's
        # explicitly disclosed continuation or produce an exact-environment pass.
        second = self.execute_direct(path, digest, resume=True)
        self.assertEqual(second["committed_steps"], 2)
        self.assertEqual(second["qualification"], "unmeasured")
        self.assertTrue(second["attempts"][-1]["metrics"]["worker"]["reproducibility_limits"])

    def assertEnvironmentRefusalDoesNotRebase(self, path, digest, first):
        from nca.ik.device import configure_device, environment_identity
        initial_environment = first["environment_sha256"]
        committed = first["committed_steps"]
        charged = first["charged_steps"]
        self.assertIsNotNone(initial_environment)

        def changed_device(*arguments, **options):
            context = configure_device(*arguments, **options)
            context.report = dict(context.report, cpu_processor="synthetic continuation environment B")
            context.report["environment_sha256"] = sha256_json(environment_identity(context.report))
            self.assertNotEqual(context.report["environment_sha256"], initial_environment)
            return context

        with patch("nca.ik.device.configure_device", side_effect=changed_device):
            for _ in range(2):
                with patch.object(SmokeWorkload, "step") as step:
                    refused = self.execute_direct(path, digest, resume=True)
                step.assert_not_called()
                self.assertEqual(refused["attempts"][-1]["status"], "failed")
                self.assertFalse(refused["attempts"][-1]["metrics"]["worker"]["environment_admitted"])
                self.assertEqual(refused["environment_sha256"], initial_environment)
                self.assertEqual(refused["committed_steps"], committed)
                self.assertEqual(refused["charged_steps"], charged)
                self.assertTrue(refused["lineage_limits"])
            continued = self.execute_direct(path, digest, resume=True, allow_environment_change=True)
        self.assertEqual(continued["committed_steps"], 2)
        self.assertEqual(continued["attempts"][-1]["status"], "completed")
        self.assertTrue(continued["attempts"][-1]["metrics"]["worker"]["environment_admitted"])
        self.assertNotEqual(continued["environment_sha256"], initial_environment)
        self.assertEqual(continued["qualification"], "unmeasured")
        self.assertTrue(continued["attempts"][-1]["metrics"]["worker"]["reproducibility_limits"])
        self.assertEqual(continued["charged_steps"], 2)

    def test_refused_environment_cannot_replace_checkpoint_environment_authority(self):
        path, digest = self.prepare(steps=2, max_optimizer_steps=2, max_samples=6)
        first = self.execute_direct(path, digest, stop_after=1)
        self.assertEqual(first["committed_steps"], 1)
        self.assertIsNotNone(first["checkpoint"])
        self.assertEnvironmentRefusalDoesNotRebase(path, digest, first)

    def test_refused_environment_cannot_replace_preflight_authority_without_checkpoint(self):
        path, digest = self.prepare(steps=2, max_optimizer_steps=2, max_samples=6)
        with patch("nca.ik.checkpoint.write_checkpoint", side_effect=ValueError("synthetic initial checkpoint failure")):
            first = self.execute_direct(path, digest)
        self.assertEqual(first["attempts"][0]["status"], "failed")
        self.assertTrue(first["attempts"][0]["metrics"]["worker"]["environment_admitted"])
        self.assertEqual(first["committed_steps"], 0)
        self.assertEqual(first["charged_steps"], 0)
        self.assertIsNone(first["checkpoint"])
        self.assertEnvironmentRefusalDoesNotRebase(path, digest, first)


if __name__ == "__main__":
    unittest.main()
