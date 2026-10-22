"""External process-envelope conformance with the child boundary mocked.

Retained sample semantics are covered by test_ik_dataset_readback. These tests
exercise process timing, termination, receipt binding, and single-use admission
without running another teacher or claiming a measured dataset allocation.
"""

import importlib.util
from itertools import count
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from nca.contracts import canonical_json, sha256_json
from nca.runs import immutable_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
TOOL_SPEC = importlib.util.spec_from_file_location("ik_dataset_envelope_dependency", ROOT / "tools/ik_dataset.py")
dataset = importlib.util.module_from_spec(TOOL_SPEC)
TOOL_SPEC.loader.exec_module(dataset)
SPEC = importlib.util.spec_from_file_location("ik_dataset_supervise_tests", ROOT / "tools/ik_dataset_supervise.py")
envelope = importlib.util.module_from_spec(SPEC)
with patch.dict(sys.modules, {"ik_dataset": dataset}):
    SPEC.loader.exec_module(envelope)


class ProcessEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.registry = self.root / "registry"
        self.output = self.root / "artifact"
        self.path = self.root / "declaration.json"
        relative = "configs/ik/datasets/tiny_v1.json"
        files = {name: sha256_file(ROOT / name) for name in (
            relative, dataset.ACCEPTANCE, "src/nca/ik/data.py",
            "tools/ik_dataset.py", "tools/ik_dataset_supervise.py",
        )}
        source = {"revision": "b" * 40, "tree": "c" * 40, "files": files,
                  "source_sha256": sha256_json(files)}
        with patch.object(dataset, "_source", return_value=(source, relative)):
            self.declaration = dataset.declare(ROOT / relative, self.path, self.registry)
        self.key = self.declaration["sha256"]
        self.child = Mock(pid=123456)
        self.child.wait.return_value = 0
        self.child.poll.return_value = 0
        self.mutate_inner = None
        self.change_declaration = False
        self.started = patch.object(envelope, "_STARTED", 0.0)
        self.clock = patch.object(envelope.time, "perf_counter", side_effect=count(0.125, 0.125))
        self.started.start()
        self.clock_mock = self.clock.start()
        self.addCleanup(self.started.stop)
        self.addCleanup(self.clock.stop)

    @property
    def receipt_path(self):
        return self.output.with_name(self.output.name + ".envelope.json")

    @property
    def claim_path(self):
        return self.registry / (self.key + ".envelope-started.json")

    def publish_child_receipts(self, *args, **kwargs):
        """Mock only what the exited inner process publishes to its parent."""
        self.output.mkdir()
        # The envelope consumes these manifest identity fields; the separate
        # semantic readback tests exercise the complete manifest and its rows.
        manifest = dataset._seal({
            "schema": dataset.MANIFEST_SCHEMA, "run_id": self.declaration["run_id"],
            "declaration_sha256": self.key, "status": "complete",
        })
        immutable_json(self.output / "manifest.json", manifest)
        tasks = self.declaration["schedule"]["tasks"]
        inner = {
            "schema": "nca-ik-dataset-supervisor/v1", "declaration_sha256": self.key,
            "status": "complete", "checks": dict.fromkeys(dataset.SUPERVISOR_CHECKS, True),
            "child_returncode": 0, "child_timed_out": False, "interrupted": False, "execution_failure": None,
            "child_whole_process_seconds": 0.1, "aggregate_through_readback_seconds": 0.2,
            "max_wall_seconds": self.declaration["config"]["budget"]["max_wall_seconds"],
            "final_receipt_write_excluded_from_elapsed": True,
            "journal": {"valid_prefix": True, "error": None, "started": tasks, "finished": tasks,
                        "unknown_attempts": 0, "unknown_additional_work": False,
                        "teacher_iterations_actual_lower_bound": 0, "teacher_iterations_conservative": 0},
            "manifest_file_sha256": sha256_file(self.output / "manifest.json"),
            "finished_utc": "mocked-process-boundary-only",
        }
        if self.mutate_inner is not None:
            self.mutate_inner(inner)
        immutable_json(self.output / "supervisor.json", dataset._seal(inner))
        if self.change_declaration:
            value = {key: value for key, value in self.declaration.items() if key != "sha256"}
            value["changed_after_spawn"] = True
            self.path.write_text(canonical_json(dataset._seal(value)) + "\n")
        return self.child

    def run_envelope(self):
        return envelope.supervise(self.path, self.output, self.registry)

    def test_success_binds_exited_child_and_published_receipt_bytes(self):
        with patch.object(envelope.subprocess, "Popen", side_effect=self.publish_child_receipts) as spawn:
            result = self.run_envelope()
        self.assertEqual(result["status"], "complete")
        self.assertTrue(all(value is True for value in result["checks"].values()))
        self.assertEqual(result["declaration_sha256"], self.key)
        self.assertEqual(result["declaration_file_sha256"], sha256_file(self.path))
        self.assertEqual(result["artifact_file_sha256"], {
            name: sha256_file(self.output / name) for name in ("manifest.json", "supervisor.json")
        })
        self.assertEqual(dataset._read(self.receipt_path, sealed=True), result)
        self.assertTrue(self.claim_path.is_file())
        self.assertGreater(result["generator_process_and_receipt_readback_seconds"], 0)
        self.assertGreaterEqual(result["module_through_readback_seconds"],
                                result["generator_process_and_receipt_readback_seconds"])
        self.assertTrue(result["envelope_final_receipt_write_excluded_from_elapsed"])
        self.assertEqual(spawn.call_args.kwargs["cwd"], ROOT)
        self.assertEqual(spawn.call_args.kwargs["start_new_session"], envelope.os.name == "posix")
        self.assertIn("generate", spawn.call_args.args[0])
        self.assertLess(self.child.wait.call_args.kwargs["timeout"], result["max_wall_seconds"])

    def test_rehashed_inner_receipt_with_a_failed_check_is_incomplete(self):
        self.mutate_inner = lambda value: value["checks"].update(counts_match_journal=False)
        with patch.object(envelope.subprocess, "Popen", side_effect=self.publish_child_receipts):
            result = self.run_envelope()
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["checks"]["inner_semantic_readback_passed"])
        self.assertTrue(self.claim_path.is_file())

    def test_rehashed_inner_receipt_cannot_name_a_different_manifest(self):
        self.mutate_inner = lambda value: value.update(manifest_file_sha256="0" * 64)
        with patch.object(envelope.subprocess, "Popen", side_effect=self.publish_child_receipts):
            result = self.run_envelope()
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["checks"]["artifact_bindings"])

    def test_declaration_mutation_during_the_child_is_detected(self):
        self.change_declaration = True
        with patch.object(envelope.subprocess, "Popen", side_effect=self.publish_child_receipts):
            result = self.run_envelope()
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["checks"]["declaration_file_unchanged"])

    def test_timeout_terminates_the_child_and_preserves_an_incomplete_receipt(self):
        self.child.wait.side_effect = subprocess.TimeoutExpired("mocked child", 1)
        self.child.poll.return_value = None
        def terminate(child):
            self.assertIs(child, self.child)
            self.child.poll.return_value = -9
            return -9
        with patch.object(envelope.subprocess, "Popen", side_effect=self.publish_child_receipts), \
                patch.object(envelope, "_terminate", side_effect=terminate) as stop:
            result = self.run_envelope()
        stop.assert_called_once_with(self.child)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["child_returncode"], -9)
        self.assertTrue(result["child_timed_out"])
        self.assertFalse(result["checks"]["complete_generator_process"])
        self.assertEqual(dataset._read(self.receipt_path, sealed=True)["status"], "incomplete")

    def test_termination_failure_still_publishes_failure_and_consumes_the_attempt(self):
        self.child.wait.side_effect = subprocess.TimeoutExpired("mocked child", 1)
        self.child.poll.return_value = None
        with patch.object(envelope.subprocess, "Popen", side_effect=self.publish_child_receipts), \
                patch.object(envelope, "_terminate", side_effect=OSError("private test path must not leak")):
            result = self.run_envelope()
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(result["child_timed_out"])
        self.assertIn("process_termination_OSError", result["failures"])
        self.assertNotIn("private test path", canonical_json(result))
        self.assertTrue(self.claim_path.is_file())
        self.assertEqual(dataset._read(self.receipt_path, sealed=True)["status"], "incomplete")

    def test_spawn_failure_cannot_be_retried_under_another_output_name(self):
        with patch.object(envelope.subprocess, "Popen", side_effect=OSError("fixture launch failure")) as spawn:
            result = self.run_envelope()
            self.assertEqual(result["status"], "incomplete")
            self.assertIn("OSError", result["failures"])
            self.assertTrue(self.claim_path.is_file())
            self.output = self.root / "different-output"
            with self.assertRaises(ValueError):
                self.run_envelope()
        spawn.assert_called_once()

    def test_undeclared_helper_bytes_are_refused_before_child_admission(self):
        value = {key: value for key, value in self.declaration.items() if key != "sha256"}
        value["source"]["files"]["tools/ik_dataset_supervise.py"] = "0" * 64
        value["source"]["source_sha256"] = sha256_json(value["source"]["files"])
        self.path.write_text(canonical_json(dataset._seal(value)) + "\n")
        with patch.object(envelope.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(dataset.DatasetToolError, "envelope_source_not_declared"):
                self.run_envelope()
        spawn.assert_not_called()
        self.assertFalse(self.claim_path.exists())

    def test_exhausted_admission_budget_never_spawns_work(self):
        maximum = self.declaration["config"]["budget"]["max_wall_seconds"]
        self.clock_mock.side_effect = [maximum - 1.0, maximum - 0.8, maximum - 0.6, maximum - 0.4]
        with patch.object(envelope.subprocess, "Popen") as spawn:
            result = self.run_envelope()
        spawn.assert_not_called()
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("envelope_admission_budget_exhausted", result["failures"])
        self.assertTrue(self.claim_path.is_file())

    def test_receipt_publication_overrun_leaves_an_explicit_marker(self):
        maximum = self.declaration["config"]["budget"]["max_wall_seconds"]
        self.clock_mock.side_effect = [0.1, 0.2, 0.3, maximum - 0.1, maximum + 0.1]
        with patch.object(envelope.subprocess, "Popen", side_effect=self.publish_child_receipts):
            with self.assertRaisesRegex(dataset.DatasetToolError, "envelope_publication_budget_exhausted"):
                self.run_envelope()
        self.assertTrue(self.receipt_path.is_file())
        marker = self.receipt_path.with_name(self.receipt_path.name + ".overrun.json")
        self.assertEqual(dataset._read(marker), {"status": "incomplete", "declaration_sha256": self.key})
        self.assertTrue(self.claim_path.is_file())

    def test_posix_termination_targets_the_child_process_group(self):
        self.child.wait.return_value = -9
        with patch.object(envelope.os, "name", "posix"), patch.object(envelope.os, "killpg", create=True) as kill:
            result = envelope._terminate(self.child)
        kill.assert_called_once_with(self.child.pid, envelope.signal.SIGKILL)
        self.child.wait.assert_called_once_with(timeout=5)
        self.assertEqual(result, -9)


if __name__ == "__main__":
    unittest.main()
