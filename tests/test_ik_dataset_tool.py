"""Failure-preserving allocation and final-suite lifecycle conformance."""

import importlib.util
from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nca.contracts import canonical_json, sha256_json
from nca.runs import immutable_bytes, immutable_json, sha256_file


SPEC = importlib.util.spec_from_file_location("ik_dataset_tool_tests", Path(__file__).parents[1] / "tools/ik_dataset.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def record(value):
    return SimpleNamespace(to_dict=lambda: value, sha256=sha256_json(value))


def fixture_source():
    relative = "configs/ik/datasets/tiny_v1.json"
    files = {name: sha256_file(tool.ROOT / name) for name in
             (relative, tool.ACCEPTANCE, "tools/ik_dataset.py", "tools/ik_dataset_supervise.py", "src/nca/ik/data.py")}
    return ({"revision": "b" * 40, "tree": "c" * 40, "files": files,
             "source_sha256": sha256_json(files)}, relative)


def tiny_artifact(root, *, supervisor=True):
    """Actual tiny retained rows; only dirty-source and process-cap admission are mocked.

    Unit tests run within one interpreter, so applying an irreversible process
    memory ceiling here would interfere with other test classes. Separate
    subprocess conformance tests exercise the physical memory ceiling.
    """
    declaration_path, registry = root / "declaration.json", root / "registry"
    with patch.object(tool, "_source", return_value=fixture_source()):
        declaration = tool.declare(tool.ROOT / "configs/ik/datasets/tiny_v1.json", declaration_path, registry)
        immutable_json(registry / (declaration["sha256"] + ".generation-started.json"),
                       {"declaration_sha256": declaration["sha256"], "run_id": declaration["run_id"]})
        with patch.object(tool, "_memory_limit", return_value="RLIMIT_AS"), \
                patch.object(tool, "_peak_memory_mib", return_value=10), \
                patch.object(tool, "_STARTED", time.perf_counter()):
            manifest = tool._worker(declaration_path, root / "artifact", registry)
    if manifest["status"] != "complete":
        raise AssertionError(manifest["failures"])
    if supervisor:
        config = tool._api().DatasetConfig.from_dict(declaration["config"])
        seconds = manifest["generation_cost"]["worker_module_seconds"]
        accounting = tool._journal_accounting(root / "artifact/attempts.jsonl", config.teacher.max_iterations,
                     {"run_id": declaration["run_id"], "declaration_sha256": declaration["sha256"]})
        immutable_json(root / "artifact/supervisor.json", tool._seal({
            "schema": "nca-ik-dataset-supervisor/v1", "declaration_sha256": declaration["sha256"],
            "status": "complete", "checks": dict.fromkeys(tool.SUPERVISOR_CHECKS, True),
            "child_returncode": 0, "child_timed_out": False, "interrupted": False, "execution_failure": None,
            "child_whole_process_seconds": seconds + .01, "aggregate_through_readback_seconds": seconds + .02,
            "max_wall_seconds": config.budget.max_wall_seconds, "final_receipt_write_excluded_from_elapsed": True,
            "journal": accounting, "manifest_file_sha256": sha256_file(root / "artifact/manifest.json"),
            "finished_utc": "fixture-only"}))
    return declaration, manifest


class ArtifactLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_directory = tempfile.TemporaryDirectory()
        cls.fixture_root = Path(cls.fixture_directory.name)
        tiny_artifact(cls.fixture_root)
        tool.verify_artifact(cls.fixture_root / "artifact")

    @classmethod
    def tearDownClass(cls):
        cls.fixture_directory.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registry = self.root / "registry"
        self.artifact = self.root / "artifact"
        shutil.copytree(self.fixture_root / "registry", self.registry)
        shutil.copytree(self.fixture_root / "artifact", self.artifact)
        self.manifest = tool._read(self.artifact / "manifest.json", sealed=True)
        self.rows = {p: [tool._api().GeometricSample.from_dict(value)
                         for shard in self.manifest["shards"] if shard["partition"] == p
                         for value in tool._jsonl_rows(self.artifact / shard["path"], shard["rows"])]
                     for p in tool.PARTITIONS}
        # Exercise a sealed lifecycle with the authentic tiny artifact, without
        # running a prospective population. All integrity checks remain real;
        # only the fixture/development classification is changed at this seam.
        original_verify = tool.verify_artifact
        def sealed_fixture(*args, **kwargs):
            return {**original_verify(*args, **kwargs), "final_state": "sealed"}
        self.verification = patch.object(tool, "verify_artifact", side_effect=sealed_fixture)
        self.verification.start()

    def tearDown(self):
        self.verification.stop()
        self.temporary.cleanup()

    @property
    def opening_path(self):
        return self.registry / (self.manifest["final_suite_identity_sha256"] + ".final-opened.json")

    def test_inspection_does_not_expose_final_rows(self):
        result = tool.inspect(self.artifact, self.registry)
        self.assertEqual(result["final_state"], "sealed")
        self.assertFalse(result["raw_final_rows_returned"])
        self.assertNotIn('"target_root"', canonical_json(result))

    def test_train_and_tune_are_readable_but_final_is_sealed(self):
        for partition in ("training", "tuning"):
            self.assertEqual(tool.read_partition(self.artifact, partition, self.registry), self.rows[partition])
        for development in (False, True):
            with self.assertRaisesRegex(tool.DatasetToolError, "final_suite_sealed"):
                tool.read_partition(self.artifact, "final", self.registry, development=development)

    def test_opening_consumes_final_before_export_and_copies_cannot_reset_it(self):
        receipt = tool.open_final(self.artifact, self.registry, "experiment-one", ["1" * 64, "2" * 64], self.root / "opened")
        self.assertEqual(receipt["future_experiments_status"], "development_only")
        self.assertEqual((self.root / "opened/final-00000.jsonl").read_bytes(), (self.artifact / "final-00000.jsonl").read_bytes())
        copied = self.root / "copied"
        shutil.copytree(self.artifact, copied)
        with self.assertRaises(ValueError):
            tool.open_final(copied, self.registry, "experiment-two", ["3" * 64], self.root / "opened-again")
        with self.assertRaises(tool.DatasetToolError):
            tool.read_partition(copied, "final", self.registry)
        self.assertEqual(tool.read_partition(copied, "final", self.registry, development=True), self.rows["final"])
        self.assertEqual(tool.inspect(copied, self.registry)["final_state"], "opened_development_for_later_experiments")

    def test_failed_export_still_consumes_opening(self):
        with patch.object(tool, "immutable_bytes", side_effect=OSError("fixture write failure")):
            with self.assertRaises(OSError):
                tool.open_final(self.artifact, self.registry, "experiment-one", ["1" * 64], self.root / "failed-open")
        self.assertTrue(self.opening_path.is_file())
        with self.assertRaises(ValueError):
            tool.open_final(self.artifact, self.registry, "experiment-two", ["2" * 64], self.root / "retry")

    def test_opening_requires_frozen_candidates(self):
        for hashes in ([], ["not-a-hash"], ["1" * 64, "1" * 64]):
            with self.subTest(hashes=hashes), self.assertRaises(tool.DatasetToolError):
                tool.open_final(self.artifact, self.registry, "experiment", hashes, self.root / "bad-open")
        self.assertFalse(self.opening_path.exists())

    def test_other_registry_is_not_a_fresh_final_suite(self):
        different = self.root / "other-registry"
        tool._registry(different, create=True)
        with self.assertRaisesRegex(tool.DatasetToolError, "registry_identity_mismatch"):
            tool.inspect(self.artifact, different)
        with self.assertRaisesRegex(tool.DatasetToolError, "registry_identity_mismatch"):
            tool.open_final(self.artifact, different, "experiment", ["1" * 64], self.root / "bad-open")

    def test_shard_mutation_and_missing_file_are_rejected(self):
        path = self.artifact / "final-00000.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(tool.DatasetToolError, "artifact_bytes_mismatch"):
            tool.inspect(self.artifact, self.registry)
        path.unlink()
        with self.assertRaisesRegex(tool.DatasetToolError, "artifact_bytes_mismatch"):
            tool.verify_artifact(self.artifact)

    def test_supervisor_overrun_cannot_be_inspected_as_complete(self):
        immutable_json(self.artifact / "supervisor-overrun.json", {"status": "incomplete"})
        with self.assertRaisesRegex(tool.DatasetToolError, "generation_not_complete"):
            tool.inspect(self.artifact, self.registry)

    def test_corrupted_opening_does_not_unlock_final(self):
        immutable_json(self.opening_path, {"fake": True})
        with self.assertRaises(tool.DatasetToolError):
            tool.read_partition(self.artifact, "final", self.registry, development=True)


class AttemptAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "journal.jsonl"

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, events, tail=b""):
        self.path.write_bytes(b"".join((canonical_json(event) + "\n").encode() for event in events) + tail)

    def test_incomplete_teacher_is_charged_its_reserved_iterations(self):
        self.write([
            {"event": "allocation_started", "run_id": "fixture", "declaration_sha256": "a" * 64},
            {"event": "teacher_attempt_started", "index": 0, "sample_id": "one", "reserved_iterations": 64},
            {"event": "teacher_attempt_finished", "index": 0, "sample_id": "one", "iterations": 3},
            {"event": "teacher_attempt_started", "index": 1, "sample_id": "two", "reserved_iterations": 64},
        ])
        result = tool._journal_accounting(self.path, 64)
        self.assertTrue(result["valid_prefix"])
        self.assertEqual(result["started"], 2)
        self.assertEqual(result["finished"], 1)
        self.assertEqual(result["unknown_attempts"], 1)
        self.assertEqual(result["teacher_iterations_conservative"], 67)

    def test_truncated_tail_is_not_complete_or_exact(self):
        self.write([{"event": "allocation_started"}], b'{"event":"teacher_attempt_started"')
        result = tool._journal_accounting(self.path, 64)
        self.assertFalse(result["valid_prefix"])
        self.assertTrue(result["unknown_additional_work"])

    def test_wrong_declaration_binding_is_rejected(self):
        self.write([{"event": "allocation_started", "run_id": "wrong", "declaration_sha256": "a" * 64}])
        result = tool._journal_accounting(self.path, 64, {"run_id": "right", "declaration_sha256": "a" * 64})
        self.assertFalse(result["valid_prefix"])

    def test_invalid_start_does_not_crash_failure_readback(self):
        self.path.write_bytes(b"[]\n")
        self.assertFalse(tool._journal_accounting(self.path, 64)["valid_prefix"])

    def test_out_of_order_completion_is_rejected(self):
        self.write([{"event": "allocation_started"},
                    {"event": "teacher_attempt_finished", "index": 0, "sample_id": "one", "iterations": 0}])
        self.assertFalse(tool._journal_accounting(self.path, 64)["valid_prefix"])


class DeclarationAdmissionTests(unittest.TestCase):
    def test_rehashed_profile_threshold_split_and_environment_tampering_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Path(__file__).parents[1] / "configs/ik/datasets/tiny_v1.json"
            with patch.object(tool, "_source", return_value=fixture_source()):
                declaration = tool.declare(config, root / "declaration.json", root / "registry")
                admitted, _, _ = tool._declaration(root / "declaration.json", root / "registry")
                self.assertEqual(canonical_json(admitted), canonical_json(declaration))
                different_config = tool._api().DatasetConfig.from_dict({**declaration["config"], "batch_size": 32})
                different_plan = tool._api().build_split_plan(different_config)
                changes = (
                    {"profile_sha256": "0" * 64}, {"environment_sha256": "0" * 64},
                    {"split_plan_sha256": "0" * 64}, {"acceptance_file_sha256": "0" * 64},
                    {"acceptance": {**declaration["acceptance"], "teacher_solved_residual_max_cells": 100}},
                    {"attempts": True}, {"retry_or_resume": True}, {"unexpected_override": True},
                    {"config": different_config.to_dict(), "config_sha256": different_config.sha256,
                     "split_plan": different_plan, "split_plan_sha256": sha256_json(different_plan),
                     "schedule": tool._schedule(different_config)},
                )
                for i, change in enumerate(changes):
                    mutated = {key: value for key, value in declaration.items() if key != "sha256"}
                    mutated.update(change)
                    path = root / f"changed-{i}.json"
                    immutable_json(path, tool._seal(mutated))
                    with self.subTest(change=change), self.assertRaises(tool.DatasetToolError):
                        tool._declaration(path, root / "registry")


class WorkerFailureTests(unittest.TestCase):
    def test_hard_memory_admission_precedes_source_and_schedule_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = tool._api().DatasetConfig.from_json((tool.ROOT / "configs/ik/datasets/tiny_v1.json").read_text())
            immutable_json(root / "declaration.json", tool._seal({"config": config.to_dict()}))
            sequence = []
            def limited(_):
                sequence.append("memory_cap")
                return "RLIMIT_AS"
            def admission(*_):
                sequence.append("source_and_schedule")
                raise tool.DatasetToolError("fixture_stop_before_generation")
            with patch.object(tool, "_memory_limit", side_effect=limited), \
                    patch.object(tool, "_declaration", side_effect=admission):
                with self.assertRaisesRegex(tool.DatasetToolError, "fixture_stop_before_generation"):
                    tool._worker(root / "declaration.json", root / "output", root / "registry")
            self.assertEqual(sequence, ["memory_cap", "source_and_schedule"])

    def test_tiny_cli_worker_publishes_checked_shards_and_training_normalizer(self):
        from nca.ik.data import DatasetConfig, Normalization
        config = DatasetConfig.from_json((Path(__file__).parents[1] / "configs/ik/datasets/tiny_v1.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry"
            with patch.object(tool, "_source", return_value=fixture_source()):
                declaration = tool.declare(tool.ROOT / "configs/ik/datasets/tiny_v1.json", root / "declaration.json", registry)
                immutable_json(registry / (declaration["sha256"] + ".generation-started.json"),
                               {"declaration_sha256": declaration["sha256"], "run_id": declaration["run_id"]})
                output = root / "artifact"
                with patch.object(tool, "_memory_limit", return_value="RLIMIT_AS"), \
                        patch.object(tool, "_peak_memory_mib", return_value=10), \
                        patch.object(tool, "_STARTED", time.perf_counter()), redirect_stdout(io.StringIO()):
                    code = tool.main(["_worker", "--declaration", str(root / "declaration.json"),
                                      "--output", str(output), "--registry", str(registry)])
            self.assertEqual(code, 0)
            manifest = tool.verify_artifact(output, require_supervisor=False)
            self.assertEqual(manifest["counts"]["completed"], config.expected_tasks)
            self.assertEqual(manifest["counts"]["unknown_attempts"], 0)
            self.assertEqual(sum(shard["rows"] for shard in manifest["shards"]), config.expected_tasks)
            self.assertEqual(manifest["final_state"], "development_fixture")
            normalization = Normalization.from_dict(tool._read(output / "normalization.json"))
            self.assertEqual(normalization.partition, "training")
            self.assertGreaterEqual(normalization.count, 72 * config.points_per_bin[0]
                                    + config.trajectories_per_partition * config.trajectory_steps)
            self.assertLessEqual(normalization.count, declaration["schedule"]["partitions"]["training"])
            coverage = tool._read(output / "coverage.json")
            self.assertFalse(coverage["accepted_final_population_minima_met"])
            self.assertFalse(coverage["final_scoring_opened"])

    def test_failed_teacher_retains_rows_and_consumes_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry"
            _, registry_id = tool._registry(registry, create=True)
            key = "a" * 64
            declaration = {"sha256": key, "run_id": "fixture-worker", "registry_id": registry_id,
                           "split_plan_sha256": "b" * 64,
                           "schedule": {"tasks": 3, "final_suite_identity_sha256": "f" * 64}}
            immutable_json(registry / (key + ".generation-started.json"),
                           {"declaration_sha256": key, "run_id": declaration["run_id"]})
            tasks = [SimpleNamespace(id=str(i), partition=partition, sha256=str(i) * 64)
                     for i, partition in enumerate(tool.PARTITIONS)]
            sample = SimpleNamespace(task=tasks[0], label=SimpleNamespace(teacher=SimpleNamespace(iterations=2, status="solved")),
                                     to_dict=lambda: {"task": "zero", "raw_teacher": "retained"}, sha256="d" * 64)
            api = SimpleNamespace(iter_tasks=lambda config: iter(tasks),
                                  generate_sample=lambda task, config: sample if task.id == "0" else (_ for _ in ()).throw(RuntimeError("fixture")))
            config = SimpleNamespace(budget=SimpleNamespace(max_memory_mib=512, max_wall_seconds=30,
                publication_reserve_seconds=5, max_unique_tasks=3, max_teacher_attempts=3, max_teacher_iterations=192),
                teacher=SimpleNamespace(max_iterations=64), batch_size=64, mode="fixture", sha256="c" * 64)
            output = root / "output"
            with patch.object(tool, "_declaration", return_value=(declaration, config, registry)), \
                    patch.object(tool, "_api", return_value=api), \
                    patch.object(tool, "_worker_admission", return_value="RLIMIT_AS"), \
                    patch.object(tool, "_peak_memory_mib", return_value=10), patch.object(tool, "_STARTED", time.perf_counter()):
                result = tool._worker(root / "declaration.json", output, registry)
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["counts"]["attempted"], 2)
                self.assertEqual(result["counts"]["completed"], 1)
                self.assertEqual(result["counts"]["unknown_attempts"], 1)
                self.assertEqual(result["counts"]["teacher_iterations_conservative"], 66)
                self.assertIn(b"raw_teacher", (output / "pending.jsonl").read_bytes())
                self.assertEqual(result["shards"][0]["rows"], 1)
                with self.assertRaises(ValueError):
                    tool._worker(root / "copied-declaration.json", root / "different-output", registry)

    def test_startup_failure_leaves_supervisor_receipt_and_consumed_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry"
            _, registry_id = tool._registry(registry, create=True)
            declaration = {"sha256": "a" * 64, "run_id": "fixture", "registry_id": registry_id}
            config = SimpleNamespace(budget=SimpleNamespace(max_wall_seconds=30, publication_reserve_seconds=5),
                                     teacher=SimpleNamespace(max_iterations=64), expected_tasks=3)
            with patch.object(tool, "_declaration", return_value=(declaration, config, registry)), \
                    patch.object(tool.subprocess, "Popen", side_effect=OSError("fixture startup failure")), \
                    patch.object(tool, "_STARTED", time.perf_counter()):
                result = tool.generate(root / "declaration.json", root / "output", registry)
                self.assertEqual(result["status"], "incomplete")
                self.assertTrue(result["journal"]["unknown_additional_work"])
                self.assertTrue((root / "output/supervisor.json").is_file())
                with self.assertRaises(ValueError):
                    tool.generate(root / "copied-declaration.json", root / "another-output", registry)


if __name__ == "__main__":
    unittest.main()
