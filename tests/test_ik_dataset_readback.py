"""Semantic readback of a tiny real worker result, including rehashed attacks.

The fixture is generated once in a disposable child process. Only clean-source
admission is mocked; task construction, the teacher, hard memory admission,
publication, and every readback validator run normally. It is development
conformance data and provides no production, training, or physical evidence.
"""

from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from nca.contracts import canonical_json, sha256_json, strict_json_loads
from nca.ik.data import DatasetConfig


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ik_dataset_readback_tests", ROOT / "tools/ik_dataset.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


BUILD_FIXTURE = r'''
import importlib.util
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from nca.contracts import canonical_json, sha256_json
from nca.runs import immutable_json, sha256_file, utc_now

root, destination = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("ik_dataset_real_fixture", root / "tools/ik_dataset.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
relative = "configs/ik/datasets/tiny_v1.json"
snapshot = tool.source_snapshot(root, require_clean=False)
files = dict(snapshot["files"])
for name in (relative, tool.ACCEPTANCE, "tools/ik_dataset.py", "tools/ik_dataset_supervise.py"):
    files[name] = sha256_file(root / name)
source = {"revision": snapshot["revision"],
          "tree": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True).strip(),
          "files": files, "source_sha256": sha256_json(files)}
registry = destination / "registry"
declaration_path = destination / "declaration.json"
with patch.object(tool, "_source", return_value=(source, relative)):
    declaration = tool.declare(root / relative, declaration_path, registry)
    immutable_json(registry / (declaration["sha256"] + ".generation-started.json"),
                   {"declaration_sha256": declaration["sha256"], "run_id": declaration["run_id"],
                    "started_utc": utc_now(), "attempt": 1, "retry_or_resume": False})
    manifest = tool._worker(declaration_path, destination / "artifact", registry)
print(canonical_json({"status": manifest["status"], "failures": manifest["failures"],
                      "counts": manifest["counts"]}))
raise SystemExit(0 if manifest["status"] == "complete" else 1)
'''


class SemanticArtifactReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        result = subprocess.run([sys.executable, "-c", BUILD_FIXTURE, str(ROOT), str(cls.base)],
                                cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError("tiny real worker did not finish: " + result.stdout + result.stderr)
        cls.original = cls.base / "artifact"
        cls.config = DatasetConfig.from_json((ROOT / "configs/ik/datasets/tiny_v1.json").read_bytes())
        cls.manifest = tool.verify_artifact(cls.original, require_supervisor=False)

    def setUp(self):
        self.case = tempfile.TemporaryDirectory()
        self.addCleanup(self.case.cleanup)
        self.artifact = Path(self.case.name) / "artifact"
        shutil.copytree(self.original, self.artifact)
        self.value = deepcopy(self.manifest)

    def rows(self, name):
        return [strict_json_loads(line) for line in (self.artifact / name).read_bytes().splitlines()]

    def write_rows(self, name, rows):
        (self.artifact / name).write_bytes(b"".join((canonical_json(row) + "\n").encode() for row in rows))

    def write_json(self, name, value):
        (self.artifact / name).write_text(canonical_json(value) + "\n")

    def reseal(self):
        """Make every declared byte count/hash honest after the deliberate edit."""
        for item in [*self.value["artifacts"], *self.value["shards"]]:
            item.update(tool._artifact(self.artifact / item["path"], self.artifact))
        self.value.pop("sha256", None)
        self.value = tool._seal(self.value)
        self.write_json("manifest.json", self.value)

    def refused(self):
        self.reseal()
        with self.assertRaises(tool.DatasetToolError) as caught:
            tool.verify_artifact(self.artifact, require_supervisor=False)
        self.assertNotEqual(caught.exception.code, "artifact_bytes_mismatch")
        self.assertNotEqual(caught.exception.code, "content_digest_mismatch")

    def test_intact_real_fixture_has_checked_rows_and_a_real_worker_memory_limit(self):
        manifest = tool.verify_artifact(self.artifact, require_supervisor=False)
        self.assertEqual(manifest["counts"]["completed"], self.config.expected_tasks)
        self.assertEqual(manifest["counts"]["attempted"], self.config.expected_tasks)
        self.assertEqual(manifest["counts"]["unknown_attempts"], 0)
        self.assertGreater(manifest["counts"]["teacher_iterations_actual_lower_bound"], 0)
        self.assertEqual(manifest["generation_cost"]["hard_memory_limit_mib"], self.config.budget.max_memory_mib)
        self.assertIn(manifest["generation_cost"]["memory_limit_method"],
                      ("RLIMIT_AS", "Windows Job Object process committed-memory limit"))
        self.assertEqual(manifest["final_state"], "development_fixture")
        self.assertEqual(len(self.rows("pending.jsonl")), self.config.expected_tasks)

    def test_rehashed_manifest_cannot_omit_any_required_artifact(self):
        for name in ("declaration.json", "attempts.jsonl", "pending.jsonl", "normalization.json", "coverage.json"):
            with self.subTest(name=name):
                self.value = deepcopy(self.manifest)
                self.value["artifacts"] = [item for item in self.value["artifacts"] if item["path"] != name]
                self.refused()

    def test_rehashed_manifest_cannot_omit_a_shard(self):
        self.value["shards"].pop()
        self.refused()

    def test_an_unlisted_shard_cannot_hide_beside_a_complete_inventory(self):
        shard = self.value["shards"][0]
        shutil.copyfile(self.artifact / shard["path"], self.artifact / "training-99999.jsonl")
        self.refused()

    def test_rehashed_partition_label_cannot_reclassify_rows(self):
        training = next(item for item in self.value["shards"] if item["partition"] == "training")
        training["partition"] = "final"
        self.refused()

    def test_swapped_partition_contents_are_refused_even_with_matching_shard_hashes(self):
        training = next(item for item in self.value["shards"] if item["partition"] == "training")
        tuning = next(item for item in self.value["shards"] if item["partition"] == "tuning")
        training_rows, tuning_rows = self.rows(training["path"]), self.rows(tuning["path"])
        for item, rows in ((training, tuning_rows), (tuning, training_rows)):
            self.write_rows(item["path"], rows)
            item["rows"] = len(rows)
            item["sample_ids_sha256"] = sha256_json([row["task"]["id"] for row in rows])
        self.refused()

    def test_duplicate_cannot_replace_a_scheduled_row_in_shard_pending_and_journal(self):
        shard = self.value["shards"][0]
        rows = self.rows(shard["path"])
        omitted_id = rows[1]["task"]["id"]
        rows[1] = deepcopy(rows[0])
        self.write_rows(shard["path"], rows)
        shard["sample_ids_sha256"] = sha256_json([row["task"]["id"] for row in rows])
        pending = self.rows("pending.jsonl")
        index = next(i for i, row in enumerate(pending) if row["task"]["id"] == omitted_id)
        pending[index] = deepcopy(rows[0])
        self.write_rows("pending.jsonl", pending)
        journal = self.rows("attempts.jsonl")
        for event in journal:
            if event.get("index") != index:
                continue
            event["sample_id"] = rows[0]["task"]["id"]
            if event["event"] == "teacher_attempt_started":
                event["task_sha256"] = sha256_json(rows[0]["task"])
            else:
                event["sample_sha256"] = sha256_json(rows[0])
                event["iterations"] = rows[0]["label"]["teacher"]["iterations"]
                event["teacher_status"] = rows[0]["label"]["teacher"]["status"]
        self.write_rows("attempts.jsonl", journal)
        total = sum(row["label"]["teacher"]["iterations"] for row in pending)
        self.value["counts"]["teacher_iterations_actual_lower_bound"] = total
        self.value["counts"]["teacher_iterations_conservative"] = total
        self.value["all_samples_sha256"] = sha256_json([sha256_json(row) for row in pending])
        self.refused()

    def test_pending_result_cannot_diverge_from_the_published_shard(self):
        rows = self.rows("pending.jsonl")
        rows[0]["label"]["teacher"]["elapsed_seconds"] += 0.001
        self.write_rows("pending.jsonl", rows)
        self.value["all_samples_sha256"] = sha256_json([sha256_json(row) for row in rows])
        self.refused()

    def test_pending_order_must_match_the_frozen_schedule(self):
        rows = self.rows("pending.jsonl")
        rows[0], rows[1] = rows[1], rows[0]
        self.write_rows("pending.jsonl", rows)
        self.value["all_samples_sha256"] = sha256_json([sha256_json(row) for row in rows])
        self.refused()

    def test_journal_started_input_is_bound_to_the_actual_task(self):
        rows = self.rows("attempts.jsonl")
        rows[1]["task_sha256"] = "0" * 64
        self.write_rows("attempts.jsonl", rows)
        self.refused()

    def test_journal_finished_output_status_and_work_are_bound_to_the_actual_result(self):
        original = self.rows("attempts.jsonl")
        finished = next(i for i, row in enumerate(original) if row["event"] == "teacher_attempt_finished")
        changes = {"sample_sha256": "0" * 64, "teacher_status": "best_effort",
                   "iterations": original[finished]["iterations"] + 1}
        for field, changed in changes.items():
            with self.subTest(field=field):
                rows = deepcopy(original)
                rows[finished][field] = changed
                self.write_rows("attempts.jsonl", rows)
                self.value = deepcopy(self.manifest)
                if field == "iterations":
                    self.value["counts"]["teacher_iterations_actual_lower_bound"] += 1
                    self.value["counts"]["teacher_iterations_conservative"] += 1
                self.refused()

    def test_rehashed_normalization_must_equal_the_training_inputs_fit(self):
        original = tool._read(self.artifact / "normalization.json")
        for field in ("mean", "scale", "count", "training_inputs_sha256"):
            with self.subTest(field=field):
                value = deepcopy(original)
                if field in ("mean", "scale"):
                    value[field][0] += 0.5
                elif field == "count":
                    value[field] += 1
                else:
                    value[field] = "0" * 64
                self.write_json("normalization.json", value)
                self.refused()

    def test_rehashed_coverage_cannot_change_denominators_failures_or_qualification(self):
        original = tool._read(self.artifact / "coverage.json")
        for mutation in ("observed", "failures", "qualified", "denominator"):
            with self.subTest(mutation=mutation):
                value = deepcopy(original)
                if mutation == "observed":
                    value["observed_samples"] -= 1
                elif mutation == "failures":
                    value["partitions"]["training"]["retained_teacher_failures"] = 0
                elif mutation == "qualified":
                    value["accepted_final_population_minima_met"] = True
                else:
                    value["partitions"]["final"]["nominal_denominator"] -= 1
                self.write_json("coverage.json", value)
                self.refused()

    def test_rehashed_declaration_cannot_omit_executed_generator_source(self):
        declaration = tool._read(self.artifact / "declaration.json")
        declaration["source"]["files"].pop("src/nca/ik/data.py")
        declaration["source"]["source_sha256"] = sha256_json(declaration["source"]["files"])
        declaration.pop("sha256")
        declaration = tool._seal(declaration)
        self.write_json("declaration.json", declaration)
        self.value["declaration_sha256"] = declaration["sha256"]
        journal = self.rows("attempts.jsonl")
        journal[0]["declaration_sha256"] = declaration["sha256"]
        self.write_rows("attempts.jsonl", journal)
        self.refused()

    def supervisor(self):
        declaration = tool._read(self.artifact / "declaration.json", sealed=True)
        seconds = self.value["generation_cost"]["worker_module_seconds"]
        accounting = tool._journal_accounting(self.artifact / "attempts.jsonl", self.config.teacher.max_iterations,
                     {"run_id": declaration["run_id"], "declaration_sha256": declaration["sha256"]})
        return {
            "schema": "nca-ik-dataset-supervisor/v1", "declaration_sha256": declaration["sha256"],
            "status": "complete", "checks": dict.fromkeys(tool.SUPERVISOR_CHECKS, True),
            "child_returncode": 0, "child_timed_out": False, "interrupted": False, "execution_failure": None,
            "child_whole_process_seconds": seconds + 0.01, "aggregate_through_readback_seconds": seconds + 0.02,
            "max_wall_seconds": self.config.budget.max_wall_seconds, "final_receipt_write_excluded_from_elapsed": True,
            "journal": accounting, "manifest_file_sha256": tool.sha256_file(self.artifact / "manifest.json"),
            "finished_utc": "development-conformance-only",
        }

    def test_rehashed_supervisor_cannot_hide_a_failed_check(self):
        supervisor = self.supervisor()
        self.write_json("supervisor.json", tool._seal(supervisor))
        tool.verify_artifact(self.artifact)
        supervisor["checks"]["counts_match_journal"] = False
        self.write_json("supervisor.json", tool._seal(supervisor))
        with self.assertRaisesRegex(tool.DatasetToolError, "generation_not_complete"):
            tool.verify_artifact(self.artifact)

    def test_rehashed_supervisor_journal_must_equal_retained_work(self):
        supervisor = self.supervisor()
        supervisor["journal"]["teacher_iterations_actual_lower_bound"] += 1
        supervisor["journal"]["teacher_iterations_conservative"] += 1
        self.write_json("supervisor.json", tool._seal(supervisor))
        with self.assertRaisesRegex(tool.DatasetToolError, "generation_not_complete"):
            tool.verify_artifact(self.artifact)


if __name__ == "__main__":
    unittest.main()
