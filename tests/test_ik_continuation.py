"""CPU-only amendment, provenance and refusal conformance; no device workloads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from nca.contracts import sha256_json, canonical_json
from nca.train import ik_continuation as api


class IKContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.old_source = {"revision": api.ORIGINAL_REVISION,
            "source_sha256": api.ORIGINAL_SOURCE_SHA256,
            "files": {"src/nca/ik/data.py": "a"*64,
                      "configs/ik/datasets/runner_v1.json": "b"*64}}
        self.new_source = {"revision": "1"*40, "source_sha256": "c"*64,
            "files": {**self.old_source["files"], **api.UPSTREAM_SOURCE_PINS,
                      "src/nca/train/ik_continuation.py": "d"*64}}
        self.amendment = {"sha256": "e"*64, "source": self.new_source}
        self.review = {"schema": "nca-ik-continuation-admission/v1",
            "amendment_sha256": "e"*64, "source_sha256": "c"*64,
            "reviewers": ["independent-one", "independent-two"],
            "checks": {k: True for k in api.CHECKS},
            "verdict": "admit_declared_CUDA_continuation"}

    def write(self, name, value):
        path = self.root / name
        path.write_text(canonical_json(value), encoding="utf-8")
        return path

    def test_source_delta_preserves_original_contract_and_config_bytes(self):
        self.assertEqual(set(api._source_delta(self.old_source, self.new_source)),
                         {"src/nca/train/ik_continuation.py", *api.UPSTREAM_SOURCE_PINS})
        for mutation in ("config", "missing", "unrelated"):
            value = deepcopy(self.new_source)
            if mutation == "config":
                value["files"]["configs/ik/datasets/runner_v1.json"] = "f"*64
            elif mutation == "missing":
                del value["files"]["src/nca/ik/data.py"]
            else:
                value["files"]["src/nca/train/ik.py"] = "f"*64
            with self.assertRaises(ValueError):
                api._source_delta(self.old_source, value)
        upstream = deepcopy(self.new_source)
        upstream["files"].update(api.UPSTREAM_SOURCE_PINS)
        self.assertEqual(set(api._source_delta(self.old_source, upstream)),
            {"src/nca/train/ik_continuation.py", *api.UPSTREAM_SOURCE_PINS})
        for name in api.UPSTREAM_SOURCE_PINS:
            bad = deepcopy(upstream)
            bad["files"][name] = "0"*64
            with self.assertRaises(ValueError):
                api._source_delta(self.old_source, bad)

    def test_inventory_preserves_old_bytes_while_new_receipts_append(self):
        target = self.run / "retained.json"
        target.write_text("original")
        inventory = api._inventory(self.run)
        (self.run / "new-receipt.json").write_text("new")
        api._baseline_unchanged(self.run, inventory)
        target.write_text("modified")
        with self.assertRaises(ValueError):
            api._baseline_unchanged(self.run, inventory)
        with self.assertRaises(ValueError):
            api._baseline_unchanged(self.run, {"../escape": inventory["retained.json"]})

    def test_two_exact_reviewers_and_true_checks_required(self):
        path = self.write("review.json", self.review)
        self.assertEqual(api._review(path, self.amendment), self.review)
        for mutation in ("duplicate", "pending", "integer", "source", "amendment"):
            value = deepcopy(self.review)
            if mutation == "duplicate": value["reviewers"] = ["one-reviewer", "one-reviewer"]
            if mutation == "pending": value["checks"]["corpus_and_splits"] = False
            if mutation == "integer": value["checks"]["allocation_and_resume"] = 1
            if mutation == "source": value["source_sha256"] = "f"*64
            if mutation == "amendment": value["amendment_sha256"] = "f"*64
            path.write_text(canonical_json(value))
            with self.assertRaises(ValueError):
                api._review(path, self.amendment)

    def test_uncertain_opening_blocks_future_train_even_with_same_source(self):
        amendment_path = self.root / "amendment.json"
        start = self.write("evaluation-supervisor-start-0001.json", {})
        with self.assertRaises(ValueError):
            api._evaluation_outcome_guard(amendment_path)
        api._evaluation_outcome_guard(amendment_path, allow_pending_evaluation=True)
        self.write("evaluation-supervisor-result-0001.json", {
            "unknown_work_blocks_resume": True, "verified_within_charged_wall": False})
        with self.assertRaises(ValueError):
            api._evaluation_outcome_guard(amendment_path, allow_pending_evaluation=True)
        self.write("evaluation-supervisor-result-0001.json", {
            "unknown_work_blocks_resume": False, "verified_within_charged_wall": True})
        api._evaluation_outcome_guard(amendment_path)
        self.write("evaluation-supervisor-overrun.json", {"resume_forbidden": True})
        with self.assertRaises(ValueError):
            api._evaluation_outcome_guard(amendment_path, allow_pending_evaluation=True)

    def test_worker_refuses_metadata_mismatch_before_cuda_or_ledger_attempt(self):
        review_path = self.write("review.json", self.review)
        with patch.object(api, "load_amendment", return_value=({}, self.amendment)), \
                patch.object(api, "validate_runtime", side_effect=ValueError("TITAN mismatch")), \
                patch.object(api, "device_report") as gpu, \
                patch.object(api.IKRunLedger, "open") as ledger:
            with self.assertRaises(ValueError):
                api.worker(self.run, "unused", "a"*64, 41011, "unused", "unused", review_path,
                    started=0., review_sha256=sha256_json(self.review))
            gpu.assert_not_called()
            ledger.assert_not_called()

    def test_execution_identity_names_actual_source_and_original_amendment(self):
        self.assertEqual(api.execution_identity(self.amendment), {
            "amendment_sha256": "e"*64, "source_sha256": "c"*64, "source_commit": "1"*40})

    def test_cpu_prepare_freezes_new_source_and_original_inventory_immutably(self):
        first = self.run / "seed-41011"
        first.mkdir()
        (first / "supervisor-result-0001.json").write_text(canonical_json({
            "verified_within_charged_wall": True, "returncode": 0}))
        declaration = {"sha256": "a"*64, "source": self.old_source,
            "runtime": {"fixture": True}, "gpu_visibility_sha256": "b"*64,
            "dataset": {"unchanged": True}, "plan": {"qualified_environment": {"driver": "581.80"}},
            "allocations": {str(s): {"original": True} for s in api.SEEDS}}
        states = {}
        for seed in api.SEEDS:
            first_seed = seed == 41011
            states[str(seed)] = {"can_resume": True, "active": False, "unresolved_work": False,
                "allocation": {"original": True}, "attempt_count": int(first_seed),
                "completed_optimizer_steps": int(first_seed), "charged": {
                    "sample_presentations": 128 if first_seed else 0,
                    "checkpoints": 2 if first_seed else 0, "tuning_calls": 0, "evaluation_calls": 0}}
        output = self.root / "amendment-output"
        with patch.object(api, "load_declaration", return_value=declaration), \
                patch.object(api, "_source", return_value=self.new_source), \
                patch.object(api, "runtime_expectation", return_value=declaration["runtime"]), \
                patch.object(api, "_titan_visibility", return_value="b"*64), \
                patch.object(api, "read_training_corpus", return_value=object()), \
                patch.object(api, "corpus_identity", return_value=declaration["dataset"]), \
                patch.object(api, "_original_review", return_value={"retained": True}), \
                patch.object(api, "_states", return_value=states), \
                patch.object(api, "allocation_for", return_value={"original": True}), \
                patch.object(api, "device_report", side_effect=AssertionError("CUDA forbidden")):
            result = api.prepare(self.run, "unused", "unused", output)
            saved = json.loads((output / "amendment.json").read_text())
            self.assertEqual(saved["baseline_files"], api._inventory(self.run))
            self.assertEqual(saved["allocations"], declaration["allocations"])
            self.assertEqual(saved["source"], self.new_source)
            self.assertFalse(result["cuda_initialized"])
            self.assertEqual(hashlib.sha256((output / "amendment.json").read_bytes()).hexdigest(),
                             result["amendment_file_sha256"])
            with self.assertRaises(ValueError):
                api.prepare(self.run, "unused", "unused", output)

    def test_import_and_help_do_not_import_torch(self):
        code = "import sys; import nca.train.ik_continuation; assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_supervisor_timeout_retains_uncertainty_and_blocks_reuse(self):
        root = self.run / "seed-41011"
        root.mkdir()
        review_path = self.write("review.json", self.review)
        amendment_path = self.root / "amendment.json"
        declaration = {"sha256": "b"*64, "allocations": {"41011": {"unchanged": True}}}
        before = {"can_resume": True, "attempt_count": 1, "remaining_wall_seconds": 100.,
                  "charged_wall_seconds": 28.}
        after = {"active": True, "unresolved_work": True, "charged_wall_seconds": 28.}
        process = unittest.mock.Mock(returncode=-1)
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture-worker", 1.), None]
        ledger = unittest.mock.Mock()
        ledger.inspect.side_effect = [before, after]
        with patch.object(api, "load_amendment", return_value=(declaration, self.amendment)), \
                patch.object(api, "validate_runtime"), \
                patch.object(api, "identity_for", return_value={}), \
                patch.object(api.IKRunLedger, "open", return_value=ledger), \
                patch.object(api.subprocess, "Popen", return_value=process):
            with self.assertRaises(ValueError):
                api.supervise(self.run, amendment_path, "a"*64, 41011,
                               "unused", "unused", review_path)
        terminal = json.loads((root / "supervisor-result-0002.json").read_text())
        self.assertTrue(terminal["timed_out"])
        self.assertTrue(terminal["unknown_work_blocks_resume"])
        self.assertFalse(terminal["verified_within_charged_wall"])
        self.assertEqual(terminal["consumed_or_reserved_wall_seconds"], 100.)
        process.kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()

# Copyright 2026 techrote. All rights reserved.
