"""Synthetic staged-evidence controls; these never establish game qualification."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    from nca.contracts import ContractError, sha256_json
    from nca.export.build import immutable_json
    from nca.export.candidate import (BUDGET, CANDIDATE, DECLARATION_SCHEMA, FIXED_POLICY_SHA256,
        LEARNER_CONFIG, LEARNER_REVISION, LEARNER_SOURCES, MANIFEST_SCHEMA, ORIGIN_NAMES,
        ORIGIN_RECEIPT, ORIGIN_SCHEMA, ORIGIN_STATUS, SCREEN_SHA256, SUMMARY_SCHEMA,
        _verify_interruption_index, validate_declaration, verify_evaluation, verify_metrics)


def declaration():
    return {"schema": DECLARATION_SCHEMA, "id": "synthetic-contract-only", "candidate": CANDIDATE,
            "policy_sha256": FIXED_POLICY_SHA256, "learner_revision": LEARNER_REVISION,
            "learner_sources_sha256": LEARNER_SOURCES, "learner_config_sha256": LEARNER_CONFIG,
            "export_criteria_sha256": SCREEN_SHA256, "training_status": ORIGIN_STATUS,
            "training_receipt_file": ORIGIN_RECEIPT, "training_receipt_schema": ORIGIN_SCHEMA,
            "original_termination_confirmed": False, "training_files": {n: "a"*64 for n in ORIGIN_NAMES},
            "budget": deepcopy(BUDGET),
            "scenario_selection": "all original eight evaluation families, seeds and goals",
            "scope": "single fixed candidate evaluation; no efficacy comparison or deployment qualification"}


def write_record(path, value):
    value = deepcopy(value)
    value.pop("sha256", None)
    value["sha256"] = sha256_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return value, hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(HAS_NUMPY, "optional NumPy CPU export dependency unavailable")
class CandidateDeclarationTests(unittest.TestCase):
    def test_task_metrics_require_complete_finite_typed_cost_and_outcome_fields(self):
        metrics = {"success": False, "administratively_capped": False, "integrity_failure": False,
                   "purposeful_launch": False, "owner_feel_accepted": False, "failed_admissions": 0,
                   "elapsed_ticks": 240, "primary_edges": 0, "ordinary_trim_reversals": 0,
                   "success_seconds": None, "response_seconds": None, "return": -.7,
                   "final_distance_cells": 24., "best_distance_cells": 24., "capped_completion_seconds": 4.,
                   "peak_toward_speed_cells_per_second": 0., "overshoot_cells": 0., "airborne_fraction": 0.,
                   "ordinary_trim_reversals_per_second": 0., "trim_variation_per_second": 0.,
                   "mean_absolute_angle_radians": 0., "peak_angular_speed": 0., "inference_p50_ms": .1,
                   "inference_p95_ms": .2, "inference_max_ms": .3, "reason": "task_deadline"}
        verify_metrics(metrics, decisions=40, ticks=240)
        changes = (("inference_p95_ms", None), ("return", "unmeasured"), ("failed_admissions", False),
                   ("elapsed_ticks", 239), ("inference_max_ms", .15), ("airborne_fraction", 1.1),
                   ("primary_edges", 241), ("return", float("nan")), ("reason", "arrival"))
        for name, value in changes:
            other = dict(metrics)
            if value is None:
                other.pop(name)
            else:
                other[name] = value
            with self.subTest(field=name), self.assertRaises(ContractError):
                verify_metrics(other, decisions=40, ticks=240)

    def test_candidate_budget_source_and_incomplete_origin_cannot_be_redeclared(self):
        validate_declaration(declaration())
        changes = (("policy_sha256", "b"*64), ("training_status", "capped"),
                   ("training_status", "completed"), ("original_termination_confirmed", True),
                   ("training_receipt_file", "summary.json"), ("learner_revision", "b"*40),
                   ("export_criteria_sha256", "b"*64))
        for key, value in changes:
            other = declaration()
            other[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ContractError):
                validate_declaration(other)
        for key, value in (("maximum_decisions", 321), ("maximum_ticks", 1921),
                           ("maximum_wall_seconds", 301), ("attempts", True), ("automatic_retry", True)):
            other = declaration()
            other["budget"][key] = value
            with self.subTest(budget=key), self.assertRaises(ContractError):
                validate_declaration(other)

    def interruption_fixture(self, root):
        row = {"episode_id": "synthetic-prefix", "artifact": "episodes/synthetic.json.gz",
               "artifact_sha256": hashlib.sha256(b"synthetic episode bytes").hexdigest(),
               "decisions": 1, "metrics": {"elapsed_ticks": 6}}
        files = {"episodes/synthetic.json.gz": b"synthetic episode bytes",
                 "metrics/synthetic.json": json.dumps(row).encode(),
                 "checkpoints/synthetic.pt": b"synthetic checkpoint bytes",
                 "learning/residual-synthetic.json": b'{"updates":1,"learner_seconds":0.1}'}
        for name, data in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        inventory = {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "mtime_ns": 1}
                     for name, data in files.items()}
        fence = {"schema": "nca-interrupted-tool-session-fencing-receipt/v1",
                 "original_termination_confirmed": False, "original_exit_code": None,
                 "operation": "write_stdin with Ctrl-C to the original known session handle",
                 "exact_result": "write_stdin failed: Unknown process id 32967"}
        _, fence_hash = write_record(root / "interruption-fence.json", fence)
        manifest = {"config": {"budget": {"maximum_wall_seconds": 1800}}}
        index = {"schema": ORIGIN_SCHEMA, "status": ORIGIN_STATUS, "original_summary_present": False,
                 "original_termination_confirmed": False, "original_evaluation_opened_receipt_present": False,
                 "unconfirmed_pending_outcome": True, "retained_record_counts_are_original_run_totals": False,
                 "original_exit_code": None, "original_measured_wall_seconds": None,
                 "original_total_attempted_decisions": None, "original_total_consumed_ticks": None,
                 "inventory_before_and_after_copy_equal": True, "snapshot_bytes_equal_original": True,
                 "evaluation": {"complete": False, "efficacy_comparison": False}, "runtime_qualified": False,
                 "declared_budget": manifest["config"]["budget"], "learner_revision": LEARNER_REVISION,
                 "receipt_origin": "post-interruption read-only inventory of an unchanged retained prefix",
                 "original_files": inventory, "original_files_sha256": sha256_json(inventory),
                 "fencing_receipt_file": "interruption-fence.json", "fencing_receipt_file_sha256": fence_hash,
                 "results": [row], "checkpoints": [{"file": "synthetic.pt", **inventory["checkpoints/synthetic.pt"]}],
                 "ppo_updates": [{"file": "learning/residual-synthetic.json", "updates": 1, "learner_seconds": .1}],
                 "retained_record_counts": {"episodes": 1, "decisions": 1, "consumed_ticks": 6,
                                            "checkpoint_files": 1, "completed_ppo_update_ledgers": 1}}
        index, index_hash = write_record(root / ORIGIN_RECEIPT, index)
        declared = {"training_files": {**{n: r["sha256"] for n, r in inventory.items()},
                    ORIGIN_RECEIPT: index_hash, "interruption-fence.json": fence_hash}}
        return index, declared, manifest

    def test_interruption_index_preserves_unknown_totals_and_actual_prefix_members(self):
        changes = (None, "invented_total", "missing_member", "changed_metric", "fabricated_summary")
        for mutation in changes:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                index, declared, manifest = self.interruption_fixture(root)
                if mutation == "invented_total":
                    index["original_total_attempted_decisions"] = 1
                elif mutation == "missing_member":
                    del index["original_files"]["episodes/synthetic.json.gz"]
                    index["original_files_sha256"] = sha256_json(index["original_files"])
                elif mutation == "changed_metric":
                    index["results"][0]["decisions"] = 2
                elif mutation == "fabricated_summary":
                    (root / "summary.json").write_text('{"status":"capped"}')
                index, declared["training_files"][ORIGIN_RECEIPT] = write_record(root / ORIGIN_RECEIPT, index)
                if mutation is None:
                    _verify_interruption_index(root, index, declared, manifest)
                else:
                    with self.assertRaises(ContractError):
                        _verify_interruption_index(root, index, declared, manifest)

    def evaluation_fixture(self, root, mutation):
        families = [{"id": f"synthetic-family-{i}", "host_seed": i} for i in range(8)]
        training = {"manifest": {"config": {"evaluation_families": families}}}
        sources = {"synthetic-source.py": "a"*64}
        manifest = {"schema": MANIFEST_SCHEMA, "declaration": declaration(),
                    "declaration_content_sha256": sha256_json(declaration()),
                    "phase": "frozen_before_any_candidate_evaluation", "training_status": ORIGIN_STATUS,
                    "original_termination_confirmed": False, "evaluation_source_files": sources,
                    "evaluation_sources_sha256": sha256_json(sources),
                    "policy_execution_revision": LEARNER_REVISION,
                    "policy_execution_sources_sha256": LEARNER_SOURCES}
        manifest, _ = write_record(root / "manifest.json", manifest)
        rows = []
        for family in families:
            name = f"episodes/{family['id']}.json.gz"
            path = root / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"synthetic episode input")
            rows.append({"kind": "bc", "scenario_family": family["id"], "artifact": name,
                         "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if mutation == "missing_family":
            rows.pop()
        elif mutation == "duplicate_family":
            rows[-1] = deepcopy(rows[0])
        worker = {"schema": "nca-fixed-candidate-worker-result/v1", "manifest_sha256": manifest["sha256"],
                  "status": "completed", "errors": [], "host_start_attempts": 1, "new_optimizer_updates": 0,
                  "source_files_unchanged": True, "policy_sha256": FIXED_POLICY_SHA256, "results": rows,
                  "attempted_decisions": 320, "actual_decisions": 320, "actual_ticks": 1920,
                  "unconfirmed_decision_reservations": 0, "unconfirmed_pending_outcome": False}
        if mutation == "unknown_outcome":
            worker["attempted_decisions"] = 321
        worker, worker_hash = write_record(root / "worker-result.json", worker)
        summary = {**{k: worker[k] for k in ("status", "errors", "host_start_attempts", "policy_sha256", "results",
                     "attempted_decisions", "actual_decisions", "actual_ticks")},
                   "schema": SUMMARY_SCHEMA, "manifest_sha256": manifest["sha256"],
                   "complete_eight_family_coverage": True, "new_optimizer_updates": 0,
                   "runtime_qualified": False, "efficacy_comparison": False, "training_status": ORIGIN_STATUS,
                   "original_termination_confirmed": False, "wall_seconds": 10., "source_files_unchanged": True,
                   "training_files_unchanged": True, "worker_result_file_sha256": worker_hash,
                   "worker_invocations": 1, "worker_exit_code": 0, "worker_termination_confirmed": True,
                   "unconfirmed_pending_outcome": False, "unconfirmed_decision_reservations": 0}
        if mutation == "time_overrun":
            summary["wall_seconds"] = 300.001
        elif mutation == "source_drift":
            summary["source_files_unchanged"] = False
        elif mutation == "false_completed_origin":
            summary["training_status"] = "completed"
        elif mutation == "missing_artifact":
            (root / rows[0]["artifact"]).unlink()
        write_record(root / "summary.json", summary)
        return training, sources

    def test_rehashed_evaluation_envelopes_do_not_hide_incomplete_or_over_budget_evidence(self):
        mutations = (None, "missing_family", "duplicate_family", "unknown_outcome", "time_overrun",
                     "source_drift", "false_completed_origin", "missing_artifact")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                training, sources = self.evaluation_fixture(root, mutation)
                # Isolate envelope admission from separately tested typed episode
                # and original checkpoint verification. No host is used here.
                with patch("nca.export.candidate.verify_origin", return_value=training), \
                     patch("nca.export.candidate._verify_evaluation_source_record"), \
                     patch("nca.export.candidate.verify_episode", return_value=SimpleNamespace(decisions=40, consumed_ticks=240)):
                    if mutation is None:
                        self.assertEqual(len(verify_evaluation(root, root)[3]), 11)
                    else:
                        with self.assertRaises(ContractError):
                            verify_evaluation(root, root)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional Torch runner dependency unavailable")
    def test_failed_worker_receipt_preserves_reserved_but_unconfirmed_decision(self):
        from nca.bench.candidate_evaluation import run_candidate_evaluation
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_root = root / "training"
            training_root.mkdir()
            learner = root / "learner"
            learner.mkdir()
            output = training_root / "new-evaluation"
            declared = declaration()
            declared["training_files"] = {}
            declaration_path = root / "declaration.json"
            declaration_path.write_text(json.dumps(declared))
            training = {"summary": {"status": ORIGIN_STATUS}, "manifest": {"source_files": {},
                        "config": {"evaluation_families": [{"id": str(i)} for i in range(8)]}}}

            class FailedReceiptProcess:
                def __init__(self, *args, **kwargs):
                    self.pid = 987654  # Synthetic handle, never signalled.

                def wait(self, timeout=None):
                    manifest = json.loads((output / "manifest.json").read_text())
                    write_record(output / "worker-result.json", {
                        "schema": "nca-fixed-candidate-worker-result/v1", "manifest_sha256": manifest["sha256"],
                        "status": "failed", "errors": [{"code": "synthetic_rpc_failure"}], "results": [],
                        "host_start_attempts": 1, "attempted_decisions": 1, "actual_decisions": 0,
                        "actual_ticks": 0, "source_files_unchanged": True,
                        "policy_sha256": FIXED_POLICY_SHA256, "new_optimizer_updates": 0})
                    return 1

                def poll(self):
                    return 1

            with patch("nca.bench.candidate_evaluation.verify_origin", return_value=training), \
                 patch("nca.bench.candidate_evaluation.committed_sources", return_value=("a"*40, {}, "declaration.json")), \
                 patch("nca.bench.candidate_evaluation._sources", return_value={}), \
                 patch("nca.bench.candidate_evaluation.subprocess.check_output", return_value=LEARNER_REVISION), \
                 patch("nca.bench.candidate_evaluation.subprocess.Popen", FailedReceiptProcess):
                result = run_candidate_evaluation(root=root, training_root=training_root, learner_source=learner,
                    declaration_path=declaration_path, host_path=root / "no-host", godot_path=root / "no-godot",
                    output=output)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["actual_decisions"], 0)
            self.assertEqual(result["unconfirmed_decision_reservations"], 1)
            self.assertTrue(result["unconfirmed_pending_outcome"])


if __name__ == "__main__":
    unittest.main()
