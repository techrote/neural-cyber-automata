"""Software evidence for the authored diagnostic's admission and accounting."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nca.contracts import ContractError
from nca.evaluation.jolt_arrival import (DiagnosticBudget, FAMILIES, diagnostic_cases,
                                       read_declaration, summarize, write_json)

ROOT = Path(__file__).resolve().parents[1]


class ArrivalDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.config = read_declaration(ROOT / "configs/baselines/jolt-arrival-diagnostic-v1.json")
        original = json.loads((ROOT / self.config["case_source"]["path"]).read_text())
        self.cases = diagnostic_cases(original)

    def test_existing_case_seed_pairs_are_complete_and_not_relabelled_as_sealed(self):
        self.assertEqual(len(self.cases), 32)
        self.assertEqual(sum(row["origin"] == "previously_open_training" for row in self.cases), 24)
        self.assertEqual(sum(row["origin"] == "previously_open_evaluation" for row in self.cases), 8)
        self.assertEqual(len({(tuple(row["spawn"]), tuple(row["target"]), row["host_seed"])
                              for row in self.cases}), 32)

    def test_changed_declaration_refuses_before_any_runtime_work(self):
        changed = copy.deepcopy(self.config)
        changed["budget"]["maximum_decisions"] += 1
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "changed.json"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ContractError, "arrival_declaration"):
                read_declaration(path)

    def test_uncertain_request_consumes_budget_and_remains_visible(self):
        limits = {**self.config["budget"], "maximum_decisions": 1, "maximum_requested_ticks": 6}
        journal = io.BytesIO()
        budget = DiagnosticBudget(limits, journal)
        serial = budget.reserve("step", {"ticks": 6})
        with self.assertRaisesRegex(ContractError, "arrival_step_budget"):
            budget.reserve("step", {"ticks": 6})
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["step_requests"], 1)
        self.assertEqual(snapshot["requested_step_ticks"], 6)
        self.assertEqual(snapshot["confirmed_step_ticks"], 0)
        self.assertTrue(snapshot["unconfirmed_tail"])
        self.assertEqual(snapshot["pending_requests"][0]["serial"], serial)
        self.assertIn(b'request_reserved', journal.getvalue())

    def test_malformed_confirmation_cannot_erase_uncertain_tail(self):
        budget = DiagnosticBudget(self.config["budget"], io.BytesIO())
        serial = budget.reserve("step", {"ticks": 6})
        for value in (None, True, 7, -1, "6"):
            with self.subTest(value=value), self.assertRaisesRegex(ContractError, "arrival_tick_receipt"):
                budget.confirm(serial, {"actual_ticks": value})
            self.assertTrue(budget.snapshot()["unconfirmed_tail"])
        budget.confirm(serial, {"actual_ticks": 6})
        self.assertFalse(budget.snapshot()["unconfirmed_tail"])
        self.assertEqual(budget.snapshot()["confirmed_step_ticks"], 6)

    def test_cancellation_is_available_after_wall_expiry_without_extra_steps(self):
        now = [0.]
        budget = DiagnosticBudget(self.config["budget"], io.BytesIO(), clock=lambda: now[0])
        now[0] = 600.
        with self.assertRaisesRegex(ContractError, "arrival_wall_budget"):
            budget.reserve("step", {"ticks": 6})
        serial = budget.reserve("cancel", {})
        budget.confirm(serial, {})
        self.assertEqual(budget.snapshot()["requested_step_ticks"], 0)

    def test_partial_or_duplicate_outcomes_cannot_become_complete(self):
        self.assertFalse(summarize([], self.cases)["complete_valid_pairing"])
        metrics = {"success": False, "administratively_capped": False, "integrity_failure": False,
            "failed_admissions": 0, "overshoot_cells": 1., "final_distance_cells": 2.,
            "capped_completion_seconds": 4., "peak_toward_speed_cells_per_second": 20.,
            "mean_absolute_angle_radians": .1, "ordinary_trim_reversals_per_second": 0.,
            "purposeful_launch": True, "primary_edges": 1}
        rows = [{"case_id": case["id"], "family": family, "origin": case["origin"],
                 "metrics": dict(metrics), "error": None, "owner_closure_confirmed": True}
                for case in self.cases for family in FAMILIES]
        result = summarize(rows, self.cases)
        self.assertTrue(result["complete_valid_pairing"])
        self.assertFalse(result["morphology_admitted"])
        self.assertFalse(result["learned_skill_qualified"])
        self.assertEqual(result["families"][FAMILIES[1]]["successes"], 0)
        self.assertFalse(summarize(rows[:-1], self.cases)["complete_valid_pairing"])
        with self.assertRaisesRegex(ContractError, "arrival_pairing"):
            summarize(rows + [rows[0]], self.cases)
        rows[0]["owner_closure_confirmed"] = False
        self.assertFalse(summarize(rows, self.cases)["complete_valid_pairing"])

    def test_failure_records_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            write_json(path, {"status": "failed"})
            with self.assertRaises(FileExistsError):
                write_json(path, {"status": "completed"})
            self.assertEqual(json.loads(path.read_bytes()), {"status": "failed"})

    def test_import_does_not_load_neural_runtime_or_start_host(self):
        code = "import sys; import nca.evaluation.jolt_arrival; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


if __name__ == "__main__":
    unittest.main()
