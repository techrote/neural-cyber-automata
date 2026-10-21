"""Negative controls for prospective costs, teacher identity and static holdouts."""

import json
from pathlib import Path
import unittest

from nca.contracts import sha256_json
from nca.train.jolt_admission import (
    AdmissionError, BudgetLedger, POLICY_SEEDS, case_fingerprints, frozen_finals,
    load_config, terrain_fingerprint, validate_config, validate_splits,
    verify_prerequisite,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/learning/jolt-learn-002-v1.json"


def document(name):
    return json.loads((ROOT / "configs/learning" / name).read_text())


class StaticAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.config = document("jolt-learn-002-v1.json")
        self.split = document("jolt-static-split-v1.json")
        self.opened = document("jolt-opened-inventory-v1.json")

    def test_complete_prospective_template_is_static_and_matches_files(self):
        self.assertEqual(load_config(CONFIG, root=ROOT), self.config)
        validate_config(ROOT, self.config)
        report = validate_splits(self.split, self.opened)
        self.assertEqual(report["case_counts"], {"teacher": 64, "open": 4, "sealed": 16})
        self.assertFalse(report["physical_outcomes_inspected"])

    def test_a_new_seed_or_distant_pile_is_not_new_contact_geometry(self):
        original = [3, 4, 5, 6, 7]
        changed = [9, 4, 5, 6, 9]
        self.assertEqual(terrain_fingerprint(original, 376.5), terrain_fingerprint(changed, 376.5))
        changed[2] = 6
        self.assertNotEqual(terrain_fingerprint(original, 376.5), terrain_fingerprint(changed, 376.5))

    def test_material_overlap_removes_fake_sand_bed_novelty(self):
        # Heights <=6 are entirely hidden by the existing six-cell sand bed.
        a, b = [3] * 5, [3] * 5
        b[4] = 6
        self.assertEqual(terrain_fingerprint(a, 544.5), terrain_fingerprint(b, 544.5))
        b[4] = 7
        self.assertNotEqual(terrain_fingerprint(a, 544.5), terrain_fingerprint(b, 544.5))

    def test_renaming_seed_without_realization_is_rejected(self):
        self.split["sealed"][0]["host_seed"] += 1
        with self.assertRaisesRegex(AdmissionError, "realization"):
            validate_splits(self.split, self.opened)

    def test_actual_contact_and_whole_layout_overlap_are_rejected(self):
        row = self.split["sealed"][0]
        self.opened["opened_contact_sha256"].append(row["contact_sha256"])
        with self.assertRaisesRegex(AdmissionError, "contact geometry"):
            validate_splits(self.split, self.opened)
        self.opened["opened_contact_sha256"].pop()
        self.opened["opened_layout_height_keys"].append("".join(map(str, row["pile_heights"])))
        with self.assertRaisesRegex(AdmissionError, "whole layout"):
            validate_splits(self.split, self.opened)

    def test_historical_case_labels_and_partial_attempts_remain_open(self):
        self.assertGreaterEqual(len(self.opened["declared_cases"]), 186)
        self.assertGreaterEqual(len(self.opened["opened_layout_height_keys"]), 5000)
        files = self.opened["source_files"]
        for version in (1, 2, 3, 4):
            self.assertTrue(any(f"jolt-recovery-v{version}-" in name for name in files))
        row = self.split["teacher"][0]
        row["spawn"] = [264.0, 346.5]
        row["target"] = [380.0, 346.5]
        row.update(case_fingerprints(row))
        with self.assertRaisesRegex(AdmissionError, "previously opened"):
            validate_splits(self.split, self.opened)

    def test_missing_or_duplicate_case_and_unrelated_terrain_are_rejected(self):
        self.split["sealed"].pop()
        with self.assertRaisesRegex(AdmissionError, "16 cases"):
            validate_splits(self.split, self.opened)
        row = self.split["open"][0]
        row["contact_anchor_x"] = 208.5
        with self.assertRaisesRegex(AdmissionError, "distant"):
            case_fingerprints(row)

    def test_no_gate_or_counter_can_be_left_unset(self):
        self.config["budget"]["counters"].pop("bootstrap_forward")
        with self.assertRaisesRegex(AdmissionError, "unset"):
            validate_config(self.config)
        self.config = document("jolt-learn-002-v1.json")
        self.config["learned_gate"]["minimum_every_seed_success"] = .25
        with self.assertRaisesRegex(AdmissionError, "thresholds"):
            validate_config(self.config)

    def test_teacher_prerequisite_is_not_a_learned_qualification(self):
        record = document("jolt-prerequisite-v1.json")
        result = verify_prerequisite(ROOT, record)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["learned_skill_qualified"])
        self.assertFalse(result["morphology_admitted"])
        record["physical_source_revision"] = "0" * 40
        with self.assertRaisesRegex(AdmissionError, "substituted"):
            verify_prerequisite(ROOT, record)

    def test_evidence_and_preserved_source_changes_are_rejected(self):
        record = document("jolt-prerequisite-v1.json")
        record["source_files"]["src/nca/controllers/jolt_recovery_v4.py"] = "0" * 64
        with self.assertRaisesRegex(AdmissionError, "source changed"):
            verify_prerequisite(ROOT, record)
        record = document("jolt-prerequisite-v1.json")
        record["evidence_files"][record["teacher_review"]] = "0" * 64
        with self.assertRaisesRegex(AdmissionError, "evidence changed"):
            verify_prerequisite(ROOT, record)

    def test_six_final_hashes_are_required_even_for_authored_sealed_reference(self):
        checkpoints = [{"family": family, "seed": seed, "phase": "final",
                        "sha256": sha256_json([family, seed])}
                       for family in ("bc", "residual") for seed in POLICY_SEEDS]
        self.assertEqual(len(frozen_finals(checkpoints)), 6)
        with self.assertRaises(AdmissionError):
            frozen_finals(checkpoints[:-1])
        checkpoints[-1]["phase"] = "initial"
        with self.assertRaises(AdmissionError):
            frozen_finals(checkpoints)


class ReservationTests(unittest.TestCase):
    def setUp(self):
        self.clock = [100.0]
        self.budget = document("jolt-learn-002-v1.json")["budget"]
        self.ledger = BudgetLedger(self.budget, started=100.0, monotonic=lambda: self.clock[0])

    def test_related_operation_reservations_are_atomic(self):
        ledger = self.ledger
        with self.assertRaises(AdmissionError):
            ledger.reserve_many({"decisions": 1, "requested_ticks": 120001}, stage="teacher")
        self.assertEqual(ledger.counts["decisions"], 0)
        self.assertEqual(ledger.counts["requested_ticks"], 0)
        self.assertEqual(len(ledger.reservations), 0)

    def test_unused_aggregate_margin_cannot_expand_a_stage(self):
        ledger = self.ledger
        ledger.reserve("decisions", 2560, stage="teacher")
        with self.assertRaisesRegex(AdmissionError, "stage counter"):
            ledger.reserve("decisions", stage="teacher")
        self.assertEqual(ledger.counts["decisions"], 2560)
        with self.assertRaisesRegex(AdmissionError, "original stage"):
            ledger.reserve("decisions")

    def test_refused_or_uncertain_work_remains_charged(self):
        ledger = self.ledger
        ledger.reserve_many({"decisions": 1, "requested_ticks": 6,
                             "owner_requests": 1, "owner_step": 1}, stage="ppo")
        ledger.reserve("uncertain_ticks", 6, stage="ppo")
        self.assertEqual(ledger.counts["decisions"], 1)
        self.assertEqual(ledger.counts["requested_ticks"], 6)
        self.assertFalse(ledger.snapshot()["refused_or_uncertain_work_refunded"])

    def test_original_start_deadline_survives_episodes_and_cleanup(self):
        ledger = self.ledger
        self.clock[0] = 1770.0  # Ten active seconds remain; transport needs15.
        with self.assertRaisesRegex(AdmissionError, "reserve exhausted"):
            ledger.reserve("owner_requests", allowance_seconds=15)
        ledger.reserve("episodes_started", stage="ppo")
        self.assertEqual(ledger.active_deadline, 1780.0)
        self.clock[0] = 1780.0
        with self.assertRaises(AdmissionError):
            ledger.reserve("bc_optimizer")
        ledger.reserve("owner_close", cleanup=True)
        self.assertEqual(ledger.remaining_seconds(cleanup=True), 120.0)
        self.clock[0] = 1900.0
        with self.assertRaises(AdmissionError):
            ledger.reserve("host_terminations", cleanup=True)

    def test_config_copy_invalid_amount_unknown_work_and_closed_ledger(self):
        self.budget["counters"]["host_starts"] = 99
        self.ledger.reserve("host_starts")
        with self.assertRaises(AdmissionError):
            self.ledger.reserve("host_starts")
        for key, value in (("bc_optimizer", True), ("bc_optimizer", -1), ("unknown_work", 1)):
            with self.assertRaises(AdmissionError):
                self.ledger.reserve(key, value)
        report = self.ledger.close()
        self.assertTrue(report["closed"])
        with self.assertRaises(AdmissionError):
            self.ledger.reserve("owner_requests", cleanup=True)


if __name__ == "__main__":
    unittest.main()
