"""Negative controls for the prospective authored teacher gate; no physics."""

import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from nca.contracts import ContractError, sha256_json
from nca.evaluation.jolt_recovery_acceptance import (
    CANDIDATE, CANDIDATE_V2, REFERENCE, HISTORICAL_FAILURES, HISTORICAL_MISSING_AIRBORNE,
    UNSUPPORTED_FAILURE, acceptance_specification, assess_teacher, verify_replay_source,
)


def fixture():
    named = list(dict.fromkeys(HISTORICAL_FAILURES + HISTORICAL_MISSING_AIRBORNE))
    cases = named + [f"paired-{index}" for index in range(32 - len(named))]
    rows = []
    for family in (REFERENCE, CANDIDATE):
        for case in cases:
            failed = case in HISTORICAL_FAILURES if family == REFERENCE else case in {
                "demo-floor-left-60-2", "reference-floor-left-63", "paired-0"}
            rows.append({"family": family, "case_id": case, "error": None,
                "owner_closure_confirmed": True,
                "metrics": {"success": not failed, "administratively_capped": False,
                    "integrity_failure": False, "failed_admissions": 0,
                    "purposeful_launch": family == CANDIDATE or case not in HISTORICAL_MISSING_AIRBORNE,
                    "ordinary_trim_reversals_per_second": .25,
                    "mean_absolute_angle_radians": .05,
                    "peak_toward_speed_cells_per_second": 48., "inference_p95_ms": 1.},
                "character_metrics": {"requested_primary_edges": 2 if family == CANDIDATE else 1}})
    return rows


class RecoveryAcceptanceTests(unittest.TestCase):
    def assess(self, rows):
        return assess_teacher(rows, pulse_authority_verified=True)

    def test_complete_candidate_needs_both_failure_classes_and_character(self):
        result = self.assess(fixture())
        self.assertTrue(result["teacher_accepted"])
        self.assertEqual(result["measurements"][CANDIDATE]["arrivals"], 29)
        self.assertEqual(len(result["historical_rescues"]), 3)
        self.assertFalse(result["learned_skill_qualified"])
        self.assertFalse(result["morphology_admitted"])

    def test_missing_duplicate_or_foreign_pair_refuses_qualification(self):
        rows = fixture()
        with self.assertRaises(ContractError):
            self.assess(rows[:-1])
        rows[-1] = copy.deepcopy(rows[-2])
        with self.assertRaises(ContractError):
            self.assess(rows)
        rows = fixture()
        rows[-1]["case_id"] = "foreign-case"
        with self.assertRaises(ContractError):
            self.assess(rows)

    def test_more_total_success_cannot_hide_two_regressions(self):
        rows = fixture()
        for row in rows:
            if row["family"] == CANDIDATE:
                row["metrics"]["success"] = row["case_id"] not in {"paired-0", "paired-1"}
        result = self.assess(rows)
        self.assertTrue(result["checks"]["strict_arrival_improvement"])
        self.assertFalse(result["checks"]["at_most_one_regression"])
        self.assertFalse(result["teacher_accepted"])

    def test_unsupported_failure_cannot_be_replaced_by_other_rescues(self):
        rows = fixture()
        for row in rows:
            if row["family"] == CANDIDATE:
                row["metrics"]["success"] = row["case_id"] != UNSUPPORTED_FAILURE
        result = self.assess(rows)
        self.assertTrue(result["checks"]["at_least_three_historical_rescues"])
        self.assertFalse(result["checks"]["unsupported_drop_rescued"])
        self.assertFalse(result["teacher_accepted"])

    def test_retained_total_launch_count_does_not_prove_character_restoration(self):
        rows = fixture()
        for row in rows:
            if row["family"] == CANDIDATE:
                row["metrics"]["purposeful_launch"] = row["case_id"] not in HISTORICAL_MISSING_AIRBORNE
        result = self.assess(rows)
        self.assertTrue(result["checks"]["retain_overall_airborne_character"])
        self.assertFalse(result["checks"]["restore_airborne_character"])
        self.assertFalse(result["teacher_accepted"])

    def test_individual_latency_outlier_is_not_hidden_by_average(self):
        rows = fixture()
        rows[-1]["metrics"]["inference_p95_ms"] = 5.00001
        self.assertFalse(self.assess(rows)["checks"]["every_case_latency_limit"])
        rows[-1]["metrics"]["inference_p95_ms"] = 5.
        self.assertTrue(self.assess(rows)["checks"]["every_case_latency_limit"])

    def test_unproven_pulse_reason_or_extra_pulse_cannot_pass(self):
        self.assertFalse(assess_teacher(fixture(), pulse_authority_verified=False)["teacher_accepted"])
        rows = fixture()
        rows[-1]["character_metrics"]["requested_primary_edges"] = 3
        self.assertFalse(self.assess(rows)["teacher_accepted"])

    def test_missing_or_invalid_measurement_is_not_zero(self):
        for value in (None, float("nan"), True):
            with self.subTest(value=value):
                rows = fixture()
                rows[-1]["metrics"]["inference_p95_ms"] = value
                with self.assertRaises(ContractError):
                    self.assess(rows)
        rows = fixture()
        rows[-1]["metrics"]["administratively_capped"] = True
        with self.assertRaises(ContractError):
            self.assess(rows)

    def test_specification_is_a_fresh_copy(self):
        changed = acceptance_specification()
        changed["historical_failures"].clear()
        self.assertEqual(len(acceptance_specification()["historical_failures"]), 5)

    def test_v2_uses_identical_numeric_rules_without_relabeling_v1_rows(self):
        old, new = acceptance_specification(), acceptance_specification(CANDIDATE_V2)
        self.assertEqual({k: v for k, v in old.items() if k not in ("schema", "candidate")},
                         {k: v for k, v in new.items() if k not in ("schema", "candidate")})
        rows = fixture()
        with self.assertRaises(ContractError):
            assess_teacher(rows, pulse_authority_verified=True, candidate_family=CANDIDATE_V2)
        for row in rows:
            if row["family"] == CANDIDATE:
                row["family"] = CANDIDATE_V2
        self.assertTrue(assess_teacher(rows, pulse_authority_verified=True,
                                      candidate_family=CANDIDATE_V2)["teacher_accepted"])
        with self.assertRaises(ContractError):
            acceptance_specification("unpublished-candidate")

    def test_replay_refuses_source_drift_despite_unchanged_revision_label(self):
        frozen = {"src/nca/controllers/jolt_recovery.py": "a" * 64,
                  "src/nca/controllers/recovery_supervisor.py": "b" * 64}
        manifest = {"source_files": frozen, "source_files_sha256": sha256_json(frozen)}
        revision = "c" * 40
        for altered in (dict(frozen, **{"src/nca/controllers/jolt_recovery.py": "d" * 64}),
                        {**frozen, "src/nca/extra_replay.py": "e" * 64}):
            with self.subTest(altered=altered), patch("nca.data.capture.source_identity",
                    return_value=(revision, altered, sha256_json(altered))):
                with self.assertRaisesRegex(ContractError, "imported replay package differs"):
                    verify_replay_source(manifest, expected_revision=revision)

    def test_replay_refuses_another_checkout_revision(self):
        frozen = {"src/nca/controllers/jolt_recovery.py": "a" * 64}
        manifest = {"source_files": frozen, "source_files_sha256": sha256_json(frozen)}
        with patch("nca.data.capture.source_identity", return_value=("b" * 40, frozen, sha256_json(frozen))):
            with self.assertRaisesRegex(ContractError, "published revision"):
                verify_replay_source(manifest, expected_revision="c" * 40)

    def test_matching_replay_source_reaches_specification_and_refuses_substitution(self):
        from nca.controllers.jolt_recovery import JoltRecoveryController
        relative = "src/nca/controllers/jolt_recovery.py"
        checkout = Path(__file__).resolve().parents[1]
        body = (checkout / relative).read_bytes()
        frozen = {relative: hashlib.sha256(body).hexdigest()}
        revision = "c" * 40
        manifest = {"source_files": frozen, "source_files_sha256": sha256_json(frozen),
                    "declaration": {"families": ["authored-balanced-v1", REFERENCE, CANDIDATE],
                        "candidate_specifications": {
                        CANDIDATE: JoltRecoveryController().specification()}}}
        with patch("nca.data.capture.source_identity", return_value=(revision, frozen, sha256_json(frozen))), \
                patch("nca.evaluation.jolt_recovery_acceptance.subprocess.check_output", return_value=body), \
                patch("nca.evaluation.jolt_recovery_acceptance.__file__",
                      str(checkout / "installed/nca/evaluation/jolt_recovery_acceptance.py")), \
                patch.dict("os.environ", {}, clear=True), patch.object(Path, "cwd", return_value=checkout):
            self.assertTrue(verify_replay_source(manifest, expected_revision=revision,
                                                source_root=checkout)["verified"])
            self.assertTrue(verify_replay_source(manifest, expected_revision=revision)["verified"])
            manifest["declaration"]["candidate_specifications"][CANDIDATE] = {}
            with self.assertRaisesRegex(ContractError, "specification differs"):
                verify_replay_source(manifest, expected_revision=revision, source_root=checkout)


if __name__ == "__main__":
    unittest.main()
