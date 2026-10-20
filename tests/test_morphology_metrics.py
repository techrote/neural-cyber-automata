"""Complete and adversarial synthetic comparisons; never gameplay evidence."""

from copy import deepcopy
from itertools import product
import math
import unittest

from nca.contracts import ContractError, sha256_json, strict_json_loads
from nca.controllers.morphology import SemanticAction
from nca.train.morphology_task import TASK_SPEC, REFERENCE_SPEC, summarize_comparison
from nca.train.morphology_metrics import COMPARISON_METRICS_SPEC
from test_morphology_task import ROOT, PROFILES


def declaration():
    """Resolve the fixed published plan geometry with synthetic prerequisites."""
    spec = COMPARISON_METRICS_SPEC
    result = {"schema": "nca-morphology-continuation-declaration/v1", "comparison_sha256": spec["comparison_sha256"],
              "task": deepcopy(TASK_SPEC), "reference": deepcopy(REFERENCE_SPEC), "comparison_metrics": deepcopy(spec)}
    groups = {
        "zero_shot": (list(product(spec["zero_shot_profiles"], spec["zero_shot_families"])), 81100001),
        "adapted_evaluation": (list(product(spec["zero_shot_profiles"], spec["adaptation_evaluation_families"])), 81300001),
        "tuning": ([(row["profile"], row["family"]) for row in spec["tuning_assignments"]], 81020001),
    }
    for phase, (assignments, seed) in groups.items():
        result[phase] = []
        for i, (profile, family) in enumerate(assignments):
            height, side, distance = family.split("-")
            drop = 0 if height == "floor" else int(height[4:])
            x = 264. if side == "left" else 176.
            row = PROFILES["profiles"][profile]
            result[phase].append({"id": f"{profile}-{family}", "profile": profile, "family": family,
                "host_seed": seed + i, "spawn": [x, 352. - row["definition"]["spawn_floor_clearance"] - drop],
                "target": [x + row["definition"]["parts"][0]["offset"][0] + (-1 if side == "left" else 1) * int(distance),
                           row["observation"]["parts"][0]["position"][1]]})
    return result


def complete_rows(config):
    """Fabricated full148-row schedule, with a consistent +.5 paired graph gain."""
    rows = []
    for phase in COMPARISON_METRICS_SPEC["phase_row_counts"]:
        cases = config[phase if phase in ("tuning", "adapted_evaluation") else "zero_shot"]
        policies = [(kind, None) for kind in ("reference", "idle")] if phase == "authored_and_idle" else product(
            COMPARISON_METRICS_SPEC["candidates"], COMPARISON_METRICS_SPEC["training_seeds"])
        for kind, seed in policies:
            for i, case in enumerate(cases):
                suffix = "initial" if phase == "zero_shot_initial" else "adapted-" + case["profile"] if phase == "adapted_evaluation" else "shared_final"
                policy_id = ("waypoint-reference-v1" if kind == "reference" else "waypoint-idle-v1" if kind == "idle"
                             else f"{kind}-seed{seed}-{suffix}")
                policy_hash = (sha256_json(REFERENCE_SPEC) if kind == "reference" else
                               sha256_json({"kind": "morphology-idle/v1", "semantic_request": SemanticAction().to_dict()})
                               if kind == "idle" else sha256_json({"synthetic_policy": policy_id}))
                success = (kind == "reference" or phase == "zero_shot_final" and (kind == "shared_graph" or i % 4 < 2))
                success = bool(success)
                identity = [phase, case["id"], kind, seed]
                rows.append({"scenario_family": case["id"], "profile": case["profile"], "host_seed": case["host_seed"],
                    "phase": phase, "kind": kind, "training_seed": seed, "policy_id": policy_id, "policy_sha256": policy_hash,
                    "profile_sha256": PROFILES["profiles"][case["profile"]]["profile_sha256"],
                    "record_schema": "nca-morphology-episode/v1", "record_sha256": sha256_json({"synthetic_record": identity}),
                    "artifact_sha256": sha256_json({"synthetic_bytes": identity}),
                    "boundary_audit_sha256": sha256_json({"synthetic_boundary": identity}),
                    "task_spec_sha256": sha256_json(TASK_SPEC), "actual_owner_closed": True, "error": None,
                    "source_boundary_regressions": 0, "decisions": 10 if success else 40, "completed_ticks": 60 if success else 240,
                    "metrics": {"success": success, "reason": "arrival" if success else "task_deadline",
                        "success_seconds": 1. if success else None, "capped_completion_seconds": 1. if success else 4.,
                        "integrity_failure": False, "failed_admissions": 0, "administratively_capped": False,
                        "owner_feel_accepted": False, "elapsed_ticks": 60 if success else 240, "return": .7 if success else -.6,
                        "ordinary_trim_reversals_per_second": .1, "mean_absolute_angle_radians": .1,
                        "peak_useful_speed_cells_per_second": 10., "inference_p50_ms": 1., "inference_p95_ms": 2.,
                        "response_seconds": .1, "settled_after_landing_seconds": None}})
    return rows


def outcome(row, success):
    """Keep synthetic outcome/counter metadata mutually consistent after mutation."""
    row.update(decisions=10 if success else 40, completed_ticks=60 if success else 240)
    row["metrics"].update(success=success, reason="arrival" if success else "task_deadline",
        success_seconds=1. if success else None, capped_completion_seconds=1. if success else 4.,
        elapsed_ticks=60 if success else 240)


class MorphologyComparisonTests(unittest.TestCase):
    def setUp(self):
        self.config = declaration()

    def test_source_spec_preserves_every_original_threshold_split_and_seed(self):
        original = strict_json_loads((ROOT / "configs/morphology/comparison-v1.json").read_bytes())
        spec = COMPARISON_METRICS_SPEC
        self.assertEqual(spec["comparison_sha256"], sha256_json(original))
        self.assertEqual(spec["every_seed_and_profile_bounds"], original["future_metrics"]["every_seed_and_profile_bounds"])
        self.assertEqual(spec["training_seeds"], original["future_learning"]["policy_seeds"])
        for key in ("zero_shot_profiles", "zero_shot_families", "adaptation_evaluation_families", "tuning_assignments"):
            self.assertEqual(spec[key], original["splits"][key])
        self.assertEqual(spec["primary"]["minimum_mean_paired_success_gain"], .125)
        self.assertEqual(sum(spec["phase_row_counts"].values()), 148)

    def test_full_synthetic_schedule_requires_all_phases_and_enumerates_exact_paired_resamples(self):
        result = summarize_comparison(complete_rows(self.config), self.config, run_completed=True)
        self.assertTrue(result["complete"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["coverage"]["valid_unique"], 148)
        gain = result["zero_shot"]["paired_final_success_gain"]
        self.assertEqual(gain["resamples"], 27)
        self.assertEqual((gain["mean"], gain["lower95"], gain["upper95"]), (.5, .5, .5))
        self.assertEqual(gain["individual_seeds"], [.5, .5, .5])
        self.assertEqual(len(result["zero_shot"]["character_gates"]), 12)
        self.assertEqual(len(result["zero_shot"]["per_seed"]), 6)
        self.assertTrue(result["adaptation"]["complete"])
        self.assertIsNone(result["adaptation"]["matched_reference_comparison"])
        self.assertIsNone(result["adaptation"]["paired_gain_over_shared"])
        self.assertFalse(result["owner_feel_accepted"])
        self.assertEqual(result["errors"], [])

    def test_large_mean_does_not_hide_uncertain_between_seed_paired_gain(self):
        rows = complete_rows(self.config)
        for row in rows:
            if row["phase"] == "zero_shot_final" and row["kind"] == "shared_graph" and row["training_seed"] == 6113:
                outcome(row, False)
        result = summarize_comparison(rows, self.config, run_completed=True)
        gain = result["zero_shot"]["paired_final_success_gain"]
        self.assertGreater(gain["mean"], .125)
        self.assertLess(gain["lower95"], 0.)
        self.assertFalse(result["passed"])
        self.assertEqual(result["recommendation"], "retain_padded_recurrent")

    def test_equal_architectures_keep_simpler_policy_and_one_eighth_boundary_is_inclusive(self):
        for gain, expected in ((0., False), (.125, True)):
            rows = complete_rows(self.config)
            for row in rows:
                if row["phase"] == "zero_shot_final" and row["kind"] == "shared_graph":
                    index = next(i for i, case in enumerate(self.config["zero_shot"]) if case["id"] == row["scenario_family"])
                    outcome(row, index % 4 < 2 or bool(gain) and index == 2)
            result = summarize_comparison(rows, self.config, run_completed=True)
            self.assertEqual(result["zero_shot"]["paired_final_success_gain"]["mean"], gain)
            self.assertEqual(result["passed"], expected)

    def test_missing_nonprimary_phase_and_unconfirmed_run_cannot_open_final_go(self):
        all_rows = complete_rows(self.config)
        for missing_phase in ("tuning", "adapted_evaluation"):
            rows = [row for row in all_rows if row["phase"] != missing_phase]
            result = summarize_comparison(rows, self.config, run_completed=True)
            self.assertTrue(result["zero_shot"]["complete"])
            self.assertTrue(result["zero_shot"]["passed"])
            self.assertFalse(result["complete"])
            self.assertFalse(result["passed"])
            self.assertEqual(result["recommendation"], "inconclusive")
        result = summarize_comparison(all_rows, self.config, run_completed=False)
        self.assertTrue(result["zero_shot"]["passed"])
        self.assertFalse(result["passed"])
        with self.assertRaises(ContractError):
            summarize_comparison(all_rows, self.config, run_completed=1)

    def test_one_missing_or_duplicate_case_is_reported_and_never_averaged_as_partial_coverage(self):
        rows = complete_rows(self.config)
        removed = next(row for row in rows if row["phase"] == "zero_shot_final")
        for changed in ([row for row in rows if row is not removed], rows + [deepcopy(removed)]):
            result = summarize_comparison(changed, self.config, run_completed=True)
            self.assertFalse(result["complete"])
            self.assertFalse(result["zero_shot"]["complete"])
            self.assertIsNone(result["zero_shot"]["paired_final_success_gain"])
            incomplete = [row for row in result["zero_shot"]["per_seed"] if not row["final"]["complete"]]
            self.assertEqual(len(incomplete), 1)
            self.assertNotIn("success", incomplete[0]["final"])
            self.assertTrue(result["coverage"]["missing_or_invalid"])

    def test_character_is_per_candidate_seed_and_profile_not_pooled_across_bodies(self):
        for metric, value in (("ordinary_trim_reversals_per_second", .601),
                              ("mean_absolute_angle_radians", .201),
                              ("peak_useful_speed_cells_per_second", 6.99),
                              ("peak_useful_speed_cells_per_second", 15.01)):
            rows = complete_rows(self.config)
            for row in rows:
                if (row["phase"], row["kind"], row["training_seed"], row["profile"]) == ("zero_shot_final", "shared_graph", 6113, "monowheel"):
                    row["metrics"][metric] = value
            result = summarize_comparison(rows, self.config, run_completed=True)
            with self.subTest(metric=metric, value=value):
                self.assertTrue(result["complete"])
                self.assertFalse(result["passed"])
                failed = [row for row in result["zero_shot"]["character_gates"] if not row["passed"]]
                self.assertEqual([(row["family"], row["training_seed"], row["profile"]) for row in failed],
                                 [("shared_graph", 6113, "monowheel")])

    def test_zero_reference_speed_does_not_create_a_fabricated_denominator(self):
        rows = complete_rows(self.config)
        for row in rows:
            if row["kind"] == "reference" and row["profile"] == "ripper":
                row["metrics"]["peak_useful_speed_cells_per_second"] = 0.
        result = summarize_comparison(rows, self.config, run_completed=True)
        self.assertFalse(result["passed"])
        self.assertTrue(all(row["peak_useful_speed_ratio_to_reference"] is None for row in
                            result["zero_shot"]["character_gates"] if row["profile"] == "ripper"))

    def test_every_phase_retains_runtime_integrity_and_censor_failures(self):
        for phase, defect in product(COMPARISON_METRICS_SPEC["phase_row_counts"], ("latency", "integrity", "admission", "cap", "source", "owner", "error")):
            rows = complete_rows(self.config)
            row = next(row for row in rows if row["phase"] == phase)
            if defect == "latency": row["metrics"]["inference_p95_ms"] = 5.001
            elif defect == "integrity": row["metrics"]["integrity_failure"] = True
            elif defect == "admission": row["metrics"]["failed_admissions"] = 1
            elif defect == "cap": row["metrics"]["administratively_capped"] = True
            elif defect == "source": row["source_boundary_regressions"] = 1
            elif defect == "owner": row["actual_owner_closed"] = False
            else: row["error"] = {"code": "synthetic_failure"}
            with self.subTest(phase=phase, defect=defect):
                self.assertFalse(summarize_comparison(rows, self.config, run_completed=True)["passed"])

    def test_source_weight_identity_nonfinite_and_boolean_corruption_close_the_gate(self):
        for defect in ("family", "seed", "seed_bool", "case", "host_seed", "profile", "policy_id", "weights", "profile_hash",
                       "record_reuse", "missing_hash", "schema", "nan", "success_bool", "cap_bool", "ticks", "return", "task_hash"):
            rows = complete_rows(self.config)
            row = next(row for row in rows if row["phase"] == "zero_shot_final")
            if defect == "family": row["kind"] = "selected_lucky_seed"
            elif defect == "seed": row["training_seed"] = 1103
            elif defect == "seed_bool": row["training_seed"] = True
            elif defect == "case": row["scenario_family"] = "unplanned-case"
            elif defect == "host_seed": row["host_seed"] += 1
            elif defect == "profile": row["profile"] = "jev"
            elif defect == "policy_id": row["policy_id"] += "-replacement"
            elif defect == "weights": row["policy_sha256"] = "f" * 64
            elif defect == "profile_hash": row["profile_sha256"] = "f" * 64
            elif defect == "record_reuse": row["record_sha256"] = rows[0]["record_sha256"]
            elif defect == "missing_hash": row["boundary_audit_sha256"] = None
            elif defect == "schema": row["record_schema"] = "nca-morphology-incomplete-owner-receipts/v1"
            elif defect == "nan": row["metrics"]["peak_useful_speed_cells_per_second"] = float("nan")
            elif defect == "success_bool": row["metrics"]["success"] = 1
            elif defect == "cap_bool": row["metrics"]["administratively_capped"] = 0
            elif defect == "ticks": row["completed_ticks"] -= 1
            elif defect == "return": row["metrics"]["return"] = float("inf")
            else: row["task_spec_sha256"] = "a" * 64
            with self.subTest(defect=defect):
                result = summarize_comparison(rows, self.config, run_completed=True)
                self.assertFalse(result["complete"])
                self.assertFalse(result["passed"])
                self.assertTrue(result["errors"])

    def test_missing_landing_remains_unmeasured_and_adaptation_does_not_invent_a_pair(self):
        result = summarize_comparison(complete_rows(self.config), self.config, run_completed=True)
        for row in result["adaptation"]["per_seed_profile"]:
            self.assertEqual(row["measured_settled_after_landing_seconds_count"], 0)
            self.assertIsNone(row["mean_measured_settled_after_landing_seconds"])
        self.assertIsNone(result["adaptation"]["paired_gain_over_shared"])
        self.assertIn("no allocated matched reference", result["adaptation"]["limit"])

    def test_phase_schedule_and_original_predicates_cannot_be_relaxed(self):
        for defect in ("bounds", "seeds", "task", "reference", "comparison", "case", "host_seed", "geometry", "missing_case"):
            config = deepcopy(self.config)
            if defect == "bounds": config["comparison_metrics"]["every_seed_and_profile_bounds"]["every_scenario_inference_p95_maximum_ms"] = 6.
            elif defect == "seeds": config["comparison_metrics"]["training_seeds"] = [1, 2, 3]
            elif defect == "task": config["task"]["deadline_ticks"] = 300
            elif defect == "reference": config["reference"]["position_gain"] = 2.
            elif defect == "comparison": config["comparison_sha256"] = "f" * 64
            elif defect == "case": config["zero_shot"][0]["id"] = "another-case"
            elif defect == "host_seed": config["zero_shot"][0]["host_seed"] += 1
            elif defect == "geometry": config["zero_shot"][0]["spawn"] = [float("nan"), 0.]
            else: config["zero_shot"].pop()
            with self.subTest(defect=defect), self.assertRaises(ContractError):
                summarize_comparison([], config, run_completed=False)


if __name__ == "__main__":
    unittest.main()
