"""Tiny CPU conformance for scoring; no accepted corpus or final registry is opened."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from nca.contracts import ContractError, sha256_json
from nca.evaluation.ik_training import (
    COMPARATORS, SEEDS, Prediction, aggregate_comparisons, apply_cpu_utility,
    collect_predictions, evaluate_comparison, freeze_candidates, measure_cpu_latency,
    model_predictor, nearest_rank, reference_predictors, score_raw,
    select_tuning_checkpoint, validate_final_opening,
)
from nca.ik.data import DatasetConfig, Normalization, RuntimeInputs, generate_sample, iter_tasks
from nca.ik.kinematics import chain_from_profile
from nca.ik.profiles import runner_right_sole_profile
from nca.ik.reference import analytic_two_link


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = (importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("numpy") is not None)


def seal(value):
    return dict(value, sha256=sha256_json(value))


class IKScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = DatasetConfig.from_json((ROOT / "configs/ik/datasets/tiny_v1.json").read_bytes())
        # Retained zero-iteration teacher failures keep these fixtures small.
        cls.config = replace(config, teacher=replace(config.teacher, max_iterations=0, max_seconds=1.))
        cls.tuning = tuple(generate_sample(task, cls.config) for task in iter_tasks(cls.config, "tuning"))
        cls.final = tuple(generate_sample(task, cls.config) for task in iter_tasks(cls.config, "final"))
        cls.chain = chain_from_profile(runner_right_sole_profile())

    def analytic(self, inputs):
        result = analytic_two_link(self.chain, inputs.target_root, inputs.current,
                                   lower=(-.72, -.27), upper=(.72, .27))
        return Prediction(result.configuration, result.status)

    def predictions(self, samples=None):
        return collect_predictions(self.tuning if samples is None else samples, self.analytic)

    def report(self):
        return score_raw(self.tuning, self.predictions(), self.config)

    def frozen(self):
        selection = select_tuning_checkpoint([
            {"step": 1, "checkpoint_sha256": "a" * 64, "tuning": self.report()},
        ])
        candidates = {seed: dict(selection, checkpoint_sha256=letter * 64)
                      for seed, letter in zip(SEEDS, "abc")}
        frozen = freeze_candidates(
            candidates, experiment="cpu-conformance-only", dataset_manifest_sha256="d" * 64,
            final_suite_identity_sha256="e" * 64, registry_id="fixture-registry",
        )
        opening = seal({
            "schema": "nca-ik-final-opening/v1", "experiment": frozen["experiment"],
            "frozen_candidates_sha256": [entry["checkpoint_sha256"] for entry in frozen["candidates"]],
            "opened_utc": "2026-10-22T00:00:00Z",
            "suite_identity_sha256": frozen["final_suite_identity_sha256"],
            "dataset_manifest_sha256": frozen["dataset_manifest_sha256"],
            "registry_id": frozen["registry_id"], "future_experiments_status": "development_only",
        })
        return frozen, opening

    def test_import_does_not_import_optional_runtime(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import nca.evaluation.ik_training; "
             "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_nearest_rank_is_not_interpolated(self):
        self.assertEqual(nearest_rank(range(1, 21), .95), 19)
        self.assertEqual(nearest_rank((9, 1, 4), .5), 4)
        self.assertIsNone(nearest_rank((), .95))
        with self.assertRaises(ContractError):
            nearest_rank((float("nan"),), .95)

    def test_score_uses_independent_geometry_without_solving(self):
        predictions = self.predictions()
        with patch("nca.evaluation.ik_training.solve_numeric", side_effect=AssertionError("teacher called")), \
             patch("nca.evaluation.ik_training.analytic_two_link", side_effect=AssertionError("analytic called")):
            report = score_raw(self.tuning, predictions, self.config)
        self.assertTrue(report["complete"])
        self.assertLess(report["nominal"]["worst_error_cells"], 1e-8)
        self.assertEqual(report["nominal"]["feasible_success_fraction"], 1.)
        self.assertEqual(report["continuity"]["admitted_branch_switches"], 0)
        self.assertFalse(report["qualified"])

    def test_missing_prediction_keeps_denominator_and_nulls_selection(self):
        predictions = self.predictions()
        absent = next(sample.task.id for sample in self.tuning if sample.task.state.kind == "point")
        del predictions[absent]
        report = score_raw(self.tuning, predictions, self.config)
        self.assertEqual(report["scheduled_calls"], len(self.tuning))
        self.assertEqual(report["nominal"]["missing_predictions"], 1)
        self.assertFalse(report["complete"])
        self.assertIsNone(report["selection_metric"])
        self.assertIsNone(report["nominal"]["p95_error_cells"])
        self.assertIsNotNone(report["nominal"]["observed_only_diagnostic"]["p95_error_cells"])

    def test_missing_sample_is_not_replaced_by_supplied_prediction(self):
        report = score_raw(self.tuning[1:], self.predictions(), self.config)
        self.assertEqual(report["scheduled_calls"], len(self.tuning))
        self.assertFalse(report["complete"])
        self.assertTrue(report["rows"][0]["sample_missing"])

    def test_nonfinite_and_bound_outputs_remain_visible(self):
        predictions = self.predictions()
        ids = [sample.task.id for sample in self.tuning if sample.task.state.kind == "point"][:2]
        predictions[ids[0]] = Prediction((float("nan"), 0.))
        predictions[ids[1]] = Prediction((4., 0.))
        report = score_raw(self.tuning, predictions, self.config)
        self.assertEqual(report["nominal"]["nonfinite_outputs"], 1)
        self.assertEqual(report["nominal"]["hard_bound_violations"], 1)
        self.assertFalse(report["complete"])
        self.assertIsNone(report["selection_metric"])
        # JSON evidence remains strict: no NaN/Infinity escapes the report.
        self.assertEqual(len(sha256_json(report)), 64)

    def test_direct_model_cannot_hide_refinement_fallback_or_projection(self):
        for changes in (
            {"correction_iterations": 1}, {"full_reference_fallback_calls": 1},
            {"projection_max_radians": .001}, {"solver_iterations": 1},
        ):
            with self.subTest(changes=changes):
                predictions = self.predictions()
                task_id = self.tuning[0].task.id
                predictions[task_id] = replace(predictions[task_id], **changes)
                report = score_raw(self.tuning, predictions, self.config)
                self.assertFalse(report["complete"])
                self.assertIn("direct_mode_has_solver_or_projection_work",
                              report["rows"][0]["protocol_violations"])

    def test_predictor_receives_only_runtime_inputs_and_invalid_rows_are_guarded(self):
        calls = []
        def predictor(inputs):
            self.assertIsInstance(inputs, RuntimeInputs)
            self.assertNotIn("teacher", inputs.to_dict())
            calls.append(inputs.sha256)
            return self.analytic(inputs)
        predictions = collect_predictions(self.tuning, predictor)
        self.assertEqual(len(predictions), len(self.tuning))
        self.assertLess(len(calls), len(self.tuning))
        report = score_raw(self.tuning, predictions, self.config)
        invalid = [row for row in report["rows"] if row["expected_refusal"] is not None]
        self.assertTrue(invalid)
        self.assertTrue(all(row["expected_refusal_observed"] for row in invalid))
        self.assertEqual(report["nominal"]["scheduled"],
                         sum(sample.task.state.kind != "stress" for sample in self.tuning))

    def test_predictor_exception_retains_failed_attempt_without_private_message(self):
        def broken(inputs):
            raise RuntimeError("private-machine-path-not-for-evidence")
        predictions = collect_predictions(self.tuning, broken)
        report = score_raw(self.tuning, predictions, self.config)
        self.assertEqual(report["scheduled_calls"], len(self.tuning))
        self.assertFalse(report["complete"])
        self.assertNotIn("private-machine", str(report))
        self.assertIn("prediction_exception", report["nominal"]["status_counts"])

    def test_unknown_predictions_and_training_split_are_refused(self):
        predictions = self.predictions()
        predictions["f" * 64] = Prediction((0., 0.))
        with self.assertRaises(ContractError):
            score_raw(self.tuning, predictions, self.config)
        with self.assertRaises(ContractError):
            score_raw(self.tuning, {}, self.config, partition="training")

    def test_all_five_controls_have_identical_scheduled_denominators(self):
        result = evaluate_comparison(self.tuning, self.config, trained=self.analytic,
                                     untrained=lambda inputs: Prediction(inputs.current))
        self.assertEqual(set(result["comparators"]), set(COMPARATORS))
        self.assertEqual({r["scheduled_calls"] for r in result["comparators"].values()}, {len(self.tuning)})
        numerical = result["comparators"]["independent_numerical_reference"]
        self.assertEqual(numerical["nominal"]["solver_iterations"], 0)
        self.assertFalse(numerical["complete"])
        self.assertEqual(result["practical_utility"], "inconclusive")
        self.assertFalse(result["qualified"])

    def test_tuning_selection_uses_lowest_p95_and_earliest_tie(self):
        report = self.report()
        rows = [{"step": step, "checkpoint_sha256": letter * 64, "tuning": report}
                for step, letter in ((9, "a"), (3, "b"), (6, "c"))]
        selected = select_tuning_checkpoint(rows)
        self.assertEqual(selected["step"], 3)
        self.assertEqual(selected["checkpoint_sha256"], "b" * 64)
        self.assertEqual(selected["tuning_report_sha256"], sha256_json(report))

    def test_selection_refuses_final_data_failed_reports_and_changed_schedule(self):
        base = {"step": 1, "checkpoint_sha256": "a" * 64, "tuning": self.report()}
        for field, value in (("partition", "final"), ("complete", False), ("selection_metric", float("nan"))):
            with self.subTest(field=field):
                changed = deepcopy(base)
                changed["tuning"][field] = value
                with self.assertRaises(ContractError):
                    select_tuning_checkpoint([changed])
        changed = deepcopy(base)
        changed["step"] = 2
        changed["tuning"]["schedule_sha256"] = "b" * 64
        with self.assertRaises(ContractError):
            select_tuning_checkpoint([base, changed])

    def test_candidate_freeze_requires_three_distinct_selected_seeds(self):
        frozen, _ = self.frozen()
        selected = {entry["seed"]: {k: v for k, v in entry.items() if k != "seed"}
                    for entry in frozen["candidates"]}
        for mutation in ("missing", "duplicate"):
            with self.subTest(mutation=mutation):
                changed = deepcopy(selected)
                if mutation == "missing":
                    changed.pop(SEEDS[0])
                else:
                    changed[SEEDS[1]]["checkpoint_sha256"] = changed[SEEDS[0]]["checkpoint_sha256"]
                with self.assertRaises(ContractError):
                    freeze_candidates(changed, experiment="fixture", dataset_manifest_sha256="d" * 64,
                                      final_suite_identity_sha256="e" * 64, registry_id="fixture")

    def test_final_opening_binds_all_seeds_suite_registry_and_manifest(self):
        frozen, opening = self.frozen()
        self.assertEqual(validate_final_opening(opening, frozen, seed=SEEDS[1])["checkpoint_sha256"], "b" * 64)
        for name, value in (
            ("frozen_candidates_sha256", ["a" * 64]),
            ("suite_identity_sha256", "f" * 64),
            ("registry_id", "another-registry"),
            ("dataset_manifest_sha256", "f" * 64),
        ):
            with self.subTest(name=name):
                changed = {k: v for k, v in opening.items() if k != "sha256"}
                changed[name] = value
                with self.assertRaises(ContractError):
                    validate_final_opening(seal(changed), frozen, seed=SEEDS[0])

    def test_final_requires_complete_frozen_schedule_and_valid_receipt(self):
        frozen, opening = self.frozen()
        ids = [sample.task.id for sample in self.final]
        predictions = self.predictions(self.final)
        for schedule in (None, ids[:-1]):
            with self.subTest(schedule_length=None if schedule is None else len(schedule)):
                with self.assertRaises(ContractError):
                    score_raw(self.final, predictions, self.config, partition="final",
                              expected_task_ids=schedule, final_opening=opening,
                              frozen_candidates=frozen, seed=SEEDS[0])
        report = score_raw(self.final, predictions, self.config, partition="final",
                           expected_task_ids=ids, final_opening=opening,
                           frozen_candidates=frozen, seed=SEEDS[0])
        self.assertTrue(report["complete"])
        self.assertGreater(report["withheld"]["scheduled"], 0)
        self.assertEqual(report["frozen_checkpoint_sha256"], "a" * 64)

    def test_final_refuses_tampered_freeze(self):
        frozen, opening = self.frozen()
        frozen["candidates"][0]["step"] = 999
        with self.assertRaises(ContractError):
            validate_final_opening(opening, frozen, seed=SEEDS[0])

    def test_partial_seed_aggregate_is_never_qualified(self):
        with self.assertRaises(ContractError):
            aggregate_comparisons({SEEDS[0]: {}})

    def test_fixture_timing_cannot_promote_utility(self):
        result = evaluate_comparison(self.tuning, self.config, trained=self.analytic, untrained=self.analytic)
        with self.assertRaises(ContractError):
            apply_cpu_utility(result, {"schema": "nca-ik-matched-cpu-latency/v1", "fixture": True})


@unittest.skipUnless(OPTIONAL, "optional pinned IK Torch/NumPy runtime is not installed")
class IKScoringTensorTests(unittest.TestCase):
    def test_cpu_predictor_has_complete_features_and_batch_one_output(self):
        from nca.ik.device import configure_device
        from nca.ik.models import build_model
        context = configure_device("cpu", 41011)
        normalization = Normalization(1, (0.,) * 14, (1.,) * 14, "a" * 64)
        model = build_model(context, normalization)
        inputs = RuntimeInputs(runner_right_sole_profile().sha256, (0., 0.), (4.6, 9.), (0., 0.))
        predictor = model_predictor(context, model)
        prediction = predictor(inputs)
        self.assertIsInstance(prediction, Prediction)
        self.assertEqual(len(prediction.raw_reference), 2)
        self.assertLessEqual(abs(prediction.raw_reference[0]), .72)
        self.assertLessEqual(abs(prediction.raw_reference[1]), .27)

    def test_small_timing_fixture_retains_scope_and_measured_counts(self):
        from nca.ik.device import configure_device
        context = configure_device("cpu", 41011)
        inputs = RuntimeInputs(runner_right_sole_profile().sha256, (0., 0.), (4.6, 9.), (0., 0.))
        predictors = {name: lambda inputs: Prediction((0., 0.)) for name in COMPARATORS}
        report = measure_cpu_latency(
            [inputs], predictors, context=context, environment={"identity": "fixture-only"},
            warmup_calls=1, measured_calls=3, fixture=True,
        )
        self.assertFalse(report["qualified"])
        self.assertTrue(report["fixture"])
        self.assertEqual({row["measured_calls"] for row in report["comparators"].values()}, {3})
        self.assertTrue(all(row["outcome_counts"] == {"complete": 3} for row in report["comparators"].values()))
        self.assertIsNone(report["memory_bytes"])


if __name__ == "__main__":
    unittest.main()
