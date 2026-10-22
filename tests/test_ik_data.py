"""Small dataset conformance: replay, leakage, evidence and permitted inputs."""

from dataclasses import replace
from pathlib import Path
import math
import unittest
from unittest.mock import patch

from nca.contracts import ContractError, sha256_json
from nca.ik.data import (
    CURRICULUM, FEATURE_NAMES, PARTITIONS, STRESS_CLASSES, DatasetConfig,
    GeometricSample, RuntimeInputs, TaskState, augment_task, build_split_plan,
    coverage_report, fit_normalization, generate_sample, iter_task_batches,
    iter_tasks, replay_task, runtime_features, validate_dataset, validate_sample,
    validate_task,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "ik" / "datasets"


class DatasetConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = DatasetConfig.from_json((CONFIGS / "tiny_v1.json").read_bytes())

    def test_default_declaration_preserves_accepted_minima_without_generation(self):
        config = DatasetConfig.from_json((CONFIGS / "runner_v1.json").read_bytes())
        plan = build_split_plan(config)
        self.assertEqual(config.mode, "prospective")
        self.assertEqual(config.split_seed, 61003)
        self.assertEqual(config.expected_tasks, 10220)
        self.assertEqual(plan["grid_shape"], [10, 8])
        self.assertGreaterEqual(config.points_per_bin[2] * 80, 2000)
        self.assertGreaterEqual(config.points_per_bin[2] * len(config.withheld_bins), 200)
        self.assertGreaterEqual(config.trajectories_per_partition, 40)
        self.assertGreaterEqual(config.trajectory_steps, 20)
        self.assertGreaterEqual(config.stress_per_class, 20)
        self.assertEqual(config.expected_tasks, config.budget.max_unique_tasks)
        self.assertGreaterEqual(config.budget.max_teacher_iterations,
                                config.expected_tasks * config.teacher.max_iterations)
        # This test checks the recipe, not a production corpus or learned result.
        self.assertEqual(self.config.mode, "fixture")
        self.assertEqual(self.config.expected_tasks, 254)
        self.assertEqual(set(STRESS_CLASSES), {
            "inner_limit_boundary", "hard_limit_boundary", "certified_geometric_infeasible",
            "unsupported_profile_or_effector", "missing_measurement_or_inactive_dof",
            "nonfinite_or_malformed_input", "iteration_exhaustion",
        })

    def test_exact_family_splits_and_withheld_regions_precede_sampling(self):
        plan = build_split_plan(self.config)
        families = plan["family_assignments"]
        self.assertEqual([len(families[p]) for p in PARTITIONS], [70, 15, 15])
        self.assertEqual(len(set().union(*(set(families[p]) for p in PARTITIONS))), 100)
        for left, right in (("training", "tuning"), ("training", "final"), ("tuning", "final")):
            self.assertFalse(set(families[left]) & set(families[right]))
        withheld = [r for r in plan["regions"] if r["kind"] == "withheld_region"]
        self.assertEqual(len(withheld), 8)
        self.assertEqual({r["bin"] for r in withheld}, set(self.config.withheld_bins))
        self.assertTrue(all(r["partition"] == "final" for r in withheld))
        self.assertEqual(plan["curriculum"], CURRICULUM)
        self.assertEqual(plan, build_split_plan(self.config))

    def test_strict_codec_refuses_unknown_fields_boolean_counts_and_seed_changes(self):
        for changes in ({"hidden_teacher_feature": True}, {"points_per_bin": [True, 1, 1]},
                        {"split_seed": 61004}, {"withheld_bins": [6] * 8},
                        {"point_guard_cells": 0}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                DatasetConfig.from_dict(self.config.to_dict() | changes)
        self.assertEqual(DatasetConfig.from_json(self.config.to_json()), self.config)

    def test_prospective_minima_and_resource_ceilings_cannot_be_silently_lowered(self):
        with self.assertRaises(ContractError):
            replace(self.config, mode="prospective")
        for budget in (replace(self.config.budget, max_unique_tasks=self.config.expected_tasks - 1),
                       replace(self.config.budget, max_teacher_iterations=1)):
            with self.subTest(budget=budget), self.assertRaises(ContractError):
                replace(self.config, budget=budget)
        with self.assertRaises(ContractError):
            replace(self.config, hip_lanes=((.06, .5), (.51, .7), (.82, .94)))


class DatasetConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = DatasetConfig.from_json((CONFIGS / "tiny_v1.json").read_bytes())
        cls.tasks = tuple(iter_tasks(cls.config))
        cls.samples = tuple(generate_sample(task, cls.config) for task in cls.tasks)
        cls.by_partition = {p: tuple(s for s in cls.samples if s.task.partition == p) for p in PARTITIONS}

    def test_tiny_schedule_batches_and_curriculum_have_exact_denominators(self):
        self.assertEqual(len(self.tasks), self.config.expected_tasks)
        self.assertEqual(len({t.id for t in self.tasks}), self.config.expected_tasks)
        self.assertEqual([len(self.by_partition[p]) for p in PARTITIONS], [82, 82, 90])
        self.assertEqual([t.stage for t in self.tasks], sorted(t.stage for t in self.tasks))
        batches = tuple(iter_task_batches(self.config))
        self.assertTrue(all(0 < len(batch) <= self.config.batch_size for batch in batches))
        self.assertEqual(tuple(task for batch in batches for task in batch), self.tasks)
        validate_dataset(self.samples, self.config, require_complete=True)

    def test_generator_state_replays_in_arbitrary_order_and_through_json(self):
        for task in self.tasks[::17][::-1]:
            with self.subTest(state=task.state):
                replayed = replay_task(self.config, task.state.to_dict())
                self.assertEqual(replayed, task)
                self.assertEqual(replayed.to_json(), task.to_json())
                self.assertEqual(replayed.id, task.id)
                self.assertEqual(replayed.inputs.sha256, task.input_sha256)
                self.assertEqual(TaskState.from_json(task.state.to_json()), task.state)
        with self.assertRaises(ContractError):
            replay_task(self.config, TaskState("point", "training", 0, 1))
        with self.assertRaises(ContractError):
            replay_task(self.config, TaskState("point", "training", self.config.withheld_bins[0], 0))

    def test_frame_augmentation_inherits_family_group_split_and_canonical_target(self):
        original = self.tasks[0]
        base_sample = self.samples[0]
        augmented_samples = [base_sample]
        for variant in (1, 2):
            augmented = augment_task(original, self.config, variant)
            self.assertNotEqual(augmented.id, original.id)
            self.assertNotEqual(augmented.original_point, original.original_point)
            self.assertEqual(augmented.inputs, original.inputs)
            self.assertEqual(augmented.input_sha256, original.input_sha256)
            self.assertEqual(augmented.partition, original.partition)
            self.assertEqual(augmented.family_id, original.family_id)
            self.assertEqual(augmented.trajectory_id, original.trajectory_id)
            sample = generate_sample(augmented, self.config)
            validate_sample(sample, self.config)
            augmented_samples.append(sample)
        validate_dataset(augmented_samples, self.config)
        with self.assertRaises(ContractError):
            augment_task(original, self.config, 3)

    def test_renaming_or_reassigning_groups_cannot_hide_duplicates(self):
        sample = self.samples[0]
        with self.assertRaises(ContractError):
            validate_dataset([sample, sample], self.config)
        for task in (replace(sample.task, id=sha256_json("renamed-copy")),
                     replace(sample.task, family_id="different-family"),
                     replace(sample.task, trajectory_id=sha256_json("another-path"))):
            with self.subTest(task=task.id), self.assertRaises(ContractError):
                validate_dataset([sample, replace(sample, task=task)], self.config)

    def test_partition_change_after_augmentation_or_generation_is_rejected(self):
        final_sample = self.by_partition["final"][0]
        for variant in (0, 1, 2):
            task = augment_task(final_sample.task, self.config, variant)
            counterfeit = replace(task, partition="training",
                                  state=replace(task.state, partition="training"))
            with self.subTest(variant=variant), self.assertRaises(ContractError):
                validate_task(counterfeit, self.config)
            with self.assertRaises(ContractError):
                fit_normalization([replace(final_sample, task=counterfeit)], self.config)

    def test_nearby_canonical_targets_cannot_be_smuggled_between_partitions(self):
        training = self.by_partition["training"][0]
        final = self.by_partition["final"][0]
        nearby = (training.task.inputs.target_root[0] + self.config.point_guard_cells / 2,
                  training.task.inputs.target_root[1])
        changed = replace(final.task, inputs=replace(final.task.inputs, target_root=nearby))
        with self.assertRaises(ContractError):
            validate_dataset([training, replace(final, task=changed)], self.config)

    def test_partial_trajectories_and_missing_cases_never_pass_complete_validation(self):
        with self.assertRaises(ContractError):
            validate_dataset(self.samples[:-1], self.config, require_complete=True)
        reduced = tuple(s for s in self.samples
                        if not (s.task.state.kind == "trajectory" and s.task.state.step == 1))
        with self.assertRaises(ContractError):
            validate_dataset(reduced, self.config, require_complete=True)
        validate_dataset(reduced, self.config, require_complete=False)

    def test_teacher_and_frame_corruption_are_detected_without_relabelling(self):
        sample = self.samples[0]
        self.assertTrue(sample.label.pose_loss)
        bad_teacher = replace(sample.label.teacher, point=(0., 0.))
        for label in (replace(sample.label, teacher=bad_teacher),
                      replace(sample.label, branch_id="positive"),
                      replace(sample.label, pose_loss=False)):
            with self.subTest(label=label), self.assertRaises(ContractError):
                validate_sample(replace(sample, label=label), self.config)
        with self.assertRaises(ContractError):
            validate_sample(replace(sample, task=replace(sample.task, original_point=(0., 0.))), self.config)
        self.assertEqual(GeometricSample.from_json(sample.to_json()), sample)

    def test_teacher_status_cannot_contradict_retained_work_and_residual(self):
        from nca.ik.data import _checked_label
        from nca.ik.kinematics import chain_from_profile
        from nca.ik.profiles import runner_right_sole_profile

        sample = self.samples[0]
        inputs = sample.task.inputs
        solver = self.config.teacher
        self.assertEqual(sample.label.teacher.status, "solved")
        self.assertLessEqual(sample.label.teacher.residual_cells, solver.tolerance_cells)
        self.assertGreater(solver.max_iterations, 0)
        self.assertGreater(solver.max_seconds, 0)
        chain = chain_from_profile(runner_right_sole_profile())
        for teacher in (
            replace(sample.label.teacher, status="iteration_budget_exhausted", iterations=0),
            replace(sample.label.teacher, status="best_effort"),
            replace(sample.label.teacher, status="time_budget_exhausted", elapsed_seconds=0.),
        ):
            # Recompute every derived label to isolate the terminal-status guard;
            # a stale loss flag or altered branch must not cause the refusal.
            label = _checked_label(chain, inputs.target_root, inputs.current, teacher, solver,
                lower=(-.72, -.27), upper=(.72, .27), present=inputs.present,
                measured=inputs.measured, task_mask=inputs.task_mask)
            self.assertTrue(label.verified)
            with self.subTest(status=teacher.status), self.assertRaises(ContractError) as caught:
                validate_sample(replace(sample, label=label), self.config)
            self.assertEqual(caught.exception.code, "ik_dataset_teacher")

    def test_report_retains_failed_teacher_denominators_and_no_hardware_claims(self):
        report = coverage_report(self.samples, self.config)
        self.assertEqual(report["observed_samples"], self.config.expected_tasks)
        self.assertFalse(report["accepted_final_population_minima_met"])
        self.assertFalse(report["training_performed"])
        self.assertFalse(report["cuda_qualified"])
        self.assertFalse(report["physical_qualified"])
        self.assertFalse(report["final_scoring_opened"])
        self.assertEqual(report["runner_singular_cases"], 0)
        self.assertEqual(report["runner_multiple_branch_cases"], 0)
        for partition in PARTITIONS:
            rows = self.by_partition[partition]
            actual = report["partitions"][partition]
            self.assertEqual(actual["teacher_failure_denominator"], len(rows))
            self.assertEqual(sum(actual["teacher_status_counts"].values()), len(rows))
            self.assertEqual(actual["retained_teacher_failures"], len(STRESS_CLASSES) - 1)
            self.assertEqual(actual["stress_populations"], dict.fromkeys(STRESS_CLASSES, 1))
            self.assertEqual(actual["status_loss_samples"], len(rows))
            self.assertEqual(actual["nominal_denominator"], len(rows) - len(STRESS_CLASSES))
            self.assertEqual(actual["complete_trajectories"], 1)
            self.assertEqual(actual["trajectory_steps"], 3)
        self.assertEqual(report["partitions"]["final"]["withheld_point_queries"], 8)

    def test_failure_labels_keep_candidate_corrections_without_pose_loss(self):
        rows = [s for s in self.by_partition["training"] if s.task.state.kind == "stress"]
        classes = {s.task.state.stress_class: s for s in rows}
        exhausted = classes["iteration_exhaustion"]
        self.assertEqual(exhausted.label.teacher.status, "iteration_budget_exhausted")
        self.assertEqual(exhausted.label.analytic.status, "solved")
        self.assertIsNotNone(exhausted.label.teacher.configuration)
        self.assertFalse(exhausted.label.pose_loss)
        unreachable = classes["certified_geometric_infeasible"]
        self.assertEqual(unreachable.label.analytic.status, "geometrically_infeasible")
        self.assertNotEqual(unreachable.label.teacher.status, "geometrically_infeasible")
        self.assertFalse(unreachable.label.pose_loss)
        self.assertFalse(unreachable.label.endpoint_loss)
        self.assertTrue(any(s.label.teacher.correction_norm > 0 for s in self.samples))

    def test_hard_limit_targets_do_not_acquire_inner_command_labels(self):
        boundary = {s.task.state.stress_class: s for s in self.by_partition["training"]
                    if s.task.state.stress_class in ("inner_limit_boundary", "hard_limit_boundary")}
        inner = boundary["inner_limit_boundary"]
        hard = boundary["hard_limit_boundary"]
        self.assertAlmostEqual(abs(inner.task.target_configuration[0]), .72, places=12)
        self.assertAlmostEqual(abs(hard.task.target_configuration[0]), .75, places=12)
        self.assertTrue(inner.label.pose_loss)
        self.assertEqual(inner.label.analytic.status, "solved")
        self.assertFalse(hard.label.pose_loss)
        self.assertEqual(hard.label.analytic.status, "joint_limited")
        self.assertLessEqual(abs(hard.label.teacher.configuration[0]), .72)
        self.assertNotEqual(inner.task.region_kind, "interpolation")
        self.assertNotEqual(hard.task.region_kind, "interpolation")

    def test_malformed_original_is_refused_without_solving_its_finite_placeholder(self):
        task = next(t for t in self.tasks if t.state.stress_class == "nonfinite_or_malformed_input")
        self.assertIn("NaN", task.invalid_specimen)
        with patch("nca.ik.data.solve_numeric", side_effect=AssertionError("must not solve a substitute")):
            sample = generate_sample(task, self.config)
            validate_sample(sample, self.config)
        self.assertEqual(sample.label.teacher.status, "invalid_input")
        self.assertEqual(sample.label.teacher.reason, "nonfinite")
        self.assertIsNone(sample.label.teacher.configuration)
        self.assertFalse(sample.label.pose_loss)
        self.assertFalse(sample.label.endpoint_loss)
        self.assertTrue(sample.label.status_loss)

    def test_runtime_features_have_exact_whitelist_and_no_privileged_sample_fields(self):
        sample = self.samples[0]
        features = runtime_features(sample.task.inputs)
        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(v) for v in features))
        self.assertEqual(features[:6], (*sample.task.inputs.current,
                                       *sample.task.inputs.target_root, *sample.task.inputs.posture))
        for field in ("target_configuration", "teacher", "analytic", "branch_id",
                      "partition", "region_kind", "pose_loss", "iterations"):
            with self.subTest(field=field), self.assertRaises(ContractError):
                RuntimeInputs.from_dict(sample.task.inputs.to_dict() | {field: 0})
        for wrong in (sample, sample.to_dict(), sample.task.inputs.to_dict()):
            with self.subTest(type=type(wrong)), self.assertRaises(ContractError):
                runtime_features(wrong)

    def test_missing_measurements_remain_null_and_are_distinguishable_from_zero(self):
        missing = next(s for s in self.samples if s.task.state.stress_class == "missing_measurement_or_inactive_dof")
        inputs = missing.task.inputs
        self.assertIsNone(inputs.current[0])
        self.assertFalse(inputs.measured[0])
        vector = runtime_features(inputs)
        self.assertEqual(vector[0], 0.)
        self.assertEqual(vector[8], 0.)
        measured_zero = replace(inputs, current=(0., inputs.current[1]), measured=(True, True))
        self.assertNotEqual(runtime_features(measured_zero), vector)
        with self.assertRaises(ContractError):
            replace(inputs, current=(0., inputs.current[1]))
        unsupported = next(s for s in self.samples if s.task.state.stress_class == "unsupported_profile_or_effector")
        with self.assertRaises(ContractError):
            runtime_features(unsupported.task.inputs)

    def test_normalization_uses_only_valid_training_inputs_with_exact_lineage(self):
        training = self.by_partition["training"]
        normalizer = fit_normalization(training, self.config)
        self.assertEqual(normalizer.count, 76)
        self.assertEqual(normalizer.feature_names, FEATURE_NAMES)
        self.assertEqual(normalizer, fit_normalization(training[::-1], self.config))
        self.assertTrue(all(math.isfinite(v) and v > 0 for v in normalizer.scale))
        self.assertTrue(all(math.isfinite(v) for v in normalizer.apply(training[0].task.inputs)))
        for bad in (self.by_partition["tuning"], self.by_partition["final"],
                    training + self.by_partition["tuning"][:1], training + training[:1], ()):
            with self.subTest(length=len(bad)), self.assertRaises(ContractError):
                fit_normalization(bad, self.config)

    def test_normalization_selection_does_not_depend_on_teacher_success(self):
        ordinary = self.by_partition["training"][:3]
        timed_config = replace(self.config, teacher=replace(self.config.teacher, max_seconds=0.))
        timed_samples = tuple(generate_sample(replay_task(timed_config, s.task.state), timed_config)
                              for s in ordinary)
        self.assertTrue(all(s.label.teacher.status == "time_budget_exhausted" for s in timed_samples))
        self.assertTrue(all(not s.label.pose_loss for s in timed_samples))
        first = fit_normalization(ordinary, self.config)
        second = fit_normalization(timed_samples, timed_config)
        self.assertEqual(first.count, second.count)
        self.assertEqual(first.mean, second.mean)
        self.assertEqual(first.scale, second.scale)
        self.assertNotEqual(first.training_inputs_sha256, second.training_inputs_sha256)


if __name__ == "__main__":
    unittest.main()
