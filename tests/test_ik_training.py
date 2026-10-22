"""Small CPU tests for the IK training foundation, not a quality experiment."""

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from nca.contracts import ContractError
from nca.ik.data import (DatasetConfig, TaskState, fit_normalization,
                         generate_sample, replay_task, runtime_features)
from nca.ik.device import configure_device
from nca.ik.kinematics import chain_from_profile, cpu_fk
from nca.ik.profiles import runner_right_sole_profile
from nca.ik.reference import SolverConfig
from nca.train.ik import (IKTrainer, LENGTH_CELLS, LEARNING_RATE, LossWeights,
                          MAX_GRADIENT_NORM, prepare_training_batch)


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None


class IKTrainingContractTests(unittest.TestCase):
    def test_registered_discovery_does_not_import_optional_runtime(self):
        program = """
import sys
sys.path.insert(0, sys.argv[1])
from nca.builtins import builtin_registry
import nca.train.ik
registry = builtin_registry()
assert registry.describe("policy", "runner-compact-ik", 1)
assert registry.describe("trainer", "runner-compact-ik-step", 1)
assert "torch" not in sys.modules
assert "numpy" not in sys.modules
"""
        subprocess.run([sys.executable, "-S", "-c", program, str(ROOT / "src")],
                       check=True, capture_output=True)

    def test_candidate_records_scope_and_matches_implemented_loss(self):
        candidate = json.loads((ROOT / "configs/ik/experiments/runner-compact-v1.json").read_text())
        self.assertEqual(candidate["status"], "software_candidate_not_execution_declaration")
        self.assertFalse(candidate["measured_run_admitted"])
        self.assertEqual(candidate["training_seeds"], [41011, 42013, 43019])
        self.assertEqual(candidate["mode"], "direct_joint_position")
        self.assertEqual(candidate["loss"]["weights"], LossWeights().to_dict())
        self.assertEqual(candidate["optimizer"]["learning_rate"], LEARNING_RATE)
        self.assertEqual(candidate["optimizer"]["max_gradient_norm"], MAX_GRADIENT_NORM)
        self.assertEqual(candidate["loss"]["endpoint_length_cells"], LENGTH_CELLS)
        config = DatasetConfig.from_json((ROOT / "configs/ik/datasets/runner_v1.json").read_bytes())
        self.assertEqual(candidate["dataset"]["config_sha256"], config.sha256)
        self.assertEqual(candidate["numerical_correction_iterations"], 0)
        self.assertEqual(candidate["full_reference_fallback_calls"], 0)

    def test_invalid_loss_weights_are_refused_without_torch(self):
        for changes in ({"joint": 0.0}, {"endpoint": -1.0}, {"step": float("nan")},
                        {"posture": True}, {"bounds": -1.0}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                LossWeights.from_dict(LossWeights().to_dict() | changes)


@unittest.skipUnless(OPTIONAL, "optional pinned IK Torch/NumPy runtime is not installed")
class IKTrainingCPUConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate only these tiny fixture rows. No corpus archive or final
        # registry is accessed, and no learned-quality benchmark is run.
        cls.config = DatasetConfig(mode="fixture", points_per_bin=(1, 1, 1),
                                   trajectories_per_partition=1, trajectory_steps=20,
                                   stress_per_class=1,
                                   teacher=SolverConfig(tolerance_cells=1e-7, max_seconds=1.0))
        addresses = [TaskState("point", "training", b, 0) for b in (0, 8, 16, 32)]
        addresses += [TaskState("trajectory", "training", 0, 0, step=i) for i in (0, 1, 2)]
        addresses += [TaskState("stress", "training", 0, 0, stress_class=name) for name in (
            "iteration_exhaustion", "missing_measurement_or_inactive_dof",
            "unsupported_profile_or_effector", "nonfinite_or_malformed_input")]
        cls.samples = tuple(generate_sample(replay_task(cls.config, a), cls.config) for a in addresses)
        cls.normalization = fit_normalization(cls.samples, cls.config)

    def setUp(self):
        self.context = configure_device("cpu", 41011)
        self.torch = self.context.torch
        self.trainer = IKTrainer(self.context, self.normalization, self.config)
        self.batch = prepare_training_batch(self.samples, self.config, self.context)

    def test_only_training_rows_are_accepted_before_label_work(self):
        original = self.samples[0]
        for partition in ("tuning", "final"):
            state = replace(original.task.state, partition=partition)
            foreign = replace(original, task=replace(original.task, partition=partition, state=state))
            with self.subTest(partition=partition), patch(
                    "nca.train.ik.validate_sample", side_effect=AssertionError("must reject first")):
                with self.assertRaisesRegex(ContractError, "training"):
                    prepare_training_batch((foreign,), self.config, self.context)
        with self.assertRaises(ContractError):
            prepare_training_batch((self.samples[0].to_dict(),), self.config, self.context)
        with self.assertRaises(ContractError):
            prepare_training_batch((original, original), self.config, self.context)

    def test_retained_failures_stay_in_batch_with_separate_supervision_masks(self):
        self.assertEqual(self.batch.features.shape, (11, 14))
        self.assertEqual(sum(self.batch.teacher_status_counts.values()), 11)
        self.assertEqual(len(self.batch.sample_ids), 11)
        # The exhausted solver has valid reachable geometry but no pose target.
        self.assertEqual(self.samples[7].label.teacher.status, "iteration_budget_exhausted")
        self.assertFalse(bool(self.batch.pose_mask[7]))
        self.assertTrue(bool(self.batch.endpoint_mask[7]))
        self.assertTrue(self.torch.equal(self.batch.teacher_reference[7], self.torch.zeros(2)))
        self.assertFalse(bool(self.batch.endpoint_mask[8:].any()))
        expected = self.torch.tensor(runtime_features(self.samples[0].task.inputs), dtype=self.torch.float32)
        self.torch.testing.assert_close(self.batch.features[0], expected, rtol=0, atol=0)
        self.assertEqual(self.trainer.loss(self.batch)[2].valid.tolist()[8:], [False, False, False])

    def test_endpoint_loss_matches_independent_binary64_geometry(self):
        total, terms, prediction, _ = self.trainer.loss(self.batch)
        chain = chain_from_profile(runner_right_sole_profile())
        raw = prediction.reference.detach().cpu().tolist()
        targets = self.batch.features[:, 2:4].cpu().tolist()
        mask = self.batch.endpoint_mask.cpu().tolist()
        expected = [math.dist(cpu_fk(chain, q).point, target) ** 2 / LENGTH_CELLS ** 2
                    for q, target, enabled in zip(raw, targets, mask) if enabled]
        self.assertGreater(len(expected), 0)
        self.assertAlmostEqual(float(terms["endpoint"].detach()), sum(expected) / len(expected), places=6)
        self.assertTrue(math.isfinite(float(total.detach())))
        self.assertEqual(float(terms["bounds"].detach()), 0.0)

    def test_real_adjacent_trajectory_pairs_exclude_random_starts_and_gaps(self):
        self.assertEqual(self.batch.continuity_pairs.tolist(), [[4, 5], [5, 6]])
        gap = prepare_training_batch((self.samples[4], self.samples[6], self.samples[0]),
                                     self.config, self.context)
        self.assertEqual(gap.continuity_pairs.shape, (0, 2))
        shuffled = prepare_training_batch((self.samples[6], self.samples[4], self.samples[5]),
                                          self.config, self.context)
        self.assertEqual(sorted(map(tuple, shuffled.continuity_pairs.tolist())), [(1, 2), (2, 0)])
        for i, j in self.batch.continuity_pairs.tolist():
            a, b = self.samples[i].task, self.samples[j].task
            self.assertEqual(a.trajectory_id, b.trajectory_id)
            self.assertEqual(a.family_id, b.family_id)
            self.assertEqual(b.state.step - a.state.step, 1)
            self.assertLessEqual(math.dist(a.inputs.target_root, b.inputs.target_root), .10)

    def test_two_cpu_updates_change_weights_and_verify_every_device_category(self):
        before = {n: p.detach().clone() for n, p in self.trainer.model.named_parameters()}
        receipts = [self.trainer.step(self.batch) for _ in range(2)]
        self.assertTrue(any(not self.torch.equal(before[n], p) for n, p in self.trainer.model.named_parameters()))
        for index, receipt in enumerate(receipts, 1):
            self.assertEqual(receipt["committed_steps"], index)
            self.assertEqual(receipt["attempted_steps"], index)
            self.assertEqual(receipt["sample_presentations"], index * 11)
            self.assertTrue(receipt["weights_changed"])
            self.assertTrue(receipt["placement"]["all_on_selected_device"])
            self.assertTrue(receipt["placement"]["finite_checked"])
            self.assertIn("optimizer", receipt["placement"]["tensor_counts"])
            self.assertIn("gradients", receipt["placement"]["tensor_counts"])
            self.assertEqual(receipt["refused_input_rows"], 3)
            self.assertEqual(receipt["correction_iterations"], 0)
            self.assertEqual(receipt["fallback_calls"], 0)
            self.assertTrue(all(math.isfinite(v) for v in receipt["terms"].values()))

    def test_unlabelled_batch_fails_without_weights_update_and_cannot_resume_silently(self):
        invalid = prepare_training_batch(self.samples[8:], self.config, self.context)
        before = [p.detach().clone() for p in self.trainer.model.parameters()]
        with self.assertRaisesRegex(ContractError, "supervision"):
            self.trainer.step(invalid)
        self.assertEqual(self.trainer.committed_steps, 0)
        self.assertEqual(self.trainer.attempted_steps, 1)
        self.assertEqual(self.trainer.sample_presentations, 3)
        self.assertTrue(self.trainer.failed)
        self.assertTrue(all(self.torch.equal(a, b) for a, b in zip(before, self.trainer.model.parameters())))
        with self.assertRaisesRegex(ContractError, "cannot continue"):
            self.trainer.step(self.batch)

    def test_nonfinite_gradient_is_rejected_before_adam_mutates_weights(self):
        before = [p.detach().clone() for p in self.trainer.model.parameters()]
        hook = next(self.trainer.model.parameters()).register_hook(lambda g: g * float("nan"))
        try:
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                self.trainer.step(self.batch)
        finally:
            hook.remove()
        self.assertTrue(self.trainer.failed)
        self.assertEqual(self.trainer.committed_steps, 0)
        self.assertTrue(all(self.torch.equal(a, b) for a, b in zip(before, self.trainer.model.parameters())))

    def test_batch_dtype_and_device_are_never_silently_converted(self):
        for changed in (self.batch.features.double(), self.batch.features.to("meta")):
            with self.subTest(dtype=changed.dtype, device=changed.device), self.assertRaises(ContractError):
                self.trainer.loss(replace(self.batch, features=changed))
        with self.assertRaises(ContractError):
            self.trainer.loss(replace(self.batch, dataset_config_sha256="0" * 64))

    def test_existing_checkpoint_primitives_restore_model_normalizer_and_adam(self):
        from nca.ik.checkpoint import load_checkpoint, restore_checkpoint, write_checkpoint
        self.trainer.step(self.batch)
        metadata = {"config_sha256": self.config.sha256, "source_sha256": "a" * 64,
                    "profile_sha256": runner_right_sole_profile().sha256,
                    "dataset_sha256": "b" * 64,
                    "normalization_sha256": self.normalization.sha256}
        state = {"boundary": "optimizer_step", "committed_steps": 1,
                 "attempted_steps": 1, "sample_presentations": 11,
                 "evidence_kind": "tiny_software_fixture"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            receipt = write_checkpoint(path, self.trainer.model, self.trainer.optimizer,
                                       metadata=metadata, training_state=state)
            payload = load_checkpoint(path, expected_sha256=receipt["sha256"], expected_metadata=metadata)
            expected = self.trainer.step(self.batch)
            reference = {n: t.detach().clone() for n, t in self.trainer.model.state_dict().items()}
            fresh = IKTrainer(configure_device("cpu", 43019), self.normalization, self.config)
            restored, limits = restore_checkpoint(payload, fresh.model, fresh.optimizer)
            self.assertEqual(restored, state)
            self.assertEqual(limits, [])
            # This explicitly exercises tensor restoration, not an implemented
            # persistent run ledger or an admitted continuation.
            fresh.committed_steps = restored["committed_steps"]
            fresh.attempted_steps = restored["attempted_steps"]
            fresh.sample_presentations = restored["sample_presentations"]
            actual = fresh.step(self.batch)
            self.assertEqual(actual["loss"], expected["loss"])
            self.assertEqual(actual["sample_presentations"], 22)
            for name, tensor in fresh.model.state_dict().items():
                self.assertTrue(self.torch.equal(tensor, reference[name]), name)


if __name__ == "__main__":
    unittest.main()
