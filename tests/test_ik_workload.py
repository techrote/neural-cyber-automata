"""Tiny IK workload conformance, not learned-quality or device qualification.

CPU cases perform at most three small optimizer steps per trajectory.  The
manual CUDA case is opt-in, performs two steps, and fails (without a CPU fallback)
if an explicitly requested device/runtime is unavailable.  It is skipped in
routine CPU CI and has not been executed merely because these tests pass.
"""

from collections.abc import Mapping
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import tempfile
import unittest

from nca.contracts import sha256_json
from nca.ik.checkpoint import (capture_rng, load_checkpoint, restore_checkpoint,
                               write_checkpoint)
from nca.ik.device import configure_device
from nca.ik.profiles import nominal_pose, runner_right_sole_profile
from nca.ik.workload import (NORMALIZATION, SPLIT, SmokeWorkload,
                            dataset_identity, model_digest, runner_points)


OPTIONAL = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None
CUDA_SELECTION = os.environ.get("NCA_IK_CUDA_TEST_DEVICE")


def fixture_config(**overrides):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/ik/runtime-smoke-v1.json").read_text())
    config.update(batch_size=7, steps=3)
    config.update(overrides)
    return config


def tensors(value, torch):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from tensors(child, torch)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensors(child, torch)


class IKWorkloadIdentityTests(unittest.TestCase):
    def test_fixed_generator_split_and_normalization_have_content_identity(self):
        config = fixture_config()
        identity = dataset_identity(config)
        self.assertEqual(identity["size"], 256)
        self.assertEqual(identity["generator"], "runner-16x16-lattice/v1")
        self.assertEqual(identity["split"], SPLIT)
        self.assertEqual(identity["normalization"], NORMALIZATION)
        self.assertEqual(identity["sha256"],
                         sha256_json({key: value for key, value in identity.items() if key != "sha256"}))
        self.assertNotEqual(identity["sha256"], dataset_identity({**config, "seed": config["seed"] + 1})["sha256"])
        # Changing optimizer settings changes the run config, not the dataset.
        self.assertEqual(identity, dataset_identity({**config, "learning_rate": .002}))
        self.assertEqual(identity["split"]["evaluation"], ["conformance-only-no-quality-holdout"])


@unittest.skipUnless(OPTIONAL, "optional pinned IK Torch/NumPy runtime is not installed")
class IKWorkloadCPUConformanceTests(unittest.TestCase):
    def setUp(self):
        self.config = fixture_config()
        self.context = configure_device("cpu", self.config["seed"])
        self.torch = self.context.torch

    def assertNestedEqual(self, left, right):
        if isinstance(left, self.torch.Tensor):
            self.assertIsInstance(right, self.torch.Tensor)
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.device, right.device)
            self.assertTrue(self.torch.equal(left, right))
        elif isinstance(left, Mapping):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assertNestedEqual(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for actual, expected in zip(left, right):
                self.assertNestedEqual(actual, expected)
        else:
            self.assertEqual(left, right)

    def run_steps(self, work, count):
        rows = []
        for _ in range(count):
            ids, batch = work.batch()
            rows.append((ids, work.step(batch)))
        return rows

    def test_fp32_compact_fk_agrees_with_independent_anchor_profile(self):
        profile, torch = runner_right_sole_profile(), self.torch
        pairs = [(h, a) for h in (-.72, -.6, 0., .6, .72) for a in (-.27, -.2, 0., .2, .27)]
        q = torch.tensor(pairs, dtype=torch.float32, device=self.context.device)
        actual = runner_points(q, torch)
        expected = torch.tensor([nominal_pose(profile, {"hip_1": h, "ankle_1": a}).point
                                 for h, a in pairs], dtype=torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0., atol=2e-6)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.device, self.context.device)
        # Independent neutral Jacobian from the accepted authored geometry.
        neutral = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(lambda value: runner_points(value, torch), neutral)
        torch.testing.assert_close(jacobian[0, :, 0, :],
                                   torch.tensor([[-6.2, -1.25], [1., 1.]]), rtol=0., atol=1e-6)

    def test_two_actual_inverse_steps_change_weights_with_all_finite_device_state(self):
        work, torch = SmokeWorkload(self.context, self.config), self.torch
        before = model_digest(work.model)
        for ids, receipt in self.run_steps(work, 2):
            self.assertEqual(len(ids), self.config["batch_size"])
            self.assertTrue(math.isfinite(receipt["loss"]))
            self.assertTrue(receipt["finite_gradients"])
            self.assertTrue(receipt["weights_changed"])
            self.assertTrue(receipt["placement"]["all_on_selected_device"])
            self.assertEqual(receipt["placement"]["device"], "cpu")
            self.assertEqual(set(receipt["placement"]["tensor_counts"]),
                             {"data", "inputs", "model", "output", "geometry", "loss", "gradients", "optimizer"})
        self.assertNotEqual(before, model_digest(work.model))
        for parameter in work.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertEqual(parameter.device, self.context.device)
        state_tensors = list(tensors(work.optimizer.state, torch))
        self.assertTrue(state_tensors)
        for value in state_tensors:
            self.assertEqual(value.device, self.context.device)
            self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertEqual((work.committed_steps, work.cursor), (2, 14))
        self.assertEqual(work.initial_model_sha256, before)

    def test_sampler_crosses_epoch_once_without_dropping_or_duplicating_positions(self):
        config = fixture_config(batch_size=136, steps=2)
        work, torch = SmokeWorkload(self.context, config), self.torch
        initial_permutation = work.permutation.tolist()
        first_ids, first_batch = work.batch()
        self.assertEqual(first_ids, initial_permutation[:136])
        # Generator cursor here is after the first current-posture draw.  The
        # next operation at the epoch edge is one independent randperm draw.
        independent_generator = torch.Generator(device="cpu")
        independent_generator.set_state(work.generator.get_state().clone())
        expected_next_permutation = torch.randperm(256, generator=independent_generator).tolist()
        work.step(first_batch)
        second_ids, second_batch = work.batch()
        work.step(second_batch)
        self.assertEqual(second_ids, initial_permutation[136:] + expected_next_permutation[:16])
        self.assertEqual(sorted(first_ids + second_ids[:120]), list(range(256)))
        self.assertEqual((work.epoch, work.position, work.cursor, work.committed_steps), (1, 16, 272, 2))
        self.assertEqual(work.state()["permutation"], expected_next_permutation)
        next_ids, _ = work.batch()
        self.assertEqual(next_ids, expected_next_permutation[16:152])

    def test_checkpoint_restores_real_workload_optimizer_dropout_sampler_counters_and_rng(self):
        torch = self.torch
        baseline = SmokeWorkload(self.context, self.config)
        baseline_rows = self.run_steps(baseline, 3)
        baseline_weights = copy.deepcopy(baseline.model.state_dict())
        baseline_optimizer = copy.deepcopy(baseline.optimizer.state_dict())
        baseline_state = baseline.state()
        baseline_rng = capture_rng(baseline.generators)
        baseline_next_ids, baseline_next_batch = baseline.batch()
        expected_draws = (random.random(), float(self.context.numpy.random.random()), torch.rand(3))

        # Begin the exact same trajectory, stop after one committed step, then
        # construct fresh components and deliberately disturb every RNG stream.
        context = configure_device("cpu", self.config["seed"])
        interrupted = SmokeWorkload(context, self.config)
        first_rows = self.run_steps(interrupted, 1)
        metadata = {"config_sha256": sha256_json(self.config), "source_sha256": "b" * 64,
                    "profile_sha256": runner_right_sole_profile().sha256,
                    "dataset_sha256": dataset_identity(self.config)["sha256"],
                    "environment_sha256": context.report["environment_sha256"]}
        allocation = {"attempts": 1, "reserved_steps": 1, "remaining_steps": 255,
                      "remaining_samples": 8185, "consumed_seconds": .1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "committed.pt"
            reference = write_checkpoint(path, interrupted.model, interrupted.optimizer,
                                         metadata=metadata, training_state=interrupted.state(allocation),
                                         generators=interrupted.generators)
            random.seed(991)
            context.numpy.random.seed(992)
            torch.manual_seed(993)
            resumed = SmokeWorkload(context, self.config)
            resumed.generator.manual_seed(994)
            payload = load_checkpoint(path, expected_sha256=reference["sha256"], expected_metadata=metadata)
            state, limitations = restore_checkpoint(payload, resumed.model, resumed.optimizer,
                                                    generators=resumed.generators)
            self.assertEqual(state["allocation"], allocation)
            resumed.restore_state(state)
            self.assertEqual(limitations, [])
            resumed_rows = first_rows + self.run_steps(resumed, 2)

        self.assertEqual([row[0] for row in resumed_rows], [row[0] for row in baseline_rows])
        self.assertEqual([row[1]["loss"] for row in resumed_rows], [row[1]["loss"] for row in baseline_rows])
        self.assertNestedEqual(resumed.model.state_dict(), baseline_weights)
        self.assertNestedEqual(resumed.optimizer.state_dict(), baseline_optimizer)
        self.assertEqual(resumed.state(), baseline_state)
        self.assertNestedEqual(capture_rng(resumed.generators), baseline_rng)
        next_ids, next_batch = resumed.batch()
        self.assertEqual(next_ids, baseline_next_ids)
        self.assertNestedEqual(next_batch, baseline_next_batch)
        actual_draws = (random.random(), float(context.numpy.random.random()), torch.rand(3))
        self.assertEqual(actual_draws[:2], expected_draws[:2])
        self.assertTrue(torch.equal(actual_draws[2], expected_draws[2]))

    def test_incompatible_or_malformed_sampler_state_is_refused_before_mutation(self):
        work = SmokeWorkload(self.context, self.config)
        self.run_steps(work, 1)
        original = work.state()
        replacements = (
            {"boundary": "partial_backward"}, {"committed_steps": True}, {"cursor": original["cursor"] + 1},
            {"epoch": original["epoch"] + 1}, {"sampler_position": -1},
            {"normalization": {"target_scale": 1.}},
            {"dataset_identity": dataset_identity({**self.config, "seed": self.config["seed"] + 1})},
            {"permutation": [0] * 256}, {"permutation": [True] + list(range(1, 256))},
            {"initial_model_sha256": "invalid"},
        )
        for replacement in replacements:
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                work.restore_state({**copy.deepcopy(original), **replacement})
            self.assertEqual(work.state(), original)


@unittest.skipUnless(CUDA_SELECTION is not None, "manual CUDA workload conformance not requested; CUDA unmeasured")
class IKWorkloadManualCUDAConformanceTests(unittest.TestCase):
    def test_explicit_selected_cuda_device_executes_two_real_optimizer_steps(self):
        self.assertRegex(CUDA_SELECTION, r"^cuda:(0|[1-9][0-9]*)$")
        config = fixture_config(batch_size=7, steps=2)
        # No skip-on-unavailable: an explicit manual request is a failing check
        # when the dependency, selected device or compiled architecture is absent.
        context = configure_device(CUDA_SELECTION, config["seed"], memory_fraction=config["memory_fraction"])
        self.assertEqual(context.device.type, "cuda")
        work = SmokeWorkload(context, config)
        before = model_digest(work.model)
        context.reset_peak_memory()
        for _ in range(2):
            _, batch = work.batch()
            receipt, timing = context.timed(lambda: work.step(batch))
            self.assertTrue(receipt["finite_gradients"])
            self.assertTrue(receipt["weights_changed"])
            self.assertEqual(receipt["placement"]["device"], CUDA_SELECTION)
            self.assertIsNotNone(timing["cuda_event_seconds"])
            self.assertGreaterEqual(timing["cuda_event_seconds"], 0)
            self.assertEqual(receipt["placement"]["tensor_counts"]["gradients"],
                             len(list(work.model.parameters())))
            for tensor in tensors(work.optimizer.state, context.torch):
                self.assertEqual(tensor.device, context.device)
        self.assertNotEqual(before, model_digest(work.model))
        memory = context.memory_stats()
        self.assertGreater(memory["peak_allocated_bytes"], 0)
        self.assertGreaterEqual(memory["peak_reserved_bytes"], memory["peak_allocated_bytes"])
        self.assertEqual((work.committed_steps, work.cursor), (2, 14))


if __name__ == "__main__":
    unittest.main()
