"""CPU-only model conformance; no candidate training or CUDA qualification."""

import importlib.util
import subprocess
import sys
import unittest

from nca.contracts import ContractError
from nca.ik.data import Normalization
from nca.ik.models import (INACTIVE_CHANNEL, INVALID_MASK, MISSING_MEASUREMENT,
                           NONFINITE_INPUT, NONFINITE_OUTPUT, OUT_OF_BOUNDS, build_model)


OPTIONAL = (importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("numpy") is not None)


def normalization():
    # Deliberately synthetic fit for local conformance; not corpus provenance.
    return Normalization(1, (0.1, -0.05, 4., 8., 0., 0.) + (42.,) * 8,
                         (.5, .2, 2., 2., .5, .2) + (3.,) * 8, "a" * 64)


class IKModelImportTests(unittest.TestCase):
    def test_model_discovery_does_not_import_optional_runtime(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; import nca.ik.models; "
             "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(OPTIONAL, "optional pinned IK Torch/NumPy runtime is not installed")
class IKModelCPUConformanceTests(unittest.TestCase):
    def setUp(self):
        from nca.ik.device import configure_device
        self.context = configure_device("cpu", 54011)
        self.torch = self.context.torch
        self.model = build_model(self.context, normalization())

    def inputs(self, count=1):
        return self.torch.tensor([[0., 0., 4.6, 9., 0., 0.] + [1.] * 8] * count,
                                 dtype=self.torch.float32)

    def test_fk_loss_backpropagates_finite_model_gradients(self):
        from nca.ik.kinematics import TorchFK, chain_from_profile, cpu_fk
        from nca.ik.profiles import runner_right_sole_profile
        torch, profile = self.torch, runner_right_sole_profile()
        chain = chain_from_profile(profile)
        rows = self.inputs(2)
        rows[:, 2:4] = torch.tensor([cpu_fk(chain, q).point for q in
                                    ((.2, -.1), (-.1, .1))], dtype=torch.float32)
        prediction = self.model(rows)
        self.assertTrue(bool(prediction.valid.all()))
        geometry = TorchFK(chain, self.context)(prediction.reference)
        self.assertTrue(bool(geometry.valid.all()))
        loss = ((geometry.point - rows[:, 2:4]) / profile.nominal_reach_cells).square().mean()
        loss.backward()
        gradients = {name: value.grad for name, value in self.model.named_parameters()}
        self.assertTrue(all(value is not None for value in gradients.values()))
        self.assertTrue(any(bool(value.abs().sum() > 0) for value in gradients.values()))
        receipt = self.context.verify_tensors({
            "model": self.model.state_dict(), "inputs": rows, "output": prediction.reference,
            "geometry": geometry.point, "loss": loss, "gradients": gradients})
        self.assertTrue(receipt["all_on_selected_device"])

    def test_numeric_normalization_keeps_mask_bits_explicit(self):
        captured = []
        hook = self.model.network[0].register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach().clone()))
        rows = self.inputs()
        try:
            self.model(rows)
        finally:
            hook.remove()
        expected = (rows[:, :6] - self.model.normalization_mean) / self.model.normalization_scale
        self.torch.testing.assert_close(captured[0][:, :6], expected, rtol=0, atol=0)
        self.torch.testing.assert_close(captured[0][:, 6:], rows[:, 6:], rtol=0, atol=0)

    def test_saturated_references_stay_inside_binary64_command_bounds(self):
        torch = self.torch
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.network[-1].bias.copy_(torch.tensor([1e6, -1e6]))
        prediction = self.model(self.inputs())
        self.assertTrue(bool(prediction.valid.all()))
        hip, ankle = prediction.reference[0].tolist()
        self.assertLessEqual(hip, .72)
        self.assertGreaterEqual(hip, -.72)
        self.assertLessEqual(ankle, .27)
        self.assertGreaterEqual(ankle, -.27)
        self.assertGreater(hip, .719)
        self.assertLess(ankle, -.269)

    def test_measured_zero_and_missing_zero_are_different(self):
        rows = self.inputs(2)
        rows[1, 8] = 0
        prediction = self.model(rows)
        self.assertEqual(prediction.valid.tolist(), [True, False])
        self.assertEqual(int(prediction.invalid_code[0]), 0)
        self.assertTrue(int(prediction.invalid_code[1]) & MISSING_MEASUREMENT)
        self.assertEqual(prediction.reference[1].tolist(), [0., 0.])
        self.assertTrue(bool(self.torch.isfinite(prediction.reference).all()))

    def test_every_required_mask_has_explicit_refusal(self):
        rows = self.inputs(8)
        for index in range(8):
            rows[index, 6 + index] = 0
        prediction = self.model(rows)
        self.assertFalse(bool(prediction.valid.any()))
        for index, code in enumerate(prediction.invalid_code.tolist()):
            self.assertTrue(code & (MISSING_MEASUREMENT if index < 4 else INACTIVE_CHANNEL))
        self.assertEqual(prediction.reference.tolist(), [[0., 0.]] * 8)

    def test_nonfinite_and_nonbinary_inputs_are_sanitized_before_network(self):
        torch, rows = self.torch, self.inputs(4)
        rows[0, 2] = float("nan")
        rows[1, 0] = float("inf")
        rows[2, 6] = .5
        rows[3, 12] = float("nan")
        prediction = self.model(rows)
        self.assertFalse(bool(prediction.valid.any()))
        self.assertTrue(int(prediction.invalid_code[0]) & NONFINITE_INPUT)
        self.assertTrue(int(prediction.invalid_code[1]) & NONFINITE_INPUT)
        self.assertTrue(int(prediction.invalid_code[2]) & INVALID_MASK)
        self.assertTrue(int(prediction.invalid_code[3]) & INVALID_MASK)
        self.assertEqual(prediction.reference.tolist(), [[0., 0.]] * 4)
        prediction.reference.square().sum().backward()
        for parameter in self.model.parameters():
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_hard_current_and_inner_posture_bounds_are_distinct(self):
        rows = self.inputs(3)
        rows[0, 0] = .74
        rows[1, 0] = .76
        rows[2, 4] = .73
        prediction = self.model(rows)
        self.assertEqual(prediction.valid.tolist(), [True, False, False])
        self.assertTrue(int(prediction.invalid_code[1]) & OUT_OF_BOUNDS)
        self.assertTrue(int(prediction.invalid_code[2]) & OUT_OF_BOUNDS)

    def test_fp32_input_endpoints_use_strict_descriptor_bounds(self):
        from nca.ik.data import RuntimeInputs, runtime_features
        from nca.ik.profiles import runner_right_sole_profile
        inputs = RuntimeInputs(runner_right_sole_profile().sha256, (0., .30),
                               (4.6, 9.), (0., 0.))
        rows = self.torch.tensor([runtime_features(inputs)], dtype=self.torch.float32)
        result = self.model(rows)
        self.assertFalse(bool(result.valid[0]))
        self.assertTrue(int(result.invalid_code[0]) & OUT_OF_BOUNDS)
        rows[0, 1] = self.model.current_upper[1]
        self.assertTrue(bool(self.model(rows).valid[0]))

    def test_wrong_shape_dtype_and_device_are_refused(self):
        torch = self.torch
        bad_inputs = (self.inputs().double(), torch.zeros(14), torch.zeros((1, 13)),
                      torch.zeros((0, 14)), torch.empty((1, 14), device="meta"), {})
        for value in bad_inputs:
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ContractError):
                self.model(value)

    def test_nonfinite_model_output_is_not_an_actionable_reference(self):
        with self.torch.no_grad():
            self.model.network[-1].bias.fill_(float("nan"))
        prediction = self.model(self.inputs())
        self.assertFalse(bool(prediction.valid.any()))
        self.assertTrue(int(prediction.invalid_code[0]) & NONFINITE_OUTPUT)
        self.assertEqual(prediction.reference.tolist(), [[0., 0.]])

    def test_state_dict_roundtrip_preserves_fit_bounds_and_predictions(self):
        torch = self.torch
        expected = self.model(self.inputs(2))
        state = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        self.assertTrue({"normalization_mean", "normalization_scale", "command_lower",
                         "command_upper", "current_lower", "current_upper"} <= set(state))
        rebuilt = build_model(self.context, normalization())
        rebuilt.load_state_dict(state, strict=True)
        actual = rebuilt(self.inputs(2))
        torch.testing.assert_close(actual.reference, expected.reference, rtol=0, atol=0)
        self.assertTrue(torch.equal(actual.valid, expected.valid))
        self.assertTrue(torch.equal(actual.invalid_code, expected.invalid_code))
        self.assertEqual(rebuilt.normalization_sha256, normalization().sha256)

    def test_unrepresentable_normalization_is_refused(self):
        bad = Normalization(1, (0.,) * 14, (1e-100,) + (1.,) * 13, "a" * 64)
        with self.assertRaises(ContractError):
            build_model(self.context, bad)


if __name__ == "__main__":
    unittest.main()
