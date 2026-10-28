"""Frozen CPU inference must preserve all v3 row checks and snapshot weights."""
import importlib.util
import unittest

from nca.contracts import ContractError


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional Torch runtime')
class FrozenCPUInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nca.ik.device import configure_device
        from nca.ik.polar import PolarNormalization
        cls.context = configure_device('cpu', 41011)
        cls.torch = cls.context.torch
        cls.torch.set_num_threads(1)
        cls.normalization = PolarNormalization(1, (0.,)*14, (1.,)*14, 'a'*64, 'b'*64)

    def model(self, **kwargs):
        from nca.ik.models_v3 import build_model
        return build_model(self.context, self.normalization, **kwargs).eval()

    def test_heads_widths_random_and_adversarial_rows_preserve_reference_and_codes(self):
        from nca.ik.inference_v1 import freeze_batch_one
        t = self.torch
        rows = t.randn((80, 14), dtype=t.float32)
        rows[:, :2] *= .1; rows[:, 4:6] *= .1; rows[:, 6:] = 1
        rows[1, 6] = 0; rows[2, 10] = .5; rows[3, 0] = 100
        rows[4, 2] = float('nan'); rows[5, 3] = float('inf')
        rows[6, 12] = 0; rows[7, 8] = 0; rows[8, 4] = .73
        for head in ('tanh', 'interior_linear'):
            for width in (64, 128):
                with self.subTest(head=head, width=width), t.inference_mode():
                    model = self.model(head=head, hidden_width=width)
                    frozen = freeze_batch_one(self.context, model)
                    for row in rows.split(1):
                        expected, actual = model(row), frozen(row)
                        self.assertTrue(t.equal(expected.valid, actual.valid))
                        self.assertTrue(t.equal(expected.invalid_code, actual.invalid_code))
                        self.assertLessEqual((expected.reference-actual.reference).abs().max().item(), 2e-5)

    def test_snapshot_is_independent_and_invalid_tensor_contract_refuses(self):
        from nca.ik.inference_v1 import freeze_batch_one
        t = self.torch; model = self.model(head='tanh')
        frozen = freeze_batch_one(self.context, model)
        row = t.zeros((1, 14)); row[:, 6:] = 1
        before = frozen(row).reference.clone()
        with t.no_grad():
            model.network[-1].bias.add_(1)
        self.assertTrue(t.equal(before, frozen(row).reference))
        for bad in (row.double(), row.expand(2, 14), row[0], None):
            with self.assertRaises(ContractError): frozen(bad)

    def test_nonfinite_network_output_retains_refusal(self):
        from nca.ik.inference_v1 import freeze_batch_one
        t = self.torch; model = self.model(head='tanh')
        with t.no_grad(): model.network[-1].bias[0] = float('nan')
        frozen = freeze_batch_one(self.context, model)
        row = t.zeros((1, 14)); row[:, 6:] = 1
        result = frozen(row)
        self.assertFalse(result.valid.item())
        self.assertEqual(result.invalid_code.item(), 32)
        self.assertTrue(t.equal(result.reference, t.zeros((1, 2))))


if __name__ == '__main__': unittest.main()

# Copyright 2026 techrote. All rights reserved.
