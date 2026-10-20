"""Explicit Torch backend contract and original-operator fidelity checks."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_RUNTIME:
    import numpy as np
    import torch
    from nca.contracts import ContractError, sha256_json
    from nca.controllers.baseline import SkillGoal
    from nca.export import ExportController, RuntimeBundle
    from nca.export.build import immutable_json
    from nca.export.torch_runtime import TorchGRU, TorchRuntimeBundle, make_torch_bundle, torch_export_controller
    from test_export_runtime import bundle, view


@unittest.skipUnless(HAS_RUNTIME, "optional Torch and NumPy export dependencies unavailable")
class TorchExportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.numeric = bundle()
        self.payload = make_torch_bundle(self.numeric.value, self.numeric.context, numeric_file_sha256="a" * 64)

    def item(self):
        return TorchRuntimeBundle(self.payload, self.numeric.context)

    def test_original_operator_graph_matches_reference_exactly_across_recurrence_and_resets(self):
        from nca.controllers.learned import RecurrentPolicy
        item = self.item()
        policy = RecurrentPolicy("bc")
        policy.load_state_dict({key: torch.tensor(value) for key, value in self.numeric.value["weights"].items()})
        policy.eval()
        rng = np.random.default_rng(73109)
        h = np.zeros(32, dtype=np.float32)
        th = policy.zero_hidden()
        with torch.no_grad():
            for index in range(128):
                x = rng.uniform(-2, 2, 26).astype(np.float32)
                reset = index in (0, 32, 64, 96)
                if reset:
                    th.zero_()
                result, value, h, logits = item.network.forward(x, h, reset=reset)
                tl, tv, th = policy(torch.tensor(x).unsqueeze(0), th)
                np.testing.assert_array_equal(h, th[0].numpy())
                np.testing.assert_array_equal(logits, tl[0].numpy())
                np.testing.assert_array_equal(result, policy.clone_outputs(tl)[0].numpy())
                self.assertEqual(value, float(tv[0]))

    def test_wrapper_preserves_original_numeric_identity_and_cannot_claim_qualification(self):
        item = self.item()
        self.assertEqual(item.numeric.value, self.numeric.value)
        self.assertEqual(item.network.policy_sha256, self.numeric.network.policy_sha256)
        self.assertFalse(item.value["qualification"]["runtime_qualified"])
        for modify in (lambda p: p.update(format="nca-cpu-policy/v1"),
                       lambda p: p["backend"].update(torch="2.8.1"),
                       lambda p: p["backend"].update(threads=2),
                       lambda p: p.update(backend_sources={}),
                       lambda p: p.update(source_numeric_content_sha256="b" * 64),
                       lambda p: p["qualification"].update(runtime_qualified=True)):
            changed = deepcopy(self.payload)
            modify(changed)
            changed.pop("sha256")
            changed["sha256"] = sha256_json(changed)
            with self.assertRaises(ContractError):
                TorchRuntimeBundle(changed, self.numeric.context)
        with self.assertRaises(ContractError):
            RuntimeBundle(self.payload, self.numeric.context)
        exposed = deepcopy(self.payload)
        self.payload["backend"]["threads"] = 4
        fresh = make_torch_bundle(self.numeric.value, self.numeric.context, numeric_file_sha256="a" * 64)
        self.assertEqual(fresh, exposed)

    def test_runtime_context_and_thread_change_are_fenced_before_new_inference(self):
        with self.assertRaises(ContractError):
            TorchRuntimeBundle(self.payload, replace(self.numeric.context, decision_ticks=12))
        controller = torch_export_controller(self.item())
        goal = SkillGoal("hop", "directed_hop", (200, 346.5), timeout_seconds=4)
        self.assertEqual(controller.step(view(), goal).status, "accepted")
        torch.set_num_threads(2)
        try:
            result = controller.step(view(6), goal)
            self.assertTrue(result.cancel_owner)
            self.assertEqual(result.reason, "torch_runtime_context")
            np.testing.assert_array_equal(controller.hidden, np.zeros(32, dtype=np.float32))
        finally:
            torch.set_num_threads(1)

    def test_export_keeps_source_age_takeover_and_cancellation_fences(self):
        goal = SkillGoal("hop", "directed_hop", (200, 346.5), timeout_seconds=4)
        controller = torch_export_controller(self.item())
        decision = controller.step(view(), goal)
        self.assertEqual(decision.source.tick, 0)
        self.assertEqual(controller.step(view(6, control=1), goal).reason, "owner_or_mount_changed")
        self.assertTrue(controller.step(view(12, control=1), goal).cancel_owner)
        fresh = replace(goal, id="fresh")
        decision = controller.step(view(12, control=1), fresh)
        self.assertEqual(decision.hidden_before, (0.,) * 32)
        self.assertEqual(controller.step(view(18, control=1), fresh, current_tick=30).reason, "source_expired")
        with self.assertRaises(ContractError):
            torch_export_controller(self.numeric)

    def test_cpu_autocast_cannot_silently_change_declared_fp32_outputs(self):
        for dtype in (torch.float16, torch.bfloat16):
            controller = torch_export_controller(self.item())
            goal = SkillGoal("hop", "directed_hop", (200, 346.5), timeout_seconds=4)
            self.assertEqual(controller.step(view(), goal).status, "accepted")
            with torch.autocast("cpu", dtype=dtype):
                result = controller.step(view(6), goal)
                self.assertEqual(result.reason, "torch_runtime_context")
                self.assertTrue(result.cancel_owner)
                np.testing.assert_array_equal(controller.hidden, np.zeros(32, dtype=np.float32))

    def test_network_owns_weights_and_preserves_caller_rng_and_input_state(self):
        state = torch.get_rng_state().clone()
        weights = deepcopy(self.numeric.value["weights"])
        network = TorchGRU(weights)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        x, h = np.ones(26, dtype=np.float32), np.ones(32, dtype=np.float32)
        expected = network.forward(x, h)
        weights["actor.bias"][0] = 123
        actual = network.forward(x, h)
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(h, np.ones(32, dtype=np.float32))
        for inputs, hidden, reset in (([0] * 25, h, False), (x, [float("nan")] * 32, False),
                                     (x, h, 1)):
            with self.assertRaises(ContractError):
                network.forward(inputs, hidden, reset=reset)

    def test_immutable_artifact_load_and_standalone_import_exclude_learner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torch-policy.json"
            digest = immutable_json(path, self.payload)
            self.assertEqual(TorchRuntimeBundle.load(path, self.numeric.context, expected_sha256=digest).sha256,
                             self.payload["sha256"])
            with self.assertRaises(ContractError):
                TorchRuntimeBundle.load(path, self.numeric.context, expected_sha256="0" * 64)
            program = """
import json,sys,torch
torch.set_num_threads(1)
from nca.export.spec import RuntimeContext
from nca.export.torch_runtime import TorchRuntimeBundle
p=json.load(open(sys.argv[1]))
b=TorchRuntimeBundle.load(sys.argv[1], RuntimeContext.from_dict(p['numeric_bundle']['context']),expected_sha256=sys.argv[2])
b.network.forward([0.]*26,[0.]*32)
assert not any(n=='nca.controllers.learned' or n.startswith('nca.train') for n in sys.modules)
print('standalone inference without learner')
"""
            result = subprocess.run([sys.executable, "-c", program, str(path), digest], text=True,
                                    capture_output=True, check=True, timeout=30)
            self.assertIn("standalone inference without learner", result.stdout)


if __name__ == "__main__":
    unittest.main()
