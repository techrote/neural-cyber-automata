"""Evidence predicates must reject divergent actions and physical traces."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_LEARNER = importlib.util.find_spec("nca.controllers.learned") is not None
if HAS_NUMPY:
    from nca.contracts import WireAction
    from nca.bench.deployment import _arithmetic, _compare_episodes

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_NUMPY, "optional NumPy export dependency unavailable")
class ExportBenchTests(unittest.TestCase):
    def test_trajectory_gate_rejects_action_pose_clock_and_cancellation_differences(self):
        config = json.loads((ROOT / "configs/export/cpu-bc-v1.json").read_text())
        frame = {"tick": 6, "parts": [{"id": "core", "index": 0, "position": [0., 0.],
                                        "rotation": 0., "detached": False}]}
        episode = {"frames": [frame], "steps": [{"issued_tick": 0, "actual_ticks": 6,
                    "terminated": False, "truncated": False, "action": WireAction(effort=0).to_dict(),
                    "hidden_after": [0.]*32, "features": [0.]*26, "value": 0.,
                    "trace": [{"tick_before": 0, "tick_after": 1, "world_tick": 1,
                               "cellular_sample_tick": 0, "admission": [{"accepted": True, "reason": ""}]}]}],
                   "cancellation_exact": True}
        self.assertTrue(_compare_episodes(episode, episode, config)["passed"])
        mutations = (lambda value: value["frames"][0]["parts"][0].update(position=[.051, 0.]),
                     lambda value: value["steps"][0]["action"].update(primary=True),
                     lambda value: value["steps"][0]["trace"][0].update(world_tick=2),
                     lambda value: value.update(cancellation_exact=False))
        for mutate in mutations:
            other = deepcopy(episode)
            mutate(other)
            self.assertFalse(_compare_episodes(episode, other, config)["passed"])

    @unittest.skipUnless(HAS_TORCH and HAS_LEARNER, "accepted Torch learner source not installed")
    def test_arithmetic_screen_uses_valid_sequential_feedback_and_exact_reference(self):
        import torch
        from nca.controllers.learned import RecurrentPolicy
        from test_export_runtime import bundle
        torch.set_num_threads(1)
        item = bundle()
        policy = RecurrentPolicy("bc")
        policy.load_state_dict({name: torch.tensor(value.copy()) for name, value in item.network.weights.items()})
        config = json.loads((ROOT / "configs/export/cpu-bc-v1.json").read_text())
        result = _arithmetic(item, policy, config)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["samples"], 128)
        self.assertEqual(result["maximum_absolute_differences"]["features"], 0)


if __name__ == "__main__":
    unittest.main()
