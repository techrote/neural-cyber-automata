"""Evidence predicates must reject divergent actions and physical traces."""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_LEARNER = importlib.util.find_spec("nca.controllers.learned") is not None
if HAS_NUMPY:
    from nca.contracts import ContractError, WireAction
    from nca.bench.common import pose_difference, wire_difference
    from nca.bench.deployment import _arithmetic, _compare_episodes, _expired_twelve_tick_refusals
    from nca.bench.parity_audit import _common_inputs, divergence_points

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_NUMPY, "optional NumPy export dependency unavailable")
class ExportBenchTests(unittest.TestCase):
    def expiry_fixture(self, age, reason):
        from nca.adapters.protocol import make_physical_command
        source = {"version": 2, "run_id": "expiry-test", "epoch": 1, "actor_id": 0,
                  "control_generation": 0, "tick": 7, "last_sequence": 0,
                  "observation_id": "expiry-test:1:0:0:7"}
        packet = make_physical_command(source, 1, WireAction(trim=.01018200907856226).to_dict(), 12)
        requested = deepcopy(packet)
        # Actual owner receipts carry JSON-rounded floats, including integers.
        for key in ("version", "actor_id", "epoch", "sequence", "observed_tick", "lease_ticks"):
            requested[key] = float(requested[key])
        requested["action"]["trim"] = .0101820090785623
        receipt = {"tick_before": 7 + age, "tick_after": 8 + age,
                   "requested": [requested], "admission": [{"accepted": False, "reason": reason}]}
        return source, packet, [receipt]

    def test_realtime_expiry_matches_each_pinned_host_reason_at_its_actual_age(self):
        for age, reason in ((12, "expired_observation"), (18, "expired_observation"),
                            (19, "lease"), (24, "lease")):
            with self.subTest(age=age, reason=reason):
                matches = _expired_twelve_tick_refusals(*self.expiry_fixture(age, reason))
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0]["source_age_ticks"], age)
                self.assertEqual(matches[0]["lease_deadline_tick"], 19)

    def test_realtime_refusal_reason_alone_cannot_prove_expiry(self):
        for age, reason in ((11, "lease"), (11, "expired_observation"), (12, "lease"),
                            (18, "lease"), (19, "expired_observation"), (24, "sequence")):
            with self.subTest(age=age, reason=reason):
                self.assertEqual(_expired_twelve_tick_refusals(*self.expiry_fixture(age, reason)), [])
        mutations = (lambda row: row["requested"][0].update(sequence=2),
                     lambda row: row["requested"][0].update(observed_tick=8),
                     lambda row: row["requested"][0].update(lease_ticks=0),
                     lambda row: row["requested"][0].update(actor_id=False),
                     lambda row: row["requested"][0]["action"].update(trim=.2),
                     lambda row: row["admission"][0].update(accepted=True))
        for mutate in mutations:
            source, packet, receipts = self.expiry_fixture(24, "lease")
            mutate(receipts[0])
            self.assertEqual(_expired_twelve_tick_refusals(source, packet, receipts), [])

    def test_realtime_expiry_requires_original_valid_packet_and_complete_receipts(self):
        for mutate in (lambda source, packet, rows: packet.update(lease_ticks=0),
                       lambda source, packet, rows: packet.update(lease_ticks=18),
                       lambda source, packet, rows: packet.update(sequence=0),
                       lambda source, packet, rows: packet.update(observed_tick=6),
                       lambda source, packet, rows: packet.update(observation_id="unissued"),
                       lambda source, packet, rows: packet.update(action={}),
                       lambda source, packet, rows: source.update(tick=True),
                       lambda source, packet, rows: rows[0].update(tick_after=33),
                       lambda source, packet, rows: rows[0]["admission"].clear()):
            source, packet, receipts = self.expiry_fixture(24, "lease")
            mutate(source, packet, receipts)
            with self.assertRaises(ContractError):
                _expired_twelve_tick_refusals(source, packet, receipts)

    def fixture(self):
        config = json.loads((ROOT / "configs/export/cpu-bc-v1.json").read_text())
        frame = {"tick": 0, "parts": [{"id": "core", "index": 0, "position": [0., 0.],
                                        "rotation": 0., "detached": False}]}
        frames = [dict(deepcopy(frame), tick=tick) for tick in range(7)]
        episode = {"frames": frames, "steps": [{"issued_tick": 0, "actual_ticks": 6,
                    "terminated": False, "truncated": False, "action": WireAction(effort=0).to_dict(),
                    "hidden_after": [0.]*32, "features": [0.]*26, "value": 0.,
                    "trace": [{"tick_before": tick, "tick_after": tick+1, "world_tick": tick+1,
                               "cellular_sample_tick": 0, "admission": [{"accepted": True, "reason": ""}]}
                              for tick in range(6)]}],
                   "cancellation_exact": True}
        return config, episode

    def test_trajectory_gate_rejects_action_pose_clock_and_cancellation_differences(self):
        config, episode = self.fixture()
        self.assertTrue(_compare_episodes(episode, episode, config)["passed"])
        mutations = (lambda value: value["frames"][0]["parts"][0].update(position=[.051, 0.]),
                     lambda value: value["steps"][0]["action"].update(primary=True),
                     lambda value: value["steps"][0]["trace"][0].update(world_tick=2),
                     lambda value: value.update(cancellation_exact=False))
        for mutate in mutations:
            other = deepcopy(episode)
            mutate(other)
            self.assertFalse(_compare_episodes(episode, other, config)["passed"])

    def test_trajectory_gate_requires_every_frame_and_consumed_receipt(self):
        config, episode = self.fixture()
        mutations = (lambda value: value["steps"][0]["trace"].pop(),
                     lambda value: value["steps"][0]["trace"].clear(),
                     lambda value: value["frames"].pop(),
                     lambda value: value["steps"][0].update(actual_ticks=0),
                     lambda value: value["steps"][0]["trace"][1].update(tick_before=0),
                     lambda value: value["frames"][1].update(tick=2),
                     lambda value: value.update(frames=[], steps=[]))
        for mutate in mutations:
            other = deepcopy(episode)
            mutate(other)
            self.assertFalse(_compare_episodes(episode, other, config)["passed"])
            self.assertFalse(_compare_episodes(other, other, config)["passed"])

    def test_missing_or_nonfinite_recorded_network_state_cannot_appear_equal(self):
        config, episode = self.fixture()
        changes = (("hidden_after", [0.]), ("hidden_after", [float("nan")]*32),
                   ("features", [0.]), ("features", [float("nan")]*26),
                   ("value", float("nan")), ("value", True))
        for field, value in changes:
            other = deepcopy(episode)
            other["steps"][0][field] = value
            result = _compare_episodes(episode, other, config)
            self.assertFalse(result["passed"])
            self.assertEqual(result["reason"], "invalid_numeric_evidence")
            self.assertFalse(_compare_episodes(other, other, config)["passed"])

    def test_nonfinite_or_incomplete_values_cannot_become_zero_difference(self):
        _, episode = self.fixture()
        for mutate in (lambda value: value["parts"][0].update(position=[float("nan"), 0]),
                       lambda value: value["parts"][0].update(position=[0]),
                       lambda value: value["parts"][0].update(rotation=float("inf")),
                       lambda value: value.update(parts=[])):
            other = deepcopy(episode["frames"][0])
            mutate(other)
            with self.assertRaises(ContractError):
                pose_difference(episode["frames"][0], other)
        action = WireAction(effort=0).to_dict()
        for key, value in (("trim", float("nan")), ("move", [0])):
            with self.assertRaises(ContractError):
                wire_difference(action, dict(action, **{key: value}))
        with self.assertRaisesRegex(ContractError, "trajectory_action_shape"):
            wire_difference({}, WireAction().to_dict())

    def test_resume_requires_its_declared_residual_feature_width(self):
        config, episode = self.fixture()
        episode["steps"][0]["features"] = [0.]*32
        self.assertFalse(_compare_episodes(episode, episode, config)["passed"])
        self.assertTrue(_compare_episodes(episode, episode, config, feature_width=32)["passed"])
        episode["steps"][0]["features"].pop()
        self.assertFalse(_compare_episodes(episode, episode, config, feature_width=32)["passed"])
        with self.assertRaisesRegex(ContractError, "trajectory_feature_spec"):
            _compare_episodes(episode, episode, config, feature_width=31)

    def test_divergence_diagnostic_preserves_original_failure_and_onset(self):
        config, episode = self.fixture()
        other = deepcopy(episode)
        other["steps"][0]["action"]["trim"] = 1e-6
        other["frames"][3]["parts"][0]["position"][0] = .06
        result = divergence_points(episode, other, config)
        self.assertFalse(result["comparison"]["passed"])
        self.assertEqual(result["first_action_difference"]["issued_tick"], 0)
        self.assertEqual(result["first_pose_difference"]["tick"], 3)
        self.assertEqual(result["first_pose_tolerance_failure"]["tick"], 3)
        self.assertTrue(result["initial_pose_exact"])

    @unittest.skipUnless(HAS_TORCH and HAS_LEARNER, "accepted Torch learner source not installed")
    def test_common_input_audit_distinguishes_arithmetic_errors_from_recorded_outcomes(self):
        import torch
        from unittest.mock import patch
        from nca.controllers.baseline import SkillGoal
        from nca.controllers.learned import RecurrentPolicy
        from nca.export import ExportController
        from test_export_runtime import bundle, view
        torch.set_num_threads(1)
        item = bundle()
        policy = RecurrentPolicy("bc")
        policy.load_state_dict({name: torch.tensor(value.copy()) for name, value in item.network.weights.items()})
        observation = view()
        goal = SkillGoal("audit-only", "directed_hop", (200, 346.5), timeout_seconds=4)
        decision = ExportController(item).step(observation, goal)
        episode = {"controller": "numpy", "initial": observation, "observations": [observation],
                   "goal": asdict(goal), "steps": [{"issued_tick": 0,
                       "issued_observation_id": observation["observation_id"],
                       "features": decision.features, "hidden_before": decision.hidden_before,
                       "hidden_after": decision.hidden_after, "action": decision.action.to_dict(),
                       "value": decision.value}]}
        config, _ = self.fixture()
        result = _common_inputs(episode, item, policy, config)
        self.assertTrue(result["arithmetic_passed"])
        self.assertTrue(result["recorded_reconstruction_passed"])
        forward = item.network.forward
        def altered_output(*args, **kwargs):
            output, value, hidden, logits = forward(*args, **kwargs)
            output[0] += .001
            return output, value, hidden, logits
        with patch.object(item.network, "forward", altered_output):
            result = _common_inputs(episode, item, policy, config)
        self.assertFalse(result["arithmetic_passed"])
        self.assertFalse(result["recorded_reconstruction_passed"])

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
