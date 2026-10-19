"""Protocol-only tests. These intentionally make no physical-host claims."""

import copy
import math
import unittest

from nca.adapters.protocol import (
    MAX_PACKET_BYTES, ProtocolError, RewardDifference, decode_packet, encode_packet,
    make_physical_command, make_sampled_command,
)
from nca.envs.recorded import RecordedEnv


class HostProtocolTests(unittest.TestCase):
    def setUp(self):
        self.view = {"run_id": "fixture-run", "actor_id": 0, "epoch": 1, "tick": 20,
                     "observation_id": "issued-fixture-view", "control_mode": "wire"}

    def test_physical_command_keeps_six_repeat_distinct_from_lease(self):
        packet = make_physical_command(self.view, 4, {"move": [1, 0]})
        self.assertEqual(packet["lease_ticks"], 12)
        self.assertEqual(packet["observed_tick"], 20)
        self.assertEqual(packet["observation_id"], self.view["observation_id"])
        self.assertEqual(set(packet), {"version", "kind", "run_id", "actor_id", "epoch", "sequence",
                                      "observed_tick", "observation_id", "lease_ticks", "action"})

    def test_commands_own_nested_input(self):
        action = {"move": [1, 0]}
        packet = make_physical_command(self.view, 0, action)
        action["move"][0] = -1
        self.assertEqual(packet["action"]["move"], [1, 0])

    def test_legacy_wire_names_are_not_reinterpreted(self):
        action = {"movement": [1, 0]}
        packet = make_sampled_command(self.view, 3, action)
        self.assertEqual(packet["version"], 1)
        self.assertEqual(packet["kind"], "rig_action")
        self.assertEqual(packet["action"], action)
        self.assertFalse(packet["cancel"])

    def test_legacy_metal_lane_is_explicit(self):
        self.view["control_mode"] = "metal"
        packet = make_sampled_command(self.view, 3, {"actuators": [1, 0, 0, 0], "tag_aim": [1, 0]})
        self.assertEqual(packet["kind"], "actuator_action")
        self.assertEqual(len(packet["action"]["actuators"]), 4)

    def test_invalid_sequence_and_lease(self):
        for sequence, lease in ((-1, 12), (True, 12), (0, 0), (0, 19), (0, False)):
            with self.subTest(sequence=sequence, lease=lease), self.assertRaises(ValueError):
                make_physical_command(self.view, sequence, {"move": [0, 0]}, lease)

    def test_json_rejects_nonfinite_duplicate_oversize_and_wrong_roots(self):
        for packet in (b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}', b'{"a":1,"a":2}',
                       b'[]', b'null', b'\xff', b' ', b'x' * (MAX_PACKET_BYTES + 1)):
            with self.subTest(packet=packet[:40]), self.assertRaises(ProtocolError):
                decode_packet(packet)
        with self.assertRaises(ProtocolError):
            encode_packet({"bad": math.nan})

    def test_reward_differences_not_cumulative_awards(self):
        rewards = RewardDifference()
        rewards.reset("run", 0, 0)
        self.assertEqual(rewards.consume("run", 6, 3), 3)
        self.assertEqual(rewards.consume("run", 12, 3), 0)
        self.assertEqual(rewards.consume("run", 13, 1), -2)
        with self.assertRaises(ProtocolError):
            rewards.consume("run", 13, 1)
        with self.assertRaises(ProtocolError):
            rewards.consume("replacement", 14, 100)
        rewards.reset("replacement", 0, 0)
        self.assertEqual(rewards.consume("replacement", 1, 0), 0)

    def test_recorded_playback_is_owned_exact_and_explicitly_nonlive(self):
        source = {"format": "nca-recorded-host/1", "source_environment": "protocol-fixture",
                  "reset": {"observation": {"position": [0, 0]}, "info": {}},
                  "transitions": [{"action": {"move": [1, 0]}, "observation": {"position": [1, 0]},
                                   "reward": 1, "terminated": False, "truncated": True, "info": {}}]}
        env = RecordedEnv(source)
        source["reset"]["observation"]["position"][0] = 1000
        obs, info = env.reset()
        self.assertEqual(obs["position"], [0, 0])
        self.assertEqual(info["evidence_class"], "recorded-playback")
        with self.assertRaises(ProtocolError):
            env.step({"move": [-1, 0]})
        self.assertEqual(env.step({"move": [1, 0]})[1:4], (1.0, False, True))
        with self.assertRaises(ValueError):
            env.reset(seed=1)


if __name__ == "__main__":
    unittest.main()
