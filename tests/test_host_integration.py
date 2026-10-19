"""Opt-in actual Cybersand tests. A skip is never game qualification.

NCA_HOST_CHECKOUT must name the pinned accepted checkout with materialized
Linux/Windows native libraries; NCA_GODOT names the pinned executable.
"""

import copy
import os
import time
import unittest

from nca.adapters.godot import CybersandEnv, GodotHost, physical_config, sampled_config
from nca.adapters.protocol import EpisodeFinished, HostError, ProtocolError, make_physical_command


HOST = os.environ.get("NCA_HOST_CHECKOUT")
GODOT = os.environ.get("NCA_GODOT")
ENABLED = bool(HOST and GODOT)


@unittest.skipUnless(ENABLED, "actual Cybersand runtime not configured; no ACT004 evidence")
class PhysicalHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = GodotHost(HOST, GODOT)

    @classmethod
    def tearDownClass(cls):
        cls.host.close()
        diagnostics = "\n".join(cls.host.diagnostics)
        if "ERROR:" in diagnostics or "SCRIPT ERROR:" in diagnostics:
            raise AssertionError(diagnostics)

    def env(self, model="ripper", ticks=600):
        env = CybersandEnv(self.host, physical_config(model, ticks=ticks))
        observation, _ = env.reset()
        return env, observation

    def test_persistent_simple_and_articulated_profiles(self):
        pid = self.host.pid
        for model, part_count in (("monowheel", 2), ("ripper", 5)):
            with self.subTest(model=model):
                env, initial = self.env(model, ticks=120)
                self.assertEqual(initial["part_count"], part_count)
                self.assertEqual(len(initial["joints"]), initial["part_count"] - 1)
                for _ in range(20):
                    final, _, _, _, info = env.step({"move": [1, -0.15], "effort": 0.7})
                    for trace in info["trace"]:
                        self.assertEqual(trace["tick_after"], trace["tick_before"] + 1)
                        self.assertEqual(trace["world_tick"], trace["tick_after"])
                        self.assertEqual(trace["cellular_sample_tick"], trace["tick_after"] - 1)
                self.assertNotEqual(initial["position"], final["position"])
                self.assertEqual(final["tick"], 120)
                self.assertEqual(self.host.pid, pid)

    def test_repeat_stops_on_first_terminal_tick_and_cannot_award_again(self):
        env, initial = self.env(ticks=13)
        durations = []
        for _ in range(3):
            final, reward, terminated, truncated, info = env.step({"move": [1, 0]})
            durations.append(info["actual_ticks"])
            self.assertEqual(reward, 0.0)
        self.assertEqual(durations, [6, 6, 1])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(final["run_id"], initial["run_id"])
        self.assertEqual(final["tick"], 13)
        with self.assertRaises(EpisodeFinished):
            env.step({"move": [1, 0]})

    def test_owned_observation_and_trace_do_not_alias_host_or_next_result(self):
        env, view = self.env()
        original_position = view["position"].copy()
        view["position"][0] = 999999
        view["cell_patch"]["materials"][0] = 255
        fresh = env.observe()
        self.assertEqual(fresh["position"], original_position)
        first, _, _, _, info = env.step({"move": [1, 0]})
        first_saved = copy.deepcopy(first)
        info["trace"][0]["observation"]["position"][0] = -999999
        env.step({"move": [-1, 0]})
        self.assertEqual(first, first_saved)
        self.assertNotIn("actors", fresh)
        self.assertNotIn("cellular_results", fresh)
        self.assertNotIn("effect_totals", fresh)

    def test_duplicate_reordered_and_stale_packets_are_refused_by_actual_host(self):
        env, view = self.env()
        packet = make_physical_command(view, 10, {"move": [1, 0]})
        _, _, _, _, info = env.step_packet(packet, ticks=1)
        self.assertTrue(info["trace"][0]["admission"][0]["accepted"])
        for sequence in (10, 9):
            bad = copy.deepcopy(packet)
            bad["sequence"] = sequence
            _, _, _, _, info = env.step_packet(bad, ticks=1)
            self.assertEqual(info["trace"][0]["admission"][0]["reason"], "sequence")
        env.step_packet(None, ticks=10)
        stale = copy.deepcopy(packet)
        stale["sequence"] = 11
        _, _, _, _, info = env.step_packet(stale, ticks=1)
        self.assertFalse(info["trace"][0]["admission"][0]["accepted"])
        self.assertEqual(info["trace"][0]["admission"][0]["reason"], "expired_observation")

    def test_observation_anchored_lease_boundaries(self):
        for lease in (1, 12, 18):
            with self.subTest(lease=lease):
                env, view = self.env()
                packet = make_physical_command(view, 0, {"move": [1, 0]}, lease)
                obs, _, _, _, info = env.step_packet(packet, ticks=lease)
                self.assertTrue(info["trace"][0]["admission"][0]["accepted"])
                self.assertEqual(obs["tick"], lease)
                expired = copy.deepcopy(packet)
                expired["sequence"] = 1
                _, _, _, _, info = env.step_packet(expired, ticks=1)
                self.assertFalse(info["trace"][0]["admission"][0]["accepted"])
                self.assertEqual(info["trace"][0]["admission"][0]["reason"], "expired_observation")

    def test_reset_replaces_bodies_and_fences_old_actor_epoch_profile(self):
        env, view = self.env("needlejet")
        packet = make_physical_command(view, 0, {"move": [1, 0]})
        fresh, _ = env.reset(options={"config": physical_config("ripper")})
        self.assertNotEqual(view["run_id"], fresh["run_id"])
        self.assertGreater(fresh["epoch"], view["epoch"])
        self.assertNotEqual(view["part_count"], fresh["part_count"])
        _, _, _, _, info = env.step_packet(packet, ticks=1)
        self.assertEqual(info["trace"][0]["admission"][0]["reason"], "run_or_epoch")
        with self.assertRaises(ProtocolError):
            env.command({"move": [1, 0]}, observation=view)
        incompatible = env.observe()
        incompatible["profile_version"] += 1
        with self.assertRaises(ProtocolError):
            env.command({"move": [1, 0]}, observation=incompatible)

    def test_cancellation_fences_issued_views_without_erasing_motion(self):
        env, _ = self.env("astra")
        previous, _, _, _, _ = env.step({"move": [1, -1], "effort": 0.9})
        stale = env.command({"move": [1, 0]})
        cancelled = env.cancel()
        self.assertEqual(cancelled["tick"], previous["tick"])
        self.assertEqual(cancelled["velocity"], previous["velocity"])
        self.assertGreater(cancelled["control_generation"], previous["control_generation"])
        _, _, _, _, info = env.step_packet(stale, ticks=1)
        self.assertEqual(info["trace"][0]["admission"][0]["reason"], "observation_not_issued")

    def test_cancel_during_actual_charge_never_becomes_release_launch(self):
        env, _ = self.env("pouncer")
        # Let the real compliant rig establish ground contact before charging.
        for _ in range(20):
            view, _, _, _, _ = env.step({"move": [0, 0], "effort": 0.7})
            if view["grounded"]:
                break
        self.assertTrue(view["grounded"])
        charged, _, _, _, _ = env.step({"move": [0, 0], "primary": True, "effort": 0.7})
        self.assertGreater(charged["charge"], 0)
        cancelled = env.cancel()
        self.assertEqual(cancelled["charge"], 0.0)
        self.assertEqual(cancelled["velocity"], charged["velocity"])
        next_view, _, _, _, _ = env.step_packet(None, ticks=1)
        self.assertEqual(next_view["cooldown"], 0.0)
        self.assertLess(abs(next_view["velocity"][1] - cancelled["velocity"][1]), 20)

    def test_transport_assignment_cannot_control_companion(self):
        env = CybersandEnv(self.host, physical_config("ripper", companions=("copter",)))
        view, _ = env.reset()
        packet = make_physical_command(view, 0, {"move": [1, 0]})
        packet["actor_id"] = 1
        with self.assertRaisesRegex(HostError, "actor_assignment"):
            env.step_packet(packet)
        self.assertEqual(env.observe()["tick"], 0)

    def test_invalid_reset_does_not_destroy_previous_world(self):
        env, view = self.env()
        with self.assertRaises(HostError):
            env.reset(options={"config": physical_config("unknown-rig")})
        same = env.observe()
        self.assertEqual(same["run_id"], view["run_id"])
        self.assertEqual(same["tick"], view["tick"])

    def test_malformed_reset_actor_and_fractional_assignment_refuse_transactionally(self):
        env, view = self.env()
        for actor in (7, None, "ripper", []):
            config = physical_config()
            config["actors"] = [actor]
            with self.subTest(actor=actor), self.assertRaises(HostError):
                self.host.request("reset", config=config, actor_id=0)
            self.assertEqual(env.observe()["run_id"], view["run_id"])
        with self.assertRaises(HostError):
            self.host.request("reset", config=physical_config(), actor_id=0.5)
        self.assertEqual(env.observe()["run_id"], view["run_id"])


@unittest.skipUnless(ENABLED, "actual sampled runtime not configured; no sampled-host evidence")
class SampledHostTests(unittest.TestCase):
    def test_real_sampled_tag_score_is_awarded_as_an_increment_only_once(self):
        with GodotHost(HOST, GODOT, lane="sampled") as host:
            config = sampled_config("copter", ticks=300)
            config["actors"].append(dict(config["actors"][0], team=1))
            env = CybersandEnv(host, config)
            view, _ = env.reset()
            awarded = 0
            for index in range(40):
                direction = view["neighbors"][0]["relative_position"] if view["neighbors"] else [1, 0]
                magnitude = max(1, (direction[0] ** 2 + direction[1] ** 2) ** 0.5)
                action = {"movement": [0, -0.6], "aim": [value / magnitude for value in direction],
                          "effort": 0.7, "trim": 0, "reel": 0,
                          "primary": False, "secondary": False, "auxiliary": False}
                view, reward, terminated, truncated, _ = env.step_packet(env.command(action, tag=index % 6 == 0))
                if reward:
                    awarded = reward
                    break
                self.assertFalse(terminated or truncated)
            self.assertEqual(awarded, 1.0)
            self.assertEqual(env.step_packet(env.command(action, tag=False))[1], 0.0)

    def test_actual_sampled_wire_and_metal_remain_separate(self):
        for metal in (False, True):
            with self.subTest(metal=metal), GodotHost(HOST, GODOT, lane="sampled") as host:
                env = CybersandEnv(host, sampled_config(ticks=7, metal=metal))
                first, _ = env.reset()
                action = ({"actuators": [0.8, 0, 0, 0], "tag_aim": [1, 0]} if metal else
                          {"movement": [1, 0], "aim": [1, 0], "effort": 0.7, "trim": 0,
                           "reel": 0, "primary": False, "secondary": False, "auxiliary": False})
                for _ in range(2):
                    final, reward, terminated, truncated, info = env.step(action)
                    self.assertEqual(reward, 0.0)
                self.assertEqual(info["actual_ticks"], 1)
                self.assertTrue(truncated)
                self.assertFalse(terminated)
                self.assertEqual(first["control_mode"], "metal" if metal else "wire")
                self.assertEqual(info["evidence_class"], "actual-sampled-host")


@unittest.skipUnless(ENABLED, "actual realtime runtime not configured; no delay evidence")
class ContinuousHostTests(unittest.TestCase):
    def test_terminal_reached_during_inference_returns_valid_final_transition_once(self):
        with GodotHost(HOST, GODOT, mode="realtime") as host:
            env = CybersandEnv(host, physical_config("astra", ticks=12))
            first, _ = env.reset()
            time.sleep(0.35)
            last, reward, terminated, truncated, info = env.step({"move": [1, 0]})
            self.assertEqual(last["tick"], 12)
            self.assertEqual(last["run_id"], first["run_id"])
            self.assertEqual(info["actual_ticks"], 0)
            self.assertEqual(info["elapsed_ticks"], 12)
            self.assertEqual(len(info["idle_trace"]), 12)
            self.assertEqual(len(info["unconsumed_terminal_commands"]), 1)
            self.assertEqual(reward, 0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            with self.assertRaises(EpisodeFinished):
                env.step({"move": [1, 0]})

    def test_world_keeps_advancing_and_delayed_command_is_not_restamped(self):
        with GodotHost(HOST, GODOT, mode="realtime") as host:
            env = CybersandEnv(host, physical_config("astra", ticks=180))
            initial, _ = env.reset()
            packet = env.command({"move": [1, -1]})
            time.sleep(0.38)
            final, _, _, _, info = env.step_packet(packet)
            self.assertGreater(info["elapsed_ticks"], 12)
            self.assertGreater(info["delay_ticks_before_repeat"], 0)
            self.assertEqual(info["actual_ticks"], 6)
            self.assertFalse(info["trace"][0]["admission"][0]["accepted"])
            self.assertEqual(info["trace"][0]["requested"][0]["observed_tick"], initial["tick"])
            self.assertGreater(final["tick"], initial["tick"] + 12)
            self.assertEqual(info["idle_trace_dropped"], 0)
            self.assertEqual(len(info["idle_trace"]), info["delay_ticks_before_repeat"])
            self.assertTrue(info["consumed_trace_complete"])


if __name__ == "__main__":
    unittest.main()
