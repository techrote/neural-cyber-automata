"""Actual ACT004 consumed-input replay. A skip supplies no physical evidence."""

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from nca.adapters.godot import CybersandEnv, GodotHost, physical_config
from nca.contracts import ContractError, sha256_json
from nca.data import ActionSource, EpisodeRecorder, read_episode, replay_episode, write_episode

HOST = os.environ.get("NCA_HOST_CHECKOUT")
GODOT = os.environ.get("NCA_GODOT")


@unittest.skipUnless(HOST and GODOT, "actual ACT004 runtime not configured; no dataset/replay qualification")
class ActualDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = GodotHost(HOST, GODOT)

    @classmethod
    def tearDownClass(cls):
        cls.host.close()
        diagnostics = "\n".join(cls.host.diagnostics)
        if "ERROR:" in diagnostics or "SCRIPT ERROR:" in diagnostics:
            raise AssertionError(diagnostics)

    def recorder(self, id="actual-data", ticks=32):
        env = CybersandEnv(self.host, physical_config("jev", ticks=ticks, seed=7))
        return EpisodeRecorder(env, id, ActionSource("script-a", "scripted", "test-script/v1", "a" * 64))

    def test_full_consumed_trace_replay_with_refusal_cancel_and_takeover(self):
        recorder = self.recorder()
        recorder.step({"move": [.7, 0], "effort": .7})
        first = recorder.events[-1].requested_packet
        recorder.step_packet(first, ticks=1)
        self.assertFalse(recorder.events[-1].imitation_eligible)
        self.assertEqual(recorder.events[-1].ticks[0].admission[0]["reason"], "sequence")
        loaded = recorder.view
        self.assertTrue(any(joint["anchor_error"] > 0 for joint in loaded["joints"]))
        canceled = recorder.cancel("test_cancel")
        self.assertEqual(canceled["velocity"], loaded["velocity"])
        self.assertEqual(canceled["parts"], loaded["parts"])
        for previous, current in zip(loaded["joints"], canceled["joints"]):
            for field in ("anchor_error", "angle", "speed"):
                self.assertEqual(current[field], previous[field])
            self.assertEqual(current["target"], 0.0)
        recorder.step({"move": [-.4, 0], "effort": .5})
        queued = recorder.env.command({"move": [1, 0], "primary": True})
        recorder.takeover(ActionSource("script-b", "scripted", "test-correction/v1", "b" * 64))
        recorder.step_packet(queued, ticks=1)
        refused = recorder.events[-1]
        self.assertEqual(refused.ticks[0].admission[0]["reason"], "observation_not_issued")
        self.assertIsNone(refused.ticks[0].consumed_request)
        while recorder.view["tick"] < 32:
            recorder.step({"move": [.1, 0], "effort": .4})
        episode = recorder.finish()
        self.assertEqual(episode.consumed_ticks, 32)
        self.assertEqual(episode.decisions, 7)
        self.assertTrue(episode.events[-1].truncated)
        self.assertTrue(episode.events[-1].imitation_eligible)
        self.assertTrue(all(not event.on_policy_eligible for event in episode.events))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "actual.json.gz"
            write_episode(path, episode)
            restored = read_episode(path)
        report = replay_episode(restored, self.host)
        self.assertTrue(report.passed, report.mismatches)
        self.assertEqual(report.replayed_ticks, 32)
        self.assertEqual(report.replayed_requests, 5)
        self.assertEqual(report.omitted_rejected_requests, 2)
        self.assertEqual(report.cancellations, 2)
        self.assertTrue(all(pair.original_run_id != pair.replay_run_id for pair in report.identities))
        self.assertTrue(all(pair.observed_tick <= pair.consumed_tick for pair in report.identities))

    def test_expiring_input_is_not_relabelled_as_fresh_or_as_continuous_consumption(self):
        recorder = self.recorder("one-tick-lease", ticks=6)
        packet = recorder.env.command({"move": [.5, 0]}, lease_ticks=1)
        recorder.step_packet(packet)
        episode = recorder.finish()
        ticks = episode.events[0].ticks
        self.assertIsNotNone(ticks[0].consumed_request)
        self.assertTrue(all(tick.consumed_request is None for tick in ticks[1:]))
        self.assertFalse(episode.events[0].imitation_eligible)
        self.assertTrue(replay_episode(episode, self.host).passed)

    def test_reset_creates_a_distinct_episode_and_fences_prior_packet(self):
        recorder = self.recorder("before-reset", ticks=60)
        recorder.step({"move": [1, 0], "effort": .8})
        old = recorder.env.command({"move": [1, 0]})
        before = recorder.finish(status="canceled", reason="explicit_reset_boundary")
        fresh = EpisodeRecorder(recorder.env, "after-reset", recorder.source)
        self.assertNotEqual(before.initial_observation["run_id"], fresh.view["run_id"])
        self.assertEqual(fresh.view["tick"], 0)
        self.assertEqual(fresh.view["velocity"], [0.0, 0.0])
        self.assertEqual(fresh.view["charge"], 0.0)
        fresh.step_packet(old, ticks=1)
        self.assertEqual(fresh.events[-1].ticks[0].admission[0]["reason"], "run_or_epoch")
        self.assertIsNone(fresh.events[-1].ticks[0].consumed_request)
        fresh.finish(status="canceled", reason="reset_fence_checked")

    def test_delayed_request_uses_original_policy_view_and_observation_age(self):
        recorder = self.recorder("delayed-source", ticks=9)
        original_view = recorder.view
        original = recorder.env.command({"move": [.5, 0]})
        recorder.step_packet(None, ticks=3)
        recorder.step_packet(original, ticks=6)
        event = recorder.events[-1]
        self.assertEqual(event.issued_observation["tick"], 3)
        self.assertEqual(event.source_observation["tick"], 0)
        self.assertEqual(event.policy_observation["observation_id"], original_view["observation_id"])
        self.assertTrue(event.imitation_eligible)
        report = replay_episode(recorder.finish(), self.host)
        self.assertTrue(report.passed, report.mismatches)
        self.assertEqual(report.identities[0].observed_tick, 0)
        self.assertEqual(report.identities[0].consumed_tick, 3)

    def test_runtime_profile_and_missing_receipts_refuse_replay(self):
        recorder = self.recorder("refusal-data", ticks=6)
        recorder.step({"move": [.2, 0]})
        episode = recorder.finish()
        drift = dict(episode.host_identity)
        drift["native_artifact_sha256"] = "0" * 64
        with self.assertRaisesRegex(ContractError, "replay_host_mismatch"):
            replay_episode(replace(episode, host_identity=drift), self.host)
        with self.assertRaisesRegex(ContractError, "replay_profile"):
            replay_episode(replace(episode, profile={"changed": True}), self.host)
        incomplete = replace(episode.events[0], valid=False, trace_complete=False)
        with self.assertRaisesRegex(ContractError, "replay_incomplete"):
            replay_episode(replace(episode, events=(incomplete,)), self.host)


if __name__ == "__main__":
    unittest.main()
