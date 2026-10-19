"""Opt-in real native/Rapier usable checks; no surrogate acceptance path."""

import math
import os
import unittest

from nca.envs import CybersandEnv, GodotHost, physical_config
from nca.skills import SkillRuntime, interval_observations
from nca.skills.tasks import SkillSpec, TaskPlan, TaskStep, run_task


@unittest.skipUnless(os.environ.get("NCA_HOST_CHECKOUT") and os.environ.get("NCA_GODOT"),
                     "real accepted Cybersand runtime not configured; no usable qualification")
class HostSkillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = GodotHost(os.environ["NCA_HOST_CHECKOUT"], os.environ["NCA_GODOT"])

    @classmethod
    def tearDownClass(cls):
        cls.host.close()

    def environment(self, model="astra", position=(300, 300), ticks=360):
        config = physical_config(model, ticks=ticks, seed=73)
        if position is not None:
            config["actors"][0]["position"] = list(position)
        config["local_patch_radius"] = 8
        env = CybersandEnv(self.host, config)
        view, _ = env.reset()
        runtime = SkillRuntime(self.host.request("profile")["profile"])
        return env, view, runtime

    def drive(self, env, runtime, view, skill, parameters, source="script", maximum=50):
        receipt = runtime.start(skill, parameters, view, source=source)
        self.assertTrue(receipt.accepted, receipt.reason)
        observations = ()
        traces = []
        for _ in range(maximum + 1):
            result = runtime.progress(view, observations=observations)
            if result.terminal:
                return receipt, result, view, traces
            view, _, terminated, truncated, info = env.step(result.action.to_dict())
            self.assertFalse(terminated or truncated)
            observations = interval_observations(info)
            traces.extend(info["trace"])
        self.fail("bounded usable fixture did not terminate")

    def test_launcher_requires_actual_local_effect_and_preserves_interval_receipt(self):
        env, view, runtime = self.environment()
        receipt, result, view, trace = self.drive(env, runtime, view, "aim_use", {"aim": [0, -1], "require_change": True})
        self.assertTrue(receipt.accepted)
        self.assertTrue(result.local_success)
        self.assertIsNone(result.evaluator_success)
        self.assertEqual(result.effects[0]["status"], "applied")
        self.assertLess(result.effects[0]["tick"], view["tick"])
        self.assertEqual(view["last_effects"], [])
        self.assertTrue(any(t["observation"]["applied_impulse"][1] > 0 for t in trace))
        self.assertTrue(result.cancel_host)
        env.cancel()

    def test_clearance_refusal_can_recoil_without_skill_success(self):
        env, view, runtime = self.environment()
        _, result, view, trace = self.drive(env, runtime, view, "aim_use", {"aim": [1, 0]})
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "effect_blocked")
        self.assertTrue(any(t["observation"]["applied_impulse"][0] < -1 for t in trace))
        cancelled = env.cancel()
        self.assertEqual(cancelled["velocity"], view["velocity"])

    def test_applied_native_cells_and_motion_survive_owner_cancellation(self):
        env, view, runtime = self.environment(position=(320, 243))
        _, result, view, _ = self.drive(env, runtime, view, "aim_use",
                                       {"aim": [0.4, -math.sqrt(0.84)], "require_change": True})
        self.assertTrue(result.local_success)
        self.assertGreater(result.effects[0]["changed_cells"], 0)
        self.assertIn(6, view["cell_patch"]["materials"])
        cancelled = env.cancel()
        self.assertEqual(cancelled["cell_patch"], view["cell_patch"])
        self.assertEqual(cancelled["velocity"], view["velocity"])
        advanced, _, _, _, info = env.step_packet(None, ticks=6)
        self.assertEqual(advanced["tick"], view["tick"] + 6)
        self.assertTrue(all(not t["observation"]["last_effects"] for t in info["trace"]))

    def test_actual_detach_and_early_grip_reuse_existing_articulated_part(self):
        env, view, runtime = self.environment("hammerhead")
        initial_names = [part["id"] for part in view["parts"]]
        _, released, view, _ = self.drive(env, runtime, view, "release", {})
        self.assertTrue(released.local_success)
        self.assertTrue(view["parts"][4]["detached"])
        self.assertTrue(any(not joint["enabled"] for joint in view["joints"] if joint["b"] == 4))
        view = env.cancel()
        _, gripped, view, _ = self.drive(env, runtime, view, "grip", {})
        self.assertTrue(gripped.local_success, gripped.reason)
        self.assertFalse(view["parts"][4]["detached"])
        self.assertEqual([part["id"] for part in view["parts"]], initial_names)
        self.assertTrue(any(joint["enabled"] for joint in view["joints"] if joint["b"] == 4))
        env.cancel()

    def test_articulated_tool_reports_no_change_without_claiming_target_damage(self):
        env, view, runtime = self.environment("hammerhead")
        for _ in range(4):
            view, *_ = env.step({"move": [0, 0]})
        _, result, view, _ = self.drive(env, runtime, view, "aim_use", {"aim": [1, 0]})
        self.assertTrue(result.local_success, result.reason)
        self.assertEqual(result.reason, "local_effect_consumed_no_change")
        self.assertEqual(result.effects[0]["changed_cells"], 0)
        self.assertIsNone(result.evaluator_success)
        env.cancel()

    def test_cancel_keeps_real_connection_generation_and_fences_old_control(self):
        env, view, runtime = self.environment("hammerhead")
        runtime.mount_ref(view)
        _, released, view, _ = self.drive(env, runtime, view, "release", {})
        self.assertTrue(released.local_success)
        detached = runtime.mount_ref(view)
        self.assertEqual(detached.attachment_generation, 1)
        before_cancel = view
        view = env.cancel()
        current = runtime.mount_ref(view)
        self.assertEqual(current.attachment_generation, detached.attachment_generation)
        self.assertGreater(current.actor.control_generation, detached.actor.control_generation)
        self.assertEqual(view["velocity"], before_cancel["velocity"])
        self.assertFalse(runtime.start("grip", {}, view, mount_ref=detached).accepted)
        _, gripped, view, _ = self.drive(env, runtime, view, "grip", {})
        self.assertTrue(gripped.local_success)
        self.assertEqual(runtime.mount_ref(view).attachment_generation, 2)
        view = env.cancel()
        self.assertEqual(runtime.mount_ref(view).attachment_generation, 2)

    def test_charge_cancel_does_not_become_release_launch(self):
        env, view, runtime = self.environment("pouncer", position=None)
        self.assertTrue(runtime.start("charge_hold", {}, view).accepted)
        observations = ()
        for _ in range(10):
            result = runtime.progress(view, observations=observations)
            self.assertFalse(result.terminal)
            view, _, _, _, info = env.step(result.action.to_dict())
            observations = interval_observations(info)
            if view["charge"] >= 0.3:
                break
        self.assertGreaterEqual(view["charge"], 0.3)
        cancelled = runtime.cancel("human_takeover")
        self.assertTrue(cancelled.cancel_host)
        before = view
        view = env.cancel()
        self.assertEqual(view["charge"], 0)
        self.assertEqual(view["velocity"], before["velocity"])
        view, _, _, _, info = env.step_packet(None, ticks=6)
        self.assertTrue(all(t["observation"]["charge"] == 0 for t in info["trace"]))
        self.assertTrue(all(math.hypot(*t["observation"]["applied_impulse"]) < 0.001 for t in info["trace"]))

    def test_interrupted_aim_has_no_later_unrequested_trigger(self):
        env, view, runtime = self.environment()
        runtime.start("aim_use", {"settle_ticks": 12}, view)
        aiming = runtime.progress(view)
        self.assertFalse(aiming.action.tool)
        view, *_ = env.step(aiming.action.to_dict())
        runtime.cancel("interrupted_aim")
        view = env.cancel()
        for _ in range(3):
            self.assertEqual(runtime.progress(view).status, "cancelled")
            view, _, _, _, info = env.step_packet(None)
            self.assertTrue(all(not t["observation"]["last_effects"] for t in info["trace"]))

    def test_reset_fences_actual_old_mount_and_object_lifetime(self):
        env, view, runtime = self.environment("hammerhead")
        old_mount = runtime.mount_ref(view)
        view, _ = env.reset()
        receipt = runtime.start("release", {}, view, mount_ref=old_mount)
        self.assertFalse(receipt.accepted)
        self.assertEqual(receipt.reason, "stale_mount_reference")
        self.assertFalse(view["parts"][4]["detached"])

    def test_compound_task_records_clearance_failure_and_declared_recovery(self):
        env, view, runtime = self.environment(ticks=600)
        plan = TaskPlan("host-compound", (
            TaskStep(SkillSpec("approach", {"target": [304, 300], "tolerance": 2.5})),
            TaskStep(SkillSpec("aim_use", {"aim": [1, 0]}),
                     (SkillSpec("recover"), SkillSpec("aim_use", {"aim": [0, -1], "require_change": True}))),
        ))
        result, view, trace = run_task(env, runtime, plan, view)
        self.assertTrue(result.local_success)
        self.assertEqual([a.result.status for a in result.attempts], ["succeeded", "failed", "succeeded", "succeeded"])
        self.assertEqual(result.attempts[1].result.reason, "effect_blocked")
        self.assertEqual([a.role for a in result.attempts], ["primary", "primary", "recovery", "recovery"])
        self.assertGreater(view["position"][0], 301)
        self.assertEqual(len(trace), result.decisions)

    def test_three_source_labels_share_the_actual_launcher_owner(self):
        sequences = []
        for source in ("human", "script", "model"):
            env, view, runtime = self.environment()
            _, result, _, traces = self.drive(env, runtime, view, "aim_use", {"aim": [0, -1]}, source=source)
            self.assertTrue(result.local_success)
            sequences.append([t["requested"][0]["action"] for t in traces if t["requested"]])
            env.cancel()
        self.assertEqual(sequences[0], sequences[1])
        self.assertEqual(sequences[1], sequences[2])

    def test_articulated_compound_carries_uses_and_releases(self):
        env, view, runtime = self.environment("hammerhead", ticks=360)
        plan = TaskPlan("host-held-compound", (
            TaskStep(SkillSpec("release")), TaskStep(SkillSpec("grip")),
            TaskStep(SkillSpec("carry", {"target": [306, 344], "tolerance": 4,
                                        "timeout_ticks": 240, "no_progress_ticks": 90})),
            TaskStep(SkillSpec("aim_use", {"aim": [1, 0], "require_change": True})),
            TaskStep(SkillSpec("release")),
        ))
        result, view, trace = run_task(env, runtime, plan, view)
        self.assertTrue(result.local_success)
        self.assertEqual(len(result.attempts), 5)
        self.assertTrue(all(attempt.result.local_success for attempt in result.attempts))
        self.assertGreater(result.attempts[3].result.effects[0]["changed_cells"], 0)
        self.assertTrue(view["parts"][4]["detached"])
        carry_ticks = [tick for step in trace if step["phase"] == "carry_attached" for tick in step["consumed"]]
        self.assertTrue(carry_ticks)
        self.assertTrue(all(not tick["detached"][4] for tick in carry_ticks))

    def test_late_pickup_fails_explicitly_when_part_remains_out_of_reach(self):
        env, view, runtime = self.environment("hammerhead", position=(300, 344))
        for _ in range(4):
            view, *_ = env.step({"move": [0, 0]})
        plan = TaskPlan("host-late-pickup", (
            TaskStep(SkillSpec("release")), TaskStep(SkillSpec("grip")),
        ))
        result, view, _ = run_task(env, runtime, plan, view)
        self.assertFalse(result.local_success)
        self.assertEqual(result.attempts[-1].result.reason, "no_progress")
        self.assertTrue(view["parts"][4]["detached"])
        self.assertGreater(result.attempts[-1].result.distance, 8)


if __name__ == "__main__":
    unittest.main()
