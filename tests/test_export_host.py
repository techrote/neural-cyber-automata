"""Separate evaluated-bundle tests and synthetic-policy actual-owner conformance."""

from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
ENABLED = HAS_NUMPY and all(os.environ.get(key) for key in
                          ("NCA_HOST_CHECKOUT", "NCA_GODOT", "NCA_EXPORT_BUNDLE", "NCA_EXPORT_BUNDLE_SHA256"))
OWNER_ENABLED = HAS_NUMPY and all(os.environ.get(key) for key in ("NCA_HOST_CHECKOUT", "NCA_GODOT"))
if HAS_NUMPY:
    from nca.contracts import ContractError
    from nca.controllers.baseline import SkillGoal
    from nca.envs import CybersandEnv, GodotHost, physical_config
    from nca.export import ExportController, RuntimeBundle, context_for, dispatch
    from nca.export.build import read_json
    from nca.bench.common import pose
    from test_export_runtime import bundle as synthetic_bundle


@unittest.skipUnless(ENABLED, "NumPy, evaluated CPU export bundle and actual host are required; no deployment evidence")
class ExportHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = GodotHost(os.environ["NCA_HOST_CHECKOUT"], os.environ["NCA_GODOT"])
        cls.bundle_path = Path(os.environ["NCA_EXPORT_BUNDLE"])
        payload, _ = read_json(cls.bundle_path)
        cls.bundle = RuntimeBundle.load(cls.bundle_path, context_for(payload["profile"], cls.host.identity),
                                        expected_sha256=os.environ["NCA_EXPORT_BUNDLE_SHA256"])

    @classmethod
    def tearDownClass(cls):
        cls.host.close()
        if any("SCRIPT ERROR:" in line or "ERROR:" in line for line in cls.host.diagnostics):
            raise AssertionError("actual export integration host emitted a script/runtime error")

    def environment(self):
        config = physical_config("jev", ticks=120, seed=73)
        config["actors"][0]["position"] = [176., 346.5]
        env = CybersandEnv(self.host, config)
        view, _ = env.reset()
        self.addCleanup(env.cancel)
        return env, view, ExportController(self.bundle), SkillGoal("test-hop", "directed_hop", (200., 346.5), timeout_seconds=4)

    def test_real_profile_and_sequential_numeric_owner_requests(self):
        env, view, controller, goal = self.environment()
        self.assertEqual(context_for(self.host.request("profile")["profile"], self.host.identity), self.bundle.context)
        for _ in range(4):
            result = controller.step(view, goal)
            self.assertEqual(result.status, "accepted")
            issued = view
            view, _, _, _, info = dispatch(env, controller, result, issued)
            self.assertEqual(info["actual_ticks"], 6)
            self.assertTrue(info["trace"][0]["admission"][0]["accepted"])
            self.assertEqual(result.source.observation_id, issued["observation_id"])
        cancelled = dispatch(env, controller, controller.cancel(), view)
        self.assertEqual(pose(cancelled), pose(view))

    def test_real_newer_view_cannot_refresh_old_result_lease(self):
        env, original, controller, goal = self.environment()
        result = controller.step(original, goal)
        later, _, _, _, _ = env.step_packet(None)
        with self.assertRaisesRegex(ContractError, "decision_source_mismatch"):
            dispatch(env, controller, result, later)
        cancelled = env.observe()
        self.assertEqual(cancelled["control_generation"], later["control_generation"] + 1)
        self.assertEqual(pose(cancelled), pose(later))
        self.assertEqual(cancelled["last_sequence"], original["last_sequence"])

    def test_real_takeover_retires_active_goal_and_preserves_motion(self):
        env, original, controller, goal = self.environment()
        result = controller.step(original, goal, source_generation=1)
        view, _, _, _, _ = dispatch(env, controller, result, original)
        takeover = controller.step(view, goal, source_generation=2)
        self.assertEqual(takeover.reason, "owner_or_mount_changed")
        cancelled = dispatch(env, controller, takeover, view)
        self.assertEqual(pose(cancelled), pose(view))
        self.assertEqual(cancelled["control_generation"], view["control_generation"] + 1)

    def test_real_old_actor_result_is_rejected_after_reset(self):
        env, original, controller, goal = self.environment()
        result = controller.step(original, goal)
        fresh, _ = env.reset()
        with self.assertRaisesRegex(ContractError, "decision_source_mismatch"):
            dispatch(env, controller, result, fresh)
        current = env.observe()
        self.assertEqual(current["run_id"], fresh["run_id"])
        self.assertEqual(pose(current), pose(fresh))


@unittest.skipUnless(OWNER_ENABLED, "NumPy and pinned actual host are required; synthetic policy is not deployment evidence")
class ExportOwnerBoundaryTests(unittest.TestCase):
    """Real native receipts with synthetic weights, never a learned-skill result."""
    @classmethod
    def setUpClass(cls):
        cls.host = GodotHost(os.environ["NCA_HOST_CHECKOUT"], os.environ["NCA_GODOT"])
        cls.bundle = synthetic_bundle()

    @classmethod
    def tearDownClass(cls):
        cls.host.close()
        if any("SCRIPT ERROR:" in line or "ERROR:" in line for line in cls.host.diagnostics):
            raise AssertionError("actual boundary host emitted a script/runtime error")

    def environment(self, *, ticks=60, duration=4):
        config = physical_config("jev", ticks=ticks, seed=73)
        config["actors"][0]["position"] = [176., 346.5]
        env = CybersandEnv(self.host, config)
        original, _ = env.reset()
        self.addCleanup(env.cancel)
        self.assertEqual(self.host.request("profile")["profile"], self.bundle.value["profile"])
        return env, original, ExportController(self.bundle), SkillGoal(
            "synthetic-boundary", "directed_hop", (200., 346.5), timeout_seconds=duration)

    def test_native_goal_deadline_caps_lease_and_final_one_tick_repeat(self):
        env, original, controller, goal = self.environment(duration=7 / 60)
        first = controller.step(original, goal)
        view, *_ = dispatch(env, controller, first, original)
        self.assertEqual(first.helper_lease_ticks, 7)
        second = controller.step(view, goal)
        final, _, _, _, info = dispatch(env, controller, second, view)
        self.assertEqual((final["tick"], info["actual_ticks"], second.helper_lease_ticks), (7, 1, 1))
        self.assertEqual(info["trace"][0]["requested"][0]["lease_ticks"], 1)
        self.assertEqual(final["lease_remaining"], 0)
        self.assertTrue(controller.owner_cleanup_unconfirmed)
        expired, _, _, _, passive = env.step_packet(None, ticks=1)
        self.assertEqual(expired["control_generation"], final["control_generation"] + 1)
        self.assertEqual(passive["trace"][0]["applied"]["action"], {})

    def test_native_delay_does_not_restamp_source(self):
        env, original, controller, goal = self.environment()
        delayed, *_ = env.step_packet(None, ticks=2)
        decision = controller.step(original, goal, current_tick=delayed["tick"])
        final, _, _, _, info = dispatch(env, controller, decision, original)
        self.assertEqual(final["tick"], 8)
        self.assertEqual(info["trace"][0]["requested"][0]["observed_tick"], 0)
        controller.confirm_owner_cancel(env.cancel())
        self.assertFalse(controller.owner_cleanup_unconfirmed)

    def test_native_repeat_dispatch_cancels_without_second_packet(self):
        env, original, controller, goal = self.environment()
        decision = controller.step(original, goal)
        before, *_ = dispatch(env, controller, decision, original)
        with self.assertRaisesRegex(ContractError, "decision_source_mismatch"):
            dispatch(env, controller, decision, original)
        after = env.observe()
        self.assertEqual(after["tick"], before["tick"])
        self.assertEqual(after["last_sequence"], before["last_sequence"])
        self.assertEqual(after["control_generation"], before["control_generation"] + 1)
        self.assertFalse(controller.owner_cleanup_unconfirmed)

    def test_native_new_source_requires_revocation_before_new_goal(self):
        env, original, controller, goal = self.environment()
        decision = controller.step(original, goal)
        view, *_ = dispatch(env, controller, decision, original)
        replacement = controller.step(view, replace(goal, id="replacement"), source_generation=2)
        self.assertEqual(replacement.reason, "owner_or_mount_changed")
        cancelled = dispatch(env, controller, replacement, view)
        self.assertEqual(cancelled["control_generation"], view["control_generation"] + 1)
        fresh = controller.step(cancelled, replace(goal, id="after-cancel"), source_generation=2)
        self.assertEqual(fresh.status, "accepted")
        self.assertEqual(fresh.hidden_before, (0.,) * 32)
        dispatch(env, controller, fresh, cancelled)

    def test_native_late_dispatch_at_goal_deadline_is_cancelled(self):
        env, original, controller, goal = self.environment(duration=.1)
        decision = controller.step(original, goal)
        delayed, *_ = env.step_packet(None, ticks=6)
        with self.assertRaisesRegex(ContractError, "dispatch_deadline"):
            dispatch(env, controller, decision, original, current_tick=delayed["tick"])
        after = env.observe()
        self.assertEqual(after["tick"], 6)
        self.assertEqual(after["last_sequence"], original["last_sequence"])
        self.assertEqual(after["control_generation"], original["control_generation"] + 1)

    def test_native_terminal_readback_retires_without_inventing_last_actuation(self):
        env, original, controller, goal = self.environment(ticks=4)
        decision = controller.step(original, goal)
        final, _, terminated, truncated, info = dispatch(env, controller, decision, original)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(final["tick"], 4)
        self.assertIsNone(info["trace"][-1]["applied"]["action"])
        self.assertTrue(controller.owner_cleanup_unconfirmed)


if __name__ == "__main__":
    unittest.main()
