"""Actual evaluated bundle + pinned host integration; absent artifacts skip explicitly."""

import importlib.util
import os
from pathlib import Path
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
ENABLED = HAS_NUMPY and all(os.environ.get(key) for key in
                          ("NCA_HOST_CHECKOUT", "NCA_GODOT", "NCA_EXPORT_BUNDLE", "NCA_EXPORT_BUNDLE_SHA256"))
if HAS_NUMPY:
    from nca.contracts import ContractError
    from nca.controllers.baseline import SkillGoal
    from nca.envs import CybersandEnv, GodotHost, physical_config
    from nca.export import ExportController, RuntimeBundle, context_for, dispatch
    from nca.export.build import read_json
    from nca.bench.common import pose


@unittest.skipUnless(ENABLED, "evaluated CPU export bundle and actual host not configured; no deployment evidence")
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
            view, _, _, _, info = dispatch(env, result, issued)
            self.assertEqual(info["actual_ticks"], 6)
            self.assertTrue(info["trace"][0]["admission"][0]["accepted"])
            self.assertEqual(result.source.observation_id, issued["observation_id"])
        cancelled = dispatch(env, controller.cancel(), view)
        self.assertEqual(pose(cancelled), pose(view))

    def test_real_newer_view_cannot_refresh_old_result_lease(self):
        env, original, controller, goal = self.environment()
        result = controller.step(original, goal)
        later, _, _, _, _ = env.step_packet(None)
        with self.assertRaisesRegex(ContractError, "decision_source_mismatch"):
            dispatch(env, result, later)
        cancelled = env.observe()
        self.assertEqual(cancelled["control_generation"], later["control_generation"] + 1)
        self.assertEqual(pose(cancelled), pose(later))
        self.assertEqual(cancelled["last_sequence"], original["last_sequence"])

    def test_real_takeover_retires_active_goal_and_preserves_motion(self):
        env, original, controller, goal = self.environment()
        result = controller.step(original, goal, source_generation=1)
        view, _, _, _, _ = dispatch(env, result, original)
        takeover = controller.step(view, goal, source_generation=2)
        self.assertEqual(takeover.reason, "owner_or_mount_changed")
        cancelled = dispatch(env, takeover, view)
        self.assertEqual(pose(cancelled), pose(view))
        self.assertEqual(cancelled["control_generation"], view["control_generation"] + 1)

    def test_real_old_actor_result_is_rejected_after_reset(self):
        env, original, controller, goal = self.environment()
        result = controller.step(original, goal)
        fresh, _ = env.reset()
        with self.assertRaisesRegex(ContractError, "decision_source_mismatch"):
            dispatch(env, result, fresh)
        current = env.observe()
        self.assertEqual(current["run_id"], fresh["run_id"])
        self.assertEqual(pose(current), pose(fresh))


if __name__ == "__main__":
    unittest.main()
