"""Behavioral checks for the distinct Jolt arrival reference; no game or fit."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import unittest

from nca.contracts import sha256_json
from nca.controllers.baseline import AuthoredController, FeedbackObservation, SkillGoal
from nca.controllers.jolt_arrival import (BASELINE_SOURCE_SHA256, CONTROLLER_ID,
                                         JoltArrivalController)


def observation(tick=0, **changes):
    values = dict(tick=tick, lifetime="episode:one", actor="jev", position=(0.0, 0.0),
                  velocity=(0.0, 0.0), angle=0.0, angular_velocity=0.0,
                  supported=True, cooldown=0.0, energy=1.0,
                  ground_distance=2.0, ceiling_distance=45.0)
    values.update(changes)
    return FeedbackObservation(**values)


class JoltArrivalTests(unittest.TestCase):
    def setUp(self):
        self.goal = SkillGoal("hop-one", "directed_hop", (48.0, 0.0), timeout_seconds=4.0)

    def test_distinct_specification_preserves_original_source(self):
        location = Path(__file__).resolve().parents[1] / "src/nca/controllers/baseline.py"
        self.assertEqual(hashlib.sha256(location.read_bytes()).hexdigest(), BASELINE_SOURCE_SHA256)
        controller = JoltArrivalController()
        specification = controller.specification()
        self.assertEqual(json.loads(json.dumps(specification)), specification)
        self.assertEqual(controller.specification_sha256, sha256_json(specification))
        self.assertEqual(controller.controller_id, CONTROLLER_ID)
        self.assertNotEqual(controller.controller_id, AuthoredController("jev").controller_id)
        frame = controller.step(observation(), self.goal)
        self.assertEqual(frame.controller_id, controller.controller_id)
        self.assertEqual(frame.tuning_hash, controller.specification_sha256)

    def test_distant_supported_goal_gets_one_purposeful_aimed_pulse(self):
        controller = JoltArrivalController()
        self.assertFalse(controller.step(observation(), self.goal).action.primary)
        launch = controller.step(observation(6), self.goal)
        self.assertTrue(launch.action.primary)
        self.assertEqual(launch.phase, "launch")
        self.assertGreater(launch.action.aim[0], .5)
        self.assertLess(launch.action.aim[1], -.4)
        self.assertGreaterEqual(launch.action.effort, .35)
        self.assertLessEqual(launch.action.effort, self.goal.pace)
        # Even a repeated fully recharged, supported observation cannot turn
        # the same invocation into the old periodic relaunch controller.
        for tick in range(12, 240, 6):
            frame = controller.step(observation(tick), self.goal)
            self.assertFalse(frame.action.primary)
            self.assertFalse(frame.action.secondary)

    def test_short_launch_spends_less_impulse_than_distant_launch(self):
        efforts = []
        for target in (24., 60.):
            controller = JoltArrivalController()
            goal = replace(self.goal, target=(target, 0.0))
            controller.step(observation(), goal)
            launch = controller.step(observation(6), goal)
            self.assertTrue(launch.action.primary)
            efforts.append(launch.action.effort)
        self.assertLess(efforts[0], efforts[1])

    def test_fast_near_goal_approach_withholds_even_first_pulse(self):
        # This is the observed dangerous relaunch state, used here as a new
        # invocation to prove that velocity/room gating itself withholds it.
        for direction in (-1., 1.):
            controller = JoltArrivalController()
            goal = replace(self.goal, target=(direction * 13.61, 0.0))
            for tick in (0, 6, 12):
                frame = controller.step(observation(tick, velocity=(direction * 27.29, 0.0)), goal)
                self.assertFalse(frame.action.primary)
        controller = JoltArrivalController()
        near = replace(self.goal, target=(8.0, 0.0))
        for tick in (0, 6, 12):
            self.assertFalse(controller.step(observation(tick, velocity=(45., 0.)), near).action.primary)

    def test_existing_away_motion_and_fast_descent_do_not_get_unqualified_launch(self):
        for velocity in ((-20., 0.), (0., 150.)):
            controller = JoltArrivalController()
            for tick in (0, 6, 12):
                self.assertFalse(controller.step(observation(tick, velocity=velocity), self.goal).action.primary)

    def test_airborne_command_does_not_retain_ground_traction_or_rebound(self):
        controller = JoltArrivalController()
        controller.step(observation(), self.goal)
        controller.step(observation(6), self.goal)
        for tick in (12, 18, 24):
            frame = controller.step(observation(tick, supported=False, velocity=(45., 20.)), self.goal)
            self.assertEqual(frame.action.move, (0.0, 0.0))
            self.assertFalse(frame.action.primary)
            self.assertFalse(frame.action.secondary)
            self.assertEqual(frame.phase, "airborne_coast")

    def test_support_resources_and_ceiling_are_needed_for_launch(self):
        for changes in ({"supported": False}, {"supported": None}, {"energy": .01},
                        {"energy": None}, {"cooldown": None}, {"cooldown": .5},
                        {"ceiling_distance": 5.0}):
            with self.subTest(changes=changes):
                controller = JoltArrivalController()
                for tick in (0, 6, 12, 18):
                    self.assertFalse(controller.step(observation(tick, **changes), self.goal).action.primary)

    def test_one_tick_support_loss_disarms_a_ready_launch_immediately(self):
        for unavailable in (False, None):
            with self.subTest(supported=unavailable):
                controller = JoltArrivalController()
                # Acquire support before the initial anticipation time ends.
                for tick in range(6):
                    self.assertFalse(controller.step(observation(tick), self.goal).action.primary)
                lost = controller.step(observation(6, supported=unavailable), self.goal)
                self.assertFalse(lost.action.primary)
                for tick in (7, 8):
                    self.assertFalse(controller.step(observation(tick), self.goal).action.primary)
                self.assertTrue(controller.step(observation(9), self.goal).action.primary)

    def test_unsupported_tilted_and_fast_goal_observations_cannot_settle(self):
        for changes in ({"supported": False}, {"supported": None}, {"angle": .5},
                        {"velocity": (6.01, 0.)}):
            with self.subTest(changes=changes):
                controller = JoltArrivalController()
                for tick in (0, 6, 12, 18, 24):
                    frame = controller.step(observation(tick, position=self.goal.target, **changes), self.goal)
                    self.assertNotEqual(frame.status, "succeeded")

    def test_arrival_dwell_starts_at_first_sample_and_resets_on_interruption(self):
        controller = JoltArrivalController()
        controller.step(observation(), self.goal)
        for tick in (6, 12):
            self.assertEqual(controller.step(observation(tick, position=self.goal.target), self.goal).status, "running")
        self.assertEqual(controller.step(observation(18, position=self.goal.target, supported=False), self.goal).status, "running")
        for tick in (24, 30):
            self.assertEqual(controller.step(observation(tick, position=self.goal.target), self.goal).status, "running")
        settled = controller.step(observation(36, position=self.goal.target), self.goal)
        self.assertEqual(settled.status, "succeeded")
        self.assertEqual(settled.reason, "supported_arrival_estimate")
        self.assertEqual(settled.action.effort, 0.0)
        self.assertEqual(controller.step(observation(42), self.goal).status, "succeeded")

    def test_sparse_samples_do_not_backfill_arrival_dwell(self):
        controller = JoltArrivalController()
        for tick in (0, 12, 24):
            self.assertEqual(controller.step(observation(tick, position=self.goal.target), self.goal).status, "running")
        self.assertEqual(controller.step(observation(30, position=self.goal.target), self.goal).status, "running")
        self.assertEqual(controller.step(observation(36, position=self.goal.target), self.goal).status, "succeeded")

    def test_supported_near_goal_overspeed_requests_countertraction(self):
        controller = JoltArrivalController()
        near = replace(self.goal, target=(1., 0.))
        for tick in (0, 6):
            frame = controller.step(observation(tick, velocity=(20., 0.)), near)
        self.assertLess(frame.action.move[0], 0.)
        self.assertFalse(frame.action.primary)

    def test_cancel_retires_goal_and_fresh_goal_rearms_one_launch(self):
        controller = JoltArrivalController()
        controller.step(observation(), self.goal)
        controller.step(observation(6), self.goal)
        self.assertEqual(controller.cancel().action.effort, 0.0)
        self.assertEqual(controller.step(observation(12), self.goal).status, "canceled")
        fresh = replace(self.goal, id="hop-two")
        self.assertFalse(controller.step(observation(18), fresh).action.primary)
        self.assertTrue(controller.step(observation(24), fresh).action.primary)
        with self.assertRaises(ValueError):
            controller.step(observation(30), self.goal)

    def test_generation_change_cancels_and_rollback_never_rearms(self):
        controller = JoltArrivalController()
        controller.step(observation(), self.goal)
        controller.step(observation(6), self.goal)
        changed = controller.step(observation(12, control_generation=2), self.goal)
        self.assertEqual(changed.status, "canceled")
        self.assertEqual(changed.action.effort, 0.)
        with self.assertRaises(ValueError):
            controller.step(observation(18, control_generation=1), replace(self.goal, id="rolled-back"))
        self.assertEqual(controller.step(observation(18, control_generation=2), self.goal).status, "canceled")

    def test_stale_duplicate_goal_mutation_and_invalid_feedback_disarm(self):
        controller = JoltArrivalController()
        controller.step(observation(), self.goal)
        controller.step(observation(6), self.goal)
        for tick in (6, 0):
            with self.assertRaises(ValueError):
                controller.step(observation(tick), self.goal)
        self.assertEqual(controller.step(observation(12), self.goal).status, "canceled")
        fresh = replace(self.goal, id="new")
        controller.step(observation(18), fresh)
        with self.assertRaises(ValueError):
            controller.step(observation(24), replace(fresh, target=(99., 0.)))
        self.assertEqual(controller.step(observation(24), fresh).status, "canceled")
        controller.step(observation(30), replace(self.goal, id="third"))
        self.assertEqual(controller.step(observation(36, valid=False), replace(self.goal, id="third")).status, "canceled")

    def test_new_lifetime_resets_trace_and_previous_lifetime_stays_retired(self):
        controller = JoltArrivalController()
        first = [asdict(controller.step(observation(tick), self.goal)) for tick in (0, 6, 12)]
        second = [asdict(controller.step(observation(tick, lifetime="episode:two"), self.goal)) for tick in (0, 6, 12)]
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            controller.step(observation(18), self.goal)

    def test_deadline_is_exact_and_another_instance_has_independent_state(self):
        controller = JoltArrivalController()
        short = replace(self.goal, timeout_seconds=.3)
        for tick in (0, 6, 12):
            self.assertEqual(controller.step(observation(tick), short).status, "running")
        self.assertEqual(controller.step(observation(18), short).status, "timed_out")
        other = JoltArrivalController()
        other.step(observation(), self.goal)
        self.assertTrue(other.step(observation(6), self.goal).action.primary)

    def test_recovery_uses_shared_auxiliary_and_suppresses_launch(self):
        controller = JoltArrivalController()
        controller.step(observation(), self.goal)
        frame = controller.step(observation(6, angle=1.4, angular_velocity=5.), self.goal)
        self.assertEqual(frame.phase, "recovery")
        self.assertTrue(frame.action.auxiliary)
        self.assertFalse(frame.action.primary)
        self.assertLess(frame.action.trim, 0.)

    def test_declared_mapping_and_finite_bounded_outputs(self):
        with self.assertRaises(ValueError):
            JoltArrivalController("astra")
        controller = JoltArrivalController()
        with self.assertRaises(ValueError):
            controller.step(observation(), replace(self.goal, skill="recover_level"))
        for index in range(40):
            frame = controller.step(observation(index * 6, velocity=(80. * math.sin(index), 80. * math.cos(index)),
                                                angle=math.sin(index) * 2., supported=index % 3 != 0), self.goal)
            self.assertLessEqual(math.hypot(*frame.action.move), 1.)
            self.assertLessEqual(math.hypot(*frame.action.aim), 1. + 1e-12)
            self.assertLessEqual(abs(frame.action.trim), 1.)
            self.assertGreaterEqual(frame.action.effort, 0.)
            self.assertLessEqual(frame.action.effort, 1.)


if __name__ == "__main__":
    unittest.main()
