"""Behavioral controller/supervisor checks using explicit synthetic feedback.

These test software decisions and refusal boundaries, not physical recovery
rates, actual impulses or actor feel. The matched diagnostic owns those claims.
"""

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import unittest

from nca.builtins import builtin_registry
from nca.contracts import sha256_json
from nca.controllers.baseline import FeedbackObservation, SkillGoal
from nca.controllers.jolt_recovery import (CONTROLLER_ID, PRESERVED_SOURCE_SHA256,
                                          JoltRecoveryController)
from nca.controllers.recovery_supervisor import PhysicalRecoverySupervisor


def observation(tick=0, **changes):
    values = dict(tick=tick, lifetime="episode:one", actor="jev", position=(0., 0.),
                  velocity=(0., 0.), angle=0., angular_velocity=0., supported=True,
                  cooldown=0., energy=1., ground_distance=2., ceiling_distance=45.)
    values.update(changes)
    return FeedbackObservation(**values)


def goal(**changes):
    values = dict(id="hop-one", skill="directed_hop", target=(48., 0.), timeout_seconds=4.)
    values.update(changes)
    return SkillGoal(**values)


def first_launch(controller, target=None):
    target = target or goal()
    controller.step(observation(), target)
    return controller.step(observation(6), target)


def supported_stall(controller, target=None, *, position=(0., 0.), blocked=True):
    target = target or goal()
    first_launch(controller, target)
    for tick in range(12, 49, 6):
        frame = controller.step(observation(tick, position=position, blocked=blocked,
                                            cooldown=max(0., (48 - tick) / 60)), target)
    return frame


class JoltRecoveryTests(unittest.TestCase):
    def test_distinct_serializable_contract_and_preserved_sources(self):
        controller = JoltRecoveryController()
        self.assertEqual(controller.controller_id, CONTROLLER_ID)
        spec = controller.specification()
        self.assertEqual(spec, json.loads(json.dumps(spec, allow_nan=False)))
        self.assertEqual(controller.specification_sha256, sha256_json(spec))
        root = Path(__file__).resolve().parents[1] / "src/nca/controllers"
        for name, digest in PRESERVED_SOURCE_SHA256.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
        frame = controller.step(observation(), goal())
        self.assertEqual(frame.tuning_hash, controller.specification_sha256)
        self.assertEqual(frame.controller_id, CONTROLLER_ID)
        registered = builtin_registry().create("policy", "authored-jolt-recovery", 1)
        self.assertEqual(registered.controller_id, CONTROLLER_ID)

    def test_initial_anticipation_then_bounded_steep_short_hop(self):
        launches = {}
        for distance in (-60., -24., 24., 60.):
            with self.subTest(distance=distance):
                controller = JoltRecoveryController()
                target = goal(target=(distance, 0.))
                anticipation = controller.step(observation(), target)
                self.assertEqual(anticipation.phase, "anticipate")
                self.assertEqual(anticipation.action.move, (0., 0.))
                self.assertFalse(anticipation.action.primary)
                frame = controller.step(observation(6), target)
                self.assertTrue(frame.action.primary)
                self.assertEqual(frame.phase, "launch")
                self.assertGreater(frame.action.aim[0] * distance, 0.)
                self.assertGreaterEqual(-frame.action.aim[1] * frame.action.effort * 86., 20.)
                self.assertGreaterEqual(frame.action.effort, .55)
                self.assertLessEqual(frame.action.effort, target.pace)
                self.assertFalse(frame.action.secondary)
                self.assertFalse(frame.action.auxiliary)
                self.assertEqual(controller.diagnostic_state()["pulse_reason"], "initial_supported_launch")
                launches[distance] = frame
        self.assertLess(abs(launches[24.].action.aim[0]), abs(launches[60.].action.aim[0]))

    def test_supported_progress_watchdog_completes_its_own_unload_and_rehop(self):
        controller = JoltRecoveryController()
        anticipation = supported_stall(controller)
        self.assertEqual(anticipation.phase, "contact_recovery_anticipate")
        self.assertEqual(anticipation.action.move, (0., 0.))
        self.assertFalse(anticipation.action.primary)
        launch = controller.step(observation(54, blocked=True), goal())
        self.assertTrue(launch.action.primary)
        self.assertEqual(launch.phase, "contact_recovery_launch")
        self.assertEqual(launch.reason, "supported_blocked_progress")
        state = controller.diagnostic_state()
        self.assertEqual(state["pulse_count"], 2)
        self.assertEqual(state["recovery_epoch"], 1)
        self.assertFalse(state["supervisor_intervention"])
        self.assertIsNone(state["supervisor_reason"])
        # New physical stalls and replenished resource feedback cannot acquire
        # a third pulse in this fixed two-pulse contract.
        for tick in range(60, 235, 6):
            frame = controller.step(observation(tick, blocked=True, cooldown=.2 if tick == 60 else 0.), goal())
            self.assertFalse(frame.action.primary)

    def test_watchdog_does_not_depend_on_host_blocked_time_heuristic(self):
        controller = JoltRecoveryController()
        frame = supported_stall(controller, blocked=False)
        self.assertEqual(frame.reason, "supported_no_progress")
        self.assertTrue(controller.step(observation(54), goal()).action.primary)

    def test_near_goal_stall_requests_vertical_unweight_without_full_forward_launch(self):
        controller = JoltRecoveryController()
        supported_stall(controller, position=(43.3, 0.))
        frame = controller.step(observation(54, position=(43.3, 0.), blocked=True), goal())
        self.assertTrue(frame.action.primary)
        self.assertLess(abs(frame.action.aim[0]), .1)
        self.assertLess(frame.action.aim[1], -.99)

    def test_fresh_motion_support_loss_unknown_contact_or_gap_cancels_pending_rehop(self):
        for changes in ({"position": (2., 0.)}, {"supported": False}, {"supported": None}):
            with self.subTest(changes=changes):
                controller = JoltRecoveryController()
                supported_stall(controller)
                frame = controller.step(observation(54, blocked=True, **changes), goal())
                self.assertFalse(frame.action.primary)
                self.assertNotEqual(frame.phase, "contact_recovery_launch")
        controller = JoltRecoveryController()
        supported_stall(controller)
        frame = controller.step(observation(60, blocked=True), goal())
        self.assertFalse(frame.action.primary)

    def test_blocked_flag_with_fresh_progress_does_not_invent_stall(self):
        controller = JoltRecoveryController()
        target = goal(target=(120., 0.))
        self.assertTrue(first_launch(controller, target).action.primary)
        for tick in range(12, 150, 6):
            frame = controller.step(observation(tick, position=((tick - 6) * .25, 0.),
                                                velocity=(15., 0.), blocked=True,
                                                cooldown=.2 if tick == 12 else 0.), target)
            self.assertFalse(frame.action.primary)
            self.assertNotIn("recovery", frame.phase)

    def test_cooldown_timer_alone_and_unacknowledged_first_request_never_rehop(self):
        controller = JoltRecoveryController()
        first_launch(controller)
        for tick in range(12, 235, 6):
            frame = controller.step(observation(tick, blocked=True), goal())
            self.assertFalse(frame.action.primary)
        self.assertEqual(controller.diagnostic_state()["pulse_count"], 1)
        self.assertFalse(controller.diagnostic_state()["pulse_acknowledged_by_cooldown"])

    def test_known_unsupported_edge_gets_one_fresh_physical_recovery_pulse(self):
        controller = JoltRecoveryController()
        first_launch(controller)
        for tick in range(12, 49, 6):
            frame = controller.step(observation(tick, supported=False,
                                                cooldown=max(0., (48 - tick) / 60)), goal())
        self.assertEqual(frame.phase, "support_loss_anticipate")
        launch = controller.step(observation(54, supported=False), goal())
        self.assertEqual(launch.phase, "support_loss_launch")
        self.assertEqual(launch.reason, "unsupported_edge_stall")
        self.assertTrue(launch.action.primary)
        self.assertEqual(launch.action.move, (0., 0.))
        self.assertFalse(controller.diagnostic_state()["supervisor_intervention"])

    def test_unknown_floor_or_normal_ballistic_progress_does_not_acquire_air_hop(self):
        for unknown in (None, 30.):
            controller = JoltRecoveryController()
            first_launch(controller)
            for tick in range(12, 120, 6):
                frame = controller.step(observation(tick, supported=False, ground_distance=unknown,
                                                    cooldown=.3 if tick == 12 else 0.), goal())
                self.assertFalse(frame.action.primary)
        controller = JoltRecoveryController()
        first_launch(controller)
        for tick in range(12, 120, 6):
            frame = controller.step(observation(tick, supported=False, position=(tick * .3, -tick * .1),
                                                velocity=(18., -6.), cooldown=.3 if tick == 12 else 0.), goal())
            self.assertFalse(frame.action.primary)

    def test_observed_unexpected_support_loss_can_request_bounded_descent_correction(self):
        controller = JoltRecoveryController()
        target = goal(target=(80., 0.))
        controller.step(observation(cooldown=.4), target)
        for tick, y in ((6, 3.), (12, 5.), (18, 7.)):
            frame = controller.step(observation(tick, supported=False, position=(0., y),
                                                velocity=(0., 16.), cooldown=.2 if tick < 18 else 0.), target)
        self.assertEqual(frame.reason, "unexpected_support_loss_descent")
        self.assertEqual(frame.phase, "support_loss_anticipate")
        frame = controller.step(observation(24, supported=False, position=(0., 9.), velocity=(0., 20.)), target)
        self.assertTrue(frame.action.primary)
        self.assertGreaterEqual(-frame.action.aim[1] * frame.action.effort * 86. - 20., 20. - 1e-9)

    def test_air_pulse_is_refused_when_too_fast_to_arrest(self):
        controller = JoltRecoveryController()
        target = goal(target=(80., 0.))
        controller.step(observation(cooldown=.1), target)
        for tick in (6, 12, 18, 24, 30):
            frame = controller.step(observation(tick, supported=False, position=(0., tick), velocity=(0., 90.)), target)
            self.assertFalse(frame.action.primary)

    def test_landing_uses_fresh_support_and_no_unconditional_bounce(self):
        controller = JoltRecoveryController()
        first_launch(controller)
        controller.step(observation(12, supported=False, position=(3., -4.), velocity=(20., -10.), cooldown=.4), goal())
        landing = controller.step(observation(18, position=(10., 0.), velocity=(15., 8.), cooldown=.3), goal())
        self.assertEqual(landing.phase, "landing")
        self.assertFalse(landing.action.primary)
        self.assertFalse(landing.action.secondary)
        self.assertFalse(landing.action.auxiliary)

    def test_launch_room_resource_ceiling_and_deadline_limits(self):
        cases = ({"velocity": (45., 0.)}, {"velocity": (-20., 0.)}, {"energy": None},
                 {"energy": .01}, {"cooldown": None}, {"cooldown": .2},
                 {"ceiling_distance": None}, {"ceiling_distance": 6.}, {"supported": None})
        for changes in cases:
            with self.subTest(changes=changes):
                controller = JoltRecoveryController()
                near = goal(target=(8., 0.))
                for tick in (0, 6, 12):
                    self.assertFalse(controller.step(observation(tick, **changes), near).action.primary)
        controller = JoltRecoveryController()
        for tick in (0, 6):
            self.assertFalse(controller.step(observation(tick), goal(timeout_seconds=.2)).action.primary)

    def test_supported_quiet_arrival_requires_fresh_observed_dwell(self):
        controller = JoltRecoveryController()
        target = goal()
        for tick in (0, 6):
            self.assertEqual(controller.step(observation(tick, position=target.target), target).status, "running")
        self.assertEqual(controller.step(observation(12, position=target.target, supported=None), target).status, "running")
        for tick in (18, 24):
            self.assertEqual(controller.step(observation(tick, position=target.target), target).status, "running")
        done = controller.step(observation(30, position=target.target), target)
        self.assertEqual(done.status, "succeeded")
        self.assertEqual(done.action.effort, 0.)
        self.assertFalse(controller.diagnostic_state()["supervisor_intervention"])
        self.assertEqual(controller.diagnostic_state()["action_lease_ticks"], 0)
        self.assertEqual(controller.step(observation(36), target).status, "succeeded")

    def test_sparse_or_unsupported_fast_tilted_samples_never_prove_quiet_arrival(self):
        for changes in ({"supported": False}, {"supported": None}, {"velocity": (6.01, 0.)}, {"angle": .33}):
            controller = JoltRecoveryController()
            for tick in (0, 6, 12, 18):
                self.assertNotEqual(controller.step(observation(tick, position=(48., 0.), **changes), goal()).status, "succeeded")
        controller = JoltRecoveryController()
        for tick in (0, 12, 24, 36):
            self.assertNotEqual(controller.step(observation(tick, position=(48., 0.)), goal()).status, "succeeded")

    def test_duplicate_goal_mutation_and_invalid_feedback_retire_motion(self):
        controller = JoltRecoveryController()
        supported_stall(controller)
        with self.assertRaises(ValueError):
            controller.step(observation(48), goal())
        self.assertEqual(controller.diagnostic_state()["action_lease_ticks"], 0)
        self.assertIsNone(controller.diagnostic_state()["pulse_reason"])
        canceled = controller.step(observation(54), goal())
        self.assertEqual(canceled.status, "canceled")
        self.assertEqual(canceled.action.effort, 0.)
        fresh = goal(id="fresh")
        controller.step(observation(60), fresh)
        with self.assertRaises(ValueError):
            controller.step(observation(66), replace(fresh, target=(99., 0.)))
        self.assertEqual(controller.step(observation(66), fresh).status, "canceled")
        other = goal(id="third")
        controller.step(observation(72), other)
        self.assertEqual(controller.step(observation(78, valid=False), other).status, "canceled")

    def test_generation_cancellation_replacement_and_lifetime_reset(self):
        controller = JoltRecoveryController()
        original = [asdict(controller.step(observation(tick), goal())) for tick in (0, 6)]
        self.assertEqual(controller.step(observation(12, control_generation=1), goal()).status, "canceled")
        with self.assertRaises(ValueError):
            controller.step(observation(18), goal(id="rollback"))
        fresh = goal(id="fresh")
        self.assertFalse(controller.step(observation(18, control_generation=1), fresh).action.primary)
        self.assertTrue(controller.step(observation(24, control_generation=1), fresh).action.primary)
        with self.assertRaises(ValueError):
            controller.step(observation(30, control_generation=1), goal())
        replay = [asdict(controller.step(observation(tick, lifetime="episode:two"), goal())) for tick in (0, 6)]
        self.assertEqual(original, replay)
        with self.assertRaises(ValueError):
            controller.step(observation(12), goal())

    def test_emergency_receives_cooldown_ack_without_ordinary_motion(self):
        controller = JoltRecoveryController()
        first_launch(controller)
        frame = controller.step(observation(12, cooldown=.5, angle=1.5), goal())
        self.assertTrue(frame.action.auxiliary)
        self.assertEqual(frame.action.move, (0., 0.))
        self.assertFalse(frame.action.primary)
        self.assertTrue(controller.diagnostic_state()["pulse_acknowledged_by_cooldown"])
        self.assertTrue(controller.diagnostic_state()["supervisor_intervention"])

    def test_current_clock_expiry_and_short_remaining_lease(self):
        controller = JoltRecoveryController()
        frame = controller.step(observation(), goal(timeout_seconds=.1), current_tick=5)
        self.assertEqual(frame.status, "running")
        self.assertEqual(controller.diagnostic_state()["action_lease_ticks"], 6)
        timed_out = controller.step(observation(6), goal(timeout_seconds=.1))
        self.assertEqual(timed_out.status, "timed_out")
        self.assertEqual(timed_out.action.effort, 0.)
        other = JoltRecoveryController()
        expired = other.step(observation(), goal(), current_tick=12)
        self.assertEqual(expired.status, "canceled")
        self.assertEqual(expired.reason, "expired_observation")

    def test_telemetry_copy_and_finite_bounded_outputs(self):
        controller = JoltRecoveryController()
        for tick in range(0, 240, 6):
            frame = controller.step(observation(tick, position=(math.sin(tick), math.cos(tick)),
                                                angle=.8 * math.sin(tick), velocity=(30. * math.sin(tick), 20. * math.cos(tick))), goal())
            self.assertLessEqual(math.hypot(*frame.action.move), 1.)
            self.assertLessEqual(math.hypot(*frame.action.aim), 1. + 1e-12)
            self.assertLessEqual(abs(frame.action.trim), 1.)
            self.assertLessEqual(frame.action.effort, 1.)
            self.assertGreaterEqual(frame.action.effort, 0.)
            json.dumps(controller.diagnostic_state(), allow_nan=False)
        state = controller.diagnostic_state()
        state["pulse_count"] = 999
        self.assertNotEqual(controller.diagnostic_state()["pulse_count"], 999)


class PhysicalRecoverySupervisorTests(unittest.TestCase):
    def test_ordinary_policy_retains_posture_and_movement_authority(self):
        supervisor = PhysicalRecoverySupervisor()
        for tick in (0, 6, 12):
            result = supervisor.admit(observation(tick, angle=1.5), goal())
            self.assertIsNone(result.override)
            self.assertFalse(result.intervention)
        result = supervisor.admit(observation(18, angle=1.5), goal(), emergency="unstable_attitude")
        self.assertTrue(result.intervention)
        self.assertTrue(result.override.auxiliary)
        self.assertFalse(result.override.primary)
        self.assertEqual(result.override.move, (0., 0.))

    def test_unwarranted_or_unknown_emergency_cannot_become_motion_helper(self):
        supervisor = PhysicalRecoverySupervisor()
        result = supervisor.admit(observation(), goal(), emergency="unstable_attitude")
        self.assertIsNone(result.override)
        with self.assertRaises(ValueError):
            supervisor.admit(observation(6), goal(), emergency="find_path")
        self.assertEqual(supervisor.admit(observation(6), goal()).status, "canceled")

    def test_emergency_is_bounded_without_repeated_request(self):
        supervisor = PhysicalRecoverySupervisor()
        supervisor.admit(observation(angle=1.5), goal(), emergency="unstable_attitude")
        for tick in range(6, 48, 6):
            result = supervisor.admit(observation(tick, angle=1.5), goal())
            self.assertTrue(result.override.auxiliary)
        expired = supervisor.admit(observation(48, angle=1.5), goal())
        self.assertEqual(expired.status, "canceled")
        self.assertEqual(expired.reason, "emergency_recovery_deadline")
        self.assertEqual(expired.override.effort, 0.)

    def test_emergency_quiet_dwell_is_observed_and_sparse_samples_restart_it(self):
        supervisor = PhysicalRecoverySupervisor()
        supervisor.admit(observation(angle=1.5), goal(), emergency="unstable_attitude")
        for tick in (6, 18, 24):
            self.assertTrue(supervisor.admit(observation(tick), goal()).intervention)
        recovered = supervisor.admit(observation(30), goal())
        self.assertIsNone(recovered.override)
        self.assertFalse(recovered.intervention)

    def test_emergency_lease_cannot_outlast_original_emergency_deadline(self):
        supervisor = PhysicalRecoverySupervisor()
        supervisor.admit(observation(angle=1.5), goal(), emergency="unstable_attitude")
        late = supervisor.admit(observation(42, angle=1.5), goal())
        self.assertEqual(late.action_lease_ticks, 6)
        aged = supervisor.admit(observation(43, angle=1.5), goal(), current_tick=47)
        self.assertEqual(aged.action_lease_ticks, 5)
        self.assertEqual(43 + aged.action_lease_ticks - 47, 1)
        expired = supervisor.admit(observation(44, angle=1.5), goal(), current_tick=48)
        self.assertEqual(expired.status, "canceled")
        self.assertEqual(expired.action_lease_ticks, 0)

    def test_source_anchored_lease_is_not_renewed_by_dispatch_clock(self):
        supervisor = PhysicalRecoverySupervisor()
        result = supervisor.admit(observation(), goal(), current_tick=10)
        self.assertEqual(result.action_lease_ticks, 12)
        self.assertEqual(0 + result.action_lease_ticks - 10, 2)
        with self.assertRaises(ValueError):
            supervisor.admit(observation(6), goal(), current_tick=9)
        self.assertEqual(supervisor.admit(observation(12), goal()).status, "canceled")


if __name__ == "__main__":
    unittest.main()
