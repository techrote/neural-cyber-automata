"""Movement decisions on retained observations and synthetic boundary cases.

These fixtures qualify control/lifecycle arithmetic. They do not establish a
counterfactual native outcome, physical rescue, or renewed diagnostic budget.
"""

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import unittest

from nca.builtins import builtin_registry
from nca.contracts import WireAction, sha256_json
from nca.controllers.baseline import FeedbackObservation, SkillGoal
from nca.controllers.jolt_recovery_v2 import JoltRecoveryControllerV2
from nca.controllers.jolt_recovery_v3 import (
    CONTROLLER_ID, PRESERVED_ADVISORY_SOURCE_SHA256, JoltRecoveryControllerV3,
)
from nca.train.task import HopTask


# Native source observations from the immutable v2 right60-0 episode, artifact
# SHA256 d1efb48390a1ef3fc3a40dbd5ded36743749d2c613226875f3b07054df2ba2a9.
# This is a four-decision prefix; later original observations cannot establish
# what would physically follow a changed decision at18.
RETAINED_CONTACT = {
    0: ((176., 346.5), (0., 0.), 0., 0., False, 96.),
    6: ((175.990142822266, 346.485626220703), (.178647845983505, 1.37513220310211), -.000529368815477937, -.122038200497627, False, 2.01252147893408),
    12: ((176.024917602539, 346.491424560547), (.148333609104156, .846784234046936), -.00280909682624042, -.014207330532372, True, 1.99875742277646),
    18: ((176.047149658203, 346.488647460938), (.103528544306755, .962234914302826), -.00165262573864311, .0388154871761799, False, 2.0055731311598),
}


def observation(tick=0, **changes):
    values = dict(tick=tick, lifetime="synthetic:movement-v3", actor="jev", position=(0., 0.),
                  velocity=(0., 0.), angle=0., angular_velocity=0., supported=True,
                  cooldown=0., energy=1., ground_distance=2., ceiling_distance=45.)
    values.update(changes)
    return FeedbackObservation(**values)


def goal(**changes):
    values = dict(id="movement-goal", skill="directed_hop", target=(48., 0.), timeout_seconds=4.)
    values.update(changes)
    return SkillGoal(**values)


def retained_contact(tick):
    position, velocity, angle, omega, supported, ground = RETAINED_CONTACT[tick]
    return observation(tick, position=position, velocity=velocity, angle=angle,
                       angular_velocity=omega, supported=supported, ground_distance=ground)


def pending_initial(controller, target=None):
    target = target or goal(target=(236., 346.5))
    for tick in (0, 6, 12):
        frame = controller.step(retained_contact(tick), target)
    return frame


def two_pulses(controller, *, position=(0., 0.)):
    target = goal()
    controller.step(observation(), target)
    first = controller.step(observation(6), target)
    for tick in range(12, 49, 6):
        controller.step(observation(tick, position=position, cooldown=max(0., (48 - tick) / 60)), target)
    second = controller.step(observation(54, position=position), target)
    return target, first, second


class JoltMovementV3Tests(unittest.TestCase):
    def test_versioned_contract_preserves_prior_controller_and_supervisor(self):
        controller = JoltRecoveryControllerV3()
        self.assertEqual(controller.controller_id, CONTROLLER_ID)
        self.assertEqual(controller.specification()["schema"], "nca-jolt-recovery-reference/v3")
        self.assertEqual(controller.specification_sha256, sha256_json(controller.specification()))
        previous = JoltRecoveryControllerV2()
        self.assertEqual(controller.supervisor.specification(), previous.supervisor.specification())
        path = Path(__file__).resolve().parents[1] / "src/nca/controllers/jolt_recovery_v2.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), PRESERVED_ADVISORY_SOURCE_SHA256)
        self.assertIs(type(builtin_registry().create("policy", "authored-jolt-recovery", 3)),
                      JoltRecoveryControllerV3)

    def test_retained_contact_change_finishes_original_anticipation(self):
        old, new = JoltRecoveryControllerV2(), JoltRecoveryControllerV3()
        target = goal(target=(236., 346.5))
        for tick in (0, 6, 12):
            self.assertEqual(old.step(retained_contact(tick), target).action,
                             new.step(retained_contact(tick), target).action)
        self.assertFalse(old.step(retained_contact(18), target).action.primary)
        frame = new.step(retained_contact(18), target)
        self.assertTrue(frame.action.primary)
        self.assertEqual(frame.reason, "initial_contact_loss_launch")
        self.assertEqual(frame.status, "running")
        state = new.diagnostic_state()
        self.assertEqual(state["initial_contact_witness_tick"], 12)
        self.assertTrue(state["initial_contact_transition"])
        self.assertEqual(state["action_lease_ticks"], 12)
        self.assertEqual(state["pulse_count"], 1)

    def test_contact_transition_requires_current_local_witness(self):
        invalid = (
            {"supported": None}, {"ground_distance": None}, {"ground_distance": 16.01},
            {"position": (177., 346.5)}, {"velocity": (0., 6.01)},
            {"energy": .149}, {"cooldown": .01}, {"ceiling_distance": 6.},
        )
        for changed in invalid:
            with self.subTest(changed=changed):
                controller = JoltRecoveryControllerV3()
                pending_initial(controller)
                frame = controller.step(replace(retained_contact(18), **changed),
                                        goal(target=(236., 346.5)))
                self.assertFalse(frame.action.primary)
                self.assertIsNone(controller._pending_since)
                # A rejected witness does not grant the old launch at24.
                self.assertFalse(controller.step(replace(retained_contact(18), tick=24),
                                                  goal(target=(236., 346.5))).action.primary)

    def test_contact_transition_cannot_use_delayed_observation_or_gap(self):
        for tick, now in ((18, 19), (24, 24)):
            with self.subTest(tick=tick, now=now):
                controller = JoltRecoveryControllerV3()
                pending_initial(controller)
                frame = controller.step(replace(retained_contact(18), tick=tick),
                                        goal(target=(236., 346.5)), current_tick=now)
                self.assertFalse(frame.action.primary)
                self.assertFalse(controller.diagnostic_state()["initial_contact_transition"])

    def test_unknown_support_cancels_witness_and_reacquisition_anticipates_again(self):
        controller = JoltRecoveryControllerV3()
        target = goal(target=(236., 346.5))
        pending_initial(controller)
        controller.step(replace(retained_contact(18), supported=None), target)
        frame = controller.step(replace(retained_contact(12), tick=24), target)
        self.assertFalse(frame.action.primary)
        self.assertEqual(frame.phase, "anticipate")
        self.assertTrue(controller.step(replace(retained_contact(12), tick=30), target).action.primary)

    def test_cancel_generation_expiry_and_duplicate_retire_pending_launch(self):
        for boundary in ("cancel", "generation", "expiry", "duplicate"):
            with self.subTest(boundary=boundary):
                controller = JoltRecoveryControllerV3()
                target = goal(target=(236., 346.5), timeout_seconds=.3 if boundary == "expiry" else 4.)
                pending_initial(controller, target)
                if boundary == "cancel":
                    controller.cancel()
                if boundary == "duplicate":
                    with self.assertRaises(ValueError):
                        controller.step(retained_contact(12), target)
                else:
                    view = retained_contact(18)
                    if boundary == "generation":
                        view = replace(view, control_generation=1)
                    frame = controller.step(view, target)
                    self.assertFalse(frame.action.primary)
                self.assertEqual(controller.diagnostic_state()["action_lease_ticks"], 0)
                self.assertIsNone(controller._pending_since)

    def test_unchanged_supported_first_launch_for_both_directions(self):
        for distance in (-60., -24., 24., 60.):
            with self.subTest(distance=distance):
                old, new = JoltRecoveryControllerV2(), JoltRecoveryControllerV3()
                target = goal(target=(distance, 0.))
                for tick in (0, 6):
                    self.assertEqual(old.step(observation(tick), target).action,
                                     new.step(observation(tick), target).action)

    def test_second_pulse_is_shorter_and_has_more_horizontal_impulse(self):
        # Use the retained right48 second-pulse source as an arithmetic probe.
        view = observation(position=(215.644393920898, 348.463165283203),
                           velocity=(.376617312431335, .332374632358551), ground_distance=0.)
        target = goal(target=(224., 346.5))
        old, new = JoltRecoveryControllerV2(), JoltRecoveryControllerV3()
        dx, dy = target.target[0] - view.position[0], target.target[1] - view.position[1]
        old_pulse = old._pulse(view, target, dx, dy, recovery=True)
        new_pulse = new._pulse(view, target, dx, dy, recovery=True)
        self.assertIsNotNone(old_pulse)
        self.assertIsNotNone(new_pulse)
        old_aim, old_effort = old_pulse
        aim, effort = new_pulse
        self.assertGreater(aim[0] * effort, old_aim[0] * old_effort)
        self.assertLess(-aim[1] * effort, -old_aim[1] * old_effort)
        self.assertLess(effort, old_effort)
        self.assertGreaterEqual(-view.velocity[1] - aim[1] * effort * 86, 20.)

    def test_recovery_vector_compensates_signed_opposing_velocity(self):
        for direction in (-1., 1.):
            controller = JoltRecoveryControllerV3()
            view = observation(velocity=(-6. * direction, 0.))
            target = goal(target=(5. * direction, 0.))
            pulse = controller._pulse(view, target, 5. * direction, 0., recovery=True)
            self.assertIsNotNone(pulse)
            aim, effort = pulse
            result = direction * (view.velocity[0] + aim[0] * effort * 86.)
            self.assertGreater(result, 0.)
            flight = 2 * (-aim[1] * effort * 86.) / 156.
            self.assertLessEqual(result * (flight + .1) + result ** 2 / 200., 3. + 1e-10)

    def test_recovery_envelope_rechecks_room_time_ceiling_and_effort(self):
        controller = JoltRecoveryControllerV3()
        for changed, target in (
            ({"velocity": (-6.01, 0.)}, goal(target=(10., 0.))),
            ({"velocity": (50., 0.)}, goal(target=(10., 0.))),
            ({"velocity": (0., 70.)}, goal(target=(10., 0.))),
            ({"ceiling_distance": None}, goal(target=(10., 0.))),
            ({"ceiling_distance": 7.}, goal(target=(10., 0.))),
            ({}, goal(target=(10., 0.), timeout_seconds=.4)),
        ):
            with self.subTest(changed=changed, target=target):
                view = observation(**changed)
                self.assertIsNone(controller._pulse(view, target, target.target[0], 0., recovery=True))
        low_pace = goal(target=(10., 0.), pace=.1)
        pulse = controller._pulse(observation(), low_pace, 10., 0., recovery=True)
        self.assertIsNotNone(pulse)
        self.assertLessEqual(pulse[1], .45)

    def test_second_edge_still_needs_acknowledged_cooldown_and_new_stall(self):
        controller = JoltRecoveryControllerV3()
        target, first, second = two_pulses(controller)
        self.assertTrue(first.action.primary)
        self.assertTrue(second.action.primary)
        self.assertIn(second.reason, {"supported_no_progress", "supported_blocked_progress"})
        for tick in range(60, 235, 6):
            self.assertFalse(controller.step(observation(tick, cooldown=.2 if tick == 60 else 0.), target).action.primary)
        self.assertEqual(controller.diagnostic_state()["pulse_count"], 2)
        unacknowledged = JoltRecoveryControllerV3()
        target = goal()
        for tick in range(0, 91, 6):
            unacknowledged.step(observation(tick), target)
        self.assertEqual(unacknowledged.diagnostic_state()["pulse_count"], 1)

    def test_progress_or_unknown_support_does_not_grant_second_pulse(self):
        for unknown in (False, True):
            controller = JoltRecoveryControllerV3()
            target = goal()
            controller.step(observation(), target)
            controller.step(observation(6), target)
            for tick in range(12, 91, 6):
                view = observation(tick, position=((tick - 12) * .2, 0.),
                                   supported=None if unknown else True,
                                   velocity=(10., 0.), cooldown=.2 if tick < 48 else 0.)
                self.assertFalse(controller.step(view, target).action.primary)

    def test_post_cap_ground_correction_has_new_observed_window_and_bounded_duration(self):
        controller = JoltRecoveryControllerV3()
        target, _, second = two_pulses(controller, position=(42., 0.))
        self.assertTrue(second.action.primary)
        for tick in (60, 66, 72):
            frame = controller.step(observation(tick, position=(42., 0.), cooldown=.3), target)
            self.assertFalse(controller.diagnostic_state()["ground_correction"])
        frame = controller.step(observation(78, position=(42., 0.), cooldown=.3), target)
        self.assertEqual(frame.phase, "ground_correction")
        self.assertGreater(frame.action.move[0], .9)
        self.assertFalse(frame.action.primary)
        self.assertFalse(controller.diagnostic_state()["supervisor_intervention"])
        self.assertTrue(controller.step(observation(84, position=(42., 0.)), target).action.move[0] > .9)
        controller.step(observation(90, position=(42., 0.)), target)
        self.assertFalse(controller.diagnostic_state()["ground_correction"])
        for tick in (96, 102, 108):
            controller.step(observation(tick, position=(42., 0.)), target)
            self.assertFalse(controller.diagnostic_state()["ground_correction"])

    def test_ground_correction_ends_on_support_loss_motion_gap_or_arrival(self):
        for changed in ({"supported": False}, {"supported": None}, {"velocity": (7., 0.)},
                        {"position": (42.5, 0.)}, {"position": (46., 0.)}, {"tick": 90}):
            with self.subTest(changed=changed):
                controller = JoltRecoveryControllerV3()
                target, _, _ = two_pulses(controller, position=(42., 0.))
                for tick in (60, 66, 72, 78):
                    controller.step(observation(tick, position=(42., 0.), cooldown=.3), target)
                values = dict(tick=84, position=(42., 0.))
                values.update(changed)
                controller.step(observation(**values), target)
                self.assertFalse(controller.diagnostic_state()["ground_correction"])

    def test_inside_tolerance_brakes_fast_motion_before_advisory_dwell(self):
        controller = JoltRecoveryControllerV3()
        view = observation(position=(233.213088989258, 348.483032226562),
                           velocity=(7.66390943527222, 2.20659351348877))
        frame = controller.step(view, goal(target=(236., 346.5)))
        self.assertEqual(frame.phase, "settle_brake")
        self.assertLess(frame.action.move[0], 0.)
        self.assertFalse(controller.diagnostic_state()["arrival_advisory"])
        self.assertEqual(frame.status, "running")
        self.assertEqual(controller.diagnostic_state()["action_lease_ticks"], 12)

    def test_quiet_hold_is_immediate_advisory_and_disturbance_resumes_control(self):
        controller = JoltRecoveryControllerV3()
        target = goal(target=(0., 0.))
        for tick in (0, 6, 12):
            frame = controller.step(observation(tick, cooldown=.2), target)
            self.assertEqual(frame.action, WireAction(effort=0.))
            self.assertEqual(frame.status, "running")
            self.assertEqual(controller.diagnostic_state()["arrival_advisory"], tick == 12)
        disturbed = controller.step(observation(18, position=(-8., 0.), cooldown=.2), target)
        self.assertGreater(disturbed.action.move[0], 0.)
        self.assertFalse(controller.diagnostic_state()["arrival_advisory"])
        self.assertEqual(controller.cancel("caller_verified_outcome").status, "canceled")
        self.assertEqual(controller.diagnostic_state()["action_lease_ticks"], 0)

    def test_vertical_motion_unknown_support_or_gap_cannot_supply_quiet_advisory(self):
        for changed in ({"velocity": (0., 6.1)}, {"supported": None}, {"supported": False}, {"tick": 18}):
            controller = JoltRecoveryControllerV3()
            target = goal(target=(0., 0.))
            controller.step(observation(cooldown=.2), target)
            controller.step(observation(6, cooldown=.2), target)
            values = dict(tick=12, cooldown=.2)
            values.update(changed)
            frame = controller.step(observation(**values), target)
            self.assertEqual(frame.status, "running")
            self.assertFalse(controller.diagnostic_state()["arrival_advisory"])

    def test_tick_223_aliasing_cannot_retire_the_original_deadline(self):
        # The failed v1 native tick223 velocity is retained exactly. The other
        # native views below are labelled synthetic quiet fixtures, not evidence
        # that the historical host continued or arrived.
        target = goal(target=(0., 0.))
        controller = JoltRecoveryControllerV3()
        for tick in (0, 54, 108, 162):
            controller.step(observation(tick, position=(-20., 0.), supported=False,
                                        ground_distance=None, cooldown=.2), target)
        for tick in (216, 222, 228, 234):
            frame = controller.step(observation(tick, cooldown=.2), target)
        self.assertEqual(frame.status, "running")
        self.assertTrue(controller.diagnostic_state()["arrival_advisory"])
        self.assertEqual(controller.diagnostic_state()["action_lease_ticks"], 6)
        def native(tick):
            return {"version": 2, "representation": "articulated-rapier",
                    "run_id": "synthetic-alias-boundary", "epoch": 1, "actor_id": 0,
                    "profile_version": 1, "model": "jev", "observation_id": f"alias-{tick}",
                    "tick": tick, "position": (0., 0.), "velocity":
                    (7.00780153274536, 2.93841552734375) if tick == 223 else (0., 0.),
                    "grounded": True, "rotation": 0., "angular_velocity": 0.,
                    "valid": True, "cooldown": 0., "energy": 1., "ground_distance": 0.,
                    "ceiling_distance": 45.}
        task = HopTask(target, native(222))
        for start in (223, 229):
            task.consume(WireAction(effort=0.), [
                {"tick_before": tick - 1, "observation": native(tick)} for tick in range(start, start + 6)])
        self.assertEqual(task.quiet_ticks, 11)
        self.assertFalse(task.done)
        expired = controller.step(observation(240), target)
        self.assertEqual(expired.status, "timed_out")
        self.assertEqual(controller.diagnostic_state()["action_lease_ticks"], 0)


if __name__ == "__main__":
    unittest.main()
