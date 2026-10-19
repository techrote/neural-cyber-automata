"""Behavioral and lifecycle tests; synthetic observations are not physics proof."""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import unittest

from nca.contracts import WireAction
from nca.controllers import (AuthoredController, BaselineTuning, FeedbackObservation,
                             SkillGoal, ContactLatch, RequestShaper, ShapeTuning,
                             soft_stop_request)
from nca.evaluation.feel import (FeelFixture, FixtureEvent, MotionSample,
                                 fixture_suite, measure_response, run_fixture)


def observation(tick=0, actor="jev", **changes):
    base = FeedbackObservation(tick, "episode:1", actor, (0.0, 0.0), (0.0, 0.0),
                               0.0, 0.0, True, cooldown=0.0, energy=1.0,
                               ground_distance=0.0, ceiling_distance=100.0)
    return replace(base, **changes)


def host_view(tick=0, actor="jev", **changes):
    view = {"version": 2, "representation": "articulated-rapier", "run_id": "fixture-run",
            "epoch": 1, "actor_id": 0, "profile_version": 1, "tick": tick,
            "observation_id": f"fixture-run:1:0:0:{tick}", "model": actor,
            "position": [0.0, 0.0], "velocity": [0.0, 0.0], "rotation": 0.0,
            "angular_velocity": 0.0, "grounded": True, "energy": 1.0,
            "cooldown": 0.0, "control_generation": 0}
    view.update(changes)
    return view


class RequestShapingTests(unittest.TestCase):
    def test_hysteresis_does_not_chatter_around_entry_threshold(self):
        shaper = RequestShaper(ShapeTuning(.03, .08, 100.0, 100.0, 0.0))
        self.assertEqual(shaper.step(.06, .1), 0.0)
        self.assertAlmostEqual(shaper.step(.1, .1), .1)
        self.assertAlmostEqual(shaper.step(.06, .1), .06)
        self.assertEqual(shaper.step(.02, .1), 0.0)
        self.assertEqual(shaper.step(.06, .1), 0.0)

    def test_slew_depends_on_elapsed_time_not_number_of_calls(self):
        tuning = ShapeTuning(0.0, 0.0, 2.0, 4.0, 0.0)
        coarse, fine = RequestShaper(tuning), RequestShaper(tuning)
        self.assertAlmostEqual(coarse.step(1.0, .1), .2)
        for _ in range(10):
            fine.step(1.0, .01)
        self.assertAlmostEqual(coarse.output, fine.output)
        self.assertAlmostEqual(coarse.step(-1.0, .1), -.2)

    def test_cancel_bypasses_old_request_and_preserves_play_until_reset(self):
        shaper = RequestShaper(ShapeTuning(.02, .04, 8.0, 10.0, .1))
        for _ in range(10):
            shaper.step(1.0, .1)
        loaded_play = shaper.play
        self.assertGreater(loaded_play, .8)
        self.assertEqual(shaper.cancel(), 0.0)
        self.assertEqual(shaper.play, loaded_play)
        self.assertEqual(shaper.step(0.0, .1), 0.0)
        shaper.reset()
        self.assertEqual(shaper.play, 0.0)

    def test_backlash_is_bounded_repeatable_and_does_not_accumulate(self):
        shaper = RequestShaper(ShapeTuning(0.0, 0.0, 100.0, 100.0, .1))
        self.assertAlmostEqual(shaper.step(.8, .1), .7)
        self.assertAlmostEqual(shaper.step(.7, .1), .7)
        self.assertAlmostEqual(shaper.step(.5, .1), .6)
        for _ in range(1000):
            value = shaper.step(-.8, .1)
            self.assertLessEqual(abs(value), 1.0)
        self.assertAlmostEqual(value, -.7)

    def test_unknown_and_lost_support_withhold_action_without_coyote_renewal(self):
        latch = ContactLatch(.08)
        self.assertFalse(latch.step(True, .04))
        self.assertTrue(latch.step(True, .04))
        self.assertFalse(latch.step(None, .01))
        self.assertFalse(latch.step(True, .04))
        self.assertFalse(latch.step(False, .01))
        self.assertFalse(latch.step(True, .04))

    def test_soft_stop_attenuates_outward_but_allows_return(self):
        self.assertEqual(soft_stop_request(1.0, 0.0, 0.0, -1.0, 1.0, .2), 1.0)
        self.assertAlmostEqual(soft_stop_request(1.0, .9, 0.0, -1.0, 1.0, .2), .5)
        self.assertEqual(soft_stop_request(1.0, 1.0, .2, -1.0, 1.0, .2), 0.0)
        self.assertEqual(soft_stop_request(-1.0, 1.0, .2, -1.0, 1.0, .2), -1.0)
        damped = soft_stop_request(1.0, .9, 2.0, -1.0, 1.0, .2)
        self.assertLess(damped, .5)
        self.assertGreaterEqual(damped, 0.0)

    def test_nonfinite_and_invalid_time_never_enter_controller_state(self):
        shaper = RequestShaper()
        for value in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                shaper.step(value, .1)
        for dt in (0.0, -1.0, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                shaper.step(.5, dt)
        self.assertEqual(shaper.output, 0.0)
        with self.assertRaises(ValueError):
            soft_stop_request(1.0, 0.0, 0.0, 1.0, -1.0, .1)


class AuthoredControllerTests(unittest.TestCase):
    def setUp(self):
        self.hop = SkillGoal("hop-1", "directed_hop", (48.0, 0.0))

    def test_jolt_issues_one_pulse_then_releases_during_flight(self):
        controller = AuthoredController("jev")
        self.assertFalse(controller.step(observation(), self.hop).action.primary)
        launch = controller.step(observation(6), self.hop)
        self.assertTrue(launch.action.primary)
        self.assertEqual(launch.phase, "launch")
        self.assertGreater(launch.action.aim[0], .5)
        self.assertLess(launch.action.aim[1], -.4)
        for tick in range(12, 60, 6):
            flight = controller.step(observation(tick, supported=False, ground_distance=20.0,
                                                 velocity=(12.0, -20.0)), self.hop)
            self.assertFalse(flight.action.primary)
            self.assertFalse(flight.action.auxiliary)

    def test_jolt_no_burst_on_support_loss_stale_sensor_or_low_ceiling(self):
        cases = ({"supported": False}, {"supported": None}, {"energy": .01},
                 {"energy": None}, {"cooldown": None}, {"cooldown": .5},
                 {"ceiling_distance": 5.0})
        for changed in cases:
            with self.subTest(changed=changed):
                controller = AuthoredController("jev")
                for tick in range(0, 120, 6):
                    frame = controller.step(observation(tick, **changed), self.hop)
                    self.assertFalse(frame.action.primary)

    def test_jolt_cooldown_and_pulse_edges_are_not_continuous_charge(self):
        controller = AuthoredController("jev")
        pulse_ticks = []
        last_pressed = False
        for tick in range(0, 240, 6):
            frame = controller.step(observation(tick), self.hop)
            if frame.action.primary:
                self.assertFalse(last_pressed)
                pulse_ticks.append(tick)
            last_pressed = frame.action.primary
        self.assertGreaterEqual(len(pulse_ticks), 3)
        self.assertTrue(all(b - a >= 51 for a, b in zip(pulse_ticks, pulse_ticks[1:])))

    def test_jolt_recovery_suppresses_launch_and_uses_shared_auxiliary(self):
        controller = AuthoredController("jev")
        controller.step(observation(), self.hop)
        frame = controller.step(observation(6, angle=1.4, angular_velocity=5.0), self.hop)
        self.assertEqual(frame.phase, "recovery")
        self.assertTrue(frame.action.auxiliary)
        self.assertFalse(frame.action.primary)
        self.assertLess(frame.action.trim, 0.0)
        frame = controller.step(observation(12, angle=.6, angular_velocity=1.0), self.hop)
        self.assertTrue(frame.action.auxiliary)
        # Crossing the entry threshold back is insufficient; require settled dwell.
        self.assertTrue(controller.step(observation(18), self.hop).action.auxiliary)
        self.assertFalse(controller.step(observation(24), self.hop).action.auxiliary)

    def test_rebound_requires_visible_contact_region_and_descent(self):
        controller = AuthoredController("jev")
        self.assertFalse(controller.step(observation(0, supported=False,
                         velocity=(8.0, 20.0), ground_distance=30.0), self.hop).action.secondary)
        self.assertTrue(controller.step(observation(6, supported=False,
                        velocity=(8.0, 20.0), ground_distance=8.0), self.hop).action.secondary)
        self.assertFalse(controller.step(observation(12, supported=False,
                         velocity=(8.0, -3.0), ground_distance=8.0), self.hop).action.secondary)

    def test_vector_brakes_reversal_and_never_toggles_auxiliary_mode(self):
        controller = AuthoredController("astra")
        goal = SkillGoal("vector-1", "burst_traverse", (-50.0, 0.0))
        for tick in range(0, 60, 6):
            frame = controller.step(observation(tick, "astra", velocity=(35.0, 0.0)), goal)
            self.assertEqual(frame.phase, "counterthrust")
            self.assertTrue(frame.action.secondary)
            self.assertFalse(frame.action.primary)
            self.assertFalse(frame.action.auxiliary)
        frame = controller.step(observation(60, "astra"), goal)
        self.assertTrue(frame.action.primary)
        self.assertLess(frame.action.aim[0], -.9)

    def test_vector_attitude_recovery_uses_brake_and_trim_not_mode_toggle(self):
        controller = AuthoredController("astra")
        goal = SkillGoal("recover-1", "recover_level")
        frame = controller.step(observation(actor="astra", angle=1.4), goal)
        self.assertTrue(frame.action.secondary)
        self.assertFalse(frame.action.auxiliary)
        self.assertFalse(frame.action.primary)

    def test_matched_tunings_keep_character_choices_with_different_response(self):
        first_moves = {}
        for name in ("firm", "balanced", "compliant"):
            controller = AuthoredController("jev", BaselineTuning.named(name))
            first_moves[name] = controller.step(observation(), self.hop).action.move[0]
            for tick in range(6, 120, 6):
                frame = controller.step(observation(tick), self.hop)
                self.assertLessEqual(math.hypot(*frame.action.move), 1.0 + 1e-12)
                self.assertFalse(frame.action.tool)
        self.assertGreater(first_moves["firm"], first_moves["balanced"])
        self.assertGreater(first_moves["balanced"], first_moves["compliant"])

    def test_cancel_disarms_and_does_not_implicitly_resume_same_goal(self):
        controller = AuthoredController("jev")
        controller.step(observation(), self.hop)
        self.assertTrue(controller.step(observation(6), self.hop).action.primary)
        canceled = controller.cancel()
        self.assertEqual(canceled.status, "canceled")
        self.assertEqual(canceled.action.effort, 0.0)
        self.assertFalse(canceled.action.primary)
        for tick in range(12, 48, 6):
            frame = controller.step(observation(tick), self.hop)
            self.assertEqual(frame.status, "canceled")
            self.assertEqual(frame.action.move, (0.0, 0.0))
        replacement = replace(self.hop, id="hop-2")
        self.assertFalse(controller.step(observation(48), replacement).action.primary)
        self.assertTrue(controller.step(observation(54), replacement).action.primary)

    def test_generation_change_retires_old_goal_without_resetting_actor_motion(self):
        controller = AuthoredController("jev")
        controller.step(observation(), self.hop)
        source = observation(6, control_generation=1, velocity=(12.0, -5.0))
        frame = controller.step(source, self.hop)
        self.assertEqual(frame.reason, "owner_generation_changed")
        self.assertFalse(frame.action.primary)
        self.assertEqual(source.velocity, (12.0, -5.0))
        self.assertEqual(controller.step(observation(12, control_generation=1), self.hop).status, "canceled")

    def test_new_invocation_after_long_canceled_interval_uses_fresh_ramp(self):
        controller = AuthoredController("jev")
        controller.step(observation(), self.hop)
        controller.step(observation(6), self.hop)
        controller.cancel()
        frame = controller.step(observation(600, control_generation=1), replace(self.hop, id="new-hop"))
        self.assertEqual(frame.status, "running")
        self.assertFalse(frame.action.primary)
        self.assertLessEqual(frame.action.move[0], 8.0 / 60.0)

    def test_episode_reset_reproduces_trace_without_recurrent_leak(self):
        controller = AuthoredController("jev")
        first = [asdict(controller.step(observation(tick), self.hop)) for tick in range(0, 60, 6)]
        second = [asdict(controller.step(observation(tick, lifetime="episode:2"), self.hop))
                  for tick in range(0, 60, 6)]
        self.assertEqual(first, second)

    def test_stale_tick_and_mutated_goal_are_rejected(self):
        controller = AuthoredController("jev")
        controller.step(observation(), self.hop)
        with self.assertRaises(ValueError):
            controller.step(observation(), self.hop)
        with self.assertRaises(ValueError):
            controller.step(observation(6), replace(self.hop, target=(99.0, 0.0)))

    def test_deadline_terminates_at_exact_boundary_not_one_step_later(self):
        controller = AuthoredController("jev")
        goal = replace(self.hop, timeout_seconds=.3)
        for tick in (0, 6, 12):
            self.assertEqual(controller.step(observation(tick), goal).status, "running")
        timed = controller.step(observation(18), goal)
        self.assertEqual(timed.status, "timed_out")
        self.assertEqual(timed.action.move, (0.0, 0.0))

    def test_success_requires_observed_arrival_and_quiet_dwell(self):
        controller = AuthoredController("jev")
        controller.step(observation(), self.hop)
        for tick in (6, 12, 18):
            frame = controller.step(observation(tick, position=(48.0, 0.0), velocity=(15.0, 0.0)), self.hop)
            self.assertNotEqual(frame.status, "succeeded")
        controller.step(observation(24, position=(48.0, 0.0)), self.hop)
        frame = controller.step(observation(30, position=(48.0, 0.0)), self.hop)
        self.assertEqual(frame.status, "succeeded")
        self.assertEqual(frame.reason, "target_reached_estimate")
        self.assertFalse(frame.action.primary)

    def test_unavailable_observation_cancels_without_filling_motion_with_zero(self):
        controller = AuthoredController("jev")
        controller.step(observation(), self.hop)
        self.assertEqual(controller.step(observation(6, valid=False), self.hop).reason,
                         "unavailable_observation")
        with self.assertRaises(ValueError):
            observation(velocity=(float("nan"), 0.0))
        view = host_view()
        del view["velocity"]
        with self.assertRaises(ValueError):
            FeedbackObservation.from_host(view)

    def test_host_observation_whitelist_copies_motion_and_discards_privilege(self):
        view = host_view(evaluator={"hidden_target": [99, 99]}, nearest_hidden_target=[30, 30])
        copied = FeedbackObservation.from_host(view)
        view["position"][0] = 123.0
        self.assertEqual(copied.position, (0.0, 0.0))
        self.assertNotIn("evaluator", asdict(copied))
        self.assertNotIn("nearest_hidden_target", asdict(copied))

    def test_terminal_suspension_keeps_valid_final_motion_but_disarms_control(self):
        copied = FeedbackObservation.from_host(host_view(suspended=True))
        self.assertTrue(copied.valid)
        self.assertFalse(copied.controllable)
        controller = AuthoredController("jev")
        self.assertEqual(controller.step(copied, self.hop).status, "canceled")

    def test_future_profiles_and_mixed_rig_identity_require_explicit_mapping(self):
        for change in ({"profile_version": 2}, {"profile_version": True}, {"epoch": .5},
                       {"run_id": ""}, {"rig_id": "astra"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                FeedbackObservation.from_host(host_view(**change))

    def test_finite_wide_sensor_sweep_produces_only_bounded_wire_requests(self):
        for actor, skill in (("jev", "directed_hop"), ("astra", "burst_traverse")):
            controller = AuthoredController(actor)
            goal = SkillGoal("wide", skill, (100.0, -30.0), timeout_seconds=120.0)
            for i in range(600):
                frame = controller.step(observation(i * 6, actor, angle=math.sin(i * .07) * 3.1,
                         angular_velocity=math.cos(i * .03) * 20.0,
                         velocity=(math.sin(i * .09) * 200.0, math.cos(i * .09) * 200.0),
                         supported=i % 7 != 0), goal)
                self.assertLessEqual(math.hypot(*frame.action.move), 1.0 + 1e-12)
                self.assertLessEqual(abs(frame.action.trim), 1.0)
                self.assertLessEqual(math.hypot(*frame.action.aim), 1.0 + 1e-12)


class FeelMetricsTests(unittest.TestCase):
    @staticmethod
    def sample(tick, x=0.0, velocity=0.0, trim=0.0, phase="stance", valid=True):
        return MotionSample(tick, (x, 0.0), (velocity, 0.0), 0.0, 0.0,
                            WireAction(trim=trim), phase, True, valid=valid)

    def test_response_overshoot_settle_and_signed_progress_use_actual_motion(self):
        rows = [self.sample(0), self.sample(6, 1, 5), self.sample(12, 15, -10),
                self.sample(18, 10, 0), self.sample(24, 10, 0), self.sample(30, 10, 0)]
        result = measure_response(rows, origin=(0, 0), target=(10, 0), tolerance=.5)
        self.assertAlmostEqual(result.response_delay_seconds, .1)
        self.assertEqual(result.maximum_overshoot, 5.0)
        self.assertEqual(result.net_useful_progress, 10.0)
        self.assertAlmostEqual(result.settling_seconds, .3)

    def test_immobility_and_invalid_samples_cannot_appear_as_fast_success(self):
        result = measure_response([self.sample(0), self.sample(60)], origin=(0, 0), target=(10, 0))
        self.assertIsNone(result.response_delay_seconds)
        self.assertIsNone(result.settling_seconds)
        self.assertEqual(result.net_useful_progress, 0.0)
        invalid = measure_response([self.sample(0), self.sample(6, 10, 5, valid=False),
                                    self.sample(60, 10)], origin=(0, 0), target=(10, 0))
        self.assertEqual(invalid.invalid_samples, 1)
        self.assertIsNone(invalid.net_useful_progress)
        self.assertIsNone(invalid.response_delay_seconds)

    def test_chatter_excludes_intentional_launch_recoil_and_recovery(self):
        chatter = [self.sample(i * 6, trim=(-1.0 if i % 2 else 1.0) * .2) for i in range(5)]
        result = measure_response(chatter, origin=(0, 0), target=(10, 0))
        self.assertEqual(result.corrective_sign_changes, 4)
        intentional = [replace(row, phase="recoil") for row in chatter]
        self.assertEqual(measure_response(intentional, origin=(0, 0), target=(10, 0)).corrective_sign_changes, 0)

    def test_duplicate_and_reordered_samples_are_not_accepted(self):
        with self.assertRaises(ValueError):
            measure_response([self.sample(6), self.sample(0)], origin=(0, 0), target=(10, 0))
        with self.assertRaises(ValueError):
            measure_response([self.sample(0), self.sample(0)], origin=(0, 0), target=(10, 0))

    def test_fixture_ids_hashes_and_profile_variants_are_repeatable(self):
        for actor in ("jev", "astra"):
            fixtures = fixture_suite(actor)
            self.assertEqual(len(fixtures), 8)
            self.assertEqual(len({fixture.id for fixture in fixtures}), 8)
            self.assertEqual([fixture.digest for fixture in fixtures],
                             [fixture.digest for fixture in fixture_suite(actor)])
        self.assertNotEqual(BaselineTuning.named("firm").digest, BaselineTuning.named("compliant").digest)

    def test_fixture_runner_records_terminal_interval_and_refuses_noop_perturbation(self):
        class RecordedHost:
            evidence_lane = "recorded"
            def __init__(self):
                self.tick = 0
                self.steps = 0
            def observe(self):
                return host_view(self.tick)
            def step(self, request, ticks):
                self.steps += 1
                self.tick += 2  # The real interval ended before its requested six ticks.
                return host_view(self.tick, terminal_reason="task_failure")
            def cancel(self):
                pass
            def fixture_event(self, event):
                return {"accepted": False, "reason": "unsupported"}
        host = RecordedHost()
        result = run_fixture(host, fixture_suite()[0])
        self.assertEqual(host.steps, 1)
        self.assertEqual(result["samples"][-1]["tick"], 2)
        self.assertEqual(result["terminal_reason"], "task_failure")
        self.assertEqual(result["evidence_lane"], "recorded")
        self.assertEqual(result["owner_feel_judgment"], "not_recorded")
        with self.assertRaises(ValueError):
            run_fixture(RecordedHost(), fixture_suite()[-1])


if __name__ == "__main__":
    unittest.main()
