"""Advisory-arrival lifecycle checks; no host work or inferred continuation.

The small kinematic excerpt is retained from the failed v1 diagnostic's
demo-floor-right-36-1 episode (archive SHA256
4c430f5bfdaa5e0f53299c1c456e06cb311df4a0b5c8a8a978fe72900db1d07e).
It demonstrates why three quiet policy samples do not prove twelve quiet
physical ticks. Other fixtures below are explicitly synthetic software checks.
"""

import hashlib
import math
from pathlib import Path
import unittest

from nca.builtins import builtin_registry
from nca.contracts import WireAction, sha256_json
from nca.controllers.baseline import FeedbackObservation, SkillGoal
from nca.controllers.jolt_recovery import JoltRecoveryController
from nca.controllers.jolt_recovery_v2 import (CONTROLLER_ID, PRESERVED_RECOVERY_SOURCE_SHA256,
                                             JoltRecoveryControllerV2)
from nca.train.task import HopTask


RETAINED = {
    216: ((208.419357299805, 348.448059082031), (7.07849168777466, -2.71284008026123), -0.0149239040911198, -0.771615862846375),
    222: ((209.106338500977, 348.438690185547), (5.95710325241089, 0.489370316267014), -0.0165028516203165, 0.146792933344841),
    223: ((209.223922729492, 348.473510742188), (7.00780153274536, 2.93841552734375), -0.00757909892126918, 0.779920876026154),
    224: ((209.264236450195, 348.490692138672), (0.921420454978943, -1.18545460700989), 0.00264623109251261, -0.420053154230118),
    225: ((209.329040527344, 348.485778808594), (3.71947050094604, 0.425480484962463), -0.00355331995524466, 0.185788050293922),
    226: ((209.396820068359, 348.478576660156), (2.71052384376526, -1.4449942111969), -0.00613188929855824, -0.322425752878189),
    227: ((209.458572387695, 348.459411621094), (3.0716700553894, -0.772887885570526), -0.0116554573178291, -0.048111479729414),
    228: ((209.526580810547, 348.472686767578), (3.19564032554626, 1.25758051872253), -0.00783545523881912, 0.579393684864044),
    229: ((209.577301025391, 348.470306396484), (1.94831192493439, -1.1305992603302), -0.00864280294626951, -0.191742107272148),
    230: ((209.609558105469, 348.451538085938), (1.43694949150085, -0.276159405708313), -0.013969442807138, -0.20423649251461),
    231: ((209.663024902344, 348.470275878906), (3.03556776046753, 1.84892868995667), -0.00830558035522699, 0.454341679811478),
    232: ((209.717849731445, 348.478759765625), (2.70530581474304, -0.841222763061523), -0.00596206774935126, -0.405402600765228),
    233: ((209.768218994141, 348.457916259766), (1.69830822944641, -1.0378851890564), -0.0121010281145573, -0.12311302870512),
    234: ((209.81916809082, 348.469482421875), (2.57931733131409, 1.25527346134186), -0.00875078048557043, 0.468330681324005),
}


def observation(tick=0, **changes):
    values = dict(tick=tick, lifetime="synthetic:advisory", actor="jev", position=(0., 0.),
                  velocity=(0., 0.), angle=0., angular_velocity=0., supported=True,
                  cooldown=.2, energy=1., ground_distance=2., ceiling_distance=45.)
    values.update(changes)
    return FeedbackObservation(**values)


def goal(**changes):
    values = dict(id="advisory-goal", skill="directed_hop", target=(0., 0.), timeout_seconds=4.)
    values.update(changes)
    return SkillGoal(**values)


def retained_view(tick):
    position, velocity, rotation, omega = RETAINED[tick]
    return {"version": 2, "representation": "articulated-rapier", "run_id": "retained-excerpt",
            "epoch": 1, "actor_id": 0, "profile_version": 1, "model": "jev", "tick": tick,
            "observation_id": f"retained-excerpt-{tick}", "position": position, "velocity": velocity,
            "rotation": rotation, "angular_velocity": omega, "grounded": True,
            "cooldown": 0., "energy": 1., "ground_distance": 0., "ceiling_distance": 45.}


def hold(controller, target=None):
    target = target or goal()
    for tick in (0, 6, 12):
        frame = controller.step(observation(tick), target)
    return frame


class JoltRecoveryAdvisoryTests(unittest.TestCase):
    def test_distinct_registered_contract_preserves_v1_source_and_supervisor(self):
        root = Path(__file__).resolve().parents[1] / "src/nca/controllers"
        for name, digest in PRESERVED_RECOVERY_SOURCE_SHA256.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
        registry = builtin_registry()
        original = registry.create("policy", "authored-jolt-recovery", 1)
        advisory = registry.create("policy", "authored-jolt-recovery", 2)
        self.assertIs(type(original), JoltRecoveryController)
        self.assertIs(type(advisory), JoltRecoveryControllerV2)
        self.assertEqual(advisory.controller_id, CONTROLLER_ID)
        self.assertEqual(advisory.specification()["schema"], "nca-jolt-recovery-reference/v2")
        self.assertEqual(advisory.specification_sha256, sha256_json(advisory.specification()))
        self.assertEqual(original.supervisor.specification(), advisory.supervisor.specification())
        self.assertEqual(original.specification()["constants"], advisory.specification()["constants"])
        self.assertEqual(hold(original).status, "succeeded")
        self.assertEqual(hold(advisory).status, "running")

    def test_retained_sample_aliasing_does_not_retire_the_unfinished_task(self):
        target = goal(target=(212., 346.5))
        controllers = (JoltRecoveryController(), JoltRecoveryControllerV2())
        for controller in controllers:
            # Synthetic nonquiet observations establish the original tick-zero
            # invocation; the retained kinematic boundary begins at tick216.
            for tick in (0, 54, 108, 162):
                controller.step(observation(tick, lifetime="retained-excerpt:1:0:1:jev",
                                             supported=False, ground_distance=None), target)
            for tick in (216, 222, 228, 234):
                frame = controller.step(FeedbackObservation.from_host(retained_view(tick)), target)
            if isinstance(controller, JoltRecoveryControllerV2):
                self.assertEqual(frame.status, "running")
                self.assertEqual(frame.reason, "supported_arrival_advisory")
                self.assertEqual(frame.action, WireAction(effort=0.))
                self.assertEqual(controller.diagnostic_state()["action_lease_ticks"], 6)
                self.assertTrue(controller.diagnostic_state()["arrival_advisory"])
            else:
                self.assertEqual(frame.status, "succeeded")
                self.assertEqual(controller.diagnostic_state()["action_lease_ticks"], 0)
        task = HopTask(target, retained_view(222))
        for start in (223, 229):
            task.consume(WireAction(effort=0.), [
                {"tick_before": tick - 1, "observation": retained_view(tick)} for tick in range(start, start + 6)])
        self.assertGreater(math.hypot(*RETAINED[223][1]), 6.)
        self.assertEqual(task.quiet_ticks, 11)
        self.assertFalse(task.done)
        self.assertFalse(task.succeeded)
        # No invented tick235 or new physical outcome is used to pass this test.

    def test_advisory_holds_without_renewing_original_deadline_or_observation_age(self):
        controller = JoltRecoveryControllerV2()
        target = goal(timeout_seconds=.4)
        self.assertEqual(hold(controller, target).phase, "settle_hold")
        frame = controller.step(observation(18), target, current_tick=20)
        lease = controller.diagnostic_state()["action_lease_ticks"]
        self.assertEqual(frame.status, "running")
        self.assertEqual(lease, 6)
        self.assertEqual(18 + lease - 20, 4)
        frame = controller.step(observation(24), target)
        self.assertEqual(frame.status, "timed_out")
        self.assertEqual(controller.diagnostic_state()["action_lease_ticks"], 0)
        self.assertFalse(controller.diagnostic_state()["arrival_advisory"])

    def test_new_disturbance_resumes_ordinary_correction_and_resets_quiet_span(self):
        controller = JoltRecoveryControllerV2()
        hold(controller)
        frame = controller.step(observation(18, position=(-8., 0.)), goal())
        self.assertEqual(frame.status, "running")
        self.assertGreater(frame.action.move[0], .1)
        self.assertFalse(controller.diagnostic_state()["arrival_advisory"])
        for tick in (24, 30):
            controller.step(observation(tick), goal())
            self.assertFalse(controller.diagnostic_state()["arrival_advisory"])
        self.assertEqual(controller.step(observation(36), goal()).phase, "settle_hold")

    def test_missing_support_or_observation_gap_cannot_continue_the_advisory(self):
        for changed in (observation(18, supported=None), observation(18, supported=False), observation(24)):
            with self.subTest(changed=changed):
                controller = JoltRecoveryControllerV2()
                hold(controller)
                self.assertEqual(controller.step(changed, goal()).status, "running")
                self.assertFalse(controller.diagnostic_state()["arrival_advisory"])

    def test_stale_or_revoked_authority_still_refuses_without_renewal(self):
        controller = JoltRecoveryControllerV2()
        hold(controller)
        with self.assertRaises(ValueError):
            controller.step(observation(12), goal())
        self.assertEqual(controller.diagnostic_state()["action_lease_ticks"], 0)
        controller = JoltRecoveryControllerV2()
        hold(controller)
        frame = controller.step(observation(18, control_generation=1), goal())
        self.assertEqual(frame.status, "canceled")
        self.assertFalse(controller.diagnostic_state()["arrival_advisory"])
        self.assertEqual(controller.diagnostic_state()["action_lease_ticks"], 0)

    def test_new_lifetime_discards_old_advisory_and_starts_new_invocation(self):
        controller = JoltRecoveryControllerV2()
        hold(controller)
        for tick in (0, 6):
            frame = controller.step(observation(tick, lifetime="synthetic:new-lifetime"), goal())
            self.assertEqual(frame.status, "running")
            self.assertFalse(controller.diagnostic_state()["arrival_advisory"])
        self.assertEqual(controller.step(observation(12, lifetime="synthetic:new-lifetime"), goal()).phase,
                         "settle_hold")
        self.assertEqual(controller.cancel("caller_verified_outcome").status, "canceled")
        self.assertEqual(controller.diagnostic_state()["action_lease_ticks"], 0)

    def test_explicit_cancel_and_new_goal_cannot_resurrect_old_advisory(self):
        controller = JoltRecoveryControllerV2()
        hold(controller)
        self.assertEqual(controller.cancel().status, "canceled")
        self.assertFalse(controller.diagnostic_state()["arrival_advisory"])
        self.assertEqual(controller.step(observation(18), goal()).status, "canceled")
        new_goal = goal(id="next-goal", target=(24., 0.))
        self.assertEqual(controller.step(observation(24), new_goal).status, "running")
        with self.assertRaises(ValueError):
            controller.step(observation(30), goal())

    def test_emergency_override_is_separate_from_advisory_holding(self):
        controller = JoltRecoveryControllerV2()
        hold(controller)
        frame = controller.step(observation(18, angle=1.3), goal())
        self.assertEqual(frame.phase, "emergency_recovery")
        self.assertTrue(frame.action.auxiliary)
        self.assertTrue(controller.diagnostic_state()["supervisor_intervention"])
        self.assertFalse(controller.diagnostic_state()["arrival_advisory"])

    def test_ordinary_launch_and_contact_recovery_actions_remain_identical(self):
        original, advisory = JoltRecoveryController(), JoltRecoveryControllerV2()
        target = goal(target=(48., 0.))
        pulses = []
        for tick in range(0, 61, 6):
            view = observation(tick, cooldown=max(0., (48 - tick) / 60) if tick >= 12 else 0.,
                               blocked=tick >= 12)
            before, after = original.step(view, target), advisory.step(view, target)
            self.assertEqual(before.action, after.action)
            self.assertEqual((before.phase, before.reason, before.status), (after.phase, after.reason, after.status))
            self.assertFalse(advisory.diagnostic_state()["arrival_advisory"])
            if after.action.primary:
                pulses.append(tick)
        self.assertEqual(pulses, [6, 54])


if __name__ == "__main__":
    unittest.main()
