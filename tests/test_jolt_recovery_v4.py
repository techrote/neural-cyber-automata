"""Retained-state and synthetic v4 software checks, with no host invocation.

Retained samples describe the consumed v3 trajectory. Changed requests below
are arithmetic probes; their later physical effects are not inferred from the
old trajectory. The independent diagnostic owns native outcomes and acceptance.
"""

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import unittest

from nca.builtins import builtin_registry
from nca.contracts import WireAction, sha256_json
from nca.controllers.baseline import FeedbackObservation, SkillGoal
from nca.controllers.jolt_recovery_v3 import JoltRecoveryControllerV3
from nca.controllers.jolt_recovery_v4 import (
    CONTROLLER_ID, PRESERVED_MOVEMENT_SOURCE_SHA256, JoltRecoveryControllerV4,
)
from nca.train.task import HopTask


def observation(tick=0, **changes):
    values = dict(tick=tick, lifetime="synthetic:settling-v4", actor="jev", position=(0., 0.),
                  velocity=(0., 0.), angle=0., angular_velocity=0., supported=True,
                  cooldown=0., energy=1., ground_distance=2., ceiling_distance=45.)
    values.update(changes)
    return FeedbackObservation(**values)


def goal(**changes):
    values = dict(id="settling-goal", skill="directed_hop", target=(48., 0.), timeout_seconds=4.)
    values.update(changes)
    return SkillGoal(**values)


def two_pulses(controller):
    target = goal()
    controller.step(observation(), target)
    first = controller.step(observation(6), target)
    for tick in range(12, 49, 6):
        controller.step(observation(tick, cooldown=max(0., (48-tick)/60)), target)
    second = controller.step(observation(54), target)
    return target, first, second


class JoltSettlingV4Tests(unittest.TestCase):
    def test_distinct_registry_contract_preserves_all_prior_motion_and_supervision(self):
        previous, current = JoltRecoveryControllerV3(), JoltRecoveryControllerV4()
        source = Path(__file__).resolve().parents[1] / "src/nca/controllers/jolt_recovery_v3.py"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), PRESERVED_MOVEMENT_SOURCE_SHA256)
        self.assertIs(type(builtin_registry().create("policy", "authored-jolt-recovery", 4)),
                      JoltRecoveryControllerV4)
        self.assertEqual(current.controller_id, CONTROLLER_ID)
        self.assertEqual(current.specification()["schema"], "nca-jolt-recovery-reference/v4")
        self.assertEqual(current.specification_sha256, sha256_json(current.specification()))
        for section in ("constants", "movement_constants", "physical_supervisor"):
            self.assertEqual(current.specification()[section], previous.specification()[section])
        self.assertIs(JoltRecoveryControllerV4._pulse, JoltRecoveryControllerV3._pulse)
        self.assertIs(JoltRecoveryControllerV4._initial_contact_transition,
                      JoltRecoveryControllerV3._initial_contact_transition)

    def test_original_supported_and_fresh_support_loss_launch_decisions_are_unchanged(self):
        # Four native source views from v2 right60-0, artifact SHA256
        # d1efb48390a1ef3fc3a40dbd5ded36743749d2c613226875f3b07054df2ba2a9.
        retained = (
            observation(0, position=(176.,346.5), supported=False, ground_distance=96.),
            observation(6, position=(175.990142822266,346.485626220703), velocity=(.178647845983505,1.37513220310211),
                        angle=-.000529368815477937, angular_velocity=-.122038200497627, supported=False, ground_distance=2.01252147893408),
            observation(12, position=(176.024917602539,346.491424560547), velocity=(.148333609104156,.846784234046936),
                        angle=-.00280909682624042, angular_velocity=-.014207330532372, ground_distance=1.99875742277646),
            observation(18, position=(176.047149658203,346.488647460938), velocity=(.103528544306755,.962234914302826),
                        angle=-.00165262573864311, angular_velocity=.0388154871761799, supported=False, ground_distance=2.0055731311598),
        )
        old, new = JoltRecoveryControllerV3(), JoltRecoveryControllerV4()
        target = goal(target=(236.,346.5))
        for view in retained:
            before, after = old.step(view,target), new.step(view,target)
            self.assertEqual(before.action,after.action)
            self.assertEqual((before.phase,before.reason),(after.phase,after.reason))
        self.assertTrue(after.action.primary)
        self.assertEqual(after.reason,"initial_contact_loss_launch")
        self.assertEqual(new.diagnostic_state()["initial_contact_witness_tick"],12)
        for distance in (-60., -24., 24., 60.):
            old, new = JoltRecoveryControllerV3(), JoltRecoveryControllerV4()
            for tick in (0,6):
                self.assertEqual(old.step(observation(tick),goal(target=(distance,0.))).action,
                                 new.step(observation(tick),goal(target=(distance,0.))).action)

    def test_retained_outer_tolerance_sample_keeps_slow_inward_traction(self):
        # V3 reference-drop28-left57 at tick156, from episode artifact SHA256
        # bc0324f30065f026735aa85a8da7da3ffe05a464831f9772fe3cc32b4491d3ef.
        # A standalone decision probe starts its own invocation at this sample.
        view = observation(156, position=(210.283004760742,348.486511230469),
                           velocity=(-2.65659928321838,1.86916565895081), angle=.00389439798891544,
                           angular_velocity=-.698528707027435, ground_distance=0.)
        target = goal(target=(207.,346.5))
        old = JoltRecoveryControllerV3().step(view,target)
        controller = JoltRecoveryControllerV4()
        frame = controller.step(view,target)
        self.assertEqual(old.action,WireAction(effort=0.))
        self.assertLess(frame.action.move[0],-.05)
        self.assertEqual(frame.phase,"settle_capture")
        self.assertTrue(controller.diagnostic_state()["settling_capture"])
        self.assertFalse(controller.diagnostic_state()["interior_quiet_hold"])
        self.assertEqual(frame.status,"running")

    def test_interior_margin_uses_euclidean_distance_and_preserves_braking(self):
        target = goal(target=(0.,0.))
        for dx in (-2.9,2.9):
            controller = JoltRecoveryControllerV4()
            frame = controller.step(observation(position=(dx,2.),cooldown=.2),target)
            self.assertEqual(frame.phase,"settle_capture")
            self.assertLess(frame.action.move[0]*dx,0.)
        controller = JoltRecoveryControllerV4()
        controller.step(observation(position=(-2.5,0.),velocity=(5.5,0.),cooldown=.2),target)
        state = controller.diagnostic_state()
        self.assertFalse(state["interior_quiet_hold"])
        self.assertLess(controller._previous_move,0.)
        frame = controller.step(observation(6,position=(-2.,0.),velocity=(1.,0.),cooldown=.2),target)
        self.assertEqual(frame.action,WireAction(effort=0.))
        self.assertEqual(frame.status,"running")

    def test_capture_hysteresis_releases_on_distance_support_uncertainty_or_gap(self):
        target = goal(target=(0.,0.))
        controller = JoltRecoveryControllerV4()
        controller.step(observation(position=(-3.8,0.),cooldown=.2),target)
        frame = controller.step(observation(6,position=(-4.2,0.),cooldown=.2),target)
        self.assertEqual(frame.phase,"settle_capture")
        self.assertLessEqual(abs(frame.action.move[0]),.125000001)
        controller.step(observation(12,position=(-5.1,0.),cooldown=.2),target)
        self.assertFalse(controller.diagnostic_state()["settling_capture"])
        for changed in ({"supported":False},{"supported":None},{"tick":12}):
            controller = JoltRecoveryControllerV4()
            controller.step(observation(position=(-3.8,0.),cooldown=.2),target)
            view = replace(observation(6,position=(-4.2,0.),cooldown=.2),**changed)
            controller.step(view,target)
            self.assertFalse(controller.diagnostic_state()["settling_capture"])

    def test_capture_cancel_takeover_emergency_and_expiry_keep_original_authority(self):
        for boundary in ("cancel","takeover","emergency","expiry","duplicate"):
            with self.subTest(boundary=boundary):
                controller = JoltRecoveryControllerV4()
                target = goal(target=(0.,0.),timeout_seconds=.1 if boundary=="expiry" else 4.)
                controller.step(observation(position=(-3.8,0.),cooldown=.2),target)
                if boundary=="cancel": controller.cancel()
                if boundary=="duplicate":
                    with self.assertRaises(ValueError):
                        controller.step(observation(position=(-3.8,0.),cooldown=.2),target)
                    self.assertEqual(controller.diagnostic_state()["action_lease_ticks"],0)
                    continue
                view = observation(6,position=(-4.2,0.),cooldown=.2,
                                   control_generation=int(boundary=="takeover"),
                                   angle=1.3 if boundary=="emergency" else 0.)
                frame = controller.step(view,target)
                self.assertFalse(controller.diagnostic_state()["settling_capture"])
                self.assertEqual(frame.action.move,(0.,0.))
                self.assertFalse(frame.action.primary)
                if boundary!="emergency":
                    self.assertEqual(controller.diagnostic_state()["action_lease_ticks"],0)

    def test_fresh_acknowledged_second_pulse_can_coast_one_interval(self):
        controller = JoltRecoveryControllerV4()
        target,first,second = two_pulses(controller)
        self.assertTrue(first.action.primary and second.action.primary)
        coast = controller.step(observation(60,position=(.5,-2.),velocity=(3.4,-12.9),cooldown=.53),target)
        self.assertEqual(coast.phase,"post_pulse_coast")
        self.assertEqual(coast.action.move,(0.,0.))
        self.assertFalse(coast.action.primary or coast.action.secondary or coast.action.auxiliary)
        self.assertEqual(controller.diagnostic_state()["action_lease_ticks"],6)
        resumed = controller.step(observation(66,position=(.7,-2.4),velocity=(3.5,-8.),cooldown=.43),target)
        self.assertFalse(controller.diagnostic_state()["post_pulse_coast"])
        self.assertEqual(resumed.status,"running")
        self.assertEqual(controller.diagnostic_state()["action_lease_ticks"],12)
        self.assertGreater(resumed.action.move[0],0.)

    def test_post_pulse_coast_requires_current_consecutive_rising_raw_support(self):
        invalid = ({"supported":False},{"supported":None},{"velocity":(3.,-5.99)},
                   {"cooldown":None},{"cooldown":0.},{"tick":66},{"control_generation":1})
        for changed in invalid:
            with self.subTest(changed=changed):
                controller = JoltRecoveryControllerV4()
                target,_,_ = two_pulses(controller)
                view = replace(observation(60,velocity=(3.,-12.),cooldown=.53),**changed)
                controller.step(view,target)
                self.assertFalse(controller.diagnostic_state()["post_pulse_coast"])
        controller = JoltRecoveryControllerV4()
        target,_,_ = two_pulses(controller)
        controller.step(observation(60,velocity=(3.,-12.),cooldown=.53),target,current_tick=61)
        self.assertFalse(controller.diagnostic_state()["post_pulse_coast"])

    def test_pulse_coast_does_not_relax_acknowledgement_or_two_edge_limit(self):
        controller = JoltRecoveryControllerV4()
        target,_,_ = two_pulses(controller)
        for tick in range(60,235,6):
            frame = controller.step(observation(tick,cooldown=.2 if tick==60 else 0.),target)
            self.assertFalse(frame.action.primary)
        self.assertEqual(controller.diagnostic_state()["pulse_count"],2)
        controller = JoltRecoveryControllerV4()
        for tick in range(0,91,6): controller.step(observation(tick),goal())
        self.assertEqual(controller.diagnostic_state()["pulse_count"],1)

    def test_sampled_arrival_is_advisory_and_tick_223_still_prevents_completion(self):
        target = goal(target=(0.,0.))
        controller = JoltRecoveryControllerV4()
        for tick in (0,54,108,162):
            controller.step(observation(tick,position=(-20.,0.),supported=False,ground_distance=None,cooldown=.2),target)
        for tick in (216,222,228,234):
            frame = controller.step(observation(tick,cooldown=.2),target)
        self.assertEqual(frame.status,"running")
        self.assertTrue(controller.diagnostic_state()["arrival_advisory"])
        self.assertEqual(controller.diagnostic_state()["action_lease_ticks"],6)
        def native(tick):
            return {"version":2,"representation":"articulated-rapier","run_id":"synthetic-alias-v4",
                    "epoch":1,"actor_id":0,"profile_version":1,"model":"jev","observation_id":f"alias-{tick}",
                    "tick":tick,"position":(0.,0.),"velocity":
                    (7.00780153274536,2.93841552734375) if tick==223 else (0.,0.),
                    "grounded":True,"rotation":0.,"angular_velocity":0.,"valid":True,
                    "cooldown":0.,"energy":1.,"ground_distance":0.,"ceiling_distance":45.}
        task=HopTask(target,native(222))
        for start in (223,229):
            task.consume(WireAction(effort=0.),[
                {"tick_before":tick-1,"observation":native(tick)} for tick in range(start,start+6)])
        self.assertEqual(task.quiet_ticks,11)
        self.assertFalse(task.done)
        self.assertEqual(controller.step(observation(240),target).status,"timed_out")
        self.assertEqual(controller.diagnostic_state()["action_lease_ticks"],0)

    def test_case_names_cannot_steer_capture(self):
        views = [observation(tick,position=(-3.8+.05*tick,1.),cooldown=.2) for tick in (0,6,12)]
        a,b=JoltRecoveryControllerV4(),JoltRecoveryControllerV4()
        for view in views:
            before=a.step(view,goal(id="arbitrary-a",target=(0.,0.)))
            after=b.step(view,goal(id="arbitrary-b",target=(0.,0.)))
            self.assertEqual(before.action,after.action)
            self.assertEqual((before.phase,before.status,before.reason),(after.phase,after.status,after.reason))


if __name__ == "__main__":
    unittest.main()
