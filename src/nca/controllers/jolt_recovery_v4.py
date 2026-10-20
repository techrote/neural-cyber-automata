"""Interior settling and a fresh, bounded second-pulse coast for Jolt.

These local feedback changes are motivated by retained native observations.
Neither an interior quiet sample nor a pulse-coast decision proves a physical
outcome. The caller still owns continuous arrival and native cancellation.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from nca.contracts import WireAction
from .baseline import ControllerFrame, FeedbackObservation, SkillGoal
from .jolt_recovery_v3 import JoltRecoveryControllerV3
from .shaping import bounded_vector, clamp, wrapped_angle


CONTROLLER_ID = "authored-jolt-recovery/v4"
PRESERVED_MOVEMENT_SOURCE_SHA256 = "367447b4dd732a8e999f1d172795685fa4591028026ebbdd2697d12dc0c2aab3"


class JoltRecoveryControllerV4(JoltRecoveryControllerV3):
    """Retain arrival margin while respecting the same launch authority.

    A supported entry into the caller's tolerance starts slow inward capture.
    The capture stays active through small outward disturbances and releases on
    loss/uncertainty of support, a sample gap or the outer release distance.
    Only an interior, slower sample permits neutral holding. The inherited
    sampled arrival detector remains advisory and never retires the invocation.

    A newly acknowledged second pulse can coast for its next six-tick interval
    if the next current observation is still rising despite raw support. This
    is one bounded ordinary motion choice, not an inferred contact state.
    """

    CAPTURE_INTERIOR_RATIO = .75
    CAPTURE_RELEASE_RATIO = 1.25
    CAPTURE_SPEED = 2.0
    CAPTURE_REQUEST = .10
    POST_PULSE_RISE_SPEED = 6.0
    POST_PULSE_COAST_TICKS = 6

    def __init__(self, actor: str = "jev") -> None:
        source = Path(__file__).with_name("jolt_recovery_v3.py")
        if hashlib.sha256(source.read_bytes()).hexdigest() != PRESERVED_MOVEMENT_SOURCE_SHA256:
            raise ValueError("the settling controller's preserved movement source changed")
        super().__init__(actor)

    @property
    def controller_id(self) -> str:
        return CONTROLLER_ID

    def specification(self) -> dict:
        result = super().specification()
        result.update({
            "schema": "nca-jolt-recovery-reference/v4",
            "preserved_movement_source_sha256": PRESERVED_MOVEMENT_SOURCE_SHA256,
            "settling_constants": {name: getattr(self, name) for name in (
                "CAPTURE_INTERIOR_RATIO", "CAPTURE_RELEASE_RATIO", "CAPTURE_SPEED",
                "CAPTURE_REQUEST", "POST_PULSE_RISE_SPEED", "POST_PULSE_COAST_TICKS")},
            "ordinary_arrival_hold": "neutral only on current raw support inside .75*caller Euclidean tolerance, speed<=2 and angle<=.32; status remains running and sampled arrival remains advisory",
            "ordinary_arrival_braking": "inside capture, bounded .10 inward traction targets slow approach; inside the inner region with abs(vx)>2, bounded .10 counter-traction; zero requested motion does not claim an active host brake",
            "settling_capture": "enter on current supported distance<=caller tolerance; retain until distance>1.25*tolerance, support loss/uncertainty, sample gap or authority boundary; keep ordinary shaped traction inside outer tolerance until interior quiet",
            "capture_shaping": "unchanged Schmitt/play/slew shaping; .10 requested magnitude yields at most .125 shaped magnitude from preserved .025 play, except finite slew from prior ordinary drive",
            "post_second_pulse_coast": "at exactly six ticks after the second requested pulse, its fresh current cooldown acknowledgement, consecutive source sample and raw supported vy<=-6 can request one six-tick traction unload with source lease capped at6; no renewed witness, extra pulse or inferred unsupported state",
            "pulse_vectors_unchanged": True,
            "initial_contact_lifecycle_unchanged": True,
            "completion": "caller validates continuous task outcome and current ownership, then cancels local/native intent; capture never supplies evaluator state or changes success thresholds",
        })
        return result

    def _clear_motion(self) -> None:
        super()._clear_motion()
        self._capture_active = False

    def _clear_windows(self) -> None:
        super()._clear_windows()
        self._capture_active = False

    def _update_capture(self, observation: FeedbackObservation, goal: SkillGoal,
                        distance: float) -> None:
        if observation.supported is not True or distance > goal.tolerance * self.CAPTURE_RELEASE_RATIO:
            self._capture_active = False
        elif distance <= goal.tolerance:
            self._capture_active = True

    def _capture_request(self, observation: FeedbackObservation, goal: SkillGoal,
                         distance: float, dx: float) -> tuple[float, bool]:
        interior = distance <= goal.tolerance * self.CAPTURE_INTERIOR_RATIO
        quiet = (interior and math.hypot(*observation.velocity) <= self.CAPTURE_SPEED
                 and abs(wrapped_angle(observation.angle)) <= self.MAXIMUM_ARRIVAL_ANGLE)
        if quiet:
            return 0.0, True
        if not interior and abs(dx) > 1e-12:
            return math.copysign(self.CAPTURE_REQUEST, dx), False
        if abs(observation.velocity[0]) > self.CAPTURE_SPEED:
            return -math.copysign(self.CAPTURE_REQUEST, observation.velocity[0]), False
        return 0.0, False

    def _fresh_post_pulse_rise(self, observation: FeedbackObservation, *, synchronous: bool) -> bool:
        previous = self._previous_observation
        return bool(synchronous and self._pulse_count == self.MAXIMUM_PULSES
                    and self._last_pulse_tick is not None and self._pulse_acknowledged
                    and previous is not None and previous.tick == self._last_pulse_tick
                    and observation.tick - self._last_pulse_tick == self.POST_PULSE_COAST_TICKS
                    and observation.supported is True and observation.cooldown is not None
                    and observation.cooldown > 0 and observation.velocity[1] <= -self.POST_PULSE_RISE_SPEED)

    def step(self, observation: FeedbackObservation, goal: SkillGoal, *,
             current_tick: int | None = None) -> ControllerFrame:
        if not isinstance(goal, SkillGoal) or goal.skill != "directed_hop":
            self.cancel("invalid_goal_mapping")
            raise ValueError("the recovery policy requires an explicit Jolt directed-hop goal")
        emergency = None
        if isinstance(observation, FeedbackObservation) and (
                abs(wrapped_angle(observation.angle)) >= self.supervisor.EMERGENCY_ANGLE
                or abs(observation.angular_velocity) >= self.supervisor.EMERGENCY_ANGULAR_SPEED):
            emergency = "unstable_attitude"
        try:
            self._latest = self.supervisor.admit(observation, goal, current_tick=current_tick, emergency=emergency)
        except ValueError:
            self.cancel(self.supervisor.reason)
            raise
        if self._latest.context != self._context:
            self._clear_motion()
            self._context = self._latest.context
        if (self._latest.status == "running" and self._last_pulse_tick is not None
                and observation.tick > self._last_pulse_tick and observation.cooldown is not None
                and observation.cooldown > 0):
            self._pulse_acknowledged = True
        self._arrival_advisory = False
        pulse_reason = None
        contact_transition = False
        witness_tick = None
        ground_correction = False
        capture_hold = False
        post_pulse_coast = False
        distance = math.dist(observation.position, goal.target)
        if self._latest.override is not None:
            self._clear_windows()
            self._x.cancel()
            self._turn.cancel()
            self._phase = "emergency_recovery" if self._latest.status == "running" else "coast"
            self._reason = self._latest.reason
            action = self._latest.override
        else:
            dx, dy = goal.target[0] - observation.position[0], goal.target[1] - observation.position[1]
            previous_support = self._previous_observation.supported if self._previous_observation else None
            self._observe_motion(observation, dx)
            self._quiet_arrival(observation, goal, distance)
            self._update_capture(observation, goal, distance)
            effort = max(.45, goal.pace)
            maximum_speed = self.SPEED * effort
            desired_x = clamp(self.POSITION_GAIN * dx, -maximum_speed, maximum_speed)
            move = desired_x / maximum_speed if observation.supported is True else 0.0
            self._phase = ("settle" if distance <= goal.tolerance * 2 else "approach") \
                if observation.supported is True else "airborne_coast"
            if observation.supported is True and previous_support is False:
                self._phase = "landing"
            self._reason = "ordinary_control"
            recovery_reason = self._recovery_reason(observation, goal, dx, dy)
            ground_correction = self._ground_correction(observation, distance, goal, recovery_reason)
            if self._pending_recovery_still_observed(observation, goal, dx, dy):
                recovery_reason = self._pending_reason
            contact_transition = self._initial_contact_transition(
                observation, goal, synchronous=current_tick in (None, observation.tick))
            witness_tick = self._pending_since if contact_transition else None
            if self._pending_since is not None and observation.tick - self._pending_since > self.ANTICIPATION_TICKS:
                self._clear_pending()
            ready = (self._pulse_count < self.MAXIMUM_PULSES and observation.cooldown is not None
                     and observation.cooldown <= 0 and observation.energy is not None and observation.energy >= .15
                     and (self._last_pulse_tick is None or observation.tick > self._last_pulse_tick + 6))
            initial = self._pulse_count == 0 and observation.supported is True and distance > goal.tolerance
            reason = ("initial_contact_loss_launch" if contact_transition else
                      "initial_supported_launch" if initial else recovery_reason)
            if self._pulse_count and not self._pulse_acknowledged:
                reason = None
            pulse = self._pulse(observation, goal, dx, dy, recovery=self._pulse_count > 0) if ready and reason else None
            if pulse is None:
                self._clear_pending()
            elif contact_transition:
                # This is a new raw physical observation, not retained support.
                # Keep its one-interval antecedent rather than starting a fresh
                # anticipation and silently carrying it to a later observation.
                self._pending_reason = reason
            elif self._pending_reason != reason:
                self._pending_reason, self._pending_since = reason, observation.tick
                self._pending_position = observation.position
                if not initial:
                    self._recovery_epoch += 1
            support_ready = (observation.supported is True and self._support_since is not None
                             and observation.tick - self._support_since >= self.SUPPORT_DWELL_TICKS)
            airborne_ready = observation.supported is False and (contact_transition or reason in {
                "unsupported_edge_stall", "unexpected_support_loss_descent"})
            primary = False
            aim = (math.copysign(.6, dx), -.8)
            if pulse is not None and self._pending_since is not None:
                aim = pulse[0]
                move = 0.0
                self._reason = reason
                contact = reason in {"supported_blocked_progress", "supported_no_progress"}
                self._phase = "anticipate" if initial else "contact_recovery_anticipate" if contact else "support_loss_anticipate"
                if (observation.tick - self._pending_since == self.ANTICIPATION_TICKS
                        and (support_ready or airborne_ready)):
                    primary, effort, pulse_reason = True, pulse[1], reason
                    self._phase = "launch" if initial or contact_transition else "contact_recovery_launch" if contact else "support_loss_launch"
                    self._pulse_count += 1
                    self._last_pulse_tick = observation.tick
                    self._pulse_acknowledged = False
                    self._clear_pending()
                    self._history.clear()
            elif ground_correction:
                move = math.copysign(self.GROUND_CORRECTION_REQUEST, dx)
                self._phase, self._reason = "ground_correction", "supported_stall_ground_correction"
            post_pulse_coast = (not primary and self._fresh_post_pulse_rise(
                observation, synchronous=current_tick in (None, observation.tick)))
            if post_pulse_coast:
                move = 0.0
                self._phase, self._reason = "post_pulse_coast", "fresh_rising_second_pulse"
            elif not primary and "anticipate" not in self._phase and self._capture_active:
                move, capture_hold = self._capture_request(observation, goal, distance, dx)
                self._phase = "settle_hold" if capture_hold else "settle_capture"
                self._reason = ("supported_arrival_advisory" if self._arrival_advisory else
                                "supported_interior_quiet_hold") if capture_hold else "supported_interior_capture"
            lean = .16 * clamp(desired_x / self.SPEED)
            if "anticipate" in self._phase:
                lean = -.10 * math.copysign(1.0, dx)
            elif self._capture_active or post_pulse_coast:
                lean = 0.0
            trim = clamp(wrapped_angle(lean - observation.angle) * .55 - observation.angular_velocity * .20, -.85, .85)
            dt = self._latest.sample_ticks / 60
            if observation.supported is not True or "anticipate" in self._phase or primary or post_pulse_coast:
                self._x.cancel()
                shaped_move = 0.0
            else:
                shaped_move = self._x.step(move, dt)
            action = WireAction(move=bounded_vector((shaped_move, 0.0)), aim=aim, effort=effort,
                                trim=self._turn.step(trim, dt), primary=primary, secondary=False)
            if capture_hold:
                self._clear_pending()
                self._x.cancel()
                self._turn.cancel()
                self._phase = "settle_hold"
                self._reason = "supported_arrival_advisory" if self._arrival_advisory else "supported_interior_quiet_hold"
                action = WireAction(effort=0.0)
            self._previous_move = action.move[0]
            self._previous_observation = observation
        self._diagnostics = {
            "tick": observation.tick, "phase": self._phase, "reason": self._reason,
            "pulse_count": self._pulse_count, "pulse_reason": pulse_reason,
            "pulse_acknowledged_by_cooldown": self._pulse_acknowledged,
            "recovery_epoch": self._recovery_epoch,
            "support_state": "supported" if observation.supported is True else "unsupported" if observation.supported is False else "unknown",
            "supported_stall_ticks": self._stall_ticks if observation.supported is True else 0,
            "support_loss_ticks": 0 if self._loss_since is None else observation.tick - self._loss_since,
            "progress_cells": self._progress,
            "supervisor_intervention": self._latest.intervention,
            "supervisor_reason": self._latest.reason if self._latest.intervention else None,
            "action_lease_ticks": min(self._latest.action_lease_ticks, self.POST_PULSE_COAST_TICKS)
                                  if post_pulse_coast else self._latest.action_lease_ticks,
            "arrival_advisory": self._arrival_advisory,
            "initial_contact_transition": contact_transition,
            "initial_contact_witness_tick": witness_tick,
            "ground_correction": ground_correction,
            "settling_capture": self._capture_active,
            "interior_quiet_hold": capture_hold,
            "post_pulse_coast": post_pulse_coast,
        }
        return self._frame(action, distance)
