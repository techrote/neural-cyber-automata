"""Jolt contact-transition launches and bounded, shorter recovery arcs.

The retained v2 observations motivate these local control changes. This module
does not replay a trajectory, inspect an evaluator, or predict contact forces.
The pulse envelope is an admission heuristic; native receipts establish motion.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from nca.contracts import WireAction
from .baseline import ControllerFrame, FeedbackObservation, SkillGoal
from .jolt_recovery_v2 import JoltRecoveryControllerV2
from .shaping import bounded_vector, clamp, wrapped_angle


CONTROLLER_ID = "authored-jolt-recovery/v3"
PRESERVED_ADVISORY_SOURCE_SHA256 = "e919d10d78ba1d9dacecf135014f9e344a1ea7e43817f84d29e067577565fdae"


class JoltRecoveryControllerV3(JoltRecoveryControllerV2):
    """Keep fresh contact transitions actionable without extending authority.

    A pending first launch may consume a newly observed support loss exactly
    one decision later, but only while the body remains slow and near its prior
    supported pose and a measured floor. Otherwise its witness is discarded.
    Later pulses still require a new physical recovery window. Ground holding
    is advisory even when its local observations look quiet.
    """

    MINIMUM_RECOVERY_EFFORT = .40
    RECOVERY_APEX_ROOM_FRACTION = .25
    MAXIMUM_RECOVERY_APEX = 7.0
    GROUND_CORRECTION_TICKS = 12
    GROUND_CORRECTION_REQUEST = 1.0
    ARRIVAL_COUNTER_REQUEST = .10

    def __init__(self, actor: str = "jev") -> None:
        if hashlib.sha256(Path(__file__).with_name("jolt_recovery_v2.py").read_bytes()).hexdigest() != PRESERVED_ADVISORY_SOURCE_SHA256:
            raise ValueError("the movement controller's preserved advisory source changed")
        super().__init__(actor)

    @property
    def controller_id(self) -> str:
        return CONTROLLER_ID

    def specification(self) -> dict:
        result = super().specification()
        result.update({
            "schema": "nca-jolt-recovery-reference/v3",
            "preserved_advisory_source_sha256": PRESERVED_ADVISORY_SOURCE_SHA256,
            "movement_constants": {name: getattr(self, name) for name in (
                "MINIMUM_RECOVERY_EFFORT", "RECOVERY_APEX_ROOM_FRACTION",
                "MAXIMUM_RECOVERY_APEX", "GROUND_CORRECTION_TICKS",
                "GROUND_CORRECTION_REQUEST", "ARRIVAL_COUNTER_REQUEST")},
            "initial_launch": "fresh supported anticipation or its exact six-tick supported-to-unsupported transition; current floor<=16, displacement<=.40 and speed<=6; synchronous fresh observation; all pulse bounds rechecked",
            "pending_launch": "one six-tick anticipation witness; unknown support, gap, motion, admission failure or lost authority discards it; a contact transition keeps only its original start",
            "first_pulse_aim": "preserved initial room/60 projection for the first pulse regardless of its fresh physical reason",
            "pulse_aim": "first pulse uses the preserved room/60 projection; recovery pulses normalize their separately solved horizontal/vertical impulse components",
            "pulse_room": "first pulse uses the preserved positive impulse quadratic; recovery solves the total horizontal velocity quadratic for its fixed bounded lift, then subtracts signed observed toward velocity",
            "second_launch": "same new physical recovery window, cooldown acknowledgment, energy and two-edge limit; shorter local room-bound recovery arc",
            "recovery_vertical_impulse": "max(86*.40, sqrt(2*156*min(7,.25*room)) + current_vy); capped by caller effort, ceiling and original remaining time; post-pulse upward speed>=20",
            "recovery_horizontal_impulse": "solve total positive horizontal speed from V*(conservative_flight+.1)+V^2/(2*100)<=room; subtract signed existing toward speed; cap vector by caller effort and horizontal aim .82; require positive resulting toward speed",
            "recovery_flight": "2*(existing positive upward speed + vertical impulse)/156 plus sqrt(2*max(0,dy)/156); conservative local bound, not a contact simulation",
            "recovery_effort": "normalize the solved impulse vector; .40 minimum and max(.45,caller pace) maximum; no raised caller budget",
            "ordinary_arrival_hold": "on current raw support inside caller tolerance, speed<=6 and angle<=.32, request neutral immediately with status running; twelve sampled quiet ticks remain advisory only",
            "ordinary_arrival_braking": "inside caller tolerance with abs(vx)>6, shaped -sign(vx)*.10 traction; no free-air brake and no success claim",
            "ordinary_ground_correction": "after both primary edges, a fresh supported no-progress window can request bounded full traction for at most12 ticks; support loss, new motion, gap, quiet envelope or authority change ends it; a new window is required to restart",
            "supervisor_unchanged": True,
        })
        return result

    def _clear_motion(self) -> None:
        super()._clear_motion()
        self._ground_correction_since: int | None = None
        self._ground_correction_position: tuple[float, float] | None = None

    def _clear_windows(self) -> None:
        super()._clear_windows()
        self._ground_correction_since = None
        self._ground_correction_position = None

    def _clear_pending(self) -> None:
        self._pending_reason = self._pending_since = None
        self._pending_position = None

    def _initial_contact_transition(self, observation: FeedbackObservation,
                                    goal: SkillGoal, *, synchronous: bool) -> bool:
        previous = self._previous_observation
        return bool(
            synchronous and self._pulse_count == 0
            and self._pending_reason == "initial_supported_launch"
            and self._pending_since is not None and self._pending_position is not None
            and previous is not None and previous.supported is True
            and previous.tick == self._pending_since
            and observation.tick - self._pending_since == self.ANTICIPATION_TICKS
            and observation.supported is False
            and observation.ground_distance is not None
            and observation.ground_distance <= self.NEAR_GROUND_DISTANCE
            and math.dist(observation.position, self._pending_position) <= self.MAXIMUM_STALL_PROGRESS
            and math.hypot(*observation.velocity) <= self.MAXIMUM_STALL_SPEED
            and math.dist(observation.position, goal.target) > goal.tolerance
        )

    def _pulse(self, observation: FeedbackObservation, goal: SkillGoal, dx: float,
               dy: float, *, recovery: bool) -> tuple[tuple[float, float], float] | None:
        if not recovery:
            return super()._pulse(observation, goal, dx, dy, recovery=False)
        room = abs(dx) - self.SETTLING_MARGIN
        toward = math.copysign(1.0, dx) * observation.velocity[0]
        ceiling = observation.ceiling_distance
        if (room <= 0 or toward < -self.MAXIMUM_OPPOSING_LAUNCH_SPEED
                or ceiling is None or ceiling <= self.CEILING_MARGIN):
            return None
        maximum_impulse = self.JUMP * max(.45, goal.pace)
        upward = max(0.0, -observation.velocity[1])
        extra_flight = math.sqrt(2 * max(0.0, dy) / self.GRAVITY)
        available = (goal.timeout_seconds - self._latest.elapsed_ticks / 60
                     - self.SETTLING_TIME_RESERVE - extra_flight)
        maximum_upward = min(math.sqrt(2 * self.GRAVITY * (ceiling - self.CEILING_MARGIN)),
                             .5 * self.GRAVITY * max(0.0, available))
        desired_upward = math.sqrt(2 * self.GRAVITY * min(
            self.MAXIMUM_RECOVERY_APEX, self.RECOVERY_APEX_ROOM_FRACTION * room))
        vertical = max(self.JUMP * self.MINIMUM_RECOVERY_EFFORT,
                       desired_upward + observation.velocity[1],
                       self.MINIMUM_UPWARD_LAUNCH_SPEED + observation.velocity[1])
        # Existing upward velocity counts against both ceiling and time. A
        # downward velocity is not counted as beneficial in horizontal travel.
        if vertical > min(maximum_impulse, maximum_upward - upward):
            return None
        flight = 2 * (upward + vertical) / self.GRAVITY + extra_flight
        duration = flight + self.REACTION_SECONDS
        maximum_velocity = 2 * room / (duration + math.sqrt(
            duration * duration + 2 * room / self.BRAKING_ACCELERATION))
        horizontal = min(maximum_velocity - toward,
                         math.sqrt(max(0.0, maximum_impulse ** 2 - vertical ** 2)),
                         vertical * self.MAXIMUM_HORIZONTAL_AIM / math.sqrt(1 - self.MAXIMUM_HORIZONTAL_AIM ** 2))
        if horizontal <= 0 or toward + horizontal <= 0:
            return None
        impulse = math.hypot(horizontal, vertical)
        aim = (math.copysign(horizontal / impulse, dx), -vertical / impulse)
        effort = impulse / self.JUMP
        if (abs(aim[0]) < self.MINIMUM_HORIZONTAL_AIM
                or effort < self.MINIMUM_RECOVERY_EFFORT
                or observation.velocity[1] - vertical > -self.MINIMUM_UPWARD_LAUNCH_SPEED):
            return None
        return aim, effort

    def _ground_correction(self, observation: FeedbackObservation, distance: float,
                           goal: SkillGoal, reason: str | None) -> bool:
        active = self._ground_correction_since is not None
        if active:
            still_observed = (
                observation.supported is True and distance > goal.tolerance
                and self._ground_correction_position is not None
                and math.dist(observation.position, self._ground_correction_position) <= self.MAXIMUM_STALL_PROGRESS
                and math.hypot(*observation.velocity) <= self.MAXIMUM_STALL_SPEED
                and observation.tick - self._ground_correction_since < self.GROUND_CORRECTION_TICKS
            )
            if still_observed:
                return True
            self._ground_correction_since = self._ground_correction_position = None
            self._history.clear()
            self._stall_ticks = 0
            return False
        if (self._pulse_count == self.MAXIMUM_PULSES
                and reason in {"supported_no_progress", "supported_blocked_progress"}):
            self._ground_correction_since = observation.tick
            self._ground_correction_position = observation.position
            self._recovery_epoch += 1
            return True
        return False

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
            quiet = (observation.supported is True and distance <= goal.tolerance
                     and math.hypot(*observation.velocity) <= self.MAXIMUM_ARRIVAL_SPEED
                     and abs(wrapped_angle(observation.angle)) <= self.MAXIMUM_ARRIVAL_ANGLE)
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
            if (not primary and observation.supported is True and distance <= goal.tolerance
                    and abs(observation.velocity[0]) > self.MAXIMUM_ARRIVAL_SPEED):
                move = -math.copysign(self.ARRIVAL_COUNTER_REQUEST, observation.velocity[0])
                self._phase, self._reason = "settle_brake", "supported_arrival_braking"
            lean = .16 * clamp(desired_x / self.SPEED)
            if "anticipate" in self._phase:
                lean = -.10 * math.copysign(1.0, dx)
            elif distance <= goal.tolerance:
                lean = 0.0
            trim = clamp(wrapped_angle(lean - observation.angle) * .55 - observation.angular_velocity * .20, -.85, .85)
            dt = self._latest.sample_ticks / 60
            if observation.supported is not True or "anticipate" in self._phase or primary:
                self._x.cancel()
                shaped_move = 0.0
            else:
                shaped_move = self._x.step(move, dt)
            action = WireAction(move=bounded_vector((shaped_move, 0.0)), aim=aim, effort=effort,
                                trim=self._turn.step(trim, dt), primary=primary, secondary=False)
            if quiet:
                self._clear_pending()
                self._x.cancel()
                self._turn.cancel()
                self._phase = "settle_hold"
                self._reason = "supported_arrival_advisory" if self._arrival_advisory else "supported_quiet_hold"
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
            "action_lease_ticks": self._latest.action_lease_ticks,
            "arrival_advisory": self._arrival_advisory,
            "initial_contact_transition": contact_transition,
            "initial_contact_witness_tick": witness_tick,
            "ground_correction": ground_correction,
        }
        return self._frame(action, distance)
