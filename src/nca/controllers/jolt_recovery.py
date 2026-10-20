"""Versioned Jolt motion with fresh-contact recovery and a separate supervisor.

This authored policy chooses ordinary movement, posture, aimed pulses and
landing correction. The accompanying supervisor owns protocol fencing and a
small declared emergency self-right operation. Neither component owns physics
or reads evaluator state. Pulse counts below count requests, not actual hops.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import hashlib
import math
from pathlib import Path

from nca.contracts import WireAction, sha256_json
from .baseline import BaselineTuning, ControllerFrame, FeedbackObservation, SkillGoal
from .recovery_supervisor import PhysicalRecoverySupervisor, Supervision
from .shaping import RequestShaper, bounded_vector, clamp, wrapped_angle


CONTROLLER_ID = "authored-jolt-recovery/v1"
PRESERVED_SOURCE_SHA256 = {
    "baseline.py": "19929806429698977cdba82b855c52d6362d1dd2bb4e2aa7161e41bb1bb3f210",
    "jolt_arrival.py": "b9eb8304aa4c481c253ba1c39d39ae7a2bba03fe9e1d8bf867eef9c96f7a500b",
    "shaping.py": "655719bb3e8faf6b3c4ff8c94d6db8624285828549adfd6de86b83f2c7597de8",
}


class JoltRecoveryController:
    """One directed-hop invocation, with at most one freshly justified re-hop.

    A second request needs a completed physical progress window, a raw support
    or measured near-floor condition, readiness and a conservative launch-room
    estimate. Neither elapsed cooldown nor a carried contact latch suffices.
    ``diagnostic_state`` reports the latest decision's local evidence; actual
    impulse and recovery outcomes must be derived from native receipts.
    """

    SPEED = 48.0
    JUMP = 86.0
    GRAVITY = 156.0
    BRAKING_ACCELERATION = 100.0
    POSITION_GAIN = 2.5
    REACTION_SECONDS = .1
    SETTLING_MARGIN = 2.0
    MINIMUM_PULSE_EFFORT = .55
    MINIMUM_UPWARD_LAUNCH_SPEED = 20.0
    MAXIMUM_OPPOSING_LAUNCH_SPEED = 6.0
    MINIMUM_HORIZONTAL_AIM = .05
    MAXIMUM_HORIZONTAL_AIM = .82
    INITIAL_AIM_ROOM_SCALE = 60.0
    RECOVERY_AIM_ROOM_SCALE = 90.0
    CEILING_MARGIN = 6.0
    SETTLING_TIME_RESERVE = .20
    MAXIMUM_PULSES = 2
    ANTICIPATION_TICKS = 6
    SUPPORT_DWELL_TICKS = 3
    MAXIMUM_WINDOW_GAP_TICKS = 6
    WATCHDOG_TICKS = 18
    MAXIMUM_STALL_PROGRESS = .40
    MAXIMUM_STALL_SPEED = 6.0
    MINIMUM_DRIVE_REQUEST = .15
    NEAR_GROUND_DISTANCE = 16.0
    LOSS_DESCENT_TICKS = 12
    MINIMUM_DESCENT_SPEED = 12.0
    MAXIMUM_ARRIVAL_SPEED = 6.0
    MAXIMUM_ARRIVAL_ANGLE = .32
    QUIET_SPAN_TICKS = 12

    def __init__(self, actor: str = "jev") -> None:
        if actor != "jev":
            raise ValueError("the recovery reference implements only the declared Jolt profile")
        for name, expected in PRESERVED_SOURCE_SHA256.items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
                raise ValueError("the recovery reference's preserved source dependency changed")
        self.actor = actor
        self.tuning = BaselineTuning()
        self._x = RequestShaper(self.tuning.shaping)
        self._turn = RequestShaper(self.tuning.shaping)
        self.supervisor = PhysicalRecoverySupervisor()
        self.reset()

    @property
    def controller_id(self) -> str:
        return CONTROLLER_ID

    def specification(self) -> dict:
        return {
            "schema": "nca-jolt-recovery-reference/v1",
            "controller_id": self.controller_id,
            "actor": self.actor, "skill": "directed_hop", "profile_version": 1,
            "preserved_source_sha256": dict(PRESERVED_SOURCE_SHA256),
            "shaping": asdict(self.tuning.shaping),
            "physical_supervisor": self.supervisor.specification(),
            "constants": {name: getattr(self, name) for name in (
                "SPEED", "JUMP", "GRAVITY", "BRAKING_ACCELERATION", "POSITION_GAIN",
                "REACTION_SECONDS", "SETTLING_MARGIN", "MINIMUM_PULSE_EFFORT",
                "MINIMUM_UPWARD_LAUNCH_SPEED", "MAXIMUM_OPPOSING_LAUNCH_SPEED",
                "MINIMUM_HORIZONTAL_AIM", "MAXIMUM_HORIZONTAL_AIM", "INITIAL_AIM_ROOM_SCALE",
                "RECOVERY_AIM_ROOM_SCALE", "CEILING_MARGIN", "SETTLING_TIME_RESERVE",
                "MAXIMUM_PULSES", "ANTICIPATION_TICKS", "SUPPORT_DWELL_TICKS",
                "MAXIMUM_WINDOW_GAP_TICKS", "WATCHDOG_TICKS", "MAXIMUM_STALL_PROGRESS",
                "MAXIMUM_STALL_SPEED", "MINIMUM_DRIVE_REQUEST", "NEAR_GROUND_DISTANCE",
                "LOSS_DESCENT_TICKS", "MINIMUM_DESCENT_SPEED", "MAXIMUM_ARRIVAL_SPEED",
                "MAXIMUM_ARRIVAL_ANGLE", "QUIET_SPAN_TICKS")},
            "effort_limit": "max(.45, caller pace); never raise it to the minimum hop effort",
            "initial_launch": "fresh raw supported dwell plus six-tick anticipation; steer upward more steeply for short room",
            "second_launch": "one extra pulse only after observed cooldown acknowledgment and a new physical recovery window",
            "supported_watchdog": "18 fresh ticks of supported directed demand, <=.40 cells toward progress and <=6 cells/s current speed",
            "unsupported_edge": "18 fresh unsupported near-ground ticks, <=.40 cells displacement and <=6 cells/s current speed",
            "unexpected_descent": "observed supported-to-unsupported transition outside the launch interval; 12 fresh unsupported ticks, descent >=12 and below goal",
            "pulse_aim": "x=sign(dx)*clamp((abs(dx)-2)/room_scale,.05,.82); y=-sqrt(1-x*x)",
            "descent_aim": "reduce horizontal aim if needed to leave at least20 cells/s upward velocity within caller effort",
            "pulse_room": "positive analytic quadratic root; existing toward/upward motion, same-height flight plus conservative downward-height flight, supported stopping and reaction distance",
            "pulse_bounds": "upward result >=20 cells/s, ceiling apex plus6 cells, and remaining time minus.20 seconds settling reserve",
            "anticipation": "zero traction for six ticks; fresh condition and current room rechecked at launch",
            "ordinary_drive": "supported desired_x=clamp(2.5*dx,+/-48*effort); ground velocity servo; no free-air traction claim",
            "ordinary_posture": "lean=.16*desired_x/48; anticipation counterlean=.10; bounded .55 proportional/.20 damping trim with shared shaping",
            "arrival": "raw support, distance within caller tolerance, speed<=6, wrapped angle<=.32 for12 observed ticks with gaps<=6",
            "secondary_rebound": False,
            "host_mechanics_changed": False,
            "feedback": "only FeedbackObservation whitelist and explicit caller goal; no task evaluator or native mutation",
            "evidence": "local reasons and requested pulses are not actual impulses, airborne or recovered verdicts",
        }

    @property
    def specification_sha256(self) -> str:
        return sha256_json(self.specification())

    def reset(self) -> None:
        self.supervisor.reset()
        self._context = -1
        self._clear_motion()
        self._latest = Supervision("idle", "no_goal", 0, 0, 1, 0)
        self._phase, self._reason = "idle", "no_goal"
        self._diagnostics: dict = {}

    def _clear_motion(self) -> None:
        self._x.reset()
        self._turn.reset()
        self._history: deque[tuple[FeedbackObservation, float]] = deque()
        self._previous_observation: FeedbackObservation | None = None
        self._previous_move = 0.0
        self._support_since: int | None = None
        self._loss_since: int | None = None
        self._unexpected_loss = False
        self._quiet_since: int | None = None
        self._quiet_last: int | None = None
        self._pulse_count = 0
        self._last_pulse_tick: int | None = None
        self._pulse_acknowledged = False
        self._pending_reason: str | None = None
        self._pending_since: int | None = None
        self._pending_position: tuple[float, float] | None = None
        self._recovery_epoch = 0
        self._progress = 0.0
        self._stall_ticks = 0

    def _clear_windows(self) -> None:
        self._history.clear()
        self._support_since = self._loss_since = None
        self._unexpected_loss = False
        self._quiet_since = self._quiet_last = None
        self._pending_reason = self._pending_since = None
        self._pending_position = None
        self._previous_observation = None
        self._previous_move = 0.0
        self._progress = 0.0
        self._stall_ticks = 0

    def _frame(self, action: WireAction, distance: float | None) -> ControllerFrame:
        goal = self.supervisor.goal
        return ControllerFrame(action, self._phase, self._latest.status, self._reason,
                               goal.id if goal else None, self._x.rate_limited or self._turn.rate_limited,
                               distance, self.controller_id, self.specification_sha256)

    def diagnostic_state(self) -> dict:
        """Copy the latest decision's local reasons, never a mutable state handle."""
        return dict(self._diagnostics)

    def cancel(self, reason: str = "canceled") -> ControllerFrame:
        self._latest = self.supervisor.cancel(reason)
        self._clear_windows()
        self._x.cancel()
        self._turn.cancel()
        self._phase, self._reason = "coast", reason
        self._diagnostics = {"phase": self._phase, "reason": reason,
                             "supervisor_intervention": True, "supervisor_reason": reason,
                             "action_lease_ticks": 0, "pulse_reason": None}
        return self._frame(WireAction(effort=0.0), None)

    def _observe_motion(self, observation: FeedbackObservation, dx: float) -> None:
        previous = self._previous_observation
        if previous is not None and observation.tick - previous.tick > self.MAXIMUM_WINDOW_GAP_TICKS:
            self._clear_windows()
            previous = None
        if observation.supported is True:
            if previous is None or previous.supported is not True:
                self._support_since = observation.tick
            self._loss_since = None
            self._unexpected_loss = False
        else:
            self._support_since = None
            if observation.supported is False:
                if previous is None or previous.supported is not False:
                    self._loss_since = observation.tick
                    self._unexpected_loss = (previous is not None and previous.supported is True
                                             and (self._last_pulse_tick is None
                                                  or observation.tick - self._last_pulse_tick > 12))
            else:
                self._loss_since = None
                self._unexpected_loss = False
        self._history.append((observation, self._previous_move))
        while len(self._history) > 1 and self._history[1][0].tick <= observation.tick - self.WATCHDOG_TICKS:
            self._history.popleft()
        anchor = self._history[0][0]
        direction = math.copysign(1.0, dx)
        self._progress = direction * (observation.position[0] - anchor.position[0])
        self._stall_ticks = observation.tick - anchor.tick

    def _recovery_reason(self, observation: FeedbackObservation, goal: SkillGoal,
                         dx: float, dy: float) -> str | None:
        if math.hypot(dx, dy) <= goal.tolerance:
            return None
        full_window = self._stall_ticks >= self.WATCHDOG_TICKS
        direction = math.copysign(1.0, dx)
        slow = math.hypot(*observation.velocity) <= self.MAXIMUM_STALL_SPEED
        if observation.supported is True and full_window and slow:
            supported = all(view.supported is True for view, _ in self._history)
            demand = all(move * direction >= self.MINIMUM_DRIVE_REQUEST for _, move in list(self._history)[1:])
            if supported and demand and self._progress <= self.MAXIMUM_STALL_PROGRESS:
                return "supported_blocked_progress" if observation.blocked else "supported_no_progress"
        near_ground = observation.ground_distance is not None and observation.ground_distance <= self.NEAR_GROUND_DISTANCE
        if observation.supported is False and near_ground:
            if full_window and slow and all(view.supported is False for view, _ in self._history):
                anchor = self._history[0][0]
                if math.dist(observation.position, anchor.position) <= self.MAXIMUM_STALL_PROGRESS:
                    return "unsupported_edge_stall"
            if (self._unexpected_loss and self._loss_since is not None
                    and observation.tick - self._loss_since >= self.LOSS_DESCENT_TICKS
                    and observation.velocity[1] >= self.MINIMUM_DESCENT_SPEED
                    and dy < -self.SETTLING_MARGIN):
                return "unexpected_support_loss_descent"
        return None

    def _pending_recovery_still_observed(self, observation: FeedbackObservation, goal: SkillGoal,
                                         dx: float, dy: float) -> bool:
        """Recheck an actual recovery witness after its deliberate unload tick.

        Anticipation requests no traction, so it must not be mistaken for a
        brand-new failed-drive window. The original witness stays bounded by
        one fresh six-tick interval and still requires the same physical state.
        """
        if (self._pending_position is None or self._pending_since is None
                or observation.tick - self._pending_since > self.ANTICIPATION_TICKS
                or math.hypot(dx, dy) <= goal.tolerance):
            return False
        slow = math.hypot(*observation.velocity) <= self.MAXIMUM_STALL_SPEED
        little_motion = math.dist(observation.position, self._pending_position) <= self.MAXIMUM_STALL_PROGRESS
        if self._pending_reason in {"supported_blocked_progress", "supported_no_progress"}:
            return observation.supported is True and slow and little_motion
        near_ground = observation.ground_distance is not None and observation.ground_distance <= self.NEAR_GROUND_DISTANCE
        if self._pending_reason == "unsupported_edge_stall":
            return observation.supported is False and near_ground and slow and little_motion
        if self._pending_reason == "unexpected_support_loss_descent":
            return (observation.supported is False and near_ground and self._unexpected_loss
                    and observation.velocity[1] >= self.MINIMUM_DESCENT_SPEED and dy < -self.SETTLING_MARGIN)
        return False

    def _pulse(self, observation: FeedbackObservation, goal: SkillGoal, dx: float,
               dy: float, *, recovery: bool) -> tuple[tuple[float, float], float] | None:
        """Bound the declared host pulse by room, ceiling and remaining time."""
        room = abs(dx) - self.SETTLING_MARGIN
        toward = math.copysign(1.0, dx) * observation.velocity[0]
        if room <= 0 or toward < -self.MAXIMUM_OPPOSING_LAUNCH_SPEED:
            return None
        scale = self.RECOVERY_AIM_ROOM_SCALE if recovery else self.INITIAL_AIM_ROOM_SCALE
        horizontal = clamp(room / scale, self.MINIMUM_HORIZONTAL_AIM, self.MAXIMUM_HORIZONTAL_AIM)
        maximum_effort = max(.45, goal.pace)
        required_vertical = (max(0.0, observation.velocity[1]) + self.MINIMUM_UPWARD_LAUNCH_SPEED) / (self.JUMP * maximum_effort)
        if required_vertical >= 1.0:
            return None
        # Round inside the admitted upward envelope, rather than accepting a
        # just-below-threshold launch after two square-root round trips.
        horizontal = min(horizontal, max(0.0, math.sqrt(1.0 - required_vertical * required_vertical) - 1e-12))
        if horizontal < self.MINIMUM_HORIZONTAL_AIM:
            return None
        vertical = math.sqrt(1.0 - horizontal * horizontal)
        aim = (math.copysign(horizontal, dx), -vertical)
        velocity, upward = max(0.0, toward), max(0.0, -observation.velocity[1])
        extra_flight = math.sqrt(2.0 * max(0.0, dy) / self.GRAVITY)
        acceleration, gravity = self.BRAKING_ACCELERATION, self.GRAVITY
        a = 2 * horizontal * vertical / gravity + horizontal * horizontal / (2 * acceleration)
        b = (2 * (velocity * vertical + upward * horizontal) / gravity
             + velocity * horizontal / acceleration + (self.REACTION_SECONDS + extra_flight) * horizontal)
        c = (2 * velocity * upward / gravity + velocity * velocity / (2 * acceleration)
             + (self.REACTION_SECONDS + extra_flight) * velocity)
        if c >= room:
            return None
        remaining = room - c
        impulse = 2 * remaining / (b + math.sqrt(b * b + 4 * a * remaining))
        ceiling = observation.ceiling_distance
        if ceiling is None or ceiling <= self.CEILING_MARGIN:
            return None
        maximum_upward = math.sqrt(2 * gravity * (ceiling - self.CEILING_MARGIN))
        available_seconds = goal.timeout_seconds - self._latest.elapsed_ticks / 60 - self.SETTLING_TIME_RESERVE - extra_flight
        maximum_upward = min(maximum_upward, .5 * gravity * max(0.0, available_seconds))
        effort = min(maximum_effort, impulse / self.JUMP,
                     (maximum_upward - upward) / (self.JUMP * vertical))
        if (effort < self.MINIMUM_PULSE_EFFORT or observation.velocity[1] - self.JUMP * effort * vertical
                > -self.MINIMUM_UPWARD_LAUNCH_SPEED):
            return None
        return aim, effort

    def _quiet_arrival(self, observation: FeedbackObservation, goal: SkillGoal, distance: float) -> bool:
        quiet = (observation.supported is True and distance <= goal.tolerance
                 and math.hypot(*observation.velocity) <= self.MAXIMUM_ARRIVAL_SPEED
                 and abs(wrapped_angle(observation.angle)) <= self.MAXIMUM_ARRIVAL_ANGLE)
        if not quiet:
            self._quiet_since = self._quiet_last = None
            return False
        if (self._quiet_since is None or self._quiet_last is None
                or observation.tick - self._quiet_last > self.MAXIMUM_WINDOW_GAP_TICKS):
            self._quiet_since = observation.tick
        self._quiet_last = observation.tick
        return observation.tick - self._quiet_since >= self.QUIET_SPAN_TICKS

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
        # Fresh cooldown feedback remains useful during explicit emergency
        # gyro supervision; it must not be lost with ordinary motion windows.
        if (self._latest.status == "running" and self._last_pulse_tick is not None
                and observation.tick > self._last_pulse_tick and observation.cooldown is not None
                and observation.cooldown > 0):
            self._pulse_acknowledged = True
        pulse_reason = None
        distance = math.dist(observation.position, goal.target)
        if self._latest.override is not None:
            self._clear_windows()
            self._x.cancel()
            self._turn.cancel()
            self._phase = "emergency_recovery" if self._latest.status == "running" else "settled" if self._latest.status == "succeeded" else "coast"
            self._reason = self._latest.reason
            action = self._latest.override
        else:
            dx, dy = goal.target[0] - observation.position[0], goal.target[1] - observation.position[1]
            previous_support = self._previous_observation.supported if self._previous_observation else None
            self._observe_motion(observation, dx)
            if self._quiet_arrival(observation, goal, distance):
                self._latest = self.supervisor.complete()
                self._clear_windows()
                self._x.cancel()
                self._turn.cancel()
                self._phase, self._reason = "settled", "supported_arrival_estimate"
                action = WireAction(effort=0.0)
            else:
                effort = max(.45, goal.pace)
                maximum_speed = self.SPEED * effort
                desired_x = clamp(self.POSITION_GAIN * dx, -maximum_speed, maximum_speed)
                move = desired_x / maximum_speed if observation.supported is True else 0.0
                if (observation.supported is True and abs(dx) <= self.SETTLING_MARGIN
                        and abs(observation.velocity[0]) > self.MAXIMUM_ARRIVAL_SPEED):
                    move = -math.copysign(.1, observation.velocity[0])
                self._phase = ("settle" if distance <= goal.tolerance * 2 else "approach") \
                    if observation.supported is True else "airborne_coast"
                if observation.supported is True and previous_support is False:
                    self._phase = "landing"
                self._reason = "ordinary_control"
                recovery_reason = self._recovery_reason(observation, goal, dx, dy)
                if self._pending_recovery_still_observed(observation, goal, dx, dy):
                    recovery_reason = self._pending_reason
                ready = (self._pulse_count < self.MAXIMUM_PULSES and observation.cooldown is not None
                         and observation.cooldown <= 0 and observation.energy is not None and observation.energy >= .15
                         and (self._last_pulse_tick is None or observation.tick > self._last_pulse_tick + 6))
                initial = self._pulse_count == 0 and observation.supported is True and distance > goal.tolerance
                reason = "initial_supported_launch" if initial else recovery_reason
                if self._pulse_count and not self._pulse_acknowledged:
                    reason = None
                pulse = self._pulse(observation, goal, dx, dy, recovery=not initial) if ready and reason else None
                if pulse is None:
                    self._pending_reason = self._pending_since = None
                    self._pending_position = None
                elif self._pending_reason != reason:
                    self._pending_reason, self._pending_since = reason, observation.tick
                    self._pending_position = observation.position
                    if not initial:
                        self._recovery_epoch += 1
                support_ready = (observation.supported is True and self._support_since is not None
                                 and observation.tick - self._support_since >= self.SUPPORT_DWELL_TICKS)
                airborne_ready = observation.supported is False and reason in {
                    "unsupported_edge_stall", "unexpected_support_loss_descent"}
                primary = False
                aim = (math.copysign(.6, dx), -.8)
                if pulse is not None and self._pending_since is not None:
                    aim = pulse[0]
                    move = 0.0
                    self._reason = reason
                    contact = reason in {"supported_blocked_progress", "supported_no_progress"}
                    self._phase = "anticipate" if initial else "contact_recovery_anticipate" if contact else "support_loss_anticipate"
                    if (observation.tick - self._pending_since >= self.ANTICIPATION_TICKS
                            and (support_ready or airborne_ready)):
                        primary, effort, pulse_reason = True, pulse[1], reason
                        self._phase = "launch" if initial else "contact_recovery_launch" if contact else "support_loss_launch"
                        self._pulse_count += 1
                        self._last_pulse_tick = observation.tick
                        self._pulse_acknowledged = False
                        self._pending_reason = self._pending_since = None
                        self._pending_position = None
                        self._history.clear()
                lean = .16 * clamp(desired_x / self.SPEED)
                if "anticipate" in self._phase:
                    lean = -.10 * math.copysign(1.0, dx)
                elif distance <= goal.tolerance and math.hypot(*observation.velocity) <= self.MAXIMUM_ARRIVAL_SPEED:
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
        }
        return self._frame(action, distance)
