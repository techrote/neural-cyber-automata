"""A separately versioned, contact-aware Jolt launch and arrival reference.

The flight estimate is a bounded admission heuristic over the pinned gameplay
profile, not simulated replacement physics. Rapier still owns every impulse,
contact and body. Local arrival remains an estimate from issued observations;
it is never substituted for the evaluator's per-tick supported-arrival verdict.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import math
from pathlib import Path

from nca.contracts import WireAction, sha256_json
from .baseline import (AuthoredController, BaselineTuning, ControllerFrame,
                       FeedbackObservation, SkillGoal)
from .shaping import bounded_vector, clamp, unit, wrapped_angle


BASELINE_SOURCE_SHA256 = "19929806429698977cdba82b855c52d6362d1dd2bb4e2aa7161e41bb1bb3f210"
CONTROLLER_ID = "authored-jolt-arrival/v1"


class JoltArrivalController(AuthoredController):
    """One supported launch, followed by traction-based approach and settling.

    Only the declared Jolt directed-hop mapping is implemented. The preserved
    v1 controller supplies shaping, frame construction and cancellation
    mechanics; this implementation owns its new control and arrival rules. It
    has no access to an environment, task evaluator, dataset or checkpoint.
    """

    SPEED = 48.0
    JUMP = 86.0
    GRAVITY = 156.0
    BRAKING_ACCELERATION = 100.0
    POSITION_GAIN = 2.5
    REACTION_SECONDS = 0.1
    SETTLING_MARGIN = 2.0
    MINIMUM_PULSE_EFFORT = 0.35
    MINIMUM_UPWARD_LAUNCH_SPEED = 2.0
    MAXIMUM_OPPOSING_LAUNCH_SPEED = 6.0
    MAXIMUM_ARRIVAL_SPEED = 6.0
    MAXIMUM_ARRIVAL_ANGLE = 0.32
    QUIET_SPAN_TICKS = 12
    MAXIMUM_QUIET_SAMPLE_GAP_TICKS = 6
    NEAR_GOAL_COUNTER_REQUEST = 0.1

    def __init__(self, actor: str = "jev") -> None:
        if actor != "jev":
            raise ValueError("the arrival reference implements only the declared Jolt profile")
        inherited = Path(__file__).with_name("baseline.py")
        if hashlib.sha256(inherited.read_bytes()).hexdigest() != BASELINE_SOURCE_SHA256:
            raise ValueError("the arrival reference's pinned shaping/lifecycle dependency changed")
        super().__init__(actor, BaselineTuning())

    @property
    def controller_id(self) -> str:
        return CONTROLLER_ID

    def specification(self) -> dict:
        """Complete fixed numeric/control interpretation, safe for canonical JSON."""
        return {
            "schema": "nca-jolt-arrival-reference/v1",
            "controller_id": self.controller_id,
            "actor": "jev", "skill": "directed_hop", "profile_version": 1,
            "baseline_source_sha256": BASELINE_SOURCE_SHA256,
            "shaping_and_recovery": asdict(self.tuning),
            "speed_cells_per_second": self.SPEED,
            "jump_cells_per_second": self.JUMP,
            "gravity_cells_per_second_squared": self.GRAVITY,
            "braking_cells_per_second_squared": self.BRAKING_ACCELERATION,
            "position_gain_per_second": self.POSITION_GAIN,
            "reaction_seconds": self.REACTION_SECONDS,
            "settling_margin_cells": self.SETTLING_MARGIN,
            "maximum_pulses_per_invocation": 1,
            "minimum_pulse_effort": self.MINIMUM_PULSE_EFFORT,
            "maximum_drive_and_pulse_effort": "max(0.45, caller pace)",
            "minimum_upward_launch_speed": self.MINIMUM_UPWARD_LAUNCH_SPEED,
            "maximum_opposing_launch_speed": self.MAXIMUM_OPPOSING_LAUNCH_SPEED,
            "pulse_aim": "unit(clamp(dx/30), min(-0.55, dy/30-0.55))",
            "pulse_room": "abs(dx)-2; positive toward velocity and upward velocity only; "
                          "same-height ballistic travel + supported stopping distance + reaction travel",
            "pulse_effort": "positive analytic quadratic root for available room, capped by caller effort; "
                            "withhold if below minimum or actual post-pulse vertical speed is not upward",
            "airborne_horizontal_request": 0.0,
            "supported_drive": "desired_x=clamp(position_gain*dx, +/-speed*effort); "
                               "request=desired_x/(speed*effort); inside settling margin with "
                               "abs(vx)>arrival_speed request -sign(vx)*near_goal_counter_request",
            "near_goal_counter_request": self.NEAR_GOAL_COUNTER_REQUEST,
            "secondary_rebound": False,
            "arrival_maximum_speed": self.MAXIMUM_ARRIVAL_SPEED,
            "arrival_maximum_angle": self.MAXIMUM_ARRIVAL_ANGLE,
            "arrival_requires_raw_support": True,
            "arrival_quiet_observed_span_ticks": self.QUIET_SPAN_TICKS,
            "arrival_maximum_sample_gap_ticks": self.MAXIMUM_QUIET_SAMPLE_GAP_TICKS,
            "arrival_observation_rule": "first qualifying sample starts the span; nonqualifying or "
                                        "over-gap samples reset it; this is a local estimate only",
            "feedback": "issued own-root proprioception, support, cooldown, energy and ceiling ray; "
                        "explicit caller goal and deadline; no evaluator inputs",
            "lifecycle": "retire canceled/completed/replaced goals; reject duplicate/stale ticks and "
                         "generation rollback; new lifetime resets state and retires prior lifetime",
            "owner_cancellation": "caller must separately revoke the host lease",
        }

    @property
    def specification_sha256(self) -> str:
        return sha256_json(self.specification())

    def _frame(self, action: WireAction, distance: float | None) -> ControllerFrame:
        return replace(super()._frame(action, distance), tuning_hash=self.specification_sha256)

    def reset(self) -> None:
        super().reset()
        self._pulse_used = False
        self._quiet_since_tick: int | None = None
        self._quiet_last_tick: int | None = None
        self._retired_lifetimes: set[str] = set()

    def cancel(self, reason: str = "canceled") -> ControllerFrame:
        self._quiet_since_tick = None
        self._quiet_last_tick = None
        self._support.reset()
        return super().cancel(reason)

    def _start(self, goal: SkillGoal, tick: int) -> None:
        super()._start(goal, tick)
        self._pulse_used = False
        self._quiet_since_tick = None
        self._quiet_last_tick = None
        self._support.reset()
        for shaper in (self._x, self._y, self._turn):
            shaper.reset()

    def _quiet_arrival(self, observation: FeedbackObservation, distance: float,
                       speed: float, angle: float) -> bool:
        quiet = (observation.supported is True and distance <= self._goal.tolerance
                 and speed <= self.MAXIMUM_ARRIVAL_SPEED
                 and abs(angle) <= self.MAXIMUM_ARRIVAL_ANGLE and not self._recovering)
        if not quiet:
            self._quiet_since_tick = self._quiet_last_tick = None
            return False
        if (self._quiet_since_tick is None or self._quiet_last_tick is None
                or observation.tick - self._quiet_last_tick > self.MAXIMUM_QUIET_SAMPLE_GAP_TICKS):
            self._quiet_since_tick = observation.tick
        self._quiet_last_tick = observation.tick
        return observation.tick - self._quiet_since_tick >= self.QUIET_SPAN_TICKS

    def _pulse_effort(self, observation: FeedbackObservation, dx: float,
                      aim: tuple[float, float], maximum_effort: float) -> float | None:
        """Solve a conservative horizontal-room quadratic without sampling a sweep."""
        room = abs(dx) - self.SETTLING_MARGIN
        direction = math.copysign(1.0, dx)
        toward = direction * observation.velocity[0]
        if room <= 0.0 or toward < -self.MAXIMUM_OPPOSING_LAUNCH_SPEED:
            return None
        # Ignoring beneficial opposite/downward velocity overestimates travel.
        velocity = max(0.0, toward)
        upward = max(0.0, -observation.velocity[1])
        horizontal, vertical = abs(aim[0]), max(0.0, -aim[1])
        if horizontal <= 1e-12 or vertical <= 1e-12:
            return None
        acceleration, gravity = self.BRAKING_ACCELERATION, self.GRAVITY
        a = 2.0 * horizontal * vertical / gravity + horizontal * horizontal / (2.0 * acceleration)
        b = (2.0 * (velocity * vertical + upward * horizontal) / gravity
             + velocity * horizontal / acceleration + self.REACTION_SECONDS * horizontal)
        c = (2.0 * velocity * upward / gravity + velocity * velocity / (2.0 * acceleration)
             + self.REACTION_SECONDS * velocity)
        if c >= room:
            return None
        remaining = room - c
        # This positive-root form avoids cancellation when the remaining room
        # is small compared with existing motion. The host sees only effort.
        impulse = 2.0 * remaining / (b + math.sqrt(b * b + 4.0 * a * remaining))
        effort = min(maximum_effort, impulse / self.JUMP)
        if (effort < self.MINIMUM_PULSE_EFFORT
                or observation.velocity[1] + self.JUMP * effort * aim[1]
                > -self.MINIMUM_UPWARD_LAUNCH_SPEED):
            return None
        return effort

    def _request(self, observation: FeedbackObservation, goal: SkillGoal,
                 dx: float, dy: float, support: bool) -> WireAction:
        effort = max(.45, goal.pace)
        maximum_speed = self.SPEED * effort
        desired_x = clamp(self.POSITION_GAIN * dx, -maximum_speed, maximum_speed)
        move = desired_x / maximum_speed if observation.supported is True else 0.0
        if (observation.supported is True and abs(dx) <= self.SETTLING_MARGIN
                and abs(observation.velocity[0]) > self.MAXIMUM_ARRIVAL_SPEED):
            move = -math.copysign(self.NEAR_GOAL_COUNTER_REQUEST, observation.velocity[0])
        aim = unit((clamp(dx / 30.0), min(-.55, dy / 30.0 - .55)))
        pulse_effort = None
        if (observation.supported is True and support and not self._pulse_used and self._ready(observation)
                and (observation.ceiling_distance is None or observation.ceiling_distance >= 10.0)):
            pulse_effort = self._pulse_effort(observation, dx, aim, effort)
        if pulse_effort is not None:
            self._pulse_used = True
            self._phase = "launch"
            effort = pulse_effort
            self._next_pulse = self._elapsed + self.tuning.pulse_period
        else:
            self._phase = ("settle" if abs(dx) <= goal.tolerance * 2.0 else "approach") \
                if observation.supported is True else "airborne_coast"
        lean = .16 * clamp(desired_x / self.SPEED)
        trim = clamp(wrapped_angle(lean - observation.angle) * self.tuning.attitude_kp
                     - observation.angular_velocity * self.tuning.attitude_kd, -.85, .85)
        return WireAction(move=(move, 0.0), aim=aim, effort=effort, trim=trim,
                          primary=pulse_effort is not None, secondary=False)

    def step(self, observation: FeedbackObservation, goal: SkillGoal) -> ControllerFrame:
        if not isinstance(observation, FeedbackObservation) or observation.actor != "jev":
            raise ValueError("arrival feedback requires the declared Jolt profile")
        if not isinstance(goal, SkillGoal) or goal.skill != "directed_hop":
            raise ValueError("arrival reference requires an explicit directed-hop goal")
        if observation.lifetime in self._retired_lifetimes:
            self.cancel("retired_lifetime")
            raise ValueError("a retired physical lifetime cannot resume")
        if self._lifetime != observation.lifetime:
            retired = set(self._retired_lifetimes)
            if self._lifetime is not None:
                retired.add(self._lifetime)
            self.reset()
            self._retired_lifetimes = retired
            self._lifetime = observation.lifetime
        if self._goal is not None and self._goal.id == goal.id and self._goal != goal:
            self.cancel("changed_goal_parameters")
            raise ValueError("the same goal identity cannot change its parameters")
        if goal.id in self._retired:
            if self._retired[goal.id] != goal:
                raise ValueError("the same retired goal identity cannot change its parameters")
            if self._goal is None or self._goal.id != goal.id:
                raise ValueError("a retired goal cannot resume after a new invocation")
        if self._last_tick is not None and observation.tick <= self._last_tick:
            self.cancel("stale_feedback")
            raise ValueError("arrival feedback must advance monotonically")
        elapsed_ticks = 1 if self._last_tick is None else observation.tick - self._last_tick
        if self._generation is not None and observation.control_generation < self._generation:
            self.cancel("owner_generation_rollback")
            raise ValueError("owner generation cannot move backwards")
        generation_changed = (self._generation is not None
                              and observation.control_generation != self._generation)
        self._generation = observation.control_generation
        self._last_tick = observation.tick
        if generation_changed:
            canceled = self.cancel("owner_generation_changed")
            if self._goal is None or goal.id == self._goal.id:
                return canceled
            elapsed_ticks = 1
        if elapsed_ticks > self.tuning.base_hz or not observation.valid or not observation.controllable:
            return self.cancel("observation_gap" if elapsed_ticks > self.tuning.base_hz else "unavailable_observation")
        if self._goal is None or self._goal.id != goal.id:
            self._start(goal, observation.tick)
            elapsed_ticks = 1
        if self._status != "running":
            return self._frame(WireAction(effort=0.0), None)
        self._elapsed = (observation.tick - self._started_tick) / self.tuning.base_hz
        if self._elapsed >= goal.timeout_seconds:
            self.cancel("skill_deadline")
            self._status = "timed_out"
            return self._frame(WireAction(effort=0.0), None)
        dt = elapsed_ticks / self.tuning.base_hz
        support = self._support.step(observation.supported, dt)
        angle, omega = wrapped_angle(observation.angle), observation.angular_velocity
        if abs(angle) >= self.tuning.recovery_enter_angle or abs(omega) >= self.tuning.recovery_enter_speed:
            self._recovering = True
            self._recovery_quiet = 0.0
        if self._recovering:
            quiet = abs(angle) <= self.tuning.recovery_exit_angle and abs(omega) <= self.tuning.recovery_exit_speed
            self._recovery_quiet = self._recovery_quiet + dt if quiet else 0.0
            if self._recovery_quiet + 1e-12 >= self.tuning.recovery_dwell:
                self._recovering = False
        dx, dy = goal.target[0] - observation.position[0], goal.target[1] - observation.position[1]
        distance, speed = math.hypot(dx, dy), math.hypot(*observation.velocity)
        if self._quiet_arrival(observation, distance, speed, angle):
            self._status, self._phase, self._reason = "succeeded", "settled", "supported_arrival_estimate"
            self._retired[goal.id] = goal
            for shaper in (self._x, self._y, self._turn):
                shaper.cancel()
            return self._frame(WireAction(effort=0.0), distance)
        if self._recovering:
            self._phase = "recovery"
            requested = WireAction(effort=.9, trim=clamp(-angle * self.tuning.attitude_kp
                                                        - omega * self.tuning.attitude_kd), auxiliary=True)
        else:
            requested = self._request(observation, goal, dx, dy, support)
        self._pulse_last = requested.primary
        move = bounded_vector((self._x.step(requested.move[0], dt), self._y.step(requested.move[1], dt)))
        # A zero desired air request must not retain the shaper's prior ground
        # demand. It still cannot revoke a host lease or apply an air brake.
        if observation.supported is not True:
            self._x.cancel()
            move = (0.0, 0.0)
        return self._frame(WireAction(move=move, aim=requested.aim, effort=requested.effort,
                                      trim=self._turn.step(requested.trim, dt), primary=requested.primary,
                                      secondary=requested.secondary, auxiliary=requested.auxiliary), distance)
