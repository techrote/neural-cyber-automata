"""Authored Jolt/Vector skills over the shared ACT-004 wire boundary.

No policy mutates bodies, adds a private force/gyro, or reads evaluator records.
Skill success below is a local estimate, distinct from an evaluator verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping

from nca.contracts import WireAction
from .shaping import (ContactLatch, RequestShaper, ShapeTuning, bounded_vector,
                      clamp, finite, unit, wrapped_angle)


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain two finite values")
    return finite(value[0], name), finite(value[1], name)


@dataclass(frozen=True)
class FeedbackObservation:
    tick: int
    lifetime: str
    actor: str
    position: tuple[float, float]
    velocity: tuple[float, float]
    angle: float
    angular_velocity: float
    supported: bool | None
    cooldown: float | None = None
    energy: float | None = None
    ground_distance: float | None = None
    ceiling_distance: float | None = None
    blocked: bool = False
    valid: bool = True
    control_generation: int = 0
    controllable: bool = True

    def __post_init__(self) -> None:
        if type(self.tick) is not int or self.tick < 0 or not isinstance(self.lifetime, str) or not self.lifetime:
            raise ValueError("tick and lifetime must identify an issued observation")
        if type(self.control_generation) is not int or self.control_generation < 0:
            raise ValueError("control_generation must be nonnegative")
        if self.actor not in ("jev", "astra"):
            raise ValueError("only the declared jev and astra profiles are supported")
        object.__setattr__(self, "position", _pair(self.position, "position"))
        object.__setattr__(self, "velocity", _pair(self.velocity, "velocity"))
        finite(self.angle, "angle")
        finite(self.angular_velocity, "angular_velocity")
        for key in ("cooldown", "energy", "ground_distance", "ceiling_distance"):
            value = getattr(self, key)
            if value is not None and finite(value, key) < 0.0:
                raise ValueError(f"{key} must be nonnegative or unavailable")
        if self.supported is not None and type(self.supported) is not bool:
            raise ValueError("support must be bool or unavailable")
        if any(type(value) is not bool for value in (self.blocked, self.valid, self.controllable)):
            raise ValueError("validity fields must be booleans")

    @classmethod
    def from_host(cls, view: Mapping[str, Any]) -> "FeedbackObservation":
        """Copy a whitelist from an issued actor view, never a whole info dict.

        Coordinates use the pinned host's cells, seconds and clockwise radians
        (positive screen Y points down). No hidden nearest-target preprocessing
        or scene/evaluator fields are accepted as features.
        """
        if view.get("representation") != "articulated-rapier" or view.get("version") != 2:
            raise ValueError("expected an ACT-004 v2 issued actor observation")
        required = ("run_id", "epoch", "actor_id", "profile_version", "tick",
                    "observation_id", "position", "velocity", "rotation", "angular_velocity")
        if any(key not in view for key in required):
            raise ValueError("missing identity or proprioception in actor view")
        if type(view["profile_version"]) is not int or view["profile_version"] != 1:
            raise ValueError("baseline mapping requires the declared ACT-004 profile version 1")
        for key in ("epoch", "actor_id", "tick"):
            if type(view[key]) is not int or view[key] < (1 if key == "epoch" else 0):
                raise ValueError("host observation identity contains an invalid integer")
        if any(not isinstance(view[key], str) or not view[key] for key in ("run_id", "observation_id")):
            raise ValueError("host observation identity strings must be present")
        actor = str(view.get("model", view.get("rig_id", "")))
        if "rig_id" in view and view["rig_id"] != actor:
            raise ValueError("host model and rig identity disagree")
        lifetime = ":".join(str(view[key]) for key in ("run_id", "epoch", "actor_id", "profile_version")) + ":" + actor
        return cls(tick=view["tick"], lifetime=lifetime, actor=actor,
                   position=_pair(view["position"], "position"), velocity=_pair(view["velocity"], "velocity"),
                   angle=finite(view["rotation"], "rotation"),
                   angular_velocity=finite(view["angular_velocity"], "angular_velocity"),
                   supported=view.get("grounded"), cooldown=view.get("cooldown"), energy=view.get("energy"),
                   ground_distance=view.get("ground_distance"), ceiling_distance=view.get("ceiling_distance"),
                   blocked=view.get("blocked", False),
                   valid=not bool(view.get("fault", "")),
                   control_generation=view.get("control_generation", 0),
                   controllable=not bool(view.get("suspended", False)))


@dataclass(frozen=True)
class SkillGoal:
    id: str
    skill: str
    target: tuple[float, float] = (0.0, 0.0)
    pace: float = 0.8
    tolerance: float = 4.0
    timeout_seconds: float = 12.0

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or len(self.id) > 128:
            raise ValueError("goal id must be a short nonempty string")
        if self.skill not in ("directed_hop", "burst_traverse", "recover_level"):
            raise ValueError("unknown authored skill")
        object.__setattr__(self, "target", _pair(self.target, "target"))
        if not 0.0 < finite(self.pace, "pace") <= 1.0:
            raise ValueError("pace must be in (0, 1]")
        if not 0.1 <= finite(self.tolerance, "tolerance") <= 32.0:
            raise ValueError("tolerance must be in [.1, 32] host cells")
        if not 0.1 <= finite(self.timeout_seconds, "timeout_seconds") <= 120.0:
            raise ValueError("skill duration must be in [.1, 120] simulated seconds")


@dataclass(frozen=True)
class BaselineTuning:
    id: str = "balanced/v1"
    shaping: ShapeTuning = field(default_factory=ShapeTuning)
    base_hz: int = 60
    attitude_kp: float = 0.55
    attitude_kd: float = 0.20
    velocity_feedback: float = 0.45
    support_dwell: float = 0.05
    anticipation_seconds: float = 0.10
    pulse_period: float = 0.85
    recovery_enter_angle: float = 1.05
    recovery_exit_angle: float = 0.32
    recovery_enter_speed: float = 4.0
    recovery_exit_speed: float = 0.8
    recovery_dwell: float = 0.20
    stopping_acceleration: float = 100.0

    def __post_init__(self) -> None:
        if type(self.base_hz) is not int or not 1 <= self.base_hz <= 1000:
            raise ValueError("base_hz must be a bounded positive integer")
        if not self.id or not isinstance(self.shaping, ShapeTuning):
            raise ValueError("tuning identity and shape contract are required")
        for key in self.__dataclass_fields__:
            if key not in ("id", "shaping", "base_hz"):
                if finite(getattr(self, key), key) < 0.0:
                    raise ValueError(f"{key} must be nonnegative")
        if not 0.0 <= self.recovery_exit_angle < self.recovery_enter_angle <= math.pi:
            raise ValueError("recovery angle thresholds must be ordered")
        if not 0.0 <= self.recovery_exit_speed < self.recovery_enter_speed:
            raise ValueError("recovery speed thresholds must be ordered")
        if self.stopping_acceleration <= 0.0 or self.pulse_period < 0.1 or self.support_dwell > 1.0:
            raise ValueError("invalid bounded timing or braking tuning")

    @classmethod
    def named(cls, name: str) -> "BaselineTuning":
        if name == "firm":
            return cls(id="firm/v1", shaping=ShapeTuning(.01, .025, 12.0, 16.0, 0.0),
                       attitude_kp=.70, attitude_kd=.15, support_dwell=.025)
        if name == "balanced":
            return cls()
        if name == "compliant":
            return cls(id="compliant/v1", shaping=ShapeTuning(.045, .085, 5.0, 9.0, .05),
                       attitude_kp=.42, attitude_kd=.24, support_dwell=.08,
                       anticipation_seconds=.14, pulse_period=.95)
        raise ValueError("unknown tuning variant")

    @property
    def digest(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ControllerFrame:
    action: WireAction
    phase: str
    status: str
    reason: str
    goal_id: str | None
    response_limited: bool
    distance: float | None
    controller_id: str
    tuning_hash: str


class AuthoredController:
    """One bounded skill instance per actor; high-level choices are replaceable.

    Jolt's pulse is deliberately support-gated even though the host also permits
    airborne pulses. Vector's auxiliary mode toggle is never mistaken for a
    self-right button. Both use the actor owner's existing assistance.
    """

    SPEEDS = {"jev": 48.0, "astra": 61.0}

    def __init__(self, actor: str, tuning: BaselineTuning | None = None) -> None:
        if actor not in self.SPEEDS:
            raise ValueError("only jev and astra baselines have declared mappings")
        self.actor = actor
        self.tuning = tuning or BaselineTuning()
        self._x = RequestShaper(self.tuning.shaping)
        self._y = RequestShaper(self.tuning.shaping)
        self._turn = RequestShaper(self.tuning.shaping)
        self._support = ContactLatch(self.tuning.support_dwell)
        self.reset()

    @property
    def controller_id(self) -> str:
        return f"authored-feedback/{self.actor}/v1"

    def reset(self) -> None:
        self._x.reset()
        self._y.reset()
        self._turn.reset()
        self._support.reset()
        self._lifetime: str | None = None
        self._generation: int | None = None
        self._last_tick: int | None = None
        self._goal: SkillGoal | None = None
        self._retired: dict[str, SkillGoal] = {}
        self._elapsed = 0.0
        self._started_tick = 0
        self._next_pulse = 0.0
        self._phase = "idle"
        self._status = "idle"
        self._recovering = False
        self._recovery_quiet = 0.0
        self._arrival_quiet = 0.0
        self._pulse_last = False
        self._braking = False
        self._rebound = False
        self._reason = "no_goal"

    def cancel(self, reason: str = "canceled") -> ControllerFrame:
        """Call host cancellation too; a neutral request is not owner revocation."""
        if self._goal is not None:
            self._retired[self._goal.id] = self._goal
        self._status = "canceled"
        self._phase = "coast"
        self._reason = reason
        self._pulse_last = False
        self._next_pulse = self._elapsed + self.tuning.anticipation_seconds
        for shaper in (self._x, self._y, self._turn):
            shaper.cancel()
        return self._frame(WireAction(effort=0.0), None)

    def _frame(self, action: WireAction, distance: float | None) -> ControllerFrame:
        return ControllerFrame(action, self._phase, self._status, self._reason,
                               self._goal.id if self._goal else None,
                               any(item.rate_limited for item in (self._x, self._y, self._turn)),
                               distance, self.controller_id, self.tuning.digest)

    def _start(self, goal: SkillGoal, tick: int) -> None:
        if goal.id in self._retired:
            if self._retired[goal.id] != goal:
                raise ValueError("reused goal id has different parameters")
            return
        if goal.skill == "directed_hop" and self.actor != "jev":
            raise ValueError("directed_hop requires the declared Jolt mapping")
        if goal.skill == "burst_traverse" and self.actor != "astra":
            raise ValueError("burst_traverse requires the declared Vector mapping")
        if len(self._retired) >= 1024:
            raise ValueError("skill identity capacity reached; start a new episode")
        if self._goal is not None:
            self._retired[self._goal.id] = self._goal
        self._goal = goal
        self._elapsed = 0.0
        self._started_tick = tick
        self._arrival_quiet = 0.0
        self._next_pulse = self.tuning.anticipation_seconds
        self._pulse_last = False
        self._status = "running"
        self._phase = "anticipate"
        self._reason = "local_skill_estimate"

    def step(self, observation: FeedbackObservation, goal: SkillGoal) -> ControllerFrame:
        if observation.actor != self.actor:
            raise ValueError("observation belongs to another actor profile")
        if self._lifetime != observation.lifetime:
            self.reset()
            self._lifetime = observation.lifetime
        if goal.skill == "directed_hop" and self.actor != "jev":
            raise ValueError("directed_hop requires the declared Jolt mapping")
        if goal.skill == "burst_traverse" and self.actor != "astra":
            raise ValueError("burst_traverse requires the declared Vector mapping")
        if self._goal is not None and self._goal.id == goal.id and self._goal != goal:
            raise ValueError("same goal id cannot change its parameters")
        if goal.id in self._retired and self._retired[goal.id] != goal:
            raise ValueError("same retired goal id cannot change parameters")
        if self._last_tick is not None and observation.tick <= self._last_tick:
            raise ValueError("feedback observations must advance monotonically")
        elapsed_ticks = 1 if self._last_tick is None else observation.tick - self._last_tick
        dt = elapsed_ticks / self.tuning.base_hz
        self._last_tick = observation.tick
        generation_changed = self._generation is not None and self._generation != observation.control_generation
        self._generation = observation.control_generation
        if generation_changed:
            canceled = self.cancel("owner_generation_changed")
            if self._goal is None or goal.id == self._goal.id:
                return canceled
            # A separately identified fresh invocation may begin after fencing.
            dt = 1.0 / self.tuning.base_hz
        if dt > 1.0 and self._status in {"idle", "canceled", "succeeded", "timed_out"} \
                and (self._goal is None or goal.id != self._goal.id):
            dt = 1.0 / self.tuning.base_hz
        if dt > 1.0 or not observation.valid or not observation.controllable:
            return self.cancel("observation_gap" if dt > 1.0 else "unavailable_observation")
        if self._goal is None or self._goal.id != goal.id:
            if goal.id in self._retired:
                raise ValueError("retired goal cannot resume after a new invocation")
            self._start(goal, observation.tick)
        if self._status != "running":
            return self._frame(WireAction(effort=0.0), None)
        self._elapsed = (observation.tick - self._started_tick) / self.tuning.base_hz
        if self._elapsed >= goal.timeout_seconds:
            self.cancel("skill_deadline")
            self._status = "timed_out"
            return self._frame(WireAction(effort=0.0), None)
        support = self._support.step(observation.supported, dt)
        angle = wrapped_angle(observation.angle)
        omega = observation.angular_velocity
        if abs(angle) >= self.tuning.recovery_enter_angle or abs(omega) >= self.tuning.recovery_enter_speed:
            self._recovering = True
            self._recovery_quiet = 0.0
        if self._recovering:
            quiet = abs(angle) <= self.tuning.recovery_exit_angle and abs(omega) <= self.tuning.recovery_exit_speed
            self._recovery_quiet = self._recovery_quiet + dt if quiet else 0.0
            if self._recovery_quiet + 1e-12 >= self.tuning.recovery_dwell:
                self._recovering = False
                self._next_pulse = max(self._next_pulse, self._elapsed + self.tuning.anticipation_seconds)
        dx, dy = goal.target[0] - observation.position[0], goal.target[1] - observation.position[1]
        distance = math.hypot(dx, dy)
        speed = math.hypot(*observation.velocity)
        if goal.skill == "recover_level":
            arrived = abs(angle) <= self.tuning.recovery_exit_angle and abs(omega) <= self.tuning.recovery_exit_speed
        else:
            arrived = distance <= goal.tolerance and speed <= 6.0 and not self._recovering
        self._arrival_quiet = self._arrival_quiet + dt if arrived else 0.0
        if self._arrival_quiet + 1e-12 >= self.tuning.recovery_dwell:
            self._status = "succeeded"
            self._phase = "settled"
            self._reason = "target_reached_estimate"
            self._retired[goal.id] = goal
            for shaper in (self._x, self._y, self._turn):
                shaper.cancel()
            return self._frame(WireAction(effort=0.0), distance)
        if self._recovering or goal.skill == "recover_level":
            self._phase = "recovery"
            requested = WireAction(effort=.9, trim=clamp(-angle * self.tuning.attitude_kp - omega * self.tuning.attitude_kd),
                                   secondary=self.actor == "astra", auxiliary=self.actor == "jev")
        elif self.actor == "jev":
            requested = self._jolt(observation, goal, dx, dy, distance, support)
        else:
            requested = self._vector(observation, goal, dx, dy, distance)
        self._pulse_last = requested.primary
        move = bounded_vector((self._x.step(requested.move[0], dt), self._y.step(requested.move[1], dt)))
        action = WireAction(move=move, aim=requested.aim, effort=requested.effort,
                            trim=self._turn.step(requested.trim, dt),
                            primary=requested.primary, secondary=requested.secondary,
                            auxiliary=requested.auxiliary)
        return self._frame(action, distance)

    def _ready(self, observation: FeedbackObservation) -> bool:
        return (not self._pulse_last and self._elapsed + 1e-12 >= self._next_pulse
                and observation.cooldown is not None and observation.cooldown <= 0.0
                and observation.energy is not None and observation.energy >= .15)

    def _jolt(self, observation: FeedbackObservation, goal: SkillGoal,
              dx: float, dy: float, distance: float, support: bool) -> WireAction:
        desired_speed = min(self.SPEEDS[self.actor] * goal.pace,
                            math.sqrt(2.0 * self.tuning.stopping_acceleration * abs(dx)))
        desired_x = math.copysign(desired_speed, dx) if abs(dx) > goal.tolerance else 0.0
        move = clamp(desired_x / self.SPEEDS[self.actor] + self.tuning.velocity_feedback *
                     (desired_x - observation.velocity[0]) / self.SPEEDS[self.actor])
        if not support:
            move *= .45
        aim = unit((clamp(dx / 30.0), min(-.55, dy / 30.0 - .55)))
        pulse = support and self._ready(observation) and distance > goal.tolerance * 1.5
        if observation.ceiling_distance is not None and observation.ceiling_distance < 10.0:
            pulse = False
        if pulse:
            self._next_pulse = self._elapsed + self.tuning.pulse_period
            self._phase = "launch"
        else:
            self._phase = "stance" if support else "airborne"
        near_floor = support or (observation.ground_distance is not None and observation.ground_distance <= 12.0)
        if observation.velocity[1] > 12.0 and near_floor and distance > goal.tolerance * 2.0:
            self._rebound = True
        elif observation.velocity[1] < 2.0 or not near_floor or distance <= goal.tolerance * 2.0:
            self._rebound = False
        lean = .16 * clamp(desired_x / self.SPEEDS[self.actor])
        trim = clamp(wrapped_angle(lean - observation.angle) * self.tuning.attitude_kp
                     - observation.angular_velocity * self.tuning.attitude_kd, -.85, .85)
        return WireAction(move=(move, 0.0), aim=aim, effort=max(.45, goal.pace),
                          trim=trim, primary=pulse, secondary=self._rebound)

    def _vector(self, observation: FeedbackObservation, goal: SkillGoal,
                dx: float, dy: float, distance: float) -> WireAction:
        direction = unit((dx, dy))
        desired_speed = min(self.SPEEDS[self.actor] * goal.pace,
                            math.sqrt(2.0 * self.tuning.stopping_acceleration * max(0.0, distance - goal.tolerance)))
        desired = (direction[0] * desired_speed, direction[1] * desired_speed)
        move = bounded_vector(tuple(clamp(desired[i] / self.SPEEDS[self.actor] +
                              self.tuning.velocity_feedback * (desired[i] - observation.velocity[i]) /
                              self.SPEEDS[self.actor]) for i in range(2)))
        along_speed = sum(observation.velocity[i] * direction[i] for i in range(2))
        speed = math.hypot(*observation.velocity)
        stopping_distance = speed * speed / (2.0 * self.tuning.stopping_acceleration)
        if (along_speed > desired_speed + 8.0 or along_speed < -8.0 or
                (distance < stopping_distance + goal.tolerance * 1.5 and speed > 10.0)):
            self._braking = True
        elif speed < max(6.0, desired_speed * .8) and along_speed >= -2.0:
            self._braking = False
        pulse = (self._ready(observation) and not self._braking and distance > 24.0
                 and along_speed < desired_speed * .7 and abs(wrapped_angle(observation.angle)) < .7)
        if direction[1] < 0.0 and observation.ceiling_distance is not None and observation.ceiling_distance < 14.0:
            pulse = False
        if pulse:
            self._next_pulse = self._elapsed + self.tuning.pulse_period
        self._phase = "counterthrust" if self._braking else "burst" if pulse else "traverse"
        trim = clamp(-wrapped_angle(observation.angle) * self.tuning.attitude_kp
                     - observation.angular_velocity * self.tuning.attitude_kd, -.75, .75)
        return WireAction(move=move, aim=direction, effort=max(.45, goal.pace), trim=trim,
                          primary=pulse, secondary=self._braking)
