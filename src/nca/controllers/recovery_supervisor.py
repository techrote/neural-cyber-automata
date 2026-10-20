"""Minimal Jolt physical supervision, separate from ordinary movement.

The supervisor never chooses a target, launch, path, posture style or traction
request. It fences control context and can apply one explicitly requested,
bounded emergency self-right operation. A neutral frame does not revoke an
already issued native command: the caller must still cancel the host owner.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from nca.contracts import WireAction
from .baseline import FeedbackObservation, SkillGoal
from .shaping import clamp, wrapped_angle


@dataclass(frozen=True)
class Supervision:
    status: str
    reason: str
    context: int
    elapsed_ticks: int
    sample_ticks: int
    action_lease_ticks: int
    override: WireAction | None = None
    intervention: bool = False


class PhysicalRecoverySupervisor:
    """Fresh-context admission plus a declared emergency attitude envelope.

    ``admit`` must be called once per new observation. ``current_tick`` is the
    caller's current simulation clock; omitted means synchronous consumption
    at the observation tick. An original observation is never re-dated. A
    subsequent dispatcher must preserve the returned observation-anchored
    lease, and still use the actual native owner/observation checks.
    """

    BASE_HZ = 60
    MAXIMUM_HELPER_LEASE_TICKS = 12
    MAXIMUM_SAMPLE_GAP_TICKS = 60
    MAXIMUM_DWELL_SAMPLE_GAP_TICKS = 6
    EMERGENCY_ANGLE = 1.20
    EMERGENCY_ANGULAR_SPEED = 4.5
    EMERGENCY_EXIT_ANGLE = .32
    EMERGENCY_EXIT_SPEED = .8
    EMERGENCY_QUIET_TICKS = 12
    MAXIMUM_EMERGENCY_TICKS = 48

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._lifetime: str | None = None
        self._retired_lifetimes: set[str] = set()
        self._generation: int | None = None
        self._last_tick: int | None = None
        self._last_current_tick: int | None = None
        self._goal: SkillGoal | None = None
        self._retired: dict[str, SkillGoal] = {}
        self._started_tick = 0
        self._context = 0
        self._status = "idle"
        self._reason = "no_goal"
        self._emergency_since: int | None = None
        self._emergency_quiet_since: int | None = None

    def specification(self) -> dict:
        return {
            "schema": "nca-physical-recovery-supervisor/v1",
            "base_hz": self.BASE_HZ,
            "maximum_helper_lease_ticks": self.MAXIMUM_HELPER_LEASE_TICKS,
            "maximum_sample_gap_ticks": self.MAXIMUM_SAMPLE_GAP_TICKS,
            "emergency": "unstable_attitude must be explicitly requested and observed",
            "emergency_enter_angle": self.EMERGENCY_ANGLE,
            "emergency_enter_angular_speed": self.EMERGENCY_ANGULAR_SPEED,
            "emergency_exit_angle": self.EMERGENCY_EXIT_ANGLE,
            "emergency_exit_angular_speed": self.EMERGENCY_EXIT_SPEED,
            "emergency_quiet_ticks": self.EMERGENCY_QUIET_TICKS,
            "maximum_dwell_sample_gap_ticks": self.MAXIMUM_DWELL_SAMPLE_GAP_TICKS,
            "maximum_emergency_ticks": self.MAXIMUM_EMERGENCY_TICKS,
            "emergency_request": "auxiliary self-right; trim clamp(-angle*.55-omega*.20); effort .9",
            "ordinary_action": "passed through without target, launch or posture decisions",
            "expiry": "exclusive original observation lease, original goal deadline and active emergency deadline",
            "authority": "retire canceled/replaced/completed goals and prior lifetimes; reject rollback",
            "host_cancellation": "caller must revoke owner; neutral output alone is not cancellation",
        }

    @property
    def goal(self) -> SkillGoal | None:
        return self._goal

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str = "canceled", *, status: str = "canceled") -> Supervision:
        if self._goal is not None:
            self._retired[self._goal.id] = self._goal
        self._status, self._reason = status, reason
        self._emergency_since = self._emergency_quiet_since = None
        return Supervision(status, reason, self._context, 0, 1, 0,
                           WireAction(effort=0.0), True)

    def complete(self) -> Supervision:
        return replace(self.cancel("supported_arrival_estimate", status="succeeded"), intervention=False)

    def _begin(self, goal: SkillGoal, tick: int) -> None:
        if len(self._retired) >= 1024:
            self.cancel("goal_identity_capacity")
            raise ValueError("start a new episode before the goal identity capacity is exhausted")
        if self._goal is not None:
            self._retired[self._goal.id] = self._goal
        self._goal, self._started_tick = goal, tick
        self._status, self._reason = "running", "ordinary_control"
        self._emergency_since = self._emergency_quiet_since = None
        self._context += 1

    def admit(self, observation: FeedbackObservation, goal: SkillGoal, *,
              current_tick: int | None = None, emergency: str | None = None) -> Supervision:
        if not isinstance(observation, FeedbackObservation) or observation.actor != "jev":
            self.cancel("invalid_feedback_profile")
            raise ValueError("physical supervision requires the declared Jolt feedback profile")
        if not isinstance(goal, SkillGoal):
            self.cancel("invalid_goal")
            raise ValueError("physical supervision requires an explicit skill goal")
        now = observation.tick if current_tick is None else current_tick
        if type(now) is not int or now < observation.tick:
            self.cancel("invalid_current_clock")
            raise ValueError("the current simulation clock cannot precede its observation")
        if emergency not in (None, "unstable_attitude"):
            self.cancel("undeclared_emergency")
            raise ValueError("the requested emergency is not in the fixed supervisor contract")
        if observation.lifetime in self._retired_lifetimes:
            self.cancel("retired_lifetime")
            raise ValueError("a retired physical lifetime cannot resume")
        if self._lifetime != observation.lifetime:
            retired = set(self._retired_lifetimes)
            if self._lifetime is not None:
                retired.add(self._lifetime)
            context = self._context
            self.reset()
            self._retired_lifetimes, self._context = retired, context
            self._lifetime = observation.lifetime
        if self._goal is not None and goal.id == self._goal.id and goal != self._goal:
            self.cancel("changed_goal_parameters")
            raise ValueError("the same goal identity cannot change its parameters")
        if goal.id in self._retired:
            if self._retired[goal.id] != goal or self._goal is None or self._goal.id != goal.id:
                self.cancel("retired_goal")
                raise ValueError("a retired goal cannot change or resume after another invocation")
        if self._last_tick is not None and observation.tick <= self._last_tick:
            self.cancel("stale_feedback")
            raise ValueError("feedback must advance strictly; duplicates cannot rearm an action")
        if self._last_current_tick is not None and now < self._last_current_tick:
            self.cancel("current_clock_rollback")
            raise ValueError("the current simulation clock cannot move backwards")
        sample_ticks = 1 if self._last_tick is None else observation.tick - self._last_tick
        if self._generation is not None and observation.control_generation < self._generation:
            self.cancel("owner_generation_rollback")
            raise ValueError("owner control generation cannot move backwards")
        changed = self._generation is not None and self._generation != observation.control_generation
        self._last_tick, self._generation = observation.tick, observation.control_generation
        self._last_current_tick = now
        if changed:
            canceled = self.cancel("owner_generation_changed")
            if self._goal is None or self._goal.id == goal.id:
                return canceled
        new_goal = self._goal is None or self._goal.id != goal.id
        if new_goal:
            self._begin(goal, observation.tick)
            sample_ticks = 1
        if not observation.valid or not observation.controllable:
            return self.cancel("unavailable_observation")
        if now - observation.tick >= self.MAXIMUM_HELPER_LEASE_TICKS:
            return self.cancel("expired_observation")
        if sample_ticks > self.MAXIMUM_SAMPLE_GAP_TICKS:
            return self.cancel("observation_gap")
        elapsed = now - self._started_tick
        deadline = self._started_tick + math.ceil(goal.timeout_seconds * self.BASE_HZ - 1e-12)
        if self._status != "running":
            return Supervision(self._status, self._reason, self._context, elapsed,
                               sample_ticks, 0, WireAction(effort=0.0), False)
        if now >= deadline:
            return self.cancel("skill_deadline", status="timed_out")
        lease = min(self.MAXIMUM_HELPER_LEASE_TICKS, deadline - observation.tick)
        angle, omega = wrapped_angle(observation.angle), observation.angular_velocity
        unstable = abs(angle) >= self.EMERGENCY_ANGLE or abs(omega) >= self.EMERGENCY_ANGULAR_SPEED
        if emergency == "unstable_attitude" and unstable and self._emergency_since is None:
            self._emergency_since = observation.tick
            self._emergency_quiet_since = None
        if self._emergency_since is not None:
            quiet = abs(angle) <= self.EMERGENCY_EXIT_ANGLE and abs(omega) <= self.EMERGENCY_EXIT_SPEED
            if not quiet:
                self._emergency_quiet_since = None
            elif self._emergency_quiet_since is None or sample_ticks > self.MAXIMUM_DWELL_SAMPLE_GAP_TICKS:
                self._emergency_quiet_since = observation.tick
            if (self._emergency_quiet_since is not None
                    and observation.tick - self._emergency_quiet_since >= self.EMERGENCY_QUIET_TICKS):
                self._emergency_since = self._emergency_quiet_since = None
            elif now - self._emergency_since >= self.MAXIMUM_EMERGENCY_TICKS:
                return self.cancel("emergency_recovery_deadline")
            else:
                action = WireAction(effort=.9, trim=clamp(-angle * .55 - omega * .20), auxiliary=True)
                lease = min(lease, self._emergency_since + self.MAXIMUM_EMERGENCY_TICKS - observation.tick)
                return Supervision("running", "declared_unstable_attitude", self._context,
                                   elapsed, sample_ticks, lease, action, True)
        return Supervision("running", "ordinary_control", self._context, elapsed,
                           sample_ticks, lease)
