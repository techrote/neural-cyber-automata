"""Predeclared directed-hop task outcomes, reward terms and style measurements.

This evaluator owns its goal and metrics. Policies receive only the separately
declared caller goal and local policy view, never this evaluator object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from nca.contracts import WireAction
from nca.controllers.baseline import FeedbackObservation, SkillGoal
from nca.controllers.shaping import wrapped_angle


REWARD_SCHEMA = "jolt-arrival-best-progress/v1"


@dataclass
class HopTask:
    goal: SkillGoal
    initial: Mapping[str, Any]
    deadline_ticks: int = 240
    quiet_ticks_required: int = 12
    elapsed_ticks: int = 0
    quiet_ticks: int = 0
    succeeded: bool = False
    deadline_missed: bool = False
    integrity_failure: bool = False
    return_sum: float = 0.0
    first_response_tick: int | None = None
    success_tick: int | None = None
    primary_edges: int = 0
    purposeful_launch: bool = False
    trim_reversals: int = 0
    ordinary_ticks: int = 0
    request_variation: float = 0.0
    maximum_toward_speed: float = 0.0
    maximum_overshoot: float = 0.0
    airborne_ticks: int = 0
    absolute_angle_sum: float = 0.0
    maximum_angular_speed: float = 0.0
    failed_admissions: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)
    inference_ns: list[int] = field(default_factory=list)

    def __post_init__(self):
        initial = FeedbackObservation.from_host(self.initial)
        self._initial_position = initial.position
        self._initial_tick = initial.tick
        self._last_tick = initial.tick
        self._best_distance = math.dist(initial.position, self.goal.target)
        self._last_distance = self._best_distance
        self._previous_action = WireAction(effort=0.0)
        self._trim_sign = 0
        self._launch_tick = None
        self._last_cooldown = initial.cooldown
        self._last_consumed_primary = bool(self.initial.get("action", {}).get("primary", False))
        self._direction = math.copysign(1.0, self.goal.target[0] - initial.position[0])
        self._target_dx = abs(self.goal.target[0] - initial.position[0])
        self.samples.append(self._sample(initial, None))

    def _sample(self, observation: FeedbackObservation, action: WireAction | None) -> dict[str, Any]:
        return {"tick": observation.tick, "position": list(observation.position),
                "velocity": list(observation.velocity), "angle": observation.angle,
                "angular_velocity": observation.angular_velocity, "supported": observation.supported,
                "action": None if action is None else action.to_dict()}

    @property
    def done(self) -> bool:
        return self.succeeded or self.deadline_missed or self.integrity_failure

    def consume(self, action: WireAction, ticks: Sequence[Mapping[str, Any]], *,
                inference_ns: int = 0) -> dict[str, float]:
        if self.done:
            raise ValueError("task outcome cannot be rewarded twice")
        if not ticks:
            raise ValueError("task reward needs actual completed host receipts")
        terms = {"progress": 0.0, "time": 0.0, "effort_proxy": 0.0,
                 "wobble": 0.0, "trim_variation": 0.0, "success": 0.0, "deadline": 0.0}
        self.inference_ns.append(inference_ns)
        ordinary_interval = True
        actual_count = 0
        for receipt in ticks:
            if receipt.get("failed") or receipt.get("tick_before") != self._last_tick:
                self.integrity_failure = True
                raise ValueError("invalid or noncontiguous physical receipt cannot become task progress")
            view = FeedbackObservation.from_host(receipt["observation"])
            if not view.valid or view.tick != self._last_tick + 1:
                self.integrity_failure = True
                raise ValueError("invalid task observation")
            self._last_tick = view.tick
            self.elapsed_ticks = view.tick - self._initial_tick
            actual_count += 1
            self.failed_admissions += sum(item.get("accepted") is not True for item in receipt.get("admission", ()))
            if self.failed_admissions:
                self.integrity_failure = True
            applied = receipt.get("applied", {})
            consumed = applied.get("action") if applied.get("action_available") is True else None
            if isinstance(consumed, Mapping):
                primary = consumed.get("primary") is True
                edge = primary and not self._last_consumed_primary
                self.primary_edges += int(edge)
                impulse = applied.get("linear_impulse")
                confirmed = (edge and self._last_cooldown is not None and self._last_cooldown <= 1 / 60 + 1e-6
                             and view.cooldown is not None and view.cooldown >= .60
                             and isinstance(impulse, (tuple, list)) and len(impulse) == 2 and impulse[1] < -1e-6
                             and applied.get("energy_used", 0) > 0)
                if confirmed:
                    self._launch_tick = self.elapsed_ticks
                self._last_consumed_primary = primary
            self._last_cooldown = view.cooldown
            distance = math.dist(view.position, self.goal.target)
            best = min(self._best_distance, distance)
            # One-time approach budget. Going away and returning cannot earn
            # the same progress again, even with discounted future rewards.
            terms["progress"] += (self._best_distance - best) / 64.0
            self._best_distance, self._last_distance = best, distance
            toward_speed = self._direction * view.velocity[0]
            self.maximum_toward_speed = max(self.maximum_toward_speed, toward_speed)
            if self.first_response_tick is None and toward_speed >= 2:
                self.first_response_tick = self.elapsed_ticks
            progress = self._direction * (view.position[0] - self._initial_position[0])
            self.maximum_overshoot = max(self.maximum_overshoot, progress - self._target_dx)
            angle = abs(wrapped_angle(view.angle))
            self.absolute_angle_sum += angle
            self.maximum_angular_speed = max(self.maximum_angular_speed, abs(view.angular_velocity))
            if view.supported is False:
                self.airborne_ticks += 1
                if self._launch_tick is not None and 0 <= self.elapsed_ticks - self._launch_tick <= 12:
                    self.purposeful_launch = True
            ordinary = (view.supported is True and not any((action.primary, action.secondary, action.auxiliary))
                        and view.cooldown is not None and view.cooldown <= 0)
            ordinary_interval &= ordinary
            self.ordinary_ticks += int(ordinary)
            speed = math.hypot(*view.velocity)
            quiet = distance <= self.goal.tolerance and speed <= 6 and angle <= .32 and view.supported is True
            self.quiet_ticks = self.quiet_ticks + 1 if quiet else 0
            terms["time"] -= .05 / 60
            terms["effort_proxy"] -= .005 / 60 * (abs(action.move[0]) + abs(action.trim)) * action.effort
            terms["wobble"] -= .02 / 60 * max(0.0, angle - .32)
            if self.quiet_ticks >= self.quiet_ticks_required and not self.succeeded:
                self.succeeded = True
                self.success_tick = self.elapsed_ticks
                terms["success"] = 2.0
            if self.elapsed_ticks >= self.deadline_ticks and not self.succeeded:
                self.deadline_missed = True
                terms["deadline"] = -.5
            # The wire repeat is a declared 6-tick decision. If success happens
            # during it, score it once, preserve the remaining actual motion,
            # and let the caller terminate before issuing another request.
        variation = abs(action.trim - self._previous_action.trim)
        self.request_variation += variation
        sign = 1 if action.trim >= .08 else -1 if action.trim <= -.08 else 0
        if ordinary_interval:
            if sign and self._trim_sign and sign != self._trim_sign:
                self.trim_reversals += 1
            if sign:
                self._trim_sign = sign
            terms["trim_variation"] = -.01 * variation
        else:
            self._trim_sign = 0
        self._previous_action = action
        self.samples.append(self._sample(view, action))
        terms["total"] = sum(terms.values())
        self.return_sum += terms["total"]
        return terms

    def metrics(self, *, administratively_capped: bool = False) -> dict[str, Any]:
        def quantile(values, fraction):
            if not values:
                return None
            ordered = sorted(values)
            position = (len(ordered) - 1) * fraction
            low, high = math.floor(position), math.ceil(position)
            return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

        return {"success": self.succeeded and not self.integrity_failure,
                "reason": "integrity_failure" if self.integrity_failure else "arrival" if self.succeeded
                          else "task_deadline" if self.deadline_missed else "administrative_cutoff",
                "administratively_capped": administratively_capped,
                "integrity_failure": self.integrity_failure, "failed_admissions": self.failed_admissions,
                "elapsed_ticks": self.elapsed_ticks, "return": self.return_sum,
                "final_distance_cells": self._last_distance, "best_distance_cells": self._best_distance,
                "success_seconds": None if self.success_tick is None else self.success_tick / 60,
                "capped_completion_seconds": self.success_tick / 60 if self.succeeded else self.deadline_ticks / 60,
                "response_seconds": None if self.first_response_tick is None else self.first_response_tick / 60,
                "peak_toward_speed_cells_per_second": self.maximum_toward_speed,
                "overshoot_cells": self.maximum_overshoot, "primary_edges": self.primary_edges,
                "purposeful_launch": self.purposeful_launch,
                "airborne_fraction": self.airborne_ticks / max(1, self.elapsed_ticks),
                "ordinary_trim_reversals": self.trim_reversals,
                "ordinary_trim_reversals_per_second": self.trim_reversals / max(1 / 60, self.ordinary_ticks / 60),
                "trim_variation_per_second": self.request_variation / max(1 / 60, self.elapsed_ticks / 60),
                "mean_absolute_angle_radians": self.absolute_angle_sum / max(1, self.elapsed_ticks),
                "peak_angular_speed": self.maximum_angular_speed,
                "inference_p50_ms": None if not self.inference_ns else quantile(self.inference_ns, .5) / 1e6,
                "inference_p95_ms": None if not self.inference_ns else quantile(self.inference_ns, .95) / 1e6,
                "inference_max_ms": None if not self.inference_ns else max(self.inference_ns) / 1e6,
                "owner_feel_accepted": False}
