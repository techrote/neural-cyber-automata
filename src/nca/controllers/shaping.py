"""Bounded request shaping. These utilities never integrate or modify bodies.

The host still owns its gyro, connections, limits and loaded mechanical state.
The same shaper can be used for a human, scripted or learned wire request.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def unit(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*vector)
    return (vector[0] / length, vector[1] / length) if length > 1e-12 else (0.0, 0.0)


def bounded_vector(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*vector)
    return unit(vector) if length > 1.0 else vector


def wrapped_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class ShapeTuning:
    """Rates are normalized request units per simulated second."""

    quiet_below: float = 0.025
    activate_above: float = 0.055
    acceleration_rate: float = 8.0
    braking_rate: float = 12.0
    play_width: float = 0.025

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            finite(getattr(self, name), name)
        if not 0.0 <= self.quiet_below <= self.activate_above < 1.0:
            raise ValueError("deadband thresholds must be ordered inside [0, 1)")
        if not 0.0 <= self.play_width < 0.5:
            raise ValueError("play_width must be in [0, .5)")
        if not 0.0 < self.acceleration_rate <= 1000.0 or not 0.0 < self.braking_rate <= 1000.0:
            raise ValueError("slew rates must be positive and bounded")


class RequestShaper:
    """Schmitt deadband, deterministic play memory and asymmetric slew.

    Play is a bounded request hysteresis operator, not joint displacement. There
    is no accumulated integral demand that can wind up against a blocked actor.
    Cancellation bypasses stale motion immediately and keeps the play memory;
    reset clears all controller state. Neither operation edits physical state.
    """

    def __init__(self, tuning: ShapeTuning = ShapeTuning()) -> None:
        self.tuning = tuning
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.output = 0.0
        self.play = 0.0
        self.rate_limited = False

    def cancel(self) -> float:
        self.active = False
        self.output = 0.0
        self.rate_limited = False
        return 0.0

    def step(self, requested: float, dt: float) -> float:
        requested = clamp(finite(requested, "requested"))
        dt = finite(dt, "dt")
        if not 0.0 < dt <= 1.0:
            raise ValueError("dt must be in (0, 1] seconds")
        if abs(requested) <= self.tuning.quiet_below:
            self.active = False
        elif abs(requested) >= self.tuning.activate_above:
            self.active = True
        target = requested if self.active else 0.0
        if self.active:
            # The play operator takes up a bounded amount of reversed demand.
            self.play = clamp(self.play, target - self.tuning.play_width,
                              target + self.tuning.play_width)
            target = self.play
        braking = target * self.output < 0.0 or abs(target) < abs(self.output)
        rate = self.tuning.braking_rate if braking else self.tuning.acceleration_rate
        change = target - self.output
        limit = rate * dt
        self.rate_limited = abs(change) > limit + 1e-12
        self.output = clamp(self.output + clamp(change, -limit, limit))
        return self.output


class ContactLatch:
    """Fresh support is acquired after dwell, and lost immediately.

    Unknown/stale support never authorizes a new pulse. A controller may retain
    passive stance memory while it withholds its unsupported action.
    """

    def __init__(self, acquire_seconds: float = 0.05) -> None:
        self.acquire_seconds = finite(acquire_seconds, "acquire_seconds")
        if not 0.0 <= self.acquire_seconds <= 1.0:
            raise ValueError("support dwell must be in [0, 1]")
        self.reset()

    def reset(self) -> None:
        self.duration = 0.0
        self.supported = False

    def step(self, support: bool | None, dt: float) -> bool:
        dt = finite(dt, "dt")
        if not 0.0 < dt <= 1.0:
            raise ValueError("dt must be in (0, 1]")
        if support is not None and type(support) is not bool:
            raise ValueError("support must be bool or None")
        if support is not True:
            self.duration = 0.0
            self.supported = False
        else:
            self.duration += dt
            self.supported = self.duration + 1e-12 >= self.acquire_seconds
        return self.supported


def soft_stop_request(request: float, position: float, velocity: float,
                      lower: float, upper: float, width: float,
                      damping: float = 0.1) -> float:
    """Attenuate only outward requests near a declared travel boundary.

    This is an optional target mapper for an explicitly mapped joint channel.
    It is never automatically applied to a chassis trim or another joint. The
    host's final constraint and actuator limits remain authoritative.
    """
    values = {name: finite(value, name) for name, value in
              (("request", request), ("position", position), ("velocity", velocity),
               ("lower", lower), ("upper", upper), ("width", width), ("damping", damping))}
    if values["lower"] >= values["upper"] or not 0.0 < width <= (upper - lower) / 2.0:
        raise ValueError("soft region must fit the ordered limits")
    if damping < 0.0:
        raise ValueError("damping must be nonnegative")
    request = clamp(request)
    if request == 0.0:
        return 0.0
    sign = 1.0 if request > 0.0 else -1.0
    distance = upper - position if sign > 0.0 else position - lower
    scale = clamp(distance / width, 0.0, 1.0)
    # A damping request may slow or stop outward drive, never reverse it.
    magnitude = max(0.0, abs(request) * scale - max(0.0, velocity * sign) * damping * (1.0 - scale))
    return sign * magnitude
