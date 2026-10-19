"""Small, bounded transport primitives; no simulation or optional ML imports."""

from __future__ import annotations

import json
import math
import socket
from collections.abc import Mapping
from typing import Any

PROTOCOL = "nca-host/1"
MAX_PACKET_BYTES = 1_048_576
MAX_REPEAT_TICKS = 18
DEFAULT_DECISION_TICKS = 6
DEFAULT_LEASE_TICKS = 12


class HostError(RuntimeError):
    """The host could not complete a valid request."""


class ProtocolError(HostError):
    """Transport, schema or correlation failed; the channel is unusable."""


class HostIntegrityError(HostError):
    """An attempted game tick failed; this is not an ordinary learning sample."""


class EpisodeFinished(HostError):
    """Reset is required; no action can advance a replacement episode implicitly."""


def owned_json(value: Any) -> Any:
    """Return an independent finite JSON value, rejecting ambiguous encodings."""
    return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))


def _reject_constant(value: str) -> None:
    raise ProtocolError(f"nonfinite JSON constant: {value}")


def _unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode_packet(data: bytes) -> dict[str, Any]:
    if not data or len(data) > MAX_PACKET_BYTES:
        raise ProtocolError("empty or oversized packet")
    try:
        value = json.loads(data.decode("utf-8"), parse_constant=_reject_constant,
                           object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ProtocolError("invalid JSON packet") from exc
    if not isinstance(value, dict):
        raise ProtocolError("packet must be an object")
    # JSON exponent overflow is not routed through parse_constant.
    try:
        owned_json(value)
    except (ValueError, RecursionError) as exc:
        raise ProtocolError("packet contains nonfinite or excessively nested data") from exc
    return value


def encode_packet(value: Mapping[str, Any]) -> bytes:
    try:
        data = json.dumps(dict(value), allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ProtocolError("packet is not finite JSON") from exc
    if len(data) > MAX_PACKET_BYTES:
        raise ProtocolError("oversized packet")
    return data + b"\n"


class JsonChannel:
    """One ordered request/response stream with bounded buffering and a deadline."""

    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.buffer = bytearray()

    def send(self, value: Mapping[str, Any]) -> None:
        try:
            self.connection.sendall(encode_packet(value))
        except OSError as exc:
            raise ProtocolError("host transport write failed") from exc

    def receive(self) -> dict[str, Any]:
        try:
            while b"\n" not in self.buffer:
                packet = self.connection.recv(min(65536, MAX_PACKET_BYTES + 1 - len(self.buffer)))
                if not packet:
                    raise ProtocolError("host transport closed before a complete response")
                self.buffer.extend(packet)
                if len(self.buffer) > MAX_PACKET_BYTES and b"\n" not in self.buffer:
                    raise ProtocolError("oversized host response")
            line, _, rest = self.buffer.partition(b"\n")
            self.buffer = bytearray(rest)
            return decode_packet(bytes(line))
        except OSError as exc:
            raise ProtocolError("host response deadline or transport failure") from exc


def integer(value: Any, minimum: int, maximum: int, name: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def make_physical_command(view: Mapping[str, Any], sequence: int,
                          action: Mapping[str, Any], lease_ticks: int = DEFAULT_LEASE_TICKS) -> dict[str, Any]:
    """Preserve the ACT004 v2 packet; never refresh an old observation identity."""
    integer(sequence, 0, 9_007_199_254_740_991, "sequence")
    integer(lease_ticks, 1, 18, "lease_ticks")
    required = ("run_id", "actor_id", "epoch", "tick", "observation_id")
    if any(key not in view for key in required):
        raise ValueError("command requires an issued host observation")
    return {"version": 2, "kind": "physical_wire_action", "run_id": view["run_id"],
            "actor_id": view["actor_id"], "epoch": view["epoch"], "sequence": sequence,
            "observed_tick": view["tick"], "observation_id": view["observation_id"],
            "lease_ticks": lease_ticks, "action": owned_json(dict(action))}


def make_sampled_command(view: Mapping[str, Any], sequence: int,
                         action: Mapping[str, Any], lease_ticks: int = DEFAULT_LEASE_TICKS,
                         *, tag: bool = False, cancel: bool = False) -> dict[str, Any]:
    """Preserve ACT003 names, reserved channels and the configured wire/metal lane."""
    result = make_physical_command(view, sequence, action, lease_ticks)
    result.update(version=1, kind="actuator_action" if view.get("control_mode") == "metal" else "rig_action",
                  tag=tag, message=[], message_type="none", cancel=cancel)
    return result


class RewardDifference:
    """Convert a cumulative owner score once per strictly increasing state tick."""

    def __init__(self) -> None:
        self.run_id: str | None = None
        self.tick = -1
        self.total = 0.0

    def reset(self, run_id: str, tick: int, total: float) -> None:
        if not math.isfinite(float(total)):
            raise ProtocolError("nonfinite cumulative reward")
        self.run_id, self.tick, self.total = run_id, tick, float(total)

    def consume(self, run_id: str, tick: int, total: float) -> float:
        if run_id != self.run_id or tick <= self.tick or not math.isfinite(float(total)):
            raise ProtocolError("reward state is stale, reordered, foreign or nonfinite")
        increment = float(total) - self.total
        self.tick, self.total = tick, float(total)
        return increment
