"""Strict, immutable experiment contracts, separate from host wire packets.

The host continues to own integration, contacts and material changes. These
types describe permitted requests and copied evidence; they are not a simulator.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import types
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields, replace
from typing import Any, ClassVar, Literal, Union, get_args, get_origin, get_type_hints


class ContractError(ValueError):
    """A machine-readable contract refusal with a useful field-level message."""

    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> None:
    raise ContractError(code, detail)


def _plain(value: Any) -> Any:
    if isinstance(value, Record):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _freeze_json(value: Any, path: str = "value", depth: int = 0) -> Any:
    if depth > 64:
        _fail("nesting_limit", path)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            _fail("invalid_type", f"{path} object keys must be strings")
        return types.MappingProxyType({key: _freeze_json(item, f"{path}.{key}", depth + 1)
                                       for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[{i}]", depth + 1)
                     for i, item in enumerate(value))
    _fail("invalid_json", f"{path} is not finite JSON")


def canonical_json(value: Any) -> str:
    """Stable UTF-8 JSON, refusing NaN, duplicate keys on read, and foreign types."""
    plain = _plain(value)
    _freeze_json(plain)
    return json.dumps(plain, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def strict_json_loads(source: str | bytes, *, max_bytes: int = 1_048_576) -> Any:
    if not isinstance(source, (str, bytes)):
        _fail("invalid_type", "JSON input must be text or bytes")
    size = len(source.encode("utf-8")) if isinstance(source, str) else len(source)
    if size > max_bytes:
        _fail("size_limit", f"JSON exceeds {max_bytes} bytes")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("duplicate_field", key)
            result[key] = value
        return result

    try:
        value = json.loads(source, object_pairs_hook=object_pairs,
                           parse_constant=lambda value: _fail("nonfinite", value))
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise ContractError("invalid_json", str(exc)) from exc
    _freeze_json(value)
    return value


def _convert(annotation: Any, value: Any, path: str) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        return _freeze_json(value, path)
    if origin is Literal:
        if any(type(value) is type(item) and value == item for item in args):
            return value
        _fail("unsupported_value", f"{path}: expected one of {args!r}, got {value!r}")
    if origin in (Union, types.UnionType):
        for member in args:
            try:
                return _convert(member, value, path)
            except ContractError:
                pass
        _fail("invalid_type", f"{path} does not match its declared alternatives")
    if origin is tuple:
        if not isinstance(value, (tuple, list)):
            _fail("invalid_type", f"{path} must be an array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], item, f"{path}[{i}]")
                         for i, item in enumerate(value))
        if len(value) != len(args):
            _fail("shape_mismatch", f"{path}: expected {len(args)} values, got {len(value)}")
        return tuple(_convert(arg, item, f"{path}[{i}]")
                     for i, (arg, item) in enumerate(zip(args, value)))
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            _fail("invalid_type", f"{path} must be an object")
        return types.MappingProxyType({_convert(args[0], key, path):
                                       _convert(args[1], item, f"{path}.{key}")
                                       for key, item in value.items()})
    if annotation is float:
        if type(value) not in (int, float):
            _fail("invalid_type", f"{path} must be numeric, not a boolean")
        try:
            normalized = float(value)
        except OverflowError:
            _fail("nonfinite", path)
        if not math.isfinite(normalized):
            _fail("nonfinite", path)
        return normalized
    if annotation in (str, int, bool, type(None)):
        if type(value) is not annotation:
            _fail("invalid_type", f"{path} must be {annotation.__name__}")
        return value
    if isinstance(annotation, type) and issubclass(annotation, Record):
        if isinstance(value, annotation):
            return value
        if isinstance(value, Mapping):
            return annotation.from_dict(value)
    _fail("invalid_type", f"{path} must match {annotation}")


class Record:
    """Dataclass codec. Every constructor and JSON decoder validates identically."""

    _type_cache: ClassVar[dict[type, dict[str, Any]]] = {}

    def __post_init__(self) -> None:
        cls = type(self)
        if cls not in self._type_cache:
            self._type_cache[cls] = get_type_hints(cls)
        for field in fields(self):
            object.__setattr__(self, field.name, _convert(
                self._type_cache[cls][field.name], getattr(self, field.name), field.name))
        self.validate()

    def validate(self) -> None:
        pass

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if not isinstance(data, Mapping):
            _fail("invalid_type", f"{cls.__name__} must be an object")
        declared = {field.name: field for field in fields(cls)}
        if extra := set(data) - set(declared):
            _fail("unknown_field", f"{cls.__name__}: {sorted(extra, key=str)!r}")
        missing = [name for name, field in declared.items() if name not in data
                   and field.default is MISSING and field.default_factory is MISSING]
        if missing:
            _fail("missing_field", f"{cls.__name__}: {missing}")
        return cls(**dict(data))

    @classmethod
    def from_json(cls, source: str | bytes):
        return cls.from_dict(strict_json_loads(source))

    def to_dict(self) -> dict[str, Any]:
        return {field.name: _plain(getattr(self, field.name)) for field in fields(self)}

    def to_json(self) -> str:
        return canonical_json(self)

    @property
    def sha256(self) -> str:
        return sha256_json(self)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Structural JSON Schema; constructors additionally enforce semantic rules."""
        definitions: dict[str, Any] = {}

        def describe(annotation: Any) -> dict[str, Any]:
            origin, args = get_origin(annotation), get_args(annotation)
            if annotation is Any:
                return {}
            if origin is Literal:
                return {"enum": list(args)}
            if origin in (Union, types.UnionType):
                return {"anyOf": [describe(member) for member in args]}
            if origin is tuple:
                if len(args) == 2 and args[1] is Ellipsis:
                    return {"type": "array", "items": describe(args[0])}
                return {"type": "array", "prefixItems": [describe(arg) for arg in args],
                        "minItems": len(args), "maxItems": len(args)}
            if origin in (dict, Mapping):
                return {"type": "object", "additionalProperties": describe(args[1])}
            primitive = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}
            if annotation in primitive:
                return {"type": primitive[annotation]}
            if isinstance(annotation, type) and issubclass(annotation, Record):
                name = annotation.__name__
                if name not in definitions:
                    definitions[name] = {}
                    hints = get_type_hints(annotation)
                    descriptions = {field.name: describe(hints[field.name]) for field in fields(annotation)}
                    required = [field.name for field in fields(annotation)
                                if field.default is MISSING and field.default_factory is MISSING]
                    definitions[name] = {"type": "object", "additionalProperties": False,
                                         "properties": descriptions, "required": required}
                return {"$ref": f"#/$defs/{name}"}
            raise TypeError(f"No schema mapping for {annotation}")

        root = describe(cls)
        return {"$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": cls.__name__,
                "description": "Structural contract. The Python decoder also validates bounds, topology, masks, identity and hashes.",
                **root, "$defs": definitions}


def identifier(value: str, name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
        _fail("invalid_identifier", name)


def digest(value: str, name: str) -> None:
    if not re.fullmatch(r"[a-f0-9]{64}", value):
        _fail("invalid_digest", name)


def nonnegative(value: float | int, name: str) -> None:
    if value < 0:
        _fail("out_of_range", f"{name} must be nonnegative")


def unique(values: tuple[str, ...], name: str) -> None:
    if len(set(values)) != len(values):
        _fail("duplicate_id", name)


@dataclass(frozen=True)
class ControlCapabilities(Record):
    """Independent axes; central decisions do not imply privileged perception."""

    actuation_lane: Literal["wire", "direct"] = "wire"
    decision_placement: Literal["local", "central"] = "local"
    perception: Literal["local", "global"] = "local"
    coordination: Literal["independent", "messages", "shared_state"] = "independent"
    training_privilege: Literal["none", "privileged_teacher", "privileged_critic"] = "none"


@dataclass(frozen=True)
class Part(Record):
    id: str
    collider: Literal["box", "circle", "capsule", "convex", "host_authored"]
    frame: str = "actor_local"

    def validate(self) -> None:
        identifier(self.id, "part.id")
        identifier(self.frame, "part.frame")


@dataclass(frozen=True)
class Joint(Record):
    id: str
    parent: str
    child: str
    kind: Literal["rotary", "slide", "spring", "emulated"]
    lower: float
    upper: float
    unit: str
    play: float = 0.0
    compliance: float = 0.0
    damping: float = 0.0
    emulation: str = "host_authored"

    def validate(self) -> None:
        identifier(self.id, "joint.id")
        if self.parent == self.child:
            _fail("invalid_topology", "joint cannot connect a part to itself")
        if self.lower > self.upper:
            _fail("out_of_range", "joint lower exceeds upper")
        for name in ("play", "compliance", "damping"):
            nonnegative(getattr(self, name), f"joint.{name}")
        if not self.unit or not self.emulation:
            _fail("missing_semantics", "joint requires units and emulation identity")


@dataclass(frozen=True)
class Sensor(Record):
    id: str
    unit: str
    frame: str
    lower: float
    upper: float
    privilege: Literal["local", "global"] = "local"
    cadence_ticks: int = 6

    def validate(self) -> None:
        identifier(self.id, "sensor.id")
        if self.lower > self.upper or self.cadence_ticks < 1:
            _fail("out_of_range", f"sensor {self.id}")
        if not self.unit or not self.frame:
            _fail("missing_semantics", "sensor requires unit and frame")


@dataclass(frozen=True)
class Channel(Record):
    id: str
    part: str
    unit: str = "normalized"
    frame: str = "actor_local"
    lower: float = -1.0
    upper: float = 1.0
    neutral: float = 0.0
    response_seconds: float = 0.0
    slew_per_second: float | None = None
    held_behavior: Literal["hold_target", "pulse", "edge"] = "hold_target"
    released_behavior: Literal["neutral", "coast", "host_authored"] = "neutral"

    def validate(self) -> None:
        identifier(self.id, "channel.id")
        if not self.lower <= self.neutral <= self.upper:
            _fail("out_of_range", f"channel {self.id}: neutral outside bounds")
        nonnegative(self.response_seconds, "channel.response_seconds")
        if self.slew_per_second is not None and self.slew_per_second <= 0:
            _fail("out_of_range", "channel.slew_per_second must be positive")
        if not self.unit or not self.frame:
            _fail("missing_semantics", "channel requires unit and frame")


@dataclass(frozen=True)
class Assist(Record):
    """Authored assistance; no finite gyro momentum ledger is implied."""

    kind: Literal["none", "bounded_gyro", "host_authored"] = "none"
    effort_limit: float = 0.0
    damping: float = 0.0
    deadzone: float = 0.0
    hysteresis: float = 0.0
    ramp_seconds: float = 0.0
    cancellation: Literal["clear_pending_keep_mechanics"] = "clear_pending_keep_mechanics"
    reset: Literal["reconstruct_episode"] = "reconstruct_episode"

    def validate(self) -> None:
        for name in ("effort_limit", "damping", "deadzone", "hysteresis", "ramp_seconds"):
            nonnegative(getattr(self, name), f"assist.{name}")
        if self.kind == "none" and self.effort_limit != 0:
            _fail("invalid_assist", "an absent assist cannot supply effort")


@dataclass(frozen=True)
class GameplayProfile(Record):
    id: str
    version: int
    host_revision: str
    representation: Literal["recorded_fixture", "sampled_ACT003", "physical_ACT004"]
    parts: tuple[Part, ...]
    joints: tuple[Joint, ...]
    sensors: tuple[Sensor, ...]
    channels: tuple[Channel, ...]
    channel_mask: tuple[bool, ...]
    assist: Assist = Assist()
    capabilities: ControlCapabilities = ControlCapabilities()
    schema_version: Literal[1] = 1
    part_cap: int = 8
    base_hz: int = 60
    decision_ticks: int = 6
    default_lease_ticks: int = 12
    maximum_lease_ticks: int = 18

    def validate(self) -> None:
        identifier(self.id, "profile.id")
        if self.version < 1:
            _fail("invalid_version", "profile version must be positive")
        if not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", self.host_revision):
            _fail("invalid_revision", "profile host_revision must be a full source revision")
        if not 1 <= self.part_cap <= 8 or not 1 <= len(self.parts) <= self.part_cap:
            _fail("part_cap", "profile must contain 1..part_cap parts, cap <= 8")
        if not self.channels or len(self.channel_mask) != len(self.channels):
            _fail("shape_mismatch", "profile channel mask differs from channels")
        for name in ("parts", "joints", "sensors", "channels"):
            unique(tuple(item.id for item in getattr(self, name)), name)
        part_ids = {part.id for part in self.parts}
        for joint in self.joints:
            if joint.parent not in part_ids or joint.child not in part_ids:
                _fail("invalid_topology", f"joint {joint.id} refers to an absent part")
        for channel in self.channels:
            if channel.part not in part_ids:
                _fail("invalid_topology", f"channel {channel.id} refers to an absent part")
        if self.base_hz < 1 or self.decision_ticks < 1:
            _fail("out_of_range", "base_hz and decision_ticks must be positive")
        if not 1 <= self.default_lease_ticks <= self.maximum_lease_ticks:
            _fail("out_of_range", "profile lease bounds")


REFERENCE_HOST_REVISION = "f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497"


def recorded_fixture_profile() -> GameplayProfile:
    """Two-part schema example; these numeric settings are not tuned game mechanics."""
    return GameplayProfile(
        id="recorded-rig-v1", version=1, host_revision=REFERENCE_HOST_REVISION,
        representation="recorded_fixture",
        parts=(Part("body", "box"), Part("foot", "capsule")),
        joints=(Joint("leg", "body", "foot", "slide", -1, 1, "game_cells",
                      play=.05, compliance=.1, damping=.3),),
        sensors=(Sensor("velocity", "cells_per_second", "actor_local", -100, 100),
                 Sensor("global_target", "game_cells", "world", -1000, 1000, "global")),
        channels=(Channel("drive", "body"), Channel("reserved", "body", lower=0, upper=0)),
        channel_mask=(True, False), assist=Assist("bounded_gyro", 2, .4, .05, .1, .05))


@dataclass(frozen=True)
class WireAction(Record):
    """Already bounded ACT004 v2 wire values, not an ACT003 v1 packet."""

    move: tuple[float, float] = (0.0, 0.0)
    aim: tuple[float, float] = (0.0, 0.0)
    effort: float = 0.7
    trim: float = 0.0
    reel: float = 0.0
    primary: bool = False
    secondary: bool = False
    auxiliary: bool = False
    tool: bool = False
    release_usable: bool = False
    attach_usable: bool = False

    def validate(self) -> None:
        for name in ("move", "aim"):
            if math.hypot(*getattr(self, name)) > 1.0 + 1e-12:
                _fail("out_of_range", f"wire.{name} must have magnitude <= 1")
        if not 0 <= self.effort <= 1 or not -1 <= self.trim <= 1 or not -1 <= self.reel <= 1:
            _fail("out_of_range", "wire effort/trim/reel outside normalized bounds")


@dataclass(frozen=True)
class LegacyWireAction(Record):
    """ACT003 v1 exact action. Component bounds intentionally differ from v2."""

    movement: tuple[float, float]
    aim: tuple[float, float]
    effort: float
    trim: float
    reel: float
    primary: bool
    secondary: bool
    auxiliary: bool

    def validate(self) -> None:
        if any(not -1 <= value <= 1 for value in self.movement + self.aim + (self.trim, self.reel)):
            _fail("out_of_range", "ACT003 wire components must be in [-1,1]")
        if not 0 <= self.effort <= 1:
            _fail("out_of_range", "ACT003 effort must be in [0,1]")

    @classmethod
    def neutral(cls) -> LegacyWireAction:
        return cls((0, 0), (0, 0), 0, 0, 0, False, False, False)


LEGACY_METAL_CHANNELS = types.MappingProxyType({
    "ripper": (("track_drive", "chassis_torque", "track_brake", "reserved_3"),
               (-1.0, -1.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0)),
    "copter": (("collective", "lateral_cyclic", "rotor_torque", "rotor_brake"),
               (0.0, -1.0, -1.0, 0.0), (1.0, 1.0, 1.0, 1.0)),
    "gemini": (("left_thruster", "right_thruster", "reserved_2", "reserved_3"),
               (0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)),
})


LEGACY_METAL_OBSERVATION_FIELDS = (
    "requested_u0", "requested_u1", "requested_u2", "requested_u3",
    "applied_u0", "applied_u1", "applied_u2", "applied_u3",
    "requested_fx", "requested_fy", "requested_torque", "applied_fx", "applied_fy", "applied_torque",
    "passive_fx", "passive_fy", "passive_torque", "delta_vx", "delta_vy", "delta_omega", "mass", "inertia",
    "resource_scale", "energy_used", "contact_scale", "consumed_dt", "centre_delta_x", "centre_delta_y",
    "rotation_delta", "flags", "profile_index", "energy",
)


def legacy_metal_profile(rig: str, host_revision: str = REFERENCE_HOST_REVISION) -> GameplayProfile:
    """ACT003 four-channel mapping, explicitly sampled and distinct from ACT004."""
    if rig not in LEGACY_METAL_CHANNELS:
        _fail("unsupported_rig", f"ACT003 metal: {rig}")
    names, lower, upper = LEGACY_METAL_CHANNELS[rig]
    units = {"mass": "game_mass", "inertia": "game_inertia", "consumed_dt": "seconds",
             "flags": "bitmask", "profile_index": "index"}
    return GameplayProfile(
        id=f"{rig}-metal-v1", version=1, host_revision=host_revision,
        representation="sampled_ACT003", parts=(Part("body", "host_authored"),), joints=(),
        sensors=tuple(Sensor(name, units.get(name, "host_declared"), "host_declared", -1e12, 1e12)
                      for name in LEGACY_METAL_OBSERVATION_FIELDS),
        channels=tuple(Channel(name, "body", lower=minimum, upper=maximum)
                       for name, minimum, maximum in zip(names, lower, upper)),
        channel_mask=tuple(minimum != maximum for minimum, maximum in zip(lower, upper)),
        capabilities=ControlCapabilities(actuation_lane="direct"))


@dataclass(frozen=True)
class LegacyMetalAction(Record):
    actuators: tuple[float, float, float, float]
    tag_aim: tuple[float, float]

    def validate(self) -> None:
        if any(not -1 <= value <= 1 for value in self.actuators + self.tag_aim):
            _fail("out_of_range", "ACT003 metal components must be in [-1,1]")

    def admitted_actuators(self, rig: str) -> tuple[float, float, float, float]:
        """Preserve the request and expose the legacy owner's separate mapping."""
        if rig not in LEGACY_METAL_CHANNELS:
            _fail("unsupported_rig", f"ACT003 metal: {rig}")
        names, lower, upper = LEGACY_METAL_CHANNELS[rig]
        for name, value, minimum, maximum in zip(names, self.actuators, lower, upper):
            if minimum == maximum == 0 and value != 0:
                _fail("unsupported_channel", f"ACT003 {rig} {name} must be zero")
        return tuple(max(minimum, min(maximum, value))
                     for value, minimum, maximum in zip(self.actuators, lower, upper))


@dataclass(frozen=True)
class LegacyCommand(Record):
    """Unchanged ACT003 v1 envelope; intentionally never upgraded to v2 implicitly."""

    version: Literal[1]
    run_id: str
    actor_id: int
    epoch: int
    sequence: int
    observed_tick: int
    observation_id: str
    lease_ticks: int
    kind: Literal["rig_action", "actuator_action"]
    action: LegacyWireAction | LegacyMetalAction
    tag: bool
    message: tuple[float, ...]
    message_type: Literal["none", "goal_hint", "latent_state"]
    cancel: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        # The Godot v1 validator admits integer-valued JSON numbers, including
        # 1.0; keep that compatibility only in this explicitly legacy decoder.
        if not isinstance(data, Mapping):
            return super().from_dict(data)
        normalized = dict(data)
        for name in ("version", "actor_id", "epoch", "sequence", "observed_tick", "lease_ticks"):
            value = normalized.get(name)
            if type(value) is float and math.isfinite(value) and value.is_integer():
                normalized[name] = int(value)
        return super().from_dict(normalized)

    def validate(self) -> None:
        if not 1 <= len(self.run_id) <= 64 or not 1 <= len(self.observation_id) <= 128:
            _fail("invalid_identifier", "ACT003 run/observation ID length")
        ranges = ((self.actor_id, 0, 7), (self.epoch, 1, 2147483647),
                  (self.sequence, 0, 2147483647), (self.observed_tick, 0, 36000),
                  (self.lease_ticks, 1, 18))
        if any(not lower <= value <= upper for value, lower, upper in ranges):
            _fail("out_of_range", "ACT003 envelope integer bounds")
        if (self.kind == "rig_action") != isinstance(self.action, LegacyWireAction):
            _fail("lane_mismatch", "ACT003 kind/action shape")
        if len(self.message) not in (0, 4) or any(not -1 <= value <= 1 for value in self.message):
            _fail("shape_mismatch", "ACT003 message must be empty or four bounded values")
        if (self.message_type == "none") != (len(self.message) == 0):
            _fail("message_mismatch", "ACT003 message type/payload")
        if self.cancel:
            neutral = (self.action == LegacyWireAction.neutral() if isinstance(self.action, LegacyWireAction)
                       else self.action == LegacyMetalAction((0, 0, 0, 0), (0, 0)))
            if not neutral or self.tag or self.message:
                _fail("invalid_cancel", "ACT003 cancel is neutral and disarmed")


@dataclass(frozen=True)
class LifecycleStamp(Record):
    run_id: str
    actor_id: str
    epoch: int
    incarnation: int = 0
    control_generation: int = 0
    morphology_version: int = 1
    mount_generation: int = 0

    def validate(self) -> None:
        identifier(self.run_id, "lifecycle.run_id")
        identifier(self.actor_id, "lifecycle.actor_id")
        for name in ("epoch", "incarnation", "control_generation", "mount_generation"):
            nonnegative(getattr(self, name), name)
        if self.morphology_version < 1:
            _fail("invalid_version", "morphology_version")


@dataclass(frozen=True)
class Observation(Record):
    id: str
    tick: int
    lifecycle: LifecycleStamp
    profile_sha256: str
    values: tuple[float | None, ...]
    valid: tuple[bool, ...]
    view: Literal["local", "global"] = "local"
    schema_version: Literal[1] = 1

    def validate(self) -> None:
        identifier(self.id, "observation.id")
        nonnegative(self.tick, "observation.tick")
        digest(self.profile_sha256, "observation.profile_sha256")
        if len(self.values) != len(self.valid):
            _fail("shape_mismatch", "observation values/validity mask")
        for value, valid in zip(self.values, self.valid):
            if (value is not None) != valid:
                _fail("invalid_mask", "unavailable values must be null; valid values must be numeric")

    def validate_profile(self, profile: GameplayProfile) -> None:
        if self.profile_sha256 != profile.sha256:
            _fail("profile_mismatch", "observation profile digest")
        if self.view != profile.capabilities.perception:
            _fail("view_mismatch", "observation has a different declared perception budget")
        if len(self.values) != len(profile.sensors):
            _fail("shape_mismatch", "observation does not match sensor slots")
        for sensor, value in zip(profile.sensors, self.values):
            if sensor.privilege == "global" and self.view == "local" and value is not None:
                _fail("privilege_violation", f"global sensor {sensor.id} in local observation")
            if value is not None and not sensor.lower <= value <= sensor.upper:
                _fail("out_of_range", f"sensor {sensor.id}")


@dataclass(frozen=True)
class EvaluatorRecord(Record):
    """Separate type; never accepted by Observation.from_dict or a policy input."""

    observation_id: str
    tick: int
    metrics: Mapping[str, Any]
    schema_version: Literal[1] = 1

    def validate(self) -> None:
        identifier(self.observation_id, "evaluator.observation_id")
        nonnegative(self.tick, "evaluator.tick")


@dataclass(frozen=True)
class Command(Record):
    id: str
    lifecycle: LifecycleStamp
    profile_sha256: str
    observation_id: str
    observed_tick: int
    sequence: int
    lease_ticks: int
    action: tuple[float, ...]
    action_mask: tuple[bool, ...]
    lane: Literal["wire", "direct"] = "wire"
    schema_version: Literal[1] = 1

    def validate(self) -> None:
        identifier(self.id, "command.id")
        identifier(self.observation_id, "command.observation_id")
        digest(self.profile_sha256, "command.profile_sha256")
        nonnegative(self.observed_tick, "command.observed_tick")
        nonnegative(self.sequence, "command.sequence")
        if self.lease_ticks < 1:
            _fail("out_of_range", "command lease must be positive")
        if len(self.action) != len(self.action_mask):
            _fail("shape_mismatch", "command action/mask")

    def validate_against(self, profile: GameplayProfile, observation: Observation,
                         current_tick: int, lifecycle: LifecycleStamp,
                         *, last_sequence: int = -1) -> None:
        observation.validate_profile(profile)
        if self.profile_sha256 != profile.sha256:
            _fail("profile_mismatch", "command profile digest")
        if self.lifecycle != lifecycle or observation.lifecycle != lifecycle:
            _fail("lifecycle_mismatch", "run, actor, epoch, morphology, mount or grant changed")
        if self.observation_id != observation.id or self.observed_tick != observation.tick:
            _fail("observation_mismatch", "command source observation identity")
        if type(current_tick) is not int or current_tick < self.observed_tick:
            _fail("invalid_clock", "consumption cannot predate observation")
        # Owner input tick is exclusive at expiry, matching ACT003/004. A request
        # observed at 0 with lease 12 may drive 11->12, never 12->13.
        if current_tick >= self.observed_tick + self.lease_ticks:
            _fail("expired", "lease is anchored to source observation, never receipt time")
        if self.lease_ticks > profile.maximum_lease_ticks:
            _fail("out_of_range", "command exceeds profile lease cap")
        if self.sequence <= last_sequence:
            _fail("sequence_replay", "command sequence did not advance")
        if self.lane != profile.capabilities.actuation_lane:
            _fail("lane_mismatch", "configured actuation lane is fixed")
        if self.action_mask != profile.channel_mask or len(self.action) != len(profile.channels):
            _fail("shape_mismatch", "command differs from declared channel slots/mask")
        for channel, active, value in zip(profile.channels, self.action_mask, self.action):
            if not channel.lower <= value <= channel.upper:
                _fail("out_of_range", f"action channel {channel.id}")
            if not active and value != channel.neutral:
                _fail("unsupported_channel", f"inactive channel {channel.id} is not neutral")


@dataclass(frozen=True)
class Admission(Record):
    requested: Command
    accepted: bool
    reason: str
    admitted_tick: int
    admitted_action: tuple[float, ...] | None = None

    def validate(self) -> None:
        nonnegative(self.admitted_tick, "admission.tick")
        if self.accepted != (self.admitted_action is not None):
            _fail("invalid_receipt", "rejected requests have no admitted action")
        if not self.reason:
            _fail("invalid_receipt", "admission requires a reason")
        if self.accepted and len(self.admitted_action) != len(self.requested.action):
            _fail("shape_mismatch", "admitted and requested action dimensions")


def admit_command(command: Command, profile: GameplayProfile, observation: Observation,
                  current_tick: int, lifecycle: LifecycleStamp, *, last_sequence: int = -1) -> Admission:
    """Validate at receipt; callers must validate again at owner consumption."""
    try:
        command.validate_against(profile, observation, current_tick, lifecycle,
                                 last_sequence=last_sequence)
    except ContractError as exc:
        return Admission(command, False, exc.code, max(current_tick, 0), None)
    return Admission(command, True, "accepted", current_tick, command.action)


@dataclass(frozen=True)
class ActuationReceipt(Record):
    """Consumed tick is the owner input tick, before advancing that base step."""

    admission: Admission
    consumed_tick: int
    consumed_action: tuple[float, ...]
    applied_effort: tuple[float, ...]
    passive_motion: tuple[float, ...] = ()
    limiting_flags: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.admission.accepted:
            _fail("invalid_receipt", "rejected command cannot have consumed actuation")
        command = self.admission.requested
        if self.consumed_tick < self.admission.admitted_tick:
            _fail("invalid_clock", "consumption predates admission")
        if self.consumed_tick >= command.observed_tick + command.lease_ticks:
            _fail("expired", "command expired between admission and consumption")
        if len(self.consumed_action) != len(command.action) or len(self.applied_effort) != len(command.action):
            _fail("shape_mismatch", "consumed action/applied effort differ from channels")
        if self.consumed_action != self.admission.admitted_action:
            _fail("invalid_receipt", "consumed request differs from admitted request")


def consume_admission(admission: Admission, profile: GameplayProfile, observation: Observation,
                       current_tick: int, lifecycle: LifecycleStamp,
                       applied_effort: tuple[float, ...], *,
                       passive_motion: tuple[float, ...] = (),
                       limiting_flags: tuple[str, ...] = (),
                       last_consumed_sequence: int = -1) -> ActuationReceipt:
    """Recheck lifetime at consumption; does not step or otherwise mutate a host."""
    if not admission.accepted:
        _fail("invalid_receipt", "cannot consume a refused request")
    admission.requested.validate_against(profile, observation, current_tick, lifecycle,
                                         last_sequence=last_consumed_sequence)
    replace(admission.requested, action=admission.admitted_action).validate_against(
        profile, observation, current_tick, lifecycle, last_sequence=last_consumed_sequence)
    return ActuationReceipt(admission, current_tick, admission.admitted_action,
                            applied_effort, passive_motion, limiting_flags)
