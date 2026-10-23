"""4LY-004 Trainer-only flight protocol over a persistent real-owner transport.

This module owns learner-side contract validation only. It never integrates
physics, invents host state, launches training, or treats preparation fixtures
as actual-host evidence.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from nca.registry import registry

EPISODE_VERSION = "4ly-episode/v1"
OBSERVATION_VERSION = "4ly-observation/v1"
FLAT_OBSERVATION_VERSION = "4ly-flat-observation/v1"
ACTION_VERSION = "4ly-action/v1"
ACTION_PROFILE = "4ly_velocity_v1"
COURSE_ID = "4ly-flight-course/v1"
COURSE_VERSION = 1
SPLIT_GEOMETRY_VERSION = 3
HOST_SCHEDULE = "synchronous-mask-actuation-effects-material-feedback-terrain-rapier/v1"
TRAINER_PROFILE = "4ly_trainer"
TRAINER_PROFILE_SHA256 = "f1f9286a27b929c96caca5aaf2be70cbd82e0fd933f29968c1aaaa43ac8f1704"
CONTROLLER_SHA256 = "fb86190c68c6a87e7d2ef7e845b1fb97305f188723a326f67f07caa6971cc8ae"
ACTOR_ID = 0
PHYSICS_HZ = 60
DECISION_TICKS = 6
LEASE_TICKS = 12
FLAT_CHANNELS = 54

# Source-bound #74 case contract from FLIGHT_PREREGISTRATION.md and the
# reviewed-but-unqualified Cybersand #229 head 2e20731d. These are physical
# episode deadlines, not a universal repeat cap.
SCENARIO_MAX_TICKS = {
    "unsupported_hover": 300,
    "horizontal_reach": 1080,
    "vertical_reach": 1080,
    "diagonal_reach": 1080,
    "reverse": 1560,
    "arrival_braking": 1500,
    "wide_corridor": 1800,
    "offset_gates": 2100,
    "corner": 2100,
    "slalom": 2400,
    "wall_recovery": 2100,
}
DEVELOPMENT_SEEDS = frozenset((101, 103, 107, 109))
TUNING_SEEDS = frozenset((211, 223, 227, 229))
TRAINING_SCENARIOS = frozenset(SCENARIO_MAX_TICKS)
TUNING_SCENARIOS = frozenset(
    scenario for scenario in SCENARIO_MAX_TICKS
    if scenario != "unsupported_hover"
)


class FlightProtocolError(RuntimeError):
    """The host reply cannot be used as a trustworthy flight transition."""


class UnknownConsumptionError(FlightProtocolError):
    """A dispatched request may have been consumed, so it must never be replayed."""

    def __init__(self, operation: str, sequence: int, reason: str = "reply_lost_after_dispatch"):
        super().__init__(f"{operation} sequence {sequence}: {reason}; consumption is unknown")
        self.operation = operation
        self.sequence = sequence
        self.reason = reason


def finite_vector(value: Any, size: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError("wrong vector shape")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("finite numeric vector required")
    return tuple(float(x) for x in value)


def velocity_action(value: Any) -> dict[str, list[float]]:
    """Return the exact two-output 4LY desired-velocity action."""
    x, y = finite_vector(value, 2)
    if max(abs(x), abs(y)) > 1:
        raise ValueError("normalized velocity outputs required")
    scale = max(1.0, math.hypot(x, y))
    return {"move": [x / scale, y / scale]}


class FlightSession:
    """Validate one Trainer flight episode over an injected persistent transport.

    The transport is responsible only for carrying commands to the real
    CyberPhysicalFlightEpisode owner. Tests may inject fixtures, but a fixture is
    never actual-host evidence.

    Transport contract:
      request(command: dict) -> reply: dict
      close() -> None

    Every reply carries the caller's sequence plus consumption equal to none or
    confirmed. A transport that cannot prove whether a dispatched request was
    consumed raises UnknownConsumptionError. This session then fail-stops and
    never retries it automatically.
    """

    def __init__(self, transport: Any):
        self.transport = transport
        self.sequence = 0
        self.observation: dict[str, Any] | None = None
        self.flat_observation: tuple[float, ...] | None = None
        self.manifest: dict[str, Any] | None = None
        self.closed = False
        self.failed = False
        self.terminal = True
        self.revoked = False
        self.unknown_consumption: dict[str, Any] | None = None
        self.last_exchange: dict[str, Any] | None = None

    def _poison(self) -> None:
        self.failed = True
        self.observation = None
        self.flat_observation = None

    def _request(self, command: Mapping[str, Any]) -> dict[str, Any]:
        if self.closed or self.failed:
            raise FlightProtocolError("closed or failed session")
        self.sequence += 1
        operation = str(command.get("command", ""))
        sent = dict(command, sequence=self.sequence)
        try:
            reply = self.transport.request(deepcopy(sent))
        except UnknownConsumptionError as exc:
            self.unknown_consumption = {
                "sequence": self.sequence,
                "operation": operation,
                "state": "unknown",
                "reason": exc.reason,
                **{
                    key: deepcopy(sent[key])
                    for key in (
                        "run_id", "actor_id", "epoch", "control_generation",
                        "observation_id", "physical_tick",
                    )
                    if key in sent
                },
            }
            self._poison()
            raise
        except BaseException:
            self._poison()
            raise
        try:
            if type(reply) is not dict:
                raise FlightProtocolError("reply must be an object")
            if type(reply.get("sequence")) is not int or reply["sequence"] != self.sequence:
                raise UnknownConsumptionError(operation, self.sequence, "reply_correlation_lost")
            consumption = reply.get("consumption")
            if consumption == "unknown":
                raise UnknownConsumptionError(
                    operation, self.sequence, "host_reported_unknown_consumption")
            if consumption not in ("none", "confirmed"):
                raise UnknownConsumptionError(
                    operation, self.sequence, "missing_consumption_classification")
            return deepcopy(reply)
        except UnknownConsumptionError as exc:
            self.unknown_consumption = {
                "sequence": self.sequence,
                "operation": operation,
                "state": "unknown",
                "reason": exc.reason,
                **{
                    key: deepcopy(sent[key])
                    for key in (
                        "run_id", "actor_id", "epoch", "control_generation",
                        "observation_id", "physical_tick",
                    )
                    if key in sent
                },
            }
            self._poison()
            raise
        except BaseException:
            self._poison()
            raise

    @staticmethod
    def _host_split(split: str) -> str:
        if split == "training":
            return "development"
        if split == "tuning":
            return "tuning"
        raise ValueError("all final/sealed suites are excluded")

    @classmethod
    def _preregistered_case(
        cls, *, scenario: str, seed: int, split: str
    ) -> str:
        host_split = cls._host_split(split)
        if host_split == "development":
            scenarios, seeds = TRAINING_SCENARIOS, DEVELOPMENT_SEEDS
        else:
            scenarios, seeds = TUNING_SCENARIOS, TUNING_SEEDS
        if scenario not in scenarios:
            raise ValueError(
                f"{split} scenario is outside the preregistered #74 case set"
            )
        if seed not in seeds:
            raise ValueError(
                f"{split} seed is outside the preregistered #74 case set"
            )
        return host_split

    @staticmethod
    def _identity(observation: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(observation.get(key) for key in (
            "run_id", "actor_id", "epoch", "control_generation",
            "observation_id", "observed_tick",
        ))

    def _validate_manifest(self, manifest: Any, *, scenario: str, seed: int, split: str) -> None:
        if type(manifest) is not dict:
            raise FlightProtocolError("missing flight owner manifest")
        expected = {
            "schema": EPISODE_VERSION,
            "observation": OBSERVATION_VERSION,
            "action": ACTION_VERSION,
            "flat_observation": FLAT_OBSERVATION_VERSION,
            "flat_observation_length": FLAT_CHANNELS,
            "action_profile": ACTION_PROFILE,
            "source": "policy",
            "course_version": COURSE_VERSION,
            "split_geometry_version": SPLIT_GEOMETRY_VERSION,
            "course": scenario,
            "seed": seed,
            "split": split,
            "host_schedule": HOST_SCHEDULE,
            "physics_hz": PHYSICS_HZ,
            "decision_ticks": DECISION_TICKS,
            "lease_ticks": LEASE_TICKS,
            "profile_hash": TRAINER_PROFILE_SHA256,
            "controller_sha256": CONTROLLER_SHA256,
            "native_coupling": True,
            "host_count": 1,
            "sealed_suite_sha256": None,
            "max_ticks": SCENARIO_MAX_TICKS[scenario],
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise FlightProtocolError(f"flight manifest mismatch: {key}")
        self.manifest = deepcopy(manifest)

    def _observe(
        self, observation: Any, flat: Any, *, fresh: bool = False,
        allow_generation_advance: bool = False,
    ) -> None:
        if type(observation) is not dict:
            raise FlightProtocolError("missing flight observation")
        if (observation.get("version") != OBSERVATION_VERSION
                or observation.get("action_version") != ACTION_VERSION):
            raise FlightProtocolError("incompatible flight observation schema")
        if not isinstance(observation.get("run_id"), str) or not observation["run_id"]:
            raise FlightProtocolError("missing run identity")
        if observation.get("actor_id") != ACTOR_ID:
            raise FlightProtocolError("wrong flight actor slot")
        if (not isinstance(observation.get("observation_id"), str)
                or not observation["observation_id"]):
            raise FlightProtocolError("missing observation identity")
        for key in ("epoch", "observed_tick", "control_generation"):
            if type(observation.get(key)) is not int or observation[key] < 0:
                raise FlightProtocolError(f"invalid owner counter: {key}")

        if type(flat) is not dict or flat.get("version") != FLAT_OBSERVATION_VERSION:
            raise FlightProtocolError("incompatible flat observation schema")
        if flat.get("field_count") != FLAT_CHANNELS:
            raise FlightProtocolError("wrong flat observation field count")
        if flat.get("source_observation_id") != observation["observation_id"]:
            raise FlightProtocolError(
                "flat observation is not bound to the raw owner observation")
        values = finite_vector(flat.get("values"), FLAT_CHANNELS)

        if not fresh and self.observation is not None:
            old = self.observation
            for key in ("run_id", "actor_id", "epoch"):
                if observation[key] != old[key]:
                    raise FlightProtocolError("owner identity changed during step")
            generation_delta = (
                observation["control_generation"] - old["control_generation"])
            if generation_delta != 0 and not (
                    allow_generation_advance and generation_delta == 1):
                raise FlightProtocolError(
                    "unexpected control-generation change during step")
            if observation["observed_tick"] <= old["observed_tick"]:
                raise FlightProtocolError("stale or reordered observation")
            if observation["observation_id"] == old["observation_id"]:
                raise FlightProtocolError("duplicate observation identity")

        self.observation = deepcopy(observation)
        self.flat_observation = values

    @staticmethod
    def _transition(reply: Mapping[str, Any], *, allow_zero: bool = False) -> dict[str, Any]:
        transition = reply.get("transition")
        if type(transition) is not dict or transition.get("version") != EPISODE_VERSION:
            raise FlightProtocolError("incompatible flight transition")
        for key in ("terminated", "truncated", "failed", "success", "episode_done"):
            if type(transition.get(key)) is not bool:
                raise FlightProtocolError("explicit terminal classification required")
        ticks = transition.get("ticks_consumed")
        total = transition.get("physical_ticks_total")
        if type(ticks) is not int or not 0 <= ticks <= DECISION_TICKS:
            raise FlightProtocolError("invalid confirmed tick count")
        if type(total) is not int or total < 0:
            raise FlightProtocolError("invalid physical tick total")
        terminal = (
            transition["terminated"] or transition["truncated"] or transition["failed"])
        if transition["episode_done"] != terminal:
            raise FlightProtocolError("terminal flags disagree")
        if transition["success"] and not transition["terminated"]:
            raise FlightProtocolError("success without task termination")
        if ticks == 0 and not allow_zero and not transition["failed"]:
            raise FlightProtocolError(
                "zero-tick step requires an authoritative owner failure")
        if 0 < ticks < DECISION_TICKS and not terminal:
            raise FlightProtocolError("short nonterminal decision interval")
        return deepcopy(transition)

    def _record_exchange(
        self,
        operation: str,
        before: Mapping[str, Any] | None,
        after: Mapping[str, Any] | None,
        *,
        ticks: int,
        terminal: bool,
        consumption: str,
    ) -> None:
        """Expose a narrow join record for future accounting/capture.

        Existing ACT004 CaptureEvent/TickRecord schemas are intentionally not
        changed by this preparation task. A future flight recorder can join on
        these exact owner identities after actual-host conformance.
        """
        self.last_exchange = {
            "schema": "4ly-flight-exchange/v1",
            "sequence": self.sequence,
            "operation": operation,
            "consumption": consumption,
            "run_id": None if after is None else after.get("run_id"),
            "actor_id": ACTOR_ID,
            "epoch": None if after is None else after.get("epoch"),
            "control_generation_before": (
                None if before is None else before.get("control_generation")),
            "control_generation_after": (
                None if after is None else after.get("control_generation")),
            "observation_id_before": (
                None if before is None else before.get("observation_id")),
            "observation_id_after": (
                None if after is None else after.get("observation_id")),
            "physical_tick_before": (
                None if before is None else before.get("observed_tick")),
            "physical_tick_after": (
                None if after is None else after.get("observed_tick")),
            "ticks_consumed": ticks,
            "terminal": bool(terminal),
        }

    def reset(
        self, *, scenario: str, seed: int, split: str = "training"
    ) -> dict[str, Any]:
        if (type(seed) is not int or seed < 0
                or not isinstance(scenario, str) or not scenario):
            raise ValueError("explicit scenario and nonnegative seed required")
        host_split = self._preregistered_case(
            scenario=scenario, seed=seed, split=split
        )
        previous = deepcopy(self.observation)
        reply = self._request({
            "command": "reset",
            "profile": TRAINER_PROFILE,
            "producer": "policy",
            "scenario": scenario,
            "seed": seed,
            "split": host_split,
        })
        try:
            if reply["consumption"] != "confirmed":
                raise FlightProtocolError(
                    "successful reset requires confirmed consumption")
            self._validate_manifest(
                reply.get("manifest"), scenario=scenario, seed=seed, split=host_split)
            transition = self._transition(reply, allow_zero=True)
            if (transition["ticks_consumed"] != 0
                    or transition["physical_ticks_total"] != 0):
                raise FlightProtocolError(
                    "reset must not consume physical ticks")
            if transition["episode_done"]:
                raise FlightProtocolError("reset returned a terminal episode")
            self._observe(
                transition.get("observation"), reply.get("flat_observation"),
                fresh=True,
            )
            assert self.observation is not None
            if self.observation["observed_tick"] != 0:
                raise FlightProtocolError(
                    "reset must begin at physical tick zero")
            if previous is not None and (
                self.observation["run_id"], self.observation["epoch"]
            ) == (previous["run_id"], previous["epoch"]):
                raise FlightProtocolError(
                    "reset did not replace prior owner authority")
            self.terminal = False
            self.revoked = False
            self._record_exchange(
                "reset", previous, self.observation,
                ticks=0, terminal=False, consumption="confirmed",
            )
            return deepcopy(self.observation)
        except BaseException:
            self._poison()
            raise

    def step(self, outputs: Any, *, observation_id: str) -> dict[str, Any]:
        if self.terminal or self.revoked or self.observation is None:
            raise FlightProtocolError("live Trainer episode required")
        if observation_id != self.observation["observation_id"]:
            raise FlightProtocolError("live matching observation required")
        if self.observation["observed_tick"] % DECISION_TICKS:
            raise FlightProtocolError("decision boundary required")

        before = deepcopy(self.observation)
        old_tick = before["observed_tick"]
        reply = self._request({
            "command": "step",
            "run_id": before["run_id"],
            "actor_id": ACTOR_ID,
            "epoch": before["epoch"],
            "control_generation": before["control_generation"],
            "observation_id": observation_id,
            "physical_tick": old_tick,
            "action": velocity_action(outputs),
            "repeat_ticks": DECISION_TICKS,
        })
        try:
            if reply["consumption"] != "confirmed":
                raise FlightProtocolError(
                    "step requires authoritative consumption")
            transition = self._transition(reply)
            ticks = transition["ticks_consumed"]
            if transition["physical_ticks_total"] != old_tick + ticks:
                raise FlightProtocolError("physical tick total discontinuity")

            if ticks == 0:
                final = transition.get("observation")
                if (type(final) is not dict
                        or self._identity(final) != self._identity(before)):
                    raise FlightProtocolError(
                        "zero-tick failure changed owner state")
                self._observe(
                    final, reply.get("flat_observation"), fresh=True)
            else:
                self._observe(
                    transition.get("observation"),
                    reply.get("flat_observation"),
                    allow_generation_advance=bool(transition["failed"]),
                )
                assert self.observation is not None
                if self.observation["observed_tick"] != old_tick + ticks:
                    raise FlightProtocolError(
                        "physical tick discontinuity")

            self.terminal = bool(transition["episode_done"])
            self._record_exchange(
                "step", before, self.observation, ticks=ticks,
                terminal=self.terminal, consumption="confirmed",
            )
            return deepcopy(reply)
        except BaseException:
            self._poison()
            raise

    def _revoke(self, operation: str, *, producer: str) -> dict[str, Any]:
        if self.terminal or self.revoked or self.observation is None:
            raise FlightProtocolError("live Trainer authority required")
        before = deepcopy(self.observation)
        reply = self._request({
            "command": operation,
            "run_id": before["run_id"],
            "actor_id": ACTOR_ID,
            "epoch": before["epoch"],
            "control_generation": before["control_generation"],
            "observation_id": before["observation_id"],
            "physical_tick": before["observed_tick"],
            "producer": producer,
        })
        try:
            if reply["consumption"] != "confirmed":
                raise FlightProtocolError(
                    f"{operation} requires authoritative consumption")
            authority = reply.get("authority")
            if type(authority) is not dict:
                raise FlightProtocolError(
                    "missing revoked authority receipt")
            if (authority.get("version") != OBSERVATION_VERSION
                    or authority.get("action_version") != ACTION_VERSION):
                raise FlightProtocolError(
                    "revocation receipt schema mismatch")
            for key in ("run_id", "actor_id", "epoch", "observed_tick"):
                if authority.get(key) != before[key]:
                    raise FlightProtocolError(
                        "revocation changed episode or physical tick identity")
            if (authority.get("control_generation")
                    != before["control_generation"] + 1):
                raise FlightProtocolError(
                    "revocation did not advance control generation")
            if (not isinstance(authority.get("observation_id"), str)
                    or authority["observation_id"] == before["observation_id"]):
                raise FlightProtocolError(
                    "revocation did not invalidate issued observation")
            if authority.get("producer") != producer:
                raise FlightProtocolError(
                    "revocation producer receipt mismatch")
            self.observation = deepcopy(authority)
            self.flat_observation = None
            self.revoked = True
            self.terminal = True
            self._record_exchange(
                operation, before, authority,
                ticks=0, terminal=True, consumption="confirmed",
            )
            return deepcopy(authority)
        except BaseException:
            self._poison()
            raise

    def cancel(self) -> dict[str, Any]:
        return self._revoke("cancel", producer="policy")

    def takeover(self, producer: str = "human") -> dict[str, Any]:
        if producer != "human":
            raise ValueError(
                "flight preparation permits only explicit human takeover")
        return self._revoke("takeover", producer=producer)

    def close(self) -> None:
        if self.closed:
            return
        try:
            if not self.failed:
                reply = self._request({"command": "close"})
                if (reply.get("consumption") != "confirmed"
                        or reply.get("closed") is not True):
                    raise FlightProtocolError(
                        "host close was not confirmed")
                self._record_exchange(
                    "close", self.observation, None,
                    ticks=0, terminal=True, consumption="confirmed",
                )
        finally:
            self.closed = True
            self.terminal = True
            self.observation = None
            self.flat_observation = None
            self.transport.close()


registry.register(
    "adapter", "4ly-preparation", 1, FlightSession,
    capabilities=(
        "flight_real_owner_protocol_preparation",
        "no_execution_admission",
    ),
)
