"""Bounded asynchronous skill decisions and explicit controller ownership.

Providers propose typed skills. The owner keeps stepping and local feedback keeps
producing fresh motor commands. This module neither calls a paid service nor
grants providers access to evaluator state or body/material mutation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import queue
import threading
import time
from typing import Any, Literal, Protocol

from nca.contracts import ContractError, Record, identifier, sha256_json

PLAN_LIFETIME_TICKS = 120
MIN_REQUEST_INTERVAL_TICKS = 30


@dataclass(frozen=True)
class ControlIdentity(Record):
    run_id: str
    actor_id: int
    epoch: int
    model: str
    profile_version: int
    mount_generation: int = 0
    control_generation: int = 0

    def validate(self):
        identifier(self.run_id, "run_id")
        identifier(self.model, "model")
        if (not 0 <= self.actor_id < 8 or self.epoch < 1 or self.profile_version != 1
                or self.mount_generation < 0 or self.control_generation < 0):
            raise ContractError("invalid_identity", "actor lifetime and profile fields are invalid")

    @classmethod
    def from_view(cls, view: Mapping[str, Any], *, mount_generation: int = 0):
        return cls(view["run_id"], view["actor_id"], view["epoch"], view["model"],
                   view["profile_version"], mount_generation, view["control_generation"])


@dataclass(frozen=True)
class SourceGrant(Record):
    identity: ControlIdentity
    source: Literal["human", "script", "model"]
    generation: int

    def validate(self):
        if self.generation < 1:
            raise ContractError("invalid_grant", "ownership generation must be positive")


@dataclass(frozen=True)
class DecisionRequest(Record):
    id: str
    grant: SourceGrant
    observation_id: str
    observed_tick: int
    # A copied, whitelisted local view. The privileged info channel is excluded.
    view: Mapping[str, Any]
    allowed_skills: tuple[str, ...]
    schema_version: Literal[1] = 1

    def validate(self):
        identifier(self.id, "request.id")
        identifier(self.observation_id, "observation.id")
        if self.observed_tick < 0 or not self.allowed_skills:
            raise ContractError("invalid_request", "source tick and permitted skills are required")
        if len(self.allowed_skills) != len(set(self.allowed_skills)):
            raise ContractError("duplicate_skill", "decision skill list contains duplicates")
        for name in self.allowed_skills:
            identifier(name, "skill")


@dataclass(frozen=True)
class SkillProposal(Record):
    request_id: str
    observation_id: str
    observed_tick: int
    grant_generation: int
    skill: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    schema_version: Literal[1] = 1

    def validate(self):
        for name in ("request_id", "observation_id", "skill"):
            identifier(getattr(self, name), name)
        if self.observed_tick < 0 or self.grant_generation < 1:
            raise ContractError("invalid_proposal", "source tick and owner fence required")


class DecisionProvider(Protocol):
    """One backend call is retained until completion or acknowledged cancellation."""
    def submit(self, request: DecisionRequest, tick: int) -> None: ...
    def poll(self, tick: int) -> tuple[str, Any] | None: ...
    def cancel(self, request_id: str) -> bool: ...


class DelayedMockProvider:
    """Deterministic provider faults; this is never described as a live model."""
    def __init__(self, choose: Callable[[DecisionRequest], Any], *, delay_ticks: int = 0,
                 cancellation_ack: bool = True, outage: bool = False,
                 jitter_ticks: tuple[int, ...] = (0,)):
        if type(delay_ticks) is not int or not 0 <= delay_ticks <= 36000:
            raise ValueError("delay_ticks must be in [0,36000]")
        self.choose, self.delay_ticks = choose, delay_ticks
        self.cancellation_ack, self.outage = cancellation_ack, outage
        if (not jitter_ticks or any(type(value) is not int or abs(value) > 60 for value in jitter_ticks)
                or type(cancellation_ack) is not bool or type(outage) is not bool):
            raise ValueError("jitter must be bounded integer ticks; provider switches are boolean")
        self.jitter_ticks = jitter_ticks
        self._active: tuple[DecisionRequest, int] | None = None
        self.calls = 0
        self.max_inflight = 0

    def submit(self, request: DecisionRequest, tick: int):
        if self._active is not None:
            raise RuntimeError("provider already has an in-flight request")
        jitter = self.jitter_ticks[self.calls % len(self.jitter_ticks)]
        self._active = request, tick + max(0, self.delay_ticks + jitter)
        self.calls += 1
        self.max_inflight = 1

    def poll(self, tick: int):
        if self._active is None or self.outage or tick < self._active[1]:
            return None
        request, _ = self._active
        self._active = None
        try:
            return request.id, self.choose(request)
        except Exception as exc:
            # Provider exception text may contain private service payloads.
            return request.id, {"provider_error": type(exc).__name__}

    def cancel(self, request_id: str) -> bool:
        if self._active is None:
            return True
        if self._active[0].id != request_id:
            return False
        if self.cancellation_ack:
            self._active = None
        return self.cancellation_ack


class CallableWorkerProvider:
    """Optional local worker with a strict one-call cap, including retired work.

This wrapper does not implement a network service. A hung callable remains busy;
resetting a local request cannot start a stream of abandoned backend calls.
The worker is daemonized so a failed external provider cannot prevent shutdown.
"""
    def __init__(self, choose: Callable[[DecisionRequest], Any]):
        self.choose = choose
        self._active: str | None = None
        self._results: queue.Queue = queue.Queue(maxsize=1)

    def submit(self, request: DecisionRequest, tick: int):
        if self._active is not None:
            raise RuntimeError("provider busy, including retired work")
        self._active = request.id

        def worker():
            try:
                reply = self.choose(request)
            except Exception as exc:
                reply = {"provider_error": type(exc).__name__}
            self._results.put((request.id, reply))

        threading.Thread(target=worker, name="nca-local-decision", daemon=True).start()

    def poll(self, tick: int):
        try:
            value = self._results.get_nowait()
        except queue.Empty:
            return None
        self._active = None
        return value

    def cancel(self, request_id: str) -> bool:
        return self._active is None


# Only self-state and bounded local terrain/effect receipts go to a decision source.
LOCAL_VIEW_FIELDS = ("run_id", "actor_id", "epoch", "model", "profile_version", "control_generation",
                     "observation_id", "tick", "position", "velocity", "rotation",
                     "angular_velocity", "grounded", "energy", "charge", "cooldown",
                     "blocked", "stuck_time", "parts", "joints", "tool_ready",
                     "cell_patch", "last_effects", "ground_distance", "ceiling_distance")


def decision_view(view: Mapping[str, Any]) -> dict:
    if view.get("representation") != "articulated-rapier" or view.get("version") != 2:
        raise ContractError("view_mismatch", "decision input must be an ACT004 issued local observation")
    if view.get("observation_scope") != "self, rig rays and bounded composed native cells":
        raise ContractError("view_mismatch", "unrecognized perception scope")
    result = {key: view[key] for key in LOCAL_VIEW_FIELDS if key in view}
    # Validation and a deep owned copy, not a reference into mutable host info.
    from nca.contracts import strict_json_loads, canonical_json
    return strict_json_loads(canonical_json(result))


@dataclass(frozen=True)
class DecisionUpdate:
    status: str
    reason: str
    proposal: SkillProposal | None = None
    source_age_ticks: int | None = None
    backend_busy: bool = False
    local_control_required: bool = True


class DecisionRouter:
    """Exclusive actor ownership, source-anchored plans and nonblocking polling.

    All methods run on one serialized owner thread. A provider worker communicates
    through poll; it never calls router methods or the host directly.

    `on_revoke` cancels pending host input and the active skill synchronously. It
    must not reset body velocity or erase existing effects. When owner cancellation
    changes control_generation, it returns the newly observed ControlIdentity so
    an explicit transfer grants the new owner the post-cancellation identity.
    A hook without a changing host may return None.
    """
    def __init__(self, provider: DecisionProvider, allowed_skills: tuple[str, ...],
                 validate_parameters: Callable[[str, Mapping[str, Any]], None],
                 on_revoke: Callable[[str], ControlIdentity | None], *, wall_timeout_seconds: float = 10.0,
                 clock: Callable[[], float] = time.monotonic):
        if not 0 < wall_timeout_seconds <= 120:
            raise ValueError("wall timeout must be in (0,120]")
        if not allowed_skills or len(set(allowed_skills)) != len(allowed_skills):
            raise ValueError("skills must be nonempty and unique")
        for skill in allowed_skills:
            identifier(skill, "skill")
        self.provider, self.allowed_skills = provider, allowed_skills
        self.validate_parameters, self.on_revoke = validate_parameters, on_revoke
        self.wall_timeout_seconds, self.clock = wall_timeout_seconds, clock
        self.grant: SourceGrant | None = None
        self._generation = 0
        self._serial = 0
        self._backend: DecisionRequest | None = None
        self._retired = False
        self._started_wall = 0.0
        self._last_request_tick: int | None = None
        self._last_tick: int | None = None
        self.events: list[dict] = []
        self.dropped_events = 0
        self.accepted = 0
        self.refused = 0

    def _event(self, kind: str, tick: int | None, **fields):
        if len(self.events) >= 1024:
            del self.events[0]
            self.dropped_events += 1
        self.events.append({"event": kind, "tick": tick, **fields})

    def _retire_backend(self):
        if self._backend is not None:
            self._retired = True
            try:
                acknowledged = self.provider.cancel(self._backend.id)
            except Exception:
                acknowledged = False
            if acknowledged is True:
                self._backend = None

    def transfer(self, source: Literal["human", "script", "model"], identity: ControlIdentity) -> SourceGrant:
        # Validate before disturbing an existing valid owner. Once revocation
        # begins, an exception cannot resurrect the previous logical authority.
        candidate = SourceGrant(identity, source, self._generation + 1)
        previous = self.grant
        self.grant = None
        self._generation += 1
        self._retire_backend()
        if previous is not None:
            current = self.on_revoke("takeover" if previous.identity == identity else "lifetime_changed")
            if current is not None:
                if not isinstance(current, ControlIdentity):
                    raise ContractError("invalid_cancel_identity", "owner cancellation must return a typed identity")
                before = identity.to_dict()
                after = current.to_dict()
                before.pop("control_generation")
                after.pop("control_generation")
                if before != after or current.control_generation < identity.control_generation:
                    raise ContractError("cancel_identity_changed", "cancellation changed the requested actor or mount identity")
                candidate = SourceGrant(current, source, self._generation)
        self.grant = candidate
        self._last_request_tick = None
        self._last_tick = None
        self._event("ownership", None, source=source, generation=self._generation)
        return candidate

    def revoke(self, reason: str = "revoked"):
        previous = self.grant
        self.grant = None
        self._generation += 1
        self._retire_backend()
        self._event("revoked", self._last_tick, reason=reason, generation=self._generation)
        if previous is not None:
            self.on_revoke(reason)

    def tick(self, view: Mapping[str, Any], *, mount_generation: int = 0,
             request_allowed: bool = True) -> DecisionUpdate:
        if type(request_allowed) is not bool:
            raise ContractError("request_permission", "request_allowed must be boolean")
        identity = ControlIdentity.from_view(view, mount_generation=mount_generation)
        tick = view["tick"]
        if type(tick) is not int or tick < 0:
            raise ContractError("invalid_clock", "owner ticks must be nonnegative integers")
        if self.grant is None:
            # Drain retired work even when no source currently owns the actor.
            self._poll_retired(tick)
            return DecisionUpdate("fallback", "no_owner", backend_busy=self._backend is not None)
        if identity != self.grant.identity:
            self.revoke("lifetime_changed")
            return DecisionUpdate("fallback", "lifetime_changed", backend_busy=self._backend is not None)
        if self._last_tick is not None and tick < self._last_tick:
            raise ContractError("clock_reversal", "reset requires a new ownership grant")
        self._last_tick = tick
        now = self.clock()
        if self._backend is not None and not self._retired:
            age = tick - self._backend.observed_tick
            if age >= PLAN_LIFETIME_TICKS or now - self._started_wall >= self.wall_timeout_seconds:
                reason = "source_expired" if age >= PLAN_LIFETIME_TICKS else "wall_timeout"
                self._event("retired", tick, reason=reason, request_id=self._backend.id)
                self._retire_backend()
                self.refused += 1
                # A rejected long plan cannot become a newly anchored plan here.
                return DecisionUpdate("fallback", reason, source_age_ticks=age,
                                      backend_busy=self._backend is not None)
        try:
            reply = self.provider.poll(tick)
            if reply is not None:
                self._validate_reply(reply)
        except Exception:
            self._retire_backend()
            self.refused += 1
            self._event("refused", tick, reason="provider_poll_failed")
            return DecisionUpdate("fallback", "provider_poll_failed", backend_busy=self._backend is not None)
        if reply is not None:
            request_id, payload = reply
            if self._backend is None:
                self.refused += 1
                self._event("refused", tick, reason="duplicate_or_unsolicited", request_id=request_id)
                return DecisionUpdate("fallback", "duplicate_or_unsolicited")
            request = self._backend
            if request_id != request.id:
                # A stale reordered result cannot resolve the current backend slot.
                self.refused += 1
                self._event("refused", tick, reason="request_mismatch", request_id=request_id)
                return DecisionUpdate("fallback", "request_mismatch", backend_busy=True)
            self._backend = None
            if self._retired:
                self._retired = False
                self.refused += 1
                return DecisionUpdate("fallback", "retired_reply")
            try:
                proposal = payload if isinstance(payload, SkillProposal) else SkillProposal.from_dict(payload)
                if (proposal.request_id != request.id or proposal.observation_id != request.observation_id
                        or proposal.observed_tick != request.observed_tick
                        or proposal.grant_generation != request.grant.generation
                        or self.grant != request.grant):
                    raise ContractError("source_mismatch", "reply changed source observation or owner")
                if tick >= request.observed_tick + PLAN_LIFETIME_TICKS:
                    raise ContractError("source_expired", "late plan cannot acquire a fresh observation")
                if proposal.skill not in request.allowed_skills:
                    raise ContractError("unsupported_skill", "provider cannot choose undeclared skills")
                self.validate_parameters(proposal.skill, proposal.parameters)
            except (ContractError, ValueError, TypeError, KeyError):
                self.refused += 1
                self._event("refused", tick, reason="invalid_proposal", request_id=request_id)
                return DecisionUpdate("fallback", "invalid_proposal")
            self.accepted += 1
            self._event("accepted", tick, request_id=request.id, observed_tick=request.observed_tick,
                        observation_id=request.observation_id, proposal_sha256=proposal.sha256,
                        wall_latency_seconds=max(0.0, now - self._started_wall))
            return DecisionUpdate("accepted", "typed_skill_proposal", proposal,
                                  tick - request.observed_tick)
        if self._backend is not None:
            return DecisionUpdate("fallback", "retired_backend_busy" if self._retired else "awaiting_provider",
                                  backend_busy=True)
        if self.grant.source != "model":
            return DecisionUpdate("local", f"{self.grant.source}_owns_actor")
        if not request_allowed:
            return DecisionUpdate("local", "active_local_skill")
        if self._last_request_tick is not None and tick - self._last_request_tick < MIN_REQUEST_INTERVAL_TICKS:
            return DecisionUpdate("fallback", "request_cadence")
        self._serial += 1
        request = DecisionRequest(f"decision-{self._generation}-{self._serial}", self.grant,
                                  view["observation_id"], tick, decision_view(view), self.allowed_skills)
        self._backend = request
        self._retired = False
        self._started_wall = now
        self._last_request_tick = tick
        try:
            self.provider.submit(request, tick)
        except Exception:
            # Providers must reject before starting on submit failure. Until an
            # explicit cancel acknowledgement, conservatively retain the slot.
            self._retire_backend()
            self.refused += 1
            return DecisionUpdate("fallback", "provider_submit_failed", backend_busy=self._backend is not None)
        self._event("requested", tick, request_id=request.id, observation_id=request.observation_id,
                    view_sha256=sha256_json(request.view))
        return DecisionUpdate("fallback", "requested", backend_busy=True)

    def _poll_retired(self, tick: int):
        try:
            reply = self.provider.poll(tick)
            if reply is not None:
                self._validate_reply(reply)
        except Exception:
            return
        if reply is not None and self._backend is not None and reply[0] == self._backend.id:
            self._backend = None
            self._retired = False

    @staticmethod
    def _validate_reply(reply):
        if not isinstance(reply, (tuple, list)) or len(reply) != 2:
            raise ContractError("provider_reply_schema", "provider reply must contain request ID and payload")
        identifier(reply[0], "provider.request_id")


def mock_proposal(request: DecisionRequest, skill: str, **parameters) -> SkillProposal:
    return SkillProposal(request.id, request.observation_id, request.observed_tick,
                         request.grant.generation, skill, parameters)
