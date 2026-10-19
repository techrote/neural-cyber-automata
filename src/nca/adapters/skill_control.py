"""Serialized owner boundary joining typed intent, local skills and the real host.

No provider receives the environment or evaluator information. The caller keeps
invoking step while a provider is pending; the realtime host advances on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from nca.contracts import ContractError, WireAction
from nca.skills import SkillInitiation, SkillProgress, SkillRuntime, interval_observations, validate_parameters
from .decisions import ControlIdentity, DecisionProvider, DecisionRouter, DecisionUpdate, PLAN_LIFETIME_TICKS
from .godot import CybersandEnv


@dataclass(frozen=True)
class OwnerStep:
    decision: DecisionUpdate
    initiation: SkillInitiation | None
    progress: SkillProgress
    observation: Mapping[str, Any]
    info: Mapping[str, Any]
    terminated: bool
    truncated: bool


class SkillControlSession:
    """One explicitly owned actor and at most one live or retired backend call.

    A terminal invocation revokes its grant once. A new invocation needs an
    explicit transfer or submit_intent call. Pending model requests do not stop
    active local feedback. With no active skill, fallback is a fresh neutral
    request; owner cancellation, rather than neutral input, disarms prior charge.
    """

    def __init__(self, env: CybersandEnv, profile: Mapping[str, Any],
                 view: Mapping[str, Any], provider: DecisionProvider, *,
                 wall_timeout_seconds: float = 10):
        self.env = env
        self.runtime = SkillRuntime(profile)
        self.view = dict(view)
        self._observations: tuple = ()
        self._cancelled_terminal: str | None = None
        self._cancel_failed = False
        self._model_plan_expires: int | None = None
        self._latest_progress = SkillProgress(None, self.view["tick"], "idle", "idle", "no_skill")
        self._terminal_progress: SkillProgress | None = None
        self.cancellations: list[dict] = []
        self.router = DecisionRouter(provider, self.runtime.allowed_skills,
                                     validate_parameters, self._cancel_owner,
                                     wall_timeout_seconds=wall_timeout_seconds)
        self.identity()

    def identity(self) -> ControlIdentity:
        mount = self.runtime.mount_ref(self.view)
        return ControlIdentity.from_view(self.view, mount_generation=mount.attachment_generation)

    def _cancel_owner(self, reason: str) -> ControlIdentity:
        result = self.runtime.cancel(reason)
        self._latest_progress = result
        self._model_plan_expires = None
        if result.terminal:
            self._cancelled_terminal = result.request_id
        previous = self.view
        self._cancel_failed = True
        self.view = self.env.cancel()
        self._cancel_failed = False
        self._observations = ()
        current = self.identity()
        if len(self.cancellations) >= 512:
            del self.cancellations[0]
        self.cancellations.append({"reason": reason, "requested_tick": previous["tick"],
                                   "completed_tick": self.view["tick"],
                                   "control_generation": current.control_generation,
                                   "mount_generation": current.mount_generation})
        return current

    def transfer(self, source: Literal["human", "script", "model"]):
        """Take over at this owner boundary without awaiting backend completion."""
        if self._cancel_failed:
            raise ContractError("owner_cleanup_failed", "reset the actor before granting new intent")
        self.view, info = self.env.observe_with_info()
        if info["finished"]:
            raise ContractError("host_terminal", "reset the completed actor before granting new intent")
        grant = self.router.transfer(source, self.identity())
        self._observations = ()
        return grant

    def submit_intent(self, source: Literal["human", "script"], skill: str,
                      parameters: Mapping[str, Any], *, request_id: str | None = None) -> SkillInitiation:
        """Human shortcuts and scripts share the same strict usable admission."""
        if source not in ("human", "script"):
            raise ContractError("intent_source", "model proposals enter through the fenced provider")
        validate_parameters(skill, parameters)
        grant = self.transfer(source)
        receipt = self.runtime.start(skill, parameters, self.view, source=source,
                                     source_generation=grant.generation, request_id=request_id,
                                     mount_ref=self.runtime.mount_ref(self.view))
        if receipt.accepted:
            self._latest_progress = SkillProgress(receipt.request.id, self.view["tick"], "running",
                                                  "initiated", "accepted_without_execution")
        return receipt

    def revoke(self, reason: str = "revoked"):
        self.router.revoke(reason)
        self._observations = ()

    def reset(self, *, seed: int | None = None):
        self.revoke("reset")
        self.view, info = self.env.reset(seed=seed)
        self.runtime = SkillRuntime(self.runtime.capability)
        self._observations = ()
        self._cancelled_terminal = None
        self._cancel_failed = False
        self._model_plan_expires = None
        self._latest_progress = SkillProgress(None, self.view["tick"], "idle", "idle", "no_skill")
        self._terminal_progress = None
        self.identity()
        return self.view, info

    def _finish_host(self, progress: SkillProgress) -> SkillProgress:
        """Keep final owner receipts separate from an unadjudicated local result."""
        if self._terminal_progress is None:
            if progress.status == "running":
                progress = replace(progress, tick=self.view["tick"], status="cancelled",
                                   phase="cancelled", reason="host_terminal_censored",
                                   action=WireAction(), local_success=None, cancel_host=True)
            self.router.revoke("host_terminal")
            self._terminal_progress = progress
        self._latest_progress = self._terminal_progress
        return self._terminal_progress

    def step(self) -> OwnerStep:
        """Consume completed feedback, poll once, and issue one fresh wire repeat."""
        if self._cancel_failed:
            raise ContractError("owner_cleanup_failed", "no release or new request follows unconfirmed cancellation")
        self.view, observed_info = self.env.observe_with_info()
        try:
            idle = interval_observations(observed_info)
        except ContractError:
            self.revoke("feedback_gap")
            raise
        samples = {(sample["run_id"], sample["control_generation"], sample["tick"]): sample
                   for sample in (*self._observations, *idle)}
        # Owner cancellation does not erase already completed idle receipts.
        # Retain them in audit info, but never feed a prior controller lifetime
        # into a newly admitted local invocation.
        identity_fields = ("run_id", "actor_id", "epoch", "model", "profile_version", "control_generation")
        current_stamp = tuple(self.view[key] for key in identity_fields)
        self._observations = tuple(sorted(
            (sample for sample in samples.values()
             if tuple(sample[key] for key in identity_fields) == current_stamp),
            key=lambda sample: sample["tick"]))
        # Expected grip/release transitions must first yield their local result.
        # This cannot grant an old planner authority over the new connection.
        previous_progress = self._latest_progress
        progress = self.runtime.progress(self.view, observations=self._observations)
        self._observations = ()
        if observed_info["finished"]:
            # A terminal owner can increment control generation before its final
            # receipts reach local adjudication. Preserve an already established
            # result; censor an active invocation rather than inventing failure.
            if previous_progress.status == "running":
                progress = previous_progress
            progress = self._finish_host(progress)
            return OwnerStep(DecisionUpdate("fallback", "host_terminal"), None, progress,
                             dict(self.view), observed_info,
                             observed_info["terminated"], observed_info["truncated"])
        plan_expired = (self._model_plan_expires is not None
                        and self.view["tick"] >= self._model_plan_expires
                        and progress.status != "succeeded")
        if plan_expired:
            progress = self.runtime.cancel("smart_plan_expired")
            self.router.revoke("smart_plan_expired")
        if progress.terminal and progress.request_id != self._cancelled_terminal:
            self._cancelled_terminal = progress.request_id
            self.router.revoke("skill_terminal")
        update = self.router.tick(self.view, mount_generation=self.identity().mount_generation,
                                  request_allowed=progress.status == "idle" or progress.terminal)
        if plan_expired:
            update = DecisionUpdate("fallback", "smart_plan_expired", backend_busy=update.backend_busy)
        if update.reason == "requested":
            # One immediate nonblocking poll gives a zero-delay provider its
            # declared same-boundary meaning, without waiting for delayed work.
            immediate = self.router.tick(self.view, mount_generation=self.identity().mount_generation,
                                         request_allowed=False)
            if immediate.reason != "awaiting_provider":
                update = immediate
        initiation = None
        if update.proposal is not None:
            proposal = update.proposal
            initiation = self.runtime.start(
                proposal.skill, proposal.parameters, self.view, source="model",
                request_id=proposal.request_id, source_generation=proposal.grant_generation,
                source_observation_id=proposal.observation_id, source_tick=proposal.observed_tick,
                mount_ref=self.runtime.mount_ref(self.view))
            if initiation.accepted:
                self._model_plan_expires = proposal.observed_tick + PLAN_LIFETIME_TICKS
                progress = self.runtime.progress(self.view)
        # Identity changes and cancellation hooks may have invalidated the local
        # invocation after its feedback was read. Never dispatch that old action.
        if self.router.grant is None:
            action = WireAction()
        else:
            action = progress.action if not progress.terminal else WireAction()
        try:
            lease = self.env.lease_ticks
            if self._model_plan_expires is not None:
                lease = min(lease, self._model_plan_expires - self.view["tick"])
            packet = self.env.command(action.to_dict(), lease_ticks=lease)
            self.view, _, terminated, truncated, info = self.env.step_packet(packet)
        except Exception:
            try:
                self.revoke("step_error")
            except Exception:
                pass
            raise
        try:
            self._observations = interval_observations(info)
        except ContractError:
            self.revoke("feedback_gap")
            raise
        if terminated or truncated:
            progress = self._finish_host(progress)
        self._latest_progress = progress
        return OwnerStep(update, initiation, progress, dict(self.view), info, terminated, truncated)


def register_decisions(registry):
    registry.register("adapter", "asynchronous-skills", 1, SkillControlSession,
                      ("act004", "local-feedback", "source-fencing", "bounded-backend"))
