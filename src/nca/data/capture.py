"""Capture actual owner consumption without refreshing delayed source packets."""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nca.adapters.godot import CybersandEnv
from nca.adapters.protocol import HostError, owned_json
from nca.contracts import ContractError, sha256_json
from nca.runs import RunBudget
from .records import (ActionSource, CaptureEvent, EpisodeRecord, PolicyContext,
                      PolicySample, TickRecord, boundary_changes, json_copy, policy_view)


def source_identity(root: str | Path | None = None) -> tuple[str, dict[str, str], str]:
    """Pin executed package sources to a matching source checkout and its HEAD.

    A source-tree import uses that tree by default. An installed package needs
    an explicit root, NCA_SOURCE_ROOT, or the current Git checkout. Its complete
    Python/Godot source file set must match the checkout before a revision can
    be attributed; a wheel does not inherit whichever Git HEAD happens nearby.
    """
    package = Path(__file__).resolve().parents[1]
    source_tree = package.parents[1]
    selected = root if root is not None else os.environ.get("NCA_SOURCE_ROOT")
    if selected is None:
        selected = source_tree if (source_tree / "src/nca").resolve() == package else Path.cwd()
    try:
        project = Path(subprocess.check_output(
            ["git", "-C", str(selected), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL).strip()).resolve()
        revision = subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"],
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("source_checkout", "provide source_root or NCA_SOURCE_ROOT for a matching nca Git checkout") from exc
    checkout_package = project / "src/nca"
    if not checkout_package.is_dir() or not (project / "pyproject.toml").is_file():
        raise ContractError("source_checkout", "selected Git checkout is not the nca source project")

    def package_hashes(folder):
        files = sorted(set(folder.rglob("*.py")) | set(folder.rglob("*.gd")))
        return {"src/nca/" + path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files}

    hashes = package_hashes(package)
    if hashes != package_hashes(checkout_package):
        raise ContractError("source_mismatch", "installed nca Python/Godot sources differ from the selected checkout")
    for relative in ("pyproject.toml", "tools/capture_replay.py"):
        path = project / relative
        if path.is_file():
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return revision, hashes, sha256_json(hashes)


class CaptureBudgetExceeded(ContractError):
    def __init__(self):
        super().__init__("capture_budget", "declared decision or wall-time capture ceiling reached")


class EpisodeRecorder:
    """One actual ACT004 episode. A reset creates a new recorder and artifact.

    The recorder receives copied local observations. It never reads a whole
    evaluator snapshot or edits host mechanics. Cancellation/takeover are owner
    calls, with boundary records preserving body motion and old action identity.
    """

    def __init__(self, env: CybersandEnv, episode_id: str, source: ActionSource, *,
                 context: PolicyContext | None = None, seed: int | None = None,
                 scenario_family: str = "protocol-smoke", split: str = "evaluation",
                 budget: RunBudget = RunBudget(), source_root: str | Path | None = None):
        if env.host.lane != "act004":
            raise ContractError("environment_lane", "capture currently requires the actual ACT004 owner")
        self.env, self.id, self.source, self.context = env, episode_id, source, context
        self.scenario_family, self.split, self.budget = scenario_family, split, budget
        self.revision, self.source_files, self.source_hash = source_identity(source_root)
        self.view, self.reset_info = env.reset(seed=seed)
        self.initial = owned_json(self.view)
        self._observations = {self.view["observation_id"]: owned_json(self.view)}
        self.profile = owned_json(env.host.request("profile")["profile"])
        self.events: list[CaptureEvent] = []
        self.segment = 0
        self._active_request: dict[str, Any] | None = None
        self._active_source: str | None = None
        self._started = time.monotonic_ns()
        self._decisions = 0
        self._done = False
        self._failed = False

    def _wall(self) -> int:
        return time.monotonic_ns() - self._started

    def _check_budget(self) -> None:
        if self._done:
            raise ContractError("episode_closed", "capture artifact has already been finalized")
        if self._failed:
            raise ContractError("capture_failed", "finalize the failed capture before starting a new episode")
        if self.budget.exhausted(self._decisions, self._wall() / 1e9):
            raise CaptureBudgetExceeded()

    def step(self, action: Mapping[str, Any], *, sample: PolicySample | None = None,
             intervention: str = "none", replaced: bool = False,
             context: PolicyContext | None = None):
        self._check_budget()
        packet = self.env.command(action)
        return self.step_packet(packet, sample=sample, intervention=intervention,
                                replaced=replaced, context=context)

    def step_packet(self, packet: Mapping[str, Any] | None, *, ticks: int | None = None,
                    sample: PolicySample | None = None, intervention: str = "none",
                    replaced: bool = False, context: PolicyContext | None = None):
        self._check_budget()
        before, started = owned_json(self.view), self._wall()
        original = None if packet is None else json_copy(packet)
        if intervention not in ("none", "human_correction", "model_override", "source_takeover") or type(replaced) is not bool:
            raise ContractError("intervention_schema", "a known intervention and explicit replacement flag are required")
        if sample is not None:
            if sample.policy_sha256 != self.source.policy_sha256:
                raise ContractError("policy_mismatch", "sample identity differs before host admission")
            if (sample.space == "wire_request" and original is not None and not replaced
                    and sha256_json(sample.values) != sha256_json(original.get("action"))):
                raise ContractError("sample_mismatch", "unlabelled action replacement refused before host admission")
        try:
            result = self.env.step_packet(original, ticks=ticks)
            return self._record_step(result, original, before, started, sample, intervention, replaced, context)
        except (HostError, ContractError, KeyError, TypeError, ValueError) as exc:
            self._failure(type(exc).__name__, before, started)
            raise

    def _record_step(self, result, original, before, started, sample, intervention, replaced, context):
        after, reward, terminated, truncated, info = result
        source_observation = (None if original is None else self._observations.get(original.get("observation_id")))
        if source_observation is not None and any(
                source_observation.get(observation_key) != original.get(request_key)
                for observation_key, request_key in (("tick", "observed_tick"), ("run_id", "run_id"),
                                                     ("epoch", "epoch"), ("actor_id", "actor_id"))):
            source_observation = None
        raw_ticks = list(info.get("idle_trace", ())) + list(info["trace"])
        complete = bool(info.get("consumed_trace_complete", True))
        records = []
        previous = before
        discontinuities = []
        for raw in raw_ticks:
            request_list = raw.get("requested", [])
            admissions = raw.get("admission", [])
            if not isinstance(admissions, list) or len(admissions) != len(request_list):
                raise ContractError("admission_trace_schema", "ACT004 request/admission trace differs")
            for requested, admission in zip(request_list, admissions):
                if admission.get("accepted") is True:
                    self._active_request = owned_json(requested)
                    self._active_source = self.source.id
            if self._active_request is not None:
                expires = self._active_request["observed_tick"] + self._active_request["lease_ticks"]
                if raw["tick_before"] >= expires:
                    self._active_request, self._active_source = None, None
            view = raw["observation"]
            changes = boundary_changes(previous, view)
            if ("control_generation" in changes and (terminated or truncated)
                    and raw["tick_after"] == after["tick"] and view.get("suspended")):
                changes = tuple("terminal_control_fence" if change == "control_generation" else change
                                for change in changes)
            discontinuities.extend(change for change in changes if change not in discontinuities)
            if any(change in changes for change in ("reset", "epoch", "actor", "morphology", "profile")):
                self._active_request, self._active_source = None, None
            records.append(TickRecord(
                tick_before=raw["tick_before"], tick_after=raw["tick_after"], world_tick=raw["world_tick"],
                received_monotonic_usec=raw["received_monotonic_usec"],
                completed_monotonic_usec=raw["completed_monotonic_usec"],
                requested=tuple(request_list), admission=tuple(admissions),
                consumed_request=self._active_request, source_id=self._active_source,
                consumed_action=view.get("action", {}), applied=raw["applied"], passive=raw["passive"],
                actual_motion=raw.get("actual_motion", {}), observation=view,
                cellular_sample_tick=raw.get("cellular_sample_tick"), valid=not raw.get("failed", False)))
            if "control_generation" in changes or "terminal_control_fence" in changes:
                self._active_request, self._active_source = None, None
            self._observations[view["observation_id"]] = owned_json(view)
            previous = view
        if original and any(original.get("action", {}).get(name) for name in ("attach_usable", "release_usable")):
            # Host topology identifies detachable parts. Other mount identities
            # are not universally published; conservatively fence assumptions.
            discontinuities.append("mount_dependency_request")
        if context is not None and context != self.context:
            discontinuities.append("goal_changed")
            self.context = context
        valid = complete and bool(records) and all(record.valid for record in records)
        expected = before["tick"]
        for record in records:
            valid = valid and record.tick_before == expected
            expected = record.tick_after
        valid = valid and expected == after["tick"]
        if not valid:
            discontinuities.append("incomplete_trace")
        event = CaptureEvent(
            kind="step", index=len(self.events), segment=self.segment,
            tick_before=before["tick"], tick_after=after["tick"],
            wall_started_ns=started, wall_completed_ns=self._wall(), source=self.source,
            issued_observation=before, source_observation=source_observation,
            policy_observation=policy_view(before if source_observation is None else source_observation),
            requested_packet=original,
            sample=sample, context=self.context, ticks=tuple(records), next_observation=after,
            reward=reward, terminated=terminated, truncated=truncated,
            reason=info.get("reason", ""), intervention=intervention, replaced=replaced,
            discontinuities=tuple(discontinuities), valid=valid, trace_complete=complete)
        self.events.append(event)
        self._decisions += 1
        self.view = owned_json(after)
        if discontinuities:
            self.segment += 1
        return result

    def _failure(self, reason: str, before: Mapping[str, Any], started: int) -> None:
        self.events.append(CaptureEvent(
            kind="failure", index=len(self.events), segment=self.segment,
            tick_before=before["tick"], tick_after=before["tick"],
            wall_started_ns=started, wall_completed_ns=self._wall(), source=self.source,
            issued_observation=before, policy_observation=policy_view(before),
            reason=reason, valid=False, trace_complete=False))
        self._failed = True
        self.segment += 1

    def cancel(self, reason: str = "source_cancel") -> dict[str, Any]:
        return self._cancel(reason, None)

    def takeover(self, source: ActionSource, reason: str = "source_takeover") -> dict[str, Any]:
        if source.id == self.source.id:
            raise ContractError("takeover_identity", "takeover requires a distinct source identity")
        return self._cancel(reason, source)

    def _cancel(self, reason: str, replacement: ActionSource | None) -> dict[str, Any]:
        # Cancellation remains available at a decision/time ceiling.
        if self._done:
            raise ContractError("episode_closed", "cannot cancel a finalized capture")
        if not isinstance(reason, str) or not reason or len(reason) > 256:
            raise ContractError("event_reason", "cancellation requires a nonempty bounded reason")
        before, started = owned_json(self.view), self._wall()
        try:
            after = self.env.cancel()
        except HostError as exc:
            self._failure(type(exc).__name__, before, started)
            raise
        changes = boundary_changes(before, after)
        if "control_generation" not in changes:
            self._failure("cancel_fence", before, started)
            raise ContractError("cancel_fence", "owner did not revoke old observation identities")
        previous_source = self.source
        if replacement is not None:
            self.source = replacement
        self.events.append(CaptureEvent(
            kind="takeover" if replacement is not None else "cancel", index=len(self.events), segment=self.segment,
            tick_before=before["tick"], tick_after=after["tick"],
            wall_started_ns=started, wall_completed_ns=self._wall(), source=self.source,
            source_before=previous_source, issued_observation=before,
            policy_observation=policy_view(before), next_observation=after, context=self.context,
            reason=reason, intervention="source_takeover" if replacement is not None else "none",
            discontinuities=changes))
        self.segment += 1
        self._active_request, self._active_source = None, None
        self.view = owned_json(after)
        self._observations[after["observation_id"]] = owned_json(after)
        return owned_json(after)

    def finish(self, status: str | None = None, reason: str | None = None,
               evaluation: Mapping[str, Any] | None = None) -> EpisodeRecord:
        if self._done:
            raise ContractError("episode_closed", "capture already finalized")
        if status is None:
            status = ("failed" if self._failed else "inconclusive"
                      if any(not event.valid or not event.trace_complete for event in self.events) else "completed")
        if reason is None:
            reason = self.events[-1].reason if self.events and self.events[-1].reason else "capture_complete"
        record = EpisodeRecord(
            id=self.id, host_identity=self.env.host.identity, learner_revision=self.revision,
            learner_sources_sha256=self.source_hash, source_files=self.source_files,
            config=self.env.config, profile=self.profile, initial_observation=self.initial,
            events=tuple(self.events), status=status, reason=reason,
            scenario_family=self.scenario_family, split=self.split,
            reward_kind=self.reset_info["reward_kind"], final_observation_valid=not self._failed,
            evaluation={} if evaluation is None else evaluation)
        self._done = True
        return record
