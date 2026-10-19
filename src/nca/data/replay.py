"""Replay admitted consumed requests on the same pinned actual host build."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nca.adapters.godot import CybersandEnv, GodotHost
from nca.adapters.protocol import owned_json
from nca.contracts import ContractError, Record, sha256_json
from .capture import source_identity
from .records import EpisodeRecord, json_copy


@dataclass(frozen=True)
class ReplayTolerance(Record):
    position: float = 0.001
    velocity: float = 0.001
    rotation: float = 0.0001
    angular_velocity: float = 0.001
    applied: float = 0.001

    def validate(self) -> None:
        if any(getattr(self, name) < 0 for name in ("position", "velocity", "rotation", "angular_velocity", "applied")):
            raise ContractError("replay_tolerance", "tolerances must be finite and nonnegative")


@dataclass(frozen=True)
class IdentityMapping(Record):
    original_run_id: str
    replay_run_id: str
    original_epoch: int
    replay_epoch: int
    original_observation_id: str
    replay_observation_id: str
    observed_tick: int
    consumed_tick: int
    sequence: int
    lease_ticks: int
    original_packet_sha256: str
    replay_packet_sha256: str


@dataclass(frozen=True)
class ReplayReport(Record):
    episode_id: str
    episode_sha256: str
    passed: bool
    original_host_identity: Mapping[str, Any]
    replay_host_identity: Mapping[str, Any]
    tolerance: ReplayTolerance
    replayed_ticks: int
    replayed_requests: int
    omitted_rejected_requests: int
    cancellations: int
    numeric_comparisons: int
    maximum_errors: Mapping[str, float]
    mismatches: tuple[str, ...]
    identities: tuple[IdentityMapping, ...]
    replay_source_sha256: str
    scope: str = "copied-local-trajectory-and-consumption; not full-state or cross-platform identity"


def _compatible(original: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    for field in ("host_revision", "host_tree", "godot_version", "godot_binary_sha256",
                  "native_artifact_sha256", "rapier_artifact_sha256", "bridge_sha256", "schedule", "lane", "mode"):
        if original.get(field) != current.get(field):
            raise ContractError("replay_host_mismatch", f"incompatible {field}")
    if current.get("mode") != "stepped":
        raise ContractError("replay_mode", "causal replay currently requires stepped consumption")


def rewrite_packet_identity(packet: Mapping[str, Any], observations: Mapping[str, Mapping[str, Any]],
                            current: Mapping[str, Any]) -> tuple[dict[str, Any], IdentityMapping]:
    """Map an actually issued old observation to its replay peer; never renew age."""
    original_id = packet.get("observation_id")
    if original_id not in observations:
        raise ContractError("replay_observation", "original source observation has no issued replay peer")
    source = observations[original_id]
    if source.get("tick") != packet.get("observed_tick"):
        raise ContractError("replay_age", "identity mapping would change the original observation age")
    if source.get("actor_id") != packet.get("actor_id"):
        raise ContractError("replay_actor", "identity mapping would target another actor")
    result = json_copy(packet)
    result["run_id"] = source["run_id"]
    result["epoch"] = source["epoch"]
    result["observation_id"] = source["observation_id"]
    mapping = IdentityMapping(
        original_run_id=packet["run_id"], replay_run_id=source["run_id"],
        original_epoch=int(packet["epoch"]), replay_epoch=int(source["epoch"]),
        original_observation_id=original_id, replay_observation_id=source["observation_id"],
        observed_tick=int(packet["observed_tick"]), consumed_tick=int(current["tick"]),
        sequence=int(packet["sequence"]), lease_ticks=int(packet["lease_ticks"]),
        original_packet_sha256=sha256_json(packet), replay_packet_sha256=sha256_json(result))
    return result, mapping


def replay_episode(episode: EpisodeRecord, host: GodotHost,
                   tolerance: ReplayTolerance = ReplayTolerance()) -> ReplayReport:
    """Recreate by config/seed, issue only admitted packets at recorded input ticks.

    Holding a command across ticks is reproduced by *not* resending it; replay
    does not renew leases or create extra button edges. Original and replay
    identities are paired explicitly while sequence, age and timing remain fixed.
    """
    _compatible(episode.host_identity, host.identity)
    if not episode.final_observation_valid or any(not event.valid or not event.trace_complete for event in episode.events):
        raise ContractError("replay_incomplete", "invalid or incomplete consumption cannot establish replay")
    env = CybersandEnv(host, json_copy(episode.config), actor_id=int(episode.initial_observation["actor_id"]))
    current, _ = env.reset()
    if sha256_json(host.request("profile")["profile"]) != episode.profile_sha256:
        raise ContractError("replay_profile", "actual authored mechanics differ from recorded profile")
    if current["tick"] != episode.initial_observation["tick"]:
        raise ContractError("replay_initial_tick", "reset cannot reconstruct a mid-episode snapshot")
    observations = {episode.initial_observation["observation_id"]: owned_json(current)}
    identity_records = []
    errors = {name: 0.0 for name in ("position", "velocity", "rotation", "angular_velocity", "applied")}
    mismatches = []
    numeric = 0
    replayed = 0
    requests = 0
    omitted = 0
    cancellations = 0

    def mismatch(message):
        if len(mismatches) < 64:
            mismatches.append(message)

    def compare_number(expected, actual, category, path):
        nonlocal numeric
        if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)) and len(expected) == len(actual):
            for index, (left, right) in enumerate(zip(expected, actual)):
                compare_number(left, right, category, f"{path}[{index}]")
            return
        if type(expected) not in (float, int) or type(actual) not in (float, int):
            mismatch(f"{path}: numeric field unavailable or malformed")
            return
        error = abs(float(actual) - float(expected))
        if not math.isfinite(error):
            mismatch(f"{path}: nonfinite difference")
            return
        numeric += 1
        errors[category] = max(errors[category], error)
        if error > getattr(tolerance, category):
            mismatch(f"{path}: error {error:.8g} exceeds {getattr(tolerance, category):.8g}")

    def compare_view(expected, actual):
        for name in ("position", "velocity", "rotation", "angular_velocity"):
            compare_number(expected.get(name), actual.get(name), name, f"tick{actual['tick']}.{name}")
        for name in ("tick", "model", "profile_version", "control_mode", "part_count", "grounded", "fault"):
            if expected.get(name) != actual.get(name):
                mismatch(f"tick{actual['tick']}.{name}: discrete mismatch")
        expected_parts, actual_parts = expected.get("parts", ()), actual.get("parts", ())
        if len(expected_parts) != len(actual_parts):
            mismatch(f"tick{actual['tick']}.parts: count mismatch")
        for index, (left, right) in enumerate(zip(expected_parts, actual_parts)):
            for name in ("id", "index", "detached"):
                if left.get(name) != right.get(name):
                    mismatch(f"tick{actual['tick']}.parts[{index}].{name}: mismatch")
            for name in ("position", "velocity", "rotation", "angular_velocity"):
                compare_number(left.get(name), right.get(name), name, f"tick{actual['tick']}.parts[{index}].{name}")
        for name in ("cell_patch", "action"):
            if expected.get(name) != actual.get(name):
                # Frozen tuples and returned JSON lists must compare as values.
                if sha256_json(expected.get(name)) != sha256_json(actual.get(name)):
                    mismatch(f"tick{actual['tick']}.{name}: copied local state mismatch")

    compare_view(episode.initial_observation, current)
    for event in episode.events:
        if current["tick"] != event.tick_before:
            raise ContractError("replay_tick", "event timing no longer matches replay world")
        if event.kind in ("cancel", "takeover"):
            current = env.cancel()
            cancellations += 1
            compare_view(event.next_observation, current)
            observations[event.next_observation["observation_id"]] = owned_json(current)
            continue
        if event.kind != "step":
            raise ContractError("replay_event", f"unsupported event {event.kind}")
        for expected in event.ticks:
            if current["tick"] != expected.tick_before:
                raise ContractError("replay_tick", "consumption tick differs")
            accepted = [packet for packet, admission in zip(expected.requested, expected.admission)
                        if admission["accepted"]]
            omitted += sum(not admission["accepted"] for admission in expected.admission)
            if len(accepted) > 1:
                raise ContractError("replay_capacity", "one owned actor may admit one packet per tick")
            packet = None
            if accepted:
                packet, pair = rewrite_packet_identity(accepted[0], observations, current)
                identity_records.append(pair)
                requests += 1
            current, _, terminated, truncated, info = env.step_packet(packet, ticks=1)
            actual = info["trace"][0]
            if packet is not None and not actual["admission"][0]["accepted"]:
                mismatch(f"tick{expected.tick_before}: replay refused an original admitted request")
            if actual["tick_before"] != expected.tick_before or actual["tick_after"] != expected.tick_after:
                mismatch("consumed tick duration changed")
            if actual["world_tick"] != expected.world_tick or actual.get("cellular_sample_tick") != expected.cellular_sample_tick:
                mismatch(f"tick{expected.tick_before}: owner schedule changed")
            compare_view(expected.observation, current)
            for name in ("linear_impulse", "angular_impulse", "energy_used"):
                compare_number(expected.applied.get(name), actual["applied"].get(name), "applied",
                               f"tick{expected.tick_after}.{name}")
            observations[expected.observation["observation_id"]] = owned_json(current)
            replayed += 1
        if (terminated, truncated) != (event.terminated, event.truncated):
            mismatch(f"tick{event.tick_after}: terminal/truncation semantics changed")
    _, _, replay_source = source_identity()
    return ReplayReport(episode.id, episode.sha256, not mismatches, episode.host_identity,
                        host.identity, tolerance, replayed, requests, omitted, cancellations,
                        numeric, errors, tuple(mismatches), tuple(identity_records), replay_source)
