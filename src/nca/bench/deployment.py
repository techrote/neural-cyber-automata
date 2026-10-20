"""Predeclared CPU export screen; its game evidence always uses the real owner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np

from nca.contracts import ContractError, WireAction, sha256_json
from nca.controllers.baseline import SkillGoal
from nca.envs import CybersandEnv, GodotHost, physical_config
from nca.export import RuntimeBundle, ExportController, ExportDecision, context_for, dispatch
from nca.export.build import read_json, verify_artifact_files
from nca.export.runtime import encode, map_action
from nca.export.spec import SCREEN_SHA256, SOURCE_FILES
from .common import durations, pose, pose_difference, process_memory, wire_difference, write_artifact, numpy_build


def _goal(case, config, suffix=""):
    return SkillGoal(case["id"] + suffix, "directed_hop", tuple(case["target"]), **config["trajectory"]["goal"])


def _environment(host, case, config, *, ticks=None, model="jev", decision_ticks=6, companions=(), seed=None):
    resolved = physical_config(model, ticks=ticks or config["trajectory"]["ticks"],
                               seed=config["trajectory"]["seed"] if seed is None else seed, companions=companions)
    resolved["actors"][0]["position"] = list(case["spawn"])
    env = CybersandEnv(host, resolved, decision_ticks=decision_ticks, lease_ticks=12)
    before = time.perf_counter_ns()
    view, info = env.reset()
    return env, view, info, time.perf_counter_ns() - before


def _cancel(controller, env):
    """No replacement wire command can accidentally release a held charge."""
    controller.cancel()
    receipt = env.cancel()
    if isinstance(controller, ExportController):
        controller.confirm_owner_cancel(receipt)
    return receipt


def _sources(root: Path):
    names = set(SOURCE_FILES) | {
        "src/nca/export/__init__.py", "src/nca/export/build.py", "src/nca/controllers/learned.py",
        "src/nca/train/artifacts.py", "src/nca/adapters/godot.py", "src/nca/adapters/protocol.py",
        "src/nca/adapters/host_bridge.gd", "src/nca/envs/__init__.py", "tools/export_policy.py"}
    names.update(str(path.relative_to(root)) for path in (root / "src/nca").rglob("*.py"))
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(names)}


def _processor_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name":
                return value.strip()
    except OSError:
        pass
    return None


def _arithmetic(bundle, policy, config):
    import torch
    from nca.controllers.learned import features, clone_request
    limits = config["arithmetic"]
    rng = np.random.default_rng(limits["seed"])
    nh = np.zeros(32, dtype=np.float32)
    th = policy.zero_hidden()
    maxima = {key: 0.0 for key in ("features", "hidden", "logits", "value", "outputs", "wire")}
    exact_flags = True
    rows = []
    goal = _goal(config["trajectory"]["cases"][0], config)
    previous = WireAction(effort=0)
    with torch.no_grad():
        for index in range(limits["observations"]):
            reset = index in limits["resets_at"]
            if reset:
                th = policy.zero_hidden()
                previous = WireAction(effort=0)
            x = rng.uniform(-2, 2, 26).astype(np.float32)
            outputs, value, nh, logits = bundle.network.forward(x, nh, reset=reset)
            tl, tv, th = policy(torch.tensor(x).unsqueeze(0), th)
            to = policy.clone_outputs(tl)[0].numpy()
            row = {"index": index, "reset": reset,
                   "hidden": float(np.max(np.abs(nh - th[0].numpy()))),
                   "logits": float(np.max(np.abs(logits - tl[0].numpy()))),
                   "value": abs(value - float(tv[0])), "outputs": float(np.max(np.abs(outputs - to)))}
            # Numeric fixtures cover normalization and absent sensor validity.
            # They do not constitute a game rollout or additional training.
            tick = (index % 32) * 6
            view = {"version": 2, "representation": "articulated-rapier", "model": "jev",
                    "run_id": "arithmetic-fixture", "epoch": 1 + index // 32, "actor_id": 0,
                    "profile_version": 1, "control_generation": 0, "tick": tick,
                    "observation_id": f"arithmetic-fixture:{1 + index // 32}:0:0:{tick}",
                    "position": rng.uniform([140, 250], [290, 347]).tolist(),
                    "velocity": rng.uniform(-120, 120, 2).tolist(),
                    "rotation": float(rng.uniform(-math.pi, math.pi)),
                    "angular_velocity": float(rng.uniform(-12, 12)),
                    "grounded": (None, False, True)[index % 3],
                    "energy": None if index % 4 == 0 else float(rng.uniform(0, 1)),
                    "cooldown": None if index % 5 == 0 else float(rng.uniform(0, 1)),
                    "ground_distance": None if index % 6 == 0 else float(rng.uniform(0, 192)),
                    "ceiling_distance": None if index % 7 == 0 else float(rng.uniform(0, 192)),
                    "fault": "", "suspended": False}
            nf, tf = encode(view, goal, previous, 0), features(view, goal, previous=previous, start_tick=0)
            row["features"] = max(abs(a - b) for a, b in zip(nf, tf))
            na, ta = map_action(outputs, view, goal), clone_request(to.tolist(), view, goal)
            difference = wire_difference(na.to_dict(), ta.to_dict())
            row["wire"] = difference["maximum_continuous_difference"]
            exact_flags &= difference["boolean_actions_exact"]
            for key in maxima:
                maxima[key] = max(maxima[key], row[key])
            rows.append(row)
            previous = ta
    passed = (maxima["features"] <= limits["feature_absolute_tolerance"]
              and all(maxima[key] <= limits["hidden_logit_value_absolute_tolerance"]
                      for key in ("hidden", "logits", "value", "outputs"))
              and maxima["wire"] <= limits["continuous_action_absolute_tolerance"] and exact_flags)
    return {"passed": passed, "kind": "independent_sequential_numeric_reference",
            "policy_sha256": bundle.network.policy_sha256, "samples": len(rows),
            "resets_at": limits["resets_at"], "maximum_absolute_differences": maxima,
            "boolean_actions_exact": exact_flags, "rows": rows}


def _episode(host, case, config, controller, kind, *, ticks=None, generator=None, seed=None):
    # A short resume audit stops by owner cancellation, preserving a valid
    # continuing world rather than inventing an earlier physical terminal.
    env, view, _, reset_ns = _environment(host, case, config, seed=seed)
    profile = host.request("profile")["profile"]
    goal = _goal(case, config)
    initial, observations, steps, frames, times = view, [], [], [pose(view)], []
    final_ticks = ticks or config["trajectory"]["ticks"]
    try:
        while view["tick"] < final_ticks:
            issued = view
            observations.append(issued)
            before = time.perf_counter_ns()
            result = (controller.step(issued, goal) if kind == "numpy"
                      else controller.step(issued, goal, deterministic=generator is None))
            times.append(time.perf_counter_ns() - before)
            if kind == "numpy":
                if result.status != "accepted":
                    raise ContractError("deployment_fallback", result.reason)
                view, _, terminated, truncated, info = dispatch(env, controller, result, issued)
            else:
                lease = min(12, initial["tick"] + round(goal.timeout_seconds * 60) - issued["tick"])
                packet = env.command(result.action.to_dict(), observation=issued, lease_ticks=lease)
                view, _, terminated, truncated, info = env.step_packet(packet)
            steps.append({"issued_observation_id": issued["observation_id"], "issued_tick": issued["tick"],
                          "action": result.action.to_dict(), "features": result.features,
                          "hidden_before": result.hidden_before,
                          "hidden_after": (result.hidden_after if kind == "numpy" else controller.hidden[0].tolist()),
                          "value": result.value, "actual_ticks": info["actual_ticks"],
                          "terminated": terminated, "truncated": truncated, "trace": info["trace"]})
            frames.extend(pose(row["observation"]) for row in info["trace"])
            if terminated or truncated:
                break
        cancelled = _cancel(controller, env)
        cancellation_exact = (pose(cancelled) == pose(view) and cancelled["tick"] == view["tick"]
                              and cancelled["control_generation"] == view["control_generation"] + 1)
    except BaseException:
        try:
            _cancel(controller, env)
        except Exception:
            pass
        raise
    return {"format": "nca-export-trajectory/v1", "case": dict(case), "controller": kind,
            "config": env.config, "profile": profile, "profile_sha256": sha256_json(profile),
            "goal": asdict(goal), "initial": initial, "steps": steps, "frames": frames,
            "final": view, "cancelled": cancelled, "cancellation_exact": cancellation_exact,
            "observations": observations, "reset_ns": reset_ns, "inference": durations(times),
            "actual_ticks": sum(step["actual_ticks"] for step in steps), "decisions": len(steps),
            "stop_reason": "host_terminal" if steps[-1]["terminated"] or steps[-1]["truncated"] else "measurement_cap"}


def _compare_episodes(left, right, config, *, feature_width=26):
    if type(feature_width) is not int or feature_width not in (26, 32):
        raise ContractError("trajectory_feature_spec", "declared BC26 or residual32 feature width required")
    maximum_position = maximum_angle = maximum_wire = maximum_hidden = maximum_features = maximum_value = 0.0
    flags = clocks = detachments = True
    if (not left["frames"] or not left["steps"] or len(left["frames"]) != len(right["frames"])
            or len(left["steps"]) != len(right["steps"])):
        return {"passed": False, "reason": "different_sequence_length"}
    for episode in (left, right):
        # zip() alone silently discards missing receipts. Every consumed tick
        # needs its own frame and receipt before numeric parity is meaningful.
        cursor = episode["frames"][0]["tick"]
        frame_index = 1
        for step in episode["steps"]:
            if (any(not isinstance(step[key], (list, tuple)) or len(step[key]) != width
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in step[key])
                    for key, width in (("features", feature_width), ("hidden_after", 32)))
                    or type(step["value"]) not in (int, float) or not math.isfinite(step["value"])):
                return {"passed": False, "reason": "invalid_numeric_evidence"}
            count = step["actual_ticks"]
            if (type(count) is not int or count <= 0 or len(step["trace"]) != count
                    or step["issued_tick"] != cursor):
                return {"passed": False, "reason": "incomplete_receipt_sequence"}
            for row in step["trace"]:
                if (row["tick_before"] != cursor or row["tick_after"] != cursor + 1
                        or frame_index >= len(episode["frames"])
                        or episode["frames"][frame_index]["tick"] != cursor + 1):
                    return {"passed": False, "reason": "incomplete_receipt_sequence"}
                cursor += 1
                frame_index += 1
        if frame_index != len(episode["frames"]):
            return {"passed": False, "reason": "incomplete_receipt_sequence"}
    for a, b in zip(left["frames"], right["frames"]):
        difference = pose_difference(a, b)
        maximum_position = max(maximum_position, difference["maximum_part_position_difference_cells"])
        maximum_angle = max(maximum_angle, difference["maximum_part_angle_difference_radians"])
        detachments &= difference["detached_exact"]
    for a, b in zip(left["steps"], right["steps"]):
        wire = wire_difference(a["action"], b["action"])
        maximum_wire = max(maximum_wire, wire["maximum_continuous_difference"])
        flags &= wire["boolean_actions_exact"]
        maximum_hidden = max(maximum_hidden, max(abs(x-y) for x, y in zip(a["hidden_after"], b["hidden_after"])))
        maximum_features = max(maximum_features, max(abs(x-y) for x, y in zip(a["features"], b["features"])))
        maximum_value = max(maximum_value, abs(a["value"]-b["value"]))
        clocks &= (a["issued_tick"], a["actual_ticks"], a["terminated"], a["truncated"]) == (
            b["issued_tick"], b["actual_ticks"], b["terminated"], b["truncated"])
        for ar, br in zip(a["trace"], b["trace"]):
            clocks &= all(ar[key] == br[key] for key in ("tick_before", "tick_after", "world_tick", "cellular_sample_tick"))
            clocks &= [(v["accepted"], v["reason"]) for v in ar["admission"]] == [
                (v["accepted"], v["reason"]) for v in br["admission"]]
    limits = config["trajectory"]
    passed = (maximum_position <= limits["maximum_part_position_difference_cells"]
              and maximum_angle <= limits["maximum_part_angle_difference_radians"]
              and maximum_wire <= config["arithmetic"]["continuous_action_absolute_tolerance"]
              and flags and clocks and detachments and left["cancellation_exact"] and right["cancellation_exact"])
    return {"passed": passed, "maximum_part_position_difference_cells": maximum_position,
            "maximum_part_angle_difference_radians": maximum_angle,
            "maximum_requested_continuous_difference": maximum_wire, "boolean_requests_exact": flags,
            "maximum_hidden_difference_on_own_trajectory": maximum_hidden,
            "maximum_features_difference_on_own_trajectory": maximum_features,
            "maximum_value_difference_on_own_trajectory": maximum_value,
            "receipt_clocks_and_admission_exact": clocks, "detachments_exact": detachments,
            "cancellation_exact": left["cancellation_exact"] and right["cancellation_exact"],
            "same_world_determinism_claim": False}


def _negative_contracts(host, bundle, config):
    case = config["trajectory"]["cases"][0]
    changes = {"profile_hash": {"profile_sha256": "0"*64}, "sensor_hash": {"sensor_sha256": "0"*64},
               "action_hash": {"action_sha256": "0"*64}, "assist_hash": {"assist_sha256": "0"*64},
               "base_hz_30": {"base_hz": 30}, "decision_ticks_12": {"decision_ticks": 12},
               "helper_lease_ticks_18": {"helper_lease_ticks": 18}}
    results = []
    for name in ["bundle_version", *changes, "nonfinite_weights"]:
        env, view, _, _ = _environment(host, case, config)
        error = None
        try:
            if name == "bundle_version":
                value = bundle.value
                value["format"] = "nca-cpu-policy/v999"
                value.pop("sha256")
                value["sha256"] = sha256_json(value)
                RuntimeBundle(value, bundle.context)
            elif name == "nonfinite_weights":
                from nca.export import NumpyGRU
                weights = bundle.value["weights"]
                weights["actor.bias"][0] = float("nan")
                NumpyGRU(weights)
            else:
                RuntimeBundle(bundle.value, replace(bundle.context, **changes[name]))
        except ContractError as exc:
            error = exc.code
        fallback = ExportDecision(WireAction(effort=0), "fallback", error or "unexpected_acceptance", True)
        cancelled = env.cancel()
        after, _, _, _, info = env.step_packet(None)
        zero_impulses = all(row["applied"].get("linear_impulse") == [0.0, 0.0]
                            and row["applied"].get("angular_impulse") == 0 for row in info["trace"])
        results.append({"case": name, "passed": error is not None and pose(view) == pose(cancelled),
                        "reason": error, "fallback": fallback.action.to_dict(),
                        "owner_cancel_generation": cancelled["control_generation"],
                        "cancel_preserves_pose_and_motion": pose(view) == pose(cancelled),
                        "passive_ticks": info["actual_ticks"], "requested_packets": 0,
                        "zero_active_impulses": zero_impulses,
                        "scope": "contract variant refusal followed by actual owner cancel and passive interval"})
    for name in ("old_actor_epoch", "old_control_generation", "stale_observation", "reordered_observation"):
        env, first, _, _ = _environment(host, case, config)
        controller = ExportController(bundle)
        goal = _goal(case, config)
        decision = controller.step(first, goal)
        view, _, _, _, _ = dispatch(env, controller, decision, first)
        if name == "old_actor_epoch":
            view, _ = env.reset()
            controller.step(view, goal)
            result = controller.step(first, goal)
        elif name == "old_control_generation":
            view = env.cancel()
            result = controller.step(view, goal)
        elif name == "stale_observation":
            view, _, _, _, _ = env.step_packet(None)
            result = controller.step(first, goal, current_tick=view["tick"])
        else:
            result = controller.step(first, goal)
        cancelled = dispatch(env, controller, result, view)
        results.append({"case": name, "passed": result.cancel_owner and pose(view) == pose(cancelled),
                        "reason": result.reason, "source_tick": first["tick"], "current_tick": view["tick"],
                        "fallback": result.action.to_dict(), "requested_packets_after_refusal": 0,
                        "owner_cancel_generation": cancelled["control_generation"],
                        "cancel_preserves_pose_and_motion": pose(view) == pose(cancelled),
                        "scope": "actual reset, control revocation or completed clock observations"})
    return {"passed": all(row["passed"] for row in results), "cases": results}


def _variant(host, bundle, config, *, model="jev", decision_ticks=6):
    case = config["trajectory"]["cases"][0]
    env, view, _, _ = _environment(host, case, config, ticks=24, model=model, decision_ticks=decision_ticks)
    profile = host.request("profile")["profile"]
    reason = None
    try:
        candidate = context_for(profile, host.identity, decision_ticks=decision_ticks)
        RuntimeBundle(bundle.value, candidate)
    except ContractError as exc:
        reason = exc.code
    cancelled = env.cancel()
    frames, trace = [pose(view)], []
    while view["tick"] < 24:
        view, _, terminal, truncated, info = env.step_packet(None, ticks=min(6, 24-view["tick"]))
        trace.extend(info["trace"])
        frames.extend(pose(row["observation"]) for row in info["trace"])
        if terminal or truncated:
            break
    return {"passed": reason is not None, "case": dict(case), "profile": profile,
            "profile_sha256": sha256_json(profile), "controller": "refused-and-owner-cancelled",
            "variant": {"model": model, "decision_ticks": decision_ticks}, "reason": reason,
            "config": env.config, "cancelled": cancelled, "frames": frames, "trace": trace,
            "requested_actions": 0, "actual_ticks": view["tick"],
            "scope": "changed actual model or external cadence; no changed protected mechanics"}


def _expired_twelve_tick_refusals(source, packet, receipts):
    """Prove expiry of this valid packet under the pinned ACT004 owner.

    The owner checks its eighteen-tick observation-history window before the
    packet's twelve-tick lease. An older packet therefore reports ``lease``;
    a packet aged twelve through eighteen ticks reports ``expired_observation``.
    Neither reason alone establishes that the requested expiry case occurred.
    """
    integer_fields = ("version", "actor_id", "epoch", "sequence", "observed_tick", "lease_ticks")
    identity_fields = ("version", "kind", "run_id", "actor_id", "epoch", "sequence",
                       "observed_tick", "observation_id", "lease_ticks")
    if (not isinstance(source, Mapping) or not isinstance(packet, Mapping)
            or set(packet) != set(identity_fields) | {"action"}
            or any(type(packet[name]) is not int for name in integer_fields)
            or packet["version"] != 2 or packet["kind"] != "physical_wire_action"
            or packet["lease_ticks"] != 12 or not 0 <= packet["actor_id"] < 8
            or packet["epoch"] < 0 or packet["observed_tick"] < 0
            or not 0 <= packet["sequence"] <= 9_007_199_254_740_991
            or any(type(source.get(name)) is not int for name in
                   ("version", "epoch", "actor_id", "control_generation", "tick", "last_sequence"))
            or source["version"] != 2 or source["control_generation"] < 0
            or source["last_sequence"] < -1 or packet["sequence"] <= source["last_sequence"]
            or not isinstance(packet["run_id"], str) or not packet["run_id"]
            or any(packet[name] != source.get(name) for name in
                   ("run_id", "epoch", "actor_id", "observation_id"))
            or packet["observed_tick"] != source["tick"]
            or packet["observation_id"] != ":".join(str(source[name]) for name in
                   ("run_id", "epoch", "actor_id", "control_generation", "tick"))):
        raise ContractError("realtime_expiry_evidence", "valid original twelve-tick packet and issued source required")
    # Complete shape, strict booleans, finite numbers and normalized bounds.
    wire_difference(packet["action"], packet["action"])
    if not isinstance(receipts, list):
        raise ContractError("realtime_expiry_evidence", "complete original receipt sequence required")
    matches = []
    for row in receipts:
        if (not isinstance(row, Mapping) or type(row.get("tick_before")) is not int
                or type(row.get("tick_after")) is not int or row["tick_before"] < 0
                or row["tick_after"] != row["tick_before"] + 1
                or not isinstance(row.get("requested"), list) or not isinstance(row.get("admission"), list)
                or len(row["requested"]) != len(row["admission"])):
            raise ContractError("realtime_expiry_evidence", "complete causal request/admission receipts required")
        for requested, admission in zip(row["requested"], row["admission"]):
            if (not isinstance(requested, Mapping) or set(requested) != set(packet)
                    or not isinstance(admission, Mapping)
                    or any(type(requested[name]) not in (int, float) for name in integer_fields)
                    or any(requested[name] != packet[name] for name in identity_fields)):
                continue
            # Godot's receipt JSON rounds finite wire floats during serialization.
            # This tight comparison only correlates the original packet; it does
            # not change the independent arithmetic or trajectory tolerances.
            difference = wire_difference(requested["action"], packet["action"])
            if not difference["boolean_actions_exact"] or difference["maximum_continuous_difference"] > 1e-12:
                continue
            age = row["tick_before"] - packet["observed_tick"]
            expected_reason = ("lease" if age > 18 else "expired_observation") if age >= 12 else None
            if (expected_reason is not None and admission.get("accepted") is False
                    and admission.get("reason") == expected_reason):
                matches.append({"requested": requested, "admission": admission,
                                "tick_before": row["tick_before"], "tick_after": row["tick_after"],
                                "source_age_ticks": age, "lease_deadline_tick": packet["observed_tick"] + 12,
                                "pinned_host_observation_window_ticks": 18})
    return matches


def _realtime(host_path, godot_path, bundle, config):
    started = time.perf_counter()
    limit = config["realtime"]
    case = config["trajectory"]["cases"][0]
    calls = 0
    with GodotHost(host_path, godot_path, mode="realtime") as host:
        env, first, _, _ = _environment(host, case, config, ticks=600)
        preparation = {"move": [1., 0.], "aim": [.8, -.6], "effort": .8, "primary": True}
        first, _, _, _, preparation_info = env.step(preparation)
        controller = ExportController(bundle)
        goal = _goal(case, config)
        decision = controller.step(first, goal)
        calls += 1
        if decision.status != "accepted":
            raise ContractError("realtime_initial", decision.reason)
        # The delayed output retains this exact first view and twelve-tick lease.
        packet = env.command(decision.action.to_dict(), observation=first)
        time.sleep(limit["delay_seconds"])
        view, _, _, _, info = env.step_packet(packet)
        receipts = info["idle_trace"] + info["trace"]
        refusals = _expired_twelve_tick_refusals(first, packet, receipts)
        result = controller.step(first, goal, current_tick=view["tick"])
        calls += 1
        before_response = host.request("observe")
        before = before_response["observation"]
        cancelled = dispatch(env, controller, result, before)
        # A realtime request can straddle a tick. The immutable cancellation
        # result is compared to the last available receipt at its own tick.
        after_response = host.request("observe")
        available = before_response.get("idle_trace", []) + after_response.get("idle_trace", [])
        matching = [row["observation"] for row in available if row["tick_after"] == cancelled["tick"]]
        prior_same_tick = before if before["tick"] == cancelled["tick"] else (matching[-1] if matching else None)
        preserved = (prior_same_tick is not None and pose(prior_same_tick) == pose(cancelled)
                     and math.hypot(*cancelled["velocity"]) > 1e-3)
        # A distinct invocation is alive when the source generation changes.
        # This check therefore cannot pass solely because expiry retired the
        # earlier invocation. Cancellation clears its old cadence cursor only.
        next_goal = replace(goal, id=goal.id + "-fresh-owner")
        fresh = cancelled
        launched = controller.step(fresh, next_goal, source_generation=1)
        calls += 1
        if launched.status != "accepted":
            raise ContractError("realtime_fresh_invocation", launched.reason)
        fresh, _, _, _, fresh_info = dispatch(env, controller, launched, fresh)
        takeover = controller.step(fresh, next_goal, source_generation=2)
        calls += 1
        transfer_before = host.request("observe")
        transferred = dispatch(env, controller, takeover, fresh)
        transfer_after = host.request("observe")
        transfer_candidates = [transfer_before["observation"]]
        transfer_candidates.extend(row["observation"] for response in (transfer_before, transfer_after)
                                   for row in response.get("idle_trace", []))
        transfer_same_tick = next((row for row in reversed(transfer_candidates) if row["tick"] == transferred["tick"]), None)
        transfer_preserved = transfer_same_tick is not None and pose(transfer_same_tick) == pose(transferred)
        fresh_admitted = any(item.get("accepted") is True for row in fresh_info["trace"] for item in row["admission"])
        elapsed = time.perf_counter() - started
        checks = {"host_advances_during_delay": view["tick"] - first["tick"] >= 12,
                  "old_twelve_tick_motor_lease_refused": bool(refusals),
                  "controller_source_age_refused": result.reason == "source_expired",
                  "owner_cancel_preserves_existing_motion": preserved,
                  "source_takeover_retires_old_invocation": fresh_admitted and takeover.cancel_owner
                    and takeover.reason == "owner_or_mount_changed" and transfer_preserved,
                  "bounded_calls_and_time": calls <= limit["maximum_policy_calls"] and elapsed <= limit["maximum_seconds"]}
        return {"passed": all(checks.values()), "checks": checks, "host_identity": host.identity,
                "config": env.config, "policy_calls": calls, "wall_seconds": elapsed,
                "authored_motion_preparation": {"action": preparation, "trace": preparation_info["trace"],
                                                "purpose": "nonzero actual motion before delayed learned output and cancellation"},
                "source": first, "packet": packet, "after_delay": view, "refusals": refusals,
                "source_fallback": result.reason, "cancellation": cancelled,
                "same_tick_before_cancel": prior_same_tick, "after_takeover": fresh,
                "fresh_invocation": {"goal": asdict(next_goal), "status": launched.status,
                                     "admitted": fresh_admitted, "trace": fresh_info["trace"]},
                "takeover_reason": takeover.reason, "takeover_cancelled": transferred,
                "same_tick_before_takeover_cancel": transfer_same_tick, "takeover_sent_packets": 0,
                "trace": receipts,
                "plan_lifetime_ticks": 120, "plan_scope": "not exercised; source planning is the separately qualified T108 layer",
                "learned_performance_promotion": False}


def _throughput(host, bundle, config):
    setting = config["throughput"]
    started = time.perf_counter()
    deadline = started + setting["maximum_wall_seconds"]
    case = config["trajectory"]["cases"][0]
    rows = []
    for actors in setting["actor_configurations"]:
        wall_started = time.perf_counter_ns()
        reset_times, inference_times, transport_times = [], [], []
        ticks = decisions = episodes = 0
        trace_hash = hashlib.sha256()
        controller = None
        env = None
        while decisions < setting["decisions_per_configuration"] and time.perf_counter() < deadline:
            if controller is None:
                companions = () if actors["total_actors"] == 1 else (actors["companion_model"],)
                env, view, _, reset_ns = _environment(host, case, config, companions=companions)
                reset_times.append(reset_ns)
                controller = ExportController(bundle)
                goal = _goal(case, config, f"-{episodes}")
                episodes += 1
            before = time.perf_counter_ns()
            result = controller.step(view, goal)
            inference_times.append(time.perf_counter_ns() - before)
            if result.status != "accepted":
                raise ContractError("throughput_fallback", result.reason)
            before = time.perf_counter_ns()
            view, _, terminal, truncated, info = dispatch(env, controller, result, view)
            transport_times.append(time.perf_counter_ns() - before)
            trace_hash.update(sha256_json(info["trace"]).encode())
            ticks += info["actual_ticks"]
            decisions += 1
            if terminal or truncated:
                _cancel(controller, env)
                controller = None
        if controller is not None:
            _cancel(controller, env)
        elapsed = (time.perf_counter_ns() - wall_started) / 1e9
        rows.append({"configuration": dict(actors), "resolved_config": env.config if env else None,
                     "decisions": decisions, "actual_ticks": ticks, "episodes": episodes,
                     "wall_seconds_including_reset_capture_and_cancel": elapsed,
                     "end_to_end_decisions_per_second": decisions / elapsed,
                     "end_to_end_host_ticks_per_second": ticks / elapsed,
                     "inference": durations(inference_times), "step_transport_and_host": durations(transport_times),
                     "reset": durations(reset_times), "host_process_memory": process_memory(host.pid),
                     "controller_process_memory_including_torch_reference": process_memory(os.getpid()),
                     "owned_actor_parts": len(view["parts"]) if env else None,
                     "trace_stream_sha256": trace_hash.hexdigest(),
                     "completed": decisions == setting["decisions_per_configuration"]})
    reset_times, observe_times = [], []
    env, view, _, _ = _environment(host, case, config)
    for _ in range(setting["reset_samples"]):
        if time.perf_counter() >= deadline:
            break
        before = time.perf_counter_ns()
        view, _ = env.reset()
        reset_times.append(time.perf_counter_ns() - before)
    for _ in range(setting["observe_transport_samples"]):
        if time.perf_counter() >= deadline:
            break
        before = time.perf_counter_ns()
        env.observe()
        observe_times.append(time.perf_counter_ns() - before)
    env.cancel()
    elapsed = time.perf_counter() - started
    return {"passed": all(row["completed"] for row in rows)
            and len(reset_times) == setting["reset_samples"] and len(observe_times) == setting["observe_transport_samples"]
            and elapsed <= setting["maximum_wall_seconds"], "configurations": rows,
            "reset_rpc": durations(reset_times), "observe_rpc": durations(observe_times),
            "wall_seconds": elapsed, "maximum_wall_seconds": setting["maximum_wall_seconds"],
            "multiple_learned_controlled_actors": "unmeasured",
            "companion_scope": "second configuration contains one learned actor and one authored demo"}


def _resume(host, training_root, bundle, config):
    import torch
    from nca.controllers.learned import LearnedController, policy_state_sha256
    from nca.train.artifacts import load_training_checkpoint
    path = training_root / "checkpoints" / config["resume"]["checkpoint"]
    metadata = bundle.value["training_evaluation"]
    training_manifest, manifest_hash = read_json(training_root / "manifest.json")
    summary, summary_hash = read_json(training_root / metadata.get("training_receipt_file", "summary.json"))
    checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    matching = [row for row in summary["checkpoints"] if row["file"] == path.name]
    if (manifest_hash != metadata["manifest_file_sha256"] or summary_hash != metadata.get("training_receipt_sha256", metadata["summary_sha256"])
            or len(matching) != 1 or matching[0]["sha256"] != checkpoint_hash):
        raise ContractError("resume_lineage", "actual saved boundary differs from its verified training receipt")
    families = training_manifest["config"]["training_families"]
    kwargs = {"expected_config_sha256": metadata["config_sha256"],
              "expected_sources_sha256": metadata["learner_sources_sha256"]}
    first = load_training_checkpoint(path, **kwargs)
    second = load_training_checkpoint(path, **kwargs)
    count = config["resume"]["rng_draws"]
    generators, action_draws, scenario_draws, global_draws = [], [], [], []
    original_global = torch.get_rng_state()
    try:
        for policy, optimizer, state, _ in (first, second):
            generator = torch.Generator()
            generator.set_state(state["torch_generator_state"])
            action_draws.append(torch.randn((count, 3), generator=generator).tolist())
            # Restore again: the new episode consumes the checkpoint boundary,
            # not the eight diagnostic draws that were compared above.
            generator.set_state(state["torch_generator_state"])
            generators.append(generator)
            scenario = random.Random()
            scenario.setstate(state["scenario_rng_state"])
            scenario_draws.append([scenario.choice(families)["id"] for _ in range(count)])
            torch.set_rng_state(state["torch_global_rng_state"])
            global_draws.append(torch.rand(count).tolist())
    finally:
        torch.set_rng_state(original_global)
    def optimizer_digest(optimizer):
        state = optimizer.state_dict()
        shaped = {"param_groups": state["param_groups"], "state": {}}
        for key, values in state["state"].items():
            shaped["state"][str(key)] = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                                                 "sha256": hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()}
                                           if torch.is_tensor(value) else value for name, value in values.items()}
        return sha256_json(shaped)
    original_policy_hash = policy_state_sha256(first[0])
    policies_equal = original_policy_hash == policy_state_sha256(second[0])
    optimizer_hashes = [optimizer_digest(item[1]) for item in (first, second)]
    controllers = [LearnedController(item[0], generator=gen) for item, gen in zip((first, second), generators)]
    zero_hidden = all(bool(torch.equal(controller.hidden, torch.zeros_like(controller.hidden))) for controller in controllers)
    resumed_case = next(family for family in families if family["id"] == scenario_draws[0][0])
    # Exact next-episode host seed formula from the frozen learner loop. This
    # does not choose a favorable geometry using the evaluation result.
    next_host_seed = 100000 + first[3]["training_seed"] * 100 + first[2]["episode_counter"]
    episodes = [_episode(host, resumed_case, config, controller, "torch-resume",
                         ticks=config["resume"]["fresh_episode_ticks"], generator=gen, seed=next_host_seed)
                for controller, gen in zip(controllers, generators)]
    comparison = _compare_episodes(episodes[0], episodes[1], config, feature_width=first[0].input_size)
    checks = {"weights_exact": policies_equal, "optimizer_exact": optimizer_hashes[0] == optimizer_hashes[1],
              "next_action_rng_draws_exact": action_draws[0] == action_draws[1],
              "next_scenario_rng_draws_exact": scenario_draws[0] == scenario_draws[1],
              "next_global_rng_draws_exact": global_draws[0] == global_draws[1],
              "fresh_hidden_zero": zero_hidden, "fresh_actual_episode_outputs_and_pose": comparison["passed"],
              "weights_unchanged_after_rollout": all(policy_state_sha256(item[0]) == original_policy_hash
                                                    for item in (first, second)),
              "checkpoint_file_unchanged": hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint_hash}
    result = {"passed": all(checks.values()), "checks": checks, "checkpoint_file": path.name,
              "checkpoint_file_sha256": checkpoint_hash,
              "policy_sha256": policy_state_sha256(first[0]), "optimizer_state_sha256": optimizer_hashes,
              "optimizer_parameters_with_state": len(first[1].state), "completed_updates": first[2]["completed_updates"],
              "checkpoint_next_episode_boundary": first[2]["next_episode_boundary"],
              "rng_draws": count, "action_draws": action_draws[0], "scenario_draws": scenario_draws[0],
              "training_scenario_count": len(families), "resumed_scenario": resumed_case,
              "next_episode_counter": first[2]["episode_counter"], "next_host_seed": next_host_seed,
              "global_draws": global_draws[0], "comparison": comparison, "new_optimizer_updates": 0,
              "scope": "two actual fresh game episodes from the same restored trained optimizer/RNG boundary",
              "mid_episode_host_restore": False}
    return result, episodes


def run_screen(*, root: Path, training_root: Path, bundle_path: Path, bundle_hash: str,
               config_path: Path, host_path: Path, godot_path: Path, output: Path,
               accepted_t105_commit: str, render_python: str | None = None) -> dict[str, Any]:
    import torch
    from nca.controllers.learned import LearnedController, load_policy
    root, training_root, bundle_path, config_path, host_path, godot_path, output = (
        path.resolve() for path in (root, training_root, bundle_path, config_path, host_path, godot_path, output))
    child_environment = os.environ.copy()
    # The reference interpreter can have an editable install pointing to a
    # different worktree. Both measurement children must import this frozen
    # checkout, not whichever package that installation happens to expose.
    child_environment["PYTHONPATH"] = str(root / "src")
    torch.set_num_threads(1)
    config, config_file_hash = read_json(config_path)
    if sha256_json(config) != SCREEN_SHA256:
        raise ContractError("predeclaration", "deployment screen criteria changed")
    if str(torch.__version__) != config["runtime"]["torch_reference"]:
        raise ContractError("reference_runtime", "declared CPU Torch reference required")
    payload, _ = read_json(bundle_path)
    context = context_for(payload["profile"], payload["context"])
    bundle = RuntimeBundle.load(bundle_path, context, expected_sha256=bundle_hash)
    if payload["training_evaluation"]["accepted_t105_commit"] != accepted_t105_commit:
        raise ContractError("dependency_gate", "bundle and reviewed dependency commit differ")
    evidence = payload["training_evaluation"]
    if evidence.get("evidence_kind") == "staged_fixed_candidate/v1":
        from nca.export.candidate import verify_staged_bundle
        training_files = verify_staged_bundle(training_root, bundle)
    else:
        training_files = {"checkpoints/bc-seed1103-final.pt": evidence["checkpoint_sha256"],
                          "exports/bc-seed1103-final.weights.json": evidence["numeric_source_sha256"],
                          "manifest.json": evidence["manifest_file_sha256"], "summary.json": evidence["summary_sha256"],
                          "environment.json": evidence["environment_file_sha256"],
                          "dataset.json": evidence["dataset_file_sha256"]}
        evaluation_files = {row["name"]: row["sha256"] for row in evidence["evaluation_artifacts"]}
        if len(evaluation_files) != 8 or set(evaluation_files).intersection(training_files):
            raise ContractError("evaluation_coverage", "eight distinct evaluation artifacts required")
        training_files.update(evaluation_files)
    verify_artifact_files(training_root, training_files)
    sources = _sources(root)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                              capture_output=True, check=True).stdout.strip()
    manifest = {"format": "nca-export-screen-manifest/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "phase": "frozen_before_deployment_screen", "config": config,
                "config_content_sha256": SCREEN_SHA256, "config_file_sha256": config_file_hash,
                "base_revision": revision, "source_files": sources, "scoped_source_sha256": sha256_json(sources),
                "accepted_t105_commit": accepted_t105_commit, "bundle_file_sha256": bundle_hash,
                "training_artifact_file_sha256": training_files,
                "bundle_content_sha256": bundle.sha256, "runtime_context": context.to_dict(),
                "reference_runtime": {"torch": str(torch.__version__), "threads": torch.get_num_threads()},
                "numpy_build": numpy_build(),
                "logical_cpu_count": os.cpu_count(), "processor_model": _processor_model(),
                "training_updates": 0, "paid_model_calls": 0}
    manifest["sha256"] = sha256_json(manifest)
    artifacts = [write_artifact(output, "manifest.json", manifest)]
    report = {"format": "nca-export-screen-result/v1", "manifest_content_sha256": manifest["sha256"],
              "bundle_file_sha256": bundle_hash, "source_sha256": sha256_json(sources), "errors": [],
              "runtime_qualified": False, "qualification_scope": "Linux external Python NumPy FP32 CPU loop",
              "owner_feel_accepted": False, "whole_game_determinism": False}
    started = time.perf_counter()
    phase = "load_reference"
    try:
        policy, _ = load_policy(training_root / "checkpoints/bc-seed1103-final.pt",
                                expected_config_sha256=config["selection"]["learner_config_sha256"])
        phase = "arithmetic"
        report["arithmetic"] = _arithmetic(bundle, policy, config)
        artifacts.append(write_artifact(output, "arithmetic.json", report["arithmetic"]))
        with GodotHost(host_path, godot_path) as host:
            report["host_identity"] = host.identity
            if context_for(payload["profile"], host.identity) != context:
                raise ContractError("live_host_context", "actual host differs from bundled artifacts")
            pairs, sequences, comparisons = [], [], []
            phase = "matched_trajectories"
            for case in config["trajectory"]["cases"]:
                pair = [_episode(host, case, config, LearnedController(policy), "torch"),
                        _episode(host, case, config, ExportController(bundle), "numpy")]
                receipt = _compare_episodes(*pair, config)
                comparisons.append({"case": case["id"], **receipt})
                pair_names = []
                for episode in pair:
                    name = f"trajectories/{case['id']}-{episode['controller']}.json.gz"
                    artifacts.append(write_artifact(output, name, episode))
                    pair_names.append(name)
                pairs.append({"id": case["id"], "left": pair_names[0], "right": pair_names[1],
                              "labels": ["Torch reference", "NumPy export"], "ticks": config["trajectory"]["ticks"]})
                sequences.append({"goal": pair[1]["goal"], "observations": pair[1]["observations"]})
            report["trajectories"] = {"passed": all(row["passed"] for row in comparisons), "cases": comparisons}
            inputs_receipt = write_artifact(output, "inference-inputs.json", {"sequences": sequences})
            artifacts.append(inputs_receipt)
            phase = "negative_contracts"
            report["negative_contracts"] = _negative_contracts(host, bundle, config)
            artifacts.append(write_artifact(output, "negative-contracts.json", report["negative_contracts"]))
            report["deployment_variants"] = []
            for name, kwargs in (("decision-12-refused", {"decision_ticks": 12}),
                                 ("astra-profile-refused", {"model": "astra"})):
                variant = _variant(host, bundle, config, **kwargs)
                filename = f"trajectories/{name}.json.gz"
                artifacts.append(write_artifact(output, filename, variant))
                report["deployment_variants"].append({"case": name, "passed": variant["passed"], "reason": variant["reason"]})
                pairs.append({"id": name, "left": "trajectories/floor-right-24-numpy.json.gz", "right": filename,
                              "labels": ["Nominal NumPy request", name], "ticks": 24})
            phase = "throughput"
            report["throughput"] = _throughput(host, bundle, config)
            artifacts.append(write_artifact(output, "throughput.json", report["throughput"]))
            phase = "checkpoint_resume"
            report["resume"], resumed = _resume(host, training_root, bundle, config)
            for index, episode in enumerate(resumed):
                artifacts.append(write_artifact(output, f"trajectories/resume-{index}.json.gz", episode))
            artifacts.append(write_artifact(output, "resume.json", report["resume"]))
            pairs.append({"id": "fresh-episode-resume", "left": "trajectories/resume-0.json.gz",
                          "right": "trajectories/resume-1.json.gz", "labels": ["Restored boundary A", "Restored boundary B"], "ticks": 24})
        phase = "realtime_expiry"
        report["realtime"] = _realtime(host_path, godot_path, bundle, config)
        artifacts.append(write_artifact(output, "realtime.json.gz", report["realtime"]))
        phase = "numpy_process_cost"
        subprocess.run([sys.executable, "-m", "nca.bench.inference_worker", "--bundle", str(bundle_path),
                        "--sha256", bundle_hash, "--inputs", str(output / "inference-inputs.json"),
                        "--count", str(config["throughput"]["inference_samples"]), "--output", str(output / "inference-cost.json")],
                       check=True, timeout=60, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       cwd=root, env=child_environment)
        report["inference_cost"], cost_hash = read_json(output / "inference-cost.json")
        artifacts.append({"file": "inference-cost.json", "file_sha256": cost_hash,
                          "content_sha256": sha256_json(report["inference_cost"]),
                          "bytes": (output / "inference-cost.json").stat().st_size})
        phase = "learning_cost_attribution"
        training_receipt = evidence.get("training_receipt_file", "summary.json")
        training_summary, summary_file_hash = read_json(training_root / training_receipt)
        bc_cost, bc_cost_hash = read_json(training_root / "learning/bc-seed1103.json")
        updates = training_summary["ppo_updates"]
        report["learner_cost"] = {"source_receipt_file_sha256": summary_file_hash,
                                  "source_receipt_content_sha256": training_summary["sha256"],
                                  "source_receipt_file": training_receipt,
                                  "source_training_status": training_summary["status"],
                                  "original_run_total_cost_known": training_receipt == "summary.json",
                                  "ppo_rollout_updates": len(updates),
                                  "optimizer_updates": sum(row["updates"] for row in updates),
                                  "ppo_update_duration": durations([round(row["learner_seconds"]*1e9) for row in updates]),
                                  "bc_seed1103": bc_cost, "bc_cost_file_sha256": bc_cost_hash,
                                  "scope": ("retained completed update costs only; original interrupted-run totals unknown"
                                            if training_receipt != "summary.json" else
                                            "reused immutable T105 measured CPU costs; zero new optimizer updates")}
        artifacts.append(write_artifact(output, "learner-cost.json", report["learner_cost"]))
        phase = "matched_pose_clips"
        clip_spec = {"format": "nca-export-pose-pairs/v1", "pairs": pairs,
                     "camera": {"world_bounds": [128, 200, 320, 368], "panel_pixels": [576, 504]},
                     "source": "actual own-part poses; reconstruction, not engine-rendered frames",
                     "base_hz": 60, "frame_stride": 6, "frame_duration_ms": 100}
        artifacts.append(write_artifact(output, "clip-spec.json", clip_spec))
        if render_python:
            subprocess.run([render_python, "-m", "nca.bench.pose_clip", "--root", str(output)], check=True,
                           timeout=60, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           cwd=root, env=child_environment)
            clip_index, clip_hash = read_json(output / "clips.json")
            report["clips"] = clip_index
            artifacts.append({"file": "clips.json", "file_sha256": clip_hash,
                              "content_sha256": sha256_json(clip_index), "bytes": (output / "clips.json").stat().st_size})
        else:
            report["clips"] = {"complete": False, "reason": "Pillow renderer interpreter not supplied"}
        checks = (report["arithmetic"]["passed"], report["trajectories"]["passed"], report["negative_contracts"]["passed"],
                  all(row["passed"] for row in report["deployment_variants"]), report["throughput"]["passed"],
                  report["resume"]["passed"], report["realtime"]["passed"],
                  report["inference_cost"]["torch_imported"] is False,
                  report["inference_cost"]["inference"]["count"] == config["throughput"]["inference_samples"],
                  report["clips"].get("complete") is True)
        report["runtime_qualified"] = all(checks)
    except BaseException as exc:
        report["errors"].append({"phase": phase, "type": type(exc).__name__,
                                 "code": exc.code if isinstance(exc, ContractError) else "execution_failure"})
        # Keep diagnostics local to the caller; published evidence contains no
        # machine-local paths from exceptions or subprocess commands.
        import traceback
        traceback.print_exc(file=sys.stderr)
    if _sources(root) != sources:
        report["errors"].append({"phase": "source_verification", "type": "SourceDrift", "code": "source_changed"})
        report["runtime_qualified"] = False
    try:
        verify_artifact_files(training_root, training_files)
    except ContractError:
        report["errors"].append({"phase": "training_verification", "type": "ArtifactDrift", "code": "training_artifact_changed"})
        report["runtime_qualified"] = False
    report["wall_seconds"] = time.perf_counter() - started
    report["artifacts"] = artifacts
    report["status"] = "qualified" if report["runtime_qualified"] and not report["errors"] else "no_go"
    report["sha256"] = sha256_json(report)
    write_artifact(output, "summary.json", report)
    return report
