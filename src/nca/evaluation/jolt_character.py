"""Receipt-derived character diagnostics, separate from unchanged task reward.

These describe observed requests, aggregate actuator receipts and subsequent
motion. They do not infer a passive/contact wrench or causal hop effectiveness.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

from nca.contracts import ContractError, sha256_json


SCHEMA = "nca-jolt-character/v1"


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _sign(value, threshold=.08):
    return 1 if value >= threshold else -1 if value <= -threshold else 0


def character_metrics(initial: Mapping[str, Any], target, events, frames):
    """Audit one episode without a host, evaluator mutation or policy execution.

    Initial falls are airborne observations, never launch confirmations. A
    primary-associated actuator pulse needs a fresh consumed edge, cooldown
    transition, energy use and an actual upward aggregate impulse. The actuator
    receipt cannot identify the isolated contribution of each host assist.
    """
    steps = [event for event in events if event.kind == "step"]
    if len(frames) != len(steps):
        raise ContractError("character_frames", "each physical decision requires exactly one retained controller frame")
    initial_tick = initial["tick"]
    direction = 1. if target[0] >= initial["position"][0] else -1.
    ticks, pulse_ticks, impulses, angular_impulses = [], [], [], []
    unknown_linear = unknown_angular = unknown_support = airborne = blocked_supported = 0
    requested_edges = admitted_edges = consumed_edges = 0
    previous_requested = previous_admitted = previous_consumed = bool(initial.get("action", {}).get("primary", False))
    cooldown = initial.get("cooldown")
    support = initial.get("grounded")
    landings = support_losses = unexpected_losses = 0
    first_request = first_impulse = first_impulse_after_request = first_motion = None
    request_changes = trim_reversals = move_reversals = 0
    previous_action, previous_trim, previous_move = None, 0, 0
    trim_variation = move_variation = 0.
    latencies, inference, phases, reasons = [], [], Counter(), Counter()
    supervisor_decisions = Counter()
    recovery_events = []
    previous_recovery_epoch = None
    previous_recovery_phase = False
    retained_request_hashes = []

    for event, frame in zip(steps, frames):
        action = frame["action"]
        if (frame.get("tick") != event.tick_before or event.requested_packet is None
                or sha256_json(action) != sha256_json(event.requested_packet.get("action"))):
            raise ContractError("character_frames", "frame request differs from the physically recorded decision")
        retained_request_hashes.append(sha256_json(action))
        phase, reason = str(frame.get("phase", "")), str(frame.get("reason", ""))
        phases[phase] += 1
        reasons[reason] += 1
        latency = (event.wall_completed_ns - event.wall_started_ns) / 1e6
        latencies.append(latency)
        if _finite(frame.get("inference_ns")):
            inference.append(frame["inference_ns"] / 1e6)
        requested = action.get("primary") is True
        requested_edges += int(requested and not previous_requested)
        previous_requested = requested
        if first_request is None and (requested or action.get("secondary") is True or action.get("auxiliary") is True
                or abs(action.get("trim", 0.)) > 1e-9
                or any(abs(value) > 1e-9 for value in action.get("move", ()))):
            first_request = event.tick_before
        trim, move = action.get("trim", 0.), action.get("move", [0., 0.])[0]
        if previous_action is not None:
            request_changes += int(sha256_json(action) != sha256_json(previous_action))
            trim_variation += abs(trim - previous_action.get("trim", 0.))
            move_variation += abs(move - previous_action.get("move", [0., 0.])[0])
        trim_sign, move_sign = _sign(trim), _sign(move)
        trim_reversals += int(bool(trim_sign and previous_trim and trim_sign != previous_trim))
        move_reversals += int(bool(move_sign and previous_move and move_sign != previous_move))
        if trim_sign:
            previous_trim = trim_sign
        if move_sign:
            previous_move = move_sign
        previous_action = action
        telemetry = frame.get("diagnostic_state") or {}
        intervention = telemetry.get("supervisor_intervention")
        if intervention:
            supervisor_decisions[str(telemetry.get("supervisor_reason") or intervention)] += 1
        recovery_phase = ("recovery" in phase or "support_loss" in phase or phase == "recover")
        epoch = telemetry.get("recovery_epoch")
        begins_recovery = (epoch is not None and epoch != previous_recovery_epoch and epoch != 0)
        begins_recovery |= recovery_phase and not previous_recovery_phase
        if begins_recovery:
            source_view = event.source_observation or event.issued_observation
            recovery_events.append({"tick": event.tick_before, "phase": phase, "reason": reason,
                "supervisor": bool(intervention), "source_grounded": source_view.get("grounded"),
                "source_blocked_heuristic": source_view.get("blocked"),
                "source_position": list(source_view["position"]),
                "source_distance_cells": math.dist(source_view["position"], target),
                "recovery_epoch": epoch})
        if epoch is not None:
            previous_recovery_epoch = epoch
        previous_recovery_phase = recovery_phase

        for tick in event.ticks:
            view, applied = tick.observation, tick.applied
            if not tick.valid:
                raise ContractError("character_receipt", "invalid physical receipt cannot establish character")
            for submitted, admission in zip(tick.requested, tick.admission):
                if admission.get("accepted") is True:
                    admitted = submitted.get("action", {}).get("primary") is True
                    admitted_edges += int(admitted and not previous_admitted)
                    previous_admitted = admitted
            consumed = applied.get("action") if applied.get("action_available") is True else None
            current_primary = consumed.get("primary") is True if isinstance(consumed, Mapping) else False
            edge = current_primary and not previous_consumed
            consumed_edges += int(edge)
            previous_consumed = current_primary
            linear, angular = applied.get("linear_impulse"), applied.get("angular_impulse")
            linear_known = isinstance(linear, (tuple, list)) and len(linear) == 2 and all(_finite(x) for x in linear)
            angular_known = _finite(angular)
            if linear_known:
                impulses.append(tuple(linear))
            else:
                unknown_linear += 1
            if angular_known:
                angular_impulses.append(angular)
            else:
                unknown_angular += 1
            has_impulse = ((linear_known and math.hypot(*linear) > 1e-9)
                           or (angular_known and abs(angular) > 1e-9))
            if has_impulse:
                if first_impulse is None:
                    first_impulse = tick.tick_after
                if (first_impulse_after_request is None and first_request is not None
                        and tick.tick_before >= first_request):
                    first_impulse_after_request = tick.tick_after
            next_cooldown = view.get("cooldown")
            confirmed_pulse = (edge and _finite(cooldown) and cooldown <= 1 / 60 + 1e-6
                and _finite(next_cooldown) and next_cooldown >= .60
                and linear_known and linear[1] < -1e-6
                and _finite(applied.get("energy_used")) and applied["energy_used"] > 0)
            if confirmed_pulse:
                pulse_ticks.append(tick.tick_after)
            cooldown = next_cooldown
            grounded = view.get("grounded")
            airborne += int(grounded is False)
            unknown_support += int(type(grounded) is not bool)
            blocked_supported += int(grounded is True and view.get("blocked") is True)
            if support is True and grounded is False:
                support_losses += 1
                unexpected_losses += int(not any(0 <= tick.tick_after - launch <= 12 for launch in pulse_ticks))
            if support is False and grounded is True:
                landings += 1
            support = grounded
            velocity = view.get("velocity", (0., 0.))
            if first_motion is None and direction * velocity[0] >= 2.:
                first_motion = tick.tick_after
            ticks.append({"tick": tick.tick_after, "grounded": grounded,
                "blocked": view.get("blocked") is True, "position": list(view["position"]),
                "distance": math.dist(view["position"], target), "actual_pulse": confirmed_pulse,
                "actual_impulse": bool((linear_known and math.hypot(*linear) > 1e-9)
                    or (angular_known and abs(angular) > 1e-9))})

    for index, recovery in enumerate(recovery_events):
        end = recovery_events[index + 1]["tick"] if index + 1 < len(recovery_events) else math.inf
        following = [tick for tick in ticks if recovery["tick"] < tick["tick"] <= end]
        recovery["observed_followup_ticks"] = len(following)
        recovery["best_subsequent_progress_cells"] = (None if not following else
            recovery["source_distance_cells"] - min(tick["distance"] for tick in following))
        recovery["progress_at_least_one_cell"] = bool(following and recovery["best_subsequent_progress_cells"] >= 1.)
        recovery["later_supported"] = any(tick["grounded"] is True for tick in following)
        recovery["actual_primary_pulses"] = sum(tick["actual_pulse"] for tick in following)
        impulse_tick = next((tick["tick"] for tick in following if tick["actual_impulse"]), None)
        recovery["aggregate_actuation_latency_ticks"] = (None if impulse_tick is None else impulse_tick - recovery["tick"])
        recovery["interpretation"] = "subsequent observed response; no isolated recovery-actuator or causal claim"

    elapsed = len(ticks)
    airborne_after_pulse = [any(tick["grounded"] is False and 0 <= tick["tick"] - launch <= 12
                                for tick in ticks) for launch in pulse_ticks]
    def delay(tick):
        return None if tick is None else (tick - initial_tick) / 60.
    return {"schema": SCHEMA, "decisions": len(steps), "consumed_ticks": elapsed,
        "requested_primary_edges": requested_edges, "admitted_primary_edges": admitted_edges,
        "consumed_primary_edges": consumed_edges, "confirmed_primary_associated_pulses": len(pulse_ticks),
        "pulse_ticks": pulse_ticks, "pulses_with_airborne_confirmation": sum(airborne_after_pulse),
        "airborne_ticks": airborne, "airborne_seconds": airborne / 60.,
        "unknown_support_ticks": unknown_support, "supported_blocked_heuristic_ticks": blocked_supported,
        "support_loss_transitions": support_losses, "support_losses_without_recent_pulse": unexpected_losses,
        "landing_transitions": landings,
        "aggregate_linear_impulse_sum": (None if unknown_linear else [sum(x[i] for x in impulses) for i in (0, 1)]),
        "aggregate_linear_impulse_magnitude_sum": None if unknown_linear else sum(math.hypot(*x) for x in impulses),
        "aggregate_angular_impulse_sum": None if unknown_angular else sum(angular_impulses),
        "aggregate_absolute_angular_impulse_sum": None if unknown_angular else sum(map(abs, angular_impulses)),
        "unknown_linear_impulse_ticks": unknown_linear, "unknown_angular_impulse_ticks": unknown_angular,
        "first_active_request_seconds": delay(first_request), "first_aggregate_actuation_seconds": delay(first_impulse),
        "first_toward_motion_seconds": delay(first_motion),
        "request_to_aggregate_actuation_ticks": (None if first_request is None or first_impulse_after_request is None
            else first_impulse_after_request - first_request),
        "actuation_latency_interpretation": "first aggregate actuator receipt after active request; host assistance is not separately attributable",
        "request_changes": request_changes, "trim_sign_reversals": trim_reversals, "move_sign_reversals": move_reversals,
        "trim_total_variation": trim_variation, "move_total_variation": move_variation,
        "owner_roundtrip_p50_ms": _quantile(latencies, .5), "owner_roundtrip_p95_ms": _quantile(latencies, .95),
        "inference_p95_ms": _quantile(inference, .95), "phase_decisions": dict(phases), "reason_decisions": dict(reasons),
        "supervisor_intervention_decisions": sum(supervisor_decisions.values()),
        "supervisor_interventions_by_reason": dict(supervisor_decisions),
        "ordinary_recovery_events": sum(not event["supervisor"] for event in recovery_events),
        "supervisor_recovery_events": sum(event["supervisor"] for event in recovery_events),
        "recoveries_with_observed_progress": sum(event["progress_at_least_one_cell"] for event in recovery_events),
        "recovery_events": recovery_events, "requested_action_sequence_sha256": sha256_json(retained_request_hashes),
        "passive_contact_wrench": "unmeasured; native receipt reports aggregate actuator impulse separately from actual motion",
        "owner_feel_accepted": False}


def summarize_character(rows, families):
    result = {}
    for family in families:
        selected = [row["character_metrics"] for row in rows if row["family"] == family]
        if not selected:
            continue
        names = ("confirmed_primary_associated_pulses", "pulses_with_airborne_confirmation", "airborne_seconds",
                 "supported_blocked_heuristic_ticks", "ordinary_recovery_events", "supervisor_recovery_events",
                 "recoveries_with_observed_progress", "supervisor_intervention_decisions", "trim_sign_reversals",
                 "move_sign_reversals", "trim_total_variation", "move_total_variation")
        result[family] = {"episodes": len(selected),
            "mean": {name: sum(row[name] for row in selected) / len(selected) for name in names},
            "episodes_with_confirmed_airborne_pulse": sum(row["pulses_with_airborne_confirmation"] > 0 for row in selected)}
    return result
