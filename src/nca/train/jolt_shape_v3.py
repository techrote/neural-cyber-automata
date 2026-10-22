"""Source-shaped serialization ceilings for the frozen one-actor Jolt workload.

These are shape bounds, not observations or admitted physical evidence. Every
float uses a 24-byte finite binary64 representation, every native integer its
20-byte signed int64 envelope, and only fixed host strings are admitted. The
pinned Jev rig has three parts, two rotary joints, no tools/thrusters/effects;
its local patch contains exactly 169 unsigned bytes in each of two arrays.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

from copy import deepcopy

from nca.contracts import canonical_json

FLOAT = -1.7976931348623157e308
INTEGER = -9223372036854775808
SCHEDULE = "synchronous-mask-actuation-effects-material-feedback-terrain-rapier/v1"


def action_shape():
    result = {key: [FLOAT, FLOAT] for key in ("move", "aim")}
    result.update({key: FLOAT for key in ("effort", "trim", "reel")})
    result.update({key: False for key in ("primary", "secondary", "auxiliary", "tool", "release_usable", "attach_usable")})
    return result


def observation_shape():
    """All keys emitted by the pinned rig/session/arena for complete Jev data."""
    value = {key: INTEGER for key in ("version", "actor_slot", "epoch", "tick", "facing",
        "part_count", "mode", "ray_queries", "actor_id", "profile_version", "lease_remaining",
        "last_sequence", "control_generation", "pending_input_edges", "environment_version")}
    value.update({key: FLOAT for key in ("rotation", "angular_velocity", "energy", "charge", "cooldown",
        "gravity", "rope_length", "tension", "gait_phase", "stuck_time", "applied_torque_impulse",
        "energy_used", "ground_distance", "wall_left", "wall_right", "ceiling_distance")})
    value.update({key: [FLOAT, FLOAT] for key in ("position", "centre", "velocity", "contact_normal",
        "anchor", "applied_impulse", "thruster_levels")})
    value.update({key: False for key in ("grounded", "rope_attached", "gripped", "blocked", "tool_ready",
        "power_enabled", "powered", "flight_mode", "coast_mode", "movement_limited", "suspended")})
    value.update(representation="articulated-rapier", rig_id="jev", run_id="f" * 32,
        observation_id="f" * 32 + (":" + "9" * 20) * 4, model="jev", producer="policy",
        control_mode="wire", schedule=SCHEDULE, observation_scope="self, rig rays and bounded composed native cells",
        # Complete successful or safely retired episodes have no native fault.
        # Allow the longest source fault plus padding; failures still abort.
        fault="x" * 64, action=action_shape(), thrust_weights=[], last_effects=[])
    part = {"id": "rebound_pad_-1", "index": INTEGER, "position": [FLOAT, FLOAT], "rotation": FLOAT,
        "velocity": [FLOAT, FLOAT], "angular_velocity": FLOAT, "size": [FLOAT, FLOAT], "mass": FLOAT, "detached": False}
    joint = {"id": "pad_flex_-1", "type": "rotary", "a": INTEGER, "b": INTEGER, "enabled": False,
        "target": FLOAT, "angle": FLOAT, "speed": FLOAT, "limits": [FLOAT, FLOAT],
        "limits_enabled": False, "anchor_error": FLOAT}
    value.update(parts=[deepcopy(part) for _ in range(3)], joints=[deepcopy(joint) for _ in range(2)],
        cell_patch={"origin": [INTEGER, INTEGER], "size": [13, 13], "materials": [255] * 169,
            "body_ids": [255] * 169, "sample_tick": INTEGER, "body_mask_pose_tick": INTEGER})
    return value


def command_shape():
    view = observation_shape()
    return {"version": INTEGER, "kind": "physical_wire_action", "run_id": view["run_id"], "actor_id": INTEGER,
        "epoch": INTEGER, "sequence": INTEGER, "observed_tick": INTEGER,
        "observation_id": view["observation_id"], "lease_ticks": INTEGER, "action": action_shape()}


def trace_shape(*, first=True):
    observation = observation_shape()
    return {"tick_before": INTEGER, "tick_after": INTEGER, "world_tick": INTEGER,
        "received_monotonic_usec": INTEGER, "completed_monotonic_usec": INTEGER,
        "requested": [command_shape()] if first else [],
        "admission": [{"accepted": False, "reason": "x" * 64}] if first else [],
        "observation": observation, "failed": False,
        "applied": {"action": action_shape(), "action_available": False, "terminal_revocation": False,
            "linear_impulse": [FLOAT, FLOAT], "angular_impulse": FLOAT, "energy_used": FLOAT},
        "passive": {"available": False,
            "reason": "ACT004 publishes aggregate actuation and actual motion, not a separate passive wrench"},
        "actual_motion": {"position": [FLOAT, FLOAT], "velocity": [FLOAT, FLOAT],
            "rotation": FLOAT, "angular_velocity": FLOAT}, "cellular_sample_tick": INTEGER}


def response_shape():
    return {"observation": observation_shape(), "traces": [trace_shape(first=i == 0) for i in range(6)],
        "start_tick": INTEGER, "tick": INTEGER, "actual_ticks": INTEGER,
        "finished": False, "failed": False, "reason": "x" * 64, "terminated": False, "truncated": False,
        "cumulative_reward": None, "reward_kind": "no_task_reward", "schedule": SCHEDULE,
        "world_tick": INTEGER, "ticks_without_reader": INTEGER, "idle_trace": [], "idle_trace_dropped": INTEGER,
        "unconsumed_terminal_commands": [], "protocol": "nca-host/1", "id": INTEGER, "ok": False}


def linked_response_shape(sequence=115110):
    return {"event": "owner_response", "serial": 38338, "response": response_shape(),
        "_journal": {"sequence": sequence, "previous_sha256": "f" * 64, "record_sha256": "f" * 64}}


def shape_bounds_v3():
    observation, command, trace, response = map(lambda f: len(canonical_json(f()).encode()),
        (observation_shape, command_shape, trace_shape, response_shape))
    linked = len(canonical_json(linked_response_shape()).encode()) + 1
    # Capture repeats the before/source/after observations and a policy subset;
    # count that subset conservatively as another complete observation. Each
    # TickRecord adds consumed command/source/action to its original trace.
    tick_record = trace + command + len(canonical_json(action_shape()).encode()) + 512
    decision = 6 * tick_record + 4 * observation + command + 8192
    # Episode fixed fields include a <=16 KiB source manifest, host/profile/config
    # <=16 KiB together, initial observation, one cancellation event (3 views),
    # and <=8 KiB fixed retirement/outcome/metric metadata. Recovery-event
    # arrays, controller frames, reward terms, and per-decision phase/reason
    # counts belong to the 8 KiB PER-DECISION allowance above. There are at
    # most forty recovery events, each below 1 KiB from its fixed scalar shape;
    # the authored metrics cap is consequently 48 KiB, not a fixed-base cost.
    # Source manifest and resolved identities still require their stated bounds.
    base = 2 * 16384 + 4 * observation + 8192
    return {"schema": "nca-jolt-source-shape/v3", "binary64_json_bytes": 24, "native_int64_json_bytes": 20,
        "parts": 3, "rotary_joints": 2, "patch_cells_per_array": 169,
        "mounts_thrusters_effects": 0, "ticks_per_decision": 6, "native_observation_bytes": observation,
        "native_command_bytes": command, "native_trace_bytes": trace, "native_response_bytes": response,
        "linked_owner_response_bytes": linked, "capture_tick_record_bytes": tick_record,
        "capture_decision_bytes": decision, "episode_fixed_bytes": base,
        "per_decision_metadata_allowance_bytes": 8192, "fixed_outcome_retirement_metrics_bytes": 8192,
        "recovery_event_per_decision_allowance_bytes": 1024,
        "source_manifest_allowance_bytes": 16384, "host_profile_config_allowance_bytes": 16384,
        "requires_exact_source_and_fixed_metadata_binding": True,
        "timing_or_physical_feasibility_proved": False}
