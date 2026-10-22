"""Original elapsed clock and versioned operation contract for LEARN-JOLT-003.

Importing this module starts no host, model, process or experiment. A valid
contract is necessary but cannot replace its independently reviewed admission.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import math
import time

from nca.contracts import ContractError
from nca.train.jolt_execution_contract import validate_v3_allocation_identity

DECLARATION_SCHEMA = "nca-learn-jolt-003-execution/v3"
RUN_SCHEMA = "nca-learn-jolt-003-run/v1"
MAXIMUM_WALL_SECONDS = 7200
CLEANUP_RESERVE_SECONDS = 1200
VERIFICATION_RESERVE_SECONDS = 300
FINALIZATION_RESERVE_SECONDS = 60
# Twelve seconds of each child phase are withheld for process-tree revocation,
# reaping and independent stream fsync. This comes from the same phase, never
# from a newly started clock or the following phase's allowance.
CHILD_CLEANUP_SECONDS = 12
PHASE_OFFSETS = {
    "active": 6000,
    "closure": 6180,
    "audit": 6780,
    "publication": 6900 - CHILD_CLEANUP_SECONDS,
    "worker_cleanup": 6900,
    "verification": 7140 - CHILD_CLEANUP_SECONDS,
    "verification_cleanup": 7140,
    "finalization": 7200,
}
WORKER_PHASES = ("active", "closure", "audit", "publication")
LEGACY_TIMING_SCALARS = {
    "maximum_owner_transport_seconds": 15, "maximum_wall_seconds": 7200,
    "cleanup_reserve_seconds": 1200, "executor_verification_reserve_seconds": 300,
    "executor_finalization_reserve_seconds": 60,
}
COUNTERS = {
    "attempts": 1, "audit_forward": 37888, "bc_burn_in": 368640,
    "bc_eligible_samples": 307200, "bc_forward": 307200, "bc_optimizer": 23040,
    "bootstrap_forward": 12288, "checkpoint_executions": 21144,
    "checkpoint_loads": 48, "checkpoint_writes": 36, "confirmed_ticks": 120000,
    "decisions": 20000, "episodes_aborted": 6464, "episodes_completed": 6464,
    "episodes_started": 6464, "host_start_attempts": 1, "host_starts": 1,
    "host_terminations": 1, "inference_forward": 25248, "owner_cancel": 6465,
    "owner_close": 1, "owner_configure": 0, "owner_observe": 0,
    "owner_profile": 6464, "owner_requests": 38338, "owner_reset": 6464,
    "owner_step": 18944, "ppo_actor_samples": 24576, "ppo_base_burn_in": 33792,
    "ppo_base_forward": 24576, "ppo_burn_in": 33792, "ppo_critic_samples": 24576,
    "ppo_forward": 24576, "ppo_optimizer": 24576, "ppo_replay_base_forward": 6144,
    "ppo_replay_forward": 6144, "refused_ticks": 120000,
    "requested_ticks": 120000, "uncertain_ticks": 120000,
}
STAGES = {
    "teacher": {"decisions": 2560, "episodes_started": 64, "requested_ticks": 15360},
    "ppo": {"decisions": 6144, "episodes_started": 6144, "requested_ticks": 36864},
    "open": {"decisions": 1280, "episodes_started": 32, "requested_ticks": 7680},
    "sealed": {"decisions": 8960, "episodes_started": 224, "requested_ticks": 53760},
}
RESERVATION_RECORDS_MAX = sum(COUNTERS.values())
assert len(COUNTERS) == 39 and RESERVATION_RECORDS_MAX == 1901556


def validate_budget_v3(budget):
    expected_scalars = {
        "maximum_attempts": 1, "automatic_retry": False, "automatic_extension": False,
        "device": "cpu", "gpu_allowed": False, "torch_threads": 1,
        **LEGACY_TIMING_SCALARS,
    }
    if not isinstance(budget, dict) or set(budget) != set(expected_scalars) | {"counters", "stages"}:
        raise ContractError("jolt_v3_budget", "the exact separately versioned allocation is required")
    # Explicit null removes every execution/transport wall limit together. An
    # omitted field, partial conversion or large finite substitute is not this
    # mode. Original finite budgets retain their historical interpretation.
    if all(budget[key] is None for key in LEGACY_TIMING_SCALARS):
        expected_scalars.update({key: None for key in LEGACY_TIMING_SCALARS})
    for key, expected in expected_scalars.items():
        value = budget.get(key)
        if type(value) is not type(expected) or value != expected:
            raise ContractError("jolt_v3_budget", "an allocation scalar changed: " + key)
    for key, expected in (("counters", COUNTERS), ("stages", STAGES)):
        if budget.get(key) != expected:
            raise ContractError("jolt_v3_budget", "the complete operation and stage ceilings are fixed")
    if (any(type(value) is not int for value in budget["counters"].values())
            or any(type(value) is not int for stage in budget["stages"].values() for value in stage.values())):
        raise ContractError("jolt_v3_budget", "counter and stage ceilings must be integers")
    return budget


def phase_offsets_v3(maximum_wall_seconds=MAXIMUM_WALL_SECONDS):
    """Return the exact historical cutoffs or explicit unlimited phase values."""
    if maximum_wall_seconds is None:
        return dict.fromkeys(PHASE_OFFSETS)
    if type(maximum_wall_seconds) is not int or maximum_wall_seconds != MAXIMUM_WALL_SECONDS:
        raise ContractError("jolt_v3_clock", "the original wall limit or explicit null is required")
    return dict(PHASE_OFFSETS)


def validate_allocation_identity_v3(declaration):
    if not isinstance(declaration, dict) or declaration.get("schema") != DECLARATION_SCHEMA:
        raise ContractError("jolt_v3_declaration", "a LEARN-JOLT-003 execution-v3 declaration is required")
    # Identity syntax is shared, not its experiment or execution semantics.
    return validate_v3_allocation_identity(declaration)


class OriginalClockV3:
    """One finite monotonic origin with historical or explicitly absent limits.

    A monotonic epoch is arbitrary. Only elapsed time and non-renewal matter;
    low-uptime systems and a synthetic origin of zero are valid.
    """
    def __init__(self, started, *, maximum_wall_seconds=MAXIMUM_WALL_SECONDS,
                 monotonic=time.monotonic):
        now = monotonic()
        if (type(started) not in (int, float) or not math.isfinite(started)
                or type(now) not in (int, float) or not math.isfinite(now) or started > now):
            raise ContractError("jolt_v3_clock", "one original finite monotonic origin is required")
        self.started = float(started)
        self._clock = monotonic
        self.maximum_wall_seconds = maximum_wall_seconds
        self.phase_offsets = phase_offsets_v3(maximum_wall_seconds)

    def elapsed(self):
        value = self._clock() - self.started
        if not math.isfinite(value) or value < 0:
            raise ContractError("jolt_v3_clock", "the original monotonic clock moved backwards")
        return value

    def remaining(self, phase):
        if phase not in self.phase_offsets:
            raise ContractError("jolt_v3_phase", "an explicit fixed execution phase is required")
        elapsed = self.elapsed()
        offset = self.phase_offsets[phase]
        return None if offset is None else max(0., offset - elapsed)

    def check(self, phase, *, operation_seconds=0):
        if operation_seconds is None and self.maximum_wall_seconds is None:
            pass
        elif (type(operation_seconds) not in (int, float) or not math.isfinite(operation_seconds)
              or operation_seconds < 0):
            raise ContractError("jolt_v3_clock", "a finite nonnegative operation allowance is required")
        if phase not in self.phase_offsets:
            raise ContractError("jolt_v3_phase", "an explicit fixed execution phase is required")
        elapsed = self.elapsed()
        offset = self.phase_offsets[phase]
        if offset is not None and elapsed + operation_seconds >= offset:
            raise ContractError("jolt_v3_phase_deadline", "the original " + phase + " deadline is exhausted")
