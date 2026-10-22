"""Versioned CPU portability and approximate diagnostic normalization for Jolt v3.

Hardware identity is OBSERVATIONAL rather than an experiment-admission key.
The pinned software/device/architecture and source/host locks remain strict.
Measured speed normalization is approximate telemetry, never a policy reward,
trained-model selection, latency-quality adjustment or physical outcome.
© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import hashlib
import math
import statistics
import time

from nca.contracts import ContractError, sha256_json

RUNTIME_CONTRACT_SCHEMA = "nca-jolt-cpu-runtime-contract/v1"
CALIBRATION_SCHEMA = "nca-jolt-cpu-calibration/v1"
NORMALIZATION_SCHEMA = "nca-jolt-cpu-normalization/v1"
POLICY = "cpu-hardware-observational-and-normalized/v1"
CALIBRATION_BLOCK = bytes(range(256)) * 256
CALIBRATION_ITERATIONS = 1024
CALIBRATION_REPETITIONS = 3
FIELDS_ADVISORY = ("cpu_model", "kernel", "logical_cpus_available", "libc")


def capability_contract_v3(runtime):
    """A stable Linux CPU SOFTWARE contract for admission, not a node identity.

    Python patch level, CPU model, Linux kernel, available cores and libc
    versions may vary across ephemeral runners. Every fact remains recorded
    in the original declaration and worker receipts; none changes the model
    weights, fixed seeds, native host, or acceptance thresholds.
    """
    if not isinstance(runtime, dict):
        raise ContractError("jolt_cpu_runtime", "runtime record is required")
    pinned = {
        "torch": "2.8.0+cpu", "numpy": "2.3.5", "device": "cpu",
        "system": "Linux", "threads": 1, "interop_threads": 1,
        "deterministic_algorithms": True,
    }
    for key, expected in pinned.items():
        value = runtime.get(key)
        if type(value) is not type(expected) or value != expected:
            raise ContractError("jolt_cpu_runtime", "pinned CPU capability differs: " + key)
    version = runtime.get("python")
    if not isinstance(version, str) or len(version) > 40 or version.split(".")[:2] != ["3", "12"]:
        raise ContractError("jolt_cpu_runtime", "Python 3.12 is required")
    machine = runtime.get("machine")
    if machine not in ("x86_64", "AMD64"):
        raise ContractError("jolt_cpu_runtime", "Linux x86_64 is required")
    if (not isinstance(runtime.get("cpu_model"), str) or
            not 0 < len(runtime["cpu_model"]) <= 240 or
            not isinstance(runtime.get("kernel"), str) or
            not 0 < len(runtime["kernel"]) <= 240 or
            type(runtime.get("logical_cpus_available")) is not int or
            runtime["logical_cpus_available"] < 1):
        raise ContractError("jolt_cpu_runtime", "advisory hardware facts must be measured, not omitted")
    libc = runtime.get("libc")
    if (not isinstance(libc, (tuple, list)) or len(libc) != 2
            or any(not isinstance(x, str) or len(x) > 80 for x in libc)):
        raise ContractError("jolt_cpu_runtime", "bounded libc observation is required")
    return {
        "schema": RUNTIME_CONTRACT_SCHEMA, "policy": POLICY,
        **pinned, "python_minor": "3.12", "machine": "x86_64",
        "hardware_advisory_only": list(FIELDS_ADVISORY),
        "performance_normalization_applies_only_to_diagnostic_elapsed_times": True,
        "training_inference_latency_gate_unchanged": True,
    }


def capability_sha256(runtime):
    return sha256_json(capability_contract_v3(runtime))


def normalize_diagnostic_seconds(seconds, measured_calibration, reference_calibration):
    """Map wall seconds to a declared *reference CPU* rate, for ballpark graphs.

    Faster measured CPUs map to longer reference time, slower CPUs to shorter.
    Never use this output to judge the <=5 ms actual-case inference p95.
    """
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ContractError("jolt_cpu_normalization", "measured duration must be finite and nonnegative")
    def rate(record):
        if (not isinstance(record, dict) or record.get("schema") != CALIBRATION_SCHEMA
                or record.get("method") != "sha256_64kib_fixed_buffer_v1"
                or type(record.get("bytes_per_second")) not in (float, int)
                or not math.isfinite(record["bytes_per_second"])
                or record["bytes_per_second"] <= 0):
            raise ContractError("jolt_cpu_normalization", "matching positive calibrations are required")
        return record["bytes_per_second"]
    observed, reference = rate(measured_calibration), rate(reference_calibration)
    return {
        "schema": NORMALIZATION_SCHEMA,
        "method": "raw_seconds_times_observed_calibration_rate_over_reference_rate",
        "raw_seconds": float(seconds),
        "reference_seconds_approximate": float(seconds) * observed / reference,
        "observed_to_reference_speed_ratio": observed / reference,
        "quality_thresholds_or_physical_results_normalized": False,
        "planning_only_not_execution_admission_proof": True,
    }


def cpu_calibration_v3(*, repetitions=CALIBRATION_REPETITIONS,
                       iterations=CALIBRATION_ITERATIONS, monotonic=time.perf_counter):
    """Small deterministic single-core C/hash throughput proxy, no neural work."""
    if type(repetitions) is not int or type(iterations) is not int or not (
            1 <= repetitions <= 16 and 1 <= iterations <= 4096):
        raise ContractError("jolt_cpu_calibration", "fixed bounded calibration shape required")
    measured = []
    for _ in range(repetitions):
        before = monotonic()
        for _n in range(iterations):
            hashlib.sha256(CALIBRATION_BLOCK).digest()
        seconds = monotonic() - before
        if not math.isfinite(seconds) or seconds <= 0:
            raise ContractError("jolt_cpu_calibration", "positive monotonic time required")
        measured.append(seconds)
    return {
        "schema": CALIBRATION_SCHEMA, "method": "sha256_64kib_fixed_buffer_v1",
        "block_bytes": len(CALIBRATION_BLOCK), "iterations": iterations,
        "repetitions": repetitions,
        "measured_seconds": measured,
        "bytes_per_second": iterations * len(CALIBRATION_BLOCK) / statistics.median(measured),
        "single_core_proxy_only": True,
        "not_a_host_or_neural_workload": True,
        "runtime_identity_not_a_gate": True,
    }
