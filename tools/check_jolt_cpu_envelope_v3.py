#!/usr/bin/env python3
"""Bounded synthetic CPU conformance for the fixed LEARN-JOLT-003 GRU32 work.

No case, native host, captured episode or checkpoint is opened. Every timed
trial starts with newly initialized random policies; a synthetic optimizer
trial performs one update and then discards those weights. This is an observed
planning envelope, not physical evidence or a universal latency guarantee.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import signal
import statistics
import sys
import time

SCRIPT_STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPETITIONS = 16
WARMUP_TRIALS = 2
PLANNING_FACTOR = 4
MAXIMUM_SECONDS = 60
SYNTHETIC_SEED = 991003


def _timeout(signum, frame):
    raise TimeoutError("bounded synthetic CPU conformance exceeded 60 seconds")


def _rss_bytes():
    measured = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(measured if sys.platform == "darwin" else measured * 1024)


def _public_cpu():
    location = Path("/proc/cpuinfo")
    if location.is_file():
        for line in location.read_text().splitlines():
            if line.startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor()


def measure():
    import numpy
    import torch
    from torch.nn import functional as F
    from nca.contracts import sha256_json
    from nca.controllers.jolt_learned import (
        FEATURE_NAMES, JoltPolicy, combine_residual, joint_log_probability,
        mapped_continuous, policy_state_sha256,
    )
    from nca.train.jolt_execution_v3 import COUNTERS
    from nca.train.jolt_learning import LearningRow, clone_loss, ordered_window

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if str(torch.__version__) != "2.8.0+cpu" or numpy.__version__ != "2.3.5":
        raise RuntimeError("the prospective pinned CPU runtime is required")
    imports_seconds = time.monotonic() - SCRIPT_STARTED
    baseline_rss = _rss_bytes()
    generator = torch.Generator().manual_seed(SYNTHETIC_SEED)
    feature_rows = torch.randn(40, len(FEATURE_NAMES), generator=generator).clamp(-1., 1.)
    labels = torch.rand(40, 7, generator=generator)
    labels[:, :2] = 2 * labels[:, :2] - 1
    labels[:, 3:5] = F.normalize(labels[:, 3:5] * 2 - 1, dim=-1)
    labels[:, 5] = (labels[:, 5] >= .5).to(torch.float32)
    labels[:, 6] = 0.  # Preserve the declared absence of secondary positives.
    latent = torch.randn(16, 5, generator=generator)
    buttons = (torch.rand(16, 2, generator=generator) >= .5).to(torch.float32)
    advantages = torch.randn(16, generator=generator)
    returns = torch.randn(16, generator=generator)

    def rows_for(mask):
        return tuple(LearningRow(tuple(float(x) for x in feature_rows[index]),
            reset=index == 0 or not mask[index - 1],
            label=tuple(float(x) for x in labels[index]) if mask[index] else None,
            eligible=mask[index], advance=mask[index], reset_after=not mask[index])
            for index in range(40))

    row_sets = {"ordinary": rows_for([True] * 40),
                "mixed": rows_for([index % 3 != 1 for index in range(40)]),
                "masked": rows_for([False] * 40)}
    # Upper-mask work: retain every ordinary forward and additionally force
    # both recurrent clears on every row. Real override masks remove forwards;
    # this synthetic superset bounds their extra reset work without new data.
    row_sets["reset_every_row"] = tuple(LearningRow(row.features, reset=True,
        label=row.label, eligible=True, advance=True, reset_after=True)
        for row in row_sets["ordinary"])
    total_model_constructions = 0
    setup_durations = []

    def setup(kind, index):
        nonlocal total_model_constructions
        started = time.perf_counter_ns()
        torch.manual_seed(SYNTHETIC_SEED + index)
        policy = JoltPolicy(kind)
        base = JoltPolicy("bc").eval().requires_grad_(False) if kind == "residual" else None
        optimizer = torch.optim.Adam(policy.parameters(), lr=.0003 if base is not None else .001)
        setup_durations.append(time.perf_counter_ns() - started)
        total_model_constructions += 1 + int(base is not None)
        return policy, base, optimizer

    def window_trial(kind, start, mask, state, total_rows):
        policy, base, optimizer = state
        rows = row_sets[mask][:total_rows]
        selected = rows[start:start + 16]
        eligible = torch.tensor([row.eligible for row in selected], dtype=torch.bool)
        counters = Counter()
        operations = {"backward_calls": 0, "optimizer_calls": 0, "physical_rows": len(selected),
                      "eligible_rows": int(eligible.sum()), "padding_rows": 16 - len(selected)}
        # This is the unchanged empty-window branch in both fixed trainers.
        if not bool(eligible.any()):
            return counters, operations
        reserve = lambda name, amount=1: counters.update({name: amount})
        stage = "bc" if kind == "bc" else "ppo"
        output = ordered_window(policy, rows, start, reserve=reserve, stage=stage, base_policy=base)
        n = len(selected)
        if kind == "bc":
            loss, _ = clone_loss(output.means[:n][eligible], output.button_logits[:n][eligible],
                                 labels[start:start + n][eligible], torch.tensor([1., 1.]))
            counters["bc_eligible_samples"] += int(eligible.sum())
        else:
            loss = .5 * F.mse_loss(output.values[:n][eligible], returns[:n][eligible])
            new = joint_log_probability(latent[:n], output.means[:n], policy.log_std,
                                         buttons[:n], output.button_logits[:n])
            # Synthetic original densities exercise both sides of the fixed
            # clipping threshold; these are not captured actor observations.
            old = new.detach() + torch.linspace(-.3, .3, n)
            ratio = (new[eligible] - old[eligible]).exp()
            advantage = advantages[:n][eligible]
            surrogate = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
            normal_entropy = (policy.log_std.clamp(-5., .5) + .5 * math.log(2 * math.pi * math.e)).sum()
            button_entropy = torch.distributions.Bernoulli(logits=output.button_logits[:n][eligible]).entropy().sum(-1).mean()
            loss = loss - surrogate.mean() - .001 * (normal_entropy + button_entropy)
            counters.update(ppo_actor_samples=int(eligible.sum()), ppo_critic_samples=int(eligible.sum()))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("synthetic conformance produced a nonfinite objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
        optimizer.step()
        counters[stage + "_optimizer"] += 1
        operations.update(backward_calls=1, optimizer_calls=1)
        if base is not None and any(parameter.grad is not None for parameter in base.parameters()):
            raise RuntimeError("synthetic residual update changed the frozen base gradient contract")
        return counters, operations

    def inference_trial(kind, state):
        policy, base, _ = state
        counters = Counter()
        hidden = policy.zero_hidden()
        base_hidden = None if base is None else base.zero_hidden()
        unit_durations = []
        with torch.no_grad():
            for index in range(40):
                unit_began = time.perf_counter_ns()
                values = torch.tensor([tuple(float(x) for x in feature_rows[index])])
                means, logits, value, hidden = policy(values, hidden)
                counters["model_forwards"] += 1
                if base is not None:
                    base_means, base_logits, _, base_hidden = base(values, base_hidden)
                    counters["model_forwards"] += 1
                    _, logits = combine_residual(base_means, base_logits, means, logits)
                joint_log_probability(latent[index % 16:index % 16 + 1], means,
                                      policy.log_std, buttons[index % 16:index % 16 + 1], logits)
                mapped_continuous(means)
                # Include finite policy identity hashing and synthetic feature
                # hashing, while explicitly excluding native observation replay.
                policy_state_sha256(policy)
                if base is not None:
                    policy_state_sha256(base)
                sha256_json(tuple(float(x) for x in feature_rows[index]))
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError("nonfinite synthetic value")
                unit_durations.append(time.perf_counter_ns() - unit_began)
        return counters, {"backward_calls": 0, "optimizer_calls": 0, "synthetic_rows": 40,
                          "unit_duration_ns": unit_durations}

    specifications = []
    for kind in ("bc", "residual"):
        specifications.append((kind + "_inference40", kind, None, "ordinary", 40))
        specifications.extend((f"{kind}_window_end{end}_{mask}", kind,
                               16 * ((end - 1) // 16), mask, end)
                              for end in range(1, 41) for mask in ("ordinary", "reset_every_row"))
        specifications.extend((f"{kind}_window_end32_{mask}", kind, 16, mask, 32)
                              for mask in ("mixed", "masked"))
    results = {}
    total_operations = Counter()
    for profile, kind, start, mask, total_rows in specifications:
        elapsed, expected_counts, expected_operations = [], None, None
        all_counts, all_operations = Counter(), Counter()
        measured_unit_durations = []
        for index in range(WARMUP_TRIALS + REPETITIONS):
            # No optimizer result is reused by the next trial or profile.
            state = setup(kind, len(results) * 1000 + index)
            began = time.perf_counter_ns()
            counts, operations = (inference_trial(kind, state) if start is None
                                  else window_trial(kind, start, mask, state, total_rows))
            duration = time.perf_counter_ns() - began
            units = operations.pop("unit_duration_ns", [])
            if expected_counts is None:
                expected_counts, expected_operations = counts, operations
            if counts != expected_counts or operations != expected_operations:
                raise RuntimeError("synthetic workload changed its exact operation count")
            all_counts.update(counts)
            all_operations.update(operations)
            if index >= WARMUP_TRIALS:
                elapsed.append(duration)
                measured_unit_durations.extend(units)
            del state
        total_operations.update(all_counts)
        results[profile] = {
            "kind": kind, "window_start": start, "episode_rows": total_rows, "mask": mask,
            "timed_trials": REPETITIONS, "warmup_trials": WARMUP_TRIALS,
            "counts_per_trial": dict(expected_counts), "operations_per_trial": expected_operations,
            "all_trial_counts_including_warmup": dict(all_counts),
            "all_trial_operations_including_warmup": dict(all_operations),
            "duration_ns": elapsed, "maximum_seconds": max(elapsed) / 1e9,
            "median_seconds": statistics.median(elapsed) / 1e9,
            "measured_individual_inference_row_duration_ns": measured_unit_durations,
        }
    bc_max = max(row["maximum_seconds"] for name, row in results.items() if name.startswith("bc_window"))
    ppo_max = max(row["maximum_seconds"] for name, row in results.items() if name.startswith("residual_window"))
    # Use the largest individually timed row, never a grouped average. The
    # residual row includes two modules and is conservatively charged to each
    # module forward in the published caps.
    forward_max = max(duration for row in results.values()
                      for duration in row["measured_individual_inference_row_duration_ns"]) / 1e9

    def episode_costs(kind):
        window = {end: max(results[f"{kind}_window_end{end}_{mask}"]["maximum_seconds"]
                           for mask in ("ordinary", "reset_every_row")) for end in range(1, 41)}
        window[32] = max(window[32], results[f"{kind}_window_end32_mixed"]["maximum_seconds"],
                         results[f"{kind}_window_end32_masked"]["maximum_seconds"])
        return {length: sum(window[min(length, start + 16)] for start in range(0, length, 16))
                for length in range(1, 41)}

    def maximum_partition(weights):
        totals, partitions = [0.] + [-float("inf")] * 256, [[] for _ in range(257)]
        for total in range(1, 257):
            for length in range(1, min(40, total) + 1):
                candidate = totals[total - length] + weights[length]
                if candidate > totals[total]:
                    totals[total] = candidate
                    partitions[total] = partitions[total - length] + [length]
        return totals[256], partitions[256]

    bc_costs, ppo_costs = episode_costs("bc"), episode_costs("residual")
    ppo_update_epoch_max, ppo_partition = maximum_partition(ppo_costs)
    prefix_rows = {length: sum(range(0, length, 16)) for length in range(1, 41)}
    prefix_max, prefix_partition = maximum_partition(prefix_rows)
    if int(prefix_max) * 24 * 4 != COUNTERS["ppo_burn_in"]:
        raise RuntimeError("source-derived prefix partition differs from the frozen ceiling")
    bc_worst_length = max(bc_costs, key=bc_costs.get)
    coupled_counts = {
        "bc_optimizer": 3 * 40 * 64 * math.ceil(bc_worst_length / 16),
        "bc_forward": 3 * 40 * 64 * bc_worst_length,
        "bc_burn_in": 3 * 40 * 64 * prefix_rows[bc_worst_length],
        "ppo_optimizer": 24 * 4 * sum(math.ceil(length / 16) for length in ppo_partition),
        "ppo_forward": 24 * 4 * sum(ppo_partition),
        "ppo_burn_in": 24 * 4 * sum(prefix_rows[length] for length in ppo_partition),
    }
    if any(value > COUNTERS[key] for key, value in coupled_counts.items()):
        raise RuntimeError("coupled timing partition exceeds a frozen operation ceiling")
    active_forward_count = sum(COUNTERS[key] for key in (
        "inference_forward", "bootstrap_forward", "ppo_replay_forward", "ppo_replay_base_forward"))
    initialization_bound = 6 + 2 * COUNTERS["checkpoint_loads"]
    components = {
        "bc_64_episodes_40_epochs_3_seeds_including_prefix_and_optimizer": 3 * 40 * 64 * bc_costs[bc_worst_length],
        "ppo_256_row_partition_24_updates_4_epochs_including_prefix_and_optimizer": 24 * 4 * ppo_update_epoch_max,
        "active_nonfitting_forward_and_density_hash_units": active_forward_count * forward_max,
        "model_and_optimizer_initializations": initialization_bound * max(setup_durations) / 1e9,
    }
    source_paths = ("tools/check_jolt_cpu_envelope_v3.py", "src/nca/controllers/jolt_learned.py",
                    "src/nca/train/jolt_learning.py", "src/nca/train/jolt_execution_v3.py")
    source_files = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths}
    cgroup = {}
    for name in ("memory.max", "memory.current", "memory.peak", "cpu.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            cgroup[name] = path.read_text().strip()
    return {
        "schema": "nca-jolt-synthetic-cpu-conformance/v3", "completed": True,
        "scope": "new random GRU32 models and synthetic tensors only; zero host/case/checkpoint evidence",
        "physical_experiment": False, "prospective_experiment_admitted": False,
        "source_files": source_files, "source_sha256": sha256_json(source_files),
        "runtime": {"torch": str(torch.__version__), "numpy": numpy.__version__,
                    "python": platform.python_version(), "cpu_model": _public_cpu(),
                    "platform": platform.system(), "machine": platform.machine(),
                    "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "fixed_design": {"repetitions": REPETITIONS, "warmup_trials": WARMUP_TRIALS,
                         "planning_factor": PLANNING_FACTOR, "maximum_seconds": MAXIMUM_SECONDS,
                         "synthetic_seed": SYNTHETIC_SEED, "physical_window_length": 16,
                         "maximum_episode_rows": 40, "maximum_prefix_rows": 32,
                         "fresh_policies_every_trial": True, "weights_saved_or_reused": False},
        "profiles": results, "all_measured_operation_counts": dict(total_operations),
        "model_constructions": total_model_constructions,
        "initialization_duration_ns": setup_durations,
        "original_counter_caps_unchanged": dict(COUNTERS),
        "planning_projection": {
            "method": "maximum measured unit times declared count, multiplied by fixed factor4; not a proven worst-case bound",
            "window_method": "all feasible suffix lengths1..40 timed separately; use max ordinary/reset-mask time for each exact physical window and dynamic-program every episode partition totaling256",
            "bc_cost_seconds_by_episode_length": bc_costs,
            "ppo_cost_seconds_by_episode_length": ppo_costs,
            "bc_maximum_cost_episode_length": bc_worst_length,
            "ppo_maximum_cost_partition": ppo_partition,
            "ppo_maximum_cost_per_update_epoch_seconds": ppo_update_epoch_max,
            "source_prefix_maximum_per_update_epoch": int(prefix_max),
            "source_prefix_maximum_partition": prefix_partition,
            "coupled_operation_counts_for_timing_maximizers": coupled_counts,
            "uncoupled_rectangle_before_factor_seconds": {
                "bc": COUNTERS["bc_optimizer"] * bc_max,
                "ppo": COUNTERS["ppo_optimizer"] * ppo_max},
            "active_components_before_factor_seconds": components,
            "active_neural_and_optimizer_seconds": PLANNING_FACTOR * sum(components.values()),
            "audit_neural_seconds": PLANNING_FACTOR * COUNTERS["audit_forward"] * forward_max,
            "active_nonfitting_forward_units": active_forward_count,
            "audit_forward_units": COUNTERS["audit_forward"],
            "initialization_units": initialization_bound,
            "initialization_unit_includes_one_bc_or_one_residual_plus_base_and_adam": True,
        },
        "memory": {"baseline_process_peak_rss_bytes": baseline_rss,
                   "final_process_peak_rss_bytes": _rss_bytes(),
                   "shared_cgroup_snapshot_not_attributable_to_this_process": cgroup},
        "imports_and_runtime_initialization_seconds": imports_seconds,
        "whole_script_elapsed_before_report_seconds": time.monotonic() - SCRIPT_STARTED,
        "limitations": [
            "Observed maxima with a planning multiplier are not hard upper latency bounds under all scheduling or storage conditions.",
            "No actual native host, transport, physical observation decoding, retirement replay, durable journal, ledger, gzip, full artifact hashing or filesystem sync is timed here.",
            "CPU training-window tensors are measured; complete retained episode/auditor Python objects and the native process are outside this RSS observation.",
            "No learned movement quality, case latency, output qualification or experimental result follows from this synthetic report.",
            "Complete prospective source/runtime/time/memory/storage admission must independently resolve these missing terms before any experiment.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; original synthetic evidence is immutable")
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(MAXIMUM_SECONDS)
    try:
        result = measure()
    finally:
        signal.alarm(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps({"completed": True, "elapsed_seconds": result["whole_script_elapsed_before_report_seconds"],
                      "peak_rss_bytes": result["memory"]["final_process_peak_rss_bytes"],
                      "planning_projection": result["planning_projection"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
