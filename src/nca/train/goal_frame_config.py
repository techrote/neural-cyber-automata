"""Fixed declaration and fail-closed utility decision for the Jolt follow-up.

This module imports neither Torch nor a host. The original T105 result and
source-bound T106 entry decision are never overwritten by this new experiment.
"""

from __future__ import annotations

from itertools import product
import math
from pathlib import Path
from typing import Any, Mapping

from nca.contracts import ContractError, sha256_json, strict_json_loads
from nca.controllers.morphology.admission import entry_decision
from nca.randomness import ScenarioSplit

FOLLOWUP_SHA256 = "af936549f71bc72bc7fabfc8c75b2d04b58a8348635b5f47ddec50ea904a7687"
METRICS = ("success", "capped_completion_seconds", "ordinary_trim_reversals_per_second",
           "mean_absolute_angle_radians", "peak_toward_speed_cells_per_second", "purposeful_launch")


def read_followup_config(path: Path) -> dict[str, Any]:
    value = strict_json_loads(path.read_bytes(), max_bytes=4 * 1024 * 1024)
    validate_followup_config(value)
    return value


def validate_followup_config(config: Mapping[str, Any]) -> None:
    if sha256_json(config) != FOLLOWUP_SHA256:
        raise ContractError("goal_frame_declaration", "the pre-collection follow-up declaration differs")
    ScenarioSplit(tuple(row["id"] for row in config["training_families"]), (),
                  tuple(row["id"] for row in config["evaluation_families"]))
    families = config["candidates_in_priority_order"]
    demonstration = sum(len(row["demo_seeds"]) for row in config["training_families"]) * 40
    evaluation = len(config["evaluation_families"]) * (2 + 2 * len(families) * len(config["training_seeds"])) * 40
    if (demonstration != 960 or evaluation != 4480 or demonstration + evaluation != 5440
            or config["budget"]["maximum_total_decisions"] != 6000
            or config["budget"]["maximum_wall_seconds"] != 1800):
        raise ContractError("goal_frame_budget", "the whole fixed schedule must fit its declared allocation")


def verify_original_boundary(config: Mapping[str, Any], root: Path) -> None:
    """Bind unchanged utility thresholds and exclude every previously known case."""
    validate_followup_config(config)
    original = strict_json_loads((root / "configs/first-skill/jolt-screen-v1.json").read_bytes())
    gate = strict_json_loads((root / "configs/morphology/entry-gate-v1.json").read_bytes())
    prior_result = strict_json_loads((root / "docs/evidence/t106-prerequisite-result.json").read_bytes())
    prior_decision = entry_decision(gate, prior_result)
    if (sha256_json(original) != config["original_evidence"]["t105_config_sha256"]
            or sha256_json(gate) != config["original_evidence"]["t106_entry_declaration_sha256"]
            or prior_decision["prerequisite_result_sha256"] != config["original_evidence"]["t106_prerequisite_result_sha256"]
            or prior_decision["t105_summary_sha256"] != config["original_evidence"]["t105_summary_sha256"]
            or any(config["prerequisite"][key] != gate[key] for key in config["prerequisite"])):
        raise ContractError("goal_frame_original_gate", "original evidence and utility criteria must remain unchanged")
    known = {(tuple(row["spawn"]), tuple(row["target"]))
             for phase in ("training", "tuning", "evaluation") for row in original[f"{phase}_families"]}
    fresh = {(tuple(row["spawn"]), tuple(row["target"])) for row in config["evaluation_families"]}
    training = {(tuple(row["spawn"]), tuple(row["target"])) for row in config["training_families"]}
    if len(fresh) != 8 or fresh & (known | training):
        raise ContractError("goal_frame_holdout", "new sealed geometry must be absent from every previously known case")


def _interval(values):
    means = sorted(sum(sample) / 3 for sample in product(values, repeat=3))
    def quantile(fraction):
        cursor = (len(means) - 1) * fraction
        left, right = math.floor(cursor), math.ceil(cursor)
        return means[left] + (means[right] - means[left]) * (cursor - left)
    return {"mean": sum(values) / 3, "lower95": quantile(.025), "upper95": quantile(.975),
            "individual_seeds": list(values), "resamples": 27,
            "method": "three independently trained policy seeds; all27 paired resamples; percentile95"}


def summarize_prerequisite(results, config, *, run_completed: bool) -> dict[str, Any]:
    """Recompute every candidate gate from complete, original matched outcomes."""
    validate_followup_config(config)
    answer = {"complete": False, "mechanical_utility_prerequisite_passed": False,
              "selected_family": None, "candidates": {}, "errors": [],
              "owner_feel_accepted": False, "learned_morphology_transfer_measured": False,
              "original_entry_result_changed": False,
              "uncertainty_limit": "three policy seeds and eight fixed new scenarios; exploratory coverage"}
    rows = [row for row in results if row.get("split") == "evaluation"]
    cases = {row["id"]: row for row in config["evaluation_families"]}
    policies = {"baseline": "baseline", "idle": "idle"}
    for family in config["candidates_in_priority_order"]:
        for seed in config["training_seeds"]:
            for phase in ("initial", "final"):
                policies[f"{family}-seed{seed}-{phase}"] = family
    expected = set(product(cases, policies))
    actual = [(row.get("scenario_family"), row.get("policy_id")) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        answer["errors"].append("incomplete_or_duplicate_paired_evaluation")
        return answer
    lookup = {(row["scenario_family"], row["policy_id"]): row for row in rows}
    for (case, policy_id), row in lookup.items():
        metrics = row.get("metrics", {})
        if (row.get("kind") != policies[policy_id] or row.get("host_seed") != cases[case]["host_seed"]
                or row.get("source_boundary_regressions") != 0
                or type(row.get("source_boundary_regressions")) is not int
                or row.get("error") is not None):
            answer["errors"].append("invalid_paired_source_or_boundary_evidence")
            return answer
        if (any(type(metrics.get(key)) is not bool for key in
                ("success", "purposeful_launch", "integrity_failure", "administratively_capped"))
                or type(metrics.get("failed_admissions")) is not int or metrics["failed_admissions"] < 0
                or any(type(metrics.get(key)) not in (int, float) or not math.isfinite(metrics[key])
                       for key in (*METRICS[1:-1], "inference_p95_ms", "return"))):
            answer["errors"].append("invalid_numeric_or_missing_outcome_evidence")
            return answer
        if (any(metrics[key] < 0 for key in (*METRICS[1:-1], "inference_p95_ms"))
                or metrics["capped_completion_seconds"] > config["goal"]["deadline_ticks"] / 60
                or metrics["mean_absolute_angle_radians"] > math.pi):
            answer["errors"].append("outcome_evidence_outside_its_physical_or_clock_range")
            return answer

    def selected(policy_id):
        return [lookup[(case, policy_id)] for case in cases]

    def average(items):
        return {key: sum(float(row["metrics"][key]) for row in items) / len(items) for key in METRICS}

    baseline_rows, idle_rows = selected("baseline"), selected("idle")
    baseline, idle = average(baseline_rows), average(idle_rows)
    answer.update(baseline=baseline, idle=idle,
                  negative_control_audit={"idle_successes": sum(row["metrics"]["success"] for row in idle_rows),
                                          "idle_positive_returns": sum(row["metrics"]["return"] > 0 for row in idle_rows)})
    gate = config["prerequisite"]
    bounds = gate["required_for_every_seed"]
    references_valid = all(not row["metrics"]["integrity_failure"]
                           and row["metrics"]["failed_admissions"] == 0
                           and not row["metrics"]["administratively_capped"]
                           for row in (*baseline_rows, *idle_rows))
    for family in config["candidates_in_priority_order"]:
        per_seed, gains, seed_gates = [], [], []
        for seed in config["training_seeds"]:
            final = selected(f"{family}-seed{seed}-final")
            initial = selected(f"{family}-seed{seed}-initial")
            means, initial_means = average(final), average(initial)
            gains.append(means["success"] - initial_means["success"])
            per_seed.append({"seed": seed, **means, "initial": initial_means,
                             "paired_gain_over_initial": gains[-1]})
            ratio = means["peak_toward_speed_cells_per_second"] / max(1e-9, baseline["peak_toward_speed_cells_per_second"])
            checks = {
                "integrity_and_admissions": references_valid and all(not row["metrics"]["integrity_failure"]
                    and row["metrics"]["failed_admissions"] == 0 for row in (*initial, *final)),
                "uncensored": all(not row["metrics"]["administratively_capped"] for row in (*initial, *final)),
                "source_and_lifetime_fences": all(row["source_boundary_regressions"] == 0 for row in (*initial, *final)),
                "ordinary_trim": means["ordinary_trim_reversals_per_second"] <= baseline["ordinary_trim_reversals_per_second"]
                    + bounds["ordinary_trim_reversals_per_second_increase_over_authored_maximum"],
                "orientation": means["mean_absolute_angle_radians"] <= baseline["mean_absolute_angle_radians"]
                    + bounds["mean_absolute_angle_increase_over_authored_maximum_radians"],
                "vigor": bounds["mean_peak_toward_speed_ratio_to_authored_minimum"] <= ratio
                    <= bounds["mean_peak_toward_speed_ratio_to_authored_maximum"],
                "launch_character": means["purposeful_launch"] >= baseline["purposeful_launch"]
                    - bounds["purposeful_launch_fraction_loss_to_authored_maximum"],
                "latency": max(row["metrics"]["inference_p95_ms"] for row in final)
                    <= bounds["every_scenario_inference_p95_maximum_ms"],
            }
            seed_gates.append({"seed": seed, **checks, "passed": all(checks.values())})
        mean_success = sum(row["success"] for row in per_seed) / 3
        minimum_success = min(row["success"] for row in per_seed)
        interval = _interval(gains)
        checks = {"minimum_mean_success": mean_success >= gate["minimum_mean_success"],
                  "minimum_every_seed_success": minimum_success >= gate["minimum_every_seed_success"],
                  "minimum_gain_over_initial": interval["mean"] >= gate["minimum_mean_success_gain_over_initial"],
                  "positive_success_gain_lower95": interval["lower95"] > 0,
                  "all_original_character_runtime_integrity_gates": all(row["passed"] for row in seed_gates),
                  "completed_fixed_run": run_completed is True}
        passed = all(checks.values())
        answer["candidates"][family] = {"passed": passed, "mean_success": mean_success,
            "minimum_seed_success": minimum_success, "gain_over_initial": interval,
            "per_seed": per_seed, "character_runtime_integrity_gates": seed_gates,
            "checks": checks, "failed": [name for name, value in checks.items() if not value]}
        if passed and answer["selected_family"] is None:
            answer["selected_family"] = family
    answer["complete"] = run_completed is True
    answer["mechanical_utility_prerequisite_passed"] = answer["selected_family"] is not None
    return answer
