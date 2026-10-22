"""Unchanged whole-family Jolt qualification from complete paired receipts.

This module has no optional learner or host dependency. Policy seeds, rather
than timesteps or scenario repeats, are the uncertainty unit.
"""

from __future__ import annotations

import itertools
import math
from typing import Mapping

from nca.contracts import ContractError


SCHEMA = "nca-jolt-learning-qualification/v1"
SCHEMA_V3 = "nca-jolt-learning-qualification/v3"
POLICY_SEEDS = (190103, 190207, 190301)
METRICS = ("success", "capped_completion_seconds", "ordinary_trim_reversals_per_second",
           "mean_absolute_angle_radians", "peak_toward_speed_cells_per_second", "purposeful_launch")
BOOLEAN_METRICS = ("success", "purposeful_launch", "integrity_failure", "administratively_capped")


def _validate_metrics(row):
    """Reject malformed evidence before it can improve any qualification mean."""
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ContractError("jolt_metrics", "each paired result requires a metric record")
    for key in BOOLEAN_METRICS:
        if type(metrics.get(key)) is not bool:
            raise ContractError("jolt_metrics", f"{key} must be an explicit boolean outcome")
    for key in METRICS:
        if key in BOOLEAN_METRICS:
            continue
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ContractError("jolt_metrics", f"{key} must be finite and nonnegative")
    if metrics["capped_completion_seconds"] > 4:
        raise ContractError("jolt_metrics", "completion is capped at the unchanged four-second deadline")
    for value, field in ((metrics.get("failed_admissions"), "failed_admissions"),
                         (row.get("supervisor_interventions"), "supervisor_interventions")):
        if type(value) is not int or value < 0:
            raise ContractError("jolt_metrics", f"{field} must be a nonnegative integer count")
    if type(row.get("owner_closure_confirmed")) is not bool:
        raise ContractError("jolt_metrics", "owner closure requires an explicit boolean receipt")
    latency = metrics.get("inference_p95_ms")
    if latency is not None and (type(latency) not in (int, float)
                                or not math.isfinite(latency) or latency < 0):
        raise ContractError("jolt_metrics", "reported inference latency must be finite and nonnegative")


def seed_interval(values):
    if len(values) != 3 or any(type(x) not in (float, int) or not math.isfinite(x) for x in values):
        raise ContractError("jolt_uncertainty", "exactly three finite independent policy-seed values are required")
    draws = sorted(sum(draw) / 3 for draw in itertools.product(values, repeat=3))

    def quantile(q):
        index = (len(draws) - 1) * q
        low, high = math.floor(index), math.ceil(index)
        return draws[low] + (draws[high] - draws[low]) * (index - low)

    return {"mean": sum(values) / 3, "lower95": quantile(.025), "upper95": quantile(.975),
            "per_seed": list(values), "resamples": 27,
            "method": "all 27 three-policy-seed percentile bootstrap resamples"}


def _mean(rows):
    if not rows:
        raise ContractError("jolt_pairs", "empty outcomes cannot establish a mean")
    result = {}
    for key in METRICS:
        values = [row["metrics"][key] for row in rows]
        if any(type(x) not in (int, float, bool) or not math.isfinite(x) for x in values):
            raise ContractError("jolt_metrics", "complete finite qualification metrics are required")
        result[key] = sum(float(x) for x in values) / len(values)
    return result


def controller_ids():
    return ["authored", "idle"] + [f"{family}-seed{seed}-{phase}"
        for family in ("bc", "residual") for seed in POLICY_SEEDS for phase in ("initial", "final")]


def qualify(rows, cases, *, evidence_complete=False, execution_version=1):
    """No outcome censoring, checkpoint selection, or authored-baseline waiver."""
    if type(evidence_complete) is not bool:
        raise ContractError("jolt_evidence", "independent evidence acceptance must be an explicit boolean")
    if type(execution_version) is not int or execution_version not in (1, 2, 3):
        raise ContractError("jolt_evidence", "unknown qualification execution version")
    selected = [row for row in rows if row["split"] == "sealed"]
    case_ids = [case["id"] if isinstance(case, Mapping) else case for case in cases]
    if len(case_ids) != 16 or len(set(case_ids)) != 16:
        raise ContractError("jolt_pairs", "the unchanged qualification uses sixteen distinct sealed cases")
    expected = set(itertools.product(case_ids, controller_ids()))
    keys = [(row["case_id"], row["controller_id"]) for row in selected]
    complete = len(keys) == len(set(keys)) and set(keys) == expected
    result = {"schema": SCHEMA_V3 if execution_version == 3 else SCHEMA,
              "complete_pairs": complete, "evidence_complete": evidence_complete,
              "expected_rows": len(expected), "observed_rows": len(selected), "families": {},
              "selected_family": None, "qualified": False, "owner_feel_accepted": False,
              "morphology_transfer_admitted": False,
              "conclusion": "incomplete_paired_evidence"}
    if not complete:
        result["missing_pairs"] = [list(pair) for pair in sorted(expected - set(keys))]
        result["unexpected_pairs"] = [list(pair) for pair in sorted(set(keys) - expected)]
        result["duplicate_pairs"] = [list(pair) for pair in sorted({pair for pair in keys if keys.count(pair) > 1})]
        return result
    for row in selected:
        _validate_metrics(row)
        if execution_version == 3:
            _validate_outcome_v3(row)
    lookup = dict(zip(keys, selected))
    authored = [lookup[(case, "authored")] for case in case_ids]
    idle = [lookup[(case, "idle")] for case in case_ids]
    reference = _mean(authored)
    result["authored"] = reference
    result["idle"] = _mean(idle)
    integrity = all(row.get("error") is None and row.get("owner_closure_confirmed") is True
                    and row["metrics"].get("integrity_failure") is False
                    and row["metrics"].get("failed_admissions") == 0
                    and row["metrics"].get("administratively_capped") is False for row in selected)
    result["global_integrity"] = integrity
    if execution_version == 3:
        integrity = integrity and all(row["outcome"]["complete"] for row in selected)
        result["global_integrity"] = integrity
        result["safe_retirements"] = sum(row["outcome"]["kind"] == "safe_retirement" for row in selected)
        result["failure_denominator"] = "all sixteen declared cases for every controller and policy seed"
        result["terminal_penalty_added"] = False
    for family in ("bc", "residual"):
        seeds, initial_gains, authored_gains, time_gains = [], [], [], []
        for seed in POLICY_SEEDS:
            final = [lookup[(case, f"{family}-seed{seed}-final")] for case in case_ids]
            initial = [lookup[(case, f"{family}-seed{seed}-initial")] for case in case_ids]
            means, initial_means = _mean(final), _mean(initial)
            gain = means["success"] - initial_means["success"]
            initial_gains.append(gain)
            authored_gains.append(means["success"] - reference["success"])
            time_gains.append(reference["capped_completion_seconds"] - means["capped_completion_seconds"])
            speed_reference = reference["peak_toward_speed_cells_per_second"]
            speed_ratio = means["peak_toward_speed_cells_per_second"] / speed_reference if speed_reference > 0 else None
            latency = [row["metrics"].get("inference_p95_ms") for row in final]
            gates = {
                "minimum_seed_arrival": means["success"] >= .5,
                "ordinary_trim": means["ordinary_trim_reversals_per_second"] <= reference["ordinary_trim_reversals_per_second"] + .5,
                "orientation": means["mean_absolute_angle_radians"] <= reference["mean_absolute_angle_radians"] + .1,
                "vigor": speed_ratio is not None and .7 <= speed_ratio <= 1.5,
                "purposeful_airborne": means["purposeful_launch"] >= reference["purposeful_launch"] - .125,
                "every_case_latency": all(type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 5 for x in latency),
            }
            autonomous = [row for row in final if row.get("supervisor_interventions") == 0]
            assisted = [row for row in final if row.get("supervisor_interventions", 0) > 0]
            seeds.append({"seed": seed, "final": means, "initial": initial_means, "initial_gain": gain,
                          "speed_ratio": speed_ratio, "case_inference_p95_ms": latency, "gates": gates,
                          "autonomous_cases": len(autonomous), "autonomous_arrivals": sum(row["metrics"]["success"] for row in autonomous),
                          "emergency_assisted_cases": len(assisted), "emergency_assisted_arrivals": sum(row["metrics"]["success"] for row in assisted)})
        initial_ci, authored_ci, time_ci = map(seed_interval, (initial_gains, authored_gains, time_gains))
        mean_arrival = sum(row["final"]["success"] for row in seeds) / 3
        authored_success = authored_ci["mean"] >= .125 and authored_ci["lower95"] > 0
        completion_fraction = time_ci["mean"] / reference["capped_completion_seconds"] if reference["capped_completion_seconds"] > 0 else 0
        authored_completion = (reference["success"] >= .875 and time_ci["mean"] >= .15
                               and completion_fraction >= .1 and min(authored_gains) >= 0 and time_ci["lower95"] > 0)
        gates = {"complete_evidence": evidence_complete, "integrity": integrity,
                 "mean_arrival": mean_arrival >= .75,
                 "every_seed": all(all(seed["gates"].values()) for seed in seeds),
                 "own_initial_gain": initial_ci["mean"] >= .125 and initial_ci["lower95"] > 0,
                 "original_authored_improvement": authored_success or authored_completion}
        passed = all(gates.values())
        result["families"][family] = {"qualified": passed, "gates": gates, "mean_arrival": mean_arrival,
            "per_seed": seeds, "initial_gain_interval": initial_ci, "authored_gain_interval": authored_ci,
            "authored_completion_interval": time_ci, "authored_completion_gain_fraction": completion_fraction,
            "authored_assessment": {"success_gain_path": authored_success, "high_baseline_completion_path": authored_completion}}
        if passed and result["selected_family"] is None:
            result["selected_family"] = family
    result["qualified"] = result["selected_family"] is not None
    result["conclusion"] = "learned_prerequisite_qualified" if result["qualified"] else "retained_negative_result"
    return result


def _validate_outcome_v3(row):
    """A complete expected failure stays in every original paired denominator."""
    from nca.evaluation.jolt_retirement import OUTCOME_SCHEMA, RETIREMENT_SCHEMA, SAFE_REASON
    outcome = row.get("outcome")
    if (not isinstance(outcome, Mapping)
            or set(outcome) != {"schema", "kind", "complete", "success", "reason"}
            or outcome["schema"] != OUTCOME_SCHEMA
            or outcome["kind"] not in ("safe_retirement", "task_terminal", "administrative_cutoff", "integrity_error")
            or type(outcome["complete"]) is not bool or type(outcome["success"]) is not bool
            or not isinstance(outcome["reason"], str)):
        raise ContractError("jolt_outcome", "explicit versioned episode outcome required")
    metrics = row["metrics"]
    if outcome["success"] is not metrics["success"]:
        raise ContractError("jolt_outcome", "episode outcome changed the observed task success")
    if (outcome["reason"] != metrics.get("reason")
            or any(field not in row for field in ("cancellation_error", "retirement_validation_error", "cleanup_phase_error"))
            or outcome["complete"] and (row.get("owner_closure_confirmed") is not True
                or row.get("error") is not None or row["cancellation_error"] is not None
                or row["retirement_validation_error"] is not None or row["cleanup_phase_error"] is not None)):
        raise ContractError("jolt_outcome", "complete outcome lacks original reason and independent native closure")
    if outcome["kind"] == "safe_retirement":
        retirement = row.get("retirement")
        if (not outcome["complete"] or outcome["success"] or outcome["reason"] != SAFE_REASON
                or row.get("error") is not None or row["cancellation_error"] is not None
                or row["retirement_validation_error"] is not None
                or row.get("owner_closure_confirmed") is not True
                or metrics["integrity_failure"] or metrics["failed_admissions"] != 0
                or metrics["administratively_capped"] or metrics["capped_completion_seconds"] != 4.
                or metrics.get("reason") != SAFE_REASON
                or not isinstance(retirement, Mapping) or retirement.get("schema") != RETIREMENT_SCHEMA
                or retirement.get("reason") != SAFE_REASON
                or retirement.get("path") not in ("ordinary_attempt", "cutoff_peek")):
            raise ContractError("jolt_outcome", "safe retirement cannot become success, integrity failure or a shorter cost")
    elif row.get("retirement") is not None:
        raise ContractError("jolt_outcome", "nonretired outcome invented a safe-retirement receipt")
    if outcome["kind"] == "integrity_error":
        if outcome["complete"] or outcome["success"] or row.get("error") is None:
            raise ContractError("jolt_outcome", "integrity errors cannot establish complete successful pairs")
    elif outcome["kind"] == "administrative_cutoff":
        if not outcome["complete"] or outcome["success"] or not metrics["administratively_capped"]:
            raise ContractError("jolt_outcome", "administrative cutoff must retain its incomplete-task meaning")
    elif outcome["kind"] == "task_terminal":
        if (not outcome["complete"] or row.get("error") is not None
                or metrics["administratively_capped"] or metrics["integrity_failure"]):
            raise ContractError("jolt_outcome", "task terminal lacks receipt-complete physical evidence")


def qualify_v3(rows, cases, *, evidence_complete=False):
    """Versioned complete-failure semantics with all original numeric gates."""
    return qualify(rows, cases, evidence_complete=evidence_complete, execution_version=3)
