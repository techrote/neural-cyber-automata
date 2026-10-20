"""Pure matched morphology summaries, with no experiment or artifact admission.

Coverage is exact, failures remain visible, and each complete three-seed paired
interval enumerates all27 resamples. No partial phase can promote a candidate.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
import math

from nca.contracts import ContractError, sha256_json
from nca.controllers.morphology import SemanticAction
from nca.controllers.morphology.admission import COMPARISON_SHA256
from .morphology_task import TASK_SPEC, REFERENCE_SPEC


COMPARISON_METRICS_SPEC = {
    "schema": "nca-morphology-comparison-metrics/v1", "comparison_sha256": COMPARISON_SHA256,
    "training_seeds": [6113, 6229, 6337], "candidates": ["padded_recurrent", "shared_graph"],
    "zero_shot_profiles": ["monowheel", "ripper"],
    "zero_shot_families": ["floor-left-30", "floor-right-30", "drop16-left-54", "drop16-right-54"],
    "adaptation_evaluation_families": ["floor-left-60", "floor-right-60"],
    "tuning_assignments": [{"profile": "jev", "family": "floor-left-42"},
                           {"profile": "astra", "family": "floor-right-42"}],
    "phase_row_counts": {"tuning": 12, "zero_shot_initial": 48, "zero_shot_final": 48,
                         "authored_and_idle": 16, "adapted_evaluation": 24},
    "total_evaluation_rows": 148,
    "primary": {"phase": "zero_shot_final", "minimum_mean_paired_success_gain": .125,
                "bootstrap_lower95_strictly_greater_than": 0., "bootstrap_resamples": 27,
                "bootstrap": "all27 size-three resamples of the three paired per-seed mean gains; linear percentile .025/.975",
                "weighting": "equal fixed scenarios within each seed, then equal independently trained seeds"},
    "every_seed_and_profile_bounds": {
        "ordinary_trim_reversals_per_second_increase_over_reference_maximum": .5,
        "mean_absolute_angle_increase_over_reference_maximum_radians": .1,
        "mean_peak_useful_speed_ratio_to_reference_minimum": .7,
        "mean_peak_useful_speed_ratio_to_reference_maximum": 1.5,
        "every_scenario_inference_p95_maximum_ms": 5.,
        "integrity_failures": 0, "failed_admissions": 0, "censored_evaluations": 0,
    },
    "character_scope": "each trained shared-final candidate seed and heldout profile against its four matching reference cases",
    "runtime_integrity_scope": "every declared evaluation row, including initial, reference, idle, tuning and adapted",
    "adaptation": "separate raw outcomes at fresh60-cell geometry; no allocated matched reference or pre-adaptation score there",
    "completion": "all148 exact evaluation rows plus caller-confirmed completed bounded programme; qualification and collection audited separately by caller",
    "missing_duplicate_invalid_rows": "incomplete, explicitly reported and final gate closed; no imputation or convenient exclusion",
    "no_go": "retain the simpler padded recurrent family when a complete comparison supplies no reliable graph benefit",
    "owner_feel_accepted": False,
}

_SEEDS = tuple(COMPARISON_METRICS_SPEC["training_seeds"])
_FAMILIES = tuple(COMPARISON_METRICS_SPEC["candidates"])
_PROFILES = tuple(COMPARISON_METRICS_SPEC["zero_shot_profiles"])
_PHASES = tuple(COMPARISON_METRICS_SPEC["phase_row_counts"])
_COLLECTION_PHASES = ("qualification", "teacher", "shared_training", "adaptation")
_MEANS = ("success", "capped_completion_seconds", "ordinary_trim_reversals_per_second",
          "mean_absolute_angle_radians", "peak_useful_speed_cells_per_second", "return")


def _number(value, minimum=0., maximum=float("inf")):
    return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _require(condition, detail):
    if not condition:
        raise ContractError("morphology_metrics_declaration", detail)


def _cases(declaration):
    """Admit the row schedule, subordinate to the caller's full source admission."""
    _require(isinstance(declaration, Mapping)
             and declaration.get("schema") == "nca-morphology-continuation-declaration/v1"
             and declaration.get("comparison_sha256") == COMPARISON_SHA256,
             "the frozen comparison and new continuation must be named")
    for field, expected in (("task", TASK_SPEC), ("reference", REFERENCE_SPEC),
                            ("comparison_metrics", COMPARISON_METRICS_SPEC)):
        _require(sha256_json(declaration.get(field)) == sha256_json(expected), f"{field} differs from the frozen predicates")
    assignments = {
        "zero_shot": [(profile, family) for profile, family in product(_PROFILES, COMPARISON_METRICS_SPEC["zero_shot_families"])],
        "adapted_evaluation": [(profile, family) for profile, family in product(_PROFILES, COMPARISON_METRICS_SPEC["adaptation_evaluation_families"])],
        "tuning": [(row["profile"], row["family"]) for row in COMPARISON_METRICS_SPEC["tuning_assignments"]],
    }
    seed_base = {"zero_shot": 81100001, "adapted_evaluation": 81300001, "tuning": 81020001}
    result = {}
    for phase, expected in assignments.items():
        rows = declaration.get(phase)
        _require(isinstance(rows, (list, tuple)) and len(rows) == len(expected), f"{phase} coverage differs")
        result[phase] = {}
        for index, (profile, family) in enumerate(expected):
            row = rows[index]
            _require(isinstance(row, Mapping) and row.get("id") == f"{profile}-{family}"
                     and row.get("profile") == profile and row.get("family") == family
                     and type(row.get("host_seed")) is int and row["host_seed"] == seed_base[phase] + index,
                     f"{phase} case identity or paired host seed differs")
            _require(all(isinstance(row.get(key), (list, tuple)) and len(row[key]) == 2
                         and all(_number(value, -float("inf")) for value in row[key]) for key in ("spawn", "target")),
                     f"{phase} needs declared finite spawn and target geometry")
            result[phase][row["id"]] = row
    return result


def _schedule(cases):
    expected = {}
    for phase in _PHASES:
        context = cases[phase if phase in ("tuning", "adapted_evaluation") else "zero_shot"]
        policies = [(kind, None) for kind in ("reference", "idle")] if phase == "authored_and_idle" else product(_FAMILIES, _SEEDS)
        for kind, seed in policies:
            for case in context.values():
                suffix = "initial" if phase == "zero_shot_initial" else "adapted-" + case["profile"] if phase == "adapted_evaluation" else "shared_final"
                policy_id = "waypoint-reference-v1" if kind == "reference" else "waypoint-idle-v1" if kind == "idle" else f"{kind}-seed{seed}-{suffix}"
                expected[phase, case["id"], kind, seed] = (case, policy_id)
    return expected


def _row_errors(row, expected):
    case, policy_id = expected
    errors = []
    def check(condition, detail):
        if not condition:
            errors.append(detail)
    check(row.get("policy_id") == policy_id and row.get("profile") == case["profile"]
          and type(row.get("host_seed")) is int and row["host_seed"] == case["host_seed"], "case, policy or host seed differs")
    check(row.get("task_spec_sha256") == sha256_json(TASK_SPEC), "task specification differs")
    check(row.get("record_schema") == "nca-morphology-episode/v1", "not a complete original morphology episode")
    for key in ("policy_sha256", "profile_sha256", "record_sha256", "artifact_sha256", "boundary_audit_sha256"):
        check(_digest(row.get(key)), f"missing or invalid {key}")
    if row["kind"] == "reference":
        check(row.get("policy_sha256") == sha256_json(REFERENCE_SPEC), "reference definition differs")
    elif row["kind"] == "idle":
        check(row.get("policy_sha256") == sha256_json({"kind": "morphology-idle/v1", "semantic_request": SemanticAction().to_dict()}),
              "idle definition differs")
    check(type(row.get("actual_owner_closed")) is bool and "error" in row
          and (row["error"] is None or isinstance(row["error"], Mapping)), "owner closure or error receipt unavailable")
    check(type(row.get("source_boundary_regressions")) is int and row["source_boundary_regressions"] >= 0,
          "source boundary audit unavailable")
    decisions, ticks = row.get("decisions"), row.get("completed_ticks")
    counters = type(decisions) is int and 1 <= decisions <= 40 and type(ticks) is int and ticks == decisions * 6
    check(counters, "actual completed request/tick counters differ")
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        return errors + ["task metrics unavailable"]
    check(all(type(metrics.get(key)) is bool for key in ("success", "integrity_failure", "administratively_capped", "owner_feel_accepted")),
          "outcome flags must be explicit booleans")
    check(metrics.get("owner_feel_accepted") is False, "owner feel is separately unjudged")
    check(type(metrics.get("failed_admissions")) is int and metrics["failed_admissions"] >= 0, "admission count unavailable")
    check(counters and type(metrics.get("elapsed_ticks")) is int and metrics["elapsed_ticks"] == ticks, "task/native tick mismatch")
    for key in (*_MEANS[1:-1], "inference_p50_ms", "inference_p95_ms"):
        check(_number(metrics.get(key)), f"invalid or unmeasured {key}")
    check(_number(metrics.get("return"), -float("inf")), "invalid or unmeasured return")
    check(_number(metrics.get("capped_completion_seconds"), 0., 4.)
          and _number(metrics.get("mean_absolute_angle_radians"), 0., math.pi), "physical or intrinsic clock range exceeded")
    if _number(metrics.get("inference_p50_ms")) and _number(metrics.get("inference_p95_ms")):
        check(metrics["inference_p50_ms"] <= metrics["inference_p95_ms"], "inference quantiles reversed")
    for key in ("response_seconds", "settled_after_landing_seconds"):
        check(key in metrics and (metrics[key] is None or _number(metrics[key], 0., 4.)), f"invalid nullable {key}")
    clean = (row.get("error") is None and metrics.get("integrity_failure") is False
             and metrics.get("administratively_capped") is False)
    if clean and metrics.get("success") is True:
        seconds = metrics.get("success_seconds")
        valid_time = _number(seconds, .2, 4.)
        success_tick = round(seconds * 60) if valid_time else None
        check(valid_time and counters and abs(seconds * 60 - success_tick) <= 1e-9 and 0 <= ticks - success_tick <= 5
              and metrics.get("capped_completion_seconds") == seconds and metrics.get("reason") == "arrival",
              "quiet arrival or remainder-of-repeat accounting differs")
    elif clean and metrics.get("success") is False:
        check(ticks == 240 and metrics.get("success_seconds") is None and metrics.get("capped_completion_seconds") == 4.
              and metrics.get("reason") == "task_deadline", "a completed miss must reach the intrinsic deadline")
    return errors


def _runtime_valid(row):
    metrics = row["metrics"]
    return (row["error"] is None and row["actual_owner_closed"] is True and row["source_boundary_regressions"] == 0
            and not metrics["integrity_failure"] and not metrics["administratively_capped"]
            and metrics["failed_admissions"] == 0 and metrics["inference_p95_ms"] <= 5.)


def _average(rows):
    result = {key: sum(float(row["metrics"][key]) for row in rows) / len(rows) for key in _MEANS}
    result.update(scenarios=len(rows), success_count=sum(row["metrics"]["success"] for row in rows),
                  maximum_scenario_inference_p95_ms=max(row["metrics"]["inference_p95_ms"] for row in rows),
                  runtime_integrity_passed=all(_runtime_valid(row) for row in rows))
    for key in ("response_seconds", "settled_after_landing_seconds"):
        measured = [row["metrics"][key] for row in rows if row["metrics"][key] is not None]
        result["measured_" + key + "_count"] = len(measured)
        result["mean_measured_" + key] = sum(measured) / len(measured) if measured else None
    return result


def _paired_interval(values):
    means = sorted(sum(sample) / 3 for sample in product(values, repeat=3))
    def quantile(fraction):
        position = (len(means) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return means[low] + (means[high] - means[low]) * (position - low)
    return {"mean": sum(values) / 3, "lower95": quantile(.025), "upper95": quantile(.975),
            "individual_seeds": list(values), "resamples": len(means),
            "method": COMPARISON_METRICS_SPEC["primary"]["bootstrap"]}


def summarize_comparison(rows, declaration, *, run_completed: bool):
    """Summarize admitted row identities; artifact/budget evidence remains external."""
    _require(type(run_completed) is bool, "completion must be an explicit caller-verified boolean")
    cases = _cases(declaration)
    expected = _schedule(cases)
    answer = {"schema": "nca-morphology-comparison-summary/v1", "complete": False, "passed": False,
              "graph_justified": False, "recommendation": "inconclusive", "errors": [],
              "comparison_metrics_sha256": sha256_json(COMPARISON_METRICS_SPEC), "owner_feel_accepted": False,
              "collection_rows_observed": {phase: 0 for phase in _COLLECTION_PHASES},
              "evidence_limit": "caller independently validates raw artifact/source/checkpoint lineage, qualification, collection and complete aggregate budget; this function starts no work",
              "uncertainty_limit": "three fixed training seeds and eight heldout scenarios; no arbitrary morphology or owner-feel claim"}
    if not isinstance(rows, (list, tuple)):
        answer["errors"].append("evaluation rows must be a list")
        rows = []
    lookup, seen, duplicates = {}, set(), set()
    policy_hashes, profile_hashes, record_hashes = {}, {}, {}
    observed_evaluation = 0
    for index, row in enumerate(rows):
        label = f"row[{index}]"
        if not isinstance(row, Mapping) or type(row.get("phase")) is not str:
            answer["errors"].append(label + ": phase unavailable")
            continue
        phase = row["phase"]
        if phase in _COLLECTION_PHASES:
            answer["collection_rows_observed"][phase] += 1
            continue
        observed_evaluation += 1
        kind, seed, case = row.get("kind"), row.get("training_seed"), row.get("scenario_family")
        if (phase not in _PHASES or type(kind) is not str or type(case) is not str or "training_seed" not in row
                or not (type(seed) is int or seed is None)):
            answer["errors"].append(label + ": unknown evaluation identity")
            continue
        key = (phase, case, kind, seed)
        if key not in expected:
            answer["errors"].append(label + ": unplanned evaluation identity")
            continue
        if key in seen:
            duplicates.add(key)
            lookup.pop(key, None)
            answer["errors"].append(label + ": duplicate evaluation identity")
            continue
        seen.add(key)
        failures = _row_errors(row, expected[key])
        if not failures:
            for mapping, name, value, reason in ((policy_hashes, row["policy_id"], row["policy_sha256"], "one policy identity changed weights"),
                                                (profile_hashes, row["profile"], row["profile_sha256"], "one profile identity changed definition")):
                if name in mapping and mapping[name] != value:
                    failures.append(reason)
                mapping[name] = value
            if row["record_sha256"] in record_hashes:
                failures.append("one physical episode was reused for another planned row")
            record_hashes[row["record_sha256"]] = key
        if failures:
            answer["errors"].extend(label + ": " + failure for failure in failures)
        else:
            lookup[key] = row
    missing = sorted(set(expected) - set(lookup), key=str)
    answer["coverage"] = {"expected": len(expected), "observed": observed_evaluation, "valid_unique": len(lookup),
                          "missing_or_invalid": [list(key) for key in missing], "duplicates": [list(key) for key in sorted(duplicates, key=str)]}
    if missing:
        answer["errors"].append("missing or invalid required evaluation rows")
    phase_coverage = {phase: {"expected": count, "valid_unique": sum(key[0] == phase for key in lookup),
                              "complete": sum(key[0] == phase for key in lookup) == count}
                      for phase, count in COMPARISON_METRICS_SPEC["phase_row_counts"].items()}
    answer["phase_coverage"] = phase_coverage

    def group(phase, kind, seed, profile=None):
        keys = [key for key, (case, _) in expected.items() if key[0] == phase and key[2] == kind and key[3] == seed
                and (profile is None or case["profile"] == profile)]
        present = [lookup[key] for key in keys if key in lookup]
        value = {"complete": len(present) == len(keys), "expected_rows": len(keys), "observed_rows": len(present)}
        if keys and value["complete"]:
            value.update(_average(present))
        return value

    zero_complete = all(phase_coverage[phase]["complete"] for phase in ("zero_shot_initial", "zero_shot_final", "authored_and_idle"))
    references = {kind: {profile: group("authored_and_idle", kind, None, profile) for profile in _PROFILES}
                  for kind in ("reference", "idle")}
    per_seed, character = [], []
    bounds = COMPARISON_METRICS_SPEC["every_seed_and_profile_bounds"]
    for family, seed in product(_FAMILIES, _SEEDS):
        row = {"family": family, "training_seed": seed,
               "initial": group("zero_shot_initial", family, seed), "final": group("zero_shot_final", family, seed),
               "profiles": {profile: {"initial": group("zero_shot_initial", family, seed, profile),
                                      "final": group("zero_shot_final", family, seed, profile)} for profile in _PROFILES}}
        per_seed.append(row)
        for profile in _PROFILES:
            candidate, reference = row["profiles"][profile]["final"], references["reference"][profile]
            checks = {"matched_rows_complete": candidate["complete"] and reference["complete"]}
            ratio = None
            if checks["matched_rows_complete"]:
                denominator = reference["peak_useful_speed_cells_per_second"]
                ratio = candidate["peak_useful_speed_cells_per_second"] / denominator if denominator > 0 else None
                checks.update(
                    ordinary_trim=candidate["ordinary_trim_reversals_per_second"] <= reference["ordinary_trim_reversals_per_second"]
                        + bounds["ordinary_trim_reversals_per_second_increase_over_reference_maximum"],
                    orientation=candidate["mean_absolute_angle_radians"] <= reference["mean_absolute_angle_radians"]
                        + bounds["mean_absolute_angle_increase_over_reference_maximum_radians"],
                    vigor=ratio is not None and bounds["mean_peak_useful_speed_ratio_to_reference_minimum"] <= ratio
                        <= bounds["mean_peak_useful_speed_ratio_to_reference_maximum"],
                    runtime_integrity=candidate["runtime_integrity_passed"] and reference["runtime_integrity_passed"])
            character.append({"family": family, "training_seed": seed, "profile": profile,
                              "peak_useful_speed_ratio_to_reference": ratio, "checks": checks,
                              "passed": all(checks.values())})
    gains = None
    if phase_coverage["zero_shot_final"]["complete"]:
        gains = _paired_interval([group("zero_shot_final", "shared_graph", seed)["success"]
                                  - group("zero_shot_final", "padded_recurrent", seed)["success"] for seed in _SEEDS])
    primary_checks = {
        "complete_zero_shot_and_controls": zero_complete,
        "minimum_mean_paired_success_gain": gains is not None and gains["mean"] >= .125,
        "positive_paired_gain_lower95": gains is not None and gains["lower95"] > 0.,
        "every_candidate_seed_and_profile_character": all(row["passed"] for row in character),
        "all_zero_shot_runtime_integrity": zero_complete and all(_runtime_valid(row) for key, row in lookup.items()
            if key[0] in ("zero_shot_initial", "zero_shot_final", "authored_and_idle")),
    }
    answer["zero_shot"] = {"complete": zero_complete, "per_seed": per_seed, "references": references,
                           "paired_final_success_gain": gains, "character_gates": character,
                           "checks": primary_checks, "passed": all(primary_checks.values())}
    answer["adaptation"] = {"complete": phase_coverage["adapted_evaluation"]["complete"],
        "per_seed_profile": [{"family": family, "training_seed": seed, "profile": profile,
                              **group("adapted_evaluation", family, seed, profile)} for family, seed, profile in product(_FAMILIES, _SEEDS, _PROFILES)],
        "matched_reference_comparison": None, "paired_gain_over_shared": None,
        "limit": COMPARISON_METRICS_SPEC["adaptation"]}
    answer["tuning"] = {"complete": phase_coverage["tuning"]["complete"],
        "per_seed": [{"family": family, "training_seed": seed, **group("tuning", family, seed)} for family, seed in product(_FAMILIES, _SEEDS)],
        "checkpoint_selection": "fixed shared final; these rows do not select a checkpoint or seed"}
    coverage_complete = not missing and not answer["errors"] and observed_evaluation == len(expected)
    answer["complete"] = run_completed and coverage_complete
    answer["checks"] = {"all_evaluation_rows_complete": coverage_complete, "programme_completed": run_completed,
                         "primary_zero_shot_gate": answer["zero_shot"]["passed"],
                         "all_evaluation_runtime_integrity": coverage_complete and all(_runtime_valid(row) for row in lookup.values())}
    answer["passed"] = answer["graph_justified"] = all(answer["checks"].values())
    if answer["complete"]:
        answer["recommendation"] = "graph_benefit_supported_within_declared_screen" if answer["passed"] else "retain_padded_recurrent"
    return answer
