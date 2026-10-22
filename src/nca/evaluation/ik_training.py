"""Independent, denominator-preserving scoring for the direct compact IK pilot.

No import or operation here opens a corpus, starts training, imports Torch, or
grants host authority. Accepted final data must first pass the corpus registry's
candidate-bound opening. Geometry uses the separate binary64 two-vector checker,
never the differentiable training FK. Timing is an explicit opt-in operation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import math
import time

from nca.contracts import ContractError, digest, identifier, sha256_json, strict_json_loads
from nca.ik.data import (
    DatasetConfig, GeometricSample, RuntimeInputs, branch_identity,
    independent_two_link_point, iter_tasks, runtime_features, validate_task,
)
from nca.ik.kinematics import chain_from_profile
from nca.ik.profiles import runner_right_sole_profile
from nca.ik.reference import analytic_two_link, solve_numeric


SEEDS = (41011, 42013, 43019)
COMPARATORS = (
    "trained_raw_reference", "matched_untrained_initialization",
    "learned_disabled_current_reference", "independent_numerical_reference",
    "independent_analytic_reference",
)
LEARNED = frozenset(COMPARATORS[:2])
LENGTH_CELLS = 6.550781059358212
RAW_P95_CELLS, RAW_WORST_CELLS = .06, .18
RAW_P95_NORMALIZED, RAW_WORST_NORMALIZED = .01, .03
REFUSALS = frozenset(("invalid_input", "unsupported_profile", "missing_required_measurement",
                     "refused", "prediction_exception", "nonfinite_output"))


@dataclass(frozen=True)
class Prediction:
    """Unprojected absolute reference plus actual work, including failed work.

    The direct candidate has no corrected/fallback output. A numerical baseline
    uses solver_iterations/evaluations; these are not learned refinement or
    fallback calls. Malformed/nonfinite raw references are deliberately accepted
    by this envelope so scoring can retain and fail them rather than lose a row.
    """

    raw_reference: tuple[float, float] | None
    status: str = "candidate"
    correction_iterations: int = 0
    full_reference_fallback_calls: int = 0
    projection_max_radians: float = 0.
    solver_iterations: int = 0
    solver_evaluations: int = 0
    projection_events: int = 0
    projection_correction_norm: float = 0.


def _fail(code, detail):
    raise ContractError("ik_evaluation_" + code, detail)


def _seal(value):
    return dict(value, sha256=sha256_json(value))


def _sealed(value, schema):
    if not isinstance(value, Mapping) or value.get("schema") != schema:
        _fail("identity", "required sealed record schema differs")
    body = dict(value)
    seal = body.pop("sha256", None)
    if seal != sha256_json(body):
        _fail("identity", "sealed record content changed")
    return value


def nearest_rank(values, fraction):
    """Accepted ceil(p*N) percentile, with no interpolation."""
    values = tuple(values)
    if not 0 < fraction <= 1:
        _fail("percentile", "percentile must be in (0,1]")
    if not values:
        return None
    if any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in values):
        _fail("percentile", "finite nonnegative values required")
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def _vector(value):
    if value is None:
        return None, False, False
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None, False, True
    if any(type(x) not in (float, int) for x in value):
        return None, False, True
    if not all(math.isfinite(x) for x in value):
        return None, True, False
    return tuple(float(x) for x in value), False, False


def _refusal(task):
    if task.invalid_specimen is not None:
        try:
            RuntimeInputs.from_dict(strict_json_loads(task.invalid_specimen))
        except ContractError:
            return "invalid_input"
        _fail("specimen", "the original invalid specimen was unexpectedly admitted")
    inputs = task.inputs
    if inputs.profile_sha256 != runner_right_sole_profile().sha256:
        return "unsupported_profile"
    if not all((*inputs.present, *inputs.measured, *inputs.active)):
        return "missing_required_measurement"
    if not all(inputs.task_mask):
        return "invalid_input"
    return None


def collect_predictions(samples, predictor):
    """Call only permitted runtime inputs; retain each exception as a failed row.

    Parsing the original malformed specimen and rejecting unsupported/missing
    input happen before invoking any comparator. No target-generating q, teacher
    label, branch, split or trajectory identity is passed to the predictor.
    """
    result = {}
    for sample in samples:
        if not isinstance(sample, GeometricSample) or sample.task.id in result:
            _fail("samples", "typed unique sample records required")
        task = sample.task
        refusal = _refusal(task)
        if refusal is not None:
            result[task.id] = Prediction(None, refusal)
            continue
        try:
            result[task.id] = predictor(task.inputs)
        except Exception:
            # Exception text may contain paths or other private details.
            result[task.id] = Prediction(None, "prediction_exception")
    return result


def model_predictor(context, model):
    """Batch-one feature construction and model inference on its explicit device."""
    torch = context.torch

    def predict(inputs):
        features = torch.tensor([runtime_features(inputs)], device=context.device, dtype=torch.float32)
        with torch.inference_mode():
            output = model(features)
        if output.reference.shape != (1, 2) or output.valid.shape != (1,):
            _fail("model_shape", "model must return exactly one reference and validity flag")
        if not bool(output.valid[0]):
            code = int(output.invalid_code[0])
            return Prediction(None, "nonfinite_output" if code & 32 else "refused")
        return Prediction(tuple(float(x) for x in output.reference[0].detach().cpu().tolist()))

    return predict


def reference_predictors(config):
    """Fresh independent controls; their work must be charged by the caller."""
    profile = runner_right_sole_profile()
    chain = chain_from_profile(profile)

    def numerical(inputs):
        q = tuple(0. if v is None else v for v in inputs.current)
        result = solve_numeric(
            chain, inputs.target_root, q, config=config.teacher,
            lower=(-.72, -.27), upper=(.72, .27), posture=inputs.posture,
            present=inputs.present, measured=inputs.measured,
            active=inputs.active, task_mask=inputs.task_mask,
        )
        return Prediction(result.configuration, result.status,
                          solver_iterations=result.iterations, solver_evaluations=result.evaluations,
                          projection_events=result.projection_events,
                          projection_correction_norm=result.projection_correction_norm)

    def analytic(inputs):
        result = analytic_two_link(chain, inputs.target_root, inputs.current,
                                   lower=(-.72, -.27), upper=(.72, .27))
        return Prediction(result.configuration, result.status)

    return {
        "learned_disabled_current_reference": lambda inputs: Prediction(inputs.current, "current_reference"),
        "independent_numerical_reference": numerical,
        "independent_analytic_reference": analytic,
    }


def _population(rows):
    errors = [row["raw_error_cells"] for row in rows if row["raw_error_cells"] is not None]
    complete = len(errors) == len(rows) and all(row["raw_usable"] for row in rows)
    observed = {
        "p50_error_cells": nearest_rank(errors, .50),
        "p95_error_cells": nearest_rank(errors, .95),
        "worst_error_cells": max(errors) if errors else None,
    }
    counts = Counter(row["status"] for row in rows)
    return {
        "scheduled": len(rows), "observed_finite_endpoints": len(errors),
        "complete_raw_outputs": bool(rows) and complete,
        # A survivor-only percentile can never be used for selection/promotion.
        **{key: value if complete and rows else None for key, value in observed.items()},
        "observed_only_diagnostic": observed,
        "successes": sum(row["raw_success"] for row in rows),
        "feasible_success_fraction": sum(row["raw_success"] for row in rows) / len(rows) if rows else None,
        "status_counts": dict(sorted(counts.items())),
        "missing_predictions": sum(row["missing_prediction"] for row in rows),
        "nonfinite_outputs": sum(row["nonfinite_output"] for row in rows),
        "malformed_outputs": sum(row["malformed_output"] for row in rows),
        "hard_bound_violations": sum(row["hard_bound_violation"] for row in rows),
        "command_bound_violations": sum(row["command_bound_violation"] for row in rows),
        "unexpected_acceptances": sum(row["unexpected_acceptance"] for row in rows),
        "solver_iterations": sum(row["solver_iterations"] for row in rows),
        "solver_evaluations": sum(row["solver_evaluations"] for row in rows),
        "numerical_correction_iterations": sum(row["correction_iterations"] for row in rows),
        "full_reference_fallback_calls": sum(row["full_reference_fallback_calls"] for row in rows),
        "full_reference_fallback_fraction": (
            sum(row["full_reference_fallback_calls"] for row in rows) / len(rows) if rows else None),
        "projection_events": sum(row["projection_events"] for row in rows),
        "projected_over_2e5_radians": sum(row["projection_max_radians"] > 2e-5 for row in rows),
        "maximum_projection_radians": max((row["projection_max_radians"] for row in rows), default=0.),
    }


def _score(task, prediction, chain, comparator, *, sample_missing):
    missing = prediction is None
    malformed = prediction is not None and not isinstance(prediction, Prediction)
    if missing or malformed:
        prediction = Prediction(None, "missing_prediction" if missing else "malformed_prediction")
    q, nonfinite, wrong_shape = _vector(prediction.raw_reference)
    malformed |= wrong_shape
    violations = []
    counts = {}
    for name in ("correction_iterations", "full_reference_fallback_calls", "solver_iterations",
                 "solver_evaluations", "projection_events"):
        value = getattr(prediction, name)
        if type(value) is not int or value < 0:
            violations.append(name)
            value = 0
        counts[name] = value
    for name in ("projection_max_radians", "projection_correction_norm"):
        value = getattr(prediction, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            violations.append(name)
            value = 0.
        counts[name] = value
    if comparator in LEARNED and any(counts[name] for name in counts):
        violations.append("direct_mode_has_solver_or_projection_work")
    status = prediction.status
    if not isinstance(status, str) or not status or len(status) > 128:
        status = "malformed_status"
        malformed = True
    nonfinite |= status == "nonfinite_output"
    expected_refusal = _refusal(task)
    unexpected = expected_refusal is not None and q is not None
    hard = q is not None and any(abs(v) > limit for v, limit in zip(q, (.75, .30)))
    command = q is not None and any(abs(v) > limit for v, limit in zip(q, (.72, .27)))
    error = None
    branch = None
    if q is not None:
        point = independent_two_link_point(chain, q)
        error = math.dist(point, task.inputs.target_root)
        branch = branch_identity(chain, q)
        if not math.isfinite(error):
            error, nonfinite = None, True
    usable = (q is not None and not (hard or command or malformed or nonfinite or violations or unexpected
                                   or sample_missing) and status not in REFUSALS
              and status not in ("time_budget_exhausted", "iteration_budget_exhausted"))
    success = bool(usable and error is not None and error <= RAW_P95_CELLS
                   and error / LENGTH_CELLS <= RAW_P95_NORMALIZED)
    return {
        "task_id": task.id, "runtime_input_sha256": task.inputs.sha256,
        "kind": task.state.kind, "stress_class": task.state.stress_class,
        "region_kind": task.region_kind, "status": status,
        "sample_missing": sample_missing, "missing_prediction": missing,
        "malformed_output": bool(malformed), "nonfinite_output": bool(nonfinite),
        "protocol_violations": violations, "raw_reference": list(q) if q is not None else None,
        "raw_error_cells": error, "raw_error_normalized": error / LENGTH_CELLS if error is not None else None,
        "raw_usable": bool(usable), "raw_success": success,
        "hard_bound_violation": bool(hard), "command_bound_violation": bool(command),
        "expected_refusal": expected_refusal, "unexpected_acceptance": bool(unexpected),
        "expected_refusal_observed": expected_refusal is not None and q is None and status == expected_refusal,
        "branch": branch, **counts,
        "corrected_error_cells": None, "fallback_error_cells": None,
        "corrected_and_fallback_reason": "frozen_direct_mode_has_neither",
    }


def _continuity(tasks, rows):
    by_id = {row["task_id"]: row for row in rows}
    trajectories = {}
    for task in tasks:
        if task.state.kind == "trajectory":
            key = (task.family_id, task.trajectory_id, task.state.augmentation)
            trajectories.setdefault(key, {})[task.state.step] = task
    deltas, pairs, absent, switches, violations = [], 0, 0, 0, 0
    for sequence in trajectories.values():
        for step, task in sequence.items():
            following = sequence.get(step + 1)
            if following is None or math.dist(task.inputs.target_root, following.inputs.target_root) > .10:
                continue
            pairs += 1
            first, second = by_id[task.id], by_id[following.id]
            if not first["raw_usable"] or not second["raw_usable"]:
                absent += 1
                continue
            # Validated rest-relative coordinates: wrapping cannot hide a jump.
            change = max(abs(a - b) for a, b in zip(first["raw_reference"], second["raw_reference"]))
            deltas.append(change)
            switches += first["branch"] != second["branch"] or first["branch"] != "negative"
            violations += change > .25
    return {
        "scheduled_eligible_pairs": pairs, "observed_pairs": len(deltas),
        "missing_or_invalid_pairs": absent, "maximum_joint_change_radians": max(deltas) if deltas else None,
        "over_0_25_radian_pairs": violations, "admitted_branch_switches": switches,
        "complete": absent == 0,
    }


def _schedule(samples, config, partition, expected_task_ids):
    if not isinstance(config, DatasetConfig) or partition not in ("tuning", "final"):
        _fail("partition", "only typed tuning/final evaluation configurations are accepted")
    tasks = tuple(iter_tasks(config, partition))
    ids = tuple(task.id for task in tasks)
    if expected_task_ids is not None and tuple(expected_task_ids) != ids:
        _fail("schedule", "supplied schedule differs from the complete declared partition")
    if partition == "final" and expected_task_ids is None:
        _fail("schedule", "final scoring requires the frozen complete ordered schedule")
    supplied = {}
    for sample in samples:
        if (not isinstance(sample, GeometricSample) or sample.task.partition != partition
                or sample.task.id in supplied):
            _fail("samples", "sample partition/type/uniqueness differs from the schedule")
        validate_task(sample.task, config)
        supplied[sample.task.id] = sample
    if set(supplied) - set(ids):
        _fail("samples", "unscheduled samples cannot expand or replace the denominator")
    return tasks, supplied, ids


def score_raw(samples, predictions, config, *, partition="tuning", expected_task_ids=None,
              final_opening=None, frozen_candidates=None, seed=None, comparator="trained_raw_reference"):
    """Score every scheduled row without numerical correction or teacher calls.

    Tuning requires no opening. Final scoring additionally requires a candidate
    freeze, the existing registry's opening, and an explicit complete schedule.
    Caller-provided predictions cannot shrink the canonical schedule. Samples are
    already authenticated by the corpus reader; this API independently validates
    task replay/partition, and never uses teacher labels as the scoring truth.
    """
    if comparator not in COMPARATORS or not isinstance(predictions, Mapping):
        _fail("prediction", "known comparator and task-keyed predictions required")
    candidate = None
    if partition == "final":
        candidate = validate_final_opening(final_opening, frozen_candidates, seed=seed)
        if candidate["config_sha256"] != config.sha256:
            _fail("opening", "final scoring configuration differs from the frozen candidates")
    tasks, supplied, ids = _schedule(samples, config, partition, expected_task_ids)
    if set(predictions) - set(ids):
        _fail("prediction", "unscheduled predictions cannot replace missing calls")
    chain = chain_from_profile(runner_right_sole_profile())
    rows = [_score(task, predictions.get(task.id), chain, comparator,
                   sample_missing=task.id not in supplied) for task in tasks]
    nominal = [row for row in rows if row["kind"] != "stress"]
    withheld = [row for row in nominal if row["region_kind"] == "withheld_region"]
    populations = {"nominal": _population(nominal), "withheld": _population(withheld)}
    for cls in sorted({row["stress_class"] for row in rows if row["kind"] == "stress"}):
        populations[cls] = _population([row for row in rows if row["stress_class"] == cls])
    complete = (
        len(supplied) == len(ids) and len(predictions) == len(ids)
        and all(not (row["sample_missing"] or row["missing_prediction"] or row["malformed_output"]
                     or row["nonfinite_output"] or row["protocol_violations"]
                     or row["unexpected_acceptance"] or row["status"] == "prediction_exception")
                for row in rows)
        and populations["nominal"]["complete_raw_outputs"]
        and all(row["expected_refusal_observed"] for row in rows if row["expected_refusal"] is not None)
    )
    return {
        "schema": "nca-ik-direct-score/v1", "partition": partition, "config_sha256": config.sha256,
        "comparator": comparator, "seed": seed, "mode": "direct_joint_position",
        "frozen_checkpoint_sha256": candidate["checkpoint_sha256"] if candidate else None,
        "scheduled_calls": len(ids), "schedule_sha256": sha256_json(list(ids)),
        "received_samples": len(supplied), "received_predictions": len(predictions),
        "complete": bool(complete), "nominal": populations["nominal"], "withheld": populations["withheld"],
        "populations": populations, "continuity": _continuity(tasks, rows),
        "selection_metric": populations["nominal"]["p95_error_cells"] if complete else None,
        "rows": rows, "cpu_latency": None,
        "cpu_latency_reason": "not_measured_by_geometric_scoring",
        "cuda_throughput": None, "cuda_memory": None,
        "device_measurement_reason": "requires_separately_budgeted_measured_device_execution",
        "qualified": False,
    }


def _accuracy(report):
    if any(row["nonfinite_output"] or row["hard_bound_violation"] or row["command_bound_violation"]
           or row["unexpected_acceptance"] or row["protocol_violations"] for row in report["rows"]):
        return "no_go"
    if not report["complete"]:
        return "inconclusive"
    groups = [report["nominal"]]
    if report["partition"] == "final":
        groups.append(report["withheld"])
    for group in groups:
        if (not group["complete_raw_outputs"] or group["p95_error_cells"] > RAW_P95_CELLS
                or group["p95_error_cells"] / LENGTH_CELLS > RAW_P95_NORMALIZED
                or group["worst_error_cells"] > RAW_WORST_CELLS
                or group["worst_error_cells"] / LENGTH_CELLS > RAW_WORST_NORMALIZED
                or group["feasible_success_fraction"] < .95):
            return "no_go"
    if (not report["continuity"]["complete"] or report["continuity"]["over_0_25_radian_pairs"]
            or report["continuity"]["admitted_branch_switches"]):
        return "no_go"
    return "pass"


def evaluate_comparison(samples, config, *, trained, untrained, partition="tuning",
                        expected_task_ids=None, final_opening=None, frozen_candidates=None, seed=None):
    """Run five matched comparators, retaining all rows and actual solver work.

    This is deliberately separate from lightweight repeated tuning score_raw.
    Calling this function spends fresh numerical/analytic control calls. The
    run harness owns durable reservation and source/environment admission.
    """
    samples = tuple(samples)
    predictors = {"trained_raw_reference": trained, "matched_untrained_initialization": untrained,
                  **reference_predictors(config)}
    reports = {}
    for name in COMPARATORS:
        predictions = collect_predictions(samples, predictors[name])
        reports[name] = score_raw(
            samples, predictions, config, partition=partition, expected_task_ids=expected_task_ids,
            final_opening=final_opening, frozen_candidates=frozen_candidates, seed=seed, comparator=name,
        )
    trained_report, untrained_report = reports[COMPARATORS[0]], reports[COMPARATORS[1]]
    accuracy = _accuracy(trained_report)
    first, control = trained_report["selection_metric"], untrained_report["selection_metric"]
    contribution = ("inconclusive" if first is None or control is None else
                    "pass" if first <= .5 * control else "no_go")
    return {
        "schema": "nca-ik-direct-comparison/v1", "seed": seed, "partition": partition,
        "config_sha256": config.sha256, "schedule_sha256": trained_report["schedule_sha256"],
        "comparators": reports, "accuracy": accuracy, "learned_contribution": contribution,
        "practical_utility": "inconclusive",
        "practical_utility_reason": "matched_complete_decision_cpu_latency_is_unmeasured",
        "qualified": False,
    }


def select_tuning_checkpoint(records):
    """Complete tuning reports only; lowest raw p95, ties earliest step.

    Any scheduled failed/incomplete report prevents selection here. The caller
    additionally binds the exact expected cadence and durable attempt ledger.
    """
    records = tuple(records)
    if not records:
        _fail("selection", "at least one scheduled tuning record is required")
    steps, identities, checked = set(), set(), []
    for item in records:
        if not isinstance(item, Mapping) or set(item) != {"step", "checkpoint_sha256", "tuning"}:
            _fail("selection", "exact checkpoint, step and tuning fields required")
        step, checkpoint, report = item["step"], item["checkpoint_sha256"], item["tuning"]
        if type(step) is not int or step < 0 or step in steps:
            _fail("selection", "checkpoint steps must be unique nonnegative integers")
        digest(checkpoint, "checkpoint_sha256")
        if (not isinstance(report, Mapping) or report.get("schema") != "nca-ik-direct-score/v1"
                or report.get("partition") != "tuning" or report.get("comparator") != COMPARATORS[0]
                or not report.get("complete") or report.get("selection_metric") is None):
            _fail("selection", "complete training-independent tuning scores are required")
        metric = report["selection_metric"]
        if type(metric) not in (int, float) or not math.isfinite(metric) or metric < 0:
            _fail("selection", "selection score must be finite and nonnegative")
        if metric != report["nominal"]["p95_error_cells"]:
            _fail("selection", "selection metric differs from raw nominal p95")
        steps.add(step)
        identities.add((report["config_sha256"], report["schedule_sha256"]))
        checked.append((metric, step, checkpoint, report))
    if len(identities) != 1:
        _fail("selection", "all checkpoints must use the identical tuning partition")
    metric, step, checkpoint, report = min(checked, key=lambda item: (item[0], item[1]))
    return {
        "step": step, "checkpoint_sha256": checkpoint,
        "tuning_report_sha256": sha256_json(report), "selection_metric": metric,
        "config_sha256": report["config_sha256"], "tuning_schedule_sha256": report["schedule_sha256"],
    }


def freeze_candidates(candidates, *, experiment, dataset_manifest_sha256,
                      final_suite_identity_sha256, registry_id):
    """Seal all three selected identities before the corpus opening transition."""
    identifier(experiment, "experiment")
    identifier(registry_id, "registry_id")
    digest(dataset_manifest_sha256, "dataset_manifest_sha256")
    digest(final_suite_identity_sha256, "final_suite_identity_sha256")
    if not isinstance(candidates, Mapping) or set(candidates) != set(SEEDS):
        _fail("freeze", "exactly the three independently trained seeds must be frozen")
    entries, hashes, suites = [], set(), set()
    fields = {"step", "checkpoint_sha256", "tuning_report_sha256", "selection_metric",
              "config_sha256", "tuning_schedule_sha256"}
    for seed in SEEDS:
        item = candidates[seed]
        if not isinstance(item, Mapping) or set(item) != fields:
            _fail("freeze", "exact selected-checkpoint receipt required")
        for name in ("checkpoint_sha256", "tuning_report_sha256", "config_sha256", "tuning_schedule_sha256"):
            digest(item[name], name)
        if type(item["step"]) is not int or item["step"] < 0:
            _fail("freeze", "selected step must be a nonnegative integer")
        metric = item["selection_metric"]
        if type(metric) not in (int, float) or not math.isfinite(metric) or metric < 0:
            _fail("freeze", "finite tuning selection metric required")
        hashes.add(item["checkpoint_sha256"])
        suites.add((item["config_sha256"], item["tuning_schedule_sha256"]))
        entries.append({"seed": seed, **dict(item)})
    if len(hashes) != 3 or len(suites) != 1:
        _fail("freeze", "three distinct checkpoint identities on one tuning suite required")
    return _seal({
        "schema": "nca-ik-frozen-candidates/v1", "experiment": experiment,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "final_suite_identity_sha256": final_suite_identity_sha256, "registry_id": registry_id,
        "candidates": entries, "qualified": False,
    })


def validate_final_opening(opening, frozen, *, seed):
    _sealed(frozen, "nca-ik-frozen-candidates/v1")
    entries = frozen.get("candidates", ())
    if not isinstance(entries, list) or len(entries) != 3:
        _fail("freeze", "all three frozen candidates are required")
    try:
        reconstructed = freeze_candidates(
            {entry["seed"]: {key: value for key, value in entry.items() if key != "seed"} for entry in entries},
            experiment=frozen["experiment"], dataset_manifest_sha256=frozen["dataset_manifest_sha256"],
            final_suite_identity_sha256=frozen["final_suite_identity_sha256"], registry_id=frozen["registry_id"],
        )
    except (KeyError, TypeError):
        _fail("freeze", "malformed candidate freeze")
    if reconstructed != dict(frozen) or type(seed) is not int or seed not in SEEDS:
        _fail("freeze", "frozen candidate identities or requested seed differ")
    _sealed(opening, "nca-ik-final-opening/v1")
    hashes = [entry["checkpoint_sha256"] for entry in entries]
    if (opening.get("experiment") != frozen["experiment"]
            or opening.get("frozen_candidates_sha256") != hashes
            or opening.get("suite_identity_sha256") != frozen["final_suite_identity_sha256"]
            or opening.get("dataset_manifest_sha256") != frozen["dataset_manifest_sha256"]
            or opening.get("registry_id") != frozen["registry_id"]
            or opening.get("future_experiments_status") != "development_only"):
        _fail("opening", "registry opening must bind this experiment and every frozen seed")
    return next(entry for entry in entries if entry["seed"] == seed)


def measure_cpu_latency(inputs, predictors, *, context, environment, warmup_calls=100,
                        measured_calls=10000, fixture=False):
    """Opt-in matched one-thread batch-one complete-decision timing.

    Feature construction belongs inside each predictor (model_predictor does
    this). Each timed call includes inference/solver work and independent raw
    bounds/FK checks. No host transport or physical tracking is claimed. Fixture
    counts cannot qualify the prospective deployment gate.
    """
    if context.device.type != "cpu" or context.torch.get_num_threads() != 1:
        _fail("timing", "explicit CPU device with one Torch thread required")
    if (set(predictors) != set(COMPARATORS) or not isinstance(environment, Mapping)
            or not environment):
        _fail("timing", "all matched predictors and declared deployment environment required")
    if (type(warmup_calls) is not int or type(measured_calls) is not int
            or warmup_calls < 1 or measured_calls < 1
            or (not fixture and (warmup_calls != 100 or measured_calls != 10000))):
        _fail("timing", "the measured protocol is exactly 100 warmups and 10000 calls per comparator")
    inputs = tuple(inputs)
    if not inputs or any(not isinstance(row, RuntimeInputs) for row in inputs):
        _fail("timing", "nonempty matched permitted inputs required")
    for row in inputs:
        runtime_features(row)
        if not all((*row.present, *row.measured, *row.active, *row.task_mask)):
            _fail("timing", "use scheduled complete nominal inputs for the utility comparison")
    chain = chain_from_profile(runner_right_sole_profile())
    timings = {name: [] for name in COMPARATORS}
    outcomes = {name: Counter() for name in COMPARATORS}

    def decision(predictor, row):
        prediction = predictor(row)
        if not isinstance(prediction, Prediction):
            return "malformed"
        q, nonfinite, malformed = _vector(prediction.raw_reference)
        if (q is None or nonfinite or malformed or prediction.status in REFUSALS
                or prediction.status in ("time_budget_exhausted", "iteration_budget_exhausted")):
            return "failed"
        if any(abs(v) > limit for v, limit in zip(q, (.72, .27))):
            return "bounds"
        endpoint = independent_two_link_point(chain, q)
        if not all(math.isfinite(v) for v in endpoint):
            return "nonfinite"
        return "complete"

    for index in range(warmup_calls):
        for name in COMPARATORS:
            decision(predictors[name], inputs[index % len(inputs)])
    for index in range(measured_calls):
        # Rotate first comparator to reduce a fixed ordering advantage.
        order = COMPARATORS[index % len(COMPARATORS):] + COMPARATORS[:index % len(COMPARATORS)]
        row = inputs[index % len(inputs)]
        for name in order:
            start = time.perf_counter()
            status = decision(predictors[name], row)
            elapsed = time.perf_counter() - start
            if not math.isfinite(elapsed) or elapsed < 0:
                _fail("timing", "nonfinite or negative elapsed measurement")
            timings[name].append(elapsed * 1000)
            outcomes[name][status] += 1
    result = {}
    for name, values in timings.items():
        result[name] = {
            "measured_calls": len(values), "p50_ms": nearest_rank(values, .5),
            "p95_ms": nearest_rank(values, .95), "p99_ms": nearest_rank(values, .99),
            "maximum_ms": max(values), "raw_milliseconds": values,
            "outcome_counts": dict(sorted(outcomes[name].items())),
        }
    return {
        "schema": "nca-ik-matched-cpu-latency/v1", "fixture": bool(fixture),
        "environment": dict(environment), "environment_sha256": sha256_json(environment),
        "input_schedule_sha256": sha256_json([row.sha256 for row in inputs]),
        "threads": 1, "batch_size": 1, "warmup_calls_per_comparator": warmup_calls,
        "measured_calls_per_comparator": measured_calls, "comparators": result,
        "scope": "permitted_features_inference_or_solver_independent_bounds_and_fk",
        "startup_ms": None, "transport_ms": None, "host_step_ms": None, "memory_bytes": None,
        "excluded_measurements_reason": "requires_separate_runtime_or_host_receipts",
        "qualified": False,
    }


def apply_cpu_utility(comparison, timing):
    """Attach the declared <=0.8 numerical-cost gate without conflating accuracy.

    Only complete nominal schedules, the exact five comparators, and the frozen
    100/10000 non-fixture protocol are eligible. Analytic cost is always visible.
    This assesses geometric software evidence, never CPU export/host authority.
    """
    if (not isinstance(comparison, Mapping)
            or comparison.get("schema") != "nca-ik-direct-comparison/v1"):
        _fail("utility", "a matched comparison report is required")
    result = dict(comparison)
    if timing is None:
        return result
    if (not isinstance(timing, Mapping) or timing.get("schema") != "nca-ik-matched-cpu-latency/v1"
            or timing.get("fixture") is not False or timing.get("threads") != 1
            or timing.get("batch_size") != 1 or timing.get("warmup_calls_per_comparator") != 100
            or timing.get("measured_calls_per_comparator") != 10000
            or set(timing.get("comparators", {})) != set(COMPARATORS)
            or timing.get("environment_sha256") != sha256_json(timing.get("environment"))):
        _fail("utility", "exact declared non-fixture CPU protocol required")
    candidate = comparison["comparators"][COMPARATORS[0]]
    numerical = comparison["comparators"]["independent_numerical_reference"]
    expected = [row["runtime_input_sha256"] for row in candidate["rows"] if row["kind"] != "stress"]
    if timing.get("input_schedule_sha256") != sha256_json(expected):
        _fail("utility", "timing inputs differ from the complete nominal comparison schedule")
    for name, values in timing["comparators"].items():
        raw = values.get("raw_milliseconds", ())
        if (values.get("measured_calls") != 10000 or len(raw) != 10000
                or values.get("p95_ms") != nearest_rank(raw, .95)
                or sum(values.get("outcome_counts", {}).values()) != 10000):
            _fail("utility", "latency summaries or complete outcome counts differ from raw calls")
    values = timing["comparators"]
    candidate_ms = values[COMPARATORS[0]]["p95_ms"]
    numerical_ms = values["independent_numerical_reference"]["p95_ms"]
    result["cpu_latency"] = timing
    result["cpu_p95_ratio_to_numerical"] = candidate_ms / numerical_ms if numerical_ms else None
    result["analytic_cpu_p95_ms"] = values["independent_analytic_reference"]["p95_ms"]
    if (_accuracy(candidate) != "pass" or _accuracy(numerical) != "pass"
            or values[COMPARATORS[0]]["outcome_counts"].get("complete", 0) != 10000
            or values["independent_numerical_reference"]["outcome_counts"].get("complete", 0) != 10000):
        result["practical_utility"] = "no_go"
        result["practical_utility_reason"] = "candidate_and_numerical_pipeline_do_not_both_meet_acceptance_quality"
    elif numerical_ms <= 0:
        result["practical_utility"] = "inconclusive"
        result["practical_utility_reason"] = "numerical_latency_cannot_support_a_finite_ratio"
    else:
        result["practical_utility"] = "pass" if candidate_ms <= .8 * numerical_ms else "no_go"
        result["practical_utility_reason"] = "declared_same_quality_numerical_cpu_p95_comparison"
    return result


def aggregate_comparisons(reports):
    """Pool raw observations, never average seed percentiles or drop failed seeds."""
    if not isinstance(reports, Mapping) or set(reports) != set(SEEDS):
        _fail("aggregate", "all three declared seeds must remain visible")
    identity, verdicts = set(), {}
    for seed in SEEDS:
        report = reports[seed]
        if (not isinstance(report, Mapping) or report.get("schema") != "nca-ik-direct-comparison/v1"
                or report.get("seed") != seed or report.get("partition") != "final"
                or set(report.get("comparators", {})) != set(COMPARATORS)):
            _fail("aggregate", "each seed needs its complete final comparison record")
        identity.add((report["config_sha256"], report["schedule_sha256"]))
        verdicts[str(seed)] = {name: report[name] for name in
                              ("accuracy", "learned_contribution", "practical_utility")}
    if len(identity) != 1:
        _fail("aggregate", "all seeds must use the identical final population")
    pooled = {}
    for name in COMPARATORS:
        rows = [row for seed in SEEDS for row in reports[seed]["comparators"][name]["rows"]
                if row["kind"] != "stress"]
        pooled[name] = _population(rows)
    combined = {}
    for name in ("accuracy", "learned_contribution", "practical_utility"):
        values = [verdicts[str(seed)][name] for seed in SEEDS]
        combined[name] = ("no_go" if "no_go" in values else
                          "pass" if all(value == "pass" for value in values) else "inconclusive")
    return {
        "schema": "nca-ik-three-seed-comparison/v1", "training_seeds": list(SEEDS),
        "per_seed": verdicts, "pooled_nominal": pooled, **combined, "qualified": False,
        "qualification_reason": "immutable_run_readback_and_independent_acceptance_remain_external",
    }
