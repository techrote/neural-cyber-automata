"""Actual arrival outcomes and conservative three-seed paired utility gates."""

from __future__ import annotations

import itertools
import math

from nca.contracts import ContractError
from nca.controllers.baseline import FeedbackObservation
from nca.controllers.shaping import wrapped_angle


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class ArrivalTask:
    """Own evaluator state is never passed to graph features or the reference."""
    def __init__(self, goal, initial, deadline_ticks=180):
        self.goal = goal
        self.start_tick = int(initial["tick"])
        self.last_tick = self.start_tick
        self.deadline_ticks = deadline_ticks
        self.quiet = 0
        self.success_tick = None
        self.integrity_failure = False
        self.failed_admissions = 0
        self.angles = []
        self.inference_ms = []
        self.distance = math.dist(initial["position"], goal.target)
        self.previous_primary = False
        self.primary_edges = 0
        self.launch_tick = None
        self.purposeful_launch = False

    @property
    def done(self):
        return (self.success_tick is not None or self.integrity_failure
                or self.last_tick - self.start_tick >= self.deadline_ticks)

    def consume(self, action, receipts, inference_ns=0):
        if self.done or not receipts:
            raise ContractError("graph_evaluation", "a completed outcome cannot be counted twice")
        self.inference_ms.append(inference_ns / 1e6)
        if action.primary and not self.previous_primary:
            self.primary_edges += 1
            self.launch_tick = self.last_tick
        self.previous_primary = action.primary
        for receipt in receipts:
            if receipt.get("failed") or receipt.get("tick_before") != self.last_tick:
                self.integrity_failure = True
                raise ContractError("graph_receipt", "invalid or noncontiguous actual receipt")
            view = FeedbackObservation.from_host(receipt["observation"])
            if not view.valid or view.tick != self.last_tick + 1:
                self.integrity_failure = True
                raise ContractError("graph_receipt", "invalid next physical observation")
            self.last_tick = view.tick
            self.failed_admissions += sum(item.get("accepted") is not True for item in receipt.get("admission", ()))
            self.integrity_failure |= self.failed_admissions > 0
            angle = abs(wrapped_angle(view.angle))
            self.angles.append(angle)
            self.distance = math.dist(view.position, self.goal.target)
            quiet = (self.distance <= self.goal.tolerance and math.hypot(*view.velocity) <= 6
                     and angle <= .32 and view.supported is True)
            self.quiet = self.quiet + 1 if quiet else 0
            if self.quiet >= 12 and self.success_tick is None:
                self.success_tick = view.tick
            if (self.launch_tick is not None and view.supported is False
                    and 0 < view.tick - self.launch_tick <= 12):
                self.purposeful_launch = True

    def metrics(self):
        success = self.success_tick is not None and not self.integrity_failure
        elapsed = self.last_tick - self.start_tick
        return {"success": success, "reason": "integrity_failure" if self.integrity_failure else
                "arrival" if success else "task_deadline" if elapsed >= self.deadline_ticks else "administrative_cutoff",
                "elapsed_ticks": elapsed, "final_distance_cells": self.distance,
                "capped_completion_seconds": (self.success_tick - self.start_tick) / 60 if success else self.deadline_ticks / 60,
                "integrity_failure": self.integrity_failure, "failed_admissions": self.failed_admissions,
                "mean_absolute_angle": sum(self.angles) / len(self.angles) if self.angles else None,
                "inference_p95_ms": percentile(self.inference_ms, .95),
                "primary_edges": self.primary_edges, "purposeful_launch": self.purposeful_launch,
                "owner_feel_accepted": False}


def mean(items):
    values = tuple(items)
    if not values:
        raise ContractError("graph_evaluation", "an empty outcome population has no mean")
    return sum(values) / len(values)


def bootstrap_three_seed(values):
    if len(values) != 3:
        raise ContractError("graph_uncertainty", "the screen requires all three trained policy seeds")
    resamples = [mean(values[index] for index in indices) for indices in itertools.product(range(3), repeat=3)]
    return {"mean": mean(values), "lower95": percentile(resamples, .025),
            "upper95": percentile(resamples, .975), "unit": "paired-policy-seed-mean", "resamples": 27}


def utility_gate(records, config):
    """Never replace a missing scenario or seed with a fabricated passing zero."""
    seeds = config["training_seeds"]
    families = [item["id"] for item in config["evaluation_families"]]
    faults = [item["id"] for item in config["faults"]]
    evaluation_rows = [row for row in records if row["stage"] == "evaluation"]
    indexed = {(row["seed"], row["method"], row["family"], row["fault"]): row["metrics"]
               for row in evaluation_rows}
    required = list(itertools.product(seeds, ("authored", "graph", "no_message"), families, faults))
    if (len(indexed) != len(required) or len(evaluation_rows) != len(required)
            or any(key not in indexed for key in required)):
        return {"decision": "inconclusive", "reason": "missing-or-duplicate-paired-evaluation"}

    for item in indexed.values():
        if (type(item.get("success")) is not bool or type(item.get("integrity_failure")) is not bool
                or type(item.get("failed_admissions")) is not int or item["failed_admissions"] < 0):
            return {"decision": "inconclusive", "reason": "invalid-evaluation-metrics"}
        for metric in ("capped_completion_seconds", "mean_absolute_angle", "inference_p95_ms"):
            value = item.get(metric)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                return {"decision": "inconclusive", "reason": "missing-or-nonfinite-evaluation-metric"}

    def values(method, fault, metric, seed=None):
        return [indexed[(chosen, method, family, fault)][metric]
                for chosen in (seeds if seed is None else (seed,)) for family in families]

    success_deltas = [mean(values("graph", "normal", "success", seed))
                      - mean(values("no_message", "normal", "success", seed)) for seed in seeds]
    time_deltas = [mean(values("no_message", "normal", "capped_completion_seconds", seed))
                   - mean(values("graph", "normal", "capped_completion_seconds", seed)) for seed in seeds]
    success_ci, time_ci = bootstrap_three_seed(success_deltas), bootstrap_three_seed(time_deltas)
    graph_success = mean(values("graph", "normal", "success"))
    no_message_success = mean(values("no_message", "normal", "success"))
    reference_success = mean(values("authored", "normal", "success"))
    gates = config["utility_gate"]
    improvement = (success_ci["mean"] >= gates["minimum_normal_success_gain_vs_no_message"] and success_ci["lower95"] > 0)
    alternative = (min(graph_success, no_message_success) >= gates["alternative_both_success_minimum"]
                   and time_ci["mean"] >= gates["alternative_capped_completion_gain_seconds"] and time_ci["lower95"] > 0)
    all_metrics = list(indexed.values())
    checks = {"normal_benefit": improvement or alternative,
              "reference_success": graph_success >= reference_success,
              "integrity": all(not item["integrity_failure"] and not item["failed_admissions"] for item in all_metrics),
              "angle": mean(values("graph", "normal", "mean_absolute_angle")) <= mean(values("authored", "normal", "mean_absolute_angle")) + gates["maximum_mean_absolute_angle_increase_vs_authored"],
              "inference": max(values("graph", "normal", "inference_p95_ms")) <= gates["maximum_inference_p95_ms"],
              "stress": all(graph_success - mean(values("graph", fault, "success")) <= gates["maximum_stress_success_loss_vs_normal"] for fault in faults)}
    return {"decision": "conditional-utility" if all(checks.values()) else "no-go-at-this-screen",
            "checks": checks, "success_gain": success_ci, "completion_gain_seconds": time_ci,
            "normal_success": {"graph": graph_success, "no_message": no_message_success, "authored": reference_success},
            "owner_feel_accepted": False,
            "scope": "one actor topology, two heldout families and three policy seeds; no general graph-method ranking"}
