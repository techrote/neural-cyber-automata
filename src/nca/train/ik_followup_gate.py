"""Fail-closed final-opening gate for the fixed three-seed polar follow-up.

This is a gate component, not an experiment launcher or execution admission.
The caller must authenticate reports/checkpoints against its declared ledger.
"""
import math
from nca.evaluation.ik_training import _accuracy
from nca.train.ik_experiment import SEEDS


def tuning_stop_gate(reports):
    if not isinstance(reports, dict) or set(reports) != set(SEEDS):
        raise ValueError("all three original seed reports required before final opening")
    verdicts = {}
    for seed, report in reports.items():
        if not isinstance(report, dict) or report.get("complete") is not True:
            raise ValueError("complete authenticated tuning report required")
        nominal = report.get("nominal")
        if not isinstance(nominal, dict):
            raise ValueError("finite tuning quality statistics required")
        for key in ("p95_error_cells", "worst_error_cells", "feasible_success_fraction"):
            value = nominal.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("finite nonnegative tuning quality statistics required")
        if (nominal["feasible_success_fraction"] > 1
                or nominal["p95_error_cells"] > nominal["worst_error_cells"]):
            raise ValueError("consistent tuning quality statistics required")
        if (report.get("schema") != "nca-ik-direct-score/v1"
                or report.get("partition") != "tuning"
                or report.get("comparator") != "trained_raw_reference"
                or report.get("scheduled_calls") != 2020
                or report.get("received_samples") != 2020
                or report.get("received_predictions") != 2020
                or len(report.get("rows", [])) != 2020
                or len({row.get("task_id") for row in report["rows"]}) != 2020
                or report.get("nominal", {}).get("scheduled") != 1880):
            raise ValueError("complete fixed original tuning schedule required")
        verdicts[seed] = _accuracy(report)
    return {"schema": "nca-ik-polar-tuning-stop/v1", "per_seed": verdicts,
            "final_opening_allowed": all(v == "pass" for v in verdicts.values()),
            "reason": "all_seed_raw_quality_gate"}


def open_after_tuning(reports, opener):
    gate = tuning_stop_gate(reports)
    if not gate["final_opening_allowed"]:
        raise ValueError("tuning stop gate refused; fresh final must remain sealed")
    return opener()

# Copyright 2026 techrote. All rights reserved.
