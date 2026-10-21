#!/usr/bin/env python3
"""Declare then execute one immutable, bounded CPU IK reference comparison.

Only --run calls a solver. Publish the declaration and its exact source before
execution. The module clock includes imports/admission/warmup/measurement and
publication; interpreter startup and process teardown are explicitly unmeasured.
A consumed declaration cannot be retried or resumed by this command.
A separately published supervisor owns the whole-process cap and final verdict.
The append-only call journal survives an externally killed worker.
"""

from __future__ import annotations

import time
_PROCESS_STARTED = time.perf_counter()

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid

from nca.contracts import canonical_json, sha256_json, strict_json_loads
from nca.environment import environment_report
from nca.ik.declaration import source_snapshot, _verify_committed_sources
from nca.ik.kinematics import chain_from_profile
from nca.ik.profiles import runner_right_sole_profile
from nca.ik.reference import SolverConfig, analytic_two_link, solve_numeric
from nca.runs import immutable_json, utc_now


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "configs/ik/reference_benchmark_v1.json"
ACCEPTANCE_PATH = "configs/ik/runner_right_sole_acceptance_v1.json"
EXTRA_SOURCES = ("tools/ik_reference_benchmark.py", "tools/ik_reference_supervise.py",
                 CONFIG_PATH, ACCEPTANCE_PATH)
DECLARATION_SCHEMA = "nca-ik-reference-benchmark-declaration/v1"
RESULT_SCHEMA = "nca-ik-reference-benchmark-result/v1"
DECLARATION_FIELDS = {
    "schema", "run_id", "created_utc", "source", "environment", "environment_sha256",
    "config", "config_sha256", "profile_sha256", "host_revision", "acceptance_file_sha256", "sha256",
}


class BenchmarkError(ValueError):
    """Public-safe error code; paths and raw environment exceptions stay local."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def fixed_config():
    """The frozen v1 workload; changing it requires a new source/declaration."""
    return {
        "schema": "nca-ik-reference-benchmark-config/v1", "id": "runner-reference-cpu-v1",
        "device": "cpu", "profile": "runner-right-sole@1",
        "partition": "geometric-conformance-only-not-learned-splits",
        "generator": "independent-authored-runner-vectors/v1", "random_seed": None,
        "methods": ["dls", "analytic-2r"],
        "order": "case-major-dls-then-analytic; warmups precede measured cases",
        "grid": {"hip": [-.6, .6, 9], "ankle": [-.2, .2, 9]},
        "initial_configurations": [[0., 0.], [-.65, .24], [.65, -.24]],
        "warmup_calls_per_method": 8, "measured_calls_per_method": 243,
        "solver": {"max_iterations": 64, "max_seconds": .05, "damping": .02,
                   "max_step": .25, "max_backtracks": 8, "tolerance_cells": 1e-7,
                   "posture_weight": .002, "singular_probe": .05},
        "acceptance": {"endpoint_residual_cells_max": 1e-6, "require_solved_status": True,
                       "require_finite_and_target_bounded": True, "timing_threshold_seconds": None},
        "budget": {"max_wall_seconds": 30., "publication_reserve_seconds": 5.,
                   "supervisor_child_timeout_seconds": 25.,
                   "max_calls_total": 502, "max_numerical_iterations_total": 16064,
                   "attempts": 1, "retry_or_resume": False,
                   "whole_process_supervisor_required": True},
        "measurement": {
            "timed": "core solver call including returned diagnostic construction",
            "excluded_from_core": ["target generation", "independent endpoint check",
                                   "target and lifecycle validation", "model", "transport",
                                   "host motion", "result publication"],
            "aggregate_clock": "first module clock read after importing time, through terminal receipt preparation; interpreter startup and process teardown unmeasured",
            "whole_process_acceptance": "separately published supervisor starts before spawn, enforces 30 seconds, and retains exact code and terminal receipt",
            "percentiles": "nearest-rank ceil(p*N) over all measured calls including failures",
            "cuda": "unmeasured", "physical": "unmeasured", "learned_comparison": "not performed",
        },
    }


def _read_json(path):
    try:
        with Path(path).open("rb") as stream:
            return strict_json_loads(stream.read(1_048_577))
    except (OSError, ValueError) as exc:
        raise BenchmarkError("unreadable_or_invalid_json") from exc


def _config(value=None):
    value = _read_json(ROOT / CONFIG_PATH) if value is None else value
    if canonical_json(value) != canonical_json(fixed_config()):
        raise BenchmarkError("fixed_configuration_mismatch")
    return value


def _with_digest(value):
    return {**value, "sha256": sha256_json(value)}


def _source():
    try:
        snapshot = source_snapshot(ROOT)
        files = dict(snapshot["files"])
        for relative in EXTRA_SOURCES:
            files[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        _verify_committed_sources(ROOT, files)
        tree = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"],
            text=True, stderr=subprocess.DEVNULL).strip()
        if re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", tree) is None:
            raise ValueError("invalid tree")
        return {"revision": snapshot["revision"], "tree": tree,
                "files": files, "source_sha256": sha256_json(files)}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise BenchmarkError("source_not_clean_or_not_committed") from exc


def _cpu_model():
    # Read only the public processor model. Never publish hostname, machine ID,
    # processor serial, full /proc contents, environment variables or paths.
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()[:200]
        except OSError:
            pass
    return platform.processor()[:200] or None


def _environment():
    report = environment_report()
    report.update(cpu_model=_cpu_model(), os_release=platform.release(),
                  logical_cpu_count=os.cpu_count(), device="cpu",
                  core_parallelism="single Python execution thread; no Torch or NumPy arithmetic")
    return report


def _outside_source(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise BenchmarkError("evidence_destination_must_be_outside_source_checkout")
    return path


def _new_destination(path):
    path = _outside_source(path)
    if path.exists():
        raise BenchmarkError("immutable_destination_exists")
    return path


def declare(output):
    output = _new_destination(output)
    config, source = _config(), _source()
    environment, profile = _environment(), runner_right_sole_profile()
    declaration = _with_digest({
        "schema": DECLARATION_SCHEMA, "run_id": "ik-reference-" + uuid.uuid4().hex,
        "created_utc": utc_now(), "source": source,
        "environment": environment, "environment_sha256": sha256_json(environment),
        "config": config, "config_sha256": sha256_json(config),
        "profile_sha256": profile.sha256, "host_revision": profile.source.commit,
        "acceptance_file_sha256": source["files"][ACCEPTANCE_PATH],
    })
    immutable_json(output, declaration)
    return declaration


def _declaration(path):
    value = _read_json(path)
    if not isinstance(value, dict) or set(value) != DECLARATION_FIELDS:
        raise BenchmarkError("declaration_fields_mismatch")
    if (value["schema"] != DECLARATION_SCHEMA
            or not isinstance(value["run_id"], str)
            or re.fullmatch(r"ik-reference-[a-f0-9]{32}", value["run_id"]) is None):
        raise BenchmarkError("declaration_identity_mismatch")
    try:
        datetime.strptime(value["created_utc"], "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as exc:
        raise BenchmarkError("declaration_timestamp_invalid") from exc
    _config(value["config"])
    if (value["sha256"] != sha256_json({k: v for k, v in value.items() if k != "sha256"})
            or value["config_sha256"] != sha256_json(value["config"])
            or value["environment_sha256"] != sha256_json(value["environment"])):
        raise BenchmarkError("declaration_digest_mismatch")
    source = value["source"]
    if (not isinstance(source, dict) or set(source) != {"revision", "tree", "files", "source_sha256"}
            or not isinstance(source["files"], dict)
            or source["source_sha256"] != sha256_json(source["files"])):
        raise BenchmarkError("source_declaration_mismatch")
    return value


def runner_point(q):
    """Independent authored-vector oracle; no FK or solver helper is called."""
    hip, ankle = q
    return (3.6 - 4.95 * math.sin(hip) + math.cos(hip + ankle) - 1.25 * math.sin(hip + ankle),
            2.8 + 4.95 * math.cos(hip) + math.sin(hip + ankle) + 1.25 * math.cos(hip + ankle))


def cases(config):
    result = []
    hip_lo, hip_hi, hip_count = config["grid"]["hip"]
    ankle_lo, ankle_hi, ankle_count = config["grid"]["ankle"]
    for hi in range(hip_count):
        hip = hip_lo + (hip_hi - hip_lo) * hi / (hip_count - 1)
        for ai in range(ankle_count):
            ankle = ankle_lo + (ankle_hi - ankle_lo) * ai / (ankle_count - 1)
            for initial_index, initial in enumerate(config["initial_configurations"]):
                result.append({"case_index": len(result), "grid_bin": [hi, ai],
                               "initial_index": initial_index, "target_configuration": [hip, ankle],
                               "target": list(runner_point((hip, ankle))),
                               "initial_configuration": list(initial)})
    if len(result) != config["measured_calls_per_method"]:
        raise BenchmarkError("case_count_mismatch")
    return result


def _percentile(values, proportion):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(proportion * len(values)) - 1)]


def _numerical_accounting(rows, max_iterations):
    """Retain known work and reserve the full cap for missing diagnostics.

    A row is reserved before its durable start receipt and solver invocation.
    Charging every reserved DLS row also conservatively covers interruption
    while writing that receipt; the journal records the durable start prefix.
    """
    selected = [row for row in rows if row["method"] == "dls"]
    iterations = [(row["result"] or {}).get("iterations") for row in selected]
    projections = [(row["result"] or {}).get("projection_events") for row in selected]
    known_iterations = [value for value in iterations if type(value) is int and value >= 0]
    known_projections = [value for value in projections if type(value) is int and value >= 0]
    unknown_iterations = len(selected) - len(known_iterations)
    return {
        "numerical_iterations_known_completed_lower_bound_including_warmup": sum(known_iterations),
        "numerical_calls_without_iteration_diagnostics": unknown_iterations,
        "numerical_iterations_conservative_charge_including_warmup": (
            sum(known_iterations) + unknown_iterations * max_iterations),
        "numerical_iterations_exact": unknown_iterations == 0,
        "numerical_projection_events_known_completed_lower_bound_including_warmup": sum(known_projections),
        "numerical_calls_without_projection_diagnostics": len(selected) - len(known_projections),
    }


def _summary(rows, method, max_iterations):
    selected = [row for row in rows if row["method"] == method]
    measured = [row for row in selected if row["phase"] == "measured"]
    durations = [row["core_seconds"] for row in measured if row["core_seconds"] is not None]
    residuals = [row["independent_residual_cells"] for row in measured
                 if row["independent_residual_cells"] is not None]
    return {
        "attempted_calls": len(selected),
        "warmup_calls": sum(row["phase"] == "warmup" for row in selected),
        "measured_calls": len(measured),
        "measured_accepted": sum(row["accepted"] for row in measured),
        "measured_failed": sum(not row["accepted"] for row in measured),
        "all_call_status_counts": dict(Counter(row["status"] for row in selected)),
        "core_seconds_total_including_warmup": sum(row["core_seconds"] or 0. for row in selected),
        "measured_core_seconds": {"median": _percentile(durations, .5),
                                  "p95": _percentile(durations, .95),
                                  "maximum": max(durations, default=None)},
        "measured_independent_residual_cells_max": max(residuals, default=None),
        **_numerical_accounting(selected, max_iterations),
    }


def _append_receipt(stream, value):
    """Publish a durable prefix; a truncated tail can never claim completion."""
    stream.write(canonical_json(value) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def run(declaration_path, output):
    declaration_path = _outside_source(declaration_path)
    output = _new_destination(output)
    terminal_path = _new_destination(output.with_name(output.name + ".terminal.json"))
    journal_path = _new_destination(output.with_name(output.name + ".rows.jsonl"))
    declaration = _declaration(declaration_path)
    config = declaration["config"]
    # This immutable marker is the allocation's one-use lease. Keep it beside
    # the original declaration; moving/copying a declaration does not authorize
    # another attempt. A failed, capped or interrupted attempt stays consumed.
    consumed_path = declaration_path.with_name(declaration_path.name + ".consumed.json")
    if len({declaration_path, consumed_path, output, terminal_path, journal_path}) != 5:
        raise BenchmarkError("evidence_destinations_overlap")
    try:
        consumed_sha256 = immutable_json(consumed_path, {
            "schema": "nca-ik-reference-benchmark-consumed/v1", "run_id": declaration["run_id"],
            "declaration_sha256": declaration["sha256"], "created_utc": utc_now(),
            "attempt": 1, "retry_or_resume": False,
            "module_elapsed_seconds": time.perf_counter() - _PROCESS_STARTED,
        })
    except ValueError as exc:
        raise BenchmarkError("declaration_already_consumed") from exc
    rows, failures = [], []
    journal = None
    started_calls = finished_calls = 0
    status = "admission_failed"
    source_verified = environment_verified = config_verified = False
    completion_source_verified = completion_environment_verified = False
    start_utc = utc_now()
    budget = config["budget"]
    admission_deadline = _PROCESS_STARTED + budget["max_wall_seconds"] - budget["publication_reserve_seconds"]
    try:
        journal = journal_path.open("x", encoding="utf-8", newline="\n")
        _append_receipt(journal, {"event": "allocation_started", "run_id": declaration["run_id"],
                                 "declaration_sha256": declaration["sha256"],
                                 "elapsed_module_seconds": time.perf_counter() - _PROCESS_STARTED})
        if _source() != declaration["source"]:
            raise BenchmarkError("source_identity_changed")
        source_verified = True
        if _environment() != declaration["environment"]:
            raise BenchmarkError("environment_identity_changed")
        environment_verified = True
        if _config() != config:
            raise BenchmarkError("configuration_identity_changed")
        profile = runner_right_sole_profile()
        if (profile.sha256 != declaration["profile_sha256"]
                or profile.source.commit != declaration["host_revision"]
                or declaration["source"]["files"][ACCEPTANCE_PATH] != declaration["acceptance_file_sha256"]):
            raise BenchmarkError("profile_or_acceptance_identity_changed")
        config_verified = True
        chain = chain_from_profile(profile)
        lower = tuple(j.target_lower for j in profile.effector_chain)
        upper = tuple(j.target_upper for j in profile.effector_chain)
        solver_config = SolverConfig.from_dict(config["solver"])
        populations = cases(config)
        status = "running"
        for phase, population in (("warmup", populations[:config["warmup_calls_per_method"]]),
                                  ("measured", populations)):
            for case in population:
                for method in config["methods"]:
                    # Do not spend the publication reserve or admit a numerical
                    # call whose full per-call allocation crosses its boundary.
                    if (time.perf_counter() + solver_config.max_seconds >= admission_deadline
                            or len(rows) >= budget["max_calls_total"]):
                        raise BenchmarkError("allocation_capped")
                    row = {**case, "phase": phase, "method": method,
                           "core_seconds": None, "status": "incomplete", "result": None,
                           "independent_point": None, "independent_residual_cells": None,
                           "finite_target_bounded": False, "accepted": False, "error_type": None}
                    rows.append(row)
                    _append_receipt(journal, {"event": "call_started", "phase": phase,
                                             "method": method, "case_index": case["case_index"],
                                             "elapsed_module_seconds": time.perf_counter() - _PROCESS_STARTED})
                    started_calls += 1
                    try:
                        before = time.perf_counter()
                        try:
                            if method == "dls":
                                solution = solve_numeric(chain, case["target"], case["initial_configuration"],
                                                         config=solver_config, lower=lower, upper=upper)
                            else:
                                solution = analytic_two_link(chain, case["target"], case["initial_configuration"],
                                                             lower=lower, upper=upper)
                        finally:
                            row["core_seconds"] = time.perf_counter() - before
                        row["result"], row["status"] = solution.to_dict(), solution.status
                        q = solution.configuration
                        if (q is not None and len(q) == 2 and all(math.isfinite(value) for value in q)
                                and all(lo <= value <= hi for value, lo, hi in zip(q, lower, upper))):
                            point = runner_point(q)
                            row["finite_target_bounded"] = all(math.isfinite(value) for value in point)
                            row["independent_point"] = list(point)
                            row["independent_residual_cells"] = math.dist(point, case["target"])
                        row["accepted"] = bool(
                            solution.status == "solved" and row["finite_target_bounded"]
                            and row["independent_residual_cells"] <= config["acceptance"]["endpoint_residual_cells_max"])
                    except BaseException as exc:
                        row["error_type"] = type(exc).__name__
                        row["status"] = "exception"
                        raise
                    finally:
                        _append_receipt(journal, {"event": "call_finished", "row": row,
                                                 "elapsed_module_seconds": time.perf_counter() - _PROCESS_STARTED})
                        finished_calls += 1
        if _source() != declaration["source"]:
            raise BenchmarkError("source_identity_changed_during_execution")
        completion_source_verified = True
        if _environment() != declaration["environment"]:
            raise BenchmarkError("environment_identity_changed_during_execution")
        completion_environment_verified = True
        status = "measurements_complete" if all(row["accepted"] for row in rows) else "failed"
        if status == "failed":
            failures.append("one_or_more_reference_calls_failed_endpoint_acceptance")
    except KeyboardInterrupt:
        status = "cancelled"
        failures.append("keyboard_interrupt")
    except BaseException as exc:
        code = exc.code if isinstance(exc, BenchmarkError) else "execution_exception_" + type(exc).__name__
        status = "capped" if code == "allocation_capped" else ("admission_failed" if not config_verified else "failed")
        failures.append(code)
    finally:
        if journal is not None:
            journal.close()
    elapsed = time.perf_counter() - _PROCESS_STARTED
    if elapsed >= budget["max_wall_seconds"]:
        status = "capped"
        failures.append("module_wall_budget_exhausted")
    result = _with_digest({
        "schema": RESULT_SCHEMA, "run_id": declaration["run_id"],
        "declaration_sha256": declaration["sha256"], "consumed_marker_file_sha256": consumed_sha256,
        "source": declaration["source"], "environment": declaration["environment"],
        "config": config, "config_sha256": declaration["config_sha256"],
        "profile_sha256": declaration["profile_sha256"], "host_revision": declaration["host_revision"],
        "acceptance_file_sha256": declaration["acceptance_file_sha256"],
        "started_utc": start_utc, "status": status, "failures": failures,
        "admission": {"source_verified": source_verified, "environment_verified": environment_verified,
                      "config_profile_acceptance_verified": config_verified},
        "completion": {"source_verified": completion_source_verified,
                       "environment_verified": completion_environment_verified},
        "measurement_scope": config["measurement"],
        "allocation": {"attempt": 1, "consumed": True, "retry_or_resume": False,
                       "calls_attempted": len(rows), "calls_planned": budget["max_calls_total"],
                       "module_elapsed_before_result_publication_seconds": elapsed,
                       "whole_process_seconds": None,
                       **_numerical_accounting(rows, config["solver"]["max_iterations"])},
        "raw_journal": {"file_sha256": hashlib.sha256(journal_path.read_bytes()).hexdigest()
                        if journal_path.is_file() else None,
                        "started_calls": started_calls, "finished_calls": finished_calls,
                        "append_only": True},
        "methods": {method: _summary(rows, method, config["solver"]["max_iterations"])
                    for method in config["methods"]},
        "rows": rows,
        "qualification": {"nominal_reference_fixture_only": True, "learned": "not performed",
                          "cuda": "unmeasured", "physical": "unmeasured", "runtime_export": "unmeasured",
                          "whole_process_acceptance": "awaiting_separate_supervisor_receipt"},
    })
    result_file_sha256 = immutable_json(output, result)
    terminal_elapsed = time.perf_counter() - _PROCESS_STARTED
    terminal = _with_digest({
        "schema": "nca-ik-reference-benchmark-publication/v1", "run_id": declaration["run_id"],
        "declaration_sha256": declaration["sha256"], "result_file_sha256": result_file_sha256,
        "result_content_sha256": result["sha256"], "result_status": status,
        "status": "published" if terminal_elapsed < budget["max_wall_seconds"] else "capped",
        "module_elapsed_through_result_publication_seconds": terminal_elapsed,
        "within_module_wall_budget": terminal_elapsed < budget["max_wall_seconds"],
        "whole_process_seconds": None,
        "timing_limit": "excludes interpreter startup, this terminal receipt write and process teardown",
        "retry_or_resume": False, "completed_utc": utc_now(),
    })
    immutable_json(terminal_path, terminal)
    return result, terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--declare", type=Path, metavar="OUTPUT")
    group.add_argument("--run", type=Path, metavar="DECLARATION")
    parser.add_argument("--output", type=Path, help="new immutable result file, required with --run")
    args = parser.parse_args()
    if (args.run is not None) != (args.output is not None):
        parser.error("--output is required exactly when --run is selected")
    try:
        if args.declare is not None:
            value = declare(args.declare)
            print(json.dumps({"status": "declared", "run_id": value["run_id"],
                              "declaration_sha256": value["sha256"],
                              "source_revision": value["source"]["revision"],
                              "source_tree": value["source"]["tree"]}, sort_keys=True))
            return 0
        result, terminal = run(args.run, args.output)
        print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                          "result_content_sha256": result["sha256"], "methods": result["methods"],
                          "within_module_wall_budget": terminal["within_module_wall_budget"]}, sort_keys=True))
        return 0 if result["status"] == "measurements_complete" and terminal["within_module_wall_budget"] else 1
    except (BenchmarkError, OSError, ValueError) as exc:
        print(json.dumps({"status": "refused", "reason": exc.code if isinstance(exc, BenchmarkError)
                          else "immutable_evidence_publication_failed"}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
