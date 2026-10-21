#!/usr/bin/env python3
"""Supervise one declared reference benchmark and read back its sealed evidence.

The child timeout leaves a separate publication reserve. Child timing includes
interpreter startup and process teardown; the final tiny supervisor receipt write
is excluded from its own recorded elapsed field and checked again before exit.
"""

from __future__ import annotations

import time
_STARTED = time.perf_counter()

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from nca.contracts import sha256_json, strict_json_loads
from nca.runs import immutable_json, utc_now
from ik_reference_benchmark import (BenchmarkError, ROOT, _declaration,
                                    _new_destination, _outside_source)


def _journal_prefix(raw, declaration):
    """Retain durable starts and unknown work even without a sealed result."""
    events, starts, finished, errors = [], [], [], []
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            errors.append("truncated_journal_tail")
            break
        try:
            events.append(strict_json_loads(line))
        except (ValueError, UnicodeError):
            errors.append("invalid_journal_tail")
            break
    pending = None
    if not events or not isinstance(events[0], dict) or not (
            events[0].get("event") == "allocation_started"
            and events[0].get("run_id") == declaration["run_id"]
            and events[0].get("declaration_sha256") == declaration["sha256"]):
        errors.append("journal_start_binding_invalid")
    else:
        for event in events[1:]:
            if not isinstance(event, dict):
                errors.append("journal_event_invalid")
                break
            if pending is None and event.get("event") == "call_started":
                if event.get("method") not in declaration["config"]["methods"]:
                    errors.append("journal_method_invalid")
                    break
                starts.append(event)
                pending = event
            elif pending is not None and event.get("event") == "call_finished":
                row = event.get("row")
                if not isinstance(row, dict) or not all(
                        row.get(k) == pending.get(k) for k in ("phase", "method", "case_index")):
                    errors.append("journal_finish_binding_invalid")
                    break
                finished.append(row)
                pending = None
            else:
                errors.append("journal_event_order_invalid")
                break
    maximum = declaration["config"]["solver"]["max_iterations"]
    known_iterations, known_calls = 0, 0
    for row in finished:
        data = row.get("result")
        if row.get("method") == "dls" and isinstance(data, dict):
            count = data.get("iterations")
            if type(count) is int and 0 <= count <= maximum:
                known_iterations += count
                known_calls += 1
    dls_started = sum(e["method"] == "dls" for e in starts)
    unknown = dls_started - known_calls
    return events, {
        "valid_durable_prefix": not errors, "errors": errors,
        "parsed_events": len(events), "started_calls": len(starts),
        "finished_calls": len(finished), "incomplete_calls": len(starts) - len(finished),
        "incomplete_call_terminal_outcome": "unknown" if pending is not None else None,
        "numerical_iterations_known_completed_lower_bound_including_warmup": known_iterations,
        "numerical_calls_without_iteration_diagnostics": unknown,
        "numerical_iterations_conservative_charge_including_warmup": known_iterations + maximum * unknown,
        "numerical_iterations_exact": not errors and unknown == 0,
        "unparsed_tail_may_contain_additional_work": bool(errors),
    }


def supervise(declaration_path, output):
    declaration_path = _outside_source(declaration_path)
    output = _new_destination(output)
    receipt_path = _new_destination(output.with_name(output.name + ".supervisor.json"))
    # Bind the one-use admission to the declaration, before even spawning the
    # interpreter. Changing output paths cannot erase failed startup cost.
    started_path = _new_destination(declaration_path.with_name(declaration_path.name + ".supervisor-consumed.json"))
    worker_consumed_path = declaration_path.with_name(declaration_path.name + ".consumed.json")
    if worker_consumed_path.exists():
        raise BenchmarkError("declaration_already_consumed")
    declaration = _declaration(declaration_path)
    budget = declaration["config"]["budget"]
    maximum, reserve = budget["max_wall_seconds"], budget["publication_reserve_seconds"]
    immutable_json(started_path, {
        "schema": "nca-ik-reference-supervisor-start/v1", "run_id": declaration["run_id"],
        "declaration_sha256": declaration["sha256"], "started_utc": utc_now(),
        "aggregate_max_seconds": maximum,
        "child_timeout_max_seconds": budget["supervisor_child_timeout_seconds"],
        "publication_reserve_seconds": reserve, "attempt": 1, "retry_or_resume": False,
    })
    child_seconds, returncode, timed_out = None, None, False
    checks, artifacts, failures = {}, {}, []
    journal_readback = None
    try:
        allowance = min(budget["supervisor_child_timeout_seconds"],
                        maximum - reserve - (time.perf_counter() - _STARTED))
        if allowance <= 0:
            raise BenchmarkError("supervisor_admission_budget_exhausted")
        child_start = time.perf_counter()
        try:
            child = subprocess.run(
                [sys.executable, "tools/ik_reference_benchmark.py", "--run", str(declaration_path),
                 "--output", str(output)], cwd=ROOT, capture_output=True, timeout=allowance)
            returncode = child.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            child_seconds = time.perf_counter() - child_start
        if timed_out:
            failures.append("child_process_timeout")
        elif returncode != 0:
            failures.append("child_process_failed")

        # Inventory every surviving file before any sealed-result dependency.
        # A killed child can leave useful consumed markers and a journal prefix.
        paths = {
            "declaration": declaration_path, "supervisor_consumed": started_path,
            "journal": output.with_name(output.name + ".rows.jsonl"), "result": output,
            "worker_publication": output.with_name(output.name + ".terminal.json"),
            "consumed": worker_consumed_path,
        }
        raw_artifacts = {}
        for label, path in paths.items():
            if path.is_file():
                raw = path.read_bytes()
                raw_artifacts[label] = raw
                artifacts[label] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            else:
                artifacts[label] = {"missing": True}
        events = []
        if "journal" in raw_artifacts:
            events, journal_readback = _journal_prefix(raw_artifacts["journal"], declaration)

        def read(label, *, sealed=False):
            if label not in raw_artifacts:
                raise BenchmarkError("missing_artifact_" + label)
            raw = raw_artifacts[label]
            value = strict_json_loads(raw)
            if sealed and value.get("sha256") != sha256_json({k: v for k, v in value.items() if k != "sha256"}):
                raise BenchmarkError("artifact_self_digest_mismatch")
            return value

        observed_declaration = read("declaration", sealed=True)
        supervisor_consumed = read("supervisor_consumed")
        consumed = read("consumed")
        result = read("result", sealed=True)
        terminal = read("worker_publication", sealed=True)
        checks["schemas"] = (result["schema"] == "nca-ik-reference-benchmark-result/v1"
                              and terminal["schema"] == "nca-ik-reference-benchmark-publication/v1"
                              and consumed["schema"] == "nca-ik-reference-benchmark-consumed/v1"
                              and supervisor_consumed["schema"] == "nca-ik-reference-supervisor-start/v1")
        checks["declaration_unchanged"] = observed_declaration == declaration
        checks["identities"] = all(v["run_id"] == declaration["run_id"] for v in (result, terminal, consumed, supervisor_consumed))
        checks["bindings"] = all(v["declaration_sha256"] == declaration["sha256"] for v in (result, terminal, consumed, supervisor_consumed))
        checks["source_environment_config"] = all(result[k] == declaration[k]
                                                     for k in ("source", "environment", "config", "config_sha256",
                                                               "profile_sha256", "host_revision", "acceptance_file_sha256"))
        checks["artifact_digests"] = (terminal["result_file_sha256"] == artifacts["result"]["sha256"]
                                      and terminal["result_content_sha256"] == result["sha256"]
                                      and result["consumed_marker_file_sha256"] == artifacts["consumed"]["sha256"])
        checks["admission_and_completion"] = all(result["admission"].values()) and all(result["completion"].values())
        checks["worker_complete"] = (result["status"] == "measurements_complete" and terminal["within_module_wall_budget"]
                                      and terminal["status"] == "published" and terminal["result_status"] == result["status"]
                                      and not result["failures"])
        rows = result["rows"]
        starts = events[1::2]
        finishes = events[2::2]
        checks["journal_matches_all_calls"] = (
            journal_readback is not None and journal_readback["valid_durable_prefix"]
            and journal_readback["incomplete_calls"] == 0
            and len(events) == 1 + 2 * len(rows)
            and events[0]["event"] == "allocation_started"
            and events[0]["run_id"] == declaration["run_id"]
            and events[0]["declaration_sha256"] == declaration["sha256"]
            and all(e["event"] == "call_started" and all(e[k] == r[k] for k in ("phase", "method", "case_index"))
                    for e, r in zip(starts, rows))
            and all(e["event"] == "call_finished" and e["row"] == r for e, r in zip(finishes, rows))
            and result["raw_journal"]["file_sha256"] == artifacts["journal"]["sha256"]
            and result["raw_journal"]["started_calls"] == result["raw_journal"]["finished_calls"] == len(rows))
        checks["counts"] = (len(rows) == budget["max_calls_total"]
                            and result["allocation"]["calls_attempted"] == len(rows)
                            and result["allocation"]["numerical_iterations_conservative_charge_including_warmup"]
                            <= budget["max_numerical_iterations_total"]
                            and all(sum(r["method"] == method and r["phase"] == phase for r in rows) == count
                                    for method in declaration["config"]["methods"]
                                    for phase, count in (("warmup", 8), ("measured", 243))))
        iterations = sum(r["result"]["iterations"] for r in rows if r["method"] == "dls")
        checks["iteration_accounting"] = (
            result["allocation"]["numerical_iterations_exact"]
            and result["allocation"]["numerical_calls_without_iteration_diagnostics"] == 0
            and iterations == result["allocation"]["numerical_iterations_known_completed_lower_bound_including_warmup"]
            == result["allocation"]["numerical_iterations_conservative_charge_including_warmup"]
            == journal_readback["numerical_iterations_conservative_charge_including_warmup"]
            and iterations <= budget["max_numerical_iterations_total"]
            and all(type(r["result"]["iterations"]) is int
                    and 0 <= r["result"]["iterations"] <= declaration["config"]["solver"]["max_iterations"]
                    and 1 <= r["result"]["evaluations"] <= 1 + 64 * (8 + 4)
                    and r["result"]["maximum_step_norm"] <= .25 + 1e-12
                    for r in rows if r["method"] == "dls"))
        checks["all_returned_endpoints"] = True
        expected_calls = [(phase, method, index)
                          for phase, indexes in (("warmup", range(8)), ("measured", range(243)))
                          for index in indexes for method in ("dls", "analytic-2r")]
        checks["declared_case_order"] = [(r["phase"], r["method"], r["case_index"]) for r in rows] == expected_calls
        for row in rows:
            data = row["result"]
            q = data.get("configuration") if row["method"] == "dls" else data["configurations"][0]
            h, a = q
            point = (3.6 - 4.95 * math.sin(h) + math.cos(h + a) - 1.25 * math.sin(h + a),
                     2.8 + 4.95 * math.cos(h) + math.sin(h + a) + 1.25 * math.cos(h + a))
            grid_h, grid_a, initial = row["case_index"] // 27, (row["case_index"] // 3) % 9, row["case_index"] % 3
            th, ta = -.6 + 1.2 * grid_h / 8, -.2 + .4 * grid_a / 8
            target = (3.6 - 4.95 * math.sin(th) + math.cos(th + ta) - 1.25 * math.sin(th + ta),
                      2.8 + 4.95 * math.cos(th) + math.sin(th + ta) + 1.25 * math.cos(th + ta))
            checks["all_returned_endpoints"] &= bool(
                row["accepted"] and data["status"] == "solved"
                and -.72 <= h <= .72 and -.27 <= a <= .27
                and math.isfinite(row["core_seconds"]) and row["core_seconds"] >= 0
                and row["grid_bin"] == [grid_h, grid_a] and row["initial_index"] == initial
                and math.dist(row["target_configuration"], (th, ta)) <= 1e-12
                and row["initial_configuration"] == declaration["config"]["initial_configurations"][initial]
                and math.dist(target, row["target"]) <= 1e-12
                and math.dist(point, row["target"]) <= 1e-6)
        checks["child_exit"] = returncode == 0 and not timed_out
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        failures.append(exc.code if isinstance(exc, BenchmarkError) else "readback_or_execution_" + type(exc).__name__)
    elapsed = time.perf_counter() - _STARTED
    checks["aggregate_within_budget"] = elapsed < maximum
    status = "completed" if checks and all(checks.values()) and not failures else "capped" if timed_out or elapsed >= maximum else "failed"
    receipt = {
        "schema": "nca-ik-reference-supervisor/v1", "run_id": declaration["run_id"],
        "declaration_sha256": declaration["sha256"], "status": status,
        "checks": checks, "failures": failures, "artifacts": artifacts,
        "durable_journal_prefix": journal_readback,
        "child_whole_process_seconds": child_seconds, "child_returncode": returncode,
        "child_timed_out": timed_out, "aggregate_through_readback_seconds": elapsed,
        "aggregate_cap_seconds": maximum, "final_receipt_write_excluded_from_recorded_elapsed": True,
        "attempt": 1, "retry_or_resume": False, "completed_utc": utc_now(),
    }
    immutable_json(receipt_path, {**receipt, "sha256": sha256_json(receipt)})
    within_final_budget = time.perf_counter() - _STARTED < maximum
    return receipt, within_final_budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt, within_budget = supervise(args.declaration, args.output)
        print(json.dumps({"status": receipt["status"], "checks": receipt["checks"],
                          "child_whole_process_seconds": receipt["child_whole_process_seconds"],
                          "aggregate_through_readback_seconds": receipt["aggregate_through_readback_seconds"],
                          "within_budget_after_receipt": within_budget}, sort_keys=True))
        return 0 if receipt["status"] == "completed" and within_budget else 1
    except (BenchmarkError, ValueError, OSError) as exc:
        print(json.dumps({"status": "refused", "reason": exc.code if isinstance(exc, BenchmarkError)
                          else "immutable_supervisor_publication_failed"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
