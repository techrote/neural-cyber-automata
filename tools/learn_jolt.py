#!/usr/bin/env python3
"""Freeze, execute once, or verify the bounded LEARN-JOLT-002 allocation.

The outer process owns the original clock and process tree. Admission and
runtime import time are included; a failed invocation cannot be retried under
the same output directory. Commands never acquire a host or fit a model during
freeze or inventory verification.
"""

from __future__ import annotations

import time
ENTRY_MONOTONIC = time.monotonic()

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_RESERVATION_BYTES = 512 * 1024 * 1024


def _checked(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path, *, limit=MAX_JSON_BYTES):
    from nca.contracts import strict_json_loads
    _checked(path.stat().st_size <= limit, "JSON evidence exceeds its bounded reader")
    return strict_json_loads(path.read_bytes(), max_bytes=limit)


def _array_hash(digest):
    value = digest.copy()
    value.update(b"]")
    return value.hexdigest()


def _cost_snapshot(budget, rows, snapshot, *, closed, prefix=None):
    """Recompute conserved costs using only the immutable reservation bytes."""
    from nca.contracts import sha256_json
    maximum, reserve = budget["maximum_wall_seconds"], budget["cleanup_reserve_seconds"]
    _checked(type(maximum) is int and maximum == 1800
             and type(reserve) is int and reserve == 120,
             "closed costs changed the original wall/cleanup allocation")
    limits, stage_limits = budget["counters"], budget["stages"]
    _checked(isinstance(limits, dict) and isinstance(stage_limits, dict)
             and all(type(value) is int and value >= 0 for value in limits.values())
             and all(isinstance(values, dict) and set(values) <= set(limits)
                     and all(type(value) is int and value >= 0 for value in values.values())
                     for values in stage_limits.values()), "invalid prospective counter inventory")
    counts = {key: 0 for key in limits}
    stages = {stage: {key: 0 for key in values} for stage, values in stage_limits.items()}
    previous = 0.
    digest = hashlib.sha256(b"[")
    _checked(isinstance(rows, list), "reservation history must be an ordered array")
    if prefix is not None and prefix[0] == 0:
        _checked(prefix[1] == _array_hash(digest), "readback rewrote the immutable pre-audit reservation prefix")
    for index, row in enumerate(rows):
        _checked(isinstance(row, dict)
                 and set(row) == {"index", "elapsed_seconds", "stage", "cleanup", "increments"}
                 and type(row["index"]) is int and row["index"] == index
                 and type(row["cleanup"]) is bool, "closed reservation schema/order changed")
        elapsed, stage, increments = row["elapsed_seconds"], row["stage"], row["increments"]
        _checked(type(elapsed) in (int, float) and math.isfinite(elapsed)
                 and previous <= elapsed < maximum - (0 if row["cleanup"] else reserve),
                 "closed reservation exceeded the original monotonic deadline")
        _checked(stage is None or stage in stages, "closed reservation used an undeclared stage")
        _checked(isinstance(increments, dict) and bool(increments), "empty or malformed closed reservation")
        for counter, amount in increments.items():
            _checked(counter in counts and type(amount) is int and amount > 0,
                     "closed reservation invented or refunded work")
            _checked(stage is not None or counter not in {"decisions", "requested_ticks", "episodes_started"},
                     "physical reservation lost its original stage")
            counts[counter] += amount
            _checked(counts[counter] <= limits[counter], "closed counter exceeded its prospective ceiling")
            if stage is not None and counter in stages[stage]:
                stages[stage][counter] += amount
                _checked(stages[stage][counter] <= stage_limits[stage][counter],
                         "closed stage exceeded its prospective ceiling")
        previous = elapsed
        # Strict JSON input already excludes foreign types and nonfinite
        # values. Hash the canonical array incrementally rather than copying
        # and serializing a potentially million-row list again.
        if index:
            digest.update(b",")
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode())
        if prefix is not None and index + 1 == prefix[0]:
            _checked(prefix[1] == _array_hash(digest), "readback rewrote the immutable pre-audit reservation prefix")
    _checked(snapshot.get("schema") == "nca-jolt-learn-budget/v1"
             and snapshot.get("closed") is closed
             and type(snapshot.get("reservation_count")) is int
             and snapshot["reservation_count"] == len(rows)
             and snapshot.get("reservations_sha256") == _array_hash(digest)
             and snapshot.get("refused_or_uncertain_work_refunded") is False
             and snapshot.get("maximum_wall_seconds") == maximum
             and snapshot.get("cleanup_reserve_seconds") == reserve,
             "closed snapshot changed its reservation identity or allocation")
    _checked(sha256_json(snapshot.get("counts")) == sha256_json(counts)
             and sha256_json(snapshot.get("stage_counts")) == sha256_json(stages),
             "closed snapshot totals differ from original reservations")
    elapsed = snapshot.get("wall_seconds")
    _checked(type(elapsed) in (int, float) and math.isfinite(elapsed)
             and previous <= elapsed < maximum, "closed snapshot exceeded the original wall deadline")
    return counts


def _verify_closed_costs(output, result, audit, *, execution_version=1):
    from nca.contracts import sha256_json
    from nca.evaluation.durable_journal import file_identity

    def read(name):
        return _read_json(output / name, limit=MAX_RESERVATION_BYTES
                          if name.endswith("budget-reservations.json") else MAX_JSON_BYTES)

    worker, config, declaration = (read(name) for name in ("worker-result.json", "config.json", "declaration.json"))
    prefix, full = (read(name) for name in ("budget-reservations.json", "closed-budget-reservations.json"))
    _checked(type(execution_version) is int and execution_version in (1, 2), "unknown execution version")
    _checked(worker.get("schema") == result.get("schema") == f"nca-learn-jolt-002-run/v{execution_version}"
             and audit.get("schema") == f"nca-learn-jolt-002-independent-readback/v{execution_version}",
             "unknown final learning receipt contract")
    if execution_version == 2:
        from nca.train.jolt_execution_contract import validate_v2_allocation_identity
        validate_v2_allocation_identity(declaration)
        _checked(declaration.get("schema") == "nca-learn-jolt-002-execution/v2"
                 and type(declaration.get("attempts")) is int and declaration["attempts"] == 1,
                 "v2 final receipt has another declaration version")
    _checked(prefix.get("schema") == full.get("schema") == "jolt-budget-reservations/v1",
             "unknown final reservation contract")
    before, after = prefix["reservations"], full["reservations"]
    _checked(isinstance(before, list) and isinstance(after, list) and len(before) <= len(after),
             "readback rewrote the immutable pre-audit reservation prefix")
    budget = config["budget"]
    _checked(sha256_json(declaration.get("budget")) == sha256_json(budget)
             and declaration.get("config_sha256") == sha256_json(config)
             and declaration.get("automatic_retry") is False
             and declaration.get("automatic_extension") is False,
             "final costs disagree with the frozen declaration")
    manifest = read("manifest.json")
    if execution_version == 2:
        _checked(manifest.get("schema") == "nca-learn-jolt-002-run/v2",
                 "v2 final receipt has another manifest version")
    _checked(sha256_json(worker.get("manifest")) == sha256_json(manifest)
             and manifest.get("config_sha256") == sha256_json(config)
             and manifest.get("declaration_sha256") == sha256_json(declaration)
             and manifest.get("source_sha256") == declaration.get("source_sha256"),
             "final allocation changed the original worker manifest")
    original = _cost_snapshot(budget, before, worker["costs_before_readback"], closed=False)
    final = _cost_snapshot(budget, after, result["costs"], closed=True,
                           prefix=(len(before), worker["costs_before_readback"]["reservations_sha256"]))
    _checked(result["costs"]["wall_seconds"] >= worker["costs_before_readback"]["wall_seconds"],
             "readback renewed the original invocation clock")
    readback_counters = {"audit_forward", "checkpoint_loads"}
    for row in after[len(before):]:
        _checked(row["cleanup"] is True and row["stage"] is None
                 and set(row["increments"]) <= readback_counters,
                 "post-audit prefix contains new ordinary work")
        _checked(row["elapsed_seconds"] >= worker["costs_before_readback"]["wall_seconds"],
                 "readback reservation predates its original worker snapshot")
    for counter in final:
        reported = audit.get(counter) if counter in readback_counters else 0
        _checked(type(reported) is int and reported >= 0 and final[counter] - original[counter] == reported,
                 "final readback cost differs from independently reported work")
    worker_sha, audit_sha = (file_identity(output / name)["sha256"] for name in
                             ("worker-result.json", "independent-readback.json"))
    _checked(result.get("worker_result_sha256") == worker_sha and result.get("audit_sha256") == audit_sha,
             "final result changed its original worker or audit binding")
    _checked(result.get("status") == worker.get("status")
             and worker["status"] in {"completed", "capped", "failed", "incomplete"}
             and sha256_json(result.get("error")) == sha256_json(worker.get("error"))
             and sha256_json(result.get("closure")) == sha256_json(worker.get("closure")),
             "final readback upgraded or changed the original execution outcome")
    _checked(all(type(audit.get(key)) is bool for key in ("verified", "complete", "qualified")),
             "audit verdicts must be explicit booleans")
    complete = worker["status"] == "completed" and audit["verified"] and audit["complete"]
    if audit["verified"]:
        _checked(audit.get("worker_result_sha256") == worker_sha
                 and audit.get("worker_status") == worker["status"]
                 and audit.get("config_sha256") == sha256_json(config)
                 and audit.get("source_sha256") == declaration.get("source_sha256")
                 and sha256_json(audit.get("costs_reconstructed")) == sha256_json(original),
                 "audit reviewed another worker outcome or cost allocation")
    _checked(not audit["complete"] or complete, "an incomplete worker or failed audit cannot acquire complete evidence")
    expected = audit.get("qualification") if complete else worker["qualification"]
    _checked(isinstance(expected, dict)
             and sha256_json(result.get("qualification")) == sha256_json(expected),
             "final qualification differs from its independently reviewed evidence")
    qualification = result["qualification"]
    _checked(type(qualification.get("qualified")) is bool
             and qualification.get("evidence_complete") is complete
             and (not qualification["qualified"] or complete)
             and (not audit["qualified"] or complete)
             and (not complete or audit["qualified"] is qualification["qualified"]),
             "incomplete readback cannot upgrade qualification")
    return {"closed": True, "reservation_count": len(after),
            "reservations_sha256": result["costs"]["reservations_sha256"], "counts": final,
            "readback_deltas": {key: final[key] - original[key] for key in sorted(readback_counters)}}


def verify_inventory(output, *, execution_version=1):
    from nca.evaluation.durable_journal import file_identity, validate_record_chain
    output = Path(output)
    value = _read_json(output / "inventory.json")
    _checked(value.get("schema") == "jolt-closed-inventory/v1" and isinstance(value.get("files"), dict),
             "unknown closed output inventory contract")
    files = value["files"]
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if actual != set(files) | {"inventory.json"}:
        raise ValueError("closed output inventory has missing or additional files")
    for relative, expected in files.items():
        path = output / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink()
                or not path.resolve().is_relative_to(output.resolve()) or file_identity(path) != expected):
            raise ValueError("closed output file changed")
    chain = validate_record_chain(output / "owner-journal.jsonl")
    result = _read_json(output / "result.json")
    audit = _read_json(output / "independent-readback.json")
    costs = _verify_closed_costs(output, result, audit, execution_version=execution_version)
    if execution_version == 2:
        durable = result["closure"]["durability"]
        _checked(durable.get("schema") == "nca-owner-journal-durability/v2"
                 and durable.get("closed_verified") is True
                 and durable.get("poisoned") is False
                 and durable.get("closed_identity") == chain
                 and durable.get("readback_records") == chain["records"],
                 "v2 inventory lacks independently verified original journal closure")
    verdict = {"inventory_verified": True, "files": len(files), "journal": chain,
            "readback_verified": audit["verified"], "readback_complete": audit["complete"], "status": result["status"],
            "qualified": result["qualification"]["qualified"], "costs_verified": True,
            "closed_costs": costs}
    if execution_version == 2:
        verdict["inventory_identity"] = file_identity(output / "inventory.json")
    return verdict


def verify_execution_v2(output, *, terminal_stdout, observed_exit_code):
    """Bind original lifecycle completion to this exact worker's evidence."""
    from nca.evaluation.durable_journal import file_identity
    from nca.train.jolt_executor import verify_lifecycle
    output = Path(output)
    lifecycle = output.with_name(output.name + "-lifecycle")
    result = verify_lifecycle(lifecycle, terminal_stdout=terminal_stdout,
        observed_exit_code=observed_exit_code,
        expected_declaration=file_identity(output / "declaration.json"))
    if not result["completed"]:
        return {**result, "scope": "bound_execution_and_worker_output", "qualified": False}
    inventory = verify_inventory(output, execution_version=2)
    manifest = _read_json(output / "manifest.json")
    _checked(result["worker_inventory"] == inventory,
             "executor independently verified a different worker inventory")
    _checked(manifest.get("schema") == "nca-learn-jolt-002-run/v2"
             and manifest.get("whole_process_started_monotonic") == result["whole_process_started_monotonic"],
             "executor and worker do not share the original invocation clock")
    return {**result, "scope": "bound_execution_and_worker_output", "qualified": inventory["qualified"]}


def freeze(args, *, execution_version=1):
    import numpy
    import torch
    from nca.adapters.godot import inspect_host
    from nca.contracts import sha256_json
    from nca.evaluation.durable_journal import file_identity, publish_json
    from nca.train.jolt_admission import validate_config
    from nca.train.jolt_screen import DECLARATION_SCHEMA, DECLARATION_SCHEMA_V2, source_inventory, runtime_identity
    if type(execution_version) is not int or execution_version not in (1, 2):
        raise ValueError("unknown execution version")
    if execution_version == 2:
        from nca.train.jolt_execution_contract import validate_v2_allocation_identity
        allocation = validate_v2_allocation_identity({
            "allocation_id": getattr(args, "allocation_id", None),
            "authorization_ref": getattr(args, "authorization_ref", None)})
    config = json.loads(args.config.read_text())
    validate_config(ROOT, config)
    inventory = source_inventory(ROOT)
    reviews, reviewers = [], set()
    for path in args.review:
        resolved = path.resolve()
        relative = resolved.relative_to(ROOT).as_posix()
        record = json.loads(resolved.read_text())
        if execution_version == 2:
            from nca.train.jolt_execution_contract import require_prospective_review
            require_prospective_review(record)
        if record.get("accepted") is not True or record.get("source_sha256") != sha256_json(inventory):
            raise ValueError("review does not accept the complete current source closure")
        reviewer = record.get("reviewer")
        if (not isinstance(reviewer, str) or not reviewer or reviewer in reviewers
                or relative in {review["path"] for review in reviews}
                or not relative.startswith("docs/evidence/")):
            raise ValueError("two distinct repository evidence reviews with independent identities are required")
        reviewers.add(reviewer)
        reviews.append({"path": relative, "sha256": file_identity(resolved)["sha256"]})
    if len(reviews) < 2:
        raise ValueError("two independent source/admission reviews are required")
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip():
        raise ValueError("publish the clean reviewed source before freezing execution")
    declaration = {"schema": DECLARATION_SCHEMA if execution_version == 1 else DECLARATION_SCHEMA_V2, "attempts": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "Owner continuation of issue #62; one prospective allocation after independent implementation and admission review.",
        "source_revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_files": inventory, "source_sha256": sha256_json(inventory),
        "config_sha256": sha256_json(config), "host_identity": inspect_host(args.checkout, args.godot),
        "runtime": runtime_identity(),
        "independent_reviews": reviews, "budget": config["budget"],
        "sealed_rule": "all six fixed final checkpoint hashes precede every sealed outcome, including authored and idle",
        "automatic_retry": False, "automatic_extension": False}
    if execution_version == 2:
        declaration.update(**allocation,
            authorization="Separately commissioned allocation recorded at authorization_ref; this declaration does not reopen v1.")
    publish_json(args.declaration, declaration)
    print(json.dumps({"source_revision": declaration["source_revision"],
                      "declaration_sha256": sha256_json(declaration), "physical_execution_started": False}))


def execute(args):
    from nca.evaluation.durable_journal import publish_json, file_identity
    # This sibling lifecycle directory records attempts that fail before a
    # worker can create its own evidence directory. Never reuse either path.
    lifecycle = args.output.with_name(args.output.name + "-lifecycle")
    if args.output.exists() or lifecycle.exists():
        raise ValueError("this allocation path has already been used; automatic retry is forbidden")
    lifecycle.mkdir(parents=True)
    declaration = json.loads(args.declaration.read_text())
    seconds = declaration["budget"]["maximum_wall_seconds"]
    if seconds != 1800:
        raise ValueError("execution requires the unchanged 1,800-second aggregate backstop")
    publish_json(lifecycle / "invocation.json", {"attempts": 1, "whole_process_started_monotonic": ENTRY_MONOTONIC,
        "maximum_wall_seconds": seconds, "declaration": file_identity(args.declaration),
        "physical_execution_started": False, "automatic_retry": False})
    # Retained robust Linux cleanup helpers keep host descendants inside this
    # invocation, including when an intermediate Python worker fails.
    from execute_jolt_diagnostic import prepare_process_tree, terminate_owned_tree, owned_processes
    owner_identity = prepare_process_tree()
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--config", str(args.config),
               "--declaration", str(args.declaration), "--checkout", str(args.checkout),
               "--godot", str(args.godot), "--output", str(args.output),
               "--whole-started", repr(ENTRY_MONOTONIC)]
    timeout, failure, verifier, cleanup_confirmed, process = False, None, None, True, None
    try:
        with (lifecycle / "stdout.log").open("xb") as stdout, (lifecycle / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            # Reserve thirty seconds for independent closed-inventory readback
            # and final receipts, all under the original invocation clock.
            worker_deadline = ENTRY_MONOTONIC + seconds - 30
            while process.poll() is None:
                if time.monotonic() >= worker_deadline:
                    timeout = True
                    cleanup_confirmed = terminate_owned_tree(process, owner_identity, grace_seconds=3)
                    break
                try:
                    process.wait(timeout=min(1., max(.01, worker_deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            if owned_processes(owner_identity):
                cleanup_confirmed = terminate_owned_tree(process, owner_identity, grace_seconds=3) and cleanup_confirmed
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
        if not timeout and process.returncode == 0:
            # External readback has its own process so the original allocation
            # remains bounded if decoding an artifact unexpectedly stalls.
            remaining = ENTRY_MONOTONIC + seconds - 5 - time.monotonic()
            checked = subprocess.run([sys.executable, str(Path(__file__).resolve()), "inventory", "--output", str(args.output)],
                                     capture_output=True, text=True, timeout=max(.01, remaining), check=True)
            verifier = json.loads(checked.stdout)
    except BaseException as exc:
        failure = {"type": type(exc).__name__}
    finally:
        # Cancellation, a failed log fsync, a malformed verifier or a child
        # scan error must all revoke the worker and every host descendant.
        try:
            cleanup_confirmed = terminate_owned_tree(process, owner_identity, grace_seconds=3) and cleanup_confirmed
        except BaseException as exc:
            cleanup_confirmed = False
            failure = failure or {"type": type(exc).__name__}
            # Keep a direct handle fallback if process-tree inspection itself
            # fails. This revokes the known worker without claiming that an
            # uninspectable descendant tree has been verified empty.
            if process is not None:
                try:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=3)
                except BaseException:
                    pass
    try:
        owned_remaining = len(owned_processes(owner_identity))
    except BaseException as exc:
        owned_remaining = None
        cleanup_confirmed = False
        failure = failure or {"type": type(exc).__name__}
    elapsed = time.monotonic() - ENTRY_MONOTONIC
    complete = bool(not timeout and failure is None and verifier and verifier["inventory_verified"]
                    and verifier["readback_verified"] and verifier["status"] == "completed"
                    and verifier.get("readback_complete") is True and verifier.get("costs_verified") is True
                    and cleanup_confirmed and owned_remaining == 0 and elapsed <= seconds)
    receipt = {"schema": "jolt-learning-executor/v1", "attempts": 1,
        "returncode": None if process is None else process.returncode,
        "timeout": timeout, "error": failure, "completion_candidate": complete, "inventory": verifier,
        "wall_seconds": elapsed, "maximum_wall_seconds": seconds,
        "whole_process_started_monotonic": ENTRY_MONOTONIC,
        "cleanup_confirmed": cleanup_confirmed, "owned_processes_remaining": owned_remaining,
        "streams": {name: file_identity(lifecycle / name) if (lifecycle / name).is_file() else None
                    for name in ("stdout.log", "stderr.log")}}
    publish_json(lifecycle / "executor.json", receipt)
    finalization = {"schema": "jolt-learning-executor-finalization/v1",
        "executor": file_identity(lifecycle / "executor.json"),
        "wall_seconds_after_executor_publication": time.monotonic() - ENTRY_MONOTONIC,
        "maximum_wall_seconds": seconds, "whole_process_started_monotonic": ENTRY_MONOTONIC}
    publish_json(lifecycle / "finalization.json", finalization)
    # This terminal verdict is measured after final receipt hashing, publication
    # and directory fsync. A late publication can never preserve completion.
    elapsed = time.monotonic() - ENTRY_MONOTONIC
    terminal = {"schema": "jolt-learning-executor-terminal/v1", "completed": complete and elapsed <= seconds,
        "wall_seconds": elapsed, "maximum_wall_seconds": seconds,
        "executor": finalization["executor"], "finalization": file_identity(lifecycle / "finalization.json"),
        "cleanup_confirmed": cleanup_confirmed, "owned_processes_remaining": owned_remaining}
    terminal["wall_seconds"] = time.monotonic() - ENTRY_MONOTONIC
    terminal["completed"] = terminal["completed"] and terminal["wall_seconds"] <= seconds
    print(json.dumps(terminal), flush=True)
    return 0 if terminal["completed"] and time.monotonic() - ENTRY_MONOTONIC <= seconds else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run", "worker", "inventory",
                                         "freeze-v2", "run-v2", "worker-v2", "inventory-v2", "lifecycle-v2"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/learning/jolt-learn-002-v1.json")
    parser.add_argument("--declaration", type=Path)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--godot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review", type=Path, action="append", default=[])
    parser.add_argument("--whole-started", type=float)
    parser.add_argument("--allocation-id")
    parser.add_argument("--authorization-ref")
    parser.add_argument("--terminal-stdout", type=Path)
    parser.add_argument("--observed-exit-code", type=int)
    args = parser.parse_args()
    execution_version = 2 if args.mode.endswith("-v2") else 1
    if args.mode == "lifecycle-v2":
        if args.output is None or args.terminal_stdout is None or args.observed_exit_code is None:
            parser.error("lifecycle-v2 requires output, original terminal stdout, and independently observed CLI exit")
        result = verify_execution_v2(args.output, terminal_stdout=args.terminal_stdout,
                                     observed_exit_code=args.observed_exit_code)
        print(json.dumps(result))
        return 0 if result["completed"] else 2
    if args.mode in ("inventory", "inventory-v2"):
        print(json.dumps(verify_inventory(args.output, execution_version=execution_version)))
        return 0
    if any(value is None for value in (args.declaration, args.checkout, args.godot)):
        parser.error("declaration, checkout and godot are required")
    if args.mode in ("freeze", "freeze-v2"):
        freeze(args, execution_version=execution_version)
        return 0
    if args.output is None:
        parser.error("output is required")
    if args.mode == "run":
        return execute(args)
    if args.mode == "run-v2":
        from nca.train.jolt_executor import execute as execute_v2
        return execute_v2(args, whole_started=ENTRY_MONOTONIC, script=Path(__file__).resolve())
    if args.whole_started is None or not 0 < args.whole_started <= ENTRY_MONOTONIC:
        parser.error("worker requires the original supervised invocation clock")
    from nca.train.jolt_screen import run
    result = run(root=ROOT, checkout=args.checkout, godot=args.godot, output=args.output,
                 config_path=args.config, declaration_path=args.declaration, whole_started=args.whole_started,
                 execution_version=execution_version)
    return 0 if result["status"] in ("completed", "capped", "failed") else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Original evidence carries stable failure codes. Public execution logs
        # must not leak absolute workspace paths or host diagnostic traces.
        print(json.dumps({"error": {"type": type(exc).__name__, "code": getattr(exc, "code", None)}}),
              file=sys.stderr, flush=True)
        raise SystemExit(2)
