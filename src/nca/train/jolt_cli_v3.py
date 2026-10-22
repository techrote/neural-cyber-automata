"""Versioned freeze and independent inventory/cost verification for Jolt 003.

The external inventory reader uses no neural operations and streams every cost
record. Its result binds immutable worker output to the original outer clock.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess

from nca.contracts import ContractError, sha256_json, strict_json_loads
from nca.evaluation.durable_journal import file_identity, publish_json
from nca.train.jolt_admission_v3 import (
    validate_config_v3, validate_execution_v3, validate_declaration_reviews,
    validate_resource_admission,
)
from nca.train.jolt_budget_v3 import reservation_records, SNAPSHOT_SCHEMA
from nca.train.jolt_execution_v3 import (
    DECLARATION_SCHEMA, RUN_SCHEMA, PHASE_OFFSETS, WORKER_PHASES, validate_budget_v3,
    validate_allocation_identity_v3,
)

AUDIT_SCHEMA = "nca-learn-jolt-003-independent-readback/v3"
MAX_JSON_BYTES = 64 * 1024 * 1024


def _need(condition, reason):
    if not condition:
        raise ContractError("jolt_v3_inventory", reason)


def _read(path):
    path = Path(path)
    _need(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_JSON_BYTES,
          "a bounded original regular JSON file is required")
    return strict_json_loads(path.read_bytes(), max_bytes=MAX_JSON_BYTES)


def _array_digest(digest):
    value = digest.copy()
    value.update(b"]")
    return value.hexdigest()


def _same(left, right, reason):
    _need(sha256_json(left) == sha256_json(right), reason)


def _phase_history(snapshot):
    history, phase = snapshot.get("phase_history"), snapshot.get("phase")
    _need(phase in WORKER_PHASES and isinstance(history, list)
          and len(history) == WORKER_PHASES.index(phase) + 1,
          "snapshot lost its complete original phase history")
    previous = 0.
    for name, entry in zip(WORKER_PHASES, history):
        _need(isinstance(entry, dict) and set(entry) == {"phase", "elapsed_seconds"}
              and entry["phase"] == name, "phase order or shape changed")
        elapsed = entry["elapsed_seconds"]
        _need(type(elapsed) in (int, float) and math.isfinite(elapsed)
              and previous <= elapsed < PHASE_OFFSETS[name], "phase reset or exceeded the original clock")
        previous = elapsed
    return {entry["phase"]: entry["elapsed_seconds"] for entry in history}


def cost_snapshot_v3(budget, rows, snapshot, *, closed, prefix=None, audit=None):
    """Independent constant-memory counter replay, including exact audit suffix."""
    validate_budget_v3(budget)
    _need(snapshot.get("schema") == SNAPSHOT_SCHEMA and snapshot.get("closed") is closed
          and snapshot.get("maximum_wall_seconds") == 7200
          and snapshot.get("cleanup_reserve_seconds") == 1200
          and snapshot.get("refused_or_uncertain_work_refunded") is False,
          "versioned closed snapshot changed its allocation or closure")
    history = _phase_history(snapshot)
    if closed:
        _need(snapshot["phase"] == "publication", "closed costs omitted the full reserved tail")
    _need(type(snapshot.get("reservation_count")) is int and snapshot["reservation_count"] == len(rows),
          "snapshot reservation count differs from its artifact")
    counts = {key: 0 for key in budget["counters"]}
    stages = {stage: {key: 0 for key in limits} for stage, limits in budget["stages"].items()}
    suffix = {"audit_forward": 0, "checkpoint_loads": 0}
    digest = hashlib.sha256(b"[")
    previous, phase_index = 0., 0
    if prefix is not None:
        _need(type(prefix["reservation_count"]) is int and 0 <= prefix["reservation_count"] <= len(rows),
              "readback replaced an impossible worker prefix")
        _same(snapshot["phase_history"][:len(prefix["phase_history"])], prefix["phase_history"],
              "readback rewrote the worker phase history")
    for index, row in enumerate(rows):
        phase, elapsed, stage, increments = row["phase"], row["elapsed_seconds"], row["stage"], row["increments"]
        current = WORKER_PHASES.index(phase)
        _need(current >= phase_index and phase in history and history[phase] <= elapsed
              and previous <= elapsed < PHASE_OFFSETS[phase], "reservation renewed or exceeded its phase")
        _need(phase == "active" or row["cleanup"] is True, "ordinary work consumed the evidence tail")
        if phase == "closure":
            _need(set(increments) <= {"owner_requests", "owner_cancel", "owner_close", "host_terminations",
                       "uncertain_ticks", "episodes_aborted", "episodes_completed"},
                  "closure contains new physical or neural work")
        if phase == "audit":
            _need(stage is None and set(increments) <= set(suffix), "audit contains ordinary work")
        _need(phase != "publication", "publication contains new model or physical work")
        for counter, amount in increments.items():
            _need(counter in counts and type(amount) is int and amount > 0,
                  "reservation invented or refunded work")
            _need(stage is not None or counter not in {"decisions", "requested_ticks", "episodes_started"},
                  "physical reservation lost its original stage")
            counts[counter] += amount
            _need(counts[counter] <= budget["counters"][counter], "aggregate counter exceeded its ceiling")
            if stage is not None and counter in stages[stage]:
                stages[stage][counter] += amount
                _need(stages[stage][counter] <= budget["stages"][stage][counter], "stage exceeded its ceiling")
        if index:
            digest.update(b",")
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        if prefix is not None:
            before = prefix["reservation_count"]
            if index + 1 == before:
                _need(_array_digest(digest) == prefix["reservations_sha256"], "immutable worker prefix changed")
                _same(counts, prefix["counts"], "worker prefix counts changed")
                _same(stages, prefix["stage_counts"], "worker prefix stages changed")
            if index >= before:
                _need(phase == "audit" and row["cleanup"] and stage is None
                      and set(increments) <= set(suffix) and elapsed >= prefix["wall_seconds"],
                      "post-worker suffix performed undeclared work or reset its clock")
                for counter, amount in increments.items():
                    suffix[counter] += amount
        previous, phase_index = elapsed, current
    if prefix is not None and prefix["reservation_count"] == 0:
        _need(prefix["reservations_sha256"] == hashlib.sha256(b"[]").hexdigest(), "empty prefix changed")
    _need(snapshot["reservations_sha256"] == _array_digest(digest), "closed cost digest differs")
    _same(snapshot["counts"], counts, "closed counts differ from original reservations")
    _same(snapshot["stage_counts"], stages, "closed stages differ from original reservations")
    elapsed = snapshot.get("wall_seconds")
    _need(type(elapsed) in (int, float) and math.isfinite(elapsed) and previous <= elapsed
          and max(history.values()) <= elapsed < PHASE_OFFSETS[snapshot["phase"]],
          "snapshot exceeded the original phase")
    if audit is not None:
        _need(all(type(audit.get(key)) is int and audit[key] == value for key, value in suffix.items()),
              "charged readback costs differ from the independent auditor")
    return counts


def verify_closed_costs(output, result, audit):
    output = Path(output)
    worker, config, declaration, manifest = (_read(output / name) for name in
                          ("worker-result.json", "config.json", "declaration.json", "manifest.json"))
    _need(worker.get("schema") == result.get("schema") == manifest.get("schema") == RUN_SCHEMA
          and audit.get("schema") == AUDIT_SCHEMA and declaration.get("schema") == DECLARATION_SCHEMA,
          "cross-version execution/cost evidence")
    validate_config_v3(config)
    validate_allocation_identity_v3(declaration)
    validate_resource_admission(declaration)
    validate_declaration_reviews(declaration)
    _same(declaration.get("budget"), config["budget"], "declaration budget changed")
    _need(type(declaration.get("attempts")) is int and declaration["attempts"] == 1
          and declaration.get("allocation_id") == config["allocation_id"]
          and declaration.get("authorization_ref") == config["authorization_ref"]
          and declaration.get("config_sha256") == sha256_json(config)
          and declaration.get("automatic_retry") is False and declaration.get("automatic_extension") is False,
          "costs describe a different prospective configuration")
    _same(worker.get("manifest"), manifest, "worker manifest was replaced")
    _need(manifest.get("config_sha256") == sha256_json(config)
          and manifest.get("declaration_sha256") == sha256_json(declaration)
          and manifest.get("source_sha256") == declaration.get("source_sha256"), "manifest identity changed")
    before, after = (reservation_records(output / name) for name in
                    ("budget-reservations.json", "closed-budget-reservations.json"))
    original = cost_snapshot_v3(config["budget"], before, worker["costs_before_readback"], closed=False)
    final = cost_snapshot_v3(config["budget"], after, result["costs"], closed=True,
                            prefix=worker["costs_before_readback"], audit=audit)
    _need(result["costs"]["wall_seconds"] >= worker["costs_before_readback"]["wall_seconds"],
          "readback renewed the original invocation clock")
    for counter in final:
        expected = audit.get(counter) if counter in {"audit_forward", "checkpoint_loads"} else 0
        _need(type(expected) is int and expected >= 0 and final[counter] - original[counter] == expected,
              "independent audit delta differs from charged costs")
    worker_sha, audit_sha = (file_identity(output / name)["sha256"] for name in
                            ("worker-result.json", "independent-readback.json"))
    _need(result.get("worker_result_sha256") == worker_sha and result.get("audit_sha256") == audit_sha,
          "final result changed its original worker/audit identities")
    _need(worker.get("status") in {"completed", "failed", "capped", "incomplete"}
          and result.get("status") == worker["status"], "final evidence upgraded the worker status")
    _same(result.get("error"), worker.get("error"), "original worker failure changed")
    _same(result.get("closure"), worker.get("closure"), "original closure changed")
    _need(all(type(audit.get(key)) is bool for key in ("verified", "complete", "qualified")),
          "independent audit verdicts must be explicit")
    complete = worker["status"] == "completed" and audit["verified"] and audit["complete"]
    if audit["verified"]:
        _need(audit.get("worker_result_sha256") == worker_sha and audit.get("worker_status") == worker["status"]
              and audit.get("config_sha256") == sha256_json(config)
              and audit.get("source_sha256") == declaration.get("source_sha256"), "audit bound another run")
        _same(audit.get("costs_reconstructed"), original, "audit reconstructed different costs")
    _need(not audit["complete"] or complete, "failed/incomplete worker cannot acquire complete evidence")
    expected = audit.get("qualification") if complete else worker["qualification"]
    _same(result.get("qualification"), expected, "final qualification differs from complete independent evidence")
    qualification = result["qualification"]
    _need(type(qualification.get("qualified")) is bool and qualification.get("evidence_complete") is complete
          and (not qualification["qualified"] or complete) and (not audit["qualified"] or complete)
          and (not complete or audit["qualified"] is qualification["qualified"]),
          "incomplete evidence cannot qualify a learner")
    return {"schema": "nca-jolt-closed-cost-verification/v3", "closed": True,
            "reservation_count": len(after), "reservations_sha256": result["costs"]["reservations_sha256"],
            "counts": final, "readback_deltas": {key: final[key] - original[key]
                                                 for key in ("audit_forward", "checkpoint_loads")}}


def verify_inventory(output):
    output = Path(output)
    _need(stat.S_ISDIR(output.lstat().st_mode), "closed inventory requires an original directory")
    inventory_path = output / "inventory.json"
    inventory_info = inventory_path.lstat()
    _need(stat.S_ISREG(inventory_info.st_mode) and inventory_info.st_nlink == 1,
          "closed inventory requires an original regular manifest without links")
    inventory = _read(inventory_path)
    _need(isinstance(inventory, dict) and inventory.get("schema") == "jolt-closed-inventory/v1"
          and isinstance(inventory.get("files"), dict),
          "unknown closed inventory")
    files = inventory["files"]
    directories = set()
    for relative in files:
        _need(isinstance(relative, str) and relative and "\\" not in relative,
              "closed inventory requires canonical relative file names")
        name = PurePosixPath(relative)
        _need(not name.is_absolute() and name.as_posix() == relative and name.parts
              and ".." not in name.parts and relative != "inventory.json",
              "closed inventory requires canonical relative file names")
        directories.update(parent.as_posix() for parent in name.parents if parent != PurePosixPath("."))
    expected_files = set(files) | {"inventory.json"}
    actual = set()
    pending = [output]
    while pending:
        # Inspect metadata without opening special entries or following links.
        # Only ancestors of declared files may be directories: an unexpected
        # empty tree is evidence drift just as an extra ordinary file is.
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(output).as_posix()
                metadata = entry.stat(follow_symlinks=False)
                mode = metadata.st_mode
                _need(not stat.S_ISLNK(mode), "closed inventory cannot contain links")
                if stat.S_ISDIR(mode):
                    _need(relative in directories, "closed inventory has an unexpected directory")
                    pending.append(path)
                else:
                    _need(stat.S_ISREG(mode), "closed inventory contains a special or nonregular file")
                    _need(metadata.st_nlink == 1, "closed inventory cannot contain hard links")
                    _need(relative in expected_files, "closed inventory has missing or extra files")
                    actual.add(relative)
    _need(actual == expected_files, "closed inventory has missing or extra files")
    for relative, expected in files.items():
        path = output / relative
        _need(file_identity(path) == expected, "inventoried bytes changed")
    from nca.evaluation.jolt_journal_v3 import journal_identity_v3
    chain = journal_identity_v3(output / "owner-journal.jsonl")
    result, audit = _read(output / "result.json"), _read(output / "independent-readback.json")
    costs = verify_closed_costs(output, result, audit)
    durable = result["closure"]["durability"]
    from nca.train.jolt_storage_v3 import MAX_JOURNAL_RECORD_BYTES, MAX_JOURNAL_RECORDS, MAX_JOURNAL_BYTES
    _need(durable.get("schema") == "nca-owner-journal-durability/v3"
          and durable.get("maximum_record_bytes") == MAX_JOURNAL_RECORD_BYTES
          and durable.get("maximum_records") == MAX_JOURNAL_RECORDS
          and durable.get("maximum_stream_bytes") == MAX_JOURNAL_BYTES
          and durable.get("closed_verified") is True and durable.get("poisoned") is False
          and durable.get("closed_identity") == chain and durable.get("readback_records") == chain["records"],
          "original owner journal lacks complete independently verified closure")
    return {"inventory_verified": True, "files": len(files), "journal": chain,
            "readback_verified": audit["verified"], "readback_complete": audit["complete"],
            "status": result["status"], "qualified": result["qualification"]["qualified"],
            "costs_verified": True, "closed_costs": costs,
            "inventory_identity": file_identity(output / "inventory.json")}


def verify_execution(output, *, terminal_stdout, observed_exit_code, observed_exit_elapsed_seconds):
    from nca.train.jolt_executor_v3 import verify_lifecycle
    from nca.train.jolt_storage_v3 import file_identity_v3
    output = Path(output)
    _need(stat.S_ISDIR(output.lstat().st_mode), "execution requires an original output directory")
    # This binding is checked before the complete tree inventory. Refuse a
    # substituted FIFO/link here instead of opening it in the initial hash.
    declaration_identity = file_identity_v3(output / "declaration.json", maximum_bytes=1_048_576)
    result = verify_lifecycle(output.with_name(output.name + "-lifecycle"),
        terminal_stdout=terminal_stdout, observed_exit_code=observed_exit_code,
        observed_exit_elapsed_seconds=observed_exit_elapsed_seconds,
        expected_declaration=declaration_identity)
    if not result["completed"]:
        return {**result, "scope": "bound_execution_and_worker_output", "qualified": False}
    inventory = verify_inventory(output)
    _same(result["worker_inventory"], inventory, "executor verified another immutable worker inventory")
    manifest = _read(output / "manifest.json")
    _need(manifest.get("schema") == RUN_SCHEMA
          and manifest.get("whole_process_started_monotonic") == result["whole_process_started_monotonic"],
          "executor and worker clocks differ")
    return {**result, "scope": "bound_execution_and_worker_output", "qualified": inventory["qualified"]}


def build_prospective(args, root):
    from nca.adapters.godot import inspect_host
    from nca.train.jolt_screen import source_inventory, runtime_identity
    root = Path(root)
    config = _read(args.config)
    validate_config_v3(root, config)
    _need(not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip(),
          "publish the complete clean source before prospective review")
    _need(args.output is not None and args.output.name == config["allocation_id"]
          and not args.output.exists() and not args.output.with_name(args.output.name + "-lifecycle").exists(),
          "the allocation requires its unused bound output identity")
    _need(args.allocation_id == config["allocation_id"] and args.authorization_ref == config["authorization_ref"],
          "the actual commission must be explicitly selected")
    source = source_inventory(root)
    declaration = {"schema": DECLARATION_SCHEMA, "attempts": 1,
        "allocation_id": config["allocation_id"], "authorization_ref": config["authorization_ref"],
        "authorization": "Owner-issued LEARN-JOLT-003 prompt; one new allocation conditional on all final prospective gates.",
        "source_revision": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "source_files": source, "source_sha256": sha256_json(source), "config_sha256": sha256_json(config),
        "host_identity": inspect_host(args.checkout, args.godot), "runtime": runtime_identity(),
        "teacher_prerequisite": config["prerequisite"], "opened_inventory": config["opened_inventory"],
        "split": config["split"], "ppo_case_stream_sha256": sha256_json(config["ppo_case_stream"]),
        "comparison_order_sha256": sha256_json(config["comparison_order"]),
        "budget": config["budget"], "output_identity": {"basename": config["allocation_id"],
            "lifecycle_basename": config["allocation_id"] + "-lifecycle", "reuse_forbidden": True},
        "resource_admission": _read(args.resource_admission),
        "sealed_rule": "all six fixed final checkpoint hashes precede every sealed outcome, including authored and idle",
        "automatic_retry": False, "automatic_extension": False}
    validate_resource_admission(declaration, output_parent=args.output.parent)
    return declaration


def freeze(args, root):
    declaration = build_prospective(args, root)
    reviews = []
    for path in args.review:
        # A review can be published after the executable source commit. Keep
        # exact original bytes, with a stable repository-relative locator.
        parts = Path(path).parts
        _need("docs" in parts and "evidence" in parts, "source reviews must be public repository evidence")
        index = next((i for i in range(len(parts) - 1) if parts[i:i+2] == ("docs", "evidence")), None)
        _need(index is not None, "review has no public evidence locator")
        content = Path(path).read_text(encoding="utf-8")
        reviews.append({"path": Path(*parts[index:]).as_posix(), "content": content,
                        "file_sha256": hashlib.sha256(content.encode()).hexdigest()})
    declaration["independent_reviews"] = reviews
    declaration["created_utc"] = datetime.now(timezone.utc).isoformat()
    config = _read(args.config)
    validate_execution_v3(root, declaration, config=config, host_identity=declaration["host_identity"])
    publish_json(args.declaration, declaration)
    return {"source_revision": declaration["source_revision"], "declaration_sha256": sha256_json(declaration),
            "physical_execution_started": False}
