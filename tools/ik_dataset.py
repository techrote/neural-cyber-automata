#!/usr/bin/env python3
"""Declare, generate and inspect one bounded immutable geometric IK dataset.

Generation is a supervised CPU process. A durable registry binds both the
one-use generation allocation and the later final-suite opening. The registry
must be retained with the artifacts; it is an audit boundary, not encryption.
"""

from __future__ import annotations

import time
_STARTED = time.perf_counter()

import argparse
from collections import Counter
import hashlib
import math
import os
from pathlib import Path
import platform
import re
import sqlite3
import subprocess
import sys
import tempfile
import uuid

from nca.contracts import ContractError, canonical_json, sha256_json, strict_json_loads
from nca.environment import environment_report
from nca.ik.declaration import source_snapshot, _verify_committed_sources
from nca.ik.profiles import runner_right_sole_profile
from nca.runs import immutable_bytes, immutable_json, sha256_file, utc_now


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE = "configs/ik/runner_right_sole_acceptance_v1.json"
SCHEMA = "nca-ik-dataset-declaration/v1"
MANIFEST_SCHEMA = "nca-ik-dataset-manifest/v1"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_ROW_BYTES = 256 * 1024
PARTITIONS = ("training", "tuning", "final")
DECLARATION_FIELDS = {"schema", "run_id", "created_utc", "registry_id", "source", "environment",
                      "environment_sha256", "config_path", "config", "config_sha256", "split_plan",
                      "split_plan_sha256", "schedule", "profile_sha256", "host_revision",
                      "acceptance_file_sha256", "acceptance", "attempts", "retry_or_resume",
                      "generation_is_not_candidate_evaluation", "sha256"}
ARTIFACT_NAMES = {"declaration.json", "attempts.jsonl", "pending.jsonl", "normalization.json", "coverage.json"}
MANIFEST_FIELDS = {"schema", "run_id", "status", "declaration_sha256", "registry_id", "config_sha256",
                   "split_plan_sha256", "schedule", "shards", "artifacts", "final_suite_identity_sha256",
                   "final_state", "all_samples_sha256", "counts", "generation_cost", "failures",
                   "completed_utc", "sha256"}
SUPERVISOR_CHECKS = {"child_success", "journal_complete", "all_tasks_retained", "manifest_complete",
                     "declaration_binding", "counts_match_journal", "whole_process_within_budget"}


class DatasetToolError(ValueError):
    """Path-free, public-safe failure code."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _api():
    from nca.ik import data
    return data


def _seal(value):
    return {**value, "sha256": sha256_json(value)}


def _read(path, *, sealed=False):
    try:
        with Path(path).open("rb") as stream:
            value = strict_json_loads(stream.read(MAX_JSON_BYTES + 1), max_bytes=MAX_JSON_BYTES)
    except (OSError, ValueError) as exc:
        raise DatasetToolError("unreadable_or_invalid_json") from exc
    if not isinstance(value, dict):
        raise DatasetToolError("object_required")
    if sealed and value.get("sha256") != sha256_json({k: v for k, v in value.items() if k != "sha256"}):
        raise DatasetToolError("content_digest_mismatch")
    return value


def _outside(path):
    result = Path(path).resolve()
    if result == ROOT or ROOT in result.parents:
        raise DatasetToolError("artifact_destination_must_be_outside_checkout")
    return result


def _portable_name(value):
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value):
        raise DatasetToolError("portable_identifier_required")
    return value


def _digest(value):
    if type(value) is not str or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise DatasetToolError("sha256_required")
    return value


def _registry(path, *, create=False):
    root = _outside(path)
    marker = root / "registry.json"
    if create:
        root.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            try:
                immutable_json(marker, {"schema": "nca-ik-dataset-registry/v1", "id": uuid.uuid4().hex})
            except ValueError:
                pass  # A concurrent creator is read and checked below.
    value = _read(marker)
    if set(value) != {"schema", "id"} or value["schema"] != "nca-ik-dataset-registry/v1":
        raise DatasetToolError("registry_schema_mismatch")
    if not re.fullmatch(r"[a-f0-9]{32}", value["id"]):
        raise DatasetToolError("registry_identity_invalid")
    return root, value["id"]


def _source(config_path):
    try:
        relative = Path(config_path).resolve().relative_to(ROOT).as_posix()
        snapshot = source_snapshot(ROOT)
        files = dict(snapshot["files"])
        for name in (relative, ACCEPTANCE, "tools/ik_dataset.py", "tools/ik_dataset_supervise.py"):
            files[name] = sha256_file(ROOT / name)
        _verify_committed_sources(ROOT, files)
        tree = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
        return {"revision": snapshot["revision"], "tree": tree, "files": files,
                "source_sha256": sha256_json(files)}, relative
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise DatasetToolError("source_must_be_clean_committed_and_matching") from exc


def _environment():
    report = environment_report()
    report.update(device="cpu", arithmetic="Python binary64; one execution thread",
                  os_release=platform.release(), logical_cpu_count=os.cpu_count())
    return report


def _acceptance(config):
    return {"independent_endpoint_tolerance_cells": 1e-8, "teacher_solved_residual_max_cells": 1e-6,
            "all_scheduled_rows_retained": True, "failed_labels_are_not_pose_supervision": True,
            "train_only_normalization": True, "final_minima_required": config.mode == "prospective"}


def _schedule(config):
    """Replay without teachers; spill identity sorting to disk with a 1 MiB cache.

    Even the largest admissible schedule cannot grow an in-memory set of a
    million IDs or final-input digests. SQLite is temporary scratch, never an
    artifact or an allocation receipt.
    """
    partitions = Counter()
    ids_hash, values_hash = hashlib.sha256(b"["), hashlib.sha256(b"[")
    count = 0
    final_hash = hashlib.sha256(b"[")
    with tempfile.TemporaryDirectory(prefix="ik-schedule-") as directory:
        connection = sqlite3.connect(str(Path(directory) / "identities.sqlite"))
        try:
            connection.execute("PRAGMA cache_size=-1024")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, final_input BLOB)")
            for task in _api().iter_tasks(config):
                if count >= config.expected_tasks or task.partition not in PARTITIONS:
                    raise DatasetToolError("declared_schedule_count_or_identity_mismatch")
                final_input = bytes.fromhex(task.input_sha256) if task.partition == "final" else None
                try:
                    connection.execute("INSERT INTO tasks VALUES (?, ?)", (task.id, final_input))
                except sqlite3.IntegrityError as exc:
                    raise DatasetToolError("declared_schedule_count_or_identity_mismatch") from exc
                partitions[task.partition] += 1
                if count:
                    ids_hash.update(b",")
                    values_hash.update(b",")
                ids_hash.update(canonical_json(task.id).encode())
                values_hash.update(canonical_json(task.sha256).encode())
                count += 1
            for index, (digest,) in enumerate(connection.execute(
                    "SELECT final_input FROM tasks WHERE final_input IS NOT NULL ORDER BY final_input")):
                if index:
                    final_hash.update(b",")
                final_hash.update(canonical_json(digest.hex()).encode())
        finally:
            connection.close()
    ids_hash.update(b"]")
    values_hash.update(b"]")
    final_hash.update(b"]")
    if count != config.expected_tasks:
        raise DatasetToolError("declared_schedule_count_or_identity_mismatch")
    return {"tasks": count, "partitions": dict(partitions),
            "task_ids_sha256": ids_hash.hexdigest(), "task_values_sha256": values_hash.hexdigest(),
            "final_suite_identity_sha256": final_hash.hexdigest()}


def declare(config_path, output, registry_path):
    output = _outside(output)
    if output.exists():
        raise DatasetToolError("immutable_declaration_exists")
    config = _api().DatasetConfig.from_dict(_read(config_path))
    source, relative = _source(config_path)
    _, registry_id = _registry(registry_path, create=True)
    environment, profile = _environment(), runner_right_sole_profile()
    plan = _api().build_split_plan(config)
    value = _seal({"schema": SCHEMA, "run_id": "ik-dataset-" + uuid.uuid4().hex,
                   "created_utc": utc_now(), "registry_id": registry_id,
                   "source": source, "environment": environment,
                   "environment_sha256": sha256_json(environment), "config_path": relative,
                   "config": config.to_dict(), "config_sha256": config.sha256,
                   "split_plan": plan, "split_plan_sha256": sha256_json(plan),
                   "schedule": _schedule(config), "profile_sha256": profile.sha256,
                   "host_revision": profile.source.commit,
                   "acceptance_file_sha256": source["files"][ACCEPTANCE],
                   "acceptance": _acceptance(config),
                   "attempts": 1, "retry_or_resume": False,
                   "generation_is_not_candidate_evaluation": True})
    immutable_json(output, value)
    return value


def _validate_declaration(value):
    """Validate the portable declaration without requiring its original checkout."""
    if (set(value) != DECLARATION_FIELDS or value.get("schema") != SCHEMA
            or type(value.get("attempts")) is not int or value.get("attempts") != 1
            or value.get("retry_or_resume") is not False
            or value.get("generation_is_not_candidate_evaluation") is not True):
        raise DatasetToolError("declaration_schema_mismatch")
    _portable_name(value["run_id"])
    config = _api().DatasetConfig.from_dict(value["config"])
    if value["config_sha256"] != config.sha256:
        raise DatasetToolError("configuration_digest_mismatch")
    source = value["source"]
    relative = value["config_path"]
    if (not isinstance(source, dict) or set(source) != {"revision", "tree", "files", "source_sha256"}
            or type(relative) is not str or not relative.startswith("configs/")
            or ".." in Path(relative).parts or Path(relative).is_absolute()
            or not isinstance(source["files"], dict)
            or not {relative, ACCEPTANCE, "tools/ik_dataset.py", "tools/ik_dataset_supervise.py",
                    "src/nca/ik/data.py"}.issubset(source["files"])
            or source["source_sha256"] != sha256_json(source["files"])
            or not re.fullmatch(r"[a-f0-9]{40}", str(source["revision"]))
            or not re.fullmatch(r"[a-f0-9]{40}", str(source["tree"]))
            or not re.fullmatch(r"[a-f0-9]{32}", str(value["registry_id"]))):
        raise DatasetToolError("declaration_source_identity_invalid")
    for name, digest in source["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise DatasetToolError("declaration_source_identity_invalid")
        _digest(digest)
    profile = runner_right_sole_profile()
    if (value["profile_sha256"] != profile.sha256 or value["host_revision"] != profile.source.commit
            or value["acceptance"] != _acceptance(config)
            or value["environment_sha256"] != sha256_json(value["environment"])
            or value["split_plan_sha256"] != sha256_json(value["split_plan"])
            or value["acceptance_file_sha256"] != value["source"]["files"].get(ACCEPTANCE)
            or value["acceptance_file_sha256"] != sha256_file(ROOT / ACCEPTANCE)):
        raise DatasetToolError("declaration_identity_or_acceptance_mismatch")
    if (value["split_plan_sha256"] != sha256_json(_api().build_split_plan(config))
            or value["schedule"] != _schedule(config)):
        raise DatasetToolError("split_or_schedule_mismatch")
    return config


def _declaration(path, registry_path, *, check_source=True):
    value = _read(path, sealed=True)
    config = _validate_declaration(value)
    registry, registry_id = _registry(registry_path)
    if value["registry_id"] != registry_id:
        raise DatasetToolError("registry_identity_mismatch")
    if check_source:
        source, _ = _source(ROOT / value["config_path"])
        committed_config = _api().DatasetConfig.from_dict(_read(ROOT / value["config_path"]))
        if (source != value["source"] or _environment() != value["environment"]
                or committed_config.sha256 != config.sha256):
            raise DatasetToolError("source_or_environment_changed")
    return value, config, registry


def _worker_admission(path):
    """Read only the bounded configuration before imposing the worker ceiling."""
    value = _read(path, sealed=True)
    config = _api().DatasetConfig.from_dict(value.get("config"))
    return _memory_limit(config.budget.max_memory_mib)


def _memory_limit(mebibytes):
    """Hard per-worker memory admission, with no silent unsupported-platform pass."""
    ceiling = int(mebibytes * 1024 * 1024)
    if os.name == "posix":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
        return "RLIMIT_AS"
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class BasicLimit(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IOCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount",
                        "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimit), ("IoInfo", IOCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        job = kernel.CreateJobObjectW(None, None)
        limits = ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
        limits.ProcessMemoryLimit = ceiling
        if not job or not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise DatasetToolError("hard_memory_limit_unavailable")
        if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
            raise DatasetToolError("hard_memory_limit_unavailable")
        return "Windows Job Object process committed-memory limit"
    raise DatasetToolError("hard_memory_limit_unavailable")


def _peak_memory_mib():
    if os.name == "posix":
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024 * 1024 if sys.platform == "darwin" else 1024)
    # The Windows hard cap is enforced above; its working-set measurement is
    # deliberately left unmeasured rather than pretending it is the cap.
    return None


def _journal_write(stream, value):
    stream.write((canonical_json(value) + "\n").encode())
    stream.flush()
    os.fsync(stream.fileno())


def _artifact(path, root):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def _worker(declaration_path, output, registry_path):
    memory_mode = _worker_admission(declaration_path)
    declaration, config, registry = _declaration(declaration_path, registry_path)
    key = declaration["sha256"]
    claim = _read(registry / (key + ".generation-started.json"))
    if claim.get("declaration_sha256") != key or claim.get("run_id") != declaration["run_id"]:
        raise DatasetToolError("supervisor_claim_required")
    immutable_json(registry / (key + ".worker-started.json"),
                   {"declaration_sha256": key, "run_id": declaration["run_id"], "started_utc": utc_now()})
    output = _outside(output)
    output.mkdir(parents=True, exist_ok=False)
    immutable_json(output / "declaration.json", declaration)
    budget = config.budget
    samples, shards, buffers, counts = [], [], {p: [] for p in PARTITIONS}, Counter()
    failures, attempts, completed, actual_iterations, reserved_iterations = [], 0, 0, 0, 0

    def flush(partition):
        rows = buffers[partition]
        if not rows:
            return
        number = counts[partition]
        name = f"{partition}-{number:05d}.jsonl"
        raw = b"".join((canonical_json(sample.to_dict()) + "\n").encode() for sample in rows)
        immutable_bytes(output / name, raw)
        shards.append({**_artifact(output / name, output), "partition": partition, "rows": len(rows),
                       "sample_ids_sha256": sha256_json([sample.task.id for sample in rows])})
        counts[partition] += 1
        rows.clear()

    try:
        with (output / "attempts.jsonl").open("xb") as journal, (output / "pending.jsonl").open("xb") as pending:
            _journal_write(journal, {"event": "allocation_started", "declaration_sha256": key,
                                     "run_id": declaration["run_id"], "started_utc": utc_now()})
            for task in _api().iter_tasks(config):
                if time.perf_counter() - _STARTED >= budget.max_wall_seconds - budget.publication_reserve_seconds:
                    raise DatasetToolError("worker_wall_budget_exhausted")
                if attempts + 1 > min(budget.max_unique_tasks, budget.max_teacher_attempts):
                    raise DatasetToolError("task_or_attempt_budget_exhausted")
                if reserved_iterations + config.teacher.max_iterations > budget.max_teacher_iterations:
                    raise DatasetToolError("teacher_iteration_reservation_exhausted")
                attempts += 1
                reserved_iterations += config.teacher.max_iterations
                _journal_write(journal, {"event": "teacher_attempt_started", "index": attempts - 1,
                                         "sample_id": task.id, "task_sha256": task.sha256,
                                         "reserved_iterations": config.teacher.max_iterations})
                sample = _api().generate_sample(task, config)
                # Persist the exact result before acknowledging completion. A
                # killed worker leaves this complete or truncated raw prefix.
                raw = (canonical_json(sample.to_dict()) + "\n").encode()
                if len(raw) > MAX_ROW_BYTES:
                    raise DatasetToolError("sample_exceeds_bounded_row_size")
                pending.write(raw)
                pending.flush()
                os.fsync(pending.fileno())
                iterations = sample.label.teacher.iterations
                if not 0 <= iterations <= config.teacher.max_iterations:
                    raise DatasetToolError("teacher_iteration_accounting_invalid")
                completed += 1
                actual_iterations += iterations
                _journal_write(journal, {"event": "teacher_attempt_finished", "index": attempts - 1,
                                         "sample_id": task.id, "sample_sha256": sample.sha256,
                                         "iterations": iterations, "teacher_status": sample.label.teacher.status})
                samples.append(sample)
                buffers[task.partition].append(sample)
                if len(buffers[task.partition]) >= config.batch_size:
                    flush(task.partition)
            for partition in PARTITIONS:
                flush(partition)
        _api().validate_dataset(samples, config, require_complete=True)
        normalization = _api().fit_normalization([s for s in samples if s.task.partition == "training"], config)
        immutable_json(output / "normalization.json", normalization.to_dict())
        coverage = _api().coverage_report(samples, config)
        immutable_json(output / "coverage.json", coverage)
        status = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        failures.append(exc.code if isinstance(exc, (DatasetToolError, ContractError)) else type(exc).__name__)
        for partition in PARTITIONS:
            try:
                flush(partition)
            except (OSError, ValueError, MemoryError):
                failures.append("partial_shard_publication_failed")
        status = "incomplete"
    peak = _peak_memory_mib()
    if peak is not None and peak > budget.max_memory_mib:
        failures.append("measured_peak_memory_exceeded")
        status = "incomplete"
    elapsed = time.perf_counter() - _STARTED
    if elapsed >= budget.max_wall_seconds:
        failures.append("worker_wall_budget_exceeded")
        status = "incomplete"
    artifact_names = ("declaration.json", "attempts.jsonl", "pending.jsonl", "normalization.json", "coverage.json")
    artifacts = [_artifact(output / name, output) for name in artifact_names if (output / name).is_file()]
    manifest = _seal({"schema": MANIFEST_SCHEMA, "run_id": declaration["run_id"], "status": status,
                      "declaration_sha256": key, "registry_id": declaration["registry_id"],
                      "config_sha256": config.sha256, "split_plan_sha256": declaration["split_plan_sha256"],
                      "schedule": declaration["schedule"], "shards": shards, "artifacts": artifacts,
                      "final_suite_identity_sha256": declaration["schedule"]["final_suite_identity_sha256"],
                      "final_state": "sealed" if config.mode == "prospective" else "development_fixture",
                      "all_samples_sha256": sha256_json([s.sha256 for s in samples]),
                      "counts": {"attempted": attempts, "completed": completed, "unknown_attempts": attempts - completed,
                                 "teacher_iterations_actual_lower_bound": actual_iterations,
                                 "teacher_iterations_conservative": actual_iterations + (attempts - completed) * config.teacher.max_iterations,
                                 "teacher_iterations_reserved": reserved_iterations},
                      "generation_cost": {"worker_module_seconds": elapsed, "peak_rss_mib": peak,
                                          "hard_memory_limit_mib": budget.max_memory_mib, "memory_limit_method": memory_mode,
                                          "whole_process_receipt": "supervisor.json", "device": "cpu"},
                      "failures": failures, "completed_utc": utc_now()})
    immutable_json(output / "manifest.json", manifest)
    return manifest


def _journal_accounting(path, maximum, declaration=None):
    starts, finished, iterations, error = 0, 0, 0, None
    pending = None
    if not path.is_file():
        return {"valid_prefix": False, "error": "missing_journal", "started": 0, "finished": 0,
                "unknown_attempts": 0, "unknown_additional_work": True}
    with path.open("rb") as stream:
        first = stream.readline(MAX_ROW_BYTES + 1)
        try:
            opening = strict_json_loads(first)
            if not first.endswith(b"\n") or not isinstance(opening, dict) or opening.get("event") != "allocation_started":
                raise ValueError()
            if declaration is not None and any(opening.get(key) != declaration[key]
                                               for key in ("run_id", "declaration_sha256")):
                raise ValueError()
        except ValueError:
            error = "invalid_start"
        while error is None:
            raw = stream.readline(MAX_ROW_BYTES + 1)
            if not raw:
                break
            try:
                if not raw.endswith(b"\n"):
                    raise ValueError()
                row = strict_json_loads(raw)
                if row["event"] == "teacher_attempt_started" and pending is None and row["index"] == starts:
                    if row["reserved_iterations"] != maximum:
                        raise ValueError()
                    pending = row["sample_id"]
                    starts += 1
                elif row["event"] == "teacher_attempt_finished" and row["sample_id"] == pending and row["index"] == finished:
                    if type(row["iterations"]) is not int or not 0 <= row["iterations"] <= maximum:
                        raise ValueError()
                    iterations += row["iterations"]
                    pending = None
                    finished += 1
                else:
                    raise ValueError()
            except (ValueError, KeyError, TypeError):
                error = "invalid_or_truncated_tail"
    return {"valid_prefix": error is None, "error": error, "started": starts, "finished": finished,
            "unknown_attempts": starts - finished, "unknown_additional_work": error is not None,
            "teacher_iterations_actual_lower_bound": iterations,
            "teacher_iterations_conservative": iterations + (starts - finished) * maximum}


def generate(declaration_path, output, registry_path):
    declaration, config, registry = _declaration(declaration_path, registry_path)
    output = _outside(output)
    if output.exists():
        raise DatasetToolError("immutable_output_exists")
    key = declaration["sha256"]
    claim = {"declaration_sha256": key, "run_id": declaration["run_id"], "started_utc": utc_now(),
             "attempt": 1, "retry_or_resume": False}
    immutable_json(registry / (key + ".generation-started.json"), claim)
    budget = config.budget
    child, returncode, timed_out, interrupted = None, None, False, False
    execution_failure = None
    child_start = time.perf_counter()
    try:
        allowance = budget.max_wall_seconds - budget.publication_reserve_seconds - (time.perf_counter() - _STARTED)
        if allowance <= 0:
            raise DatasetToolError("supervisor_admission_budget_exhausted")
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_worker",
                                  "--declaration", str(Path(declaration_path).resolve()),
                                  "--output", str(output), "--registry", str(registry)],
                                 cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            returncode = child.wait(timeout=allowance)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            returncode = child.wait(timeout=5)
        except KeyboardInterrupt:
            interrupted = True
            child.kill()
            returncode = child.wait(timeout=5)
    except Exception as exc:
        execution_failure = exc.code if isinstance(exc, DatasetToolError) else type(exc).__name__
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    child_seconds = time.perf_counter() - child_start
    output.mkdir(parents=True, exist_ok=True)
    accounting = _journal_accounting(output / "attempts.jsonl", config.teacher.max_iterations,
                                    {"run_id": declaration["run_id"], "declaration_sha256": key})
    checks = {"child_success": returncode == 0 and not timed_out and not interrupted,
              "journal_complete": accounting["valid_prefix"] and accounting["unknown_attempts"] == 0,
              "all_tasks_retained": accounting["finished"] == config.expected_tasks}
    try:
        manifest = verify_artifact(output, require_supervisor=False)
        checks["manifest_complete"] = manifest["status"] == "complete"
        checks["declaration_binding"] = manifest["declaration_sha256"] == key
        checks["counts_match_journal"] = (manifest["counts"]["attempted"] == accounting["started"]
            and manifest["counts"]["completed"] == accounting["finished"]
            and manifest["counts"]["teacher_iterations_actual_lower_bound"] == accounting["teacher_iterations_actual_lower_bound"])
        manifest_sha256 = sha256_file(output / "manifest.json")
    except (Exception, KeyboardInterrupt) as exc:
        checks["manifest_complete"] = False
        manifest_sha256 = None
        execution_failure = exc.code if isinstance(exc, (DatasetToolError, ContractError)) else type(exc).__name__
        if isinstance(exc, KeyboardInterrupt):
            interrupted = True
            checks["child_success"] = False
    elapsed = time.perf_counter() - _STARTED
    checks["whole_process_within_budget"] = elapsed < budget.max_wall_seconds
    receipt = _seal({"schema": "nca-ik-dataset-supervisor/v1", "declaration_sha256": key,
                     "status": "complete" if all(checks.values()) else "incomplete", "checks": checks,
                     "child_returncode": returncode, "child_timed_out": timed_out, "interrupted": interrupted,
                     "execution_failure": execution_failure,
                     "child_whole_process_seconds": child_seconds, "aggregate_through_readback_seconds": elapsed,
                     "max_wall_seconds": budget.max_wall_seconds,
                     "final_receipt_write_excluded_from_elapsed": True, "journal": accounting,
                     "manifest_file_sha256": manifest_sha256, "finished_utc": utc_now()})
    immutable_json(output / "supervisor.json", receipt)
    if time.perf_counter() - _STARTED >= budget.max_wall_seconds:
        immutable_json(output / "supervisor-overrun.json", {"declaration_sha256": key, "status": "incomplete"})
        raise DatasetToolError("supervisor_publication_budget_exhausted")
    return receipt


def _jsonl_rows(path, maximum):
    """Read a finite canonical prefix, refusing trailing data and truncation."""
    count = 0
    with path.open("rb") as stream:
        while True:
            raw = stream.readline(MAX_ROW_BYTES + 1)
            if not raw:
                return
            if count >= maximum or len(raw) > MAX_ROW_BYTES or not raw.endswith(b"\n"):
                raise DatasetToolError("invalid_or_oversized_sample_line")
            value = strict_json_loads(raw, max_bytes=MAX_ROW_BYTES)
            if not isinstance(value, dict) or raw != (canonical_json(value) + "\n").encode():
                raise DatasetToolError("noncanonical_artifact_row")
            count += 1
            yield value


def _verified_journal(path, samples, declaration, config):
    rows = iter(_jsonl_rows(path, 2 * config.expected_tasks + 1))
    first = next(rows, None)
    if (not isinstance(first, dict)
            or set(first) != {"event", "declaration_sha256", "run_id", "started_utc"}
            or first["event"] != "allocation_started" or first["run_id"] != declaration["run_id"]
            or first["declaration_sha256"] != declaration["sha256"]
            or type(first["started_utc"]) is not str or not first["started_utc"]):
        raise DatasetToolError("journal_allocation_binding_mismatch")
    iterations = 0
    for index, sample in enumerate(samples):
        expected_start = {"event": "teacher_attempt_started", "index": index, "sample_id": sample.task.id,
                          "task_sha256": sample.task.sha256, "reserved_iterations": config.teacher.max_iterations}
        expected_finish = {"event": "teacher_attempt_finished", "index": index, "sample_id": sample.task.id,
                           "sample_sha256": sample.sha256, "iterations": sample.label.teacher.iterations,
                           "teacher_status": sample.label.teacher.status}
        # Canonical comparison also refuses booleans masquerading as integers.
        if (canonical_json(next(rows, None)) != canonical_json(expected_start)
                or canonical_json(next(rows, None)) != canonical_json(expected_finish)):
            raise DatasetToolError("journal_sample_or_reservation_mismatch")
        iterations += sample.label.teacher.iterations
    if next(rows, None) is not None:
        raise DatasetToolError("journal_has_unacknowledged_work")
    return {"valid_prefix": True, "error": None, "started": len(samples), "finished": len(samples),
            "unknown_attempts": 0, "unknown_additional_work": False,
            "teacher_iterations_actual_lower_bound": iterations, "teacher_iterations_conservative": iterations}


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _verify_supervisor(root, manifest, declaration, config, accounting):
    supervisor = _read(root / "supervisor.json", sealed=True)
    fields = {"schema", "declaration_sha256", "status", "checks", "child_returncode", "child_timed_out",
              "interrupted", "execution_failure", "child_whole_process_seconds", "aggregate_through_readback_seconds",
              "max_wall_seconds", "final_receipt_write_excluded_from_elapsed", "journal", "manifest_file_sha256",
              "finished_utc", "sha256"}
    checks = supervisor.get("checks")
    child = supervisor.get("child_whole_process_seconds")
    elapsed = supervisor.get("aggregate_through_readback_seconds")
    if (set(supervisor) != fields or supervisor.get("schema") != "nca-ik-dataset-supervisor/v1"
            or supervisor.get("status") != "complete" or supervisor.get("declaration_sha256") != declaration["sha256"]
            or not isinstance(checks, dict) or set(checks) != SUPERVISOR_CHECKS
            or any(value is not True for value in checks.values())
            or type(supervisor.get("child_returncode")) is not int or supervisor.get("child_returncode") != 0
            or supervisor.get("child_timed_out") is not False or supervisor.get("interrupted") is not False
            or supervisor.get("execution_failure") is not None
            or supervisor.get("final_receipt_write_excluded_from_elapsed") is not True
            or not _finite_nonnegative(child) or not _finite_nonnegative(elapsed)
            or not manifest["generation_cost"]["worker_module_seconds"] <= child <= elapsed < config.budget.max_wall_seconds
            or supervisor.get("max_wall_seconds") != config.budget.max_wall_seconds
            or canonical_json(supervisor.get("journal")) != canonical_json(accounting)
            or supervisor.get("manifest_file_sha256") != sha256_file(root / "manifest.json")):
        raise DatasetToolError("generation_not_complete")


def verify_artifact(output, *, require_supervisor=True):
    """Check retained results and the complete allocation, without calling a teacher.

    Byte digests alone are insufficient: a rehashed omission, duplicated row,
    partition swap, or incomplete journal must still fail. The original config
    drives exact task replay, shard boundaries, normalization, and coverage.
    """
    root = Path(output)
    if (root / "supervisor-overrun.json").exists():
        raise DatasetToolError("generation_not_complete")
    for name in ("manifest.json", "declaration.json"):
        if (root / name).is_symlink() or not (root / name).is_file():
            raise DatasetToolError("artifact_bytes_mismatch")
    manifest = _read(root / "manifest.json", sealed=True)
    if set(manifest) != MANIFEST_FIELDS or manifest.get("schema") != MANIFEST_SCHEMA:
        raise DatasetToolError("manifest_schema_mismatch")
    if manifest["status"] != "complete" or manifest["failures"] != []:
        raise DatasetToolError("generation_not_complete")
    declaration = _read(root / "declaration.json", sealed=True)
    config = _validate_declaration(declaration)
    if (manifest["declaration_sha256"] != declaration["sha256"]
            or any(manifest[key] != declaration[key] for key in
                   ("run_id", "registry_id", "config_sha256", "split_plan_sha256", "schedule"))
            or manifest["final_suite_identity_sha256"] != declaration["schedule"]["final_suite_identity_sha256"]
            or manifest["final_state"] != ("sealed" if config.mode == "prospective" else "development_fixture")):
        raise DatasetToolError("manifest_declaration_binding_mismatch")
    partitions = declaration["schedule"]["partitions"]
    expected_shards = {f"{p}-{i:05d}.jsonl" for p in PARTITIONS
                       for i in range((partitions[p] + config.batch_size - 1) // config.batch_size)}
    if not isinstance(manifest["artifacts"], list) or not isinstance(manifest["shards"], list):
        raise DatasetToolError("artifact_inventory_invalid")
    names, shard_names, artifact_names = set(), set(), set()
    for group, required_fields in (("artifacts", {"path", "bytes", "sha256"}),
                                    ("shards", {"path", "bytes", "sha256", "partition", "rows", "sample_ids_sha256"})):
        for item in manifest[group]:
            if not isinstance(item, dict) or set(item) != required_fields:
                raise DatasetToolError("artifact_inventory_invalid")
            name = item["path"]
            if (type(name) is not str or Path(name).name != name or name in names
                    or type(item["bytes"]) is not int or item["bytes"] <= 0):
                raise DatasetToolError("artifact_inventory_invalid")
            names.add(name)
            (shard_names if group == "shards" else artifact_names).add(name)
            _digest(item["sha256"])
            path = root / name
            bound = (config.batch_size * MAX_ROW_BYTES if group == "shards" else
                     (config.expected_tasks * MAX_ROW_BYTES if name == "pending.jsonl" else
                      (2 * config.expected_tasks + 1) * MAX_ROW_BYTES if name == "attempts.jsonl" else MAX_JSON_BYTES))
            if (item["bytes"] > bound or path.is_symlink() or not path.is_file()
                    or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]):
                raise DatasetToolError("artifact_bytes_mismatch")
    if artifact_names != ARTIFACT_NAMES or shard_names != expected_shards:
        raise DatasetToolError("artifact_inventory_incomplete_or_unexpected")
    allowed = names | {"manifest.json"}
    if require_supervisor or (root / "supervisor.json").exists():
        allowed.add("supervisor.json")
    observed = set()
    for path in root.iterdir():
        if path.name not in allowed or path.is_symlink() or not path.is_file():
            raise DatasetToolError("unauthenticated_artifact_entry")
        observed.add(path.name)
    if observed != allowed:
        raise DatasetToolError("artifact_inventory_incomplete_or_unexpected")

    api = _api()
    samples, task_iterator = [], iter(api.iter_tasks(config))
    sample_hash = hashlib.sha256(b"[")
    for index, value in enumerate(_jsonl_rows(root / "pending.jsonl", config.expected_tasks)):
        sample = api.GeometricSample.from_dict(value)
        if sample.task != next(task_iterator, None):
            raise DatasetToolError("pending_sample_differs_from_declared_schedule")
        if index:
            sample_hash.update(b",")
        sample_hash.update(canonical_json(sample.sha256).encode())
        samples.append(sample)
    sample_hash.update(b"]")
    if (len(samples) != config.expected_tasks or next(task_iterator, None) is not None
            or sample_hash.hexdigest() != manifest["all_samples_sha256"]):
        raise DatasetToolError("pending_samples_incomplete_or_digest_mismatch")

    def partition_rows(partition):
        return (sample for sample in samples if sample.task.partition == partition)

    cursors = {p: iter(partition_rows(p)) for p in PARTITIONS}
    shard_index, observed_rows = Counter(), Counter()
    for item in manifest["shards"]:
        partition = item["partition"]
        if partition not in PARTITIONS or item["path"] != f"{partition}-{shard_index[partition]:05d}.jsonl":
            raise DatasetToolError("shard_partition_or_order_mismatch")
        expected_rows = min(config.batch_size, partitions[partition] - observed_rows[partition])
        if type(item["rows"]) is not int or item["rows"] != expected_rows:
            raise DatasetToolError("shard_row_count_mismatch")
        ids = []
        for value in _jsonl_rows(root / item["path"], expected_rows):
            sample = next(cursors[partition], None)
            if sample is None or sha256_json(value) != sample.sha256:
                raise DatasetToolError("shard_sample_differs_from_retained_result")
            ids.append(sample.task.id)
        if len(ids) != expected_rows or item["sample_ids_sha256"] != sha256_json(ids):
            raise DatasetToolError("shard_row_identity_mismatch")
        observed_rows[partition] += len(ids)
        shard_index[partition] += 1
    if dict(observed_rows) != partitions or any(next(cursor, None) is not None for cursor in cursors.values()):
        raise DatasetToolError("shard_population_incomplete")

    accounting = _verified_journal(root / "attempts.jsonl", samples, declaration, config)
    actual = accounting["teacher_iterations_actual_lower_bound"]
    expected_counts = {"attempted": config.expected_tasks, "completed": config.expected_tasks, "unknown_attempts": 0,
                       "teacher_iterations_actual_lower_bound": actual, "teacher_iterations_conservative": actual,
                       "teacher_iterations_reserved": config.expected_tasks * config.teacher.max_iterations}
    if canonical_json(manifest["counts"]) != canonical_json(expected_counts):
        raise DatasetToolError("manifest_counts_disagree_with_retained_work")
    cost = manifest["generation_cost"]
    cost_fields = {"worker_module_seconds", "peak_rss_mib", "hard_memory_limit_mib", "memory_limit_method",
                   "whole_process_receipt", "device"}
    if (not isinstance(cost, dict) or set(cost) != cost_fields
            or not _finite_nonnegative(cost["worker_module_seconds"])
            or not cost["worker_module_seconds"] < config.budget.max_wall_seconds
            or cost["hard_memory_limit_mib"] != config.budget.max_memory_mib
            or cost["memory_limit_method"] not in ("RLIMIT_AS", "Windows Job Object process committed-memory limit")
            or cost["whole_process_receipt"] != "supervisor.json" or cost["device"] != "cpu"
            or (cost["peak_rss_mib"] is not None and (not _finite_nonnegative(cost["peak_rss_mib"])
                                                   or cost["peak_rss_mib"] > config.budget.max_memory_mib))):
        raise DatasetToolError("generation_cost_or_memory_admission_invalid")
    # These checks inspect the recorded teacher output, never solve the task.
    api.validate_dataset(samples, config, require_complete=True)
    normalization = api.fit_normalization(partition_rows("training"), config)
    if canonical_json(_read(root / "normalization.json")) != canonical_json(normalization.to_dict()):
        raise DatasetToolError("training_normalization_mismatch")
    coverage = api.coverage_report(samples, config)
    if canonical_json(_read(root / "coverage.json")) != canonical_json(coverage):
        raise DatasetToolError("coverage_mismatch")
    if config.mode == "prospective" and coverage["accepted_final_population_minima_met"] is not True:
        raise DatasetToolError("accepted_final_population_minima_not_met")
    if require_supervisor or (root / "supervisor.json").exists():
        _verify_supervisor(root, manifest, declaration, config, accounting)
    return manifest


def _opening(registry, manifest):
    path = registry / (manifest["final_suite_identity_sha256"] + ".final-opened.json")
    if not path.exists():
        return None
    receipt = _read(path, sealed=True)
    if (receipt.get("schema") != "nca-ik-final-opening/v1"
            or receipt.get("suite_identity_sha256") != manifest["final_suite_identity_sha256"]
            or receipt.get("registry_id") != manifest["registry_id"]
            or receipt.get("future_experiments_status") != "development_only"):
        raise DatasetToolError("final_opening_binding_invalid")
    return receipt


def inspect(output, registry_path):
    manifest = verify_artifact(output)
    registry, registry_id = _registry(registry_path)
    if registry_id != manifest["registry_id"]:
        raise DatasetToolError("registry_identity_mismatch")
    opened = _opening(registry, manifest)
    return {"status": manifest["status"], "manifest_sha256": manifest["sha256"], "counts": manifest["counts"],
            "schedule": manifest["schedule"], "generation_cost": manifest["generation_cost"],
            "final_state": "opened_development_for_later_experiments" if opened is not None else manifest["final_state"],
            "coverage": _read(Path(output) / "coverage.json"), "raw_final_rows_returned": False}


def read_partition(output, partition, registry_path, *, development=False):
    if partition not in PARTITIONS:
        raise DatasetToolError("partition_unknown")
    manifest = verify_artifact(output)
    registry, registry_id = _registry(registry_path)
    if registry_id != manifest["registry_id"]:
        raise DatasetToolError("registry_identity_mismatch")
    if partition == "final" and manifest["final_state"] != "development_fixture":
        opened = _opening(registry, manifest)
        if not development or opened is None:
            raise DatasetToolError("final_suite_sealed_or_requires_development_label")
    rows = []
    for item in manifest["shards"]:
        if item["partition"] != partition:
            continue
        with (Path(output) / item["path"]).open("rb") as stream:
            while True:
                line = stream.readline(MAX_ROW_BYTES + 1)
                if not line:
                    break
                if not line.endswith(b"\n") or len(line) > MAX_ROW_BYTES:
                    raise DatasetToolError("invalid_or_oversized_sample_line")
                rows.append(_api().GeometricSample.from_dict(strict_json_loads(line, max_bytes=MAX_ROW_BYTES)))
    return rows


def open_final(output, registry_path, experiment, candidate_hashes, destination):
    _portable_name(experiment)
    hashes = tuple(_digest(value) for value in candidate_hashes)
    if not hashes or len(set(hashes)) != len(hashes):
        raise DatasetToolError("distinct_frozen_candidate_hashes_required")
    manifest = verify_artifact(output)
    registry, registry_id = _registry(registry_path)
    if registry_id != manifest["registry_id"]:
        raise DatasetToolError("registry_identity_mismatch")
    if manifest["final_state"] != "sealed":
        raise DatasetToolError("development_fixture_is_not_pristine_final_data")
    destination = _outside(destination)
    if destination.exists():
        raise DatasetToolError("immutable_opening_destination_exists")
    receipt = _seal({"schema": "nca-ik-final-opening/v1", "experiment": experiment,
                     "frozen_candidates_sha256": list(hashes), "opened_utc": utc_now(),
                     "suite_identity_sha256": manifest["final_suite_identity_sha256"],
                     "dataset_manifest_sha256": manifest["sha256"], "registry_id": registry_id,
                     "future_experiments_status": "development_only"})
    # This transition is durable before any raw final sample or exported path is
    # returned. A copy of the dataset cannot reset its identity in this registry.
    immutable_json(registry / (manifest["final_suite_identity_sha256"] + ".final-opened.json"), receipt)
    destination.mkdir(parents=True, exist_ok=False)
    immutable_json(destination / "opening.json", receipt)
    for item in manifest["shards"]:
        if item["partition"] == "final":
            immutable_bytes(destination / item["path"], (Path(output) / item["path"]).read_bytes())
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    declaration = commands.add_parser("declare")
    declaration.add_argument("--config", type=Path, required=True)
    declaration.add_argument("--output", type=Path, required=True)
    declaration.add_argument("--registry", type=Path, required=True)
    for name in ("generate", "_worker"):
        command = commands.add_parser(name)
        command.add_argument("--declaration", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--registry", type=Path, required=True)
    inspection = commands.add_parser("inspect")
    inspection.add_argument("--artifact", type=Path, required=True)
    inspection.add_argument("--registry", type=Path, required=True)
    opening = commands.add_parser("open-final")
    opening.add_argument("--artifact", type=Path, required=True)
    opening.add_argument("--registry", type=Path, required=True)
    opening.add_argument("--experiment", required=True)
    opening.add_argument("--candidate-sha256", action="append", required=True)
    opening.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "declare":
            result = declare(args.config, args.output, args.registry)
            shown = {key: result[key] for key in ("run_id", "sha256", "config_sha256", "schedule")}
        elif args.command == "generate":
            result = shown = generate(args.declaration, args.output, args.registry)
        elif args.command == "_worker":
            result = _worker(args.declaration, args.output, args.registry)
            shown = {"status": result["status"], "counts": result["counts"]}
        elif args.command == "inspect":
            result = shown = inspect(args.artifact, args.registry)
        else:
            result = shown = open_final(args.artifact, args.registry, args.experiment,
                                        args.candidate_sha256, args.output)
        print(canonical_json(shown))
        return 0 if result.get("status", "complete") == "complete" else 1
    except (Exception, KeyboardInterrupt) as exc:
        print(canonical_json({"status": "refused", "reason": exc.code if isinstance(exc, (DatasetToolError, ContractError))
                              else type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
