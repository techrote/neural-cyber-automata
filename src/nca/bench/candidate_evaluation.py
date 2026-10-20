"""One declared evaluation of an unchanged BC candidate; never trains weights."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from nca.contracts import ContractError, sha256_json
from nca.export.build import immutable_json, read_json, verify_artifact_files, verify_self_hash
from nca.export.candidate import (BUDGET, LEARNER_REVISION, LEARNER_SOURCES, MANIFEST_SCHEMA,
                                  SUMMARY_SCHEMA, verify_origin)
from .deployment import _sources


def _signal_owned_group(process, signum):
    """Only the child group created by this invocation is signalled."""
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def committed_sources(root: Path, declaration_path: Path):
    """The declaration and all execution bytes must belong to this Git head."""
    relative = declaration_path.resolve().relative_to(root.resolve()).as_posix()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=root)
    if committed != declaration_path.read_bytes():
        raise ContractError("candidate_publication", "declaration bytes differ from the recorded head")
    sources = _sources(root)
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0"))
    if not set(sources) <= tracked or subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *sorted(sources)], cwd=root).returncode:
        raise ContractError("candidate_publication", "all execution sources must match the committed head")
    return revision, sources, relative


def run_candidate_evaluation(*, root: Path, training_root: Path, learner_source: Path,
                             declaration_path: Path, host_path: Path, godot_path: Path, output: Path):
    import torch
    torch.set_num_threads(1)
    root, training_root, learner_source, declaration_path, output = (
        p.resolve() for p in (root, training_root, learner_source, declaration_path, output))
    started = time.monotonic()
    if output.exists() or not output.is_relative_to(training_root) or output == training_root:
        raise ContractError("candidate_output", "new separate child collection required; original artifacts stay immutable")
    declaration, declaration_file_hash = read_json(declaration_path)
    revision, sources, relative = committed_sources(root, declaration_path)
    training = verify_origin(training_root, declaration)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=learner_source, text=True).strip() != LEARNER_REVISION:
        raise ContractError("candidate_original_source", "original immutable learner checkout required")
    verify_artifact_files(learner_source, training["manifest"]["source_files"])
    manifest = {"schema": MANIFEST_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                "phase": "frozen_before_any_candidate_evaluation", "declaration": declaration,
                "declaration_content_sha256": sha256_json(declaration), "declaration_file": relative,
                "declaration_file_sha256": declaration_file_hash, "evaluation_revision": revision,
                "evaluation_source_files": sources, "evaluation_sources_sha256": sha256_json(sources),
                "policy_execution_revision": LEARNER_REVISION, "policy_execution_sources_sha256": LEARNER_SOURCES,
                "runtime": {"torch": str(torch.__version__), "torch_threads": 1},
                "training_status": training["summary"]["status"], "original_termination_confirmed": False,
                "new_optimizer_updates": 0}
    manifest["sha256"] = sha256_json(manifest)
    immutable_json(output / "manifest.json", manifest)
    result, errors, expired, process, worker_file_hash = None, [], False, None, None
    worker_exit_code = None
    environment = dict(os.environ, PYTHONPATH=str(learner_source / "src"))
    remaining = BUDGET["maximum_wall_seconds"] - (time.monotonic() - started)
    try:
        if remaining <= 0:
            raise TimeoutError("candidate preflight exhausted its total declared wall budget")
        command = [sys.executable, str(root / "src/nca/bench/candidate_worker.py"),
                   "--training-root", str(training_root), "--output", str(output),
                   "--host", str(host_path), "--godot", str(godot_path), "--wall-budget", str(remaining)]
        with (output / "worker-local-diagnostics.txt").open("x") as diagnostics:
            process = subprocess.Popen(command, cwd=learner_source, env=environment,
                                       stdout=diagnostics, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                worker_exit_code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                expired = True
                _signal_owned_group(process, signal.SIGINT)
                try:
                    worker_exit_code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _signal_owned_group(process, signal.SIGKILL)
                    worker_exit_code = process.wait(timeout=5)
            if worker_exit_code:
                errors.append({"phase": "worker", "type": "WorkerFailure", "code": "candidate_worker_failed"})
        if (output / "worker-result.json").is_file():
            candidate_result, worker_file_hash = read_json(output / "worker-result.json")
            verify_self_hash(candidate_result)
            if (candidate_result["schema"] != "nca-fixed-candidate-worker-result/v1"
                    or candidate_result["manifest_sha256"] != manifest["sha256"]
                    or candidate_result["status"] not in ("completed", "failed", "capped")
                    or not isinstance(candidate_result["results"], list) or not isinstance(candidate_result["errors"], list)
                    or any(type(candidate_result[k]) is not int or candidate_result[k] < 0 for k in
                           ("attempted_decisions", "actual_decisions", "actual_ticks", "host_start_attempts"))
                    or type(candidate_result["source_files_unchanged"]) is not bool
                    or candidate_result["policy_sha256"] != declaration["policy_sha256"]
                    or candidate_result["new_optimizer_updates"] != 0):
                raise ContractError("candidate_worker_lineage", "worker result differs from its frozen manifest")
            result = candidate_result
        else:
            errors.append({"phase": "worker", "type": "MissingResult", "code": "candidate_worker_incomplete"})
    except BaseException as exc:
        errors.append({"phase": "execution", "type": type(exc).__name__,
                       "code": exc.code if isinstance(exc, ContractError) else "candidate_execution_failure"})
    finally:
        if process is not None and process.poll() is None:
            try:
                _signal_owned_group(process, signal.SIGINT)
                try:
                    worker_exit_code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _signal_owned_group(process, signal.SIGKILL)
                    worker_exit_code = process.wait(timeout=5)
            except BaseException as exc:
                errors.append({"phase": "cleanup", "type": type(exc).__name__, "code": "worker_termination_unconfirmed"})
    worker_termination_confirmed = process is None or process.poll() is not None
    source_unchanged = _sources(root) == sources
    training_unchanged = True
    try:
        verify_artifact_files(training_root, declaration["training_files"])
        verify_artifact_files(learner_source, training["manifest"]["source_files"])
        verify_artifact_files(root, {relative: declaration_file_hash})
    except ContractError:
        training_unchanged = False
    if not source_unchanged or not training_unchanged:
        errors.append({"phase": "verification", "type": "InputDrift", "code": "candidate_source_or_input_changed"})
    elapsed = time.monotonic() - started
    within_budget = elapsed <= BUDGET["maximum_wall_seconds"] and not expired
    rows = [] if result is None else result["results"]
    confirmed_decisions = confirmed_ticks = 0
    if result is None:
        # If process control is interrupted, retained completed episode files
        # still establish a lower bound. An unknown in-flight outcome stays
        # explicitly unknown; it cannot be recorded as zero executed work.
        from nca.data import read_episode
        for path in sorted((output / "metrics").glob("*.json")):
            try:
                row, _ = read_json(path)
                verify_artifact_files(output, {row["artifact"]: row["artifact_sha256"]})
                episode = read_episode(output / row["artifact"])
                rows.append(row)
                confirmed_decisions += episode.decisions
                confirmed_ticks += episode.consumed_ticks
            except Exception as exc:
                errors.append({"phase": "retained_prefix", "type": type(exc).__name__, "code": "unreadable_retained_record"})
    families = {f["id"] for f in training["manifest"]["config"]["evaluation_families"]}
    complete = (result is not None and result["status"] == "completed" and not result["errors"]
                and len(rows) == 8 and {row["scenario_family"] for row in rows} == families
                and result["attempted_decisions"] == result["actual_decisions"] <= BUDGET["maximum_decisions"]
                and result["actual_ticks"] <= BUDGET["maximum_ticks"] and result["source_files_unchanged"]
                and result["policy_sha256"] == declaration["policy_sha256"]
                and result["new_optimizer_updates"] == 0 and source_unchanged and training_unchanged and within_budget
                and worker_exit_code == 0 and worker_termination_confirmed)
    summary = {"schema": SUMMARY_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
               "manifest_sha256": manifest["sha256"], "status": "completed" if complete and not errors else
                   "capped" if not within_budget else "failed", "errors": errors,
               "training_status": training["summary"]["status"], "original_termination_confirmed": False,
               "policy_sha256": declaration["policy_sha256"],
               "complete_eight_family_coverage": complete, "results": rows,
               "actual_decisions": confirmed_decisions if result is None else result["actual_decisions"],
               "attempted_decisions": None if result is None else result["attempted_decisions"],
               "actual_ticks": confirmed_ticks if result is None else result["actual_ticks"],
               "unconfirmed_pending_outcome": (process is not None if result is None else
                   result["attempted_decisions"] > result["actual_decisions"]),
               "unconfirmed_decision_reservations": None if result is None else
                   max(0, result["attempted_decisions"] - result["actual_decisions"]),
               "worker_invocations": int(process is not None),
               "worker_exit_code": worker_exit_code, "worker_termination_confirmed": worker_termination_confirmed,
               "worker_result_file_sha256": worker_file_hash,
               "host_start_attempts": None if result is None else result["host_start_attempts"],
               "wall_seconds": elapsed, "source_files_unchanged": source_unchanged,
               "training_files_unchanged": training_unchanged, "new_optimizer_updates": 0,
               "efficacy_comparison": False, "runtime_qualified": False, "owner_feel_accepted": False,
               "scope": "single fixed candidate evaluation; no efficacy comparison or deployment qualification"}
    summary["sha256"] = sha256_json(summary)
    immutable_json(output / "summary.json", summary)
    return summary
