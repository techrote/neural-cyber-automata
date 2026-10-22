#!/usr/bin/env python3
"""Measure one dataset generator process through exit and receipt readback.

The envelope includes Python startup, worker startup/exit, semantic readback,
and the inner supervisor's final receipt publication. Its own receipt write is
reported separately rather than claiming a clock can include its own exit.
"""

from __future__ import annotations

import time
_STARTED = time.perf_counter()

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

from nca.contracts import ContractError, canonical_json
from nca.runs import immutable_json, sha256_file, utc_now
from ik_dataset import (DatasetToolError, ROOT, SUPERVISOR_CHECKS, _api,
                        _outside, _read, _registry, _seal)


def _terminate(child):
    """Terminate this envelope's own generator and descendants after timeout."""
    if os.name == "posix":
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=5, check=False)
    else:
        child.kill()
    return child.wait(timeout=5)


def supervise(declaration_path, output, registry_path):
    declaration_path, output = _outside(declaration_path), _outside(output)
    declaration = _read(declaration_path, sealed=True)
    config = _api().DatasetConfig.from_dict(declaration["config"])
    maximum = config.budget.max_wall_seconds
    registry, registry_id = _registry(registry_path)
    key = declaration["sha256"]
    if declaration["registry_id"] != registry_id:
        raise DatasetToolError("registry_identity_mismatch")
    helper = "tools/ik_dataset_supervise.py"
    if declaration["source"]["files"].get(helper) != sha256_file(ROOT / helper):
        raise DatasetToolError("envelope_source_not_declared")
    receipt_path = output.with_name(output.name + ".envelope.json")
    if output.exists() or receipt_path.exists() or (registry / (key + ".generation-started.json")).exists():
        raise DatasetToolError("allocation_or_output_already_consumed")
    declaration_file_hash = sha256_file(declaration_path)
    immutable_json(registry / (key + ".envelope-started.json"), {
        "schema": "nca-ik-dataset-process-start/v1", "declaration_sha256": key,
        "run_id": declaration["run_id"], "started_utc": utc_now(),
        "attempt": 1, "retry_or_resume": False, "max_wall_seconds": maximum,
    })
    child = None
    returncode = None
    timed_out = False
    failures, checks, hashes = [], {}, {}
    process_started = time.perf_counter()
    try:
        allowance = maximum - 2.0 - (process_started - _STARTED)
        if allowance <= 0:
            raise DatasetToolError("envelope_admission_budget_exhausted")
        child = subprocess.Popen([
            sys.executable, "tools/ik_dataset.py", "generate",
            "--declaration", str(declaration_path), "--output", str(output),
            "--registry", str(registry),
        ], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=os.name == "posix")
        try:
            returncode = child.wait(timeout=allowance)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = _terminate(child)
        process_seconds = time.perf_counter() - process_started
        checks["complete_generator_process"] = returncode == 0 and not timed_out
        checks["declaration_file_unchanged"] = sha256_file(declaration_path) == declaration_file_hash
        inner = _read(output / "supervisor.json", sealed=True)
        manifest = _read(output / "manifest.json", sealed=True)
        hashes = {name: sha256_file(output / name) for name in ("supervisor.json", "manifest.json")}
        checks["inner_semantic_readback_passed"] = (
            inner.get("schema") == "nca-ik-dataset-supervisor/v1"
            and inner.get("status") == "complete"
            and set(inner.get("checks", {})) == SUPERVISOR_CHECKS
            and all(value is True for value in inner["checks"].values())
            and inner.get("child_returncode") == 0
            and inner.get("execution_failure") is None
            and inner.get("child_timed_out") is False
            and inner.get("interrupted") is False)
        checks["artifact_bindings"] = (
            inner.get("declaration_sha256") == manifest.get("declaration_sha256") == key
            and inner.get("manifest_file_sha256") == hashes["manifest.json"]
            and manifest.get("status") == "complete"
            and manifest.get("run_id") == declaration["run_id"])
    except (Exception, KeyboardInterrupt) as exc:
        failures.append(exc.code if isinstance(exc, (DatasetToolError, ContractError)) else type(exc).__name__)
    finally:
        if child is not None and child.poll() is None:
            try:
                returncode = _terminate(child)
            except (Exception, KeyboardInterrupt) as exc:
                failures.append("process_termination_" + type(exc).__name__)
        process_seconds = time.perf_counter() - process_started
    elapsed = time.perf_counter() - _STARTED
    checks["process_exit_and_receipt_readback_within_budget"] = elapsed < maximum
    receipt = _seal({
        "schema": "nca-ik-dataset-process-envelope/v1", "run_id": declaration["run_id"],
        "declaration_sha256": key, "declaration_file_sha256": declaration_file_hash,
        "status": "complete" if not failures and checks and all(checks.values()) else "incomplete",
        "checks": checks, "failures": failures, "child_returncode": returncode,
        "child_timed_out": timed_out, "generator_process_and_receipt_readback_seconds": process_seconds,
        "module_through_readback_seconds": elapsed, "max_wall_seconds": maximum,
        "envelope_final_receipt_write_excluded_from_elapsed": True,
        "artifact_file_sha256": hashes, "completed_utc": utc_now(),
        "attempt": 1, "retry_or_resume": False,
    })
    immutable_json(receipt_path, receipt)
    if time.perf_counter() - _STARTED >= maximum:
        immutable_json(receipt_path.with_name(receipt_path.name + ".overrun.json"),
                       {"status": "incomplete", "declaration_sha256": key})
        raise DatasetToolError("envelope_publication_budget_exhausted")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = supervise(args.declaration, args.output, args.registry)
        print(canonical_json(result))
        return 0 if result["status"] == "complete" else 1
    except (Exception, KeyboardInterrupt) as exc:
        print(canonical_json({"status": "refused", "reason": exc.code if isinstance(exc, (DatasetToolError, ContractError))
                              else type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
