#!/usr/bin/env python3
"""Observe one admitted Jolt v3 CLI exit without renewing its original clock.

Original stdout/stderr and the actual wait status are retained independently of
the worker's result. This observer never substitutes an earlier completion flag
for process exit, retries a child, or performs another neural/physical audit.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import time
ENTRY_MONOTONIC = time.monotonic()

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def observe(args, *, started, command=None):
    from nca.contracts import strict_json_loads
    from nca.evaluation.durable_journal import file_identity, sync_directory
    from nca.evaluation.durable_publication import publish_json_v2
    from nca.train.jolt_execution_v3 import OriginalClockV3, validate_budget_v3
    from nca.train.jolt_executor_v3 import verify_lifecycle
    from nca.train.jolt_storage_v3 import file_identity_v3
    from nca.train.jolt_streams_v3 import wait_bounded_child
    import execute_jolt_diagnostic as helper

    # Select timing only from the explicit declaration budget. Omitted nulls
    # cannot turn a historical bounded invocation into an unlimited one.
    declaration_identity = file_identity_v3(args.declaration, maximum_bytes=1_048_576)
    declaration = strict_json_loads(Path(args.declaration).read_bytes())
    budget = validate_budget_v3(declaration.get("budget") if isinstance(declaration, dict) else None)
    maximum_wall_seconds = budget["maximum_wall_seconds"]
    clock = OriginalClockV3(started, maximum_wall_seconds=maximum_wall_seconds,
                            monotonic=time.monotonic)
    output = Path(args.output)
    names = {channel: output.with_name(output.name + "-outer-" + channel + ".log")
             for channel in ("stdout", "stderr")}
    receipt_path = output.with_name(output.name + "-observed-exit.json")
    lifecycle = output.with_name(output.name + "-lifecycle")
    if any(os.path.lexists(path) for path in (output, lifecycle, receipt_path, *names.values())):
        raise ValueError("an original output or observation path already exists; retry is forbidden")
    output.parent.mkdir(parents=True, exist_ok=True)
    child_command = command or [sys.executable, str(ROOT / "tools/learn_jolt.py"), "run-v3",
        "--config", str(args.config), "--declaration", str(args.declaration),
        "--checkout", str(args.checkout), "--godot", str(args.godot),
        "--output", str(output), "--whole-started", repr(started)]
    process = identity = exit_elapsed = None
    cleanup_confirmed = False
    errors = []

    def failed(exc, stage):
        code = getattr(exc, "code", None)
        errors.append({"stage": stage, "type": type(exc).__name__,
                       "code": code if isinstance(code, str) and len(code) <= 80 else None})

    with names["stdout"].open("xb") as stdout, names["stderr"].open("xb") as stderr:
        sync_directory(output.parent)
        try:
            clock.check("finalization")
            identity = helper.prepare_process_tree()
            process = subprocess.Popen(child_command, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            wait_bounded_child(process, stdout, stderr,
                               deadline=None if maximum_wall_seconds is None else started + maximum_wall_seconds,
                               monotonic=time.monotonic, maximum_bytes=65536)
            exit_elapsed = clock.elapsed()
        except BaseException as exc:
            failed(exc, "original_cli")
        finally:
            try:
                if identity is not None:
                    remaining = clock.remaining("finalization")
                    cleanup_confirmed = helper.terminate_owned_tree(process, identity,
                        grace_seconds=None if remaining is None else min(3., remaining))
                elif process is not None:
                    process.kill()
                    process.wait(timeout=None if maximum_wall_seconds is None else 3)
                if process is not None and process.poll() is not None and exit_elapsed is None:
                    exit_elapsed = clock.elapsed()
            except BaseException as exc:
                failed(exc, "independent_cleanup")
            for channel, stream in (("stdout", stdout), ("stderr", stderr)):
                try:
                    stream.flush()
                    os.fsync(stream.fileno())
                except BaseException as exc:
                    failed(exc, channel + "_sync")
    code = None if process is None else process.poll()
    original = None
    if code is not None and exit_elapsed is not None:
        try:
            original = verify_lifecycle(lifecycle, terminal_stdout=names["stdout"],
                observed_exit_code=code, observed_exit_elapsed_seconds=exit_elapsed,
                expected_declaration=declaration_identity, expected_budget=budget)
        except BaseException as exc:
            failed(exc, "original_lifecycle_readback")
    receipt = {"schema": "nca-jolt-003-independently-observed-exit/v1",
        "whole_process_started_monotonic": started, "maximum_wall_seconds": maximum_wall_seconds,
        "allocation_id": output.name, "declaration": declaration_identity,
        "exit_observed": code is not None, "observed_exit_code": code,
        "observed_exit_elapsed_seconds": exit_elapsed,
        "stdout": file_identity(names["stdout"]), "stderr": file_identity(names["stderr"]),
        "cleanup_confirmed": cleanup_confirmed, "lifecycle_readback": original,
        "wall_seconds_before_receipt_publication": clock.elapsed(), "errors": errors,
        "automatic_retry": False, "automatic_extension": False}
    publish_json_v2(receipt_path, receipt)
    elapsed = clock.elapsed()
    completed = bool(not errors and cleanup_confirmed and code == 0 and original
                     and original.get("completed") is True
                     and (maximum_wall_seconds is None or elapsed <= maximum_wall_seconds))
    # Receipt fsync is observed after publication; it cannot attest to itself.
    print(json.dumps({"schema": "nca-jolt-003-observer-terminal/v1", "completed": completed,
        "observed_exit": file_identity(receipt_path), "wall_seconds": elapsed,
        "original_exit_code": code}), flush=True)
    final_elapsed = clock.elapsed()
    return 0 if completed and (maximum_wall_seconds is None or final_elapsed <= maximum_wall_seconds) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/learning/jolt-learn-003-v2.json")
    for name in ("declaration", "checkout", "godot", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    return observe(args, started=ENTRY_MONOTONIC)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"error": {"type": type(exc).__name__, "code": getattr(exc, "code", None)}}),
              file=sys.stderr, flush=True)
        raise SystemExit(2)
