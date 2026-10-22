"""Opt-in learning executor with independently retained lifecycle receipts.

This module starts at most one worker. It does not repair an earlier allocation,
load checkpoints or interpret a failed journal as complete evidence. Every
receipt is written once; a partial publication remains an honest failure.

The durable terminal is a completion *candidate*: its own fsync and the Python
process exit cannot be observed from inside the bytes being written. Acceptance
also requires the final post-publication stdout verdict and an observed zero
CLI exit. A lost terminal stream, SIGKILL or unavailable filesystem never
becomes proof of completion merely because an earlier stage receipt exists.

© 2026 techrote. All rights reserved.
"""

from __future__ import annotations

from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from nca.contracts import ContractError, strict_json_loads
from nca.evaluation.durable_journal import file_identity
from nca.evaluation.durable_publication import publish_json_v2
from nca.train.jolt_execution_contract import validate_v2_allocation_identity


MAXIMUM_WALL_SECONDS = 1800
VERIFICATION_RESERVE_SECONDS = 30
FINALIZATION_RESERVE_SECONDS = 5
MAX_VERIFIER_BYTES = 65_536
DECLARATION_SCHEMA = "nca-learn-jolt-002-execution/v2"


class _StopExecution(Exception):
    """A retained failure already explains why no ordinary work may follow."""


def _error(exc, stage):
    # Arbitrary exception text may contain local paths or host traces. Retain
    # only the exception class and a bounded machine-readable contract code.
    code = getattr(exc, "code", None)
    if (not isinstance(code, str) or not 0 < len(code) <= 80
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                   for character in code)):
        code = None
    return {"stage": stage, "type": type(exc).__name__, "code": code}


def execute(args, *, whole_started, script, worker_mode="worker-v2", inventory_mode="inventory-v2"):
    """Supervise one explicitly admitted v2 worker under its original clock.

    ``script`` is the owning repository CLI, not a shell command. The modes are
    explicit integration points for a separately versioned worker/inventory
    contract. There is no recovery, automatic retry or alternate worker.
    """
    if (type(whole_started) not in (float, int) or not math.isfinite(whole_started)
            or not 0 < whole_started <= time.monotonic()):
        raise ValueError("the original positive monotonic invocation origin is required")
    output = Path(args.output)
    lifecycle = output.with_name(output.name + "-lifecycle")
    if output.exists() or lifecycle.exists():
        raise ValueError("this allocation path has already been used; automatic retry is forbidden")
    # Exclusive directory creation also arbitrates competing executors. A
    # startup failure consumes this path even when it starts no worker.
    lifecycle.mkdir(parents=True)
    deadline = whole_started + MAXIMUM_WALL_SECONDS
    errors, receipts, processes = [], {}, {}
    helper = owner_identity = None
    cleanup_confirmed, owned_remaining = False, None
    timed_out, verifier = False, None
    declaration_identity = None

    def elapsed():
        return time.monotonic() - whole_started

    def record_error(exc, stage):
        errors.append(_error(exc, stage))

    def persist(name, value):
        try:
            publish_json_v2(lifecycle / name, value)
            identity = file_identity(lifecycle / name)
            receipts[name] = identity
            return identity
        except BaseException as exc:
            record_error(exc, "publish_" + name.removesuffix(".json"))
            receipts[name] = None
            return None

    def required_receipt(name, value):
        if persist(name, value) is None:
            raise _StopExecution()

    def check_deadline(stage, reserve=0):
        nonlocal timed_out
        if time.monotonic() >= deadline - reserve:
            timed_out = True
            errors.append({"stage": stage, "type": "DeadlineExceeded", "code": "original_wall_deadline"})
            raise _StopExecution()

    def cleanup(process, phase):
        nonlocal cleanup_confirmed, owned_remaining
        current_confirmed = False
        try:
            if helper is None or owner_identity is None:
                raise RuntimeError("owned process identity was not established")
            current_confirmed = helper.terminate_owned_tree(process, owner_identity, grace_seconds=3)
            owned_remaining = len(helper.owned_processes(owner_identity))
            current_confirmed = current_confirmed is True and owned_remaining == 0
        except BaseException as exc:
            record_error(exc, phase + "_cleanup")
            owned_remaining = None
            # A direct handle can revoke the known worker without inventing a
            # complete descendant-tree claim after /proc inspection fails.
            if process is not None:
                try:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=3)
                except BaseException as fallback:
                    record_error(fallback, phase + "_direct_cleanup")
        cleanup_confirmed = current_confirmed
        persist(phase + "-cleanup.json", {
            "schema": "jolt-learning-executor-cleanup/v2", "phase": phase,
            "cleanup_confirmed": current_confirmed, "owned_processes_remaining": owned_remaining,
            "returncode": None if process is None else process.returncode,
            "wall_seconds": elapsed(), "errors": list(errors),
        })

    def run_child(phase, command, reserve):
        nonlocal timed_out
        process = None
        # Each stream is synchronized independently even if its peer fails.
        streams = []
        try:
            check_deadline(phase + "_before_spawn", reserve)
            with ExitStack() as stack:
                for channel in ("stdout", "stderr"):
                    streams.append(stack.enter_context((lifecycle / f"{phase}-{channel}.log").open("xb")))
                try:
                    process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                        stdout=streams[0], stderr=streams[1], start_new_session=True)
                    processes[phase] = process
                    required_receipt(phase + "-started.json", {
                        "schema": "jolt-learning-executor-process-start/v2", "phase": phase,
                        "spawned": True, "wall_seconds": elapsed(),
                        "whole_process_started_monotonic": whole_started,
                    })
                    check_deadline(phase + "_after_spawn", reserve)
                    process.wait(timeout=max(.001, deadline - reserve - time.monotonic()))
                    check_deadline(phase + "_after_exit", reserve)
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    record_error(exc, phase + "_wait")
                except _StopExecution:
                    pass
                except BaseException as exc:
                    record_error(exc, phase + "_execution")
                finally:
                    cleanup(process, phase)
                    for channel, stream in zip(("stdout", "stderr"), streams):
                        try:
                            stream.flush()
                            os.fsync(stream.fileno())
                        except BaseException as exc:
                            record_error(exc, phase + "_" + channel + "_sync")
        except _StopExecution:
            pass
        except BaseException as exc:
            record_error(exc, phase + "_streams")
            if phase not in processes:
                cleanup(process, phase)
        try:
            check_deadline(phase + "_after_cleanup_and_sync", reserve)
        except _StopExecution:
            pass
        persist(phase + "-exit.json", {
            "schema": "jolt-learning-executor-process-exit/v2", "phase": phase,
            "spawned": process is not None,
            "process_exited": process is not None and process.poll() is not None,
            "returncode": None if process is None else process.returncode,
            "timeout": timed_out, "wall_seconds": elapsed(), "errors": list(errors),
        })
        try:
            check_deadline(phase + "_after_exit_publication", reserve)
        except _StopExecution:
            pass
        return process

    try:
        declaration_path = Path(args.declaration)
        if declaration_path.stat().st_size > 1_048_576:
            raise ValueError("declaration exceeds the bounded executor reader")
        declaration = strict_json_loads(declaration_path.read_bytes())
        if (not isinstance(declaration, dict) or declaration.get("schema") != DECLARATION_SCHEMA
                or type(declaration.get("attempts")) is not int or declaration["attempts"] != 1
                or declaration.get("automatic_retry") is not False
                or declaration.get("automatic_extension") is not False):
            raise ValueError("a separately admitted v2 single-invocation declaration is required")
        allocation_identity = validate_v2_allocation_identity(declaration)
        seconds = declaration["budget"]["maximum_wall_seconds"]
        if type(seconds) is not int or seconds != MAXIMUM_WALL_SECONDS:
            raise ValueError("execution requires the unchanged 1,800-second aggregate backstop")
        declaration_identity = file_identity(declaration_path)
        required_receipt("invocation.json", {
            "schema": "jolt-learning-executor-invocation/v2", "attempts": 1,
            "whole_process_started_monotonic": whole_started,
            "maximum_wall_seconds": seconds, "declaration": declaration_identity,
            "declaration_schema": DECLARATION_SCHEMA,
            **allocation_identity,
            "physical_execution_started": False, "automatic_retry": False,
            "automatic_extension": False,
        })
        import execute_jolt_diagnostic as helper
        owner_identity = helper.prepare_process_tree()
        required_receipt("ownership.json", {
            "schema": "jolt-learning-executor-ownership/v2", "prepared": True,
            "wall_seconds": elapsed(), "linux_subreaper_and_pidfd_required": True,
        })
        worker = [sys.executable, str(script), worker_mode,
            "--config", str(args.config), "--declaration", str(args.declaration),
            "--checkout", str(args.checkout), "--godot", str(args.godot),
            "--output", str(output), "--whole-started", repr(whole_started)]
        process = run_child("worker", worker, VERIFICATION_RESERVE_SECONDS)
        if not errors and not timed_out and process is not None and process.returncode == 0 and cleanup_confirmed:
            verify_command = [sys.executable, str(script), inventory_mode, "--output", str(output)]
            checked = run_child("verification", verify_command, FINALIZATION_RESERVE_SECONDS)
            if not errors and not timed_out and checked is not None and checked.returncode == 0:
                stdout = lifecycle / "verification-stdout.log"
                if stdout.stat().st_size > MAX_VERIFIER_BYTES:
                    raise ValueError("inventory verdict exceeds its bounded reader")
                verifier = strict_json_loads(stdout.read_bytes(), max_bytes=MAX_VERIFIER_BYTES)
                if not isinstance(verifier, dict):
                    raise ValueError("inventory verdict must be an object")
                required_receipt("verification.json", {
                    "schema": "jolt-learning-executor-verification/v2", "inventory": verifier,
                    "wall_seconds": elapsed(),
                })
    except _StopExecution:
        pass
    except BaseException as exc:
        record_error(exc, "executor")
    finally:
        # Startup ownership errors are retained as unconfirmed, never silently
        # promoted to an empty tree. No process has been spawned on that path.
        if not processes:
            cleanup(None, "startup")

    stream_identities = {}
    for phase in ("worker", "verification"):
        for channel in ("stdout", "stderr"):
            name = f"{phase}-{channel}.log"
            path = lifecycle / name
            try:
                stream_identities[name] = file_identity(path) if path.is_file() else None
            except BaseException as exc:
                record_error(exc, phase + "_" + channel + "_identity")
                stream_identities[name] = None

    worker = processes.get("worker")
    complete = bool(not errors and not timed_out and worker is not None and worker.returncode == 0
        and verifier and all(verifier.get(key) is True for key in
            ("inventory_verified", "readback_verified", "readback_complete", "costs_verified"))
        and verifier.get("status") == "completed" and cleanup_confirmed
        and owned_remaining == 0 and elapsed() <= MAXIMUM_WALL_SECONDS)
    receipt = {
        "schema": "jolt-learning-executor/v2", "attempts": 1,
        "declaration": declaration_identity, "whole_process_started_monotonic": whole_started,
        "maximum_wall_seconds": MAXIMUM_WALL_SECONDS, "wall_seconds": elapsed(),
        "returncode": None if worker is None else worker.returncode,
        "timeout": timed_out, "errors": list(errors), "completion_candidate": complete,
        "cleanup_confirmed": cleanup_confirmed, "owned_processes_remaining": owned_remaining,
        "inventory": verifier, "streams": stream_identities, "stage_receipts": dict(receipts),
    }
    executor_identity = persist("executor.json", receipt)
    finalization_identity = persist("finalization.json", {
        "schema": "jolt-learning-executor-finalization/v2", "executor": executor_identity,
        "wall_seconds_after_executor_publication": elapsed(),
        "whole_process_started_monotonic": whole_started,
        "maximum_wall_seconds": MAXIMUM_WALL_SECONDS, "errors": list(errors),
    })
    candidate = complete and not errors and elapsed() <= MAXIMUM_WALL_SECONDS
    terminal_identity = persist("terminal.json", {
        "schema": "jolt-learning-executor-terminal-receipt/v2",
        "completion_candidate": candidate, "requires_observed_zero_exit": True,
        "requires_post_publication_stdout_verdict": True,
        "wall_seconds_before_terminal_publication": elapsed(),
        "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
        "whole_process_started_monotonic": whole_started,
        "executor": executor_identity, "finalization": finalization_identity,
        "cleanup_confirmed": cleanup_confirmed, "owned_processes_remaining": owned_remaining,
        "errors": list(errors),
    })
    terminal = {
        "schema": "jolt-learning-executor-terminal/v2", "completed": False,
        "wall_seconds": elapsed(), "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
        "whole_process_started_monotonic": whole_started,
        "terminal_receipt": terminal_identity, "executor": executor_identity,
        "finalization": finalization_identity, "cleanup_confirmed": cleanup_confirmed,
        "owned_processes_remaining": owned_remaining, "errors": list(errors),
    }
    terminal["completed"] = bool(candidate and not errors and terminal_identity is not None
                                 and terminal["wall_seconds"] <= MAXIMUM_WALL_SECONDS)
    try:
        print(json.dumps(terminal), flush=True)
    except BaseException as exc:
        record_error(exc, "terminal_stdout")
        persist("terminal-output-error.json", {
            "schema": "jolt-learning-executor-output-error/v2", "completed": False,
            "terminal_receipt": terminal_identity, "wall_seconds": elapsed(), "errors": list(errors),
        })
        return 2
    if elapsed() > MAXIMUM_WALL_SECONDS:
        persist("late-exit.json", {
            "schema": "jolt-learning-executor-late-exit/v2", "completed": False,
            "terminal_receipt": terminal_identity, "wall_seconds": elapsed(),
            "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
        })
        return 2
    return 0 if terminal["completed"] else 2


def verify_lifecycle(lifecycle, *, terminal_stdout, observed_exit_code, expected_declaration=None):
    """Read an original terminal/exit pair against all v2 lifecycle evidence.

    The caller supplies the *observed* CLI exit, never a return-code field copied
    from the worker. ``terminal_stdout`` is its original bytes or retained file.
    No process, model, checkpoint or original auditor is started. Negative CLI
    outcomes return an explicit nonacceptance after binding their terminal
    receipts; they do not claim a complete verification of partial stages.
    This verifies executor lifecycle only. The caller must separately verify
    the original worker output and match its declaration; ``expected_declaration``
    may supply that independently read identity directly.
    """
    lifecycle = Path(lifecycle)

    def require(condition, reason):
        if not condition:
            raise ContractError("learning_executor_lifecycle", reason)

    require(type(observed_exit_code) is int, "an independently observed integer CLI exit is required")
    require(not lifecycle.is_symlink() and lifecycle.is_dir(), "an original lifecycle directory is required")
    files = {}
    for path in lifecycle.iterdir():
        require(not path.is_symlink() and path.is_file(), "lifecycle contains a link, directory or special file")
        files[path.name] = file_identity(path)

    def read(name, schema):
        path = lifecycle / name
        require(name in files and path.stat().st_size <= 1_048_576, "missing or oversized lifecycle receipt")
        value = strict_json_loads(path.read_bytes(), max_bytes=1_048_576)
        require(isinstance(value, dict) and value.get("schema") == schema, "unknown lifecycle receipt schema")
        return value

    if isinstance(terminal_stdout, Path):
        require(not terminal_stdout.is_symlink() and terminal_stdout.is_file()
                and terminal_stdout.stat().st_size <= MAX_VERIFIER_BYTES, "a bounded original CLI stdout file is required")
        raw_stdout = terminal_stdout.read_bytes()
    else:
        require(isinstance(terminal_stdout, bytes) and len(terminal_stdout) <= MAX_VERIFIER_BYTES,
                "bounded original CLI stdout bytes are required")
        raw_stdout = terminal_stdout
    terminal = strict_json_loads(raw_stdout, max_bytes=MAX_VERIFIER_BYTES)
    require(isinstance(terminal, dict) and terminal.get("schema") == "jolt-learning-executor-terminal/v2"
            and type(terminal.get("completed")) is bool, "the original v2 post-publication verdict is required")
    durable = read("terminal.json", "jolt-learning-executor-terminal-receipt/v2")
    executor = read("executor.json", "jolt-learning-executor/v2")
    finalization = read("finalization.json", "jolt-learning-executor-finalization/v2")
    require(terminal.get("terminal_receipt") == files["terminal.json"], "stdout changed its durable terminal binding")
    require(terminal.get("executor") == durable.get("executor") == finalization.get("executor") == files["executor.json"]
            and terminal.get("finalization") == durable.get("finalization") == files["finalization.json"],
            "terminal or finalization changed its original executor binding")
    require(durable.get("requires_observed_zero_exit") is True
            and durable.get("requires_post_publication_stdout_verdict") is True,
            "a durable candidate alone cannot establish executor completion")
    require(expected_declaration is None or executor.get("declaration") == expected_declaration,
            "executor lifecycle belongs to another independently bound declaration")
    if observed_exit_code != 0 or terminal["completed"] is not True:
        return {"schema": "jolt-learning-executor-readback/v2", "verified": False, "completed": False,
                "scope": "executor_lifecycle_only",
                "terminal_bound": True, "observed_exit_code": observed_exit_code,
                "reason": "negative_or_unconfirmed_executor_exit"}

    stages = {
        "invocation.json": "jolt-learning-executor-invocation/v2",
        "ownership.json": "jolt-learning-executor-ownership/v2",
        "worker-started.json": "jolt-learning-executor-process-start/v2",
        "worker-cleanup.json": "jolt-learning-executor-cleanup/v2",
        "worker-exit.json": "jolt-learning-executor-process-exit/v2",
        "verification-started.json": "jolt-learning-executor-process-start/v2",
        "verification-cleanup.json": "jolt-learning-executor-cleanup/v2",
        "verification-exit.json": "jolt-learning-executor-process-exit/v2",
        "verification.json": "jolt-learning-executor-verification/v2",
    }
    streams = {f"{phase}-{channel}.log" for phase in ("worker", "verification")
               for channel in ("stdout", "stderr")}
    require(set(files) == set(stages) | streams | {"executor.json", "finalization.json", "terminal.json"},
            "complete lifecycle has missing or surplus original artifacts")
    expected_stages = {name: files[name] for name in stages}
    require(executor.get("stage_receipts") == expected_stages, "executor changed an immutable stage receipt")
    require(executor.get("streams") == {name: files[name] for name in streams},
            "executor changed its original worker or verification streams")
    values = {name: read(name, schema) for name, schema in stages.items()}
    invocation, ownership = values["invocation.json"], values["ownership.json"]
    origin = invocation.get("whole_process_started_monotonic")
    require(type(origin) in (int, float) and math.isfinite(origin) and origin > 0,
            "original monotonic origin is absent or invalid")
    for value in (invocation, executor, finalization, durable, terminal):
        require(type(value.get("maximum_wall_seconds")) is int
                and value["maximum_wall_seconds"] == MAXIMUM_WALL_SECONDS
                and value.get("whole_process_started_monotonic") == origin,
                "an executor receipt renewed or changed the original allocation")
    require(invocation.get("declaration_schema") == DECLARATION_SCHEMA
            and type(invocation.get("attempts")) is int and invocation["attempts"] == 1
            and type(executor.get("attempts")) is int and executor["attempts"] == 1
            and invocation.get("automatic_retry") is False and invocation.get("automatic_extension") is False
            and invocation.get("physical_execution_started") is False
            and executor.get("declaration") == invocation.get("declaration"),
            "the original v2 single-invocation declaration changed")
    validate_v2_allocation_identity(invocation)
    declaration_identity = invocation.get("declaration")
    require(isinstance(declaration_identity, dict) and set(declaration_identity) == {"bytes", "sha256"}
            and type(declaration_identity["bytes"]) is int and declaration_identity["bytes"] > 0
            and isinstance(declaration_identity["sha256"], str) and len(declaration_identity["sha256"]) == 64
            and set(declaration_identity["sha256"]) <= set("0123456789abcdef"), "invalid declaration identity")
    require(ownership.get("prepared") is True and ownership.get("linux_subreaper_and_pidfd_required") is True,
            "verified ownership was not established")
    ordered_walls = [ownership.get("wall_seconds")]
    for phase in ("worker", "verification"):
        start = values[phase + "-started.json"]
        cleanup = values[phase + "-cleanup.json"]
        exited = values[phase + "-exit.json"]
        require(all(value.get("phase") == phase for value in (start, cleanup, exited))
                and start.get("spawned") is True and start.get("whole_process_started_monotonic") == origin,
                "a process stage lost its original phase or invocation origin")
        require(exited.get("spawned") is True and exited.get("process_exited") is True
                and type(exited.get("returncode")) is int and exited["returncode"] == 0
                and exited.get("timeout") is False and exited.get("errors") == [],
                "an original child failed, timed out or has no observed exit")
        require(type(cleanup.get("returncode")) is int and cleanup["returncode"] == 0
                and cleanup.get("cleanup_confirmed") is True
                and type(cleanup.get("owned_processes_remaining")) is int
                and cleanup["owned_processes_remaining"] == 0 and cleanup.get("errors") == [],
                "an original process tree did not verifiably close")
        ordered_walls.extend(value.get("wall_seconds") for value in (start, cleanup, exited))
    ordered_walls.extend((values["verification.json"].get("wall_seconds"), executor.get("wall_seconds"),
        finalization.get("wall_seconds_after_executor_publication"),
        durable.get("wall_seconds_before_terminal_publication"), terminal.get("wall_seconds")))
    require(all(type(value) in (int, float) and math.isfinite(value)
                and 0 <= value <= MAXIMUM_WALL_SECONDS for value in ordered_walls)
            and ordered_walls == sorted(ordered_walls), "lifecycle times are nonfinite, reordered or exceed the original deadline")
    require(values["worker-exit.json"]["wall_seconds"] < MAXIMUM_WALL_SECONDS - VERIFICATION_RESERVE_SECONDS
            and values["verification-exit.json"]["wall_seconds"] < MAXIMUM_WALL_SECONDS - FINALIZATION_RESERVE_SECONDS,
            "child execution consumed the reserved finalization interval")
    require(executor.get("completion_candidate") is True and durable.get("completion_candidate") is True
            and type(executor.get("returncode")) is int and executor["returncode"] == 0
            and executor.get("timeout") is False
            and all(value.get("errors") == [] for value in (executor, finalization, durable, terminal)),
            "a failed publication or original executor verdict cannot be upgraded")
    for value in (executor, durable, terminal):
        require(value.get("cleanup_confirmed") is True
                and type(value.get("owned_processes_remaining")) is int and value["owned_processes_remaining"] == 0,
                "terminal lost its confirmed empty process tree")
    verification_bytes = (lifecycle / "verification-stdout.log").read_bytes()
    require(len(verification_bytes) <= MAX_VERIFIER_BYTES, "verification stdout exceeds its declared reader")
    inventory = strict_json_loads(verification_bytes, max_bytes=MAX_VERIFIER_BYTES)
    require(isinstance(inventory, dict) and inventory == executor.get("inventory")
            and inventory == values["verification.json"].get("inventory")
            and all(inventory.get(key) is True for key in
                    ("inventory_verified", "readback_verified", "readback_complete", "costs_verified"))
            and inventory.get("status") == "completed" and type(inventory.get("qualified")) is bool,
            "independent inventory, physical evidence or cost verification did not complete")
    require({path.name for path in lifecycle.iterdir()} == set(files)
            and all(not (lifecycle / name).is_symlink() and file_identity(lifecycle / name) == identity
                    for name, identity in files.items()), "lifecycle changed during independent readback")
    return {"schema": "jolt-learning-executor-readback/v2", "verified": True, "completed": True,
            "scope": "executor_lifecycle_only",
            "observed_exit_code": observed_exit_code, "terminal_bound": True,
            "worker_inventory": inventory, "wall_seconds": terminal["wall_seconds"],
            "maximum_wall_seconds": MAXIMUM_WALL_SECONDS, "declaration": declaration_identity,
            "whole_process_started_monotonic": origin,
            "terminal_receipt": files["terminal.json"], "artifacts": len(files)}
