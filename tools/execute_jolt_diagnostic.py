#!/usr/bin/env python3
"""Invoke the single frozen v4 diagnostic and account for its actual CLI exit.

The 900-second interval starts before Python child creation. Receipt hashing,
stream synchronization and immutable receipt publication are subsequent offline
accounting, not another physical invocation or an extension of its allocation.
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MAXIMUM_WALL_SECONDS = 900


def _process(pid):
    """Read a process identity from this executor's Linux PID namespace."""
    raw = Path(f"/proc/{pid}/stat").read_text()
    head, separator, tail = raw.rpartition(") ")
    fields = tail.split()
    if not separator or int(head.split(" ", 1)[0]) != pid or len(fields) < 20:
        raise RuntimeError("inconsistent Linux process identity")
    return {"parent": int(fields[1]), "start": int(fields[19]), "state": fields[0]}


def owned_processes(owner_start):
    """A subreaper retains ownership across worker sessions and parent exits."""
    owner = os.getpid()
    table = {}
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            try:
                table[int(path.name)] = _process(int(path.name))
            except FileNotFoundError:
                pass  # Exited during this scan; permission/format errors fail closed.
    if table.get(owner, {}).get("start") != owner_start:
        raise RuntimeError("executor and proc must share a consistent PID namespace")
    descendants = {owner}
    while True:
        added = {pid for pid, item in table.items() if item["parent"] in descendants} - descendants
        if not added:
            return {pid: table[pid] for pid in descendants if pid != owner}
        descendants.update(added)


def direct_children():
    """Non-reaping kernel query closes non-atomic proc-scan reparenting races.

    ECHILD proves absence; both a live child (None) and a waitable child (result)
    mean that ownership remains. WNOWAIT preserves Popen's original exit status.
    """
    try:
        os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return False
    return True


def prepare_process_tree():
    """Require Linux subreaping and PID-safe signalling before consuming a run."""
    if (not sys.platform.startswith("linux") or not hasattr(os, "pidfd_open")
            or not hasattr(signal, "pidfd_send_signal")
            or not all(hasattr(os, name) for name in ("waitid", "P_ALL", "WEXITED", "WNOHANG", "WNOWAIT"))):
        raise RuntimeError("verified Linux subreaper and pidfd support are required")
    if int(Path("/proc/self/stat").read_text().split(" ", 1)[0]) != os.getpid():
        raise RuntimeError("executor and proc must share a consistent PID namespace")
    libc = ctypes.CDLL(None, use_errno=True)
    # Linux prctl.h: PR_SET_CHILD_SUBREAPER=36, PR_GET_CHILD_SUBREAPER=37.
    enabled = ctypes.c_int()
    if (libc.prctl(36, 1, 0, 0, 0) != 0
            or libc.prctl(37, ctypes.byref(enabled), 0, 0, 0) != 0 or enabled.value != 1):
        raise OSError(ctypes.get_errno(), "cannot establish child subreaper")
    identity = _process(os.getpid())["start"]
    if owned_processes(identity) or direct_children():
        raise RuntimeError("executor must have no preexisting child processes")
    return identity


def _kill_identity(pid, identity):
    """Pin a PID before checking its start time; never signal a reused PID."""
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        try:
            current = _process(pid)
        except FileNotFoundError:
            return
        if current["start"] == identity["start"] and current["state"] != "Z":
            try:
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            except ProcessLookupError:
                pass
    finally:
        os.close(descriptor)


def terminate_owned_tree(process, owner_start, grace_seconds=3):
    """Kill and reap every owned descendant, regardless of its process group.

    Cleanup may run after the hard deadline only to revoke remaining processes;
    a timed-out attempt remains failed and its full elapsed time is retained.
    Explicit None waits for confirmed termination without a cleanup wall cap.
    The finite default preserves the historical diagnostic contract.
    """
    deadline = None if grace_seconds is None else time.monotonic() + grace_seconds
    while True:
        descendants = owned_processes(owner_start)
        for pid, identity in descendants.items():
            _kill_identity(pid, identity)
        if process is not None:
            process.poll()  # Popen, not waitpid below, owns its child's exit status.
        for pid, identity in descendants.items():
            if identity["parent"] == os.getpid() and (process is None or pid != process.pid):
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
        if (not owned_processes(owner_start) and (process is None or process.poll() is not None)
                and not direct_children()):
            return True
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(.01)


def _identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("an original regular artifact is required")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _original_completion(output, receipt, stdout_path):
    from nca.contracts import strict_json_loads

    primary_path = output.with_name(output.name + "-supervisor.json")
    final_path = output.with_name(output.name + "-finalization.json")
    receipt["supervisor_sha256"] = _identity(primary_path)["sha256"]
    receipt["finalization_sha256"] = _identity(final_path)["sha256"]
    primary = strict_json_loads(primary_path.read_bytes())
    final = strict_json_loads(final_path.read_bytes())
    # Only the final supervisor line is its original completion verdict.
    with stdout_path.open("rb") as stream:
        stream.seek(max(0, stream.seek(0, os.SEEK_END) - 65536))
        lines = stream.read().splitlines()
    terminal = strict_json_loads(lines[-1]) if lines else None
    receipt["terminal_report"] = terminal
    identities = ("source_revision", "declaration_sha256", "whole_process_started_monotonic")
    if (not isinstance(primary, dict) or not isinstance(final, dict) or not isinstance(terminal, dict)
            or primary.get("schema") != "nca-jolt-arrival-supervisor/v1"
            or final.get("schema") != "nca-jolt-arrival-finalization/v1"
            or any(value.get(key) != receipt[key] for value in (primary, final) for key in identities)
            or primary.get("attempts") != 1 or primary.get("returncode") != 0
            or primary.get("completed") is not True or primary.get("timeout") is not False
            or primary.get("worker_terminated") is not True
            or primary.get("owned_process_group_terminated") is not True
            or primary.get("final_output_verification", {}).get("verified") is not True
            or final.get("completed") is not True
            or final.get("supervisor_sha256") != receipt["supervisor_sha256"]
            or terminal.get("completed") is not True or terminal.get("timeout") is not False
            or terminal.get("returncode") != 0
            or terminal.get("finalization_sha256") != receipt["finalization_sha256"]):
        raise ValueError("original completion identities or verdict do not agree")
    walls = [value.get("wall_seconds") for value in (primary, final, terminal, receipt)]
    if (any(type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0 for wall in walls)
            or walls != sorted(walls) or walls[-1] > MAXIMUM_WALL_SECONDS):
        raise ValueError("original completion times do not fit the external invocation")


def execute(args, forwarded):
    from nca.evaluation.durable_publication import publish_json_v2
    from nca.evaluation.jolt_arrival import read_declaration

    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output) or any(path.name.startswith(output.name + "-") for path in output.parent.iterdir()):
        raise ValueError("diagnostic output or sibling already exists; no retry or overwrite")
    declaration = read_declaration(args.declaration, expected_sha256=args.expected_declaration_sha256)
    if (declaration["schema"] != "nca-jolt-recovery-diagnostic/v4"
            or declaration["budget"]["maximum_wall_seconds"] != MAXIMUM_WALL_SECONDS):
        raise ValueError("the frozen single 900-second v4 allocation is required")
    owner_start = prepare_process_tree()
    stdout_path = output.with_name(output.name + "-executor-stdout.log")
    stderr_path = output.with_name(output.name + "-executor-stderr.log")
    receipt = {"schema": "nca-jolt-arrival-executor/v1", "source_revision": args.expected_revision,
        "declaration_sha256": args.expected_declaration_sha256,
        "executor_source_sha256": _identity(Path(__file__).resolve())["sha256"],
        "maximum_wall_seconds": MAXIMUM_WALL_SECONDS, "attempts": 1,
        "returncode": None, "timeout": False, "completed": False, "error_type": None,
        "supervisor_terminated": False, "owned_process_tree_terminated": False,
        "supervisor_sha256": None, "finalization_sha256": None, "terminal_report": None,
        "measured_interval": "before Python supervisor spawn through observed CLI exit and any required owned-tree termination",
        "receipt_publication": "stream synchronization, artifact hashing and immutable executor receipt publication are post-execution offline accounting, excluded from the measured physical invocation"}
    process = None
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        started = time.monotonic()
        try:
            command = [sys.executable, str(ROOT / "tools/jolt_arrival.py"), *forwarded,
                "--whole-process-started", repr(started)]
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            remaining = MAXIMUM_WALL_SECONDS - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired("diagnostic-supervisor", MAXIMUM_WALL_SECONDS)
            receipt["returncode"] = process.wait(timeout=remaining)
            ended = time.monotonic()
            receipt["timeout"] = ended - started > MAXIMUM_WALL_SECONDS
        except BaseException as exc:
            ended = time.monotonic()
            receipt["timeout"] = isinstance(exc, subprocess.TimeoutExpired)
            receipt["error_type"] = type(exc).__name__
        try:
            if receipt["error_type"] is not None or owned_processes(owner_start) or direct_children():
                receipt["error_type"] = receipt["error_type"] or "OwnedProcessTreeDidNotExit"
                receipt["owned_process_tree_terminated"] = terminate_owned_tree(process, owner_start)
                ended = time.monotonic()
            else:
                receipt["owned_process_tree_terminated"] = True
        except BaseException as exc:
            receipt["error_type"] = receipt["error_type"] or type(exc).__name__
            receipt["cleanup_error_type"] = type(exc).__name__
            # A proc visibility failure cannot justify a closed-tree claim.
            if process is not None and process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
            ended = time.monotonic()
        receipt["supervisor_terminated"] = process is None or process.poll() is not None
        if process is not None:
            receipt["returncode"] = process.returncode
        receipt.update(whole_process_started_monotonic=started,
            process_exited_monotonic=ended if receipt["supervisor_terminated"] else None,
            wall_seconds=ended - started)
        receipt["timeout"] = receipt["timeout"] or receipt["wall_seconds"] > MAXIMUM_WALL_SECONDS
        for stream in (stdout, stderr):
            stream.flush()
            os.fsync(stream.fileno())
    receipt["streams"] = {path.name: _identity(path) for path in (stdout_path, stderr_path)}
    try:
        _original_completion(output, receipt, stdout_path)
    except Exception as exc:
        receipt["error_type"] = receipt["error_type"] or type(exc).__name__
    receipt["completed"] = (receipt["returncode"] == 0 and not receipt["timeout"]
        and receipt["error_type"] is None and receipt["supervisor_terminated"]
        and receipt["owned_process_tree_terminated"] is True)
    publish_json_v2(output.with_name(output.name + "-executor.json"), receipt)
    print(json.dumps({key: receipt[key] for key in ("completed", "wall_seconds", "timeout", "returncode")}))
    return 0 if receipt["completed"] else 2


def main(argv=None):
    forwarded = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--expected-declaration-sha256", required=True)
    args, _ = parser.parse_known_args(forwarded)
    # The inner parser accepts abbreviations: reject those for its private modes
    # too. All other arguments retain their original order and exact token bytes.
    private = ("--worker", "--verify-only", "--whole-process-started")
    if any(token.startswith("--") and any(flag.startswith(token.split("=", 1)[0]) for flag in private)
            for token in forwarded):
        parser.error("internal supervisor modes and invocation origins cannot be supplied")
    try:
        return execute(args, forwarded)
    except Exception as exc:
        print(json.dumps({"completed": False, "error_type": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
