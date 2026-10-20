#!/usr/bin/env python3
"""Run the single fixed Jolt teacher diagnostic from a published source freeze."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def process_group_exists(group_id):
    """A missing leader does not establish that its owned descendants exited."""
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    return True


def terminate_owned_tree(process, grace_seconds=3):
    """Revoke the whole owned group, even after its worker leader has exited."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=grace_seconds)
        # taskkill cannot prove descendant absence after an already-exited PID.
        return None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                pass
    process.wait(timeout=grace_seconds)
    deadline = time.monotonic() + grace_seconds
    while process_group_exists(process.pid) and time.monotonic() < deadline:
        time.sleep(.02)
    return not process_group_exists(process.pid)


def supervise(args):
    """One owned worker/process tree; its output finalization is inside the cap."""
    if args.output.exists():
        raise ValueError("diagnostic output already exists; a previous invocation cannot be replaced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output.with_name(args.output.name + "-supervisor.json")
    stdout_path = args.output.with_name(args.output.name + "-stdout.log")
    stderr_path = args.output.with_name(args.output.name + "-stderr.log")
    if any(path.exists() for path in (receipt_path, stdout_path, stderr_path)):
        raise ValueError("supervision output already exists; automatic retry is not supported")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--host", str(args.host),
        "--godot", str(args.godot), "--output", str(args.output), "--expected-revision", args.expected_revision]
    settings = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                else {"start_new_session": True})
    receipt = {"schema": "nca-jolt-arrival-supervisor/v1", "source_revision": args.expected_revision,
        "maximum_wall_seconds": 600, "attempts": 1, "timeout": False, "returncode": None,
        "worker_terminated": False, "owned_process_group_terminated": None, "error_type": None,
        "measured_interval": "worker spawn through exit, including source admission, host startup and evidence finalization"}
    process = None
    started = time.monotonic()
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, **settings)
            try:
                receipt["returncode"] = process.wait(timeout=max(.001, 600 - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                receipt["timeout"] = True
                receipt["owned_process_group_terminated"] = terminate_owned_tree(process)
                receipt["returncode"] = process.returncode
            finally:
                stdout.flush()
                stderr.flush()
                os.fsync(stdout.fileno())
                os.fsync(stderr.fileno())
    except BaseException as exc:
        receipt["error_type"] = type(exc).__name__
        if process is not None:
            try:
                receipt["owned_process_group_terminated"] = terminate_owned_tree(process)
            except (OSError, subprocess.SubprocessError) as cleanup:
                receipt["cleanup_error_type"] = type(cleanup).__name__
    finally:
        if process is not None and os.name != "nt":
            if process_group_exists(process.pid):
                receipt["error_type"] = receipt["error_type"] or "OwnedProcessGroupDidNotExit"
                try:
                    terminate_owned_tree(process)
                except (OSError, subprocess.SubprocessError) as cleanup:
                    receipt["cleanup_error_type"] = type(cleanup).__name__
            receipt["owned_process_group_terminated"] = not process_group_exists(process.pid)
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["worker_terminated"] = process is None or process.poll() is not None
        receipt["streams"] = {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                         "bytes": path.stat().st_size}
                              for path in (stdout_path, stderr_path) if path.is_file()}
        receipt["completed"] = (receipt["returncode"] == 0 and receipt["worker_terminated"]
            and receipt["owned_process_group_terminated"] is True
            and not receipt["timeout"] and receipt["error_type"] is None and receipt["wall_seconds"] <= 600)
        with receipt_path.open("x") as target:
            json.dump(receipt, target, sort_keys=True, allow_nan=False)
            target.write("\n")
    print(json.dumps({key: receipt[key] for key in ("completed", "wall_seconds", "timeout", "returncode")}))
    return 0 if receipt["completed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, default=os.environ.get("NCA_HOST_CHECKOUT"))
    parser.add_argument("--godot", type=Path, default=os.environ.get("NCA_GODOT"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.host or not args.godot:
        parser.error("configure NCA_HOST_CHECKOUT and NCA_GODOT or supply --host and --godot")
    if not args.worker:
        return supervise(args)
    from nca.evaluation.jolt_arrival import run_diagnostic
    result = run_diagnostic(root=ROOT, checkout=args.host, godot=args.godot, output=args.output,
        declaration_path=ROOT / "configs/baselines/jolt-arrival-diagnostic-v1.json",
        expected_revision=args.expected_revision)
    print(result["status"])
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
