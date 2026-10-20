#!/usr/bin/env python3
"""Run the single fixed Jolt teacher diagnostic from a published source freeze."""

import argparse
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
    entered = time.monotonic()
    from nca.evaluation.jolt_arrival import read_declaration, write_json
    declaration = read_declaration(args.declaration, expected_sha256=args.expected_declaration_sha256)
    limits = declaration["budget"]
    maximum_wall_seconds = limits["maximum_wall_seconds"]
    version4 = declaration["schema"] == "nca-jolt-recovery-diagnostic/v4"
    cleanup_reserve = limits["shutdown_reserve_seconds"] if version4 else 6
    verification_reserve = limits["independent_verification_reserve_seconds"] if version4 else 6
    finalization_reserve = limits["supervisor_finalization_reserve_seconds"] if version4 else 6
    supplied_origin = getattr(args, "whole_process_started", None)
    if version4 and supplied_origin is not None and (
            type(supplied_origin) not in (int, float) or not math.isfinite(supplied_origin)
            or not 0 < supplied_origin <= entered):
        raise ValueError("the external executor's finite original monotonic start is required")
    if args.output.exists():
        raise ValueError("diagnostic output already exists; a previous invocation cannot be replaced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output.with_name(args.output.name + "-supervisor.json")
    stdout_path = args.output.with_name(args.output.name + "-stdout.log")
    stderr_path = args.output.with_name(args.output.name + "-stderr.log")
    verification_stdout = args.output.with_name(args.output.name + "-verification-stdout.log")
    verification_stderr = args.output.with_name(args.output.name + "-verification-stderr.log")
    finalization_path = args.output.with_name(args.output.name + "-finalization.json")
    reserved_paths = (receipt_path, stdout_path, stderr_path, verification_stdout, verification_stderr)
    if version4:
        reserved_paths += (finalization_path,)
    if any(path.exists() for path in reserved_paths):
        raise ValueError("supervision output already exists; automatic retry is not supported")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--host", str(args.host),
        "--godot", str(args.godot), "--output", str(args.output), "--expected-revision", args.expected_revision,
        "--declaration", str(args.declaration), "--expected-declaration-sha256", args.expected_declaration_sha256]
    settings = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                else {"start_new_session": True})
    receipt = {"schema": "nca-jolt-arrival-supervisor/v1", "source_revision": args.expected_revision,
        "declaration_sha256": args.expected_declaration_sha256,
        "maximum_wall_seconds": maximum_wall_seconds, "attempts": 1, "timeout": False, "returncode": None,
        "worker_terminated": False, "owned_process_group_terminated": None, "error_type": None,
        "final_output_verification": {"verified": False},
        "cleanup_reserve_seconds": cleanup_reserve,
        "measured_interval": "worker spawn through exit and independent final artifact/journal replay, including source admission and host startup"}
    process = None
    started = (supplied_origin if supplied_origin is not None else entered) if version4 else time.monotonic()
    if version4:
        command.extend(["--whole-process-started", repr(started)])
        receipt.update(independent_verification_reserve_seconds=verification_reserve,
            supervisor_finalization_reserve_seconds=finalization_reserve,
            whole_process_started_monotonic=started,
            active_work_maximum_seconds=maximum_wall_seconds - cleanup_reserve,
            worker_maximum_seconds=maximum_wall_seconds - verification_reserve,
            measured_interval="shared invocation origin through startup, worker cleanup/sealing and independent final artifact/native replay; all reserves inside the original wall ceiling")
        from nca.evaluation.durable_publication import publish_json_v2
        write_json = publish_json_v2
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, **settings)
            try:
                receipt["returncode"] = process.wait(timeout=max(.001,
                    maximum_wall_seconds - verification_reserve - (time.monotonic() - started)))
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
        receipt["worker_terminated"] = process is None or process.poll() is not None
        if receipt["returncode"] == 0 and not receipt["timeout"] and receipt["error_type"] is None:
            try:
                remaining = maximum_wall_seconds - finalization_reserve - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired("offline-verification", 0)
                # The independent audit also runs under the original wall
                # allocation. It starts no host and cannot trigger a retry.
                verification_command = [sys.executable, str(Path(__file__).resolve()), "--verify-only",
                    "--output", str(args.output), "--expected-revision", args.expected_revision,
                    "--expected-declaration-sha256", args.expected_declaration_sha256]
                with verification_stdout.open("xb") as stdout, verification_stderr.open("xb") as stderr:
                    try:
                        verified = subprocess.run(verification_command, cwd=ROOT, stdout=stdout, stderr=stderr,
                            timeout=remaining, check=False)
                    finally:
                        stdout.flush()
                        stderr.flush()
                        os.fsync(stdout.fileno())
                        os.fsync(stderr.fileno())
                if verified.returncode != 0:
                    raise ValueError("independent offline verification failed")
                receipt["final_output_verification"] = json.loads(verification_stdout.read_text())
                if receipt["final_output_verification"].get("verified") is not True:
                    raise ValueError("independent offline verification did not confirm evidence")
            except subprocess.TimeoutExpired:
                receipt["timeout"] = True
                receipt["error_type"] = "OfflineVerificationTimeout"
                receipt["final_output_verification"] = {"verified": False, "error_type": "OfflineVerificationTimeout"}
            except Exception as exc:
                receipt["error_type"] = type(exc).__name__
                receipt["final_output_verification"] = {"verified": False, "error_type": type(exc).__name__,
                    "error_code": getattr(exc, "code", None)}
        receipt["streams"] = {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                         "bytes": path.stat().st_size}
                              for path in (stdout_path, stderr_path, verification_stdout, verification_stderr) if path.is_file()}
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["completed"] = (receipt["returncode"] == 0 and receipt["worker_terminated"]
            and receipt["owned_process_group_terminated"] is True
            and receipt["final_output_verification"]["verified"]
            and not receipt["timeout"] and receipt["error_type"] is None
            and receipt["wall_seconds"] <= maximum_wall_seconds)
        write_json(receipt_path, receipt)
        if version4:
            published_wall = time.monotonic() - started
            finalization = {"schema": "nca-jolt-arrival-finalization/v1",
                "source_revision": args.expected_revision,
                "declaration_sha256": args.expected_declaration_sha256,
                "supervisor_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                "whole_process_started_monotonic": started,
                "wall_seconds": published_wall,
                "completed": receipt["completed"] and published_wall <= maximum_wall_seconds,
                "measurement": "shared invocation origin through durable original supervisor receipt publication; terminal marker write and final CLI exit are separately checked by the invoking executor"}
            write_json(finalization_path, finalization)
    reported = {key: receipt[key] for key in ("completed", "wall_seconds", "timeout", "returncode")}
    if version4:
        # The terminal marker cannot include its own later write. Check again
        # before returning and require the invoking executor's exit/duration
        # receipt for full qualification; never rewrite either original file.
        reported["wall_seconds"] = time.monotonic() - started
        reported["completed"] = finalization["completed"] and reported["wall_seconds"] <= maximum_wall_seconds
        reported["timeout"] = receipt["timeout"] or reported["wall_seconds"] > maximum_wall_seconds
        reported["finalization_sha256"] = hashlib.sha256(finalization_path.read_bytes()).hexdigest()
    print(json.dumps(reported))
    return 0 if reported["completed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, default=os.environ.get("NCA_HOST_CHECKOUT"))
    parser.add_argument("--godot", type=Path, default=os.environ.get("NCA_GODOT"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--declaration", type=Path,
        default=ROOT / "configs/baselines/jolt-arrival-diagnostic-v1.json")
    parser.add_argument("--expected-declaration-sha256",
        default="269cef7a852fe8d690bb95ab9187576a3babe0f1fa5d1e8ee28fafc2730c474a")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verify-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--whole-process-started", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_only:
        from nca.evaluation.jolt_arrival import verify_diagnostic_output
        result = verify_diagnostic_output(args.output, expected_revision=args.expected_revision,
            expected_declaration_sha256=args.expected_declaration_sha256)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    if not args.host or not args.godot:
        parser.error("configure NCA_HOST_CHECKOUT and NCA_GODOT or supply --host and --godot")
    if not args.worker:
        return supervise(args)
    from nca.evaluation.jolt_arrival import run_diagnostic
    result = run_diagnostic(root=ROOT, checkout=args.host, godot=args.godot, output=args.output,
        declaration_path=args.declaration, expected_revision=args.expected_revision,
        expected_declaration_sha256=args.expected_declaration_sha256,
        whole_process_started=args.whole_process_started)
    print(result["status"])
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
