"""Standard-library supervisor for one immutable, bounded IK allocation.

The optional runtime runs in a separate child. Startup, failed attempts and
checkpoint publication consume the same allocation. No command retries a
failed batch, changes a batch size, or silently chooses a different device.
"""

from __future__ import annotations

import time
ENTRY_STARTED = time.monotonic()

import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

from nca.contracts import canonical_json, sha256_json, strict_json_loads
from nca.randomness import ScenarioSplit
from nca.runs import (ArtifactRef, RunBudget, RunManifest, RunStore, immutable_bytes,
                      immutable_json, sha256_file, utc_now)


FINALIZATION_RESERVE = 5.0


def read_json(path: Path) -> dict:
    if path.stat().st_size > 1_048_576:
        raise ValueError("IK JSON receipt exceeds 1 MiB")
    value = strict_json_loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("IK receipt must be an object")
    return value


def _relative(root: Path, name: str) -> Path:
    if not isinstance(name, str) or "\\" in name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("invalid artifact path")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact escapes allocation")
    return path


def _attempt_id(index: int) -> str:
    return f"attempt-{index:04d}"


def _step_inventory(attempt: Path, committed: int, batch_size: int):
    reservations = sorted(attempt.glob("reserve-*.json"))
    steps = sorted(attempt.glob("step-*.json"))
    if [path.name for path in reservations] != [f"reserve-{i:06d}.json" for i in range(1, len(reservations) + 1)]:
        raise ValueError("optimizer reservation sequence is incomplete")
    if [path.name for path in steps] != [f"step-{i:06d}.json" for i in range(1, len(steps) + 1)]:
        raise ValueError("checkpoint boundary sequence is incomplete")
    if len(steps) > len(reservations) or len(reservations) - len(steps) > 1:
        raise ValueError("optimizer reservations and boundaries differ")
    last = None
    for index, path in enumerate(reservations, 1):
        reservation = read_json(path)
        if reservation != {"attempted_step": index, "samples": batch_size,
                           "next_committed_step": committed + index}:
            raise ValueError("optimizer reservation identity differs")
        if index <= len(steps):
            receipt = read_json(steps[index - 1])
            if (receipt.get("reservation_sha256") != sha256_file(path)
                    or receipt.get("committed_steps") != committed + index
                    or receipt.get("cursor") != (committed + index) * batch_size
                    or receipt.get("boundary") != "optimizer_step"):
                raise ValueError("optimizer boundary identity differs")
            result = receipt.get("result", {})
            loss = result.get("loss")
            if (type(loss) not in (int, float) or not math.isfinite(loss)
                    or result.get("finite_gradients") is not True or result.get("weights_changed") is not True):
                raise ValueError("optimizer boundary lacks finite changed-weight evidence")
            placement = result.get("placement", {})
            categories = {"data", "inputs", "model", "output", "geometry", "loss", "gradients", "optimizer"}
            counts = placement.get("tensor_counts", {})
            if (placement.get("all_on_selected_device") is not True or placement.get("finite_checked") is not True
                    or set(counts) != categories or any(type(v) is not int or v < 1 for v in counts.values())):
                raise ValueError("optimizer boundary lacks verified placement")
            request = read_json(attempt / "request.json")
            if placement.get("device") != request["device"]:
                raise ValueError("optimizer boundary placement differs from request")
            reference = receipt.get("checkpoint", {})
            filename = reference.get("file")
            if not isinstance(filename, str) or Path(filename).name != filename or "\\" in filename:
                raise ValueError("checkpoint reference filename differs")
            checkpoint = attempt / "checkpoints" / filename
            marker = read_json(checkpoint.with_name(filename + ".json"))
            if (marker.get("checkpoint") != reference or marker.get("record_sha256") != sha256_json(reference)
                    or checkpoint.stat().st_size != reference.get("bytes")
                    or sha256_file(checkpoint) != reference.get("sha256")):
                raise ValueError("optimizer boundary checkpoint bytes differ")
            last = receipt
    return len(reservations), len(steps), last


def inspect_run(run: str | Path) -> dict[str, Any]:
    """Verify immutable records and account all attempts without importing Torch."""
    from .declaration import load_declaration
    root = Path(run)
    declaration = load_declaration(root)
    config = declaration["config"]
    declaration_sha = sha256_file(root / "declaration.json")
    attempts = sorted((root / "attempts").glob("attempt-*")) if (root / "attempts").exists() else []
    if [path.name for path in attempts] != [_attempt_id(i) for i in range(1, len(attempts) + 1)]:
        raise ValueError("attempt history is incomplete")
    charged_seconds = 0.0
    charged_steps = 0
    committed = 0
    checkpoint = None
    environment_sha = None
    parent_record_sha = None
    records = []
    lineage_limits = []
    unresolved = False
    for attempt in attempts:
        request_path = attempt / "request.json"
        if not request_path.is_file():
            unresolved = True
            break
        request = read_json(request_path)
        if (request.get("declaration_sha256") != declaration_sha
                or request.get("prior_charged_steps") != charged_steps
                or request.get("prior_committed_steps") != committed
                or request.get("prior_charged_seconds") != charged_seconds
                or request.get("parent_record_sha256") != parent_record_sha
                or request.get("checkpoint") != checkpoint):
            raise ValueError("allocation lineage or counters changed")
        record_path = root / "records" / f"{attempt.name}.json"
        seal_path = attempt / "sealed.json"
        if not record_path.is_file() or not seal_path.is_file():
            unresolved = True
            break
        record = RunStore(root / "records").read(attempt.name)
        seal = read_json(seal_path)
        if (seal.get("record_sha256") != record.sha256
                or record.metrics.get("request_sha256") != sha256_file(request_path)
                or record.metrics.get("declaration_sha256") != declaration_sha):
            raise ValueError("attempt seal differs")
        listed = set()
        for ref in record.artifacts:
            artifact = _relative(root, ref.uri)
            listed.add(artifact)
            if artifact.stat().st_size != ref.size_bytes or sha256_file(artifact) != ref.content_sha256:
                raise ValueError("attempt artifact bytes differ")
        actual = {p for p in attempt.rglob("*") if p.is_file() and p.name != "sealed.json"}
        if listed != actual:
            raise ValueError("attempt inventory differs")
        attempted, completed, last = _step_inventory(attempt, committed, config["batch_size"])
        if (record.metrics["attempted_steps"] != attempted
                or record.metrics["committed_steps"] != committed + completed
                or record.decisions != attempted):
            raise ValueError("attempt counters differ from receipts")
        qualification = record.metrics.get("qualification", "unmeasured")
        if qualification != "unmeasured" and (record.status != "completed"
                or committed + completed != config["steps"] or attempted != completed
                or lineage_limits or record.metrics.get("worker", {}).get("reproducibility_limits")
                or qualification != ("cpu_smoke_passed" if request["device"] == "cpu" else "cuda_smoke_passed")):
            raise ValueError("qualification is not supported by complete step evidence")
        charged_steps += attempted
        committed += completed
        charged_seconds += record.runtime_seconds
        if last is not None:
            checkpoint = {"path": f"attempts/{attempt.name}/checkpoints/{last['checkpoint']['file']}",
                          "reference": last["checkpoint"]}
            environment_sha = last["checkpoint"]["metadata"]["environment_sha256"]
        elif checkpoint is None and (attempt / "initial.json").is_file():
            initial = read_json(attempt / "initial.json")
            checkpoint = {"path": f"attempts/{attempt.name}/checkpoints/{initial['file']}",
                          "reference": initial}
            environment_sha = initial["metadata"]["environment_sha256"]
        if (record.environment.get("environment_sha256") is not None and
                (environment_sha is None or record.metrics.get("worker", {}).get("environment_admitted") is True)):
            environment_sha = record.environment["environment_sha256"]
        parent_record_sha = record.sha256
        records.append(record.to_dict())
        lineage_limits.extend(record.metrics.get("worker", {}).get("reproducibility_limits", []))
    if unresolved or (root / "active.json").exists():
        # Unknown consumed work is never converted to a free retry.
        remaining_seconds = 0.0
    else:
        remaining_seconds = max(0.0, config["max_wall_seconds"] - charged_seconds)
    return {"schema": "nca-ik-allocation-state/v1", "run_id": declaration["run_id"],
            "declaration_sha256": declaration_sha, "device": declaration["device"],
            "attempts": records, "attempt_count": len(attempts), "unresolved_attempt": unresolved,
            "active": (root / "active.json").exists(), "charged_seconds": charged_seconds,
            "charged_steps": charged_steps, "charged_samples": charged_steps * config["batch_size"],
            "committed_steps": committed, "cursor": committed * config["batch_size"],
            "remaining_seconds": remaining_seconds,
            "remaining_steps": max(0, config["max_optimizer_steps"] - charged_steps),
            "checkpoint": checkpoint, "environment_sha256": environment_sha,
            "lineage_limits": sorted(set(lineage_limits)),
            "parent_record_sha256": parent_record_sha,
            "qualification": records[-1]["metrics"].get("qualification", "unmeasured") if records else "unmeasured"}


def cancel_run(run: str | Path) -> dict:
    root = Path(run)
    active = read_json(root / "active.json")
    name = active.get("attempt")
    if not isinstance(name, str) or not name.startswith("attempt-"):
        raise ValueError("invalid active attempt")
    path = _relative(root, f"cancel-{name}.json")
    request = {"attempt": name, "requested_utc": utc_now()}
    if not path.exists():
        immutable_json(path, request)
    return {"cancellation_requested": name, "qualification": "unmeasured"}


def _stop_worker(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=1.0)


def execute(run: str | Path, expected_sha256: str, *, resume: bool = False,
            stop_after: int | None = None, allow_environment_change: bool = False,
            device: str | None = None, started: float | None = None) -> dict:
    """Run once, recording failures; resume keeps the original allocation."""
    from .declaration import load_declaration, verify_source
    from .device import parse_device
    started = time.monotonic() if started is None else started
    root = Path(run).resolve()
    declaration = load_declaration(root, expected_sha256)
    config = declaration["config"]
    state = inspect_run(root)
    if state["active"] or state["unresolved_attempt"]:
        raise ValueError("prior attempt is active or unsealed; allocation cannot be retried")
    if bool(state["attempt_count"]) != resume:
        raise ValueError("use smoke once, then resume the same allocation")
    if (state["committed_steps"] >= config["steps"] or state["remaining_steps"] <= 0
            or state["remaining_seconds"] <= FINALIZATION_RESERVE
            or state["charged_samples"] + config["batch_size"] > config["max_samples"]):
        # Read-only refusal: repeated invocations cannot create a new attempt or
        # manufacture allocation beyond the already immutable terminal record.
        return {**state, "admission_refused": "completed_or_exhausted_allocation"}
    if stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after <= config["steps"]):
        raise ValueError("stop-after must be within the declared step count")
    selected = declaration["device"] if device is None else device
    parse_device(selected)
    if selected != declaration["device"] and not allow_environment_change:
        raise ValueError("device changes require an explicit environment continuation")
    index = state["attempt_count"] + 1
    name = _attempt_id(index)
    active = root / "active.json"
    immutable_json(active, {"attempt": name, "declaration_sha256": expected_sha256})
    attempt = root / "attempts" / name
    attempt.mkdir(parents=True)
    remaining = state["remaining_seconds"]
    request = {"schema": "nca-ik-attempt/v1", "attempt": name,
               "declaration_sha256": expected_sha256, "device": selected,
               "stop_after": stop_after, "allow_environment_change": allow_environment_change,
               "prior_charged_steps": state["charged_steps"],
               "prior_committed_steps": state["committed_steps"],
               "prior_charged_seconds": state["charged_seconds"],
               "parent_record_sha256": state["parent_record_sha256"],
               "checkpoint": state["checkpoint"], "environment_sha256": state["environment_sha256"],
               "lineage_limits": state["lineage_limits"],
               "started_utc": utc_now(), "attempt_started_monotonic": started,
               "worker_deadline_monotonic": started + max(0, remaining - FINALIZATION_RESERVE)}
    immutable_json(attempt / "request.json", request)
    terminal = {"status": "failed", "reason": "supervisor_failure", "qualification": "unmeasured"}
    process = None
    finalized = False
    result = None
    try:
        if (state["remaining_steps"] <= 0 or remaining <= FINALIZATION_RESERVE
                or state["charged_samples"] + config["batch_size"] > config["max_samples"]):
            terminal.update(status="capped", reason="allocation_exhausted")
        elif state["committed_steps"] >= config["steps"]:
            terminal.update(status="capped", reason="declared_workload_already_completed")
        else:
            verify_source(declaration)
            if time.monotonic() >= request["worker_deadline_monotonic"]:
                terminal.update(status="capped", reason="startup_allocation_exhausted")
            else:
                log = attempt / "worker.log"
                with log.open("xb") as stream:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "nca.ik.runtime", "_worker", "--run", str(root),
                         "--attempt", name], stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                        start_new_session=os.name == "posix")
                    timed_out = canceled = False
                    cancellation_started = None
                    while process.poll() is None:
                        if time.monotonic() >= request["worker_deadline_monotonic"]:
                            timed_out = True
                            _stop_worker(process)
                            break
                        if (root / f"cancel-{name}.json").exists():
                            canceled = True
                            if cancellation_started is None:
                                cancellation_started = time.monotonic()
                            elif time.monotonic() - cancellation_started >= 0.5:
                                _stop_worker(process)
                                break
                        try:
                            process.wait(timeout=0.05)
                        except subprocess.TimeoutExpired:
                            pass
                    canceled |= (root / f"cancel-{name}.json").exists()
                    if (attempt / "worker.json").is_file():
                        terminal = read_json(attempt / "worker.json")
                    else:
                        terminal = {"status": "failed", "reason": "worker_exited_without_result",
                                    "qualification": "unmeasured"}
                    if timed_out:
                        terminal.update(status="capped", reason="wall_allocation_exhausted", qualification="unmeasured")
                    elif canceled:
                        terminal.update(status="canceled", reason="operator_cancellation", qualification="unmeasured")
                    elif process.returncode != 0 and terminal.get("status") == "completed":
                        terminal.update(status="failed", reason="worker_exit_mismatch", qualification="unmeasured")
        terminal["worker_exit_code"] = None if process is None else process.returncode
    except KeyboardInterrupt:
        cancel_run(root)
        if process is not None:
            _stop_worker(process)
        terminal.update(status="canceled", reason="operator_interrupt", qualification="unmeasured")
    except Exception as exc:
        if process is not None:
            _stop_worker(process)
        # Never publish arbitrary exception strings containing local paths.
        terminal.update(status="failed", reason=getattr(exc, "code", "admission_or_supervisor_failure"),
                        exception_type=type(exc).__name__, qualification="unmeasured")
    finally:
        if process is not None and process.poll() is None:
            _stop_worker(process)
        try:
            attempted, completed, last = _step_inventory(attempt, state["committed_steps"], config["batch_size"])
            target = config["steps"] if stop_after is None else min(config["steps"], state["committed_steps"] + stop_after)
            if terminal.get("status") == "completed" and (state["committed_steps"] + completed != target
                    or attempted != completed):
                terminal.update(status="failed", reason="completion_evidence_incomplete", qualification="unmeasured")
            if terminal.get("qualification", "unmeasured") != "unmeasured" and (
                    terminal.get("status") != "completed" or state["committed_steps"] + completed != config["steps"]
                    or state["lineage_limits"] or terminal.get("reproducibility_limits")
                    or terminal["qualification"] != ("cpu_smoke_passed" if selected == "cpu" else "cuda_smoke_passed")):
                terminal.update(status="failed", reason="qualification_evidence_incomplete", qualification="unmeasured")
            elapsed = time.monotonic() - started
            # Charge up to the sealing reserve within the remaining allocation;
            # observed elapsed work remains charged even after an overrun.
            charged = max(elapsed, min(remaining, elapsed + FINALIZATION_RESERVE))
            if elapsed > remaining:
                terminal.update(status="capped", reason="wall_allocation_exhausted", qualification="unmeasured")
            if terminal.get("status") not in ("completed", "canceled", "failed", "capped", "inconclusive"):
                terminal.update(status="failed", reason="invalid_worker_status", qualification="unmeasured")
            immutable_json(attempt / "supervisor.json", {**terminal,
                           "elapsed_through_worker_and_inventory_seconds": elapsed,
                           "finalization_reserve_charged_seconds": charged - elapsed,
                           "charged_seconds": charged})
            refs = tuple(ArtifactRef(uri=p.relative_to(root).as_posix(), content_sha256=sha256_file(p),
                                     size_bytes=p.stat().st_size, kind="ik-evidence")
                         for p in sorted(attempt.rglob("*")) if p.is_file())
            record = RunManifest(
                run_id=name, experiment="ik-device-smoke-v1", learner_revision=declaration["source"]["revision"],
                host_revision=declaration["host_revision"], profile_sha256=declaration["profile_sha256"],
                resolved_config=config, environment=terminal.get("environment", declaration["runtime"]),
                seeds=(config["seed"],), scenario_split=ScenarioSplit(("runner-smoke-lattice",), (),
                    ("conformance-only-no-quality-holdout",)), status=terminal["status"], device=selected,
                observation_schema="ik-runtime-smoke-input-v1", action_schema="ik-runtime-smoke-output-v1",
                suite="ik-device-smoke-v1", budget=RunBudget(config["max_optimizer_steps"], config["max_wall_seconds"]),
                decisions=attempted, runtime_seconds=charged, artifacts=refs,
                failures=() if terminal["status"] == "completed" else (terminal.get("reason", "incomplete"),),
                metrics={"declaration_sha256": expected_sha256, "request_sha256": sha256_file(attempt / "request.json"),
                         "parent_record_sha256": state["parent_record_sha256"],
                         "continuation_boundary": "optimizer_step" if resume else "initialization",
                         "attempted_steps": attempted, "committed_steps": state["committed_steps"] + completed,
                         "cumulative_attempted_steps": state["charged_steps"] + attempted,
                         "cumulative_charged_seconds": state["charged_seconds"] + charged,
                         "qualification": terminal.get("qualification", "unmeasured"),
                         "worker": terminal, "quality_qualification": False})
            RunStore(root / "records").write(record)
            if time.monotonic() > started + charged:
                # No seal: subsequent resume refuses unknown finalization cost.
                raise RuntimeError("finalization exceeded its charged reserve")
            immutable_json(attempt / "sealed.json", {"record_sha256": record.sha256,
                           "observed_seconds_before_seal": time.monotonic() - started,
                           "charged_seconds": charged})
            result = inspect_run(root)
            if time.monotonic() > started + charged:
                raise RuntimeError("seal/readback exceeded its charged reserve")
            finalized = True
        finally:
            if finalized:
                active.unlink()
                if time.monotonic() > started + charged:
                    immutable_json(active, {"attempt": name, "declaration_sha256": expected_sha256})
                    raise RuntimeError("finalization exceeded its charged reserve")
    result.update(active=False, remaining_seconds=max(0.0, config["max_wall_seconds"] - result["charged_seconds"]))
    return result


def compare_runs(cpu_run: str | Path, cuda_run: str | Path) -> dict:
    from .declaration import load_declaration
    cpu, cuda = (inspect_run(path) for path in (cpu_run, cuda_run))
    left, right = (load_declaration(path) for path in (cpu_run, cuda_run))
    for key in ("config_sha256", "profile_sha256", "dataset"):
        if left[key] != right[key]:
            raise ValueError("comparison workloads differ")
    if left["source"]["source_sha256"] != right["source"]["source_sha256"]:
        raise ValueError("comparison source differs")
    if cpu["device"] != "cpu" or not cuda["device"].startswith("cuda:"):
        raise ValueError("comparison requires explicit CPU and CUDA allocations")
    qualified = (cpu["qualification"] == "cpu_smoke_passed"
                 and cuda["qualification"] == "cuda_smoke_passed"
                 and not any(item["active"] or item["unresolved_attempt"] for item in (cpu, cuda)))
    return {"schema": "nca-ik-device-comparison/v1", "matched_workload": True,
            "comparison_measured": qualified, "cpu": cpu, "cuda": cuda,
            "quality_qualification": False, "cross_device_bit_identity": False,
            "speedup_claim": "none; inspect synchronized workload and full charged costs separately"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--device", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--device", required=True)
    prepare.add_argument("--output", required=True)
    for name in ("smoke", "resume"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True)
        command.add_argument("--declaration-sha256", required=True)
        command.add_argument("--stop-after", type=int)
        command.add_argument("--allow-environment-change", action="store_true")
        command.add_argument("--device")
    for name in ("cancel", "inspect", "_worker"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True)
        if name == "_worker":
            command.add_argument("--attempt", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--cpu", required=True)
    compare.add_argument("--cuda", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "_worker":
            from .runtime_worker import worker
            return worker(Path(args.run), args.attempt)
        if args.command == "doctor":
            # Inventory only; no CUDA context or optional import. Actual
            # preflight belongs to the prospectively declared smoke attempt.
            from .declaration import runtime_expectation
            from .device import parse_device
            parse_device(args.device)
            result = {"requested_device": args.device, "runtime": runtime_expectation(),
                      "qualification": "unmeasured", "device_preflight": "performed inside declared smoke"}
        elif args.command == "prepare":
            from .declaration import prepare as prepare_run
            result = prepare_run(args.config, args.device, args.output)
            result = {"declaration": result, "declaration_sha256": sha256_file(Path(args.output) / "declaration.json"),
                      "next": "publish the exact declaration before invoking smoke"}
        elif args.command in ("smoke", "resume"):
            result = execute(args.run, args.declaration_sha256, resume=args.command == "resume",
                             stop_after=args.stop_after, allow_environment_change=args.allow_environment_change,
                             device=args.device, started=ENTRY_STARTED)
        elif args.command == "cancel":
            result = cancel_run(args.run)
        elif args.command == "inspect":
            result = inspect_run(args.run)
        else:
            result = compare_runs(args.cpu, args.cuda)
        if args.command in ("smoke", "resume"):
            display = {key: value for key, value in result.items() if key not in ("attempts", "checkpoint")}
            display["attempt_statuses"] = [{"id": item["run_id"], "status": item["status"],
                                            "failures": item["failures"]} for item in result["attempts"]]
            print(canonical_json(display))
        else:
            print(canonical_json(result))
        if args.command in ("smoke", "resume"):
            if result.get("admission_refused"):
                return 1
            return 0 if result["attempts"][-1]["status"] == "completed" else 1
        if args.command == "compare" and not result["comparison_measured"]:
            return 1
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(canonical_json({"status": "refused", "error_type": type(exc).__name__,
                              "reason": getattr(exc, "code", "invalid_or_unavailable_allocation"),
                              "qualification": "unmeasured"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
