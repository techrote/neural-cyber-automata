"""Optional runtime worker, invoked only by the bounded stdlib supervisor."""

from __future__ import annotations

from pathlib import Path
import signal
import time

from nca.contracts import canonical_json, sha256_json
from nca.runs import immutable_json, sha256_file
from .runtime import read_json, _relative


def worker(root: Path, name: str) -> int:
    from .declaration import load_declaration, metadata_for, runtime_expectation, verify_source
    from .device import DeviceError, configure_device
    from .checkpoint import load_checkpoint, restore_checkpoint, write_checkpoint
    from .workload import SmokeWorkload, model_digest
    attempt = _relative(root, f"attempts/{name}")
    request = read_json(attempt / "request.json")
    declaration = load_declaration(root, request["declaration_sha256"])
    if read_json(root / "active.json") != {"attempt": name, "declaration_sha256": request["declaration_sha256"]}:
        raise ValueError("worker lacks an active supervisor allocation")
    config = declaration["config"]
    started = time.monotonic()
    deadline = request["worker_deadline_monotonic"]
    canceled = False
    def on_signal(signum, frame):
        nonlocal canceled
        canceled = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    terminal = {"schema": "nca-ik-worker-result/v1", "status": "failed", "reason": "worker_failure",
                "qualification": "unmeasured", "quality_qualification": False,
                "environment_admitted": False, "stage": "source_validation",
                "timing": {"preflight_seconds": 0.0, "loading_seconds": 0.0,
                           "generation_seconds": 0.0, "checkpoint_seconds": 0.0,
                           "synchronized_training_seconds": 0.0, "cuda_event_seconds": None},
                "reproducibility_limits": list(request["lineage_limits"]), "progress": []}
    context = workload = None
    attempted = 0
    try:
        verify_source(declaration)
        terminal["stage"] = "runtime_validation"
        current_runtime = runtime_expectation()
        environment_changed = current_runtime != declaration["runtime"]
        if environment_changed:
            terminal["reproducibility_limits"].append("runtime differs from prospective declaration")
            if not request["allow_environment_change"]:
                raise ValueError("runtime differs from prospective declaration")
        terminal["stage"] = "device_preflight"
        context = configure_device(request["device"], config["seed"], memory_fraction=config["memory_fraction"],
                                   source_sha256=declaration["source"]["source_sha256"],
                                   config_sha256=declaration["config_sha256"])
        terminal["environment"] = context.report
        immutable_json(attempt / "preflight.json", context.report)
        terminal["timing"]["preflight_seconds"] = time.monotonic() - started
        if request["environment_sha256"] is not None:
            environment_changed |= request["environment_sha256"] != context.report["environment_sha256"]
        environment_changed |= request["device"] != declaration["device"]
        if environment_changed:
            terminal["reproducibility_limits"].append("environment continuation; bit identity is not claimed")
            if not request["allow_environment_change"]:
                raise ValueError("device environment changed; explicit continuation required")
        terminal["environment_admitted"] = True
        metadata = metadata_for(declaration, context.report["environment_sha256"])
        context.reset_peak_memory()
        stage = time.monotonic()
        workload = SmokeWorkload(context, config)
        checkpoint = request["checkpoint"]
        if checkpoint is not None:
            terminal["stage"] = "checkpoint_restore"
            expected = dict(metadata)
            if request["allow_environment_change"]:
                expected.pop("environment_sha256", None)
            payload = load_checkpoint(_relative(root, checkpoint["path"]),
                                      expected_sha256=checkpoint["reference"]["sha256"],
                                      expected_metadata=expected, max_bytes=config["max_checkpoint_bytes"])
            # Validate the saved sampler and counters before restoring tensors.
            workload.restore_state(payload["training_state"])
            if workload.committed_steps != request["prior_committed_steps"]:
                raise ValueError("checkpoint counter differs from immutable attempt ledger")
            saved_allocation = payload["training_state"].get("allocation", {})
            if (saved_allocation.get("attempted_steps", 0) > request["prior_charged_steps"]
                    or saved_allocation.get("charged_seconds", 0) > request["prior_charged_seconds"]):
                raise ValueError("checkpoint allocation exceeds immutable attempt ledger")
            restored, limits = restore_checkpoint(payload, workload.model, workload.optimizer,
                                                  generators=workload.generators,
                                                  allow_device_change=request["allow_environment_change"])
            workload.restore_state(restored)
            terminal["reproducibility_limits"].extend(limits)
        terminal["timing"]["loading_seconds"] = time.monotonic() - stage
        def allocation():
            seconds = request["prior_charged_seconds"] + time.monotonic() - request["attempt_started_monotonic"]
            steps = request["prior_charged_steps"] + attempted
            return {"attempted_steps": steps, "charged_seconds": seconds,
                    "remaining_steps": max(0, config["max_optimizer_steps"] - steps),
                    "remaining_samples": max(0, config["max_samples"] - steps * config["batch_size"]),
                    "remaining_seconds": max(0.0, config["max_wall_seconds"] - seconds),
                    "authority": "final supervisor ledger overrides this earlier checkpoint snapshot"}
        def checkpoint_at(filename):
            terminal["stage"] = "checkpoint_publication"
            stage = time.monotonic()
            reference = write_checkpoint(attempt / "checkpoints" / filename, workload.model, workload.optimizer,
                                         metadata=metadata, training_state=workload.state(allocation()),
                                         generators=workload.generators, max_bytes=config["max_checkpoint_bytes"])
            terminal["timing"]["checkpoint_seconds"] += time.monotonic() - stage
            return reference
        if checkpoint is None:
            if time.monotonic() >= deadline:
                terminal.update(status="capped", reason="startup_allocation_exhausted")
            else:
                immutable_json(attempt / "initial.json", checkpoint_at("initial.pt"))
        target = config["steps"]
        if request["stop_after"] is not None:
            target = min(target, workload.committed_steps + request["stop_after"])
        while workload.committed_steps < target:
            if canceled or (root / f"cancel-{name}.json").exists():
                terminal.update(status="canceled", reason="operator_cancellation")
                break
            if time.monotonic() >= deadline:
                terminal.update(status="capped", reason="wall_allocation_exhausted")
                break
            if (request["prior_charged_steps"] + attempted >= config["max_optimizer_steps"]
                    or (request["prior_charged_steps"] + attempted + 1) * config["batch_size"] > config["max_samples"]):
                terminal.update(status="capped", reason="optimizer_or_sample_allocation_exhausted")
                break
            attempted += 1
            reservation = attempt / f"reserve-{attempted:06d}.json"
            immutable_json(reservation, {"attempted_step": attempted, "samples": config["batch_size"],
                                        "next_committed_step": workload.committed_steps + 1})
            stage = time.monotonic()
            ids, batch = workload.batch()
            context.synchronize()
            terminal["timing"]["generation_seconds"] += time.monotonic() - stage
            terminal["stage"] = "optimizer_step"
            result, timing = context.timed(lambda: workload.step(batch))
            terminal["timing"]["synchronized_training_seconds"] += timing["synchronized_host_seconds"]
            if timing["cuda_event_seconds"] is not None:
                terminal["timing"]["cuda_event_seconds"] = (terminal["timing"]["cuda_event_seconds"] or 0.0) + timing["cuda_event_seconds"]
            reference = checkpoint_at(f"step-{attempted:06d}.pt")
            receipt = {"boundary": "optimizer_step", "committed_steps": workload.committed_steps,
                       "cursor": workload.cursor, "sample_ids": ids, "sample_ids_sha256": sha256_json(ids),
                       "reservation_sha256": sha256_file(reservation), "checkpoint": reference,
                       "result": result, "timing": timing}
            immutable_json(attempt / f"step-{attempted:06d}.json", receipt)
            progress = {"committed_steps": workload.committed_steps, "cursor": workload.cursor,
                        "loss": result["loss"], "attempted_steps": request["prior_charged_steps"] + attempted,
                        "samples_per_training_second": config["batch_size"] / timing["synchronized_host_seconds"]}
            terminal["progress"].append(progress)
            print(canonical_json(progress), flush=True)
        else:
            terminal.update(status="completed", reason="declared_boundary_reached")
            if workload.committed_steps == config["steps"] and not terminal["reproducibility_limits"]:
                if context.device.type == "cpu":
                    terminal["qualification"] = "cpu_smoke_passed"
                elif context.report.get("driver") is not None:
                    terminal["qualification"] = "cuda_smoke_passed"
                else:
                    terminal.update(status="inconclusive", reason="cuda_driver_identity_missing")
        terminal["committed_steps"] = workload.committed_steps
        terminal["cursor"] = workload.cursor
        terminal["model_sha256"] = model_digest(workload.model)
        terminal["initial_model_sha256"] = workload.initial_model_sha256
        terminal["memory"] = context.memory_stats()
        terminal["stage"] = "complete"
        if canceled or (root / f"cancel-{name}.json").exists():
            terminal.update(status="canceled", reason="operator_cancellation", qualification="unmeasured")
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, DeviceError):
            terminal["environment"] = exc.report
        oom = isinstance(exc, MemoryError) or (context is not None and isinstance(exc, context.torch.OutOfMemoryError))
        terminal.update(status="canceled" if isinstance(exc, KeyboardInterrupt) else "failed",
                        reason="out_of_memory" if oom else getattr(exc, "code", terminal["stage"] + "_failed"),
                        exception_type=type(exc).__name__, qualification="unmeasured")
    terminal["attempted_steps"] = attempted
    terminal["worker_seconds"] = time.monotonic() - started
    immutable_json(attempt / "worker.json", terminal)
    return 0 if terminal["status"] == "completed" else 1
