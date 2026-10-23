"""Reproducible compact IK training orchestration and CPU-only preparation.

The CLI's prepare command authenticates train/tuning data and writes a frozen
declaration without importing Torch. Actual execution is deliberately explicit,
supervised, CUDA-only for the accepted corpus, and bound to that declaration.
Small fixture calls never qualify learned quality or open final data.
"""
from __future__ import annotations

import time
ENTRY_STARTED = time.monotonic()

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from nca.contracts import canonical_json, sha256_json, strict_json_loads
from nca.ik.declaration import (_project, _verify_committed_sources,
                                runtime_expectation, source_snapshot)
from nca.ik.device import configure_device, device_report, parse_device
from nca.ik.profiles import runner_right_sole_profile
from nca.ik.kinematics import chain_from_profile, cpu_fk
from nca.runs import immutable_json, utc_now
from nca.train.ik import IKTrainer, prepare_training_batch
from nca.train.ik_corpus import PairPreservingSampler, read_training_corpus
from nca.train.ik_run_state import IKRunLedger
from nca.evaluation.ik_training import (collect_predictions, model_predictor,
                                       score_raw)


SEEDS = (41011, 42013, 43019)
PLAN_PATH = "configs/ik/experiments/runner-compact-schedule-v1.json"
CANDIDATE_PATH = "configs/ik/experiments/runner-compact-v1.json"
ACCEPTANCE_PATH = "configs/ik/runner_right_sole_acceptance_v1.json"
STOP_RESERVE_SECONDS = 10.0


def _read(path, maximum=1_048_576):
    with Path(path).open("rb") as stream:
        return strict_json_loads(stream.read(maximum + 1), max_bytes=maximum)


def _digest(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def validate_plan(value):
    """One fixed configuration; editing its JSON does not expand this workload."""
    expected = {
        "schema": "nca-ik-training-plan/v1", "id": "runner-compact-training-v1",
        "status": "prospective_gpu_gate_not_executed", "candidate": CANDIDATE_PATH,
        "training_seeds": list(SEEDS), "epochs": 40, "batch_size": 128,
        "curriculum": "stage_order_then_seeded_whole_group_shuffle_each_epoch",
        "checkpoint_every_epochs": 1,
        "selection": "tuning_lowest_raw_p95_ties_earliest",
        "mode": "direct_joint_position", "numerical_correction_iterations": 0,
        "full_reference_fallback_calls": 0, "max_attempts_per_seed": 4,
        "max_wall_seconds_per_seed": 7200, "max_memory_bytes": 2 * 1024**3,
        "checkpoint_limit_per_seed": 45,
        "cpu_latency": {"warmup_calls": 100, "measured_calls": 10000, "threads": 1, "batch_size": 1},
        "final_comparators": 5,
        "expected_dataset": {"training_rows": 5260, "tuning_rows": 2020, "final_rows": 2940},
        "measured_execution_admitted": False,
        "qualified_environment": {
            "os": "Windows",
            "os_release": "11",
            "architecture": "AMD64",
            "python": "3.12.10",
            "torch": "2.8.0+cu126",
            "numpy": "2.3.5",
            "cuda_build": "12.6",
            "driver": "581.80",
            "gpu": {
                "allowed_names": ["TITAN RTX", "NVIDIA TITAN RTX"],
                "capability": [7, 5],
                "multiprocessors": 72,
                "minimum_total_memory_bytes": 20 * 1024**3,
            }
        },
    }
    # Canonical comparison is type-sensitive (true cannot stand for 1).
    if canonical_json(value) != canonical_json(expected):
        raise ValueError("the fixed prospective compact IK schedule differs")
    return value



_TITAN_UUID = r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"


def _titan_visibility(*, expected_sha256=None, driver="581.80"):
    """CPU-only identity gate: never create CUDA context on an unselected GPU.

    CUDA_VISIBLE_DEVICES must contain one *TITAN RTX UUID*, not an ordinal, a
    list, or the V100. Query only that UUID with nvidia-smi. Do not publish it.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if re.fullmatch(_TITAN_UUID, visible) is None:
        raise ValueError("select only the TITAN RTX UUID in CUDA_VISIBLE_DEVICES; V100 is prohibited")
    fingerprint = hashlib.sha256(visible.lower().encode("ascii")).hexdigest()
    if expected_sha256 is not None and fingerprint != expected_sha256:
        raise ValueError("TITAN RTX selection changed since prospective declaration")
    try:
        queried = subprocess.run(
            ["nvidia-smi", "-i", visible, "--query-gpu=name,uuid,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        fields = [value.strip() for value in queried.stdout.strip().split(",")]
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("selected TITAN RTX metadata unavailable; no GPU initialization authorized") from exc
    if (len(fields) != 3 or fields[0] not in ("TITAN RTX", "NVIDIA TITAN RTX")
            or fields[1].lower() != visible.lower() or fields[2] != driver):
        raise ValueError("selected device is not the declared TITAN RTX and driver; V100 is prohibited")
    return fingerprint


def _qualified_titan_gpu_bytes(report, expected):
    """Admit the selected TITAN's actual Torch capacity, not V100's old bytes."""
    gpu, target = report.get("gpu"), expected["gpu"]
    if (report.get("status") != "ready" or report.get("requested_device") != "cuda:0"
            or any(report.get(key) != expected[key] for key in (
                "os", "os_release", "architecture", "python", "torch", "numpy",
                "cuda_build", "driver"))
            or not isinstance(gpu, dict)
            or gpu.get("name") not in target["allowed_names"]
            or gpu.get("capability") != target["capability"]
            or gpu.get("multiprocessors") != target["multiprocessors"]):
        raise ValueError("TITAN RTX Windows runtime qualification mismatch; V100 forbidden")
    total = gpu.get("total_memory_bytes")
    if type(total) is not int or total < target["minimum_total_memory_bytes"]:
        raise ValueError("TITAN RTX measured device memory is insufficient or unknown")
    return total


def _source(root):
    project = _project(root)
    source = source_snapshot(project, require_clean=True)
    for relative in (PLAN_PATH, CANDIDATE_PATH, ACCEPTANCE_PATH,
                     "configs/ik/datasets/runner_v1.json"):
        source["files"][relative] = hashlib.sha256((project / relative).read_bytes()).hexdigest()
    _verify_committed_sources(project, source["files"])
    source["source_sha256"] = sha256_json(source["files"])
    return source


def corpus_identity(corpus):
    return {
        "dataset_config_sha256": corpus.config.sha256,
        "manifest_sha256": corpus.manifest_sha256,
        "normalization_sha256": corpus.normalization.sha256,
        "normalization_file_sha256": corpus.normalization_file_sha256,
        "training_inputs_sha256": corpus.training_inputs_sha256,
        "final_suite_identity_sha256": corpus.final_suite_identity_sha256,
        "registry_id": corpus.registry_id,
        "training_rows": len(corpus.training), "tuning_rows": len(corpus.tuning),
    }


def allocation_for(corpus, plan, seed):
    sampler = PairPreservingSampler(corpus.training, seed,
                                    batch_size=plan["batch_size"], epochs=plan["epochs"])
    # Every row is retained, including batches with no trainable labels.
    # Such a batch charges presentations without an optimizer update.
    final_calls = plan["final_comparators"] * plan["expected_dataset"]["final_rows"]
    timing_calls = plan["final_comparators"] * (
        plan["cpu_latency"]["warmup_calls"] + plan["cpu_latency"]["measured_calls"])
    return {
        "max_optimizer_steps": sampler.planned_batches,
        "max_sample_presentations": sampler.total_presentations,
        "max_tuning_calls": plan["epochs"] * len(corpus.tuning),
        "max_evaluation_calls": final_calls + timing_calls,
        "max_checkpoints": plan["checkpoint_limit_per_seed"],
        "max_attempts": plan["max_attempts_per_seed"],
        "max_wall_seconds": plan["max_wall_seconds_per_seed"],
        "max_memory_bytes": plan["max_memory_bytes"],
    }


def prepare(corpus_root, registry, output, *, device="cuda:0", source_root=None):
    """Authenticate the exact corpus and freeze all three allocations, CPU-only."""
    kind, _ = parse_device(device)
    if kind != "cuda" or device != "cuda:0":
        raise ValueError("accepted comparison requires isolated TITAN RTX at cuda:0")
    project = _project(source_root)
    source = _source(project)
    plan = validate_plan(_read(project / PLAN_PATH))
    gpu_visibility_sha256 = _titan_visibility(driver=plan["qualified_environment"]["driver"])
    candidate = _read(project / CANDIDATE_PATH)
    corpus = read_training_corpus(corpus_root, registry)
    if not corpus.accepted or (len(corpus.training), len(corpus.tuning)) != (5260, 2020):
        raise ValueError("accepted corpus authentication and exact train/tuning denominators required")
    bound = corpus_identity(corpus)
    if any(candidate["dataset"].get(key) != bound[key] for key in
           ("manifest_sha256", "normalization_file_sha256", "training_inputs_sha256",
            "final_suite_identity_sha256")) or candidate["dataset"]["config_sha256"] != corpus.config.sha256:
        raise ValueError("candidate and authenticated corpus identities differ")
    profile = runner_right_sole_profile()
    runtime = runtime_expectation()
    expected = plan["qualified_environment"]
    if (any(runtime.get(key) != expected[key] for key in
            ("python", "os", "os_release", "architecture"))
            or any(runtime["packages"].get(key) != expected[key] for key in ("torch", "numpy"))):
        raise ValueError("prepare on the qualified run environment; metadata differs")
    declaration = {
        "schema": "nca-ik-compact-declaration/v1",
        "id": "ik-compact-" + uuid.uuid4().hex, "created_utc": utc_now(),
        "plan": plan, "candidate_sha256": sha256_json(candidate),
        "source": source, "runtime": runtime, "device": device,
        "gpu_visibility_sha256": gpu_visibility_sha256,
        "profile_sha256": profile.sha256, "dataset": bound,
        "acceptance_file_sha256": source["files"][ACCEPTANCE_PATH],
        "allocations": {str(seed): allocation_for(corpus, plan, seed) for seed in SEEDS},
        "final_opened": False, "gpu_model_path_qualified": False,
        "next_required_action": "independent_declaration_review_then_exact_model_CUDA_execution",
    }
    declaration["sha256"] = sha256_json(declaration)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    raw_digest = immutable_json(target / "declaration.json", declaration)
    for seed in SEEDS:
        IKRunLedger.create(target / f"seed-{seed}", identity=identity_for(declaration, seed),
                           allocation=declaration["allocations"][str(seed)])
    return {"declaration": declaration, "declaration_file_sha256": raw_digest,
            "gpu_executed": False, "final_opened": False}


def identity_for(declaration, seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("one of the three frozen seeds is required")
    return {
        "config_sha256": sha256_json({"plan": declaration["plan"],
                                      "candidate_sha256": declaration["candidate_sha256"]}),
        "source_sha256": declaration["source"]["source_sha256"],
        "profile_sha256": declaration["profile_sha256"],
        "dataset_sha256": declaration["dataset"]["manifest_sha256"],
        "normalization_sha256": declaration["dataset"]["normalization_sha256"],
        "runtime_sha256": sha256_json(declaration["runtime"]),
        "declaration_sha256": declaration["sha256"], "seed": seed,
    }


def load_declaration(run, expected_sha256):
    if not _digest(expected_sha256):
        raise ValueError("an exact prospective declaration file SHA-256 is required")
    path = Path(run) / "declaration.json"
    with path.open("rb") as stream:
        data = stream.read(1_048_577)
    if len(data) > 1_048_576 or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("declaration file SHA-256 differs")
    value = strict_json_loads(data)
    fields = {"schema", "id", "created_utc", "plan", "candidate_sha256", "source",
              "runtime", "device", "gpu_visibility_sha256",
              "profile_sha256", "dataset", "acceptance_file_sha256",
              "allocations", "final_opened", "gpu_model_path_qualified", "next_required_action", "sha256"}
    if (not isinstance(value, dict) or set(value) != fields
            or value["schema"] != "nca-ik-compact-declaration/v1"
            or value["sha256"] != sha256_json({k: v for k, v in value.items() if k != "sha256"})
            or value["final_opened"] is not False
            or value["gpu_model_path_qualified"] is not False):
        raise ValueError("compact declaration schema/seal differs")
    validate_plan(value["plan"])
    if value["device"] != "cuda:0" or not _digest(value["gpu_visibility_sha256"]):
        raise ValueError("only the source-bound isolated TITAN RTX at cuda:0 is admitted")
    if set(value["allocations"]) != {str(seed) for seed in SEEDS}:
        raise ValueError("all three seed allocations are required")
    for seed in SEEDS:
        identity_for(value, seed)
    return value


def _tuning_record(corpus, trainer, step):
    trainer.model.eval()
    predictor = model_predictor(trainer.context, trainer.model)
    predictions = collect_predictions(corpus.tuning, predictor)
    report = score_raw(corpus.tuning, predictions, corpus.config, partition="tuning",
                       authenticated_corpus=corpus if corpus.accepted else None)
    return {"step": step, "tuning": report}


def independent_fk_guard(trainer, batch):
    """Check this device's tensor geometry against independent binary64 FK.

    This is a geometry guard on the original raw output, never a correction or
    an alternative model answer. It executes once at the first optimizer batch.
    """
    context = trainer.context
    with context.torch.no_grad():
        prediction = trainer.model(batch.features)
        geometry = trainer.fk(prediction.reference)
        references = prediction.reference.detach().cpu().tolist()
        points = geometry.point.detach().cpu().tolist()
        valid = prediction.valid.detach().cpu().tolist()
    chain = chain_from_profile(runner_right_sole_profile())
    errors = [math.dist(cpu_fk(chain, tuple(q)).point, tuple(point))
              for q, point, admitted in zip(references, points, valid) if admitted]
    if not errors or any(not math.isfinite(error) or error > 2e-4 for error in errors):
        raise ValueError("independent CPU/tensor FK guard failed")
    return {"schema": "nca-ik-training-fk-guard/v1", "device": str(context.device),
            "checked_rows": len(errors), "max_error_cells": max(errors),
            "tolerance_cells": 2e-4, "correction_iterations": 0, "fallback_calls": 0}


def train_attempt(ledger, attempt, corpus, context, *, epochs, batch_size,
                  pause_after_batches=None, execution=None):
    """Run an already admitted attempt; callable with tiny CPU fixtures in tests.

    The CLI owns process supervision and source/runtime admission. An accepted
    corpus is refused on CPU. No final reader or final opening runs here.
    """
    if corpus.accepted and context.device.type != "cuda":
        raise ValueError("accepted corpus training requires CUDA; fixtures remain separate")
    if (type(epochs) is not int or epochs < 1 or type(batch_size) is not int
            or not 1 <= batch_size <= 1024):
        raise ValueError("finite epochs and batch size required")
    if pause_after_batches is not None and (
            type(pause_after_batches) is not int or pause_after_batches < 1):
        raise ValueError("pause boundary must be a positive batch count")
    admitted_state = ledger.inspect()
    seed = admitted_state["identity"]["seed"]
    memory_ceiling = admitted_state["allocation"]["max_memory_bytes"]
    sampler = PairPreservingSampler(corpus.training, seed, batch_size=batch_size, epochs=epochs)
    epoch_ends = {}
    cursor = 0
    for epoch, count in enumerate(sampler.batches_per_epoch, start=1):
        cursor += count
        epoch_ends[cursor] = epoch
    trainer = IKTrainer(context, corpus.normalization, corpus.config)
    extra = ledger.resume(trainer, sampler)
    if "execution" in extra and extra["execution"] != execution:
        raise ValueError("checkpoint execution amendment differs across continuation")
    history = list(extra.get("tuning_history", []))
    completed_epochs = extra.get("completed_epochs", 0)
    initial = extra.get("initial_checkpoint")
    geometry_check = extra.get("geometry_check")
    device_identity = context.report["environment_sha256"]
    if extra.get("device_environment_sha256", device_identity) != device_identity:
        raise ValueError("actual device environment differs across continuation")
    latest = ledger.inspect()["latest_checkpoint"]
    if initial is None and latest is not None:
        if latest["state"]["consumed_batches"] != 0:
            raise ValueError("matched untrained checkpoint identity is missing")
        initial = {"path": latest["path"], "reference": latest["reference"]}
    if initial is None:
        saved_initial = attempt.checkpoint(trainer, sampler.state_dict(), extra_state={
            "tuning_history": history, "completed_epochs": 0, "initial_checkpoint": None,
            "execution": execution,
            "device_environment_sha256": device_identity})
        initial = {"path": saved_initial["path"], "reference": saved_initial["reference"]}
    batch_limit = pause_after_batches
    done = 0
    cumulative_timing = dict(extra.get("training_timing", {
        "synchronized_host_seconds": 0.0, "cuda_event_seconds": None if context.device.type == "cpu" else 0.0}))
    context.reset_peak_memory()
    while not sampler.exhausted:
        if attempt.remaining_wall_seconds <= STOP_RESERVE_SECONDS:
            break
        rows = sampler.next_batch()
        prepared = prepare_training_batch(rows, corpus.config, context,
                                         authenticated_corpus=corpus if corpus.accepted else None)
        trainable = bool((prepared.pose_mask | prepared.endpoint_mask).any())
        token = attempt.reserve("optimizer" if trainable else "presentation_only",
                                samples=len(rows), call_id=f"batch-{sampler.cursor:08d}")
        if trainable:
            if geometry_check is None:
                geometry_check = independent_fk_guard(trainer, prepared)
            receipt, elapsed = context.timed(lambda: trainer.step(prepared))
            if trainer.committed_steps == 1:
                if receipt["weights_changed"] is not True:
                    raise ValueError("first optimizer update did not change model weights")
                receipt["independent_fk_guard"] = geometry_check
            cumulative_timing["synchronized_host_seconds"] += elapsed["synchronized_host_seconds"]
            if elapsed["cuda_event_seconds"] is not None:
                cumulative_timing["cuda_event_seconds"] += elapsed["cuda_event_seconds"]
            receipt["timing"] = elapsed
        else:
            with context.torch.no_grad():
                prediction = trainer.model(prepared.features)
                context.verify_tensors({"refusal_output": prediction.reference})
            receipt = {"schema": "nca-ik-no-loss-batch/v1", "rows": len(rows),
                       "ordered_sample_ids_sha256": sha256_json(list(prepared.sample_ids)),
                       "teacher_status_counts": prepared.teacher_status_counts,
                       "optimizer_update": False}
        attempt.complete(token, receipt)
        done += 1
        epoch = epoch_ends.get(sampler.cursor)
        if epoch is not None:
            token = attempt.reserve("tuning", calls=len(corpus.tuning), call_id=f"tuning-{epoch:04d}")
            item = _tuning_record(corpus, trainer, trainer.committed_steps)
            attempt.complete(token, {"epoch": epoch, "report_sha256": sha256_json(item["tuning"]),
                                     "complete": item["tuning"]["complete"]})
            report_file = f"tuning-epoch-{epoch:04d}.json"
            report_digest = immutable_json(ledger.root / report_file, item["tuning"])
            history.append({"step": item["step"], "epoch": epoch, "file": report_file,
                            "file_sha256": report_digest,
                            "report_sha256": sha256_json(item["tuning"])})
            completed_epochs = epoch
        memory = context.memory_stats()
        if (memory["peak_reserved_bytes"] is not None
                and memory["peak_reserved_bytes"] > memory_ceiling):
            raise ValueError("declared CUDA allocator memory ceiling exceeded")
        at_epoch = epoch is not None
        paused = batch_limit is not None and done >= batch_limit
        if at_epoch or paused or attempt.remaining_wall_seconds <= STOP_RESERVE_SECONDS:
            ref = attempt.checkpoint(trainer, sampler.state_dict(), extra_state={
                "tuning_history": history, "completed_epochs": completed_epochs,
                "initial_checkpoint": initial, "training_timing": cumulative_timing,
                "execution": execution,
                "geometry_check": geometry_check, "device_environment_sha256": device_identity})
            # A checkpoint's bytes include metrics, but cannot include their own hash.
            # Later reconstruction pairs the saved last tuning entry with this ref.
        if paused:
            break
    # If time stopped us before any epoch/explicit pause, capture the exact boundary.
    info = ledger.inspect()
    if (info["latest_checkpoint"] is None
            or info["latest_checkpoint"]["state"]["charged"] != info["charged"]):
        attempt.checkpoint(trainer, sampler.state_dict(), extra_state={
            "tuning_history": history, "completed_epochs": completed_epochs,
            "initial_checkpoint": initial, "training_timing": cumulative_timing,
                "execution": execution,
                "geometry_check": geometry_check, "device_environment_sha256": device_identity})
    result = {
        "schema": "nca-ik-training-attempt-result/v1", "seed": seed,
        "training_complete": sampler.exhausted, "completed_epochs": completed_epochs,
        "planned_epochs": epochs, "sampler": sampler.state_dict(),
        "initial_checkpoint": initial, "training_timing": cumulative_timing,
        "execution": execution,
        "geometry_check": geometry_check,
        "device": str(context.device), "memory": context.memory_stats(),
        "device_report": context.report,
        "optimizer_sample_presentations": trainer.sample_presentations,
        "optimizer_samples_per_synchronized_host_second": (
            trainer.sample_presentations / cumulative_timing["synchronized_host_seconds"]
            if cumulative_timing["synchronized_host_seconds"] > 0 else None),
        "cuda_forward_backward_optimizer_observed": (
            context.device.type == "cuda" and trainer.committed_steps > 0),
        "final_opened": False, "learned_quality_qualified": False,
    }
    attempt.finish("paused", reason="training_complete_final_unopened" if sampler.exhausted
                   else "exact_training_boundary_paused")
    return result


def _review(review_path, declaration):
    review = _read(review_path)
    expected = {"schema", "declaration_sha256", "source_sha256", "reviewers", "checks", "verdict"}
    if (not isinstance(review, dict) or set(review) != expected
            or review["schema"] != "nca-ik-compact-admission/v1"
            or review["declaration_sha256"] != declaration["sha256"]
            or review["source_sha256"] != declaration["source"]["source_sha256"]
            or review["verdict"] != "admit_declared_CUDA_comparison"):
        raise ValueError("source/declaration-bound independent admission is required")
    if (not isinstance(review["reviewers"], list) or len(review["reviewers"]) < 2
            or any(type(x) is not str or re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,63}", x) is None
                   for x in review["reviewers"])
            or len(set(review["reviewers"])) != len(review["reviewers"])):
        raise ValueError("two distinct public review identifiers required")
    checks = {"source_and_cpu_checks", "corpus_and_splits", "allocation_and_resume",
              "scoring_and_selection", "environment_and_memory_feasibility"}
    if not isinstance(review["checks"], dict) or set(review["checks"]) != checks or any(
            item is not True for item in review["checks"].values()):
        raise ValueError("all independent prospective admission checks must pass")
    return review


def worker(run, declaration_digest, seed, corpus_root, registry, review_path,
           *, started, review_sha256, pause_after_batches=None, source_root=None):
    declaration = load_declaration(run, declaration_digest)
    ledger = IKRunLedger.open(Path(run) / f"seed-{seed}",
                              expected_identity=identity_for(declaration, seed),
                              expected_allocation=declaration["allocations"][str(seed)])
    attempt = ledger.begin_attempt(started=started)
    try:
        if sha256_json(_review(review_path, declaration)) != review_sha256:
            raise ValueError("admission review changed after supervisor publication")
        _titan_visibility(expected_sha256=declaration["gpu_visibility_sha256"],
                          driver=declaration["plan"]["qualified_environment"]["driver"])
        if _source(source_root) != declaration["source"]:
            raise ValueError("executed source differs from prospective source")
        if runtime_expectation() != declaration["runtime"]:
            raise ValueError("runtime differs; no silent environment continuation")
        corpus = read_training_corpus(corpus_root, registry)
        if corpus_identity(corpus) != declaration["dataset"]:
            raise ValueError("authenticated corpus differs from declaration")
        plan = declaration["plan"]
        if allocation_for(corpus, plan, seed) != declaration["allocations"][str(seed)]:
            raise ValueError("computed finite allocation differs from declaration")
        source_id = declaration["source"]["source_sha256"]
        config_id = identity_for(declaration, seed)["config_sha256"]
        # The UUID-bound metadata check has passed. Initialize the *selected*
        # TITAN only, then derive the 2 GiB allocator fraction from its actual
        # Torch-reported capacity, never from a historical V100 measurement.
        preflight = device_report(declaration["device"], source_sha256=source_id,
                                  config_sha256=config_id)
        gpu_total = _qualified_titan_gpu_bytes(preflight, plan["qualified_environment"])
        context = configure_device(declaration["device"], seed,
                                   memory_fraction=min(1.0, plan["max_memory_bytes"] / gpu_total),
                                   source_sha256=source_id, config_sha256=config_id)
        if _qualified_titan_gpu_bytes(context.report, plan["qualified_environment"]) != gpu_total:
            raise ValueError("TITAN RTX identity/capacity changed during device setup")
        # The exact prospective GPU capacity determines the allocator fraction;
        # per-step peak checks also enforce the declared byte ceiling.
        result = train_attempt(ledger, attempt, corpus, context,
                               epochs=plan["epochs"], batch_size=plan["batch_size"],
                               pause_after_batches=pause_after_batches)
        index = ledger.inspect()["attempt_count"]
        immutable_json(Path(run) / f"seed-{seed}" / f"training-result-{index:04d}.json", result)
        return result
    except BaseException as exc:
        if ledger.inspect()["active"]:
            attempt.finish("failed", reason=type(exc).__name__)
        raise


def supervise(run, declaration_digest, seed, corpus_root, registry, review_path,
              *, pause_after_batches=None, started=None):
    """Own only a disposable worker; timeout/uncertain completion blocks reuse."""
    started = time.monotonic() if started is None else started
    run = Path(run)
    declaration = load_declaration(run, declaration_digest)
    review = _review(review_path, declaration)
    _titan_visibility(expected_sha256=declaration["gpu_visibility_sha256"],
                      driver=declaration["plan"]["qualified_environment"]["driver"])
    frozen_review = run / "admission-review.json"
    if frozen_review.exists():
        if _read(frozen_review) != review:
            raise ValueError("continuation admission review differs from original")
    else:
        immutable_json(frozen_review, review)
    review_digest = sha256_json(review)
    root = run / f"seed-{seed}"
    ledger = IKRunLedger.open(root, expected_identity=identity_for(declaration, seed),
                              expected_allocation=declaration["allocations"][str(seed)])
    info = ledger.inspect()
    if not info["can_resume"]:
        raise ValueError("allocation cannot safely continue")
    # A previous supervisor without a verified terminal receipt is not replayable.
    if any(root.glob("supervisor-overrun-*.json")):
        raise ValueError("whole-process overrun is retained; continuation is forbidden")
    for previous in sorted(root.glob("supervisor-start-*.json")):
        terminal = previous.with_name(previous.name.replace("start", "result", 1))
        if not terminal.is_file() or _read(terminal).get("verified_within_charged_wall") is not True:
            raise ValueError("previous whole-process outcome is uncertain; preserve for review")
    remaining = info["remaining_wall_seconds"]
    index = info["attempt_count"] + 1
    immutable_json(root / f"supervisor-start-{index:04d}.json", {
        "schema": "nca-ik-compact-supervisor-start/v1", "declaration_sha256": declaration["sha256"],
        "seed": seed, "attempt": index, "remaining_wall_seconds": remaining,
        "admission_review_sha256": review_digest, "created_utc": utc_now()})
    command = [sys.executable, "-m", "nca.train.ik_experiment", "_worker",
               "--run", str(run), "--declaration-sha256", declaration_digest,
               "--seed", str(seed), "--corpus", str(corpus_root), "--registry", str(registry),
               "--review", str(frozen_review), "--review-sha256", review_digest,
               "--started", repr(started)]
    if pause_after_batches is not None:
        command.extend(["--pause-after-batches", str(pause_after_batches)])
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    timed_out = False
    interrupted = False
    try:
        # Leave time to observe child exit and preserve the terminal receipt.
        process.wait(timeout=max(0.001, remaining - (time.monotonic() - started) - 5.0))
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        process.wait(timeout=5.0)
    except BaseException:
        interrupted = True
        process.kill()
        process.wait(timeout=5.0)
    elapsed = time.monotonic() - started
    after = ledger.inspect()
    charged = after["charged_wall_seconds"] - info["charged_wall_seconds"]
    verified = (not timed_out and not interrupted and process.returncode == 0 and not after["active"]
                and not after["unresolved_work"] and elapsed <= charged and elapsed <= remaining)
    result = {
        "schema": "nca-ik-compact-supervisor-result/v1", "seed": seed, "attempt": index,
        "returncode": process.returncode, "timed_out": timed_out, "interrupted": interrupted,
        "whole_process_seconds": elapsed,
        "consumed_or_reserved_wall_seconds": charged if verified else remaining,
        "charged_wall_seconds": charged, "verified_within_charged_wall": verified,
        "unknown_work_blocks_resume": not verified, "final_opened": False,
    }
    immutable_json(root / f"supervisor-result-{index:04d}.json", result)
    if time.monotonic() - started > charged or time.monotonic() - started > remaining:
        immutable_json(root / f"supervisor-overrun-{index:04d}.json",
                       {"schema": "nca-ik-compact-supervisor-overrun/v1", "resume_forbidden": True})
        raise ValueError("whole-process publication exceeded the charged reserve")
    if not verified:
        raise ValueError("worker failed/capped or complete timing remains uncertain")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="authenticate train/tuning data and freeze; no GPU work")
    prep.add_argument("--corpus", required=True)
    prep.add_argument("--registry", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--device", default="cuda:0")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--run", required=True)
    inspect.add_argument("--declaration-sha256", required=True)
    for name in ("run", "_worker"):
        entry = sub.add_parser(name)
        for argument in ("run", "declaration-sha256", "corpus", "registry", "review"):
            entry.add_argument("--" + argument, required=True)
        entry.add_argument("--seed", type=int, choices=SEEDS, required=True)
        entry.add_argument("--pause-after-batches", type=int)
        if name == "_worker":
            entry.add_argument("--started", type=float, required=True)
            entry.add_argument("--review-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.corpus, args.registry, args.output, device=args.device)
        elif args.command == "inspect":
            declaration = load_declaration(args.run, args.declaration_sha256)
            result = {str(seed): IKRunLedger.open(Path(args.run) / f"seed-{seed}",
                       expected_identity=identity_for(declaration, seed),
                       expected_allocation=declaration["allocations"][str(seed)]).inspect() for seed in SEEDS}
        else:
            keywords = {"pause_after_batches": args.pause_after_batches}
            if args.command == "_worker":
                keywords["started"] = args.started
                keywords["review_sha256"] = args.review_sha256
            else:
                keywords["started"] = ENTRY_STARTED
            result = (worker if args.command == "_worker" else supervise)(
                args.run, args.declaration_sha256, args.seed, args.corpus, args.registry,
                args.review, **keywords)
        print(canonical_json(result))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        # Public logs retain a stable class, never raw filesystem paths.
        print(canonical_json({"status": "refused_or_failed", "error_class": type(exc).__name__,
                              "quality_qualified": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
