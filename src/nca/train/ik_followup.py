"""Separately allocated polar-v2 experiment with an all-seed tuning stop gate.

Original accepted train/tuning bytes are reused, never rewritten. This launcher
cannot continue v1 ledgers. A new sealed final suite is opened only after every
fixed seed completes forty epochs and its tuning-selected candidate passes the
original raw quality gate. A no-go terminates the experiment with final sealed.
"""
from __future__ import annotations

import time
ENTRY_STARTED = time.monotonic()
import argparse
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import uuid

from nca.contracts import canonical_json, sha256_json, strict_json_loads
from nca.ik.declaration import runtime_expectation, _project
from nca.ik.device import configure_device, device_report
from nca.ik.polar import PolarNormalization, TRANSFORM, fit_training_normalization, build_trainer, build_model
from nca.ik.data import GeometricSample, iter_tasks, validate_sample, validate_dataset
from nca.ik.fresh_final import FreshFinalConfig
from nca.ik.profiles import runner_right_sole_profile
from nca.runs import immutable_json, sha256_file, utc_now
from nca.train.ik_corpus import read_polar_followup_corpus, PairPreservingSampler
from nca.train.ik_experiment import (SEEDS, PLAN_PATH, ACCEPTANCE_PATH, _read, _source,
    _titan_visibility, _qualified_titan_gpu_bytes, corpus_identity, allocation_for,
    validate_plan, identity_for as original_identity, train_attempt, _review)
from nca.train.ik_run_state import IKRunLedger
from nca.train.ik_followup_gate import tuning_stop_gate
from nca.train.ik_followup_seal import verify_seal

ARCHIVED = {
    41011: "d232baa3dd1e7c081c74609a451166434af9f32607643bced369685e810bedb3",
    42013: "f137fe4050e58a62d50c5db7835a9a6cce1ccc17ff90493a248a31fb050689bf",
    43019: "38ec7ec155217598dabb7bbffc89499c0b3fbde95ec26f65f9759dada0e54cbe"}
QUALIFIED_DEVICE = {"capability": [7, 5], "multiprocessors": 72,
                    "name": "NVIDIA TITAN RTX", "total_memory_bytes": 25769476096}


def plan():
    root = _project(None)
    value = dict(validate_plan(_read(root / PLAN_PATH)))
    value.update(schema="nca-ik-polar-plan/v2", id="runner-polar-followup-v2",
                 final_comparators=6, feature_transform=TRANSFORM,
                 max_memory_bytes=20 * 1024**3,
                 execution_order="serial_seeds_serial_cpu_timing",
                 final_recipe_sha256=FreshFinalConfig().sha256,
                 tuning_stop_gate="all_three_original_raw_quality_before_final",
                 original_evidence="development_only_never_supervision_or_selection")
    return value


def identity_for(declaration, seed):
    result = original_identity(declaration, seed)
    result["normalization_sha256"] = PolarNormalization.from_dict(declaration["normalization"]).sha256
    return result


def ledger_at(run, declaration, seed):
    return IKRunLedger.open(Path(run) / f"seed-{seed}", expected_identity=identity_for(declaration, seed),
                            expected_allocation=declaration["allocations"][str(seed)])


def archived_paths(root):
    result = {}
    for seed, expected in ARCHIVED.items():
        seed_root = Path(root) / f"seed-{seed}"
        found = []
        for marker in seed_root.glob("attempts/attempt-*/checkpoints/*.pt.json"):
            record = _read(marker)
            ref = record.get("checkpoint", {})
            if ref.get("sha256") == expected:
                path = marker.with_suffix("")
                if sha256_file(path) != expected or record.get("record_sha256") != sha256_json(ref):
                    raise ValueError("archived candidate bytes/marker changed")
                found.append(path.relative_to(seed_root).as_posix())
        if len(found) != 1:
            raise ValueError("exact original tuning-selected archived candidate required")
        result[str(seed)] = {"path": found[0], "sha256": expected}
    return result


def prepare(corpus_root, registry, output, fresh_root, archived_run):
    source, schedule = _source(None), plan()
    visibility = _titan_visibility(driver=schedule["qualified_environment"]["driver"])
    corpus = read_polar_followup_corpus(corpus_root, registry, source_sha256=source["source_sha256"])
    normalization = fit_training_normalization(corpus)
    seal = verify_seal(fresh_root)
    runtime = runtime_expectation()
    expected = schedule["qualified_environment"]
    if (any(runtime.get(k) != expected[k] for k in ("python", "os", "os_release", "architecture"))
            or any(runtime["packages"].get(k) != expected[k] for k in ("torch", "numpy"))):
        raise ValueError("qualified Windows runtime differs")
    declaration = {"schema": "nca-ik-polar-declaration/v2", "id": "ik-polar-" + uuid.uuid4().hex,
        "created_utc": utc_now(), "source": source, "runtime": runtime, "device": "cuda:0",
        "gpu_visibility_sha256": visibility, "qualified_cuda_device": QUALIFIED_DEVICE,
        "profile_sha256": runner_right_sole_profile().sha256,
        "plan": schedule, "candidate_sha256": sha256_json({"policy": "runner-compact-ik@2",
            "trainer": "runner-compact-ik-step@2", "transform": TRANSFORM}),
        "dataset": corpus_identity(corpus), "normalization": normalization.to_dict(),
        "acceptance_file_sha256": source["files"][ACCEPTANCE_PATH], "fresh_final": seal,
        "archived_candidates": archived_paths(archived_run),
        "allocations": {str(s): allocation_for(corpus, schedule, s) for s in SEEDS},
        "final_opened": False, "gpu_model_path_qualified": False,
        "next_required_action": "two_independent_exact_source_declaration_admissions"}
    declaration["sha256"] = sha256_json(declaration)
    root = Path(output).resolve()
    checkout = _project(None)
    if root == checkout or checkout in root.parents:
        raise ValueError("run output must remain outside checkout")
    root.mkdir(parents=True, exist_ok=False)
    digest = immutable_json(root / "declaration.json", declaration)
    for seed in SEEDS:
        IKRunLedger.create(root / f"seed-{seed}", identity=identity_for(declaration, seed),
                           allocation=declaration["allocations"][str(seed)])
    return {"declaration_sha256": declaration["sha256"], "declaration_file_sha256": digest,
            "source_sha256": source["source_sha256"], "zero_consumed_training": True,
            "normalization_sha256": normalization.sha256, "final_opened": False}


def load(run, expected_sha256):
    path = Path(run) / "declaration.json"
    if sha256_file(path) != expected_sha256:
        raise ValueError("exact declaration file SHA-256 required")
    value = _read(path)
    if (value.get("schema") != "nca-ik-polar-declaration/v2"
            or value.get("sha256") != sha256_json({k:v for k,v in value.items() if k != "sha256"})
            or value.get("source") != _source(None) or value.get("runtime") != runtime_expectation()
            or value.get("plan") != plan() or value.get("device") != "cuda:0"
            or value.get("qualified_cuda_device") != QUALIFIED_DEVICE
            or value.get("final_opened") is not False
            or value.get("acceptance_file_sha256") != value["source"]["files"][ACCEPTANCE_PATH]
            or value.get("profile_sha256") != runner_right_sole_profile().sha256):
        raise ValueError("polar declaration/source/runtime/plan differs")
    PolarNormalization.from_dict(value["normalization"])
    _titan_visibility(expected_sha256=value["gpu_visibility_sha256"],
                       driver=value["plan"]["qualified_environment"]["driver"])
    return value


def authenticated_corpus(declaration, corpus_root, registry, fresh_root, archived_run):
    corpus = read_polar_followup_corpus(corpus_root, registry,
                                       source_sha256=declaration["source"]["source_sha256"])
    if (corpus_identity(corpus) != declaration["dataset"]
            or fit_training_normalization(corpus).to_dict() != declaration["normalization"]
            or verify_seal(fresh_root) != declaration["fresh_final"]
            or archived_paths(archived_run) != declaration["archived_candidates"]
            or any(allocation_for(corpus, plan(), s) != declaration["allocations"][str(s)] for s in SEEDS)):
        raise ValueError("authenticated follow-up corpus, seal, archived controls or allocation differs")
    return corpus


def cuda_context(declaration, seed):
    source_id, config_id = declaration["source"]["source_sha256"], identity_for(declaration, seed)["config_sha256"]
    preflight = device_report("cuda:0", source_sha256=source_id, config_sha256=config_id)
    capacity = _qualified_titan_gpu_bytes(preflight, plan()["qualified_environment"])
    context = configure_device("cuda:0", seed, memory_fraction=plan()["max_memory_bytes"] / capacity,
                               source_sha256=source_id, config_sha256=config_id)
    if _qualified_titan_gpu_bytes(context.report, plan()["qualified_environment"]) != capacity:
        raise ValueError("TITAN RTX capacity changed during setup")
    if context.report.get("gpu") != declaration["qualified_cuda_device"]:
        raise ValueError("exact qualified TITAN RTX identity/capacity differs")
    return context


def select_training(run, declaration, *, active_ledgers=None):
    """Every report and checkpoint derives from the authenticated event ledger."""
    from nca.evaluation.ik_training import select_tuning_checkpoint
    selected, reports, paths = {}, {}, {}
    for seed in SEEDS:
        ledger = ledger_at(run, declaration, seed) if active_ledgers is None else active_ledgers[seed]
        state = ledger.inspect()
        latest = state["latest_checkpoint"]
        active_ok = (active_ledgers is not None and ledger._attempt is not None
                     and not ledger._attempt.closed and ledger._attempt.sequence == 0
                     and ledger._attempt.pending is None
                     and ledger._attempt.prior["can_resume"]
                     and not ledger._attempt.prior["active"]
                     and not ledger._attempt.prior["unresolved_work"]
                     and state["charged"] == ledger._attempt.prior["charged"])
        if (state["active"] or state["unresolved_work"]) and not active_ok:
            raise ValueError("reconciled idle or authenticated empty active boundaries required")
        if (latest is None
                or latest["state"]["extra_state"].get("completed_epochs") != 40
                or state["consumed_batches"] != 1720
                or state["charged"]["sample_presentations"] != 210400
                or state["charged"]["tuning_calls"] != 80800):
            raise ValueError("all three complete reconciled forty-epoch training histories required")
        history = latest["state"]["extra_state"].get("tuning_history", [])
        if len(history) != 40:
            raise ValueError("every scheduled tuning report required")
        checkpoints = {}
        for path in sorted(ledger.root.glob("attempts/attempt-*/event-*.json")):
            event = _read(path)["record"]
            receipt = event.get("payload", {}).get("receipt", {})
            if event["kind"] == "complete" and "reference" in receipt and "state" in receipt:
                epoch = receipt["state"]["extra_state"].get("completed_epochs", 0)
                checkpoints.setdefault(epoch, []).append(receipt)
                paths[(seed, receipt["reference"]["sha256"])] = receipt["path"]
        records = []
        for epoch, item in enumerate(history, 1):
            report_path = ledger.root / item["file"]
            if sha256_file(report_path) != item["file_sha256"] or item["epoch"] != epoch:
                raise ValueError("scheduled tuning file/epoch changed")
            report = _read(report_path, maximum=32*1024**2)
            if sha256_json(report) != item["report_sha256"]:
                raise ValueError("tuning semantic receipt differs")
            matches = [c for c in checkpoints.get(epoch, ())
                       if c["state"]["completed_optimizer_steps"] == item["step"]
                       and c["state"]["extra_state"].get("tuning_history", [])[-1:] == [item]]
            if not matches:
                raise ValueError("scheduled tuning checkpoint lineage missing")
            records.append({"step": item["step"], "checkpoint_sha256": matches[0]["reference"]["sha256"],
                            "tuning": report})
        choice = selected[seed] = select_tuning_checkpoint(records)
        reports[seed] = next(r["tuning"] for r in records if r["checkpoint_sha256"] == choice["checkpoint_sha256"])
    return selected, reports, paths


def open_final(fresh_root, declaration, selected, reports, *, run, ledgers):
    """The only archive payload reader: fail closed before any file opening."""
    from nca.evaluation.ik_training import freeze_candidates
    gate = tuning_stop_gate(reports)
    if not gate["final_opening_allowed"]:
        raise ValueError("tuning stop gate refused; final remains sealed")
    actual_selected, actual_reports, _ = select_training(run, declaration, active_ledgers=ledgers)
    if actual_selected != selected or actual_reports != reports:
        raise ValueError("final gate inputs differ from authenticated tuning/checkpoint lineage")
    seal = verify_seal(fresh_root)
    if seal != declaration["fresh_final"]:
        raise ValueError("prospective final seal differs")
    archive_path = Path(fresh_root) / "sealed-final.tar.gz"
    with archive_path.open("rb") as stream:
        archive_bytes = stream.read(seal["archive_bytes"] + 1)
    if (len(archive_bytes) != seal["archive_bytes"]
            or hashlib.sha256(archive_bytes).hexdigest() != seal["archive_sha256"]):
        raise ValueError("fresh archive bytes changed before final opening")
    config = FreshFinalConfig()
    # The scorer's freeze binds its final geometry. The outer opening separately
    # retains the original training-config selections without altering them.
    final_choices = {s: dict(choice, config_sha256=config.sha256) for s, choice in selected.items()}
    frozen = freeze_candidates(final_choices, experiment=declaration["id"],
        dataset_manifest_sha256=seal["sha256"], final_suite_identity_sha256=seal["archive_sha256"],
        registry_id=seal["registry_id"])
    opening = {"schema": "nca-ik-final-opening/v1", "experiment": declaration["id"],
        "frozen_candidates_sha256": [c["checkpoint_sha256"] for c in frozen["candidates"]],
        "suite_identity_sha256": seal["archive_sha256"], "dataset_manifest_sha256": seal["sha256"],
        "registry_id": seal["registry_id"], "future_experiments_status": "development_only"}
    opening["sha256"] = sha256_json(opening)
    immutable_json(Path(fresh_root) / "opened.json", {"declaration_sha256": declaration["sha256"],
        "training_selections": {str(s): c for s,c in selected.items()}, "tuning_gate": gate,
        "frozen": frozen, "opening": opening, "created_utc": utc_now()})
    samples = []
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        if [m.name for m in members] != [f"final-{i:05d}.jsonl" for i in range(46)]:
            raise ValueError("fresh final archive member schedule differs")
        for member in members:
            if not member.isfile() or member.size > 16*1024**2:
                raise ValueError("fresh final archive member type/size differs")
            for line in archive.extractfile(member):
                sample = GeometricSample.from_dict(strict_json_loads(line, max_bytes=256*1024))
                validate_sample(sample, config)
                samples.append(sample)
    if [s.task.to_dict() for s in samples] != [t.to_dict() for t in iter_tasks(config, "final")]:
        raise ValueError("fresh final ordered rows differ")
    validate_dataset(samples, config, require_complete=True)
    return tuple(samples), config, opening, frozen


def finish_experiment(run, declaration, corpus, fresh_root, archived_run, ledgers, attempts):
    from nca.ik.checkpoint import load_checkpoint
    from nca.ik.models import build_model as build_original
    from nca.evaluation.ik_training import (evaluate_comparison, model_predictor, reference_predictors,
        collect_predictions, score_raw, measure_cpu_latency, apply_cpu_utility, aggregate_comparisons)
    # Selection is reconstructed before the supervisor opens evaluation attempts.
    selected, reports, paths = select_training(run, declaration, active_ledgers=ledgers)
    gate = tuning_stop_gate(reports)
    immutable_json(Path(run) / "tuning-stop.json", dict(gate,
        declaration_sha256=declaration["sha256"], selections={str(s):c for s,c in selected.items()}))
    if not gate["final_opening_allowed"]:
        for attempt in attempts.values():
            attempt.finish("completed", reason="bounded_followup_tuning_no_go_final_sealed", publication_reserve_seconds=60.)
        return gate
    samples, config, opening, frozen = open_final(fresh_root, declaration, selected, reports,
                                                run=run, ledgers=ledgers)
    results = {}
    normalization = PolarNormalization.from_dict(declaration["normalization"])
    for seed in SEEDS:
        ledger, attempt = ledgers[seed], attempts[seed]
        trainer = build_trainer(cuda_context(declaration, seed), normalization, corpus.config)
        sampler = PairPreservingSampler(corpus.training, seed, batch_size=128, epochs=40)
        extra = ledger.resume(trainer, sampler)
        if not sampler.exhausted:
            raise ValueError("final evaluation cannot optimize an incomplete candidate")
        token = attempt.reserve("evaluation", calls=78240, call_id="fresh-final-six-controls-and-timing")
        cpu = configure_device("cpu", seed, source_sha256=declaration["source"]["source_sha256"],
                               config_sha256=identity_for(declaration, seed)["config_sha256"])
        cpu.torch.set_num_threads(1)
        def model_at(root, relative, digest, original=False):
            marker = _read(Path(root) / (relative + ".json"))["checkpoint"]
            payload = load_checkpoint(Path(root) / relative, expected_sha256=digest,
                                      expected_metadata=marker["metadata"] if original else ledger.identity)
            model = build_original(cpu, corpus.normalization) if original else build_model(cpu, normalization)
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
            return model
        choice = selected[seed]
        trained = model_at(ledger.root, paths[(seed, choice["checkpoint_sha256"])], choice["checkpoint_sha256"])
        initial = ledger.inspect()["latest_checkpoint"]["state"]["extra_state"]["initial_checkpoint"]
        untrained = model_at(ledger.root, initial["path"], initial["reference"]["sha256"])
        archived = declaration["archived_candidates"][str(seed)]
        old_model = model_at(Path(archived_run) / f"seed-{seed}", archived["path"], archived["sha256"], True)
        predictors = {"trained_raw_reference": model_predictor(cpu, trained),
            "matched_untrained_initialization": model_predictor(cpu, untrained), **reference_predictors(config)}
        comparison = evaluate_comparison(samples, config, trained=predictors["trained_raw_reference"],
            untrained=predictors["matched_untrained_initialization"], partition="final",
            expected_task_ids=[s.task.id for s in samples], final_opening=opening, frozen_candidates=frozen, seed=seed)
        old_predictor = model_predictor(cpu, old_model)
        old_report = score_raw(samples, collect_predictions(samples, old_predictor), config, partition="final",
            expected_task_ids=[s.task.id for s in samples], final_opening=opening, frozen_candidates=frozen, seed=seed)
        old_report = dict(old_report, comparator="archived_trained_v1",
            frozen_checkpoint_sha256=archived["sha256"], diagnostic_only=True,
            suite_opening_sha256=opening["sha256"])
        nominal = tuple(s.task.inputs for s in samples if s.task.state.kind != "stress")
        timing = measure_cpu_latency(nominal, predictors, context=cpu, environment=cpu.report,
                                    archived_predictor=old_predictor)
        archived_timing = timing["comparators"].pop("archived_trained_v1")
        comparison = apply_cpu_utility(comparison, timing)
        comparison["archived_trained_v1"] = {"checkpoint_sha256": archived["sha256"],
                                              "score": old_report, "cpu_latency": archived_timing}
        digest = immutable_json(Path(run) / f"fresh-final-comparison-{seed}.json", comparison)
        attempt.complete(token, {"report_file_sha256": digest, "report_sha256": sha256_json(comparison),
                                 "scheduled_final_decisions": 17640, "timing_calls": 60600})
        extra["fresh_final_evaluation"] = {"report_file_sha256": digest,
            "report_sha256": sha256_json(comparison), "device_report": trainer.context.report,
            "memory": trainer.context.memory_stats(), "frozen_candidates_sha256": frozen["sha256"]}
        attempt.checkpoint(trainer, sampler.state_dict(), extra_state=extra)
        results[seed] = comparison
    aggregate = aggregate_comparisons(results)
    immutable_json(Path(run) / "fresh-three-seed-comparison.json", aggregate)
    for attempt in attempts.values():
        attempt.finish("completed", reason="bounded_followup_final_evaluation_complete", publication_reserve_seconds=60.)
    return aggregate


def worker(args):
    path = Path(args.run) / "declaration.json"
    if sha256_file(path) != args.declaration_sha256:
        raise ValueError("allocation declaration bytes changed after launch")
    declaration = _read(path)
    claim = _read(Path(args.run) / "active-phase.json")
    serial = _read(Path(args.run) / "serial-phase-active.json")
    if (claim.get("id") != args.phase_claim
            or claim.get("declaration_sha256") != declaration["sha256"]
            or claim.get("phase") != args.phase or claim.get("seed") != args.seed
            or serial.get("declaration_file_sha256") != args.declaration_sha256
            or serial.get("phase") != args.phase or serial.get("seed") != args.seed):
        raise ValueError("worker requires the exact live run-wide supervisor ownership")
    seeds = SEEDS if args.phase == "finish" else (args.seed,)
    ledgers, attempts = {}, {}
    try:
        # Charge launch/authentication/refusals before parsing corpus or CUDA.
        for seed in seeds:
            ledger = ledgers[seed] = ledger_at(args.run, declaration, seed)
            attempts[seed] = ledger.begin_attempt(started=args.started)
        declaration = load(args.run, args.declaration_sha256)
        if sha256_json(_review(args.review, declaration)) != args.review_sha256:
            raise ValueError("admission changed after supervisor publication")
        corpus = authenticated_corpus(declaration, args.corpus, args.registry, args.fresh, args.archived_run)
        if args.phase == "train":
            seed = args.seed
            context = cuda_context(declaration, seed)
            result = train_attempt(ledgers[seed], attempts[seed], corpus, context, epochs=40, batch_size=128,
                polar_normalization=PolarNormalization.from_dict(declaration["normalization"]),
                execution={"declaration_sha256": declaration["sha256"], "source_sha256": declaration["source"]["source_sha256"]})
            immutable_json(ledgers[seed].root / f"training-result-{attempts[seed].number:04d}.json", result)
        else:
            result = finish_experiment(args.run, declaration, corpus, args.fresh, args.archived_run, ledgers, attempts)
        return result
    except BaseException as exc:
        for attempt in attempts.values():
            if not attempt.closed:
                attempt.finish("failed", reason=type(exc).__name__, publication_reserve_seconds=60.)
        raise


def supervise(args):
    from nca.ik.phase_owner import serial_phase
    with serial_phase(args.run, phase=args.phase, seed=args.seed,
                      declaration_sha256=args.declaration_sha256):
        return _supervise_owned(args)


def _supervise_owned(args):
    started = ENTRY_STARTED
    declaration = load(args.run, args.declaration_sha256)
    review = _review(args.review, declaration)
    claim_id = uuid.uuid4().hex
    claim = {"schema": "nca-ik-polar-phase-claim/v2", "id": claim_id,
        "declaration_sha256": declaration["sha256"], "phase": args.phase,
        "seed": args.seed, "created_utc": utc_now()}
    lock = Path(args.run) / "active-phase.json"
    # Atomic run-wide ownership spans every seed, startup, device work and CPU
    # timing. A crash retains the lock and durable claim for independent review.
    with lock.open("xb") as stream:
        stream.write((canonical_json(claim)+"\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    claims = Path(args.run) / "phase-claims"
    claims.mkdir(exist_ok=True)
    immutable_json(claims / f"{claim_id}-start.json", claim)
    for other_seed in SEEDS:
        other = ledger_at(args.run, declaration, other_seed)
        state = other.inspect()
        if state["active"] or state["unresolved_work"] or any(other.root.glob("supervisor-overrun-*.json")):
            raise ValueError("another seed has pending/unknown work; phase ownership retained")
        for previous in other.root.glob("supervisor-start-*.json"):
            terminal = previous.with_name(previous.name.replace("start", "result", 1))
            if not terminal.exists() or _read(terminal).get("verified_within_charged_wall") is not True:
                raise ValueError("another seed has an uncertain supervisor; phase ownership retained")
    review_path = Path(args.run) / "admission-review.json"
    if review_path.exists():
        if _read(review_path) != review:
            raise ValueError("continuation review differs")
    else:
        immutable_json(review_path, review)
    # A single ownership claim covers every seed and matched CPU benchmark.
    # All prior phases must be reconciled, including a different selected seed.
    for prior_seed in SEEDS:
        prior_ledger = ledger_at(args.run, declaration, prior_seed)
        prior_state = prior_ledger.inspect()
        if prior_state["active"] or prior_state["unresolved_work"] or any(prior_ledger.root.glob("supervisor-overrun-*.json")):
            raise ValueError("another seed phase is active or uncertain")
        for prior_start in prior_ledger.root.glob("supervisor-start-*.json"):
            terminal = prior_start.with_name(prior_start.name.replace("start", "result", 1))
            if not terminal.exists() or _read(terminal).get("verified_within_charged_wall") is not True:
                raise ValueError("another seed supervisor is not reconciled")
    seeds = SEEDS if args.phase == "finish" else (args.seed,)
    before = {}
    for seed in seeds:
        ledger = ledger_at(args.run, declaration, seed)
        state = before[seed] = ledger.inspect()
        if not state["can_resume"] or any(ledger.root.glob("supervisor-overrun-*.json")):
            raise ValueError("allocation cannot safely resume")
        for previous in ledger.root.glob("supervisor-start-*.json"):
            terminal = previous.with_name(previous.name.replace("start", "result", 1))
            if not terminal.exists() or _read(terminal).get("verified_within_charged_wall") is not True:
                raise ValueError("previous supervisor outcome uncertain")
    for seed, state in before.items():
        ledger = ledger_at(args.run, declaration, seed)
        immutable_json(ledger.root / f"supervisor-start-{state['attempt_count']+1:04d}.json", {
            "declaration_sha256": declaration["sha256"], "phase": args.phase,
            "admission_sha256": sha256_json(review), "created_utc": utc_now()})
    maximum = min(s["remaining_wall_seconds"] for s in before.values())
    command = [sys.executable, "-m", "nca.train.ik_followup", "_worker", "--phase", args.phase,
        "--run", args.run, "--declaration-sha256", args.declaration_sha256, "--seed", str(args.seed),
        "--corpus", args.corpus, "--registry", args.registry, "--fresh", args.fresh,
        "--archived-run", args.archived_run, "--review", str(review_path),
        "--review-sha256", sha256_json(review), "--started", repr(started)]
    command.extend(["--phase-claim", claim_id])
    log = Path(args.run) / f"{args.phase}-{args.seed}-attempt-{before[args.seed]['attempt_count']+1:04d}-worker.log"
    with log.open("xb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=stream)
        timed_out = False
        try:
            process.wait(timeout=max(.001, maximum - (time.monotonic()-started) - 5))
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            timed_out = True
            process.kill()
            process.wait(timeout=5)
    elapsed = time.monotonic() - started
    receipts = {}
    for seed in seeds:
        ledger = ledger_at(args.run, declaration, seed)
        state = ledger.inspect()
        charge = state["charged_wall_seconds"] - before[seed]["charged_wall_seconds"]
        verified = (not timed_out and process.returncode == 0 and not state["active"]
                    and not state["unresolved_work"] and elapsed <= charge <= maximum)
        receipt = receipts[str(seed)] = {"schema": "nca-ik-polar-supervisor/v2", "phase": args.phase,
            "seed": seed, "returncode": process.returncode, "timed_out": timed_out,
            "whole_process_seconds": elapsed, "charged_wall_seconds": charge,
            "verified_within_charged_wall": verified, "unknown_work_blocks_resume": not verified}
        index = before[seed]["attempt_count"]+1
        immutable_json(ledger.root / f"supervisor-result-{index:04d}.json", receipt)
        if not verified:
            immutable_json(ledger.root / f"supervisor-overrun-{index:04d}.json", {"resume_forbidden": True})
        if time.monotonic() - started > charge:
            if not (ledger.root / f"supervisor-overrun-{index:04d}.json").exists():
                immutable_json(ledger.root / f"supervisor-overrun-{index:04d}.json", {"resume_forbidden": True})
            verified = False
        if not verified:
            raise ValueError("failed/uncertain supervised attempt retained; no automatic retry")
    immutable_json(claims / f"{claim_id}-result.json", {"claim_id": claim_id,
        "verified": True, "supervisor_receipts": receipts, "whole_phase_seconds": time.monotonic()-started})
    if any(time.monotonic()-started > r["charged_wall_seconds"] for r in receipts.values()):
        raise ValueError("phase publication exceeded charged reserve; ownership retained")
    if _read(lock) != claim:
        raise ValueError("phase ownership changed")
    lock.unlink()
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "inspect", "run", "_worker"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--declaration-sha256")
    parser.add_argument("--corpus")
    parser.add_argument("--registry")
    parser.add_argument("--fresh")
    parser.add_argument("--archived-run")
    parser.add_argument("--review")
    parser.add_argument("--phase", choices=("train", "finish"), default="train")
    parser.add_argument("--seed", choices=SEEDS, type=int, default=41011)
    parser.add_argument("--started", type=float)
    parser.add_argument("--review-sha256")
    parser.add_argument("--phase-claim")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.corpus, args.registry, args.run, args.fresh, args.archived_run)
    elif args.command == "inspect":
        declaration = load(args.run, args.declaration_sha256)
        result = {"declaration_sha256": declaration["sha256"], "source": declaration["source"],
                  "ledgers": {str(s): ledger_at(args.run, declaration, s).inspect() for s in SEEDS}}
    elif args.command == "_worker":
        result = worker(args)
    else:
        result = supervise(args)
    print(canonical_json(result))


if __name__ == "__main__":
    main()

# Copyright 2026 techrote. All rights reserved.
