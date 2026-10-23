"""One supervised, candidate-bound final comparison within the original budgets.

All three completed training views are authenticated before the final opening.
The same worker retains that view, reserves every comparison/timing call, and
keeps the original training checkpoint lineage while recording amended source.
No optimization, candidate reselection or final-driven retry is permitted.
"""
from __future__ import annotations

import time
ENTRY_STARTED = time.monotonic()

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

from nca.contracts import sha256_json
from nca.runs import immutable_json, utc_now
from nca.train.ik_experiment import (SEEDS, _read, identity_for, corpus_identity,
    _titan_visibility, _qualified_titan_gpu_bytes)
from nca.train.ik_run_state import IKRunLedger
from nca.train.ik_corpus import read_training_corpus, PairPreservingSampler
from nca.ik.device import configure_device, device_report
from nca.ik.models import build_model
from nca.ik.checkpoint import load_checkpoint
from nca.train.ik import IKTrainer
from nca.evaluation.ik_training import (select_tuning_checkpoint, freeze_candidates,
    evaluate_comparison, model_predictor, reference_predictors, measure_cpu_latency,
    apply_cpu_utility, aggregate_comparisons, _refusal)
from nca.evaluation.ik_final_corpus import open_final_corpus, _verified_training


def _ledger(run, declaration, seed):
    return IKRunLedger.open(Path(run) / f"seed-{seed}",
        expected_identity=identity_for(declaration, seed),
        expected_allocation=declaration["allocations"][str(seed)])


def freeze_training(run, declaration, corpus):
    """Reconstruct every tuning-only selection from authenticated ledger events."""
    choices, paths = {}, {}
    for seed in SEEDS:
        ledger = _ledger(run, declaration, seed)
        state = ledger.inspect()
        latest = state["latest_checkpoint"]
        if latest is None or state["active"] or state["unresolved_work"]:
            raise ValueError("three reconciled completed training boundaries required")
        history = latest["state"]["extra_state"].get("tuning_history", ())
        if len(history) != 40:
            raise ValueError("all forty scheduled tuning reports required")
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
            if hashlib.sha256(report_path.read_bytes()).hexdigest() != item["file_sha256"]:
                raise ValueError("scheduled tuning file changed")
            report = _read(report_path, maximum=32*1024*1024)
            if sha256_json(report) != item["report_sha256"]:
                raise ValueError("scheduled tuning semantic receipt changed")
            matches = [c for c in checkpoints.get(epoch, ())
                if c["state"]["completed_optimizer_steps"] == item["step"]
                and c["state"]["extra_state"].get("tuning_history", [])[-1:] == [item]]
            if not matches:
                raise ValueError("tuning checkpoint lineage missing")
            records.append({"step": item["step"], "checkpoint_sha256": matches[0]["reference"]["sha256"],
                            "tuning": report})
        choices[seed] = select_tuning_checkpoint(records)
    frozen = freeze_candidates(choices, experiment=declaration["id"],
        dataset_manifest_sha256=corpus.manifest_sha256,
        final_suite_identity_sha256=corpus.final_suite_identity_sha256,
        registry_id=corpus.registry_id)
    _verified_training(run, declaration, corpus, frozen)
    return frozen, paths


def worker(run, amendment_path, amendment_digest, corpus_root, registry, review_path,
           archive, *, started, review_sha256):
    from nca.train.ik_continuation import load_amendment, _review
    declaration, amendment = load_amendment(run, amendment_path, amendment_digest,
                                           allow_pending_evaluation=True)
    if sha256_json(_review(review_path, amendment)) != review_sha256:
        raise ValueError("continuation admission changed after supervision")
    execution = {"amendment_sha256": amendment["sha256"],
                 "source_sha256": amendment["source"]["source_sha256"],
                 "source_commit": amendment["source"]["revision"]}
    _titan_visibility(expected_sha256=declaration["gpu_visibility_sha256"],
        driver=declaration["plan"]["qualified_environment"]["driver"])
    corpus = read_training_corpus(corpus_root, registry)
    if corpus_identity(corpus) != declaration["dataset"]:
        raise ValueError("original corpus identity changed")
    frozen, paths = freeze_training(run, declaration, corpus)
    out = Path(amendment_path).parent
    immutable_json(out / "frozen-candidates.json", frozen)
    # No attempt exists yet: the final reader independently proves all three
    # completed, inactive training ledgers. Opening overhead is charged from
    # the common supervisor start time when the attempts begin below.
    final = open_final_corpus(corpus_root, registry, run,
        hashlib.sha256((Path(run)/"declaration.json").read_bytes()).hexdigest(), frozen,
        training_corpus=corpus, amendment_sha256=amendment["sha256"],
        source_sha256=execution["source_sha256"], archive=archive)
    ledgers, attempts, saved, trainers, samplers = {}, {}, {}, {}, {}
    try:
        for seed in SEEDS:
            ledger = _ledger(run, declaration, seed)
            info = ledger.inspect()
            index = info["attempt_count"] + 1
            immutable_json(ledger.root / f"supervisor-start-{index:04d}.json", {
                "schema": "nca-ik-continuation-supervisor-start/v1", "phase": "evaluate-all",
                "declaration_sha256": declaration["sha256"], "execution": execution,
                "seed": seed, "attempt": index, "remaining_wall_seconds": info["remaining_wall_seconds"],
                "admission_review_sha256": review_sha256, "created_utc": utc_now()})
            ledgers[seed] = ledger
            attempt = attempts[seed] = ledger.begin_attempt(started=started)
            config_id = identity_for(declaration, seed)["config_sha256"]
            preflight = device_report("cuda:0", source_sha256=execution["source_sha256"], config_sha256=config_id)
            capacity = _qualified_titan_gpu_bytes(preflight, declaration["plan"]["qualified_environment"])
            context = configure_device("cuda:0", seed,
                memory_fraction=declaration["plan"]["max_memory_bytes"] / capacity,
                source_sha256=execution["source_sha256"], config_sha256=config_id)
            trainer = trainers[seed] = IKTrainer(context, corpus.normalization, corpus.config)
            sampler = samplers[seed] = PairPreservingSampler(corpus.training, seed, batch_size=128, epochs=40)
            saved[seed] = ledger.resume(trainer, sampler)
            if saved[seed].get("execution") != execution or not sampler.exhausted:
                raise ValueError("completed amended training checkpoint required")
            # Every actual final and timing predictor call is reserved before
            # any comparator invocation. No call can evade the original cap.
            token = attempt.reserve("evaluation", calls=65200, call_id="sealed-final-and-cpu-latency")
            saved[seed]["evaluation_reservation"] = token
        reports = {}
        for seed in SEEDS:
            ledger, attempt = ledgers[seed], attempts[seed]
            cpu = configure_device("cpu", seed, source_sha256=execution["source_sha256"],
                                   config_sha256=identity_for(declaration, seed)["config_sha256"])
            cpu.torch.set_num_threads(1)
            selected = next(x for x in frozen["candidates"] if x["seed"] == seed)
            def model_at(relative, digest):
                payload = load_checkpoint(ledger.root / relative, expected_sha256=digest,
                                          expected_metadata=ledger.identity)
                model = build_model(cpu, corpus.normalization)
                model.load_state_dict(payload["model"], strict=True)
                model.eval()
                return model
            trained = model_at(paths[(seed, selected["checkpoint_sha256"])], selected["checkpoint_sha256"])
            initial = saved[seed]["initial_checkpoint"]
            untrained = model_at(initial["path"], initial["reference"]["sha256"])
            predictors = {"trained_raw_reference": model_predictor(cpu, trained),
                "matched_untrained_initialization": model_predictor(cpu, untrained),
                **reference_predictors(corpus.config)}
            comparison = evaluate_comparison(final.samples, final.config,
                trained=predictors["trained_raw_reference"], untrained=predictors["matched_untrained_initialization"],
                partition="final", expected_task_ids=[x.task.id for x in final.samples],
                final_opening=final.opening, frozen_candidates=frozen, seed=seed, authenticated_final=final)
            nominal = tuple(x.task.inputs for x in final.samples if x.task.state.kind != "stress")
            timing = measure_cpu_latency(nominal, predictors, context=cpu, environment=cpu.report)
            comparison = apply_cpu_utility(comparison, timing)
            digest = immutable_json(out / f"final-comparison-{seed}.json", comparison)
            refused = sum(_refusal(x.task) is not None for x in final.samples)
            receipt = {"file_sha256": digest, "report_sha256": sha256_json(comparison),
                "final_scheduled_decisions": 14700,
                "final_predictor_invocations": 5*(len(final.samples)-refused),
                "final_guarded_refusals": 5*refused,
                "timing_predictor_invocations": 50500, "execution": execution,
                "accuracy": comparison["accuracy"], "learned_contribution": comparison["learned_contribution"],
                "practical_utility": comparison["practical_utility"]}
            attempt.complete(saved[seed].pop("evaluation_reservation"), receipt)
            memory = trainers[seed].context.memory_stats()
            if memory["peak_reserved_bytes"] > declaration["plan"]["max_memory_bytes"]:
                raise ValueError("TITAN allocator ceiling exceeded")
            saved[seed]["final_evaluation"] = dict(receipt, memory=memory,
                device_report=trainers[seed].context.report, frozen_candidates_sha256=frozen["sha256"])
            attempt.checkpoint(trainers[seed], samplers[seed].state_dict(), extra_state=saved[seed])
            reports[seed] = comparison
        aggregate = aggregate_comparisons(reports)
        immutable_json(out / "three-seed-comparison.json", dict(aggregate, execution=execution))
        for seed in SEEDS:
            attempts[seed].finish("completed", reason="declared_training_and_sealed_final_evaluation_complete",
                                  publication_reserve_seconds=60.)
        return aggregate
    except BaseException as exc:
        for attempt in attempts.values():
            if not attempt.closed:
                attempt.finish("failed", reason=type(exc).__name__, publication_reserve_seconds=60.)
        raise


def supervise(run, amendment_path, amendment_digest, corpus_root, registry, review_path,
              archive, *, started=None):
    from nca.train.ik_continuation import load_amendment, _review
    started = time.monotonic() if started is None else started
    declaration, amendment = load_amendment(run, amendment_path, amendment_digest)
    review = _review(review_path, amendment)
    review_digest = sha256_json(review)
    out = Path(amendment_path).parent
    frozen_review = out / "admission-review.json"
    if frozen_review.exists():
        if _read(frozen_review) != review:
            raise ValueError("immutable continuation admission changed")
    else:
        immutable_json(frozen_review, review)
    before = {s: _ledger(run, declaration, s).inspect() for s in SEEDS}
    if any(not x["can_resume"] or x["charged"]["evaluation_calls"] != 0 for x in before.values()):
        raise ValueError("all three fresh, resumable final evaluation allocations required")
    remaining = min(x["remaining_wall_seconds"] for x in before.values())
    immutable_json(out / "evaluation-supervisor-start.json", {
        "schema": "nca-ik-continuation-final-supervisor/v1", "amendment_sha256": amendment["sha256"],
        "admission_review_sha256": review_digest, "remaining_wall_seconds": remaining,
        "created_utc": utc_now()})
    command = [sys.executable, "-m", "nca.evaluation.ik_continuation_eval", "_worker",
        "--run", str(run), "--amendment", str(amendment_path), "--amendment-sha256", amendment_digest,
        "--corpus", str(corpus_root), "--registry", str(registry), "--review", str(frozen_review),
        "--archive", str(archive), "--started", repr(started), "--review-sha256", review_digest]
    with (out / "evaluation-worker.log").open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        timed_out = False
        try:
            process.wait(timeout=max(.001, remaining-(time.monotonic()-started)-5))
        except BaseException:
            timed_out = True
            process.kill()
            process.wait(timeout=5)
    elapsed = time.monotonic()-started
    results = {}
    for seed in SEEDS:
        after = _ledger(run, declaration, seed).inspect()
        charge = after["charged_wall_seconds"]-before[seed]["charged_wall_seconds"]
        verified = (process.returncode == 0 and not timed_out and not after["active"]
            and not after["unresolved_work"] and after["last_status"] == "completed"
            and elapsed <= charge and elapsed <= before[seed]["remaining_wall_seconds"])
        result = {"schema": "nca-ik-continuation-supervisor-result/v1", "seed": seed,
            "attempt": before[seed]["attempt_count"]+1, "phase": "evaluate-all",
            "returncode": process.returncode, "timed_out": timed_out,
            "whole_process_seconds": elapsed, "charged_wall_seconds": charge,
            "verified_within_charged_wall": verified, "unknown_work_blocks_resume": not verified,
            "consumed_or_reserved_wall_seconds": charge if verified else before[seed]["remaining_wall_seconds"],
            "amendment_sha256": amendment["sha256"], "source_sha256": amendment["source"]["source_sha256"]}
        results[str(seed)] = result
        target = Path(run)/f"seed-{seed}"/f"supervisor-result-{result['attempt']:04d}.json"
        immutable_json(target, result)
    terminal = {"schema": "nca-ik-continuation-final-supervisor-result/v1", "seeds": results,
                "verified_within_charged_wall": all(x["verified_within_charged_wall"] for x in results.values()),
                "unknown_work_blocks_resume": any(x["unknown_work_blocks_resume"] for x in results.values())}
    immutable_json(out / "evaluation-supervisor-result.json", terminal)
    published_elapsed = time.monotonic()-started
    overrun = [seed for seed in SEEDS if
        published_elapsed > results[str(seed)]["charged_wall_seconds"]
        or published_elapsed > before[seed]["remaining_wall_seconds"]]
    if overrun:
        immutable_json(out / "evaluation-supervisor-overrun.json", {
            "schema": "nca-ik-continuation-final-overrun/v1", "resume_forbidden": True,
            "whole_process_publication_seconds": published_elapsed, "seeds": overrun})
        for seed in overrun:
            index = before[seed]["attempt_count"]+1
            immutable_json(Path(run)/f"seed-{seed}"/f"supervisor-overrun-{index:04d}.json", {
                "schema": "nca-ik-continuation-final-overrun/v1", "resume_forbidden": True})
        raise ValueError("final supervisor publication exceeded charged reserve; stop")
    if terminal["unknown_work_blocks_resume"]:
        raise ValueError("final worker outcome failed or uncertain; preserve receipts and stop")
    return terminal


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "_worker"))
    for name in ("run", "amendment", "amendment-sha256", "corpus", "registry", "review", "archive"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--started", type=float)
    p.add_argument("--review-sha256")
    a = p.parse_args(argv)
    args = (a.run, a.amendment, a.amendment_sha256, a.corpus, a.registry, a.review, a.archive)
    if a.command == "run":
        supervise(*args, started=ENTRY_STARTED)
    else:
        worker(*args, started=a.started, review_sha256=a.review_sha256)


if __name__ == "__main__":
    main()

# Copyright 2026 techrote. All rights reserved.
