"""IK54 development sweep v1. Explicit prepare/run/confirm; never runs Part B.

Only the existing authenticated training/tuning reader accesses corpus storage.
All output paths must be new and external. Historical launchers are unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from nca.contracts import canonical_json, sha256_json
from nca.runs import immutable_json, sha256_file, utc_now
from nca.train.ik_dev import (DevConfig, SEEDS, SELECTION_RULE, grid, split, sampler,
                              epoch_batches, batched_predictions, dev_score, parity,
                              point_stage_rms, rank_configs)
from nca.train.ik_experiment import _source, _titan_visibility, _read, ACCEPTANCE_PATH
from nca.train.ik_matched_prepare import TITAN_FINGERPRINT, GIB


def read_corpus(packet):
    from nca.train.ik_corpus import read_polar_followup_corpus
    return read_polar_followup_corpus(packet["corpus"], packet["registry"],
                                      source_sha256=packet["source"]["source_sha256"])


def freeze(path, value):
    immutable_json(Path(path), value)
    return sha256_file(path)


def append(stream, value):
    stream.write(canonical_json(value) + "\n")
    stream.flush()


def check_packet(root):
    root = Path(root)
    packet = _read(root / "packet.json", 16*1024**2)
    source = _source(Path(__file__).resolve().parents[3])
    if packet["source"] != source:
        raise ValueError("development executable source differs from frozen packet")
    for name in ("split", "selection-rule"):
        if sha256_file(root / (name+".json")) != packet[name+"_file_sha256"]:
            raise ValueError("frozen development file changed")
    _titan_visibility(expected_sha256=TITAN_FINGERPRINT)
    return packet


def prepare(args):
    root = Path(args.root).resolve()
    checkout = Path(__file__).resolve().parents[3]
    if root.exists() or root.is_relative_to(checkout):
        raise ValueError("new external root required")
    _titan_visibility(expected_sha256=TITAN_FINGERPRINT)
    source = _source(checkout)
    packet = {"schema": "nca-ik-dev-packet/v1", "source": source,
              "corpus": str(Path(args.corpus).resolve()), "registry": str(Path(args.registry).resolve()),
              "evidence": "development evidence; no acceptance claim", "created_utc": utc_now(),
              "seeds": list(SEEDS), "grid": [c.to_dict() for c in grid()],
              "gpu_visibility_sha256": TITAN_FINGERPRINT, "max_workers": 8,
              "allocator_bytes_per_worker": GIB, "board_memory_cap_bytes": 20*GIB,
              "host_commit_cap_per_process_bytes": 8*GIB,
              "max_wall_seconds": 12*3600, "max_attempts_per_run": 3,
              "final_calls": 0, "part_b_authorized": False,
              "owner_authority": "2026-10-23 owner prompt: #54 IK development sweep, then one strict qualification step; D-032",
              "acceptance_file_sha256": sha256_file(checkout/ACCEPTANCE_PATH)}
    corpus = read_corpus(packet)
    from nca.ik.polar import fit_training_normalization
    normalization = fit_training_normalization(corpus)
    train, dev, split_record = split(corpus.training)
    if len(split_record["family_order"]) != 70:
        raise ValueError("expected original 70 training families")
    # The original normalization is recomputed for verification only, never on a subset.
    packet.update(corpus_identity=corpus.identity, normalization=normalization.to_dict(),
                  train_rows=len(train), dev_rows=len(dev), original_training_rows=len(corpus.training))
    root.mkdir(parents=True)
    packet["split_file_sha256"] = freeze(root/"split.json", split_record)
    packet["selection-rule_file_sha256"] = freeze(root/"selection-rule.json", SELECTION_RULE)
    packet["full_v2_schedule_hashes"] = {str(seed): sampler(corpus.training, seed, DevConfig()).schedule_sha256
                                          for seed in SEEDS}
    freeze(root/"packet.json", packet)
    print(canonical_json({"root": str(root), "split": packet["split_file_sha256"],
                          "selection": packet["selection-rule_file_sha256"],
                          "train_rows": len(train), "dev_rows": len(dev)}))


def cuda_context(seed):
    _titan_visibility(expected_sha256=TITAN_FINGERPRINT)
    from nca.ik.device import configure_device, device_report
    report = device_report("cuda:0")
    gpu = report.get("gpu") or {}
    if (report["status"] != "ready" or gpu.get("name") not in ("TITAN RTX", "NVIDIA TITAN RTX")
            or gpu.get("capability") != [7, 5] or gpu.get("multiprocessors") != 72
            or report.get("driver") != "581.80"):
        raise ValueError("TITAN runtime metadata mismatch")
    context = configure_device("cuda:0", seed, memory_fraction=GIB/gpu["total_memory_bytes"])
    context.torch.set_num_interop_threads(1)
    return context


def checkpoint(context, trainer, path, metadata):
    torch = context.torch
    state = {"model": trainer.model.state_dict(), "optimizer": trainer.optimizer.state_dict(),
             "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
             "metadata": metadata}
    # Independent canonical tensor bytes identity supplements the file hash.
    digest = hashlib.sha256()
    for key, tensor in sorted(state["model"].items()):
        digest.update(key.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(canonical_json(list(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    with Path(path).open("xb") as stream:
        torch.save(state, stream)
    return {"file": Path(path).name, "file_sha256": sha256_file(path), "model_tensor_sha256": digest.hexdigest()}


def worker(args):
    began = time.monotonic()
    root = Path(args.root)
    packet = check_packet(root)
    from nca.train.ik_gpu_ownership import verify
    verify(_read(root/"ownership.json"))
    config = DevConfig(**packet["grid"][args.index])
    output = root/"runs"/config.id/f"seed-{args.seed}"/f"attempt-{args.attempt}"
    output.mkdir(parents=True, exist_ok=False)
    freeze(output/"started.json", {"utc": utc_now(), "pid": os.getpid(), "seed": args.seed,
                                  "config": config.to_dict(), "attempt": args.attempt,
                                  "packet_file_sha256": sha256_file(root/"packet.json")})
    try:
        corpus = read_corpus(packet)
        train, dev, record = split(corpus.training)
        if sha256_json(record) != sha256_json(_read(root/"split.json", 16*1024**2)):
            raise ValueError("frozen split differs")
        from nca.ik.polar import PolarNormalization
        normalization = PolarNormalization.from_dict(packet["normalization"])
        context = cuda_context(args.seed)
        from nca.train.ik_matched_supervisor import process_metrics
        from nca.train.ik_dev_trainer import DevTrainer, PreparedRows
        trainer = DevTrainer(context, normalization, corpus.config, config)
        plan = sampler(train, args.seed, config)
        prepared = PreparedRows(context, corpus, train)
        freeze(output/"manifest.json", {"normalization": normalization.to_dict(), "device": context.report,
                                       "schedule_sha256": plan.schedule_sha256,
                                       "optimizer_ceiling": plan.planned_batches,
                                       "presentation_ceiling": plan.total_presentations,
                                       "evaluation_epochs": list(config.evaluation_epochs()),
                                       "config": config.to_dict(), "seed": args.seed,
                                       "initial_model_parameters_sha256": _parameter_hash(context, trainer.model)})
        # Actual-run CUDA batch parity checks only dev rows; no tuning inference.
        spot = dev[:64]
        freeze(output/"cuda-parity.json", parity(context, trainer.model, spot))
        best = None
        step = 0
        history = []
        with (output/"steps.jsonl").open("x", encoding="utf-8") as log, (output/"dev.jsonl").open("x", encoding="utf-8") as metrics:
            for epoch in range(config.epochs):
                receipts = []
                for indexes in epoch_batches(plan, config, epoch):
                    lr = config.lr(step, plan.planned_batches)
                    trainer.optimizer.param_groups[0]["lr"] = lr
                    batch = prepared.batch(indexes)
                    receipt = trainer.step(batch)
                    receipt.update(epoch=epoch+1, stage=train[indexes[0]].task.stage, lr=lr,
                                   ordered_sample_ids_sha256=sha256_json(list(batch.sample_ids)))
                    append(log, receipt)
                    receipts.append(receipt)
                    step += 1
                point_rms = point_stage_rms(receipts)
                if epoch+1 not in config.evaluation_epochs():
                    append(metrics, {"epoch": epoch+1, "point_stage_rms": point_rms, "evaluated": False})
                    continue
                predictions = batched_predictions(context, trainer.model, dev)
                score = dev_score(dev, predictions)
                score.update(epoch=epoch+1, point_stage_rms=point_rms, evaluated=True)
                if not score["complete"]:
                    raise ValueError("incomplete dev evaluation")
                score["checkpoint"] = checkpoint(context, trainer, output/f"epoch-{epoch+1:04d}.pt",
                                                  {"epoch": epoch+1, "config": config.to_dict(), "seed": args.seed,
                                                   "schedule_sha256": plan.schedule_sha256,
                                                   "packet_file_sha256": sha256_file(root/"packet.json")})
                append(metrics, score)
                os.fsync(log.fileno()); os.fsync(metrics.fileno())
                history.append(score)
                if best is None or score["nominal"]["p95_error_cells"] < best["nominal"]["p95_error_cells"]:
                    best = score
                freeze(output/f"progress-{epoch+1:04d}.json", {"epoch": epoch+1, "wall_seconds": time.monotonic()-began,
                                                               "process": process_metrics(), "memory": context.memory_stats()})
        result = {"status": "complete", "config": config.to_dict(), "seed": args.seed,
                  "attempt": args.attempt, "selected": best, "optimizer_steps": step,
                  "sample_presentations": trainer.sample_presentations,
                  "memory": context.memory_stats(), "process": process_metrics(),
                  "worker_work_seconds": time.monotonic()-began,
                  "point_rms_last5epochs": _last_point_rms(output, config.epochs),
                  "evidence": "development evidence; no acceptance claim"}
        freeze(output/"result.json", result)
    except BaseException as exc:
        freeze(output/"failure.json", {"type": type(exc).__name__, "message": str(exc),
                                       "traceback": traceback.format_exc(), "wall_seconds": time.monotonic()-began})
        raise


def _last_point_rms(output, epochs):
    values = [json.loads(line) for line in (output/"dev.jsonl").read_text().splitlines()]
    tail = [r["point_stage_rms"] for r in values if r["epoch"] > epochs-5 and r["point_stage_rms"] is not None]
    return sum(tail)/len(tail) if tail else None


def _parameter_hash(context, model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode()); digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run(args):
    root = Path(args.root)
    packet = check_packet(root)
    if (root/"group-start.json").exists():
        raise ValueError("sweep is single-invocation; no implicit resume")
    from nca.train.ik_gpu_ownership import acquire, release
    from nca.train.ik_matched_supervisor import board_sample, stop_tree
    from nca.ik.process_limits import memory_limit
    ownership = {"kind": "ik54_dev_sweep", "claim": uuid.uuid4().hex,
                 "packet_file_sha256": sha256_file(root/"packet.json")}
    # Per-process cap applies to all inherited workers; group Job keeps children contained.
    memory_limit(8192, kill_children_on_exit=True)
    board_sample()
    acquire(ownership)
    freeze(root/"ownership.json", ownership)
    began = time.monotonic()
    freeze(root/"group-start.json", {"utc": utc_now(), "monotonic": began, "ownership": ownership})
    # Short runs first, then 200, then 1000; tanh-1000 is last to allow budget dropping.
    jobs = sorted([(i, seed, 1) for i, c in enumerate(grid()) for seed in SEEDS],
                  key=lambda x: (grid()[x[0]].epochs, grid()[x[0]].epochs == 1000 and grid()[x[0]].head == "tanh", x[0], x[1]))
    active, outcomes = {}, []
    failed = False
    try:
        with (root/"board.jsonl").open("x", encoding="utf-8") as telemetry, (root/"attempts.jsonl").open("x", encoding="utf-8") as attempts:
            while jobs or active:
                sample = board_sample()
                if sample["temperature"] > 80:
                    raise ValueError("TITAN temperature cap exceeded")
                append(telemetry, sample)
                elapsed = time.monotonic()-began
                if elapsed >= packet["max_wall_seconds"] - 120:
                    raise TimeoutError("12-hour development allocation exhausted")
                completed_epochs = sum(grid()[next(i for i,c in enumerate(grid()) if c.id == r["config"])].epochs
                                       for r in outcomes if r["exit_code"] == 0)
                if completed_epochs and len(outcomes) >= 8:
                    seconds_per_epoch = sum(r["process_seconds"] for r in outcomes) / completed_epochs
                    remaining_epochs = sum(grid()[i].epochs for i,_,_ in jobs)
                    forecast = 1.5 * seconds_per_epoch * remaining_epochs / packet["max_workers"]
                    if forecast > packet["max_wall_seconds"] - elapsed - 120:
                        eligible = {i for i,_,_ in jobs if grid()[i].head == "tanh" and grid()[i].epochs == 1000
                                    and sum(j == i for j,_,_ in jobs) == 3}
                        if eligible:
                            dropped = [j for j in jobs if j[0] in eligible]
                            jobs = [j for j in jobs if j[0] not in eligible]
                            with (root/"dropped.jsonl").open("a", encoding="utf-8") as log:
                                append(log, {"utc": utc_now(), "jobs": dropped,
                                             "reason": "projected grid exceeds remaining 12-hour allocation; tanh-1000 first",
                                             "forecast_seconds": forecast})
                # Reserve 8 GiB host commitment per new worker plus 8 GiB system headroom.
                headroom = _host_free_commit()
                while jobs and len(active) < packet["max_workers"] and headroom >= 16*GIB:
                    index, seed, attempt = jobs.pop(0)
                    config = grid()[index]
                    logdir = root/"logs"; logdir.mkdir(exist_ok=True)
                    command = [sys.executable, "-m", "nca.train.ik_dev_sweep", "_worker", "--root", str(root),
                               "--index", str(index), "--seed", str(seed), "--attempt", str(attempt)]
                    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                    with (logdir/f"{config.id}-{seed}-a{attempt}.log").open("xb") as log:
                        process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
                    active[process.pid] = (process, index, seed, attempt, time.monotonic())
                    append(attempts, {"event": "start", "utc": utc_now(), "pid": process.pid,
                                      "config": config.id, "seed": seed, "attempt": attempt})
                    headroom -= 8*GIB
                for pid, (process, index, seed, attempt, started) in list(active.items()):
                    if process.poll() is None:
                        continue
                    config = grid()[index]
                    output = root/"runs"/config.id/f"seed-{seed}"/f"attempt-{attempt}"
                    outcome = {"event": "exit", "config": config.id, "seed": seed, "attempt": attempt,
                               "exit_code": process.returncode, "process_seconds": time.monotonic()-started,
                               "utc": utc_now()}
                    append(attempts, outcome); outcomes.append(outcome)
                    del active[pid]
                    if process.returncode != 0 or not (output/"result.json").is_file():
                        if attempt < packet["max_attempts_per_run"]:
                            jobs.append((index, seed, attempt+1))
                        else:
                            failed = True
                if jobs and not active and headroom < 16*GIB:
                    raise ValueError("insufficient host committed-memory headroom")
                time.sleep(2)
        freeze(root/"group-result.json", {"status": "completed_with_failures" if failed else "complete",
                                          "wall_seconds": time.monotonic()-began, "attempts": outcomes,
                                          "utc": utc_now()})
    except BaseException as exc:
        for process, *_ in active.values():
            stop_tree(process)
        freeze(root/"group-failure.json", {"type": type(exc).__name__, "message": str(exc),
                                           "wall_seconds": time.monotonic()-began, "unstarted_jobs": jobs,
                                           "stopped_jobs": [{"pid": p.pid, "index": i, "seed": s, "attempt": a,
                                                              "exit_code": p.poll()} for p,i,s,a,_ in active.values()]})
        raise
    finally:
        # Only release after every owned process has actually exited.
        if all(p.poll() is not None for p, *_ in active.values()):
            release(ownership)


def _host_free_commit():
    if os.name != "nt":
        raise ValueError("Windows host memory telemetry required")
    import ctypes
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(k, ctypes.c_ulonglong) for k in
                    ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "extended")]
    value = Memory(); value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise ValueError("host memory telemetry unavailable")
    return value.avail_page


def results(root):
    values = []
    for path in sorted((Path(root)/"runs").glob("*/seed-*/attempt-*/result.json")):
        value = _read(path, 16*1024**2)
        value["output"] = str(path.parent)
        values.append(value)
    return values


def confirm(args):
    root = Path(args.root)
    packet = check_packet(root)
    if (root/"shortlist.json").exists():
        raise ValueError("shortlist already frozen; confirmation cannot be repeated")
    group = _read(root/"group-result.json")
    ranked = rank_configs(results(root))
    if len(ranked) < 3:
        raise ValueError("at least three complete three-seed configurations required")
    selected = [{"rank": i+1, "config": config.to_dict(), "ranking_key": list(key),
                 "runs": runs} for i, (key, config, runs) in enumerate(ranked[:3])]
    freeze(root/"shortlist.json", selected)
    from nca.train.ik_gpu_ownership import acquire, release
    ownership = {"kind": "ik54_dev_confirmation", "claim": uuid.uuid4().hex,
                 "shortlist_sha256": sha256_file(root/"shortlist.json")}
    from nca.train.ik_matched_supervisor import board_sample
    from threading import Thread, Event
    acquire(ownership)
    stopped = Event()
    def watchdog():
        try:
            with (root/"confirmation-board.jsonl").open("x", encoding="utf-8") as log:
                while not stopped.is_set():
                    sample = board_sample()
                    if sample["temperature"] > 80:
                        raise ValueError("temperature cap")
                    append(log, sample)
                    stopped.wait(2)
        except BaseException as exc:
            freeze(root/"confirmation-watchdog-failure.json", {"message": str(exc)})
            os._exit(2)
    thread = Thread(target=watchdog, daemon=True); thread.start()
    try:
        corpus = read_corpus(packet)
        from nca.ik.polar import PolarNormalization
        from nca.ik.models_v3 import build_model
        from nca.evaluation.ik_training import score_raw, _accuracy
        context = cuda_context(SEEDS[0])
        normalization = PolarNormalization.from_dict(packet["normalization"])
        confirmations = []
        for item in selected:
            config = DevConfig(**item["config"])
            for run_result in item["runs"]:
                output = Path(run_result["output"])
                chosen = run_result["selected"]["checkpoint"]
                path = output/chosen["file"]
                if sha256_file(path) != chosen["file_sha256"]:
                    raise ValueError("dev-selected checkpoint bytes changed")
                model = build_model(context, normalization, head=config.head, hidden_width=config.hidden_width)
                state = context.torch.load(path, map_location=context.device, weights_only=True)
                model.load_state_dict(state["model"])
                # Exactly one batched inference pass over all original tuning rows.
                predictions = batched_predictions(context, model, corpus.tuning)
                score = score_raw(corpus.tuning, predictions, corpus.config, authenticated_corpus=corpus,
                                  seed=run_result["seed"])
                score["development_accuracy_gate"] = _accuracy(score)
                name = f"confirmation-r{item['rank']}-s{run_result['seed']}.json"
                freeze(root/name, score)
                from nca.train.ik_dev import bias_metrics
                confirmations.append({"rank": item["rank"], "config": config.to_dict(), "seed": run_result["seed"],
                                      "checkpoint": chosen, "epoch": run_result["selected"]["epoch"],
                                      "nominal": score["nominal"], "continuity": score["continuity"],
                                      "complete": score["complete"], "gate": _accuracy(score),
                                      "bias": bias_metrics(corpus.tuning, predictions),
                                      "report": name, "report_sha256": sha256_file(root/name)})
        freeze(root/"confirmation.json", confirmations)
    finally:
        stopped.set(); thread.join(timeout=7)
        release(ownership)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True); p.add_argument("--corpus", required=True); p.add_argument("--registry", required=True)
    for verb in ("run", "confirm", "_worker"):
        p = sub.add_parser(verb); p.add_argument("--root", required=True)
        if verb == "_worker":
            p.add_argument("--index", type=int, required=True); p.add_argument("--seed", type=int, choices=SEEDS, required=True)
            p.add_argument("--attempt", type=int, choices=(1,2,3), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "confirm": confirm, "_worker": worker}[args.command](args)


if __name__ == "__main__":
    main()

# Copyright 2026 techrote. All rights reserved.
