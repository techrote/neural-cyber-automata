"""CPU-only preparation of matched IK trials; deliberately no execution verb.

This packet is not an admission. It creates new zero-consumption ledgers and
never reopens, resumes or modifies either historical experiment. No final-suite
payload, final result, model, optional numerical runtime or CUDA API is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import uuid

from nca.contracts import canonical_json, sha256_json
from nca.ik.declaration import _project, _verify_committed_sources, runtime_expectation
from nca.ik.polar import TRANSFORM, fit_training_normalization
from nca.ik.profiles import runner_right_sole_profile
from nca.runs import immutable_json, sha256_file, utc_now
from nca.train.ik import LossWeights
from nca.train.ik_corpus import PairPreservingSampler, read_polar_followup_corpus
from nca.train.ik_experiment import (ACCEPTANCE_PATH, PLAN_PATH, SEEDS, _read,
    _source, _titan_visibility, corpus_identity, validate_plan)
from nca.train.ik_run_state import IKRunLedger

CONFIG_PATH = "configs/ik/experiments/runner-polar-endpoint4-matched-v1.json"
TITAN_FINGERPRINT = "e5df8b0db3a22718306c0f68e7ec85b69f1555759649550d74fc44f563c841d9"
GIB = 1024 ** 3


def candidate_config():
    """One fixed hypothesis, not a search over weights or architectures."""
    return {
        "schema": "nca-ik-matched-preparation/v1",
        "id": "runner-polar-endpoint4-matched-v1",
        "status": "preparation_only_no_training_authorized",
        "feature_transform": TRANSFORM,
        "loss_weights": LossWeights(endpoint=4.0).to_dict(),
        "training_seeds": list(SEEDS), "epochs": 40, "batch_size": 128,
        "curriculum": "stage_order_then_seeded_whole_group_shuffle_each_epoch",
        "selection": "tuning_lowest_raw_p95_ties_earliest",
        "cpu_threads_per_worker": 1,
        "trials": {"serial": {"max_workers": 1}, "parallel": {"max_workers": 3}},
        "gpu_allocator_bytes_per_worker": 5 * GIB,
        "gpu_aggregate_budget_bytes": 20 * GIB,
        "gpu_nonallocator_and_monitor_margin_bytes": 5 * GIB,
        "max_wall_seconds_per_seed": 7200,
        "max_attempts_per_seed": 4, "max_checkpoints_per_seed": 45,
        "max_evaluation_calls": 0,
        "excluded_final_suites": ["original_opened_final", "polar_sealed_final"],
        "tuning_stop_gate": "all_three_original_raw_quality_no_final_access_even_on_pass",
        "automatic_retry": False, "trial_overlap_allowed": False,
        "execution_admitted": False,
    }


def validate_config(value):
    if canonical_json(value) != canonical_json(candidate_config()):
        raise ValueError("fixed matched endpoint4 preparation configuration differs")
    return value


def build_candidate_trainer(context, normalization, dataset_config):
    """Prepared loss ablation; the owning future supervisor must admit execution.

    The architecture, polar transform, normalizer and Adam settings are exactly
    v2. Keeping this separate leaves the historical v2 factory unchanged.
    """
    from nca.train.ik import IKTrainer
    return IKTrainer(context, normalization, dataset_config,
                     weights=LossWeights(endpoint=4.0), feature_transform=TRANSFORM)


def prepared_source():
    root = _project(None)
    source = _source(root)
    # The generic source snapshot covers Python, but not this new recipe.
    source["files"][CONFIG_PATH] = sha256_file(root / CONFIG_PATH)
    _verify_committed_sources(root, source["files"])
    source["source_sha256"] = sha256_json(source["files"])
    validate_config(_read(root / CONFIG_PATH))
    return source


def schedules(corpus):
    result = {}
    for seed in SEEDS:
        sampler = PairPreservingSampler(corpus.training, seed, batch_size=128, epochs=40)
        if (sampler.planned_batches, sampler.total_presentations) != (1720, 210400):
            raise ValueError("unchanged forty-epoch workload required")
        result[str(seed)] = sampler.state_dict()
    return result


def allocation():
    return dict(max_optimizer_steps=1720, max_sample_presentations=210400,
                max_tuning_calls=80800, max_evaluation_calls=0,
                max_checkpoints=45, max_attempts=4, max_wall_seconds=7200,
                max_memory_bytes=5 * GIB)


def identity_for(declaration, seed):
    return dict(declaration_sha256=declaration["sha256"],
                config_sha256=sha256_json(declaration["candidate"]),
                source_sha256=declaration["source"]["source_sha256"],
                profile_sha256=declaration["profile_sha256"],
                dataset_sha256=sha256_json(declaration["dataset"]),
                normalization_sha256=sha256_json(declaration["normalization"]),
                runtime_sha256=sha256_json(declaration["runtime"]), seed=seed)


def metrics_template():
    return {
        "status": "unmeasured", "group_makespan_seconds": None,
        "worker_start_finish_overlap": None,
        "worker_wall_seconds": None, "worker_cpu_user_system_seconds": None,
        "worker_process_tree_peak_resident_bytes": None,
        "worker_io_bytes": None, "monitor_overhead_seconds": None, "cuda_event_seconds": None,
        "gpu_utilization_time_series": None, "gpu_used_free_temperature_series": None,
        "worker_peak_allocated_reserved_bytes": None,
        "checkpoint_tensor_optimizer_rng_sampler_hashes": None,
        "ordered_sample_update_receipt_hashes": None,
        "all_forty_tuning_report_hashes": None, "selected_tuning_quality": None,
        "optimizer_steps_presentations_tuning_calls": None,
        "failures_timeouts_unknown_work_and_charges": None,
    }


def prepare(corpus_root, registry, output):
    """Authenticate original train/tuning only, then prepare both new trials."""
    root = Path(output).resolve()
    checkout = _project(None)
    corpus_path, registry_path = Path(corpus_root).resolve(), Path(registry).resolve()
    if (root == checkout or checkout in root.parents or root.exists()
            or any(root == p or root in p.parents or p in root.parents
                   for p in (corpus_path, registry_path))):
        raise ValueError("new output must be outside checkout and disjoint from corpus/registry")
    source = prepared_source()
    baseline = validate_plan(_read(checkout / PLAN_PATH))
    fingerprint = _titan_visibility(expected_sha256=TITAN_FINGERPRINT,
                                    driver=baseline["qualified_environment"]["driver"])
    runtime = runtime_expectation()
    expected = baseline["qualified_environment"]
    if (any(runtime.get(k) != expected[k] for k in ("python", "os", "os_release", "architecture"))
            or any(runtime["packages"].get(k) != expected[k] for k in ("torch", "numpy"))):
        raise ValueError("qualified Windows runtime metadata differs")
    corpus = read_polar_followup_corpus(corpus_root, registry, source_sha256=source["source_sha256"])
    if (not corpus.accepted or len(corpus.training) != 5260 or len(corpus.tuning) != 2020):
        raise ValueError("original authenticated training and tuning populations required")
    normalization = fit_training_normalization(corpus)
    schedule = schedules(corpus)
    common = dict(source=source, runtime=runtime, device="cuda:0",
                  gpu_visibility_sha256=fingerprint, candidate=candidate_config(),
                  profile_sha256=runner_right_sole_profile().sha256,
                  dataset=corpus_identity(corpus), normalization=normalization.to_dict(),
                  schedules=schedule, acceptance_file_sha256=source["files"][ACCEPTANCE_PATH],
                  allocations={str(s): allocation() for s in SEEDS},
                  execution_admitted=False, final_access_allowed=False)
    declarations = {}
    for mode, topology in candidate_config()["trials"].items():
        value = dict(common, schema="nca-ik-matched-trial/v1", mode=mode,
                     topology=topology, id="ik-matched-" + uuid.uuid4().hex, created_utc=utc_now(),
                     metrics=metrics_template())
        value["sha256"] = sha256_json(value)
        declarations[mode] = value
    # All authentication precedes the first output. No existing ledger is opened.
    root.mkdir(parents=True, exist_ok=False)
    files = {}
    for mode, declaration in declarations.items():
        trial = root / mode
        trial.mkdir()
        files[mode] = immutable_json(trial / "declaration.json", declaration)
        for seed in SEEDS:
            IKRunLedger.create(trial / f"seed-{seed}", identity=identity_for(declaration, seed),
                               allocation=allocation())
    packet = dict(schema="nca-ik-matched-packet/v1", created_utc=utc_now(),
                  source_sha256=source["source_sha256"], candidate_sha256=sha256_json(candidate_config()),
                  declaration_file_sha256=files, training_consumed=False,
                  execution_admitted=False, owner_instruction="prepare_both_do_not_train_yet",
                  reviews=[], resource_feasibility="unmeasured_three_workers_not_qualified",
                  required_before_future_execution=["owner_execution_instruction",
                      "exact_head_ci", "reviewed_source_bound_parallel_supervisor",
                      "aggregate_memory_monitor_and_worker_partition_validation",
                      "two_fresh_independent_titan_execution_admissions"],
                  parallel_speedup=None, tensor_reproducibility=None, final_access_allowed=False)
    packet["sha256"] = sha256_json(packet)
    immutable_json(root / "packet.json", packet)
    return packet


def inspect(output):
    """Verify prepared bytes, matched identity and six untouched ledgers."""
    root = Path(output)
    packet = _read(root / "packet.json")
    if (packet.get("sha256") != sha256_json({k: v for k, v in packet.items() if k != "sha256"})
            or packet.get("execution_admitted") is not False
            or packet.get("training_consumed") is not False
            or packet.get("final_access_allowed") is not False):
        raise ValueError("unadmitted preparation packet differs")
    values = {}
    for mode in ("serial", "parallel"):
        path = root / mode / "declaration.json"
        if sha256_file(path) != packet["declaration_file_sha256"][mode]:
            raise ValueError("prepared declaration bytes differ")
        value = _read(path)
        if (value["sha256"] != sha256_json({k: v for k, v in value.items() if k != "sha256"})
                or value["mode"] != mode or value["topology"] != candidate_config()["trials"][mode]
                or value["execution_admitted"] is not False
                or value["final_access_allowed"] is not False
                or value["allocations"] != {str(s): allocation() for s in SEEDS}
                or value["gpu_visibility_sha256"] != TITAN_FINGERPRINT):
            raise ValueError("fixed trial preparation differs")
        validate_config(value["candidate"])
        for seed in SEEDS:
            ledger = IKRunLedger.open(root / mode / f"seed-{seed}",
                                     expected_identity=identity_for(value, seed),
                                     expected_allocation=allocation())
            state = ledger.inspect()
            if (state["attempt_count"] != 0 or state["active"] or state["unresolved_work"]
                    or state["latest_checkpoint"] is not None or any(state["charged"].values())):
                raise ValueError("preparation ledger has consumed work")
        values[mode] = {k: v for k, v in value.items()
                        if k not in ("id", "created_utc", "sha256", "mode", "topology")}
    if values["serial"] != values["parallel"]:
        raise ValueError("serial and parallel candidate/data/runtime/work differ")
    if (packet["source_sha256"] != values["serial"]["source"]["source_sha256"]
            or packet["candidate_sha256"] != sha256_json(candidate_config())):
        raise ValueError("packet source or candidate binding differs")
    return dict(packet_sha256=packet["sha256"], matched=True, zero_consumption=True,
                execution_admitted=False, final_access_allowed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    for name in ("corpus", "registry", "output"):
        command.add_argument("--" + name, required=True)
    command = commands.add_parser("inspect")
    command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.corpus, args.registry, args.output)
    else:
        result = inspect(args.output)
    if "torch" in sys.modules or "numpy" in sys.modules:
        raise RuntimeError("preparation imported an optional numerical runtime")
    print(canonical_json(result))


if __name__ == "__main__":
    main()

# Copyright 2026 techrote. All rights reserved.
