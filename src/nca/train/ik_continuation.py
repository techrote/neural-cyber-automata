"""Immutable, source-bound continuation of the original compact IK allocation.

Amendments add reviewed executable provenance, never replace the original
declaration, review, checkpoint or cumulative ledger. Preparation is CPU-only.
Only an explicitly reviewed supervisor starts the sole UUID-isolated TITAN.
"""
from __future__ import annotations

import time
ENTRY_STARTED = time.monotonic()

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys

from nca.contracts import canonical_json, sha256_json, strict_json_loads
from nca.ik.declaration import runtime_expectation, _project
from nca.ik.device import device_report, configure_device
from nca.runs import immutable_json, utc_now
from nca.train.ik_corpus import read_training_corpus
from nca.train.ik_experiment import (SEEDS, _source, _titan_visibility,
    _qualified_titan_gpu_bytes, load_declaration, corpus_identity, allocation_for,
    identity_for, train_attempt, _review as _original_review)
from nca.train.ik_run_state import IKRunLedger

SCHEMA = "nca-ik-continuation-amendment/v1"
EVALUATION_SCHEMA = "nca-ik-evaluation-amendment/v1"
EVALUATION_SOURCE_DELTA = frozenset(("src/nca/evaluation/ik_final_corpus.py",
    "src/nca/train/ik_continuation.py", "src/nca/evaluation/ik_continuation_eval.py"))
ORIGINAL_REVISION = "b9d1ccda294b3111dbb227704d2b8cb867a84a75"
ORIGINAL_SOURCE_SHA256 = "2e08895a16ca82199feedf9aea0c490e84d44b6c1d3ef623595d3a718e2cb1ae"
ORIGINAL_DECLARATION_FILE_SHA256 = "c4162e81f2cfbbd9bddfb86fbafc2d4b9c4b62f141197bfbb5e15a6db18b214e"
ALLOWED_SOURCE_DELTA = frozenset((
    "src/nca/evaluation/ik_training.py", "src/nca/evaluation/ik_final_corpus.py",
    "src/nca/evaluation/ik_continuation_eval.py", "src/nca/train/ik_experiment.py",
    "src/nca/train/ik_continuation.py", "src/nca/train/ik_run_state.py", "src/nca/builtins.py"))
UPSTREAM_REVISION = "fb57c32c878485df26c3efeb57cd18d7bc0e9b01"
UPSTREAM_SOURCE_PINS = {
    "src/nca/train/jolt_admission_v3.py": "2f7123817b32017ad6db5e22c33299d532f1b32001b2f3a4a3eac2ee41d73f23",
    "src/nca/train/jolt_memory_watchdog_v3.py": "e13dd0624097f3c59a58b06bf00b808cf1a5dd81387394d5f034f35f95efd22d",
    "src/nca/train/jolt_streams_v3.py": "efe37079dd58b26db3c79c8b65d9c88bca963d79a927c5db713d849748332d97",
}
CHECKS = frozenset(("source_and_cpu_checks", "corpus_and_splits", "allocation_and_resume",
                    "scoring_and_selection", "environment_and_memory_feasibility"))
_AMENDMENT_FIELDS = frozenset(("schema", "created_utc", "original_declaration_file_sha256",
    "original_declaration_sha256", "original_admission_sha256", "source", "changed_source_files",
    "runtime", "gpu_visibility_sha256", "dataset", "plan", "allocations", "baseline_ledgers",
    "baseline_files", "final_opened", "scope", "no_new_allocations", "sha256",
    "integrated_upstream_revision", "integrated_upstream_source_pins"))
_EVALUATION_FIELDS = _AMENDMENT_FIELDS | frozenset(("parent_amendment",
    "parent_amendment_file_sha256", "parent_admission", "parent_admission_file_sha256",
    "training_execution", "evaluation_only", "frozen_candidates"))


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4*1024*1024:
        raise ValueError("ordinary bounded continuation JSON required")
    value = strict_json_loads(path.read_bytes(), max_bytes=4*1024*1024)
    if not isinstance(value, dict):
        raise ValueError("continuation JSON object required")
    return value


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _canonical_file_hash(value):
    """immutable_json uses canonical JSON and one terminal newline."""
    return hashlib.sha256((canonical_json(value) + "\n").encode("utf-8")).hexdigest()


def _evaluation_source_delta(parent, actual):
    old, new = parent["files"], actual["files"]
    if set(old) != set(new):
        raise ValueError("evaluation-only source inventory must preserve every parent file")
    changed = {path for path in new if new[path] != old[path]}
    if not changed or changed - EVALUATION_SOURCE_DELTA:
        raise ValueError("evaluation-only source changes exceed the three bounded repair modules")
    return sorted(changed)


def _amendment_record(value, *, evaluation=False):
    fields = _EVALUATION_FIELDS if evaluation else _AMENDMENT_FIELDS
    schema = EVALUATION_SCHEMA if evaluation else SCHEMA
    if (not isinstance(value, dict) or set(value) != fields or value.get("schema") != schema
            or value.get("sha256") != sha256_json({k: v for k, v in value.items() if k != "sha256"})
            or value.get("no_new_allocations") is not True or value.get("final_opened") is not False
            or value.get("integrated_upstream_revision") != UPSTREAM_REVISION
            or value.get("integrated_upstream_source_pins") != UPSTREAM_SOURCE_PINS
            or value.get("original_declaration_file_sha256") != ORIGINAL_DECLARATION_FILE_SHA256):
        raise ValueError("strict immutable original-allocation amendment required")


def _declaration_binding(run, amendment):
    declaration = load_declaration(run, amendment["original_declaration_file_sha256"])
    if (amendment["original_declaration_sha256"] != declaration["sha256"]
            or amendment["plan"] != declaration["plan"]
            or amendment["allocations"] != declaration["allocations"]
            or amendment["dataset"] != declaration["dataset"]
            or amendment["runtime"] != declaration["runtime"]
            or amendment["gpu_visibility_sha256"] != declaration["gpu_visibility_sha256"]
            or sha256_json(_original_review(Path(run) / "admission-review.json", declaration))
               != amendment["original_admission_sha256"]):
        raise ValueError("amendment altered original declaration, allocation or admission")
    return declaration


def _validate_parent(run, parent, parent_file_sha256, admission, admission_file_sha256):
    """Authenticate historical provenance without claiming its source is current."""
    _amendment_record(parent)
    if (not _digest(parent_file_sha256) or _canonical_file_hash(parent) != parent_file_sha256
            or not _digest(admission_file_sha256)
            or _canonical_file_hash(admission) != admission_file_sha256):
        raise ValueError("embedded parent or admission does not match its canonical original file")
    declaration = _declaration_binding(run, parent)
    if _source_delta(declaration["source"], parent["source"]) != parent["changed_source_files"]:
        raise ValueError("historical training amendment source delta differs")
    _review_value(admission, parent)
    _baseline_unchanged(run, parent["baseline_files"])
    return declaration


def _completed_training_baseline(run, declaration, parent, corpus):
    states = _states(run, declaration)
    execution = execution_identity(parent)
    for seed in SEEDS:
        state = states[str(seed)]
        checkpoint = state["latest_checkpoint"]
        if (state["active"] or state["unresolved_work"] or not state["can_resume"]
                or state["last_status"] != "paused" or checkpoint is None
                or state["charged"]["sample_presentations"] != 210400
                or state["charged"]["tuning_calls"] != 80800
                or state["charged"]["evaluation_calls"] != 0
                or state["charged"]["checkpoints"] >= state["allocation"]["max_checkpoints"]
                or allocation_for(corpus, declaration["plan"], seed) != state["allocation"]
                or checkpoint["state"]["extra_state"].get("execution") != execution
                or checkpoint["state"]["extra_state"].get("completed_epochs") != 40
                or checkpoint["state"]["sampler"].get("cursor") != 1720):
            raise ValueError("all three unchanged allocations require complete admitted training and unspent evaluation")
    from nca.evaluation.ik_continuation_eval import freeze_training
    frozen, _ = freeze_training(run, declaration, corpus)
    return states, frozen


def _evaluation_baselines(amendment, declaration):
    """The sealed child must retain complete original-ledger training proofs."""
    if set(amendment["baseline_ledgers"]) != {str(s) for s in SEEDS}:
        raise ValueError("all three original evaluation baseline ledgers required")
    for seed in SEEDS:
        state = amendment["baseline_ledgers"][str(seed)]
        cp = state["latest_checkpoint"]
        if (state["identity"] != identity_for(declaration, seed)
                or state["allocation"] != declaration["allocations"][str(seed)]
                or state["active"] or state["unresolved_work"] or not state["can_resume"]
                or state["last_status"] != "paused" or cp is None
                or state["remaining_wall_seconds"] <= 0
                or state["charged"]["sample_presentations"] != 210400
                or state["charged"]["tuning_calls"] != 80800
                or state["charged"]["evaluation_calls"] != 0
                or state["charged"]["checkpoints"] >= state["allocation"]["max_checkpoints"]
                or cp["state"]["charged"] != state["charged"]
                or cp["state"]["extra_state"].get("execution") != amendment["training_execution"]
                or cp["state"]["extra_state"].get("completed_epochs") != 40
                or len(cp["state"]["extra_state"].get("tuning_history", ())) != 40
                or cp["state"]["sampler"].get("cursor") != 1720
                or cp["reference"]["training_state_sha256"] != sha256_json(cp["state"])):
            raise ValueError("evaluation child lacks complete original admitted training state")
        relative = f"seed-{seed}/" + cp["path"]
        expected = amendment["baseline_files"].get(relative)
        if expected != {"bytes": cp["reference"]["bytes"], "sha256": cp["reference"]["sha256"]}:
            raise ValueError("evaluation baseline checkpoint is not in the immutable original run inventory")


def _source_delta(original, actual):
    old, new = original["files"], actual["files"]
    if (original["revision"] != ORIGINAL_REVISION
            or original["source_sha256"] != ORIGINAL_SOURCE_SHA256
            or not set(old).issubset(new)):
        raise ValueError("original source lineage or complete source inventory differs")
    changed = {path for path in new if path not in old or new[path] != old[path]}
    if (not changed or changed - ALLOWED_SOURCE_DELTA - set(UPSTREAM_SOURCE_PINS)
            or any(new.get(path) != UPSTREAM_SOURCE_PINS[path]
                   for path in UPSTREAM_SOURCE_PINS)):
        raise ValueError("continuation source changes exceed the narrow declared extension")
    return sorted(changed)


def _inventory(run):
    result = {}
    for path in sorted(Path(run).rglob("*")):
        if path.is_symlink():
            raise ValueError("original allocation contains a symlink")
        if path.is_file():
            result[path.relative_to(run).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": _hash(path)}
    return result


def _baseline_unchanged(run, inventory):
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("original immutable allocation inventory required")
    for relative, expected in inventory.items():
        path = Path(relative)
        if (path.is_absolute() or ".." in path.parts or "\\" in relative
                or path.as_posix() != relative):
            raise ValueError("unsafe allocation inventory member")
        target = Path(run) / path
        if (target.is_symlink() or not target.is_file()
                or target.stat().st_size != expected["bytes"] or _hash(target) != expected["sha256"]):
            raise ValueError("original retained allocation bytes changed")


def _states(run, declaration):
    return {str(seed): IKRunLedger.open(Path(run) / f"seed-{seed}",
        expected_identity=identity_for(declaration, seed),
        expected_allocation=declaration["allocations"][str(seed)]).inspect() for seed in SEEDS}


def execution_identity(amendment):
    return {"amendment_sha256": amendment["sha256"],
            "source_sha256": amendment["source"]["source_sha256"],
            "source_commit": amendment["source"]["revision"]}


def _evaluation_outcome_guard(amendment_path, *, allow_pending_evaluation=False):
    """An uncertain final opening/evaluation can never grant training reuse."""
    folder = Path(amendment_path).parent
    if any(folder.glob("evaluation-supervisor-overrun*.json")):
        raise ValueError("final evaluation publication exceeded its charged reserve")
    pending = 0
    for start in folder.glob("evaluation-supervisor-start*.json"):
        terminal = start.with_name(start.name.replace("start", "result", 1))
        if not terminal.is_file():
            pending += 1
            if not allow_pending_evaluation or pending > 1:
                raise ValueError("pending or uncertain final evaluation blocks continuation")
            continue
        value = _read(terminal)
        if (value.get("unknown_work_blocks_resume") is not False
                or value.get("verified_within_charged_wall") is not True):
            raise ValueError("uncertain final evaluation blocks continuation")


def prepare(run, corpus_root, registry, output, *, declaration_digest=ORIGINAL_DECLARATION_FILE_SHA256):
    """Freeze the actual new source and original unspent budget without Torch."""
    run, output = Path(run).resolve(), Path(output).resolve()
    project = _project(None)
    if (run == output or run in output.parents or output == project
            or project in output.parents or output.exists()):
        raise ValueError("new immutable amendment directory outside source/allocation required")
    if declaration_digest != ORIGINAL_DECLARATION_FILE_SHA256:
        raise ValueError("the original first-batch declaration file must be retained")
    declaration = load_declaration(run, declaration_digest)
    actual = _source(None)
    delta = _source_delta(declaration["source"], actual)
    runtime = runtime_expectation()
    if runtime != declaration["runtime"]:
        raise ValueError("continuation runtime differs from the original qualified environment")
    visibility = _titan_visibility(expected_sha256=declaration["gpu_visibility_sha256"],
                                    driver=declaration["plan"]["qualified_environment"]["driver"])
    corpus = read_training_corpus(corpus_root, registry)
    if corpus_identity(corpus) != declaration["dataset"]:
        raise ValueError("original corpus and retained normalization differ")
    original_review = _original_review(run / "admission-review.json", declaration)
    states = _states(run, declaration)
    for seed in SEEDS:
        state = states[str(seed)]
        if (not state["can_resume"] or state["active"] or state["unresolved_work"]
                or allocation_for(corpus, declaration["plan"], seed) != state["allocation"]):
            raise ValueError("original finite allocation is not reconciled or differs")
        expected = (1, 1, 128, 2) if seed == 41011 else (0, 0, 0, 0)
        observed = (state["attempt_count"], state["completed_optimizer_steps"],
                    state["charged"]["sample_presentations"], state["charged"]["checkpoints"])
        if observed != expected or state["charged"]["tuning_calls"] or state["charged"]["evaluation_calls"]:
            raise ValueError("amendment must start from the original single verified batch")
    first = _read(run / "seed-41011/supervisor-result-0001.json")
    if first.get("verified_within_charged_wall") is not True or first.get("returncode") != 0:
        raise ValueError("original first-batch whole-process receipt is not verified")
    inventory = _inventory(run)
    body = {"schema": SCHEMA, "created_utc": utc_now(),
        "original_declaration_file_sha256": declaration_digest,
        "original_declaration_sha256": declaration["sha256"],
        "original_admission_sha256": sha256_json(original_review),
        "source": actual, "changed_source_files": delta,
        "integrated_upstream_revision": UPSTREAM_REVISION,
        "integrated_upstream_source_pins": UPSTREAM_SOURCE_PINS,
        "runtime": runtime, "gpu_visibility_sha256": visibility,
        "dataset": declaration["dataset"], "plan": declaration["plan"],
        "allocations": declaration["allocations"], "baseline_ledgers": states,
        "baseline_files": inventory, "final_opened": False,
        "scope": "complete_existing_three_seed_schedule_then_frozen_candidate_final_and_CPU_latency",
        "no_new_allocations": True}
    amendment = dict(body, sha256=sha256_json(body))
    output.mkdir(parents=True, exist_ok=False)
    file_sha = immutable_json(output / "amendment.json", amendment)
    return {"amendment": amendment, "amendment_file_sha256": file_sha,
            "cuda_initialized": False, "final_opened": False}


def prepare_evaluation(run, prior_amendment_path, prior_filehash, corpus_root, registry, output):
    """Freeze a new evaluation-only source over the complete old training run."""
    run, output = Path(run).resolve(), Path(output).resolve()
    project = _project(None)
    prior_amendment_path = Path(prior_amendment_path)
    if (run == output or run in output.parents or output == project or project in output.parents
            or output.exists() or prior_amendment_path.parent.resolve() == output):
        raise ValueError("new immutable evaluation amendment directory required")
    if not _digest(prior_filehash) or _hash(prior_amendment_path) != prior_filehash:
        raise ValueError("exact historical amendment file SHA-256 required")
    parent = _read(prior_amendment_path)
    admission_path = prior_amendment_path.parent / "admission-review.json"
    admission, admission_filehash = _read(admission_path), _hash(admission_path)
    declaration = _validate_parent(run, parent, prior_filehash, admission, admission_filehash)
    _evaluation_outcome_guard(prior_amendment_path)
    actual = _source(None)
    delta = _evaluation_source_delta(parent["source"], actual)
    if runtime_expectation() != parent["runtime"]:
        raise ValueError("evaluation runtime differs from admitted training runtime")
    validate_runtime(declaration, parent)
    corpus = read_training_corpus(corpus_root, registry)
    if corpus_identity(corpus) != declaration["dataset"]:
        raise ValueError("original corpus and normalization differ")
    states, frozen = _completed_training_baseline(run, declaration, parent, corpus)
    inventory = _inventory(run)
    body = {k: v for k, v in parent.items() if k != "sha256"}
    body.update({"schema": EVALUATION_SCHEMA, "created_utc": utc_now(),
        "parent_amendment": parent, "parent_amendment_file_sha256": prior_filehash,
        "parent_admission": admission, "parent_admission_file_sha256": admission_filehash,
        "source": actual, "changed_source_files": delta, "training_execution": execution_identity(parent),
        "evaluation_only": True, "scope": "evaluate_existing_frozen_three_seed_candidates_only_no_training",
        "baseline_ledgers": states, "baseline_files": inventory, "frozen_candidates": frozen})
    amendment = dict(body, sha256=sha256_json(body))
    _amendment_record(amendment, evaluation=True)
    _evaluation_baselines(amendment, declaration)
    output.mkdir(parents=True, exist_ok=False)
    file_sha = immutable_json(output / "amendment.json", amendment)
    return {"amendment": amendment, "amendment_file_sha256": file_sha,
            "cuda_initialized": False, "final_opened": False}


def load_amendment(run, amendment_path, expected_sha256, *, allow_pending_evaluation=False):
    """Validate every retained baseline byte and exact current execution source."""
    if not _digest(expected_sha256) or _hash(amendment_path) != expected_sha256:
        raise ValueError("exact immutable amendment file SHA-256 required")
    amendment = _read(amendment_path)
    evaluation = amendment.get("schema") == EVALUATION_SCHEMA
    _amendment_record(amendment, evaluation=evaluation)
    declaration = _declaration_binding(run, amendment)
    if evaluation:
        parent = amendment["parent_amendment"]
        _validate_parent(run, parent, amendment["parent_amendment_file_sha256"],
                         amendment["parent_admission"], amendment["parent_admission_file_sha256"])
        if (amendment["evaluation_only"] is not True
                or amendment["scope"] != "evaluate_existing_frozen_three_seed_candidates_only_no_training"
                or amendment["training_execution"] != execution_identity(parent)
                or _evaluation_source_delta(parent["source"], amendment["source"])
                   != amendment["changed_source_files"]):
            raise ValueError("evaluation child changed training execution or declared repair scope")
        _evaluation_baselines(amendment, declaration)
    elif _source_delta(declaration["source"], amendment["source"]) != amendment["changed_source_files"]:
        raise ValueError("original continuation source delta differs")
    if _source(None) != amendment["source"] or runtime_expectation() != amendment["runtime"]:
        raise ValueError("current source or runtime differs from the amended declaration")
    _baseline_unchanged(run, amendment["baseline_files"])
    _evaluation_outcome_guard(amendment_path, allow_pending_evaluation=allow_pending_evaluation)
    return declaration, amendment


def _review_value(value, amendment):
    evaluation = amendment.get("schema") == EVALUATION_SCHEMA
    schema = "nca-ik-evaluation-admission/v1" if evaluation else "nca-ik-continuation-admission/v1"
    verdict = "admit_declared_final_evaluation" if evaluation else "admit_declared_CUDA_continuation"
    if (not isinstance(value, dict)
            or set(value) != {"schema", "amendment_sha256", "source_sha256", "reviewers", "checks", "verdict"}
            or value["schema"] != schema
            or value["amendment_sha256"] != amendment["sha256"]
            or value["source_sha256"] != amendment["source"]["source_sha256"]
            or value["verdict"] != verdict):
        raise ValueError("exact amendment/source-bound prospective continuation review required")
    reviewers = value["reviewers"]
    if (not isinstance(reviewers, list) or len(reviewers) < 2
            or any(type(x) is not str or re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,63}", x) is None for x in reviewers)
            or len(set(reviewers)) != len(reviewers)
            or not isinstance(value["checks"], dict) or set(value["checks"]) != CHECKS
            or any(x is not True for x in value["checks"].values())):
        raise ValueError("two genuine distinct reviewers and every completed continuation check required")
    return value


def _review(review_path, amendment):
    return _review_value(_read(review_path), amendment)


def _training_scope(amendment):
    if amendment.get("schema") != SCHEMA or amendment.get("evaluation_only") is True:
        raise ValueError("evaluation-only amendment cannot admit training or an optimizer attempt")


def validate_runtime(declaration, amendment):
    if runtime_expectation() != amendment["runtime"]:
        raise ValueError("qualified runtime changed")
    return _titan_visibility(expected_sha256=amendment["gpu_visibility_sha256"],
        driver=declaration["plan"]["qualified_environment"]["driver"])


def worker(run, amendment_path, amendment_digest, seed, corpus_root, registry, review_path,
           *, started, review_sha256):
    declaration, amendment = load_amendment(run, amendment_path, amendment_digest)
    _training_scope(amendment)
    if sha256_json(_review(review_path, amendment)) != review_sha256:
        raise ValueError("continuation review changed after supervisor publication")
    validate_runtime(declaration, amendment)
    ledger = IKRunLedger.open(Path(run) / f"seed-{seed}",
        expected_identity=identity_for(declaration, seed),
        expected_allocation=declaration["allocations"][str(seed)])
    attempt = ledger.begin_attempt(started=started)
    try:
        corpus = read_training_corpus(corpus_root, registry)
        if (corpus_identity(corpus) != declaration["dataset"]
                or allocation_for(corpus, declaration["plan"], seed) != ledger.allocation):
            raise ValueError("original corpus or remaining allocation differs")
        source_id = amendment["source"]["source_sha256"]
        config_id = identity_for(declaration, seed)["config_sha256"]
        report = device_report("cuda:0", source_sha256=source_id, config_sha256=config_id)
        total = _qualified_titan_gpu_bytes(report, declaration["plan"]["qualified_environment"])
        context = configure_device("cuda:0", seed,
            memory_fraction=min(1., declaration["plan"]["max_memory_bytes"] / total),
            source_sha256=source_id, config_sha256=config_id)
        if _qualified_titan_gpu_bytes(context.report, declaration["plan"]["qualified_environment"]) != total:
            raise ValueError("selected TITAN capacity changed during setup")
        result = train_attempt(ledger, attempt, corpus, context,
            epochs=declaration["plan"]["epochs"], batch_size=declaration["plan"]["batch_size"],
            execution=execution_identity(amendment))
        number = ledger.inspect()["attempt_count"]
        immutable_json(Path(run) / f"seed-{seed}" / f"training-result-{number:04d}.json", result)
        return result
    except BaseException as exc:
        if ledger.inspect()["active"]:
            attempt.finish("failed", reason=type(exc).__name__, publication_reserve_seconds=60.)
        raise


def supervise(run, amendment_path, amendment_digest, seed, corpus_root, registry, review_path, *, started=None):
    started = time.monotonic() if started is None else started
    run = Path(run)
    declaration, amendment = load_amendment(run, amendment_path, amendment_digest)
    _training_scope(amendment)
    review = _review(review_path, amendment)
    validate_runtime(declaration, amendment)
    frozen = Path(amendment_path).parent / "admission-review.json"
    if frozen.exists():
        if _read(frozen) != review:
            raise ValueError("immutable continuation review changed")
    else:
        immutable_json(frozen, review)
    review_digest = sha256_json(review)
    root = run / f"seed-{seed}"
    ledger = IKRunLedger.open(root, expected_identity=identity_for(declaration, seed),
                            expected_allocation=declaration["allocations"][str(seed)])
    info = ledger.inspect()
    if not info["can_resume"] or any(root.glob("supervisor-overrun-*.json")):
        raise ValueError("original allocation cannot safely continue")
    for previous in root.glob("supervisor-start-*.json"):
        terminal = previous.with_name(previous.name.replace("start", "result", 1))
        if not terminal.is_file() or _read(terminal).get("verified_within_charged_wall") is not True:
            raise ValueError("previous whole-process outcome remains uncertain")
    index, remaining = info["attempt_count"] + 1, info["remaining_wall_seconds"]
    immutable_json(root / f"supervisor-start-{index:04d}.json", {
        "schema": "nca-ik-continuation-supervisor-start/v1", "seed": seed, "attempt": index,
        "phase": "train", "declaration_sha256": declaration["sha256"],
        "execution": execution_identity(amendment), "admission_review_sha256": review_digest,
        "remaining_wall_seconds": remaining, "created_utc": utc_now()})
    command = [sys.executable, "-m", "nca.train.ik_continuation", "_worker",
        "--run", str(run), "--amendment", str(amendment_path),
        "--amendment-sha256", amendment_digest, "--seed", str(seed),
        "--corpus", str(corpus_root), "--registry", str(registry),
        "--review", str(frozen), "--review-sha256", review_digest, "--started", repr(started)]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    timed_out = interrupted = False
    try:
        process.wait(timeout=max(.001, remaining - (time.monotonic() - started) - 5.))
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill(); process.wait(timeout=5.)
    except BaseException:
        interrupted = True
        process.kill(); process.wait(timeout=5.)
    elapsed = time.monotonic() - started
    after = ledger.inspect()
    charged = after["charged_wall_seconds"] - info["charged_wall_seconds"]
    verified = (not timed_out and not interrupted and process.returncode == 0
        and not after["active"] and not after["unresolved_work"]
        and elapsed <= charged and elapsed <= remaining)
    result = {"schema": "nca-ik-continuation-supervisor-result/v1", "seed": seed,
        "attempt": index, "phase": "train", "execution": execution_identity(amendment),
        "returncode": process.returncode, "timed_out": timed_out, "interrupted": interrupted,
        "whole_process_seconds": elapsed, "charged_wall_seconds": charged,
        "consumed_or_reserved_wall_seconds": charged if verified else remaining,
        "verified_within_charged_wall": verified, "unknown_work_blocks_resume": not verified,
        "final_opened": False}
    immutable_json(root / f"supervisor-result-{index:04d}.json", result)
    if time.monotonic() - started > charged or time.monotonic() - started > remaining:
        immutable_json(root / f"supervisor-overrun-{index:04d}.json",
                       {"schema": "nca-ik-continuation-supervisor-overrun/v1", "resume_forbidden": True})
        raise ValueError("whole-process publication exceeded the original charged reserve")
    if not verified:
        raise ValueError("continuation failed or retains uncertain work; no automatic retry")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("run", "corpus", "registry", "output"):
        prep.add_argument("--" + name, required=True)
    prep.add_argument("--declaration-sha256", default=ORIGINAL_DECLARATION_FILE_SHA256)
    eval_prep = sub.add_parser("prepare-evaluation")
    for name in ("run", "prior-amendment", "prior-amendment-sha256", "corpus", "registry", "output"):
        eval_prep.add_argument("--" + name, required=True)
    for name in ("train", "_worker"):
        entry = sub.add_parser(name)
        for argument in ("run", "amendment", "amendment-sha256", "corpus", "registry", "review"):
            entry.add_argument("--" + argument, required=True)
        entry.add_argument("--seed", type=int, choices=SEEDS, required=True)
        if name == "_worker":
            entry.add_argument("--review-sha256", required=True)
            entry.add_argument("--started", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            value = prepare(args.run, args.corpus, args.registry, args.output,
                            declaration_digest=args.declaration_sha256)
        elif args.command == "prepare-evaluation":
            value = prepare_evaluation(args.run, args.prior_amendment, args.prior_amendment_sha256,
                                       args.corpus, args.registry, args.output)
        else:
            kwargs = {"started": args.started if args.command == "_worker" else ENTRY_STARTED}
            if args.command == "_worker":
                kwargs["review_sha256"] = args.review_sha256
            value = (worker if args.command == "_worker" else supervise)(args.run,
                args.amendment, args.amendment_sha256, args.seed, args.corpus, args.registry,
                args.review, **kwargs)
        print(canonical_json(value))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(canonical_json({"status": "refused_or_failed", "error_class": type(exc).__name__,
                              "final_opened": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright 2026 techrote. All rights reserved.
