"""Explicitly commissioned, frozen-candidate recovery of one failed IK read.

CPU preparation appends the historical failure charge; it never reads final
payloads or initializes CUDA. Execution requires a new exact-source admission.
All original receipts, final opening, allocations and candidates are retained.
"""
from __future__ import annotations

import time
ENTRY_STARTED = time.monotonic()

import argparse
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import re
import tarfile

from nca.contracts import canonical_json, sha256_json, strict_json_loads
from nca.runs import immutable_json, utc_now
from nca.ik.declaration import runtime_expectation, _project
from nca.train.ik_experiment import (_source, _titan_visibility, load_declaration,
    identity_for, corpus_identity, SEEDS)
from nca.train.ik_continuation import (_amendment_record, _declaration_binding,
    _validate_parent, _canonical_file_hash, _review_value, _inventory,
    _baseline_unchanged, _evaluation_outcome_guard, CHECKS)
from nca.train.ik_run_state import IKRunLedger

SCHEMA = "nca-ik-frozen-recovery-amendment/v1"
REVIEW_SCHEMA = "nca-ik-frozen-recovery-admission/v1"
VERDICT = "admit_declared_frozen_final_recovery"
SCOPE = "recover_same_opened_frozen_final_only_no_training_no_new_allocation"
FAILED_AMENDMENT_FILE_SHA256 = "ac058a6eac60bfbe5c0e87a707e28ab7fdebe1e6e930927f0b2e5125c1ea44de"
FAILED_SOURCE_SHA256 = "7f140d712e42a23b4ecc7bf1129ae2890a7d23c92d3fdd32bb58cd13a2bd09cc"
FROZEN_SHA256 = "e9e1ea0e1bb4831b4323a78fef14a9a1012c97c12a61eae8833f5933c0161b85"
ARCHIVE_SHA256 = "97e3c547c7056e6ef727dc07437f469fde54b545bb091503a791aed85a4f99ce"
MAX_JSON_BYTES = 16 * 1024 * 1024
SOURCE_DELTA = frozenset(("src/nca/evaluation/ik_recovery.py",
    "src/nca/evaluation/ik_continuation_eval.py", "src/nca/evaluation/ik_final_corpus.py",
    "src/nca/train/ik_corpus.py", "src/nca/train/ik_run_state.py",
    "src/nca/train/ik_experiment.py", "src/nca/builtins.py"))
_TOKEN = object()
_FIELDS = frozenset(("schema", "created_utc", "source", "changed_source_files",
    "failed_amendment", "failed_amendment_file_sha256", "failed_admission",
    "failed_admission_file_sha256", "original_declaration_file_sha256",
    "original_declaration_sha256", "plan", "dataset", "allocations", "runtime",
    "gpu_visibility_sha256", "training_execution", "frozen_candidates",
    "failed_artifacts", "prior_run_files", "protected_registry_files", "opening",
    "prior_binding", "historical_receipts", "ledger_recoveries", "recovery_intent",
    "baseline_files", "baseline_ledgers", "archive_metadata", "scope", "evaluation_only",
    "final_opened", "no_new_allocations", "sha256"))


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_JSON_BYTES:
        raise ValueError("ordinary bounded recovery JSON required")
    result = strict_json_loads(path.read_bytes(), max_bytes=MAX_JSON_BYTES)
    if not isinstance(result, dict):
        raise ValueError("recovery JSON object required")
    return result


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sealed(value):
    return dict(value, sha256=sha256_json(value))


def _check_seal(value):
    if not isinstance(value, dict) or value.get("sha256") != sha256_json(
            {k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("recovery record semantic digest differs")


def _source_delta(failed, actual):
    old, new = failed["files"], actual["files"]
    if failed["source_sha256"] != FAILED_SOURCE_SHA256 or not set(old).issubset(new):
        raise ValueError("the exact failed evaluation source lineage is required")
    changed = {p for p in new if old.get(p) != new[p]}
    if (not changed or changed - SOURCE_DELTA
            or set(new) - set(old) != {"src/nca/evaluation/ik_recovery.py"}):
        raise ValueError("source changes exceed the declared frozen recovery extension")
    return sorted(changed)


def _failed(run, failed_path, failed_digest):
    if failed_digest != FAILED_AMENDMENT_FILE_SHA256 or _sha(failed_path) != failed_digest:
        raise ValueError("exact original failed evaluation amendment required")
    a = _read(failed_path)
    _amendment_record(a, evaluation=True)
    d = _declaration_binding(run, a)
    _validate_parent(run, a["parent_amendment"], a["parent_amendment_file_sha256"],
                     a["parent_admission"], a["parent_admission_file_sha256"])
    if a["source"]["source_sha256"] != FAILED_SOURCE_SHA256 or a["frozen_candidates"]["sha256"] != FROZEN_SHA256:
        raise ValueError("failed source or frozen candidates differ")
    review_path = Path(failed_path).parent / "admission-review.json"
    review = _read(review_path)
    _review_value(review, a)
    return d, a, review, _sha(review_path)


def _failure_facts(run, failed_path, failed):
    folder = Path(failed_path).parent
    terminal = _read(folder / "evaluation-supervisor-result.json")
    overrun = _read(folder / "evaluation-supervisor-overrun.json")
    if (terminal.get("unknown_work_blocks_resume") is not True
            or terminal.get("verified_within_charged_wall") is not False
            or overrun.get("resume_forbidden") is not True
            or overrun.get("seeds") != list(SEEDS)):
        raise ValueError("the retained failed evaluation outcome is required")
    elapsed = overrun.get("whole_process_publication_seconds")
    if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < 130.98:
        raise ValueError("complete failed-phase publication duration required")
    historical = {}
    for seed in SEEDS:
        r = terminal["seeds"][str(seed)]
        number = failed["baseline_ledgers"][str(seed)]["attempt_count"] + 1
        if (r.get("returncode") != 1 or r.get("timed_out") is not False
                or r.get("unknown_work_blocks_resume") is not True
                or r.get("charged_wall_seconds") != 0
                or r.get("attempt") != number):
            raise ValueError("only the exact pre-attempt archive lookup failure may recover")
        for stem in ("supervisor-result", "supervisor-overrun"):
            rel = f"seed-{seed}/{stem}-{number:04d}.json"
            path = Path(run) / rel
            if stem == "supervisor-result" and _read(path) != r:
                raise ValueError("per-seed failed outcome differs")
            if stem == "supervisor-overrun" and _read(path).get("resume_forbidden") is not True:
                raise ValueError("historical overrun marker is missing")
            historical[rel] = _sha(path)
    log = (folder / "evaluation-worker.log").read_text(encoding="utf-8")
    if "KeyError:" not in log or "corpus/final-00000.jsonl" not in log:
        raise ValueError("retained archive member lookup failure required")
    if list(folder.glob("final-comparison-*.json")) or (folder / "three-seed-comparison.json").exists():
        raise ValueError("a scored prior final cannot acquire this recovery")
    return float(elapsed), max(131, math.ceil(elapsed)), historical


def _registry(registry, failed):
    suite = failed["dataset"]["final_suite_identity_sha256"]
    names = (suite + ".final-opened.json", suite + ".compact-reader-v1.json")
    opening, binding = (_read(Path(registry) / name) for name in names)
    for record in (opening, binding):
        _check_seal(record)
    if (opening["frozen_candidates_sha256"] != [c["checkpoint_sha256"] for c in failed["frozen_candidates"]["candidates"]]
            or opening["future_experiments_status"] != "development_only"
            or binding["amendment_sha256"] != failed["sha256"]
            or binding["source_sha256"] != FAILED_SOURCE_SHA256
            or binding["frozen_candidates_sha256"] != FROZEN_SHA256):
        raise ValueError("only the same original opened frozen suite may recover")
    inventory = {name: {"bytes": (Path(registry)/name).stat().st_size,
                        "sha256": _sha(Path(registry)/name)} for name in names}
    return opening, binding, inventory


def _archive_metadata(archive, corpus):
    """Authenticate archive and final member headers, without extracting payloads."""
    if _sha(archive) != ARCHIVE_SHA256:
        raise ValueError("unchanged original archive required")
    manifest = _read(Path(corpus) / "manifest.json")
    members = {}
    with tarfile.open(archive, "r:gz") as stream:
        headers = stream.getmembers()
        if len({m.name for m in headers}) != len(headers):
            raise ValueError("duplicate archive members")
        for shard in manifest["shards"]:
            if shard["partition"] != "final":
                continue
            name = "runner-ik-corpus-v1-20261022/corpus/" + shard["path"]
            member = stream.getmember(name)
            if not member.isfile() or member.size != shard["bytes"]:
                raise ValueError("original final member header differs")
            members[shard["path"]] = {"name": name, "bytes": member.size}
    return {"archive_sha256": ARCHIVE_SHA256, "final_members": members,
            "payloads_extracted": False}


def _proof_document(run, registry, a):
    suite = a["dataset"]["final_suite_identity_sha256"]
    result = {"schema": "nca-ik-final-recovery-proof/v1", "run": str(Path(run).resolve()),
        "registry": str(Path(registry).resolve()),
        "opening_file_sha256": a["protected_registry_files"][suite+".final-opened.json"]["sha256"],
        "prior_binding_file_sha256": a["protected_registry_files"][suite+".compact-reader-v1.json"]["sha256"],
        "frozen_candidates": a["frozen_candidates"], "recovery_amendment_sha256": a["sha256"],
        "source_sha256": a["source"]["source_sha256"],
        "historical_receipts": a["historical_receipts"], "ledger_recoveries": a["ledger_recoveries"]}
    if "rebind" in a:
        result["historical_source_sha256"] = a["rebind"]["parent_source_sha256"]
    return result


def _ledger(run, declaration, seed, proof):
    return IKRunLedger.open(Path(run)/f"seed-{seed}", expected_identity=identity_for(declaration, seed),
        expected_allocation=declaration["allocations"][str(seed)], recovery_proof=proof)


def prepare(run, failed_amendment, failed_digest, corpus, registry, archive, output):
    """CPU-only, append-only historical accounting and prospective recovery freeze."""
    run, output = Path(run).resolve(), Path(output).resolve()
    project = _project(None)
    failed_folder = Path(failed_amendment).resolve().parent
    if (output.exists() or output == run or run in output.parents or output == project
            or project in output.parents or output == failed_folder or failed_folder in output.parents):
        raise ValueError("new recovery directory outside source and original allocation required")
    d, failed, prior_review, prior_review_hash = _failed(run, failed_amendment, failed_digest)
    actual = _source(None)
    delta = _source_delta(failed["source"], actual)
    if runtime_expectation() != failed["runtime"]:
        raise ValueError("recovery must retain the admitted runtime")
    visibility = _titan_visibility(expected_sha256=failed["gpu_visibility_sha256"],
                                  driver=d["plan"]["qualified_environment"]["driver"])
    elapsed, charge, historical = _failure_facts(run, failed_amendment, failed)
    from nca.evaluation.ik_final_corpus import RECOVERY_HISTORICAL_RECEIPTS
    if historical != RECOVERY_HISTORICAL_RECEIPTS:
        raise ValueError("historical failed receipt bytes differ from the exact recovery pins")
    opening, binding, registry_files = _registry(registry, failed)
    metadata = _archive_metadata(archive, corpus)
    prior_files = _inventory(run)
    failed_files = _inventory(Path(failed_amendment).parent)
    _baseline_unchanged(run, failed["baseline_files"])
    intent = _sealed({"schema": "nca-ik-frozen-recovery-intent/v1", "created_utc": utc_now(),
        "source": actual, "failed_amendment_file_sha256": failed_digest,
        "historical_receipts": historical, "historical_elapsed_seconds": elapsed,
        "historical_charge_wall_seconds": charge, "frozen_candidates_sha256": FROZEN_SHA256,
        "opening_sha256": opening["sha256"], "no_new_allocations": True})
    ledger_proofs, states, planned_states = {}, {}, {}
    for seed in SEEDS:
        root = run/f"seed-{seed}"
        ledger = IKRunLedger.open(root, expected_identity=identity_for(d, seed),
            expected_allocation=d["allocations"][str(seed)])
        state = ledger.inspect()
        if state != failed["baseline_ledgers"][str(seed)]:
            raise ValueError("training baseline changed before historical recovery charge")
        planned_states[str(seed)] = state
        index = state["attempt_count"] + 1
        files = {f"{stem}-{index:04d}.json": {"bytes": (root/f"{stem}-{index:04d}.json").stat().st_size,
            "sha256": _sha(root/f"{stem}-{index:04d}.json")} for stem in ("supervisor-result", "supervisor-overrun")}
        body = {"schema": "nca-ik-ledger-recovery/v1", "identity_sha256": sha256_json(ledger.identity),
            "allocation_sha256": ledger.header["sha256"], "seed": seed,
            "prior_parent_sha256": state["parent_sha256"], "prior_attempt_count": state["attempt_count"],
            "checkpoint_sha256": state["latest_checkpoint"]["reference"]["sha256"],
            "failed_attempt": index, "failure_files": files,
            "source_sha256": actual["source_sha256"], "authority_sha256": intent["sha256"],
            "frozen_candidates_sha256": FROZEN_SHA256, "opening_sha256": opening["sha256"],
            "historical_elapsed_seconds": elapsed, "historical_charge_wall_seconds": charge}
        ledger_proofs[str(seed)] = _sealed(body)
    # Prove corpus and every original report before the first append-only debit.
    # This typed unpaid proof is only for metadata preflight; the actual final
    # reader independently requires sealed, paid accounting on all three seeds.
    from nca.evaluation.ik_final_corpus import (validate_recovery_proof,
        read_recovery_training_corpus, verify_recovery_training_preflight)
    preflight = _sealed({"schema": "nca-ik-recovery-unpaid-preflight/v1",
        "dataset": d["dataset"], "source": actual, "protected_registry_files": registry_files,
        "frozen_candidates": failed["frozen_candidates"], "historical_receipts": historical,
        "ledger_recoveries": ledger_proofs, "intent_sha256": intent["sha256"]})
    unpaid = validate_recovery_proof(_proof_document(run, registry, preflight))
    training = read_recovery_training_corpus(corpus, registry, recovery_proof=unpaid)
    if corpus_identity(training) != d["dataset"]:
        raise ValueError("original corpus differs before historical accounting")
    verify_recovery_training_preflight(run, d, training, failed["frozen_candidates"], unpaid)
    _baseline_unchanged(run, prior_files)
    if _source(None) != actual:
        raise ValueError("source changed during readonly recovery preflight")
    for seed in SEEDS:
        ledger = _ledger(run, d, seed, ledger_proofs[str(seed)])
        if ledger.inspect() != planned_states[str(seed)]:
            raise ValueError("training ledger changed during readonly recovery preflight")
    output.mkdir(parents=True, exist_ok=False)
    immutable_json(output/"recovery-intent.json", intent)
    for seed in SEEDS:
        proof = ledger_proofs[str(seed)]
        ledger = _ledger(run, d, seed, proof)
        ledger.record_recovery(proof)
        states[str(seed)] = ledger.inspect()
    body = {"schema": SCHEMA, "created_utc": utc_now(), "source": actual,
        "changed_source_files": delta, "failed_amendment": failed,
        "failed_amendment_file_sha256": failed_digest, "failed_admission": prior_review,
        "failed_admission_file_sha256": prior_review_hash,
        "original_declaration_file_sha256": failed["original_declaration_file_sha256"],
        "original_declaration_sha256": d["sha256"], "plan": d["plan"], "dataset": d["dataset"],
        "allocations": d["allocations"], "runtime": failed["runtime"],
        "gpu_visibility_sha256": visibility, "training_execution": failed["training_execution"],
        "frozen_candidates": failed["frozen_candidates"], "failed_artifacts": failed_files,
        "prior_run_files": prior_files, "protected_registry_files": registry_files,
        "opening": opening, "prior_binding": binding, "historical_receipts": historical,
        "ledger_recoveries": ledger_proofs, "recovery_intent": intent,
        "baseline_files": _inventory(run), "baseline_ledgers": states, "archive_metadata": metadata,
        "scope": SCOPE, "evaluation_only": True, "final_opened": True, "no_new_allocations": True}
    amendment = _sealed(body)
    proof = validate_recovery_proof(_proof_document(run, registry, amendment))
    training = read_recovery_training_corpus(corpus, registry, recovery_proof=proof)
    from nca.evaluation.ik_continuation_eval import freeze_training
    frozen, _ = freeze_training(run, d, training, recovery_proof=proof)
    if frozen != amendment["frozen_candidates"] or corpus_identity(training) != d["dataset"]:
        raise ValueError("recovery cannot change the original corpus or frozen selection")
    digest = immutable_json(output/"amendment.json", amendment)
    return {"amendment": amendment, "amendment_file_sha256": digest,
            "cuda_initialized": False, "new_final_opening": False, "historical_charge_recorded": True}


@dataclass(frozen=True)
class RecoveryContext:
    declaration: dict
    amendment: dict
    proof: object
    run: str
    amendment_path: str
    amendment_digest: str
    registry: str
    failed_amendment_path: str
    _authentication: object = field(default=None, repr=False, compare=False)

    def validate_arguments(self, run, path, digest, registry):
        if (self._authentication is not _TOKEN or str(Path(run).resolve()) != self.run
                or str(Path(path).resolve()) != self.amendment_path or digest != self.amendment_digest
                or str(Path(registry).resolve()) != self.registry):
            raise ValueError("exact authenticated recovery execution context required")

    def review(self, path):
        return review(path, self.amendment)

    def worker_arguments(self):
        return ["--failed-amendment", self.failed_amendment_path,
                "--failed-amendment-sha256", FAILED_AMENDMENT_FILE_SHA256]


def load(run, path, digest, failed_path, failed_digest, registry, *, allow_pending=False, _historical=False):
    if _historical and digest != REBIND_PARENT_FILE_SHA256:
        raise ValueError("only the exact refused parent can receive historical loading")
    if _sha(path) != digest:
        raise ValueError("exact immutable recovery amendment file hash required")
    a = _read(path)
    _check_seal(a)
    if (set(a) != _FIELDS | ({"rebind"} if "rebind" in a else set()) or a.get("schema") != SCHEMA or a.get("scope") != SCOPE
            or a.get("evaluation_only") is not True or a.get("final_opened") is not True
            or a.get("no_new_allocations") is not True):
        raise ValueError("strict evaluation-only recovery amendment required")
    d, failed, failed_review, failed_review_hash = _failed(run, failed_path, failed_digest)
    if (a["failed_amendment"] != failed or a["failed_amendment_file_sha256"] != failed_digest
            or a["failed_admission"] != failed_review or a["failed_admission_file_sha256"] != failed_review_hash
            or a["frozen_candidates"] != failed["frozen_candidates"]
            or a["training_execution"] != failed["training_execution"]
            or any(a[k] != failed[k] for k in ("plan", "dataset", "allocations", "runtime", "gpu_visibility_sha256"))
            or a["original_declaration_sha256"] != d["sha256"]):
        raise ValueError("recovery altered failed provenance, candidates or original budget")
    if ((not _historical and _source(None) != a["source"]) or runtime_expectation() != a["runtime"]
            or _source_delta(failed["source"], a["source"]) != a["changed_source_files"]):
        raise ValueError("recovery source or runtime differs")
    _baseline_unchanged(run, a["prior_run_files"])
    _baseline_unchanged(run, a["baseline_files"])
    _baseline_unchanged(Path(failed_path).parent, a["failed_artifacts"])
    _baseline_unchanged(registry, a["protected_registry_files"])
    opening, binding, _ = _registry(registry, failed)
    if opening != a["opening"] or binding != a["prior_binding"]:
        raise ValueError("original opened registry cannot be replaced")
    elapsed, charge, historical = _failure_facts(run, failed_path, failed)
    _check_seal(a["recovery_intent"])
    intent = a["recovery_intent"]
    if (_read(Path(path).parent/"recovery-intent.json") != intent
            or intent.get("schema") != "nca-ik-frozen-recovery-intent/v1"
            or intent.get("source", {}).get("source_sha256") != a.get("rebind", {}).get("parent_source_sha256", a["source"]["source_sha256"])
            or intent.get("failed_amendment_file_sha256") != failed_digest
            or intent.get("historical_receipts") != historical or historical != a["historical_receipts"]
            or intent.get("historical_elapsed_seconds") != elapsed
            or intent.get("historical_charge_wall_seconds") != charge
            or intent.get("no_new_allocations") is not True):
        raise ValueError("historical accounting authority intent changed")
    for seed in SEEDS:
        p = a["ledger_recoveries"][str(seed)]
        if (p["authority_sha256"] != intent["sha256"] or p["source_sha256"] != intent["source"]["source_sha256"]
                or p["historical_elapsed_seconds"] != elapsed
                or p["historical_charge_wall_seconds"] != charge):
            raise ValueError("historical accounting authority differs")
        ledger = _ledger(run, d, seed, p)
        if not ledger.allows_recovered_failure(Path(run)/f"seed-{seed}"/
                                               f"supervisor-overrun-{p['failed_attempt']:04d}.json"):
            raise ValueError("historical failed phase lacks durable charged recovery")
    if "rebind" in a:
        if _read(Path(path).parent/"rebind-intent.json") != a["rebind"]["intent"]:
            raise ValueError("durable rebind intent differs")
        _validate_rebind(run, registry, a, failed_path, failed_digest)
    _evaluation_outcome_guard(path, allow_pending_evaluation=allow_pending)
    from nca.evaluation.ik_final_corpus import validate_recovery_proof
    proof = validate_recovery_proof(_proof_document(run, registry, a))
    return RecoveryContext(d, a, proof, str(Path(run).resolve()), str(Path(path).resolve()), digest,
        str(Path(registry).resolve()), str(Path(failed_path).resolve()), _TOKEN)



REBIND_PARENT_FILE_SHA256 = "a1bb48cdf6bed34f5aa973dcad0fe1a315462d7f3276a18b3ed75d720276bece"
REBIND_PARENT_SOURCE_SHA256 = "81d2588a0aef03b4d4a73479ed6c4656c8e57605f03b4dfeca4c41b136345f70"
REFUSAL_OBSERVATION_SHA256 = "a07a912338c720cd69f987699426a89d2b2f663438dbd1bffe5b8096a3755256"


def _refusal(path):
    path = Path(path).resolve()
    if _sha(path) != REFUSAL_OBSERVATION_SHA256:
        raise ValueError("exact independently audited refused launcher observation required")
    r = _read(path)
    if (r["amendment_file_sha256"] != REBIND_PARENT_FILE_SHA256
            or r["worker_started"] is not False or r["returncode"] != 1
            or r["conservative_launcher_debit_seconds"] != 10
            or r["observed_tool_command_wall_seconds"] > 10):
        raise ValueError("only the pre-worker launcher refusal may be rebound")
    files = {str(path): {"bytes": path.stat().st_size, "sha256": _sha(path)}}
    for name, expected in r["files"].items():
        f = path.parent/name
        if f.stat().st_size != expected["bytes"] or _sha(f) != expected["sha256"]:
            raise ValueError("refusal supporting evidence changed")
        files[str(f)] = expected
    return files


def _validate_rebind(run, registry, a, failed_path, failed_digest):
    r = a["rebind"]
    if (set(r) != {"parent_amendment", "parent_file_sha256", "parent_sha256",
            "parent_source_sha256", "refusal_observation", "intent", "overhead_proofs"}
            or r["parent_file_sha256"] != REBIND_PARENT_FILE_SHA256
            or r["parent_source_sha256"] != REBIND_PARENT_SOURCE_SHA256):
        raise ValueError("exact refused recovery parent required")
    parent = load(run, r["parent_amendment"], r["parent_file_sha256"], failed_path,
                  failed_digest, registry, _historical=True).amendment
    if "rebind" in parent or parent["sha256"] != r["parent_sha256"]:
        raise ValueError("rebind cannot recursively replace accounting authority")
    unchanged = _FIELDS - {"created_utc", "source", "changed_source_files", "baseline_files", "baseline_ledgers", "sha256"}
    if any(a[k] != parent[k] for k in unchanged):
        raise ValueError("rebind changed original paid history or experiment")
    files = _refusal(r["refusal_observation"])
    intent = r["intent"]
    _check_seal(intent)
    if (intent.get("schema") != "nca-ik-recovery-rebind-intent/v1"
            or intent.get("source") != a["source"]
            or intent.get("parent_file_sha256") != r["parent_file_sha256"]
            or intent.get("refusal_files") != files or intent.get("debit_seconds") != 10):
        raise ValueError("exact rebind authority and refusal charge required")
    if set(r["overhead_proofs"]) != {str(s) for s in SEEDS}:
        raise ValueError("all original allocations must pay refused launcher work")
    for seed in SEEDS:
        p = r["overhead_proofs"][str(seed)]
        _check_seal(p)
        old = parent["baseline_ledgers"][str(seed)]
        if (p.get("source_sha256") != a["source"]["source_sha256"]
                or p.get("prior_source_sha256") != r["parent_source_sha256"]
                or p.get("prior_amendment_file_sha256") != r["parent_file_sha256"]
                or p.get("prior_amendment_sha256") != parent["sha256"]
                or p.get("authority_sha256") != intent["sha256"]
                or p.get("prior_parent_sha256") != old["parent_sha256"]
                or p.get("historical_charge_wall_seconds") != 10
                or p.get("refusal_files") != files):
            raise ValueError("overhead proof changed the refused source or charge")
        overhead = Path(run)/f"seed-{seed}"/"overheads"/"charge-0001.json"
        if _read(overhead).get("record", {}).get("proof") != p:
            raise ValueError("durable launcher charge differs from declared overhead proof")
        # Immutable baseline inventory binds the durable paid overhead receipts.
        state = a["baseline_ledgers"][str(seed)]
        if (state["attempt_count"] != old["attempt_count"] or state["charged"] != old["charged"]
                or state["latest_checkpoint"] != old["latest_checkpoint"]
                or state["charged_wall_seconds"] < old["charged_wall_seconds"] + 10):
            raise ValueError("rebind reset accounting, checkpoint or refused launcher debit")


def _copy_paid_intent(parent_path, parent, output):
    """Retain the exact original authority bytes before any new debit."""
    source = Path(parent_path).parent/"recovery-intent.json"
    if _read(source) != parent["recovery_intent"]:
        raise ValueError("original paid authority changed before rebind publication")
    raw = source.read_bytes()
    target = Path(output)/"recovery-intent.json"
    with target.open("xb") as stream:
        stream.write(raw)
    if target.read_bytes() != raw:
        raise ValueError("original paid authority copy readback differs")


def rebind(run, parent_path, parent_digest, failed_path, failed_digest, corpus, registry,
           refusal_observation, output):
    """Rebind one refused launcher, retaining original paid authority and candidates."""
    run, output = Path(run).resolve(), Path(output).resolve()
    project = _project(None)
    if (output.exists() or output == run or run in output.parents
            or output == project or project in output.parents):
        raise ValueError("new external rebind directory required")
    if parent_digest != REBIND_PARENT_FILE_SHA256:
        raise ValueError("exact refused parent amendment required")
    context = load(run, parent_path, parent_digest, failed_path, failed_digest, registry, _historical=True)
    parent, d = context.amendment, context.declaration
    if "rebind" in parent or parent["source"]["source_sha256"] != REBIND_PARENT_SOURCE_SHA256:
        raise ValueError("one explicit refused parent only")
    source = _source(None)
    if source == parent["source"]:
        raise ValueError("rebind requires the committed launcher repair")
    delta = _source_delta(parent["failed_amendment"]["source"], source)
    _titan_visibility(expected_sha256=parent["gpu_visibility_sha256"],
                      driver=d["plan"]["qualified_environment"]["driver"])
    files = _refusal(refusal_observation)
    from nca.evaluation.ik_final_corpus import read_recovery_training_corpus
    from nca.evaluation.ik_continuation_eval import freeze_training
    training = read_recovery_training_corpus(corpus, registry, recovery_proof=context.proof)
    frozen, _ = freeze_training(run, d, training, recovery_proof=context.proof)
    if frozen != parent["frozen_candidates"] or corpus_identity(training) != parent["dataset"]:
        raise ValueError("rebind cannot change corpus or frozen candidates")
    ledgers = {s: _ledger(run, d, s, parent["ledger_recoveries"][str(s)]) for s in SEEDS}
    if any(ledgers[s].inspect() != parent["baseline_ledgers"][str(s)] for s in SEEDS):
        raise ValueError("actual paid parent boundary changed")
    intent = _sealed({"schema": "nca-ik-recovery-rebind-intent/v1", "created_utc": utc_now(),
        "source": source, "parent_file_sha256": parent_digest, "refusal_files": files,
        "debit_seconds": 10})
    proofs = {}
    for seed in SEEDS:
        ledger, old = ledgers[seed], parent["baseline_ledgers"][str(seed)]
        proofs[str(seed)] = _sealed({"schema": "nca-ik-recovery-launcher-overhead/v1",
            "identity_sha256": sha256_json(ledger.identity), "allocation_sha256": ledger.header["sha256"],
            "prior_parent_sha256": old["parent_sha256"], "prior_attempt_count": old["attempt_count"],
            "checkpoint_sha256": old["latest_checkpoint"]["reference"]["sha256"],
            "prior_amendment_file_sha256": parent_digest, "prior_amendment_sha256": parent["sha256"],
            "prior_source_sha256": parent["source"]["source_sha256"],
            "source_sha256": source["source_sha256"], "authority_sha256": intent["sha256"],
            "historical_charge_wall_seconds": 10, "refusal_files": files})
    if _source(None) != source:
        raise ValueError("source changed during rebind preflight")
    output.mkdir(parents=True, exist_ok=False)
    immutable_json(output/"rebind-intent.json", intent)
    _copy_paid_intent(parent_path, parent, output)
    for seed in SEEDS:
        ledgers[seed].record_recovery_overhead(proofs[str(seed)])
    body = {k:v for k,v in parent.items() if k != "sha256"}
    body.update(created_utc=utc_now(), source=source, changed_source_files=delta,
        baseline_files=_inventory(run), baseline_ledgers={str(s):ledgers[s].inspect() for s in SEEDS},
        rebind={"parent_amendment": str(Path(parent_path).resolve()), "parent_file_sha256": parent_digest,
            "parent_sha256": parent["sha256"], "parent_source_sha256": parent["source"]["source_sha256"],
            "refusal_observation": str(Path(refusal_observation).resolve()), "intent": intent,
            "overhead_proofs": proofs})
    a = _sealed(body)
    _validate_rebind(run, registry, a, failed_path, failed_digest)
    digest = immutable_json(output/"amendment.json", a)
    load(run, output/"amendment.json", digest, failed_path, failed_digest, registry)
    return {"amendment": a, "amendment_file_sha256": digest, "cuda_initialized": False,
            "refused_launcher_charged_seconds_per_seed": 10, "new_allocations": False}


def review(path, amendment):
    r = _read(path)
    if (set(r) != {"schema", "amendment_sha256", "source_sha256", "reviewers", "checks", "verdict"}
            or r["schema"] != REVIEW_SCHEMA or r["verdict"] != VERDICT
            or r["amendment_sha256"] != amendment["sha256"]
            or r["source_sha256"] != amendment["source"]["source_sha256"]
            or not isinstance(r["reviewers"], list) or len(r["reviewers"]) < 2
            or any(type(x) is not str or re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,63}", x) is None for x in r["reviewers"])
            or len(set(r["reviewers"])) != len(r["reviewers"])
            or not isinstance(r["checks"], dict) or set(r["checks"]) != CHECKS
            or any(v is not True for v in r["checks"].values())):
        raise ValueError("two fresh source-bound actual recovery admissions required")
    return r


def supervise(run, amendment, digest, failed_amendment, failed_digest, corpus, registry, archive,
              review_path, *, started=None):
    context = load(run, amendment, digest, failed_amendment, failed_digest, registry)
    context.review(review_path)
    _titan_visibility(expected_sha256=context.amendment["gpu_visibility_sha256"],
                      driver=context.declaration["plan"]["qualified_environment"]["driver"])
    from nca.evaluation.ik_continuation_eval import supervise as execute
    return execute(run, amendment, digest, corpus, registry, review_path, archive,
                   started=started, recovery_context=context)


def worker(run, amendment, digest, failed_amendment, failed_digest, corpus, registry, archive,
           review_path, *, started, review_sha256):
    context = load(run, amendment, digest, failed_amendment, failed_digest, registry, allow_pending=True)
    if sha256_json(context.review(review_path)) != review_sha256:
        raise ValueError("recovery review differs from supervisor receipt")
    from nca.evaluation.ik_continuation_eval import worker as execute
    return execute(run, amendment, digest, corpus, registry, review_path, archive,
                   started=started, review_sha256=review_sha256, recovery_context=context)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("run", "failed-amendment", "failed-amendment-sha256", "corpus", "registry", "archive", "output"):
        prep.add_argument("--"+name, required=True)
    rebound = sub.add_parser("rebind")
    for flag in ("run", "parent-amendment", "parent-amendment-sha256", "failed-amendment", "failed-amendment-sha256", "corpus", "registry", "refusal-observation", "output"):
        rebound.add_argument("--"+flag, required=True)
    for name in ("run", "_worker"):
        command = sub.add_parser(name)
        for flag in ("run", "amendment", "amendment-sha256", "failed-amendment", "failed-amendment-sha256", "corpus", "registry", "archive", "review"):
            command.add_argument("--"+flag, required=True)
        if name == "_worker":
            command.add_argument("--started", type=float, required=True)
            command.add_argument("--review-sha256", required=True)
    a = p.parse_args(argv)
    if a.command == "prepare":
        result = prepare(a.run, a.failed_amendment, a.failed_amendment_sha256,
                         a.corpus, a.registry, a.archive, a.output)
    elif a.command == "rebind":
        result = rebind(a.run, a.parent_amendment, a.parent_amendment_sha256,
            a.failed_amendment, a.failed_amendment_sha256, a.corpus, a.registry,
            a.refusal_observation, a.output)
    else:
        common = (a.run, a.amendment, a.amendment_sha256, a.failed_amendment,
                  a.failed_amendment_sha256, a.corpus, a.registry, a.archive, a.review)
        result = (worker(*common, started=a.started, review_sha256=a.review_sha256) if a.command == "_worker"
                  else supervise(*common, started=ENTRY_STARTED))
    print(canonical_json(result))


def register(registry):
    return registry.register("evaluator", "ik-frozen-recovery", 1, supervise,
        ("original-frozen-candidates", "append-only-failure-charge", "TITAN-UUID-only"))


if __name__ == "__main__":
    # Use the canonical module for typed recovery proofs shared with the
    # evaluator. runpy's __main__ classes have a different Python identity.
    from nca.evaluation import ik_recovery as canonical_module
    canonical_module.ENTRY_STARTED = ENTRY_STARTED
    canonical_module.main()

# Copyright 2026 techrote. All rights reserved.
