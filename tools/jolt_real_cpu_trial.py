#!/usr/bin/env python3
"""Owner-approved one-shot *real physical* Jolt CPU learning smoke.

This is NOT the conditionally admitted LEARN-JOLT-003 allocation, does not
open its sealed split, and can never count as a qualified learned family.
Uses the unchanged actual v3 native capture, BC40 and PPO256 trainer paths,
a separately named evidence root, and an irreversible GitHub claim.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRIAL_ID = "jolt-real-cpu-trial-20261023-01"
MARKER = "NCA-JOLT-REAL-CPU-TRIAL/v1:" + TRIAL_ID
ISSUE_URL = "https://api.github.com/repos/techrote/neural-cyber-automata/issues/62/comments"
COMMENT_URL = "https://api.github.com/repos/techrote/neural-cyber-automata/issues/comments/"
GITHUB_REPO = "techrote/neural-cyber-automata"
BRANCH = "experiment/jolt-real-cpu-20261023"
TRIGGER_MESSAGE = "run(jolt): one owner-approved real CPU trial"
SOURCE_CONFIG = ROOT / "configs/learning/jolt-learn-003-v2.json"
APPROVAL = ROOT / "docs/operations/jolt-real-trial-owner-approval-20261023.json"
TEACHER_CASES = 8
PPO_DECISIONS = 256
TRAINING_REPLAY_CASES = 1
SEED = 190103


def _github_api(url, token, *, body=None):
    headers = {"Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28",
               "Authorization": "Bearer " + token}
    raw = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    if raw is not None:
        headers["Content-Type"] = "application/json"
    req = request.Request(url, headers=headers, data=raw,
                          method="POST" if raw is not None else "GET")
    with request.urlopen(req, timeout=45) as response:
        return json.loads(response.read(8 * 1024 * 1024).decode("utf-8"))


def _prior_claim(token):
    for page in range(1, 102):
        comments = _github_api(ISSUE_URL + "?per_page=100&page=" + str(page), token)
        if not isinstance(comments, list):
            raise ValueError("non-list issue comment history")
        if any(MARKER in str(row.get("body", "")) for row in comments):
            return True
        if len(comments) < 100:
            return False
    raise ValueError("issue claim history too long to verify fully")


def _read(path):
    from nca.contracts import strict_json_loads
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or
            path.stat().st_size > 1024 * 1024):
        raise ValueError("missing or replaced bounded source evidence")
    return strict_json_loads(path.read_bytes(), max_bytes=1024 * 1024)


def _publish(path, value, *, maximum_bytes=1024 * 1024):
    from nca.train.jolt_storage_v3 import publish_json_v3
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return publish_json_v3(path, value, maximum_bytes=maximum_bytes)


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for buf in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def _git_sha():
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   text=True).strip()


def _safe_error(exc):
    code = getattr(exc, "code", None)
    return {"type": type(exc).__name__,
            "code": code if isinstance(code, str) and len(code) <= 80 else None}


def _owner_approval():
    value = _read(APPROVAL)
    if (value.get("schema") != "nca-owner-approved-real-cpu-trial/v1"
            or value.get("trial_id") != TRIAL_ID
            or value.get("one_attempt_only") is not True
            or value.get("nonqualifying_independent_allocation") is not True
            or value.get("official_learn_jolt_003_allocation_consumed") is not False
            or value.get("sealed_cases_permitted") is not False
            or value.get("teacher_episodes") != TEACHER_CASES
            or value.get("ppo_decisions") != PPO_DECISIONS
            or value.get("training_case_replays") != TRAINING_REPLAY_CASES
            or value.get("seed") != SEED):
        raise ValueError("explicit bounded owner trial commission differs")
    return value


def _assert_environment():
    env = os.environ
    if (env.get("GITHUB_EVENT_NAME") != "push" or
            env.get("GITHUB_REPOSITORY") != GITHUB_REPO or
            env.get("GITHUB_REF_NAME") != BRANCH or
            env.get("GITHUB_RUN_ATTEMPT") != "1" or
            not env.get("GITHUB_RUN_ID", "").isdecimal() or
            env.get("GITHUB_SHA") != _git_sha()):
        raise ValueError("trial requires the unique original GitHub push identity")
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"],
                               text=True).strip():
        raise ValueError("trial source checkout must be clean")


def claim(args):
    """Durable single-use GitHub record BEFORE any physical native episode."""
    _assert_environment()
    _owner_approval()
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError("GitHub issue-write token unavailable")
    if Path(args.receipt).exists():
        raise ValueError("claim receipt already exists")
    if _prior_claim(token):
        raise ValueError("trial has already been claimed: any retry is forbidden")

    body = "\n".join([
        MARKER, "",
        "**Owner-approved one-shot REAL CPU physical Jolt smoke**, separate from",
        "the unspent formal LEARN-JOLT-003 allocation.",
        "Source commit: " + _git_sha(),
        "GitHub run: " + os.environ["GITHUB_RUN_ID"] + " (attempt 1)",
        "Actual host/teacher/BC40/PPO256 and one open-case comparison.",
        "Seed " + str(SEED) + ", 8 fixed teacher cases, no sealed split.",
        "This durable claim permanently forbids reruns even after failure.",
        "No prospective official qualification or memory admission is claimed.",
    ])
    posted = _github_api(ISSUE_URL, token, body={"body": body})
    identifier = posted.get("id")
    if type(identifier) is not int or identifier < 1:
        raise ValueError("GitHub did not return an original issue comment ID")
    reread = _github_api(COMMENT_URL + str(identifier), token)
    if reread.get("id") != identifier or reread.get("body") != body:
        raise ValueError("published original claim does not match readback")
    receipt = {
        "schema": "nca-jolt-real-cpu-trial-claim/v1",
        "trial_id": TRIAL_ID, "issue_comment_id": identifier,
        "issue_comment_url": reread.get("html_url"),
        "run_id": os.environ["GITHUB_RUN_ID"],
        "source_revision": _git_sha(),
        "owner_approval_sha256": _hash(APPROVAL),
        "one_attempt_only": True,
        "physical_work_started_at_claim": False,
        "official_learn_jolt_003_allocation_consumed": False,
    }
    _publish(args.receipt, receipt)
    print(json.dumps({"status": "single_use_claimed", "comment_id": identifier,
                      "physical_work_started": False}), flush=True)
    return 0


def run_real(args):
    """Run real pinned Godot teacher captures, BC fitting, PPO fitting and open-case test."""
    import torch
    from nca.adapters.godot import inspect_host
    from nca.contracts import sha256_json, strict_json_loads
    from nca.controllers.jolt_learned import JoltPolicy, policy_state_sha256
    from nca.controllers.jolt_recovery_v4 import JoltRecoveryControllerV4
    from nca.controllers.recovery_supervisor import PhysicalRecoverySupervisor
    from nca.train.jolt_admission_v3 import validate_config_v3
    from nca.train.jolt_budget_v3 import BudgetLedgerV3
    from nca.train.jolt_learning import (
        PPO_SPEC, convert_clone_episode_v3, train_clone, ppo_update_v3,
    )
    from nca.train.jolt_native_v3 import BoundedGodotHostV3
    from nca.train.jolt_screen import (
        MeteredOwner, capture, close_worker_v2, error_record, note,
        runtime_identity, source_inventory, _checkpoint,
    )
    from nca.train.jolt_runtime_v3 import capability_contract_v3
    from nca.train.jolt_storage_v3 import BoundedVerifiedJournalV3

    _assert_environment()
    approval = _owner_approval()
    claim_record = _read(args.claim)
    if (claim_record.get("schema") != "nca-jolt-real-cpu-trial-claim/v1"
            or claim_record.get("trial_id") != TRIAL_ID
            or claim_record.get("one_attempt_only") is not True
            or claim_record.get("run_id") != os.environ["GITHUB_RUN_ID"]
            or claim_record.get("source_revision") != _git_sha()
            or claim_record.get("owner_approval_sha256") != _hash(APPROVAL)):
        raise ValueError("original independently claimed trial identity missing")

    config = strict_json_loads(SOURCE_CONFIG.read_bytes(), max_bytes=1024 * 1024)
    validate_config_v3(ROOT, config)
    if (config["id"] != "learn-jolt-003-v2"
            or config["allocation_id"] == TRIAL_ID
            or config["training_seeds"][0] != SEED):
        raise ValueError("trial must reuse accepted fixed experiment semantics without its allocation")
    runtime = runtime_identity()
    capability_contract_v3(runtime)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    host_identity = inspect_host(args.checkout, args.godot, timeout=None)
    output = Path(args.output).resolve()
    if output.name != TRIAL_ID or output.exists():
        raise ValueError("trial requires an unused separately named output")
    if shutil.disk_usage(output.parent).free < 2 * 1024 ** 3:
        raise ValueError("trial requires at least two GiB of immediate free disk")

    source_files = source_inventory(ROOT)
    source_hash = sha256_json(source_files)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema": "nca-real-physical-jolt-trial/v1",
        "trial_id": TRIAL_ID,
        "source_revision": _git_sha(), "source_sha256": source_hash,
        "config_sha256": sha256_json(config),
        "owner_authorization_sha256": _hash(APPROVAL),
        "allocation_claim": claim_record,
        "runtime": runtime, "host_identity": host_identity,
        "teacher_cases_declared": [case["id"] for case in config["cases"]["teacher"][:TEACHER_CASES]],
        "ppo_seed": SEED,
        "ppo_confirmed_decisions_declared": PPO_DECISIONS,
        # Replay an existing training case, not any untouched comparison split.
        "open_cases_declared": [],
        "training_case_replay_declared": [config["cases"]["teacher"][0]["id"]],
        "official_open_cases_opened": False, "sealed_cases_opened": False,
        "evaluation_replays_training_cases_only": True,
        "qualification_scope": "nonqualifying_exploratory_physical_cpu",
        "formal_learn_jolt_003_allocation_consumed": False,
    }
    _publish(output / "manifest.json", manifest)
    ledger = BudgetLedgerV3(config["budget"], started=started,
                            spool_path=output / "reservation-ledger.jsonl")
    journal = BoundedVerifiedJournalV3(output / "owner-journal.jsonl")
    native = owner = None
    errors, recording_errors, rows, checkpoints, learning = [], [], [], [], []
    train_complete = False
    confirmed_ppo_decisions = 0
    stage = "host_start"
    status, failed = "failed", None
    closure_status = None
    close_data = None
    try:
        ledger.reserve("attempts")
        ledger.reserve_many({"host_starts": 1, "host_start_attempts": 1}, allowance_seconds=None)
        note(journal, "physical_trial_native_start_reserved", trial_id=TRIAL_ID, counts=ledger.snapshot())
        native = BoundedGodotHostV3(args.checkout, args.godot,
            original_clock=ledger.original_clock, verified_identity=host_identity,
            lane="act004", mode="stepped", timeout=None)
        owner = MeteredOwner(native, ledger, journal,
                             transport_seconds=None, execution_version=3)
        note(journal, "physical_trial_host_started", identity=native.identity)
        print("REAL_NATIVE_HOST_STARTED pinned=" + host_identity["host_revision"], flush=True)
        stage = "teacher"
        compact = []
        expected_teacher = JoltRecoveryControllerV4().specification_sha256
        expected_supervisor = sha256_json(PhysicalRecoverySupervisor().specification())
        for index, case in enumerate(config["cases"]["teacher"][:TEACHER_CASES]):
            episode, row, _ = capture(owner, ledger, journal, case,
                identifier="smoke-teacher-%02d" % index,
                controller_id="authored", kind="authored", stage="teacher",
                output=output, root=ROOT, config=config, source_sha256=source_hash,
                execution_version=3)
            compact.append(convert_clone_episode_v3(
                episode, expected_teacher_sha256=expected_teacher,
                expected_supervisor_sha256=expected_supervisor))
            rows.append(row)
            print("REAL_TEACHER_EPISODE " + str(index + 1) + "/" + str(TEACHER_CASES)
                  + " decisions=" + str(row["decisions"]) + " outcome=" + row["outcome"]["kind"], flush=True)

        stage = "bc"
        torch.manual_seed(SEED)
        bc = JoltPolicy("bc")
        note(journal, "real_bc_fitting_started", seed=SEED, teacher_episodes=TEACHER_CASES)
        fit = train_clone(bc, tuple(compact), seed=SEED, reserve=ledger.reserve)
        fit.pop("optimizer", None)
        learning.append({"family": "bc", "seed": SEED, "optimizer_calls": fit["optimizer_calls"]})
        _publish(output / "learning" / "real-bc-seed190103.json", fit)
        _checkpoint(output, ledger, journal, checkpoints, bc,
            "real-smoke-bc-seed190103",
            {"config_sha256": sha256_json(config), "source_sha256": source_hash,
             "seed": SEED, "phase": "exploratory_only_not_qualified"},
            execution_version=3)
        note(journal, "real_bc_fitting_completed", seed=SEED, optimizer_calls=fit["optimizer_calls"])
        print("REAL_BC_TRAINED seed=190103 epochs=40 optimizers=" + str(fit["optimizer_calls"]), flush=True)

        stage = "ppo"
        frozen_base = deepcopy(bc).eval().requires_grad_(False)
        torch.manual_seed(SEED + 100000)
        residual = JoltPolicy("residual")
        generator = torch.Generator().manual_seed(SEED + 200000)
        optimizer = torch.optim.Adam(residual.parameters(), lr=PPO_SPEC["learning_rate"])
        teacher_by_id = {case["id"]: case for case in config["cases"]["teacher"]}
        stream = config["ppo_case_stream"]["seeds"][str(SEED)]
        sequences = []
        cursor = 0
        while confirmed_ppo_decisions < PPO_DECISIONS:
            if cursor >= 256:
                raise ValueError("PPO smoke exceeded 256 physical episode starts")
            case_id = stream[cursor]
            case = teacher_by_id[case_id]
            _, row, sequence = capture(owner, ledger, journal, case,
                identifier="smoke-ppo-e%03d" % cursor,
                controller_id="real-smoke-residual-seed190103",
                kind="residual", stage="ppo", output=output, root=ROOT,
                config=config, source_sha256=source_hash, policy=residual,
                base=frozen_base, generator=generator, execution_version=3,
                maximum_decisions=min(40, PPO_DECISIONS - confirmed_ppo_decisions),
                case_stream_entry={"seed": SEED, "index": cursor, "update": 0})
            if row["decisions"] <= 0 or sequence is None:
                raise ValueError("real PPO produced no authentic original decision")
            rows.append(row)
            sequences.append(sequence)
            confirmed_ppo_decisions += row["decisions"]
            cursor += 1
            print("REAL_PPO_CAPTURE decisions=" + str(confirmed_ppo_decisions)
                  + "/256 episodes=" + str(cursor), flush=True)
        note(journal, "real_ppo_fitting_started", seed=SEED,
             physical_decisions=confirmed_ppo_decisions)
        ppo_fit = ppo_update_v3(residual, frozen_base, optimizer, sequences,
            seed=SEED, reserve=ledger.reserve, source_sha256=source_hash)
        _publish(output / "learning" / "real-ppo-seed190103-u1.json", ppo_fit)
        learning.append({"family": "residual", "seed": SEED,
                         "confirmed_decisions": confirmed_ppo_decisions,
                         "optimizer_calls": ppo_fit["optimizer_calls"]})
        _checkpoint(output, ledger, journal, checkpoints, residual,
            "real-smoke-residual-seed190103-u1",
            {"config_sha256": sha256_json(config), "source_sha256": source_hash,
             "seed": SEED, "phase": "one_exploratory_PPO_update_only"},
            base=frozen_base, execution_version=3)
        note(journal, "real_ppo_fitting_completed", seed=SEED,
             physical_decisions=confirmed_ppo_decisions,
             optimizer_calls=ppo_fit["optimizer_calls"])
        print("REAL_PPO_TRAINED confirmed_decisions=256", flush=True)

        stage = "open"
        # A separately named *training-case replay* test. Never inspect
        # the frozen open/sealed comparison cases of the official allocation.
        case = config["cases"]["teacher"][0]
        for kind, policy, base in (("bc", bc, None),
                                   ("residual", residual, frozen_base)):
            _, row, _ = capture(owner, ledger, journal, case,
                identifier="smoke-open-" + kind,
                controller_id="real-smoke-" + kind + "-seed190103",
                kind=kind, stage="open", output=output, root=ROOT,
                config=config, source_sha256=source_hash,
                policy=policy, base=base, execution_version=3)
            rows.append(row)
            print("REAL_TRAINING_REPLAY " + kind + " success=" + str(row["metrics"]["success"]), flush=True)
        train_complete = True
        status = "completed_unqualified"
    except BaseException as exc:
        failed = _safe_error(exc)
        errors.append({"stage": stage, "error": failed})
        print("REAL_TRIAL_FAILED stage=" + stage + " type=" + failed["type"]
              + " code=" + str(failed["code"]), flush=True)
        status = "incomplete_or_failed"
        try:
            note(journal, "real_trial_stop", stage=stage, error=failed)
        except BaseException as record_exc:
            errors.append({"stage": "failure_journal", "error": _safe_error(record_exc)})
    finally:
        if ledger.phase == "active":
            try:
                ledger.enter_phase("closure")
            except BaseException as exc:
                errors.append({"stage": "phase_closure", "error": _safe_error(exc)})
        try:
            closure_status, failure = close_worker_v2(
                native, owner, ledger, journal, status=status, failure=failed,
                recording_errors=recording_errors)
            close_data = {"worker_closure_status": closure_status,
                          "worker_closure_failure": failure,
                          "recording_errors": recording_errors,
                          "journal": journal.durability_receipt()}
            if closure_status == "failed":
                status = "incomplete_or_failed"
        except BaseException as exc:
            status = "incomplete_or_failed"
            errors.append({"stage": "native_or_journal_cleanup", "error": _safe_error(exc)})
        try:
            ledger.publish_reservations(output / "budget-reservations.json")
            ledger.enter_phase("audit")
            # This pilot intentionally does not run the *formal* six-final
            # independent LEARN-JOLT-003 qualification audit.
            ledger.enter_phase("publication")
            ledger.publish_reservations(output / "closed-budget-reservations.json")
            charged = ledger.close()
        except BaseException as exc:
            status = "incomplete_or_failed"
            errors.append({"stage": "reservation_readback", "error": _safe_error(exc)})
            try:
                charged = ledger.snapshot()
            except BaseException:
                charged = None
        finally:
            if not ledger._stream.closed:
                ledger._stream.close()

    summary = {
        "schema": "nca-jolt-real-cpu-trial-result/v1",
        "trial_id": TRIAL_ID, "status": status, "actual_run_completed": train_complete,
        "real_native_host_started": native is not None,
        "source_revision": _git_sha(), "source_sha256": source_hash,
        "config_sha256": sha256_json(config), "host_identity": host_identity,
        "runtime": runtime, "claim": claim_record,
        "elapsed_seconds": time.monotonic() - started,
        "physical_capture_rows": rows,
        "checkpoint_rows": checkpoints,
        "learning": learning, "confirmed_ppo_decisions": confirmed_ppo_decisions,
        "counters": charged, "closure": close_data, "errors": errors,
        "sealed_cases_opened": False, "official_learn_jolt_003_allocation_consumed": False,
        "independent_full_audit_completed": False,
        "qualified_learned_family": False,
        "valid_frozen_six_final_comparison": False,
        "prohibited_conclusion": "This exploratory pilot cannot qualify a learned family or release morphology #16.",
    }
    _publish(output / "trial-result.json", summary, maximum_bytes=4 * 1024 * 1024)
    inventory = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            inventory[path.relative_to(output).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": _hash(path)}
    _publish(output / "trial-original-file-inventory.json", {
        "schema": "nca-jolt-real-cpu-trial-file-inventory/v1",
        "trial_id": TRIAL_ID, "files": inventory,
        "readback_scope": "original_available_bytes_only",
        "formal_learn_jolt_003_audit_claimed": False
    }, maximum_bytes=4 * 1024 * 1024)
    print(json.dumps({
        "trial_id": TRIAL_ID, "status": status,
        "real_native_host_started": native is not None,
        "teacher_completed": sum(r["stage"] == "teacher" for r in rows),
        "ppo_decisions_completed": confirmed_ppo_decisions,
        "trained_families": [r["family"] for r in learning],
        "training_case_replay_evaluations": sum(r["stage"] == "open" for r in rows),
        "official_open_cases_opened": False, "sealed_opened": False, "formal_allocation_spent": False,
        "original_evidence_root": str(output)
    }), flush=True)
    return 0 if status == "completed_unqualified" and train_complete else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    claim_cmd = modes.add_parser("claim")
    claim_cmd.add_argument("--receipt", type=Path, required=True)
    run_cmd = modes.add_parser("run")
    for key in ("checkout", "godot", "output", "claim"):
        run_cmd.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    return claim(args) if args.mode == "claim" else run_real(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        print(json.dumps({"error_type": type(exc).__name__,
                          "code": getattr(exc, "code", None),
                          "trial_id": TRIAL_ID,
                          "status": "failed_incomplete_not_qualified"}), file=sys.stderr, flush=True)
        raise SystemExit(2)
