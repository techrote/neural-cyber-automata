"""One-use CPU bookkeeping for the retained polar tuning no-go.

This command cannot train, initialize CUDA, run inference or open final payloads.
Historical failed supervisors and ownership holds remain immutable. A finite new
CPU reservation is debited to each existing allocation before selection readback.
"""
from __future__ import annotations

import time
ENTRY_STARTED = time.monotonic()
import argparse
from pathlib import Path
import subprocess
import sys

from nca.contracts import canonical_json, sha256_json, strict_json_loads
from nca.ik.declaration import runtime_expectation
from nca.runs import immutable_json, sha256_file, utc_now
from nca.train.ik_experiment import _source, SEEDS
from nca.train.ik_followup import ledger_at, select_training
from nca.train.ik_followup_gate import tuning_stop_gate

ORIGINAL_FILE = "f36137d484ec54fb50cbf466cc7421785e2d73136d2f30940e35a02bc064a3c0"
ORIGINAL_SOURCE = "df75fc5a14dd3f67eb11ac47437563fa8f8117d9ae0503c64aec1ca52a322435"
PREPASS_FILE = "d7156b37dcb409d5fdf91e0a3f392a0779f1d7e682d44c9478ef1e01cab9346c"
CPU_RESERVE_SECONDS = 900.
CHANGES = {"src/nca/train/ik_followup_gate.py", "src/nca/train/ik_run_state.py",
           "src/nca/train/ik_followup_terminal.py"}
CHECKS = {"source_ci", "corpus_isolation", "allocation_failure_accounting",
          "unchanged_tuning_gate", "cpu_only_feasibility"}


def read(path):
    return strict_json_loads(Path(path).read_bytes(), max_bytes=32*1024**2)


def signed(value):
    return dict(value, sha256=sha256_json(value))


def verify_signed(value):
    if value.get("sha256") != sha256_json({k:v for k,v in value.items() if k != "sha256"}):
        raise ValueError("terminal semantic identity differs")
    return value


def pin(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("terminal evidence must be a regular file")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def historical(declaration):
    run, fresh = Path(declaration["run"]), Path(declaration["fresh"])
    if pin(run / "declaration.json")["sha256"] != ORIGINAL_FILE:
        raise ValueError("exact original admitted polar declaration required")
    original = verify_signed(read(run / "declaration.json"))
    if original["source"]["source_sha256"] != ORIGINAL_SOURCE:
        raise ValueError("historical executable source differs")
    if pin(declaration["prepass"])["sha256"] != PREPASS_FILE:
        raise ValueError("independent failure prepass changed")
    prior = read(declaration["prepass"])
    if (any(declaration["failure_files"].get(k) != v for k,v in prior["failure_files"].items())
            or declaration["absent_files"] != prior["absent_terminal_files"]):
        raise ValueError("terminal declaration omitted historical failure evidence")
    for name, expected in declaration["failure_files"].items():
        if pin(run / name) != expected:
            raise ValueError("historical failure/hold evidence changed")
    for name in declaration["absent_files"]:
        if (run / name).exists():
            raise ValueError("historically absent supervisor terminal appeared")
    if (fresh / "opened.json").exists() or (run / "tuning-stop.json").exists():
        raise ValueError("fresh final must remain unopened and old publication absent")
    for name, expected in declaration["fresh_files"].items():
        if pin(fresh / name) != expected:
            raise ValueError("sealed final metadata/archive bytes changed")
    return original


def admission(path, declaration):
    review = verify_signed(read(path))
    if (set(review) != {"schema", "declaration_sha256", "source_sha256", "reviewers", "verdict", "sha256"}
            or review["schema"] != "nca-ik-polar-terminal-admission/v1"
            or review["declaration_sha256"] != declaration["sha256"]
            or review["source_sha256"] != declaration["source"]["source_sha256"]
            or review["verdict"] != "admit_cpu_terminal_no_go_only"
            or len(review["reviewers"]) != 2
            or len({r["id"] for r in review["reviewers"]}) != 2):
        raise ValueError("two fresh independent terminal admissions required")
    for reviewer in review["reviewers"]:
        if (set(reviewer) != {"id", "artifact", "artifact_file_sha256"}
                or pin(reviewer["artifact"])["sha256"] != reviewer["artifact_file_sha256"]):
            raise ValueError("independent review artifact differs")
        result = verify_signed(read(reviewer["artifact"]))
        if (result["reviewer"] != reviewer["id"] or result["verdict"] != "approve"
                or result["declaration_sha256"] != declaration["sha256"]
                or result["source_sha256"] != declaration["source"]["source_sha256"]
                or set(result["checks"]) != CHECKS or any(v is not True for v in result["checks"].values())):
            raise ValueError("review findings do not approve exact terminal scope")
    return review


def validate_debit(ledger, proof):
    fields = {"schema", "identity_sha256", "allocation_sha256", "prior_parent_sha256",
              "prior_attempt_count", "checkpoint_sha256", "historical_charge_wall_seconds",
              "declaration_path", "declaration_file_sha256", "review_path", "review_file_sha256", "sha256"}
    if not isinstance(proof, dict) or set(proof) != fields:
        raise ValueError("exact terminal-only debit fields required")
    verify_signed(proof)
    if pin(proof["declaration_path"])["sha256"] != proof["declaration_file_sha256"]:
        raise ValueError("terminal declaration bytes changed")
    declaration = verify_signed(read(proof["declaration_path"]))
    original = historical(declaration)
    if pin(proof["review_path"])["sha256"] != proof["review_file_sha256"]:
        raise ValueError("terminal admission bytes changed")
    admission(proof["review_path"], declaration)
    seed = str(ledger.identity["seed"])
    boundary = declaration["boundaries"][seed]
    if (declaration["schema"] != "nca-ik-polar-terminal-declaration/v1"
            or declaration["cpu_reservation_seconds_per_seed"] != CPU_RESERVE_SECONDS
            or declaration["authority"] != "permanent_no_go_no_training_no_final_opening"
            or ledger.root.resolve() != Path(declaration["run"]).resolve() / f"seed-{seed}"
            or ledger.identity["declaration_sha256"] != original["sha256"]
            or proof["identity_sha256"] != sha256_json(ledger.identity)
            or proof["allocation_sha256"] != ledger.header["sha256"]
            or proof["prior_parent_sha256"] != boundary["parent_sha256"]
            or proof["prior_attempt_count"] != 2
            or proof["checkpoint_sha256"] != boundary["latest_checkpoint"]["reference"]["sha256"]
            or proof["historical_charge_wall_seconds"] != CPU_RESERVE_SECONDS):
        raise ValueError("terminal debit differs from immutable failed boundary")


def declare(run, fresh, prepass, output):
    if pin(prepass)["sha256"] != PREPASS_FILE:
        raise ValueError("independent historical failure prepass required")
    prior = read(prepass)
    source = _source(None)
    original = read(Path(run) / "declaration.json")
    changed = {k for k in set(source["files"]) | set(original["source"]["files"])
               if source["files"].get(k) != original["source"]["files"].get(k)}
    if changed != CHANGES:
        raise ValueError("terminal source must change only gate, ledger and terminal bookkeeping")
    failure = dict(prior["failure_files"])
    for path in Path(run).glob("phase-claims/*-start.json"):
        failure[path.relative_to(run).as_posix()] = pin(path)
    value = {"schema": "nca-ik-polar-terminal-declaration/v1", "created_utc": utc_now(),
        "source": source, "runtime": runtime_expectation(), "run": str(Path(run).resolve()),
        "fresh": str(Path(fresh).resolve()), "failure_files": failure,
        "prepass": str(Path(prepass).resolve()),
        "absent_files": prior["absent_terminal_files"],
        "fresh_files": {p.name:pin(p) for p in Path(fresh).iterdir() if p.is_file()},
        "cpu_reservation_seconds_per_seed": CPU_RESERVE_SECONDS,
        "authority": "permanent_no_go_no_training_no_final_opening",
        "boundaries": {str(s): ledger_at(run, original, s).inspect() for s in SEEDS}}
    historical(value)
    for boundary in value["boundaries"].values():
        if (boundary["attempt_count"] != 2 or boundary["last_status"] != "failed"
                or boundary["active"] or boundary["unresolved_work"]
                or boundary["charged"]["evaluation_calls"] != 0
                or boundary["remaining_wall_seconds"] <= CPU_RESERVE_SECONDS + 60):
            raise ValueError("exact paid idle failed finish required")
    immutable_json(output, signed(value))


def derive(run, original):
    selected, reports, _ = select_training(run, original)
    gate = tuning_stop_gate(reports)
    if gate["final_opening_allowed"] or set(gate["per_seed"].values()) != {"no_go"}:
        raise ValueError("this terminal authority only admits all-seed tuning no-go")
    if "torch" in sys.modules:
        raise ValueError("terminal bookkeeping must not import Torch")
    return {"gate": gate, "selections": {str(s):c for s,c in selected.items()},
            "fresh_final": "sealed_unmeasured", "training_authority": False,
            "historical_finish_status": "failed_preserved"}


def validate_result(result, declaration, review_path, ledgers):
    verify_signed(result)
    expected_selections = {s:v["selected"] for s,v in read(declaration["prepass"])["seeds"].items()}
    expected_gate = {"schema":"nca-ik-polar-tuning-stop/v1",
        "per_seed":{str(s):"no_go" for s in SEEDS}, "final_opening_allowed":False,
        "reason":"all_seed_raw_quality_gate"}
    debit_hashes = {str(s):sha256_file(l.root / "overheads/charge-0001.json") for s,l in ledgers.items()}
    if (result.get("terminal_declaration_sha256") != declaration["sha256"]
            or result.get("admission_file_sha256") != sha256_file(review_path)
            or result.get("debits") != debit_hashes or result.get("gate") != expected_gate
            or result.get("selections") != expected_selections
            or result.get("training_authority") is not False
            or result.get("fresh_final") != "sealed_unmeasured"
            or result.get("historical_finish_status") != "failed_preserved"):
        raise ValueError("terminal worker result differs from independently pinned no-go selections")


def publish_supervisor(root, began):
    immutable_json(root / "supervisor.json", {"returncode":0,"verified_within_paid_cpu_reservation":True,
        "whole_process_seconds":time.monotonic()-began, "terminal_file_sha256":sha256_file(root / "terminal.json")})
    if time.monotonic()-began > CPU_RESERVE_SECONDS:
        immutable_json(root / "overrun.json", {"terminal_unverified":True,"resume_forbidden":True})
        raise ValueError("final supervisor publication exceeded paid CPU reservation")


def execute(path, review_path):
    began = ENTRY_STARTED
    declaration = verify_signed(read(path))
    if declaration["source"] != _source(None) or declaration["runtime"] != runtime_expectation():
        raise ValueError("exact admitted terminal source/runtime required")
    admission(review_path, declaration)
    original = historical(declaration)
    root = Path(path).parent
    ledgers = {s:ledger_at(declaration["run"], original, s) for s in SEEDS}
    if any(l.inspect() != declaration["boundaries"][str(s)] for s,l in ledgers.items()):
        raise ValueError("pre-debit boundary changed")
    immutable_json(root / "started.json", {"declaration_sha256": declaration["sha256"], "created_utc":utc_now()})
    # No work starts until all three finite CPU reservations are durable.
    for seed, ledger in ledgers.items():
        boundary = declaration["boundaries"][str(seed)]
        proof = signed({"schema":"nca-ik-polar-terminal-debit/v1",
            "identity_sha256":sha256_json(ledger.identity), "allocation_sha256":ledger.header["sha256"],
            "prior_parent_sha256":boundary["parent_sha256"], "prior_attempt_count":2,
            "checkpoint_sha256":boundary["latest_checkpoint"]["reference"]["sha256"],
            "historical_charge_wall_seconds":CPU_RESERVE_SECONDS,
            "declaration_path":str(Path(path).resolve()), "declaration_file_sha256":sha256_file(path),
            "review_path":str(Path(review_path).resolve()), "review_file_sha256":sha256_file(review_path)})
        ledger.record_terminal_overhead(proof)
    with (root / "worker.log").open("xb") as log:
        process = subprocess.Popen([sys.executable, "-m", __name__.replace("__main__", "nca.train.ik_followup_terminal"),
                                    "_derive", "--declaration", str(path), "--review", str(review_path)], stdout=log, stderr=log)
        try:
            process.wait(timeout=max(.001, CPU_RESERVE_SECONDS-(time.monotonic()-began)-60))
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            process.kill()
            process.wait(timeout=5)
            raise ValueError("terminal CPU timeout retained; no automatic retry")
    if process.returncode != 0:
        raise ValueError("terminal CPU failure retained; no automatic retry")
    result = read(root / "worker-result.json")
    validate_result(result, declaration, review_path, ledgers)
    historical(declaration)
    states = {str(s):l.inspect() for s,l in ledgers.items()}
    for seed, state in states.items():
        old = declaration["boundaries"][seed]
        if any(state[k] != old[k] for k in ("attempt_count", "last_status", "charged", "latest_checkpoint",
                                            "completed_optimizer_steps", "optimizer_presentations", "consumed_batches")):
            raise ValueError("terminal bookkeeping changed training/candidates/accounting counters")
    elapsed = time.monotonic()-began
    if elapsed > CPU_RESERVE_SECONDS-30:
        raise ValueError("terminal publication reserve exhausted; holds retained")
    receipt = signed({"schema":"nca-ik-polar-terminal-no-go/v1", "declaration_sha256":declaration["sha256"],
        "admission_file_sha256":sha256_file(review_path), "result":result,
        "pre_terminal_ledger_states":states, "whole_process_seconds_before_publication":elapsed,
        "charged_cpu_reservation_seconds_per_seed":CPU_RESERVE_SECONDS,
        "publication_reserve_seconds_per_seed":60., "old_failure_holds_retained":True,
        "resume_forbidden":True})
    immutable_json(Path(declaration["run"]) / "terminal-no-go.json", receipt)
    immutable_json(root / "terminal.json", receipt)
    if time.monotonic()-began > CPU_RESERVE_SECONDS:
        immutable_json(root / "overrun.json", {"terminal_unverified":True,"resume_forbidden":True})
        raise ValueError("terminal publication overrun retained")
    publish_supervisor(root, began)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run", "_derive"))
    for name in ("run", "fresh", "prepass", "declaration", "review"):
        parser.add_argument("--"+name)
    args = parser.parse_args()
    if args.command == "declare":
        declare(args.run, args.fresh, args.prepass, args.declaration)
    elif args.command == "run":
        execute(args.declaration, args.review)
    else:
        declaration = verify_signed(read(args.declaration))
        root = Path(args.declaration).parent
        # Exclusive before costly CPU verification; duplicate children refuse
        # before selection and retain the first worker claim on failure.
        immutable_json(root / "worker-started.json", {
            "declaration_sha256":declaration["sha256"], "created_utc":utc_now()})
        if declaration["source"] != _source(None) or declaration["runtime"] != runtime_expectation():
            raise ValueError("exact terminal worker source/runtime required")
        admission(args.review, declaration)
        if read(root / "started.json")["declaration_sha256"] != declaration["sha256"]:
            raise ValueError("supervised one-use terminal claim required")
        for seed in SEEDS:
            proof = read(Path(declaration["run"]) / f"seed-{seed}" / "overheads/charge-0001.json")["record"]["proof"]
            if proof["declaration_file_sha256"] != sha256_file(args.declaration):
                raise ValueError("all three durable terminal CPU debits required")
            validate_debit(ledger_at(declaration["run"], read(Path(declaration["run"]) / "declaration.json"), seed), proof)
        result = derive(declaration["run"], historical(declaration))
        result.update(terminal_declaration_sha256=declaration["sha256"],
            admission_file_sha256=sha256_file(args.review),
            debits={str(s):sha256_file(Path(declaration["run"]) / f"seed-{s}" / "overheads/charge-0001.json") for s in SEEDS})
        immutable_json(root / "worker-result.json", signed(result))


if __name__ == "__main__":
    main()

# Copyright 2026 techrote. All rights reserved.
