"""CPU-first command line; optional host and learner modules load on demand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .contracts import (Command, ContractError, GameplayProfile, LegacyCommand,
                        Observation, WireAction, strict_json_loads)
from .environment import environment_report
from .randomness import ScenarioSplit
from .runs import RunBudget, RunManifest, RunStore


def source_revision() -> str:
    for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
        try:
            return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            continue
    raise ContractError("missing_revision", "pass --learner-revision for this installed package")


def print_json(value):
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="nca", description="Cybersand actor-control experiments")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="report the CPU environment without loading a GPU runtime")
    commands.add_parser("plugins", help="list the small registered CPU conformance extensions")
    validate = commands.add_parser("validate", help="validate a strict versioned JSON record")
    validate.add_argument("kind", choices=("profile", "command", "legacy-command", "observation", "wire-action"))
    validate.add_argument("path", type=Path)
    fixture = commands.add_parser("fixture", help="run a recorded contract example, not a game simulation")
    fixture.add_argument("--output", type=Path, default=Path("artifacts/records"))
    fixture.add_argument("--run-id", required=True)
    fixture.add_argument("--learner-revision")
    runs = commands.add_parser("runs", help="verify immutable records or produce a derived index")
    runs.add_argument("operation", choices=("verify", "index"))
    runs.add_argument("directory", type=Path)
    return result


def fixture_run(args) -> dict:
    from .builtins import builtin_registry
    started = time.perf_counter()
    registry = builtin_registry()
    adapter = registry.create("adapter", "recorded")
    policy = registry.create("policy", "idle")
    evaluator = registry.create("evaluator", "recorded")
    observation = adapter.observe()
    for sequence in range(3):
        observation = adapter.step(policy(observation, sequence))
    metrics = evaluator(observation)
    manifest = RunManifest(
        run_id=args.run_id, experiment="recorded-contract-example",
        learner_revision=args.learner_revision or source_revision(),
        host_revision=adapter.profile.host_revision,
        profile_sha256=adapter.profile.sha256,
        resolved_config={"decisions": 3, "adapter": "recorded@1", "policy": "idle@1"},
        environment=environment_report(), seeds=(0,),
        scenario_split=ScenarioSplit((), (), ("recorded-schema-v1",)),
        status="completed", budget=RunBudget(3, 120), decisions=3,
        runtime_seconds=time.perf_counter() - started, metrics=metrics,
    )
    RunStore(args.output).write(manifest)
    return {"run_id": manifest.run_id, "record_sha256": manifest.sha256,
            "status": manifest.status, "metrics": metrics}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            print_json(environment_report())
        elif args.command == "plugins":
            from .builtins import builtin_registry
            print_json([item.to_dict() for item in builtin_registry().list()])
        elif args.command == "validate":
            record_type = {"profile": GameplayProfile, "command": Command,
                           "legacy-command": LegacyCommand, "observation": Observation,
                           "wire-action": WireAction}[args.kind]
            record = record_type.from_dict(strict_json_loads(args.path.read_bytes()))
            print_json({"valid": True, "kind": args.kind, "sha256": record.sha256})
        elif args.command == "fixture":
            print_json(fixture_run(args))
        elif args.command == "runs":
            store = RunStore(args.directory)
            if args.operation == "verify":
                print_json({"verified": list(store.verify())})
            else:
                print("| Run | Experiment | Status | Evidence lane | Config SHA-256 |")
                print("|---|---|---|---|---|")
                for record in store.list():
                    print(f"| {record.run_id} | {record.experiment} | {record.status} | "
                          f"{record.gameplay_lane} | `{record.config_sha256}` |")
        return 0
    except (ContractError, OSError, ValueError) as exc:
        print_json({"error": getattr(exc, "code", type(exc).__name__), "detail": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
