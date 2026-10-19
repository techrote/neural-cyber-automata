"""Capture the single declared T110 launch correction without resetting budget.

The learner runs from its separate clean original checkout. This wrapper records
the launch and completion, preserves both process streams, and passes the
immutable import-failure receipt to the learner's existing budget mechanism.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "output", "python", "host", "godot", "declaration",
                 "correction", "prior-budget", "failed-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--publication-commit", required=True)
    parser.add_argument("--expected-correction-sha256", required=True)
    parser.add_argument("--check-only", action="store_true",
                        help="validate files and source identity without a host, import or launch")
    args = parser.parse_args(argv)
    declaration = json.loads(args.declaration.read_bytes())
    correction = json.loads(args.correction.read_bytes())
    prior = json.loads(args.prior_budget.read_bytes())
    failed = json.loads(args.failed_receipt.read_bytes())
    source, output = args.source_root.resolve(), args.output.resolve()
    # Do not resolve this symlink: Python uses the entry-point location to find
    # pyvenv.cfg and the environment's installed dependencies.
    python = args.python.absolute()
    if not re.fullmatch(r"[0-9a-f]{40}", args.publication_commit):
        raise ValueError("publication commit must be an exact Git SHA")
    if sha(args.correction) != args.expected_correction_sha256:
        raise ValueError("correction differs from the independently verified publication")
    if sha(args.declaration) != correction["original_declaration"]["file_sha256"]:
        raise ValueError("original reproduction declaration differs")
    if sha(args.failed_receipt) != correction["failed_invocation"]["receipt_sha256"]:
        raise ValueError("failed invocation receipt differs")
    if sha(args.prior_budget) != correction["budget"]["prior_budget_file_sha256"]:
        raise ValueError("prior-budget file differs")
    expected_prior = [{"id": "t110-launch-import-failure-20261019",
                       "content_sha256": sha(args.failed_receipt), "game_decisions": 0,
                       "wall_seconds": failed["process_wall_seconds"]}]
    if prior != {"schema": "nca-cellular-prior-budget/v1", "receipts": expected_prior}:
        raise ValueError("the failed invocation must consume the original budget")
    revision, tree = git(source, "rev-parse", "HEAD"), git(source, "rev-parse", "HEAD^{tree}")
    if (revision != declaration["learner"]["revision"]
            or tree != declaration["learner"]["tree"] or git(source, "status", "--porcelain")):
        raise ValueError("learner must be the clean original frozen checkout")
    if not python.is_file() or not args.godot.is_file() or not args.host.is_dir():
        raise ValueError("the pinned local runtime paths must exist")
    if output.exists():
        raise FileExistsError("corrected launch requires a new output directory")
    if args.check_only:
        print(json.dumps({"status": "validated-without-execution", "learner_revision": revision,
                          "correction_sha256": sha(args.correction),
                          "prior_wall_seconds": failed["process_wall_seconds"]}, sort_keys=True))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    for path, name in ((args.declaration, "original-declaration.json"),
                       (args.correction, "correction-declaration.json"),
                       (args.prior_budget, "prior-budget-input.json"),
                       (args.failed_receipt, "failed-invocation-receipt.json")):
        with (output / name).open("xb") as stream:
            stream.write(path.read_bytes())
    launch = {
        "schema": "nca-t110-corrected-launch/v1", "reproduction_id": declaration["id"],
        "invocation_number": 2, "maximum_corrected_invocations": 1,
        "launch_utc": utc(), "publication_commit": args.publication_commit,
        "correction_sha256": sha(args.correction), "launcher_sha256": sha(Path(__file__)),
        "original_declaration_sha256": sha(args.declaration),
        "prior_budget_file_sha256": sha(args.prior_budget),
        "failed_receipt_sha256": sha(args.failed_receipt),
        "learner_revision": revision, "learner_tree": tree, "learner_clean_before_launch": True,
        "prior_game_decisions": 0, "prior_wall_seconds": failed["process_wall_seconds"],
        "aggregate_game_decision_limit": 4000, "aggregate_wall_second_limit": 900,
        "command": "PYTHONPATH=src <venv-python> tools/cellular_screen.py --source-root . --config configs/cellular/jolt-three-part-hop-v1.json --output screen --prior-budget prior-budget-input.json --host <pinned-host> --godot <pinned-godot>",
        "executable_path_preserves_virtual_environment": True,
    }
    write_json(output / "launch-receipt.json", launch)
    command = [str(python), "tools/cellular_screen.py", "--source-root", ".",
               "--config", "configs/cellular/jolt-three-part-hop-v1.json",
               "--output", str(output / "screen"), "--prior-budget", str(output / "prior-budget-input.json"),
               "--host", str(args.host.resolve()), "--godot", str(args.godot.resolve())]
    environment = dict(os.environ, PYTHONPATH="src", PYTHONUNBUFFERED="1")
    started = time.monotonic()
    print(json.dumps({"event": "corrected_launch_receipt_written", "launch_utc": launch["launch_utc"]}), flush=True)
    with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
        completed = subprocess.run(command, cwd=source, env=environment, stdout=stdout, stderr=stderr, check=False)
    completion = {
        "schema": "nca-t110-corrected-completion/v1", "reproduction_id": declaration["id"],
        "invocation_number": 2, "completed_utc": utc(), "process_exit_code": completed.returncode,
        "process_wall_seconds": time.monotonic() - started,
        "learner_clean_after_execution": not bool(git(source, "status", "--porcelain")),
        "launch_receipt_sha256": sha(output / "launch-receipt.json"),
        "stdout_sha256": sha(output / "stdout.log"), "stderr_sha256": sha(output / "stderr.log"),
        "additional_corrected_invocations": 0,
    }
    write_json(output / "completion-receipt.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
