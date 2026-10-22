#!/usr/bin/env python3
"""Read-only admission diagnostics and durable one-shot GitHub CI claim.

Preflight and inventory never start a model or host session. Training is owned
exclusively by the existing freeze-v3 and observe_jolt_v3.py contracts.
© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPOSITORY = "techrote/neural-cyber-automata"
ALLOCATION = "learn-jolt-003-20261022-a1"
CLAIM_MARKER = "NCA-JOLT-003-EXECUTION-CLAIM/v1:" + ALLOCATION
ISSUE_COMMENTS = "https://api.github.com/repos/" + REPOSITORY + "/issues/62/comments"
MAX_JSON_BYTES = 1_048_576


def _read(path):
    from nca.contracts import strict_json_loads
    file = Path(path)
    if file.is_symlink() or not file.is_file() or file.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("missing, substituted or oversized JSON evidence")
    return strict_json_loads(file.read_bytes(), max_bytes=MAX_JSON_BYTES)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _publish(path, value):
    """Create rather than overwrite evidence; fsync original bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    # The production execution lane is Linux; Windows unit-test fixtures
    # cannot fsync a directory handle using the POSIX API.
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _source_revision():
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _cgroup_memory():
    result = {}
    for key, name in (
        ("limit_bytes", "memory.max"),
        ("current_bytes", "memory.current"),
        ("peak_bytes", "memory.peak"),
    ):
        path = Path("/sys/fs/cgroup") / name
        try:
            raw = path.read_text().strip()
            result[key] = int(raw) if raw.isdecimal() else None
        except OSError:
            result[key] = None
    return result


def probe(args):
    """Record a source-bound environment sample, explicitly NOT admission."""
    from nca.contracts import sha256_json
    from nca.train.jolt_admission_v3 import validate_config_v3
    from nca.train.jolt_screen import source_inventory, runtime_identity
    from nca.train.jolt_storage_v3 import storage_bounds_v3

    config = _read(args.config)
    validate_config_v3(ROOT, config)
    if config["schema"] != "nca-jolt-learn-003/v2" or config["budget"]["maximum_wall_seconds"] is not None:
        raise ValueError("manual workflow requires explicit unlimited configuration v2")

    output_parent = Path(args.output_parent)
    if not output_parent.is_dir():
        raise ValueError("preflight output parent must exist")
    bounds = storage_bounds_v3()
    disk = shutil.disk_usage(output_parent)
    required = bounds["required_disk_bytes_including_1_gib_reserve"]
    files = source_inventory(ROOT)
    report = {
        "schema": "nca-jolt-003-ci-preflight/v1",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": _source_revision(),
        "source_sha256": sha256_json(files),
        "config_sha256": sha256_json(config),
        "runtime": runtime_identity(),
        "allocation_id": ALLOCATION,
        "serialisation_bounds": bounds,
        "disk": {"free_bytes": disk.free, "required_bytes": required,
                 "minimum_available": disk.free >= required},
        "cgroup_memory_sample": _cgroup_memory(),
        "full_worker_and_native_peak_memory_qualified": False,
        "source_and_runtime_admitted_for_physical_work": False,
        "physical_execution_started": False,
        "experiment_allocation_consumed": False,
        "status": "preflight_only" if disk.free >= required else "insufficient_free_disk",
    }
    _publish(args.receipt, report)
    print(json.dumps({"status": report["status"], "disk_gate": report["disk"]["minimum_available"],
                      "admitted": False, "allocation_consumed": False}))
    return 0 if disk.free >= required else 2


def attest(args):
    """Independent declaration readback after freeze; recheck actual free disk."""
    from nca.adapters.godot import inspect_host
    from nca.train.jolt_admission_v3 import validate_execution_v3, validate_resource_admission

    declaration = _read(args.declaration)
    config = _read(args.config)
    output = Path(args.output)
    if (output.exists() or output.with_name(output.name + "-lifecycle").exists()
            or declaration.get("allocation_id") != ALLOCATION):
        raise ValueError("this execution output identity is already used or mismatched")
    host = inspect_host(args.checkout, args.godot,
                        timeout=config["budget"]["maximum_owner_transport_seconds"])
    validate_execution_v3(ROOT, declaration, config=config, host_identity=host)
    validate_resource_admission(declaration, output_parent=output.parent)
    receipt = {
        "schema": "nca-jolt-003-ci-frozen-declaration-readback/v1",
        "source_revision": _source_revision(),
        "declaration_sha256": _hash(args.declaration),
        "allocation_id": ALLOCATION,
        "full_source_runtime_resource_and_reviews_admitted": True,
        "execution_started": False,
    }
    _publish(args.receipt, receipt)
    print(json.dumps({"status": "declaration_verified", "physical_execution_started": False}))
    return 0


def _github_api(url, token, *, body=None):
    headers = {"Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28",
               "Authorization": "Bearer " + token}
    raw = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    if raw is not None:
        headers["Content-Type"] = "application/json"
    operation = request.Request(url, data=raw, headers=headers,
                                method="POST" if raw is not None else "GET")
    # Network I/O safety is separate from the experiment's removed wall limit.
    with request.urlopen(operation, timeout=30) as response:
        return json.loads(response.read(8 * 1024 * 1024).decode("utf-8"))


def _prior_claim(token):
    # Fail closed if GitHub pagination cannot be read to exhaustion.
    for page in range(1, 102):
        entries = _github_api(ISSUE_COMMENTS + "?per_page=100&page=" + str(page), token)
        if not isinstance(entries, list):
            raise ValueError("GitHub issue comments are not a list")
        if any(CLAIM_MARKER in str(row.get("body", "")) for row in entries):
            return True
        if len(entries) < 100:
            return False
    raise ValueError("GitHub claim history exceeds complete readback bound")


def claim(args):
    """Permanent cross-run claim BEFORE opening the one physical invocation."""
    env = os.environ
    if (env.get("GITHUB_REPOSITORY") != REPOSITORY
            or env.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or env.get("GITHUB_REF_NAME") != "main"
            or env.get("GITHUB_RUN_ATTEMPT") != "1"
            or not re.fullmatch(r"[0-9]+", env.get("GITHUB_RUN_ID", ""))):
        raise ValueError("single-use claim requires first manual main-branch workflow attempt")
    token = env.get("GH_TOKEN")
    if not token:
        raise ValueError("GitHub issue-write token is unavailable")
    declaration = _read(args.declaration)
    attestation = _read(args.attestation)
    if (attestation.get("schema") != "nca-jolt-003-ci-frozen-declaration-readback/v1"
            or attestation.get("full_source_runtime_resource_and_reviews_admitted") is not True
            or attestation.get("execution_started") is not False
            or attestation.get("source_revision") != declaration.get("source_revision")
            or attestation.get("declaration_sha256") != _hash(args.declaration)
            or declaration.get("allocation_id") != ALLOCATION
            or len(declaration.get("independent_reviews", [])) < 2):
        raise ValueError("unreviewed or replaced declaration cannot consume allocation")
    if _prior_claim(token):
        raise ValueError("this allocation has already been claimed; retry is forbidden")

    body = ("\n".join((
        CLAIM_MARKER,
        "",
        "Single-use LEARN-JOLT-003 execution claim **before** any worker or physical task.",
        "Any subsequent dispatch or rerun must refuse this allocation, including after failure.",
        "Run: " + env["GITHUB_RUN_ID"] + ", first attempt",
        "Executable source: " + declaration["source_revision"],
        "Declaration SHA-256: " + _hash(args.declaration),
        "No assertion of training completion or qualification.",
    )))
    posted = _github_api(ISSUE_COMMENTS, token, body={"body": body})
    identifier = posted.get("id")
    if type(identifier) is not int or identifier < 1:
        raise ValueError("GitHub did not return a durable issue claim identity")
    reread = _github_api("https://api.github.com/repos/" + REPOSITORY + "/issues/comments/" + str(identifier), token)
    if reread.get("body") != body or reread.get("id") != identifier:
        raise ValueError("independent claim readback differs from publication")
    _publish(args.receipt, {
        "schema": "nca-jolt-003-ci-single-use-claim/v1",
        "allocation_id": ALLOCATION,
        "issue_comment_id": identifier,
        "issue_comment_url": reread.get("html_url"),
        "run_id": env["GITHUB_RUN_ID"],
        "source_revision": declaration["source_revision"],
        "declaration_sha256": _hash(args.declaration),
        "original_comment_readback_equal": True,
        "physical_execution_started_at_claim": False,
        "allocation_reserved_for_one_invocation": True,
    })
    print(json.dumps({"status": "allocation_claimed", "comment_id": identifier,
                      "invocation_started": False}))
    return 0


def inventory(args):
    """Hash all original/partial bytes without following links or overwriting data."""
    root = Path(args.root)
    rows = {}
    errors = []
    if not root.is_dir() or root.is_symlink():
        errors.append({"entry": ".", "error": "missing_or_substituted_evidence_root"})
    else:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            kind = path.lstat().st_mode
            if stat.S_ISDIR(kind):
                continue
            if not stat.S_ISREG(kind):
                errors.append({"entry": relative, "error": "unexpected_nonregular_entry"})
                continue
            try:
                rows[relative] = {"bytes": path.stat().st_size, "sha256": _hash(path)}
            except OSError:
                errors.append({"entry": relative, "error": "unreadable_file"})
    report = {
        "schema": "nca-jolt-003-ci-raw-evidence-inventory/v1",
        "complete_readback": not errors,
        "file_count": len(rows), "files": rows, "errors": errors,
        "scope": "unmodified available bytes; does not repair partial output or establish qualification",
    }
    _publish(args.receipt, report)
    print(json.dumps({"files": len(rows), "readback_complete": not errors}))
    return 0 if not errors else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    probe_parser = modes.add_parser("probe")
    for key in ("config", "output-parent", "receipt"):
        probe_parser.add_argument("--" + key, required=True, type=Path)
    attest_parser = modes.add_parser("attest")
    for key in ("config", "declaration", "checkout", "godot", "output", "receipt"):
        attest_parser.add_argument("--" + key, required=True, type=Path)
    claim_parser = modes.add_parser("claim")
    for key in ("declaration", "attestation", "receipt"):
        claim_parser.add_argument("--" + key, required=True, type=Path)
    inventory_parser = modes.add_parser("inventory")
    for key in ("root", "receipt"):
        inventory_parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    return {"probe": probe, "attest": attest, "claim": claim, "inventory": inventory}[args.mode](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Never print arbitrary exception messages containing workspace paths or tokens.
        print(json.dumps({"status": "refused", "error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(2)
