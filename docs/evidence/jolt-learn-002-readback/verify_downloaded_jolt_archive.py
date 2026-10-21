#!/usr/bin/env python3
"""Independently verify a newly downloaded failed-evidence archive.

Only standard-library archive/hash operations and read-only git-show calls are
used. Checkpoints remain opaque bytes. No runtime, model, replay or training is
started, and no original artifact or downloaded archive is changed.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tarfile
import time

EXPECTED_ARCHIVE_SHA = "9c1862df7397f55b06f5e91061ac862785e3fcd84d38af7ab6887ffa7ec53fe8"
EXPECTED_ARCHIVE_BYTES = 106211945
EXPECTED_MEMBER_INDEX_SHA = "f04c225536ca31464a4cceb266e56104e8757ada183176c6df1a7ca6efab5693"
EXPECTED_MEMBERS = 765
EXPECTED_MEMBER_BYTES = 283352687
EXPECTED_INNER_GZIPS = 458
BLOCK = 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identify(stream):
    digest, size = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(BLOCK), b""):
        digest.update(block)
        size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def file_identity(path):
    with path.open("rb") as stream:
        return identify(stream)


def committed_bytes(repository, revision, path):
    return subprocess.run(["git", "-C", str(repository), "show", f"{revision}:{path}"],
                          check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--publication-commit", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    if not stat.S_ISREG(args.archive.lstat().st_mode):
        raise ValueError("download must be a regular file")
    actual_commit = subprocess.run(
        ["git", "-C", str(args.repository), "rev-parse", args.publication_commit],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.decode().strip()
    prefix = "docs/evidence/"
    member_path = prefix + "jolt-learn-002-archive-members-20261021.json"
    publication_path = prefix + "jolt-learn-002-index-20261021.json"
    member_index_bytes = committed_bytes(args.repository, actual_commit, member_path)
    publication_bytes = committed_bytes(args.repository, actual_commit, publication_path)
    if sha(member_index_bytes) != EXPECTED_MEMBER_INDEX_SHA:
        raise ValueError("committed archive index differs from declared identity")
    member_index = json.loads(member_index_bytes)
    publication_index = json.loads(publication_bytes)
    expected = member_index["members"]
    if len(expected) != EXPECTED_MEMBERS or sum(x["bytes"] for x in expected.values()) != EXPECTED_MEMBER_BYTES:
        raise ValueError("committed member inventory has unexpected scope")
    before = file_identity(args.archive)
    if before != {"bytes": EXPECTED_ARCHIVE_BYTES, "sha256": EXPECTED_ARCHIVE_SHA}:
        raise ValueError("downloaded archive differs from declared bytes")

    copied_by_member = {}
    copied_public_files = {}
    for filename, identity in publication_index["copied_original_files"].items():
        payload = committed_bytes(args.repository, actual_commit, prefix + filename)
        observed = {"bytes": len(payload), "sha256": sha(payload)}
        if observed != {k: identity[k] for k in ["bytes", "sha256"]}:
            raise ValueError("committed original copy differs from its publication index")
        copied_by_member[identity["archive_member"]] = payload
        copied_public_files[filename] = {**observed, "archive_member": identity["archive_member"]}
    if len(copied_by_member) != 6:
        raise ValueError("expected exactly six copied original files")

    observed_members, gzip_count, copied_matches = {}, 0, []
    gzip_decoded_bytes, checkpoints_hashed = 0, 0
    retained_small = {}
    small_names = {"run/result.json", "run/independent-readback.json", "outer/terminal.json"}
    with tarfile.open(args.archive, mode="r|gz") as archive:
        for member in archive:
            name, path = member.name, PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or name not in expected or name in observed_members):
                raise ValueError("undeclared, duplicate, nonregular or unsafe member")
            with archive.extractfile(member) as stream:
                if name.endswith(".gz"):
                    compressed = stream.read()
                    actual = {"bytes": len(compressed), "sha256": sha(compressed)}
                    with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as decoded:
                        decoded_identity = identify(decoded)
                    gzip_count += 1
                    gzip_decoded_bytes += decoded_identity["bytes"]
                elif name in copied_by_member or name in small_names:
                    payload = stream.read()
                    actual = {"bytes": len(payload), "sha256": sha(payload)}
                    if name in copied_by_member:
                        if payload != copied_by_member[name]:
                            raise ValueError("downloaded original differs from committed copy")
                        copied_matches.append(name)
                    if name in small_names:
                        retained_small[name] = payload
                else:
                    actual = identify(stream)
            if actual != expected[name] or actual["bytes"] != member.size:
                raise ValueError("downloaded archive member differs from committed inventory")
            observed_members[name] = actual
            checkpoints_hashed += name.endswith(".pt")
    if observed_members != expected or gzip_count != EXPECTED_INNER_GZIPS or len(copied_matches) != 6:
        raise ValueError("downloaded archive inventory or nested-gzip scope is incomplete")
    # tar readers may stop at end-of-archive blocks before the gzip trailer.
    # Read the complete gzip separately to authenticate its physical EOF.
    with gzip.open(args.archive, "rb") as decoded:
        outer_tar_identity = identify(decoded)
    after = file_identity(args.archive)
    if after != before:
        raise ValueError("downloaded archive changed during readback")
    result = json.loads(retained_small["run/result.json"])
    original_readback = json.loads(retained_small["run/independent-readback.json"])
    if (result["status"] != "failed" or result["qualification"]["qualified"] is not False
            or original_readback["verified"] is not False
            or original_readback["complete"] is not False
            or retained_small["outer/terminal.json"] != b""):
        raise ValueError("original failed verdict unexpectedly changed")
    missing_original_receipts = [name for name in
        ["lifecycle/executor.json", "lifecycle/finalization.json"] if name not in observed_members]
    receipt = {
        "schema": "nca-learn-jolt-002-downloaded-archive-readback/v1",
        "reviewer": "independent-archive-byte-audit",
        "role": "Independent result-artifact reviewer; not the original experiment executor or uploader.",
        "reviewed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "review_scope": "Read-only independent verification of a fresh separately materialized archive against the already committed original member index and six original repository copies.",
        "archive": {"filename": args.archive.name, **before},
        "committed_reference": {"revision": actual_commit,
            "member_index": {"path": member_path, "bytes": len(member_index_bytes), "sha256": sha(member_index_bytes)},
            "publication_index": {"path": publication_path, "bytes": len(publication_bytes), "sha256": sha(publication_bytes)}},
        "byte_readback": {"verified": True, "archive_matches_declared_bytes": True,
            "regular_members": len(observed_members),
            "member_bytes": sum(x["bytes"] for x in observed_members.values()),
            "every_member_matches_committed_inventory": True,
            "missing_members": [], "extra_members": [], "changed_members": [],
            "duplicates_or_unsafe_members": [], "full_outer_gzip_eof_verified": True,
            "outer_decoded_tar": outer_tar_identity,
            "nested_gzip_eofs_verified": gzip_count,
            "nested_gzip_uncompressed_bytes": gzip_decoded_bytes,
            "checkpoint_files_hashed_as_opaque_bytes": checkpoints_hashed,
            "downloaded_archive_unchanged": True},
        "copied_original_readback": {"verified": True, "files": copied_public_files,
            "matching_archive_members": sorted(copied_matches)},
        "original_experiment_verdict": {"status": result["status"],
            "qualified": result["qualification"]["qualified"],
            "original_auditor_verified": original_readback["verified"],
            "original_auditor_complete": original_readback["complete"],
            "original_auditor_error": original_readback["error"],
            "outer_terminal_bytes": len(retained_small["outer/terminal.json"]),
            "missing_original_completion_receipts": missing_original_receipts,
            "allocation_consumed": True, "retry_admitted": False,
            "complete_archive_readback_repairs_failed_experiment": False,
            "morphology_transfer_admitted": False, "owner_feel_accepted": False},
        "limits": [
            "This later readback verifies retained bytes after external preservation. It does not recreate missing owner records, final executor receipts, physical evaluation outcomes or a complete original wall interval.",
            "The same independent archive reviewer previously built the lossless transport; this is a separate pass over a fresh downloaded copy, independent of the original executor and uploader.",
            "Destination account identity and permissions are outside this content-only receipt and are not disclosed here."],
        "review_operations": {"host_starts": 0, "physical_ticks": 0, "optimizer_calls": 0,
            "model_constructions": 0, "checkpoint_deserializations": 0,
            "neural_forwards": 0, "torch_imports": 0, "original_auditor_replays": 0,
            "original_artifact_mutations": 0, "downloaded_archive_mutations": 0,
            "stdlib_only": True},
        "reproducer": {"filename": Path(__file__).name, **file_identity(Path(__file__))},
        "later_readback_wall_seconds": time.monotonic() - start,
    }
    with args.receipt.open("xb") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"receipt": args.receipt.name, **file_identity(args.receipt),
                      "archive_verified": True, "original_experiment_qualified": False}))


if __name__ == "__main__":
    main()

# © 2026 techrote. All rights reserved.
