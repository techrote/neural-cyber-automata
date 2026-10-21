#!/usr/bin/env python3
"""Lossless read-only packaging of a failed allocation; no experiment execution.

This standard-library-only program accepts existing original files and creates a
new archive and separately dated byte audit. It never imports repository code,
deserializes checkpoints, constructs models or repairs original receipts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import time

BLOCK = 1024 * 1024
NAME = "jolt-learn-002-failed-evidence-20261021"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def reject_constant(value):
    raise ValueError("nonfinite JSON constant")


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def loads(data):
    return json.loads(data, object_pairs_hook=reject_duplicates,
                      parse_constant=reject_constant)


def identity_stream(stream):
    digest, size = hashlib.sha256(), 0
    while block := stream.read(BLOCK):
        digest.update(block)
        size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def identity(path):
    with path.open("rb") as stream:
        return identity_stream(stream)


def publish(path, value):
    with path.open("xb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def require_plain(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("original source is not a regular file")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    original = args.original_root
    run = original / "jolt-learn-002-20261021"
    lifecycle = original / "jolt-learn-002-20261021-lifecycle"
    sources = {}
    for directory, prefix in [(run, "run"), (lifecycle, "lifecycle")]:
        for path in sorted(directory.rglob("*")):
            if path.is_dir() and not path.is_symlink():
                continue
            require_plain(path)
            sources[f"{prefix}/{path.relative_to(directory).as_posix()}"] = path
    extras = {
        "outer/declaration.json": "jolt-execution-learn-002-20261021.json",
        "outer/terminal.json": "jolt-learn-002-20261021-terminal.json",
        "outer/command-stderr.log": "jolt-learn-002-20261021-command-stderr.log",
        "original-review/training-postmortem.json":
            "jolt-learn-002-training-postmortem-review-20261021.json",
    }
    for name, path in extras.items():
        require_plain(original / path)
        sources[name] = original / path
    sources = dict(sorted(sources.items()))
    members = {name: identity(path) for name, path in sources.items()}
    print(json.dumps({"phase": "original_hashes", "members": len(members),
                      "bytes": sum(x["bytes"] for x in members.values())}), flush=True)

    inventory = loads((run / "inventory.json").read_bytes())
    indexed = inventory["files"]
    present = {name.removeprefix("run/"): value for name, value in members.items()
               if name.startswith("run/")}
    mismatches = [name for name, value in indexed.items() if present.get(name) != value]
    missing = sorted(set(indexed) - set(present))
    unindexed = sorted(set(present) - set(indexed))

    gzip_records, gzip_errors, text_path_hits = {}, [], []
    private_tokens = [b"/workspace/", b"/root/", b"/home/", b"/tmp/",
                      b"C:\\Users\\", b"C:/Users/"]
    for name, path in sources.items():
        if name.endswith(".gz"):
            try:
                raw = gzip.decompress(path.read_bytes())
                loads(raw)
                gzip_records[name] = {"bytes": len(raw),
                                      "sha256": hashlib.sha256(raw).hexdigest()}
                if any(token in raw for token in private_tokens):
                    text_path_hits.append(name)
            except (OSError, EOFError, ValueError) as error:
                gzip_errors.append({"member": name, "type": type(error).__name__})
        elif path.suffix in {".json", ".jsonl", ".log"}:
            tail = b""
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(BLOCK), b""):
                    combined = tail + block
                    if any(token in combined for token in private_tokens):
                        text_path_hits.append(name)
                        break
                    tail = combined[-32:]

    chain_previous, count, chain_errors = "0" * 64, 0, []
    events, operations = Counter(), Counter()
    pending, responses, tail_events = {}, 0, []
    with (run / "owner-journal.jsonl").open("rb") as stream:
        for line in stream:
            count += 1
            try:
                if not line.endswith(b"\n") or len(line) > BLOCK:
                    raise ValueError("incomplete or oversized record")
                record = loads(line)
                meta = record["_journal"]
                if set(meta) != {"previous_sha256", "record_sha256", "sequence"}:
                    raise ValueError("invalid journal metadata")
                recorded_hash = meta.pop("record_sha256")
                if meta["sequence"] != count or meta["previous_sha256"] != chain_previous:
                    raise ValueError("sequence or predecessor mismatch")
                if hashlib.sha256(canonical(record)).hexdigest() != recorded_hash:
                    raise ValueError("record hash mismatch")
                chain_previous = recorded_hash
                event = record.get("event")
                events[event] += 1
                if event == "owner_request":
                    serial = record["serial"]
                    if serial in pending:
                        raise ValueError("duplicate pending request")
                    pending[serial] = record["operation"]
                    operations[record["operation"]] += 1
                elif event == "owner_response":
                    serial = record["serial"]
                    if serial not in pending:
                        raise ValueError("response without pending request")
                    del pending[serial]
                    responses += 1
                tail_events = (tail_events + [{k: record[k] for k in
                    ["event", "serial", "operation", "episode_id", "id", "index"]
                    if k in record}])[-5:]
            except (ValueError, KeyError, TypeError) as error:
                chain_errors.append({"record": count, "reason": str(error)})
                break

    manifest = loads((run / "manifest.json").read_bytes())
    source_mismatches = [name for name, expected in manifest["source_files"].items()
                         if not (args.repository / name).is_file()
                         or identity(args.repository / name)["sha256"] != expected]
    worker = loads((run / "worker-result.json").read_bytes())
    result = loads((run / "result.json").read_bytes())
    readback = loads((run / "independent-readback.json").read_bytes())
    claimed = result["closure"]["durability"]
    checkpoint_mismatches = [r["id"] for r in worker["checkpoints"]
                            if present.get(r["path"], {}).get("sha256") != r["file_sha256"]]
    artifact_mismatches = []
    for row in worker["results"]:
        for path_key, sha_key in [("artifact", "artifact_sha256"),
                                  ("sample_artifact", "sample_sha256")]:
            if present.get(row[path_key], {}).get("sha256") != row[sha_key]:
                artifact_mismatches.append({"episode": row["id"], "field": path_key})

    archive = args.output / f"{NAME}.tgz"
    with archive.open("xb") as target:
        with gzip.GzipFile(fileobj=target, filename="", mode="wb", compresslevel=6,
                           mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as tar:
                for name, path in sources.items():
                    info = tarfile.TarInfo(name)
                    info.size = members[name]["bytes"]
                    info.mode, info.mtime, info.uid, info.gid = 0o444, 0, 0, 0
                    info.uname = info.gname = ""
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)
        target.flush()
        os.fsync(target.fileno())
    archive_id = identity(archive)
    print(json.dumps({"phase": "archive_written", **archive_id}), flush=True)

    seen, archive_errors, gz_rechecked = {}, [], 0
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar:
            pure = PurePosixPath(member.name)
            if (not member.isfile() or pure.is_absolute() or ".." in pure.parts
                    or member.name in seen or member.name not in members):
                raise ValueError("archive member type, path or identity is invalid")
            with tar.extractfile(member) as stream:
                if member.name.endswith(".gz"):
                    payload = stream.read()
                    found = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                    raw = gzip.decompress(payload)
                    loads(raw)
                    if {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} != gzip_records[member.name]:
                        raise ValueError("archived gzip content changed")
                    gz_rechecked += 1
                else:
                    found = identity_stream(stream)
            seen[member.name] = found
            if found != members[member.name]:
                archive_errors.append(member.name)
    with gzip.open(archive, "rb") as stream:
        outer_eof = identity_stream(stream)
    original_changed = [name for name, path in sources.items() if identity(path) != members[name]]
    if seen != members or archive_errors or original_changed:
        raise ValueError("archive or original changed during publication")

    member_report = {
        "schema": "nca-learn-jolt-002-archive-members/v1",
        "archive": {"filename": archive.name, **archive_id},
        "member_count": len(members), "uncompressed_file_bytes": sum(x["bytes"] for x in members.values()),
        "members": members,
        "scope": "All retained original run files, original lifecycle and terminal/log files, unchanged prospective declaration, and the original metadata-only training postmortem. No reconstructed receipts.",
        "archive_report_self_included": False,
    }
    members_path = args.output / "jolt-learn-002-archive-members-20261021.json"
    publish(members_path, member_report)
    audit = {
        "schema": "nca-learn-jolt-002-archive-audit/v1",
        "reviewer": "independent-archive-byte-audit",
        "reviewed_utc_date": "2026-10-21",
        "review_scope": "Later standard-library-only byte/inventory/chain review and independent archive readback; no physical or neural replay and no original verdict repair.",
        "source_revision": manifest["source_revision"],
        "source_sha256": manifest["source_sha256"],
        "declaration_canonical_sha256": manifest["declaration_sha256"],
        "configuration_canonical_sha256": manifest["config_sha256"],
        "archive": {"filename": archive.name, **archive_id},
        "member_manifest": {"filename": members_path.name, **identity(members_path)},
        "original_inventory": {"file": "run/inventory.json", **members["run/inventory.json"],
            "indexed_files": len(indexed), "matching_indexed_files": len(indexed) - len(mismatches),
            "missing": missing, "mismatches": mismatches, "unindexed": unindexed,
            "unindexed_inventory_itself_expected": unindexed == ["inventory.json"]},
        "original_artifacts": {"inner_files": len(present), "all_members": len(members),
            "uncompressed_file_bytes": sum(x["bytes"] for x in members.values()),
            "gzip_files": len(gzip_records), "gzip_errors": gzip_errors,
            "checkpoint_files_verified_as_opaque_bytes": len(worker["checkpoints"]),
            "checkpoint_hash_mismatches": checkpoint_mismatches,
            "episode_rows": len(worker["results"]), "episode_sample_hash_mismatches": artifact_mismatches,
            "original_changes_during_review": original_changed,
            "private_path_token_hits_in_text": text_path_hits},
        "source_file_readback": {"files": len(manifest["source_files"]),
            "mismatches": source_mismatches,
            "reconstructed_inventory_sha256": hashlib.sha256(canonical(manifest["source_files"])).hexdigest()},
        "original_owner_journal": {"file": "run/owner-journal.jsonl", **members["run/owner-journal.jsonl"],
            "retained_records": count, "chain_sha256": chain_previous,
            "chain_prefix_errors": chain_errors, "recorded_writer_records": claimed["records"],
            "recorded_writer_chain_sha256": claimed["chain_sha256"],
            "missing_records_relative_to_writer": claimed["records"] - count,
            "writer_receipt_matches": count == claimed["records"] and chain_previous == claimed["chain_sha256"],
            "event_counts": dict(events), "retained_owner_request_counts": dict(operations),
            "retained_owner_responses": responses, "unanswered_retained_requests": pending,
            "last_retained_events": tail_events,
            "compressed_owner_journal_present": "owner-journal.jsonl.gz" in present,
            "raw_compressed_equality": "not_applicable_no_compressed_owner_journal",
            "interpretation": "An internally valid prefix does not establish the missing suffix or durable original closure."},
        "archive_readback": {"verified": seen == members and not archive_errors,
            "regular_members": len(seen), "missing_members": sorted(set(members)-set(seen)),
            "changed_members": archive_errors, "inner_gzip_eofs_rechecked": gz_rechecked,
            "outer_gzip_eof_verified": True, "outer_uncompressed_tar": outer_eof,
            "deterministic_metadata": "Sorted USTAR regular members; mode 0444; zero timestamp/uid/gid; empty owner names and gzip filename; gzip mtime zero."},
        "experiment_verdict": {"original_status": result["status"],
            "original_auditor_verified": readback["verified"],
            "original_auditor_complete": readback["complete"],
            "original_auditor_error": readback["error"],
            "original_qualified": result["qualification"]["qualified"],
            "open_evaluation_rows": result["costs"]["stage_counts"]["open"]["episodes_started"],
            "sealed_evaluation_rows": result["costs"]["stage_counts"]["sealed"]["episodes_started"],
            "required_sealed_rows": result["qualification"]["expected_rows"],
            "retained_final_checkpoints": sum(x["id"].endswith("-final") for x in worker["checkpoints"]),
            "required_final_checkpoints": 6,
            "outer_terminal_file_bytes": members["outer/terminal.json"]["bytes"],
            "missing_executor_completion_receipts": [x for x in ["executor.json", "finalization.json"] if x not in present],
            "whole_process_exit_time_proven": False,
            "allocation_consumed": True, "retry_permitted": False,
            "archive_completeness_is_experiment_completeness": False},
        "review_operations": {"host_starts": 0, "physical_ticks": 0,
            "optimizer_calls": 0, "checkpoint_deserializations": 0,
            "model_constructions": 0, "neural_forwards": 0, "torch_imports": 0,
            "original_artifact_mutations": 0, "original_auditor_replays": 0,
            "retry_started": False, "stdlib_only": True},
        "later_review_wall_seconds": time.monotonic() - start,
        "limitations": [
            "Counts from worker/result metadata are retained claims where the original owner suffix is absent.",
            "Hash and gzip checks authenticate retained bytes, not missing physical observations or learned correctness.",
            "This later review is outside the consumed original allocation and cannot extend or complete its clock.",
        ],
    }
    audit_path = args.output / "jolt-learn-002-archive-audit-20261021.json"
    publish(audit_path, audit)
    print(json.dumps({"phase": "complete", "archive": archive.name,
                      **archive_id, "members": len(members),
                      "gzip_files": len(gzip_records), "journal_records": count,
                      "inventory_mismatches": mismatches,
                      "audit_sha256": identity(audit_path)["sha256"]}), flush=True)


if __name__ == "__main__":
    main()

# © 2026 techrote. All rights reserved.
