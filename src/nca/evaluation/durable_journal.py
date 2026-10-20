"""Durable owner reservations and immutable, independently readable seals.

The live evidence is plain JSONL. Compression is a post-close transport, so a
worker killed before finalization leaves an honestly incomplete readable prefix.
No helper repairs, resumes or replaces a previous invocation.
"""

from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Mapping
import zlib

from nca.contracts import ContractError, canonical_json, sha256_json, strict_json_loads


MAX_RECORD_BYTES = 1_048_576
ZERO_HASH = "0" * 64


def sync_directory(path: Path) -> bool:
    """Flush directory entries where Python exposes the required operation.

    Windows file data is flushed, but directory-fsync support is not claimed.
    Unexpected POSIX failures remain errors instead of fictional durability.
    """
    if os.name == "nt":
        return False
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def publish_json(path: Path, value: Mapping[str, Any]) -> str:
    """Publish atomically without overwriting; retain staging bytes on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".pending")
    payload = (canonical_json(value) + "\n").encode()
    with staging.open("xb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    os.link(staging, path)
    sync_directory(path.parent)
    staging.unlink()
    sync_directory(path.parent)
    return hashlib.sha256(payload).hexdigest()


class DurableJournal:
    """Persist each complete hash-linked record before permitting owner I/O."""

    def __init__(self, path: Path):
        self.path = path
        self._stream = path.open("xb")
        try:
            self.directory_fsync = sync_directory(path.parent)
        except BaseException:
            self._stream.close()
            raise
        self.records = 0
        self.chain_sha256 = ZERO_HASH
        self.fsyncs = 0
        self.poisoned = False
        self.closed = False

    def write(self, payload: bytes) -> int:
        if self.closed or self.poisoned:
            raise ContractError("journal_writer_closed", "closed or uncertain journal cannot admit another owner request")
        if not payload.endswith(b"\n") or len(payload) > MAX_RECORD_BYTES:
            raise ContractError("journal_record", "one bounded complete JSONL record is required")
        record = strict_json_loads(payload)
        if not isinstance(record, dict) or "_journal" in record:
            raise ContractError("journal_record", "journal metadata is writer-owned")
        record["_journal"] = {"sequence": self.records + 1, "previous_sha256": self.chain_sha256}
        digest = sha256_json(record)
        record["_journal"]["record_sha256"] = digest
        encoded = (canonical_json(record) + "\n").encode()
        if len(encoded) > MAX_RECORD_BYTES:
            raise ContractError("journal_record", "linked record exceeds its declared size")
        try:
            if self._stream.write(encoded) != len(encoded):
                raise OSError("short journal write")
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except BaseException:
            self.poisoned = True
            raise
        self.records += 1
        self.fsyncs += 1
        self.chain_sha256 = digest
        return len(payload)

    def flush(self):
        # write() already flushed and fsynced every complete record.
        if self.closed or self.poisoned:
            raise ContractError("journal_writer_closed", "journal persistence is not confirmed")

    def close(self):
        if self.closed:
            return
        try:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self.fsyncs += 1
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self._stream.close()
            self.closed = True
        sync_directory(self.path.parent)

    def durability_receipt(self):
        return {"closed": self.closed, "poisoned": self.poisoned,
                "records": self.records, "record_fsyncs": self.records,
                "total_fsyncs": self.fsyncs, "directory_fsync": self.directory_fsync,
                "chain_sha256": self.chain_sha256}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def validate_record_chain(path: Path, *, compressed: bool = False):
    """Read the entire stream and reject lost, changed or reordered records."""
    digest, size, count, previous = hashlib.sha256(), 0, 0, ZERO_HASH
    opener = gzip.open if compressed else open
    try:
        with opener(path, "rb") as source:
            while True:
                line = source.readline(MAX_RECORD_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_RECORD_BYTES or not line.endswith(b"\n"):
                    raise ContractError("journal_chain", "complete bounded linked records are required")
                record = strict_json_loads(line)
                if not isinstance(record, dict) or not isinstance(record.get("_journal"), dict):
                    raise ContractError("journal_chain", "each durable record requires chain metadata")
                meta = record["_journal"]
                if (set(meta) != {"sequence", "previous_sha256", "record_sha256"}
                        or type(meta["sequence"]) is not int or meta["sequence"] != count + 1
                        or meta["previous_sha256"] != previous):
                    raise ContractError("journal_chain", "journal sequence or predecessor changed")
                claimed = meta.pop("record_sha256")
                if sha256_json(record) != claimed:
                    raise ContractError("journal_chain", "journal record bytes differ from their chain")
                previous = claimed
                digest.update(line)
                size += len(line)
                count += 1
    except (OSError, EOFError, zlib.error) as exc:
        raise ContractError("journal_stream", "complete readable journal bytes are required") from exc
    return {"sha256": digest.hexdigest(), "bytes": size, "records": count, "chain_sha256": previous}


def seal_owner_journal(raw_path: Path, *, expected_counts, expected_rows,
                       durability, validate_owner: Callable, families, artifact_root: Path):
    """Seal only after gzip EOF, raw equality and owner/episode replay agree."""
    if (durability.get("closed") is not True or durability.get("poisoned") is not False
            or durability.get("record_fsyncs") != durability.get("records")):
        raise ContractError("journal_durability", "uncertain live persistence cannot be sealed")
    raw = validate_record_chain(raw_path)
    if (raw["records"] != durability.get("records")
            or raw["chain_sha256"] != durability.get("chain_sha256")):
        raise ContractError("journal_durability", "closed bytes differ from the writer's durable receipt")
    compressed_path = raw_path.with_name(raw_path.name + ".gz")
    staging = compressed_path.with_name(compressed_path.name + ".pending")
    with staging.open("xb") as target:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=target, mtime=0) as compressed:
            with raw_path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    compressed.write(block)
        target.flush()
        os.fsync(target.fileno())
    if validate_record_chain(staging, compressed=True) != raw:
        raise ContractError("journal_transport", "compressed journal differs from the durable original")
    validation = validate_owner(staging, expected_counts=expected_counts,
        expected_rows=expected_rows, families=families, artifact_root=artifact_root)
    compressed_identity = file_identity(staging)
    os.link(staging, compressed_path)
    sync_directory(compressed_path.parent)
    staging.unlink()
    sync_directory(compressed_path.parent)
    seal = {"schema": "nca-owner-journal-seal/v1", "raw": {"path": raw_path.name, **raw},
            "compressed": {"path": compressed_path.name, **compressed_identity},
            "durability": dict(durability), "owner_validation": validation,
            "expected_counts_sha256": sha256_json(expected_counts),
            "expected_rows_sha256": sha256_json(expected_rows), "families": list(families)}
    seal_path = raw_path.with_name("owner-journal-seal.json")
    publish_json(seal_path, seal)
    return {"verified": True, "seal_sha256": file_identity(seal_path)["sha256"],
            "raw_sha256": raw["sha256"], "compressed_sha256": compressed_identity["sha256"],
            "owner_validation": validation}


def verify_sealed_journal(root: Path, *, expected_counts, expected_rows,
                          validate_owner: Callable, families):
    """Independent post-exit verification; no writer state or host is consulted."""
    seal_path = root / "owner-journal-seal.json"
    seal = strict_json_loads(seal_path.read_bytes())
    if (seal.get("schema") != "nca-owner-journal-seal/v1"
            or seal.get("expected_counts_sha256") != sha256_json(expected_counts)
            or seal.get("expected_rows_sha256") != sha256_json(expected_rows)
            or seal.get("families") != list(families)):
        raise ContractError("journal_seal", "seal does not bind the expected result and declaration")
    raw, compressed = root / "owner-journal.jsonl", root / "owner-journal.jsonl.gz"
    raw_chain = validate_record_chain(raw)
    if (seal.get("raw") != {"path": raw.name, **raw_chain}
            or seal.get("compressed") != {"path": compressed.name, **file_identity(compressed)}
            or validate_record_chain(compressed, compressed=True) != raw_chain):
        raise ContractError("journal_seal", "retained raw or compressed bytes differ from their seal")
    durability = seal.get("durability", {})
    if (durability.get("closed") is not True or durability.get("poisoned") is not False
            or durability.get("records") != raw_chain["records"]
            or durability.get("record_fsyncs") != raw_chain["records"]
            or durability.get("chain_sha256") != raw_chain["chain_sha256"]):
        raise ContractError("journal_seal", "seal lacks its complete persistence receipt")
    validation = validate_owner(compressed, expected_counts=expected_counts,
        expected_rows=expected_rows, families=families, artifact_root=root)
    if validation != seal.get("owner_validation"):
        raise ContractError("journal_seal", "independent owner/episode replay differs from the seal")
    return {"verified": True, "seal_sha256": file_identity(seal_path)["sha256"],
            "raw_sha256": raw_chain["sha256"], "compressed_sha256": file_identity(compressed)["sha256"],
            "owner_validation": validation}
