"""Versioned owner journal with independent path readback before dispatch.

The legacy writer and its historical receipts retain their original semantics.
This writer additionally checks file identity and length before every append,
reads the just-synchronized bytes through a separate descriptor, and replays the
entire closed stream before issuing a verified closure receipt. It cannot repair
an earlier journal or establish the cause of a historical missing suffix.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

from nca.contracts import ContractError, canonical_json, sha256_json, strict_json_loads
from nca.evaluation.durable_journal import MAX_RECORD_BYTES, ZERO_HASH, sync_directory, validate_record_chain


RECEIPT_SCHEMA = "nca-owner-journal-durability/v2"


class VerifiedDurableJournal:
    """Single-writer, append-only journal; a persistence doubt poisons admission.

    Appends check the previous record and the new record, not the full prefix.
    A same-length modification earlier in the prefix is rejected by full close
    replay. Neither fsync nor a local readback proves survival of every possible
    storage fault; independent final artifact verification remains mandatory.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.records = 0
        self.chain_sha256 = ZERO_HASH
        self.fsyncs = 0
        self.record_fsyncs = 0
        self.readback_records = 0
        self.bytes = 0
        self.poisoned = False
        self.closed = False
        self.closed_verified = False
        self.closed_identity = None
        self.directory_fsync = False
        self.final_directory_fsync = False
        self._digest = hashlib.sha256()
        self._tail = b""
        self._stream = self.path.open("xb", buffering=0)
        self._identity = self._file_key(os.fstat(self._stream.fileno()))
        try:
            self._check_path(0, check_position=True)
            self.directory_fsync = sync_directory(self.path.parent)
        except BaseException:
            self.poisoned = True
            self._stream.close()
            self.closed = True
            raise

    @staticmethod
    def _file_key(info):
        return info.st_dev, info.st_ino

    def _check_stat(self, info, expected_bytes):
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or self._file_key(info) != self._identity):
            raise ContractError("journal_identity", "the exclusively created owner journal was replaced or linked")
        if info.st_size != expected_bytes:
            raise ContractError("journal_length", "owner journal length differs from its verified append boundary")

    def _check_path(self, expected_bytes, *, check_position=False):
        # lstat rejects a pathname replaced with a symlink, even when its target
        # happens to be the original open file.
        self._check_stat(self.path.lstat(), expected_bytes)
        if not self._stream.closed:
            self._check_stat(os.fstat(self._stream.fileno()), expected_bytes)
            if check_position and self._stream.tell() != expected_bytes:
                raise ContractError("journal_offset", "the writer left its verified append boundary")

    def _read_span(self, offset, expected, *, total_bytes):
        self._check_path(total_bytes)
        with self.path.open("rb", buffering=0) as reader:
            self._check_stat(os.fstat(reader.fileno()), total_bytes)
            reader.seek(offset)
            if reader.read(len(expected)) != expected:
                raise ContractError("journal_readback", "independent owner-journal byte readback differs from the append")
            self._check_stat(os.fstat(reader.fileno()), total_bytes)
        self._check_path(total_bytes)

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
            self._check_path(self.bytes, check_position=True)
            if self._tail:
                self._read_span(self.bytes - len(self._tail), self._tail, total_bytes=self.bytes)
            if self._stream.write(encoded) != len(encoded):
                raise OSError("short journal write")
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self.fsyncs += 1
            self.record_fsyncs += 1
            self._read_span(self.bytes, encoded, total_bytes=self.bytes + len(encoded))
        except BaseException:
            self.poisoned = True
            raise
        self.records += 1
        self.readback_records += 1
        self.chain_sha256 = digest
        self.bytes += len(encoded)
        self._digest.update(encoded)
        self._tail = encoded
        return len(payload)

    def flush(self):
        if self.closed or self.poisoned:
            raise ContractError("journal_writer_closed", "journal persistence is not confirmed")
        try:
            self._check_path(self.bytes, check_position=True)
            if self._tail:
                self._read_span(self.bytes - len(self._tail), self._tail, total_bytes=self.bytes)
        except BaseException:
            self.poisoned = True
            raise

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
            try:
                self._stream.close()
            except BaseException:
                self.poisoned = True
                raise
            finally:
                self.closed = True
        # A prior uncertain append cannot be promoted by a successful later
        # close. Keep the readable prefix and its failed admission unchanged.
        if self.poisoned:
            return
        try:
            self.final_directory_fsync = sync_directory(self.path.parent)
            self._check_path(self.bytes)
            identity = validate_record_chain(self.path)
            self._check_path(self.bytes)
            expected = {"bytes": self.bytes, "sha256": self._digest.hexdigest(),
                        "records": self.records, "chain_sha256": self.chain_sha256}
            if identity != expected:
                raise ContractError("journal_readback", "the complete closed stream differs from independently verified appends")
            self.closed_identity = identity
            self.closed_verified = True
        except BaseException:
            self.poisoned = True
            raise

    def durability_receipt(self):
        return {"schema": RECEIPT_SCHEMA, "closed": self.closed, "poisoned": self.poisoned,
                "records": self.records, "record_fsyncs": self.record_fsyncs,
                "total_fsyncs": self.fsyncs, "directory_fsync": self.directory_fsync,
                "chain_sha256": self.chain_sha256, "readback_records": self.readback_records,
                "closed_verified": self.closed_verified,
                "final_directory_fsync": self.final_directory_fsync,
                "closed_identity": None if self.closed_identity is None else dict(self.closed_identity)}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
