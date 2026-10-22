"""Bounded standard-library readback of original LEARN-JOLT-003 journal bytes.

Importing this reader neither imports the optional learner nor starts a host.
The external inventory process and neural auditor share only the immutable
byte-chain contract; their operation and learning replay remain separate.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from nca.contracts import ContractError, sha256_json, strict_json_loads
from nca.train.jolt_storage_v3 import MAX_JOURNAL_RECORD_BYTES, MAX_JOURNAL_RECORDS, MAX_JOURNAL_BYTES


def _need(condition, detail):
    if not condition:
        raise ContractError("jolt_audit", detail)


def journal_identity_v3(path, *, check=None):
    """Verify linked original bytes with a 64 KiB buffer and a fixed row cap."""
    path = Path(path)
    if check is not None:
        check()
    _need(path.stat().st_size <= MAX_JOURNAL_BYTES, "v3 journal exceeds the original disk envelope")
    digest, size, count, previous = hashlib.sha256(), 0, 0, "0" * 64
    with path.open("rb") as stream:
        while True:
            if check is not None:
                check()
            line = stream.readline(MAX_JOURNAL_RECORD_BYTES + 1)
            if not line:
                break
            _need(len(line) <= MAX_JOURNAL_RECORD_BYTES and line.endswith(b"\n")
                  and count < MAX_JOURNAL_RECORDS, "v3 journal exceeds its bounded record/stream envelope")
            record = strict_json_loads(line, max_bytes=MAX_JOURNAL_RECORD_BYTES)
            _need(isinstance(record, dict) and isinstance(record.get("_journal"), dict),
                  "v3 journal omitted its original durable chain")
            meta = record["_journal"]
            _need(set(meta) == {"sequence", "previous_sha256", "record_sha256"}
                  and type(meta["sequence"]) is int and meta["sequence"] == count + 1
                  and meta["previous_sha256"] == previous, "v3 journal reordered or replaced original records")
            claimed = meta.pop("record_sha256")
            _need(sha256_json(record) == claimed, "v3 journal record differs from its durable chain")
            previous = claimed
            digest.update(line)
            size += len(line)
            count += 1
    return {"sha256": digest.hexdigest(), "bytes": size, "records": count, "chain_sha256": previous}
