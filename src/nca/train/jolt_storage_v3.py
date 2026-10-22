"""Writer-enforced byte envelopes for LEARN-JOLT-003, without whole-file copies.

The limits couple artifacts to the fixed number of confirmed decisions and
starts. They bound the entire declared workload, including safe-retirement
metadata; a per-file reader limit alone is not an aggregate disk proof.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import stat

from nca.contracts import ContractError, Record, canonical_json, strict_json_loads
from nca.evaluation.durable_journal import file_identity, sync_directory
from nca.evaluation.verified_journal import VerifiedDurableJournal
from nca.train.jolt_budget_v3 import MAX_PUBLISHED_BYTES, MAX_SPOOL_BYTES

SERIALIZATION_SCHEMA = "nca-jolt-bounded-serialization/v3"
EPISODE_BASE_BYTES = 64 * 1024
EPISODE_DECISION_BYTES = 128 * 1024
MAX_EPISODE_DECISIONS = 40
MAX_ORDINARY_JSON_BYTES = 64 * 1024 * 1024
MAX_METRICS_BYTES = 48 * 1024
MAX_OTHER_METRICS_BYTES = 8 * 1024
MAX_AUTHORED_EPISODES = 84
MAX_WORKER_RESULT_FIXED_BYTES = 8 * 1024 * 1024
MAX_LEARNING_JSON_BYTES = 16 * 1024
MAX_JOURNAL_RECORD_BYTES = 64 * 1024
MAX_JOURNAL_RECORDS = 115110
MAX_JOURNAL_BYTES = MAX_JOURNAL_RECORD_BYTES * MAX_JOURNAL_RECORDS
MAX_SCALAR_BYTES = 64 * 1024
CHUNK_BYTES = 64 * 1024


def canonical_chunks(value, *, depth=0):
    """The existing canonical JSON bytes, streamed over immutable records.

    No thawed plain-tree or validation-tree copy is constructed. Keys and
    scalars remain strictly typed and finite under the same 64-level limit.
    Bounded strings prevent one primitive from allocating an enormous chunk.
    """
    if depth > 64:
        raise ContractError("nesting_limit", "bounded publication nesting exceeded")
    if isinstance(value, Record):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise ContractError("invalid_type", "canonical object keys must be strings")
        yield b"{"
        for index, key in enumerate(sorted(value)):
            if index:
                yield b","
            yield from canonical_chunks(key, depth=depth + 1)
            yield b":"
            yield from canonical_chunks(value[key], depth=depth + 1)
        yield b"}"
    elif isinstance(value, (list, tuple)):
        yield b"["
        for index, item in enumerate(value):
            if index:
                yield b","
            yield from canonical_chunks(item, depth=depth + 1)
        yield b"]"
    else:
        if not (value is None or type(value) in (bool, int, str)
                or type(value) is float and math.isfinite(value)):
            raise ContractError("invalid_json", "bounded publication requires finite JSON scalars")
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_SCALAR_BYTES:
            raise ContractError("jolt_storage_scalar", "a scalar exceeds the v3 publication bound")
        yield encoded


def episode_byte_limit(decisions):
    if type(decisions) is not int or not 0 <= decisions <= MAX_EPISODE_DECISIONS:
        raise ContractError("jolt_storage_decisions", "an episode has zero to forty confirmed physical decisions")
    return EPISODE_BASE_BYTES + EPISODE_DECISION_BYTES * decisions


def metrics_byte_limit_v3(kind):
    """Only the 84 authored captures can retain per-decision recovery events."""
    if kind not in ("authored", "idle", "bc", "residual"):
        raise ContractError("jolt_storage_metrics", "metrics require an original declared comparator")
    return MAX_METRICS_BYTES if kind == "authored" else MAX_OTHER_METRICS_BYTES


def validate_worker_result_v3(value, *, check=None):
    """Bind row-family multiplicity and the remaining fixed JSON before write."""
    rows = value.get("results")
    if (not isinstance(rows, list) or len(rows) > 6464
            or sum(row.get("kind") == "authored" for row in rows) > MAX_AUTHORED_EPISODES
            or sum(row.get("kind") != "authored" for row in rows) > 6464 - MAX_AUTHORED_EPISODES):
        raise ContractError("jolt_storage_worker", "worker rows exceeded the frozen comparator multiplicities")
    def bounded_size(item, maximum):
        if check is not None:
            check()
        size = 0
        boundary = CHUNK_BYTES
        for chunk in canonical_chunks(item):
            size += len(chunk)
            if check is not None and size >= boundary:
                check()
                boundary = size + CHUNK_BYTES
            if size > maximum:
                raise ContractError("jolt_storage_worker", "worker result exceeded its source-derived JSON shape")
        return size
    fixed = bounded_size({key: item for key, item in value.items() if key != "results"}, MAX_WORKER_RESULT_FIXED_BYTES)
    row_bytes = sum(bounded_size(row, metrics_byte_limit_v3(row.get("kind"))) for row in rows)
    # Restoring the results key, array separators and final LF adds fewer than
    # two bytes per row plus 32 fixed bytes. This is below the 64 MiB reader.
    upper = fixed + row_bytes + 2 * len(rows) + 32
    if upper > MAX_ORDINARY_JSON_BYTES:
        raise ContractError("jolt_storage_worker", "worker result cannot fit the unchanged ordinary reader")
    return {"fixed_bytes": fixed, "rows_bytes": row_bytes, "total_upper_bytes": upper}


def compressed_byte_limit(raw_bytes):
    # Above zlib's compressBound plus the deterministic gzip envelope for every
    # admitted raw length (and never used to enlarge an uncompressed reader).
    return raw_bytes + math.ceil(raw_bytes / 1000) + 64


def _write_chunks(stream, chunks, *, maximum_bytes, check=None):
    digest, size, buffer = hashlib.sha256(), 0, bytearray()
    for chunk in chunks:
        size += len(chunk)
        if size > maximum_bytes:
            raise ContractError("jolt_storage_bytes", "serialized artifact exceeds its prospective byte envelope")
        digest.update(chunk)
        buffer.extend(chunk)
        if len(buffer) >= CHUNK_BYTES:
            if check is not None:
                check()
            if stream.write(buffer) != len(buffer):
                raise OSError("short bounded publication")
            buffer.clear()
    if buffer:
        if check is not None:
            check()
        if stream.write(buffer) != len(buffer):
            raise OSError("short bounded publication")
    return {"bytes": size, "sha256": digest.hexdigest()}


def file_identity_v3(path, *, check=None, maximum_bytes=None):
    """Hash fixed-size chunks with deadline checks and unchanged-file evidence."""
    path = Path(path)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or maximum_bytes is not None and before.st_size > maximum_bytes):
        raise ContractError("jolt_storage_identity", "a bounded original regular artifact is required")
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while True:
            if check is not None:
                check()
            block = stream.read(CHUNK_BYTES)
            if not block:
                break
            count += len(block)
            if count > before.st_size:
                raise ContractError("jolt_storage_identity", "artifact length changed during byte readback")
            digest.update(block)
    after = path.lstat()
    attributes = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    if count != before.st_size or any(getattr(before, name) != getattr(after, name) for name in attributes):
        raise ContractError("jolt_storage_identity", "artifact identity changed during byte readback")
    return {"bytes": count, "sha256": digest.hexdigest()}


def publish_json_v3(path, value, *, maximum_bytes=MAX_ORDINARY_JSON_BYTES, check=None):
    """Immutable streaming final-path publication; interrupted bytes stay put."""
    if type(maximum_bytes) is not int or not 0 < maximum_bytes <= MAX_ORDINARY_JSON_BYTES:
        raise ContractError("jolt_storage_reader", "ordinary JSON remains within its original 64 MiB reader")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def payload():
        yield from canonical_chunks(value)
        yield b"\n"
    with path.open("xb") as stream:
        expected = _write_chunks(stream, payload(), maximum_bytes=maximum_bytes, check=check)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)
    actual = file_identity_v3(path, check=check)
    if actual != expected:
        raise ContractError("jolt_storage_readback", "bounded JSON changed during independent byte readback")
    if check is not None:
        check()
    return actual["sha256"]


def _write_episode(path, episode, *, decisions, compresslevel, check=None):
    """Same canonical episode/hash envelope, streamed and checked through EOF."""
    path = Path(path)
    maximum = episode_byte_limit(decisions)
    path.parent.mkdir(parents=True, exist_ok=True)
    def envelope():
        digest = hashlib.sha256()
        yield b'{"episode":'
        for chunk in canonical_chunks(episode):
            digest.update(chunk)
            yield chunk
        yield b',"sha256":"' + digest.hexdigest().encode() + b'"}\n'
    with path.open("xb") as target:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=compresslevel, fileobj=target, mtime=0) as zipped:
            raw = _write_chunks(zipped, envelope(), maximum_bytes=maximum, check=check)
        target.flush()
        os.fsync(target.fileno())
    sync_directory(path.parent)
    actual = file_identity_v3(path, check=check)
    if actual["bytes"] > compressed_byte_limit(maximum):
        raise ContractError("jolt_storage_compression", "compressed artifact exceeds the proven expansion envelope")
    digest, size = hashlib.sha256(), 0
    with gzip.open(path, "rb") as independent:
        for block in iter(lambda: independent.read(CHUNK_BYTES), b""):
            if check is not None:
                check()
            size += len(block)
            if size > maximum:
                raise ContractError("jolt_storage_expansion", "compressed artifact expanded beyond its original envelope")
            digest.update(block)
    if {"bytes": size, "sha256": digest.hexdigest()} != raw:
        raise ContractError("jolt_storage_readback", "complete gzip EOF readback differs from the serialized episode")
    return actual


def write_episode_v3(path, episode, *, check=None):
    # These source-bound fixed allowances make the episode-base calculation
    # reviewable independently of the per-decision payload and reader ceiling.
    for metadata in (episode.source_files, {"host_identity": episode.host_identity,
                                           "profile": episode.profile, "config": episode.config}):
        total = 0
        for chunk in canonical_chunks(metadata):
            total += len(chunk)
            if total > 16 * 1024:
                raise ContractError("jolt_storage_metadata", "fixed episode provenance exceeds its 16 KiB source allowance")
    return _write_episode(path, episode, decisions=episode.decisions, compresslevel=9, check=check)["sha256"]


def write_sample_episode_v3(path, episode, *, check=None):
    return _write_episode(path, episode, decisions=len(episode.decisions), compresslevel=1, check=check)


class BoundedVerifiedJournalV3(VerifiedDurableJournal):
    """The v2 verified durability mechanism with stricter declared byte bounds."""
    def write(self, payload):
        if self.records >= MAX_JOURNAL_RECORDS:
            raise ContractError("jolt_storage_journal", "journal exceeded the complete event-count envelope")
        # Metadata sizes are fixed regardless of their hash values. Check the
        # exact linked-record size before allowing the verified writer to append.
        if type(payload) is not bytes or len(payload) > MAX_JOURNAL_RECORD_BYTES:
            raise ContractError("jolt_storage_journal", "journal payload exceeds its bounded v3 reader")
        record = strict_json_loads(payload, max_bytes=MAX_JOURNAL_RECORD_BYTES)
        if not isinstance(record, dict) or "_journal" in record:
            raise ContractError("jolt_storage_journal", "journal metadata is writer-owned")
        record["_journal"] = {"sequence": self.records + 1,
                              "previous_sha256": self.chain_sha256, "record_sha256": "0" * 64}
        if len((canonical_json(record) + "\n").encode()) > MAX_JOURNAL_RECORD_BYTES:
            raise ContractError("jolt_storage_journal", "linked journal record exceeds the 64 KiB source envelope")
        return super().write(payload)

    def durability_receipt(self):
        result = super().durability_receipt()
        result.update(schema="nca-owner-journal-durability/v3", maximum_record_bytes=MAX_JOURNAL_RECORD_BYTES,
                      maximum_records=MAX_JOURNAL_RECORDS, maximum_stream_bytes=MAX_JOURNAL_BYTES)
        return result


def storage_bounds_v3():
    """Closed-form declared disk ceilings; no observations or training involved."""
    starts, decisions = 6464, 18944
    family_raw = starts * EPISODE_BASE_BYTES + decisions * EPISODE_DECISION_BYTES
    # ceil(n/1000) <= n/1000+1 per episode, conservatively rounded up overall.
    family_compressed = family_raw + math.ceil(family_raw / 1000) + starts * 65
    pieces = {
        "raw_episode_gzip": family_compressed,
        "sample_episode_gzip": family_compressed,
        "owner_journal": MAX_JOURNAL_BYTES,
        "live_reservation_spool": MAX_SPOOL_BYTES,
        "two_published_reservation_prefixes": 2 * MAX_PUBLISHED_BYTES,
        "metrics": MAX_AUTHORED_EPISODES * MAX_METRICS_BYTES + (starts - MAX_AUTHORED_EPISODES) * MAX_OTHER_METRICS_BYTES,
        # Worker/result/readback/inventory are individually bounded ordinary
        # JSON. Four possible partial files are retained rather than repaired.
        "fixed_ordinary_json": 13 * MAX_ORDINARY_JSON_BYTES,
        "learning_json": 27 * MAX_LEARNING_JSON_BYTES,
        "checkpoints": 36 * 1024 * 1024,
        "lifecycle_and_original_outer_evidence": 20 * 1024 * 1024,
    }
    return {"schema": SERIALIZATION_SCHEMA, "worker_files": 19469,
            "lifecycle_files": 17, "complete_lifecycle_files": 16, "outer_stdout_stderr_and_exit_files": 3,
            "maximum_journal_records": MAX_JOURNAL_RECORDS,
            "episode_starts": starts, "confirmed_decisions": decisions,
            "raw_per_family_bytes": family_raw, "pieces": pieces,
            "total_artifact_bytes": sum(pieces.values()),
            "required_disk_bytes_including_1_gib_reserve": sum(pieces.values()) + 1024 ** 3,
            "reservation_reader_bytes": 512 * 1024 * 1024,
            "ordinary_json_reader_bytes": MAX_ORDINARY_JSON_BYTES,
            "raw_episode_reader_bytes": 32 * 1024 * 1024,
            "sample_reader_bytes": MAX_ORDINARY_JSON_BYTES,
            "largest_episode_uncompressed_bytes": episode_byte_limit(40),
            "authored_metrics_bytes": MAX_METRICS_BYTES, "other_metrics_bytes": MAX_OTHER_METRICS_BYTES,
            "authored_episode_starts": MAX_AUTHORED_EPISODES,
            "maximum_worker_fixed_bytes": MAX_WORKER_RESULT_FIXED_BYTES,
            "maximum_worker_result_bytes": (MAX_AUTHORED_EPISODES * MAX_METRICS_BYTES
                + (starts-MAX_AUTHORED_EPISODES) * MAX_OTHER_METRICS_BYTES
                + MAX_WORKER_RESULT_FIXED_BYTES + 2 * starts + 32),
            "source_requires_independent_runtime_time_and_memory_admission": True}


class _BoundedCheckpointStream:
    def __init__(self, stream, *, check=None):
        self.stream, self.check = stream, check
        self.bytes = 0

    def write(self, data):
        if self.bytes + len(data) > 1024 * 1024:
            raise ContractError("jolt_storage_checkpoint", "fixed GRU32 bundle exceeds its 1 MiB publication envelope")
        if self.check is not None:
            self.check()
        written = self.stream.write(data)
        if written != len(data):
            raise OSError("short checkpoint publication")
        self.bytes += written
        return written

    def flush(self):
        return self.stream.flush()

    def tell(self):
        return self.bytes


def save_policy_v3(path, policy, *, base_policy=None, metadata=None, check=None):
    """Unchanged v2 policy bundle, bounded v3 storage and no alternate weights."""
    import torch
    from nca.controllers import jolt_learned
    if ((policy.kind == "residual") != (base_policy is not None)
            or base_policy is not None and base_policy.kind != "bc"):
        raise ValueError("residual checkpoint requires exactly one learned BC base")
    for model in (policy, base_policy):
        if model is not None and any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
            raise ValueError("cannot publish nonfinite policy weights")
    payload = {"format": jolt_learned.BUNDLE_FORMAT, **jolt_learned._bundle_model(policy),
               "base": None if base_policy is None else jolt_learned._bundle_model(base_policy),
               "metadata": dict(metadata or {}),
               "source_sha256": hashlib.sha256(Path(jolt_learned.__file__).read_bytes()).hexdigest()}
    if len(canonical_json(payload["metadata"]).encode()) > 16 * 1024:
        raise ContractError("jolt_storage_checkpoint", "checkpoint metadata exceeds the fixed bundle envelope")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, _BoundedCheckpointStream(stream, check=check))
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)
    return {"file_sha256": file_identity_v3(path, check=check, maximum_bytes=1024 * 1024)["sha256"], "policy_sha256": payload["policy_sha256"],
            "base_sha256": None if base_policy is None else jolt_learned.policy_state_sha256(base_policy)}
