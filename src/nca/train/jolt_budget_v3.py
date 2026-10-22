"""Bounded streaming reservations for the LEARN-JOLT-003 allocation.

The live ledger has O(counter count) resident state; it never builds or copies a
million-element Python list. Every operation is reserved before work. Native
operations additionally retain the independently read/fsynced owner journal.
Neural reservations are appended before work and flushed/fsynced at snapshots;
loss of a buffered tail is an incomplete failed allocation, never a retry or a
reconstructed success. Published prefixes are independently replayed in full.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import threading
import time

from nca.contracts import ContractError, strict_json_loads
from nca.evaluation.durable_journal import file_identity, sync_directory
from nca.train.jolt_execution_v3 import (
    COUNTERS, STAGES, PHASE_OFFSETS, WORKER_PHASES, RESERVATION_RECORDS_MAX,
    OriginalClockV3, phase_offsets_v3, validate_budget_v3,
)

RESERVATIONS_SCHEMA = "jolt-budget-reservations/v3"
SNAPSHOT_SCHEMA = "nca-jolt-learn-budget/v3"
MAX_RECORD_BYTES = 256
MAX_RESERVATION_BYTES = 512 * 1024 * 1024
MAX_HEADER_BYTES = 256
# Newlines/commas count as at most two extra bytes per record. This exact bound
# is below the unchanged reservation reader limit even at the loose sum of caps.
MAX_PUBLISHED_BYTES = MAX_HEADER_BYTES + RESERVATION_RECORDS_MAX * (MAX_RECORD_BYTES + 2) + 4
MAX_SPOOL_BYTES = RESERVATION_RECORDS_MAX * (MAX_RECORD_BYTES + 1)
assert MAX_PUBLISHED_BYTES < MAX_RESERVATION_BYTES


def _encoded(value):
    # All callers below already constrain plain dictionaries, strings, finite
    # numeric leaves and booleans; this is byte-identical to canonical_json.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _error(reason, code="jolt_v3_reservations"):
    raise ContractError(code, reason)


def _file_key(value):
    return value.st_dev, value.st_ino, value.st_mode, value.st_nlink


def _validate_row(row, index, *, phase_offsets=PHASE_OFFSETS):
    if (not isinstance(row, dict) or set(row) != {"index", "elapsed_seconds", "stage", "cleanup", "increments", "phase"}
            or type(row["index"]) is not int or row["index"] != index
            or type(row["cleanup"]) is not bool or row["stage"] not in (None, *STAGES)
            or row["phase"] not in WORKER_PHASES
            or type(row["elapsed_seconds"]) not in (int, float)
            or not math.isfinite(row["elapsed_seconds"])
            or row["elapsed_seconds"] < 0
            or (phase_offsets[row["phase"]] is not None
                and row["elapsed_seconds"] >= phase_offsets[row["phase"]])
            or not isinstance(row["increments"], dict) or not row["increments"]):
        _error("reservation has an invalid versioned shape, phase or original time")
    for key, amount in row["increments"].items():
        if key not in COUNTERS or type(amount) is not int or not 0 < amount <= COUNTERS[key]:
            _error("reservation has an invalid operation or amount")
    return row


class ReservationRecords:
    """Repeatable bounded stream; len() is the immutable declared record count.

    Each full iteration verifies canonical rows, exact count, closed envelope,
    EOF, and unchanged file identity/length. The caller must exhaust iteration
    before trusting it. No per-record dictionaries survive the consumer loop.
    """
    def __init__(self, path, *, budget=None):
        # Missing mode information always retains the historical finite clock.
        # Only an explicitly validated budget may interpret a later timestamp
        # under the owner's no-wall-clock-limit mode. Header bytes stay exact.
        if budget is not None:
            validate_budget_v3(budget)
        self.phase_offsets = phase_offsets_v3(7200 if budget is None else budget["maximum_wall_seconds"])
        self.path = Path(path)
        info = self.path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_size > MAX_PUBLISHED_BYTES):
            _error("a bounded regular reservation artifact is required")
        self._stat = info
        self.schema = RESERVATIONS_SCHEMA
        with self.path.open("rb") as source:
            self._header = source.readline(MAX_HEADER_BYTES + 1)
        if len(self._header) > MAX_HEADER_BYTES or not self._header.endswith(b'"reservations":[\n'):
            _error("reservation artifact has no bounded versioned header")
        prefix = self._header[:-1] + b"]}"
        value = strict_json_loads(prefix, max_bytes=MAX_HEADER_BYTES)
        if (set(value) != {"schema", "reservation_count", "reservations"}
                or value["schema"] != RESERVATIONS_SCHEMA or value["reservations"] != []
                or type(value["reservation_count"]) is not int
                or not 0 <= value["reservation_count"] <= RESERVATION_RECORDS_MAX):
            _error("reservation header changed its schema or finite count")
        self.count = value["reservation_count"]
        if self._header != _header(self.count):
            _error("reservation header is not the canonical v3 envelope")

    def __len__(self):
        return self.count

    def _unchanged(self):
        now = self.path.lstat()
        before = self._stat
        if (_file_key(now) != _file_key(before) or now.st_size != before.st_size
                or now.st_mtime_ns != before.st_mtime_ns or now.st_ctime_ns != before.st_ctime_ns):
            _error("reservation artifact changed during readback")

    def __iter__(self):
        self._unchanged()
        with self.path.open("rb") as source:
            if source.read(len(self._header)) != self._header:
                _error("reservation header changed during readback")
            for index in range(self.count):
                line = source.readline(MAX_RECORD_BYTES + 3)
                suffix = b",\n" if index + 1 < self.count else b"\n"
                if not line.endswith(suffix) or len(line) > MAX_RECORD_BYTES + len(suffix):
                    _error("reservation has a truncated or oversized canonical row")
                encoded = line[:-len(suffix)]
                row = _validate_row(strict_json_loads(encoded, max_bytes=MAX_RECORD_BYTES), index,
                                    phase_offsets=self.phase_offsets)
                if _encoded(row) != encoded:
                    _error("reservation row is not canonical")
                yield row
            if source.read(5) != b"]}\n":
                _error("reservation envelope has a missing tail, surplus row or suffix")
        self._unchanged()


def _header(count):
    return (b'{"schema":"' + RESERVATIONS_SCHEMA.encode() + b'","reservation_count":'
            + str(count).encode() + b',"reservations":[\n')


def budget_exhaustion_v3(error):
    """Classify declared wall/counter exhaustion without hiding integrity errors."""
    code = getattr(error, "code", None)
    return isinstance(error, ContractError) and isinstance(code, str) and (
        code == "jolt_v3_phase_deadline" or code.startswith("budget_counter_")
        or code.startswith("budget_stage_"))


def reservation_records(path, *, budget=None):
    return ReservationRecords(path, budget=budget)


class BudgetLedgerV3:
    """One fresh append-only ledger, exact counters and declared clock mode."""
    def __init__(self, budget, *, started=None, spool_path, monotonic=time.monotonic):
        validate_budget_v3(budget)
        self.budget = deepcopy(budget)
        self._clock = monotonic
        self.original_clock = OriginalClockV3(monotonic() if started is None else started,
            maximum_wall_seconds=budget["maximum_wall_seconds"], monotonic=monotonic)
        self.started = self.original_clock.started
        self.seconds, self.reserve_seconds = budget["maximum_wall_seconds"], budget["cleanup_reserve_seconds"]
        self.deadline = None if self.seconds is None else self.started + self.seconds
        active = self.original_clock.phase_offsets["active"]
        self.active_deadline = None if active is None else self.started + active
        self.limits, self.stage_limits = deepcopy(COUNTERS), deepcopy(STAGES)
        self.counts = {key: 0 for key in self.limits}
        self.stage_counts = {stage: {key: 0 for key in limits} for stage, limits in self.stage_limits.items()}
        self.phase = "active"
        self.phase_history = [{"phase": self.phase, "elapsed_seconds": self.original_clock.elapsed()}]
        self.path = Path(spool_path)
        self._stream = self.path.open("xb", buffering=65536)
        self._key = _file_key(self.path.lstat())
        self._count, self._bytes = 0, 0
        self._digest = hashlib.sha256(b"[")
        self._tail = b""
        self._closed = self._poisoned = False
        self._lock = threading.RLock()
        sync_directory(self.path.parent)

    def enter_phase(self, phase):
        with self._lock:
            if self._closed or self._poisoned:
                _error("closed or uncertain ledger cannot enter another phase")
            current = WORKER_PHASES.index(self.phase)
            if phase not in WORKER_PHASES or WORKER_PHASES.index(phase) != current + 1:
                _error("worker phases must advance once in their declared order", "jolt_v3_phase")
            self.original_clock.check(phase)
            self.phase = phase
            self.phase_history.append({"phase": phase, "elapsed_seconds": self.original_clock.elapsed()})
            return self.snapshot()

    def check(self, *, cleanup=False, allowance_seconds=0.):
        if self._closed or self._poisoned:
            _error("closed or uncertain ledger cannot admit more work")
        if type(cleanup) is not bool:
            _error("cleanup marker must be boolean")
        if self.phase != "active" and not cleanup:
            _error("ordinary work cannot consume a reserved worker-tail interval", "jolt_v3_phase")
        self.original_clock.check(self.phase, operation_seconds=allowance_seconds)

    def remaining_seconds(self, *, cleanup=False):
        if self.phase != "active" and not cleanup:
            return 0.
        return self.original_clock.remaining(self.phase)

    def reserve(self, counter, amount=1, *, stage=None, cleanup=False, allowance_seconds=0.):
        return self.reserve_many({counter: amount}, stage=stage, cleanup=cleanup,
                                 allowance_seconds=allowance_seconds)

    def reserve_many(self, increments, *, stage=None, cleanup=False, allowance_seconds=0.):
        with self._lock:
            self.check(cleanup=cleanup, allowance_seconds=allowance_seconds)
            if not isinstance(increments, dict) or not increments or stage not in (None, *self.stage_limits):
                _error("a nonempty declared reservation and stage are required")
            if self.phase == "publication":
                _error("publication admits no model or physical operation", "jolt_v3_phase")
            if self.phase == "audit" and (not cleanup or stage is not None
                    or not set(increments) <= {"audit_forward", "checkpoint_loads"}):
                _error("only the independent audit may consume the audit interval", "jolt_v3_phase")
            if self.phase == "closure" and (not cleanup or not set(increments) <= {
                    "owner_requests", "owner_cancel", "owner_close", "host_terminations",
                    "uncertain_ticks", "episodes_aborted", "episodes_completed"}):
                _error("closure admits only finite native revocation and accounting", "jolt_v3_phase")
            for counter, amount in increments.items():
                if counter not in self.limits or type(amount) is not int or amount <= 0:
                    _error("reservation names an invalid operation or amount")
                if self.counts[counter] + amount > self.limits[counter]:
                    _error("prospective counter exhausted: " + counter, "budget_counter_" + counter)
                if counter in {"decisions", "requested_ticks", "episodes_started"} and stage is None:
                    _error("physical work requires its original stage")
                if stage is not None and counter in self.stage_limits[stage]:
                    if self.stage_counts[stage][counter] + amount > self.stage_limits[stage][counter]:
                        _error("prospective stage counter exhausted", "budget_stage_" + stage + "_" + counter)
            record = {"index": self._count, "elapsed_seconds": self.original_clock.elapsed(),
                      "stage": stage, "cleanup": cleanup, "increments": dict(increments), "phase": self.phase}
            encoded = _encoded(record)
            if len(encoded) > MAX_RECORD_BYTES or self._count >= RESERVATION_RECORDS_MAX:
                _error("reservation serialization exceeds its prospective bound")
            line = encoded + b"\n"
            try:
                if self._stream.write(line) != len(line):
                    raise OSError("short reservation append")
            except BaseException:
                self._poisoned = True
                raise
            # Successful reservation precedes the operation. There is no refund
            # path, even for refused or uncertain physical work.
            if self._count:
                self._digest.update(b",")
            self._digest.update(encoded)
            self._count += 1
            self._bytes += len(line)
            self._tail = line
            for counter, amount in increments.items():
                self.counts[counter] += amount
                if stage is not None and counter in self.stage_counts[stage]:
                    self.stage_counts[stage][counter] += amount

    def _sync(self):
        try:
            if not self._stream.closed:
                self._stream.flush()
                os.fsync(self._stream.fileno())
            info = self.path.lstat()
            if (_file_key(info) != self._key or not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1 or info.st_size != self._bytes):
                _error("live reservation spool was replaced or changed")
            if self._tail:
                with self.path.open("rb") as independent:
                    independent.seek(self._bytes - len(self._tail))
                    if independent.read(len(self._tail)) != self._tail:
                        _error("live reservation tail failed independent readback")
        except BaseException:
            self._poisoned = True
            raise

    def snapshot(self):
        with self._lock:
            self._sync()
            digest = self._digest.copy()
            digest.update(b"]")
            return {"schema": SNAPSHOT_SCHEMA, "counts": dict(self.counts),
                    "stage_counts": deepcopy(self.stage_counts),
                    "wall_seconds": self.original_clock.elapsed(), "maximum_wall_seconds": self.seconds,
                    "cleanup_reserve_seconds": self.reserve_seconds, "reservation_count": self._count,
                    "reservations_sha256": digest.hexdigest(), "closed": self._closed,
                    "refused_or_uncertain_work_refunded": False,
                    "phase": self.phase, "phase_history": deepcopy(self.phase_history)}

    def publish_reservations(self, destination):
        """Publish and replay the exact immutable prefix without whole-file RAM."""
        with self._lock:
            self.check(cleanup=True)
            self._sync()
            destination = Path(destination)
            count, size, expected_digest = self._count, self._bytes, self._digest.copy()
            expected_digest.update(b"]")
            digest = hashlib.sha256(b"[")
            with self.path.open("rb") as source, destination.open("xb") as target:
                target.write(_header(count))
                for index in range(count):
                    if index % 512 == 0:
                        self.check(cleanup=True)
                    line = source.readline(MAX_RECORD_BYTES + 2)
                    if not line.endswith(b"\n") or len(line) > MAX_RECORD_BYTES + 1:
                        _error("reservation spool has a missing or oversized record")
                    encoded = line[:-1]
                    row = _validate_row(strict_json_loads(encoded, max_bytes=MAX_RECORD_BYTES), index,
                                        phase_offsets=self.original_clock.phase_offsets)
                    if _encoded(row) != encoded:
                        _error("reservation spool is not canonical")
                    if index:
                        digest.update(b",")
                    digest.update(encoded)
                    target.write(encoded + (b",\n" if index + 1 < count else b"\n"))
                if source.tell() != size or source.read(1):
                    _error("reservation spool has a missing prefix or surplus suffix")
                digest.update(b"]")
                if digest.hexdigest() != expected_digest.hexdigest():
                    _error("reservation spool prefix differs from charged work")
                target.write(b"]}\n")
                target.flush()
                os.fsync(target.fileno())
            sync_directory(destination.parent)
            independent = hashlib.sha256(b"[")
            for index, row in enumerate(reservation_records(destination, budget=self.budget)):
                if index % 512 == 0:
                    self.check(cleanup=True)
                if index:
                    independent.update(b",")
                independent.update(_encoded(row))
            independent.update(b"]")
            if independent.hexdigest() != expected_digest.hexdigest():
                _error("published reservations failed independent full-prefix readback")
            self.check(cleanup=True)
            identity = file_identity(destination)
            self.check(cleanup=True)
            return identity

    def close(self):
        with self._lock:
            self.check(cleanup=True)
            if self.phase != "publication":
                _error("ledger closes only after full audit and publication phase entry", "jolt_v3_phase")
            self._sync()
            self._stream.close()
            self._closed = True
            return self.snapshot()
