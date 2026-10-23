"""Cumulative, write-ahead accounting for the compact IK training harness.

The ledger authenticates complete optimizer-boundary checkpoints with the
existing IK codec. It does not launch work, import Torch or authorize a run.
A supervisor must enforce the remaining whole-process wall allocation. An
unsealed attempt or work after the latest complete checkpoint blocks resume.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import re
import time
from typing import Any

from nca.contracts import canonical_json, sha256_json, strict_json_loads
from nca.ik.checkpoint import MAX_BYTES, load_checkpoint, restore_checkpoint, write_checkpoint
from nca.runs import immutable_json, sha256_file, utc_now


SCHEMA = "nca-ik-training-ledger/v1"
_ID_HASHES = (
    "declaration_sha256", "config_sha256", "source_sha256", "profile_sha256",
    "dataset_sha256", "normalization_sha256", "runtime_sha256",
)
_LIMITS = (
    "max_optimizer_steps", "max_sample_presentations", "max_tuning_calls",
    "max_evaluation_calls", "max_checkpoints", "max_attempts",
    "max_wall_seconds", "max_memory_bytes",
)
_COUNTERS = ("optimizer_steps", "sample_presentations", "tuning_calls",
             "evaluation_calls", "checkpoints")
_KINDS = {"optimizer", "presentation_only", "tuning", "evaluation", "checkpoint"}
_STATUSES = {"paused", "completed", "failed", "capped", "canceled"}
_JSON_LIMIT = 8 * 1024 * 1024


def _copy(value):
    return strict_json_loads(canonical_json(value), max_bytes=_JSON_LIMIT)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def _identity(value):
    if not isinstance(value, Mapping) or set(value) != {*_ID_HASHES, "seed"}:
        raise ValueError("training identity fields differ")
    result = _copy(dict(value))
    for name in _ID_HASHES:
        if type(result[name]) is not str or not re.fullmatch(r"[a-f0-9]{64}", result[name]):
            raise ValueError("training " + name + " must be a SHA-256")
    if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**32:
        raise ValueError("training seed must be a uint32")
    return result


def _allocation(value):
    if not isinstance(value, Mapping) or set(value) != set(_LIMITS):
        raise ValueError("training allocation fields differ")
    result = _copy(dict(value))
    for name in _LIMITS:
        if name != "max_wall_seconds":
            _integer(result[name], name, 0 if name in {
                "max_tuning_calls", "max_evaluation_calls"} else 1)
    wall = result["max_wall_seconds"]
    if type(wall) not in (int, float) or not math.isfinite(wall) or not 0 < wall <= 7200:
        raise ValueError("training allocation exceeds the finite two-hour ceiling")
    return result


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _JSON_LIMIT:
        raise ValueError("training ledger JSON is missing, linked or oversized")
    value = strict_json_loads(path.read_bytes(), max_bytes=_JSON_LIMIT)
    if not isinstance(value, dict):
        raise ValueError("training ledger JSON must be an object")
    return value


def _publish(path, record):
    record = _copy(record)
    envelope = {"record": record, "sha256": sha256_json(record)}
    if len(canonical_json(envelope).encode()) > _JSON_LIMIT:
        raise ValueError("training ledger JSON exceeds its byte limit")
    immutable_json(path, envelope)
    return envelope


def _record(path):
    envelope = _read(path)
    if (set(envelope) != {"record", "sha256"} or not isinstance(envelope["record"], dict)
            or envelope["sha256"] != sha256_json(envelope["record"])):
        raise ValueError("training ledger record/hash differs")
    return envelope


def _cost(kind, samples, calls):
    if kind not in _KINDS:
        raise ValueError("unknown training reservation kind")
    _integer(samples, "samples")
    _integer(calls, "calls", 1)
    if kind in {"optimizer", "presentation_only"}:
        if not 1 <= samples <= 1024 or calls != 1:
            raise ValueError("one bounded batch is required for a training reservation")
    elif samples != 0 or (kind == "checkpoint" and calls != 1):
        raise ValueError("non-training reservations use explicit call counts, not presentations")
    value = dict.fromkeys(_COUNTERS, 0)
    if kind == "optimizer":
        value["optimizer_steps"] = 1
    if kind in {"optimizer", "presentation_only"}:
        value["sample_presentations"] = samples
    if kind == "tuning":
        value["tuning_calls"] = calls
    if kind == "evaluation":
        value["evaluation_calls"] = calls
    if kind == "checkpoint":
        value["checkpoints"] = 1
    return value


def _add(charged, cost, allocation):
    result = {key: charged[key] + cost[key] for key in _COUNTERS}
    if any(result[key] > allocation["max_" + key] for key in _COUNTERS):
        raise ValueError("training operation exceeds the unchanged cumulative allocation")
    return result


def _checkpoint_file(root, relative):
    if (type(relative) is not str or "\\" in relative or ":" in relative
            or Path(relative).is_absolute() or ".." in Path(relative).parts):
        raise ValueError("training checkpoint path is not relative")
    path = Path(root) / relative
    if not path.resolve().is_relative_to(Path(root).resolve()) or path.is_symlink():
        raise ValueError("training checkpoint escapes the allocation")
    return path


def _verify_checkpoint(root, value, identity):
    if not isinstance(value, dict) or set(value) != {"path", "reference", "state"}:
        raise ValueError("training checkpoint receipt fields differ")
    target = _checkpoint_file(root, value["path"])
    reference, state = value["reference"], value["state"]
    if not isinstance(reference, dict) or not isinstance(state, dict):
        raise ValueError("training checkpoint reference/state must be objects")
    marker = _read(target.with_name(target.name + ".json"))
    if (marker != {"checkpoint": reference, "record_sha256": sha256_json(reference)}
            or reference.get("file") != target.name
            or reference.get("metadata") != identity
            or reference.get("training_state_sha256") != sha256_json(state)
            or not target.is_file() or target.stat().st_size != reference.get("bytes")
            or target.stat().st_size > MAX_BYTES or sha256_file(target) != reference.get("sha256")):
        raise ValueError("training checkpoint bytes, marker or state identity differ")
    if state.get("schema") != "nca-ik-training-state/v1" or state.get("boundary") != "optimizer_step":
        raise ValueError("training checkpoint is not a complete optimizer boundary")


def _inventory(attempt):
    result = {}
    for path in sorted(attempt.rglob("*")):
        if path.is_symlink():
            raise ValueError("training attempt cannot contain linked artifacts")
        if path.is_file() and path.name not in {"finish.json", "sealed.json"}:
            result[path.relative_to(attempt).as_posix()] = {
                "sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


class IKRunLedger:
    """One immutable per-seed allocation; failed work never becomes free work."""

    def __init__(self, root, header):
        self.root = Path(root)
        self.header = header
        self.identity = header["record"]["identity"]
        self.allocation = header["record"]["allocation"]
        self._attempt = None

    @classmethod
    def create(cls, root, *, identity, allocation):
        identity, allocation = _identity(identity), _allocation(allocation)
        destination = Path(root)
        destination.mkdir(parents=True, exist_ok=False)
        header = _publish(destination / "allocation.json", {
            "schema": SCHEMA, "identity": identity, "allocation": allocation})
        return cls(destination, header)

    @classmethod
    def open(cls, root, *, expected_identity, expected_allocation=None):
        header = _record(Path(root) / "allocation.json")
        record = header["record"]
        if (set(record) != {"schema", "identity", "allocation"}
                or record["schema"] != SCHEMA
                or _identity(record["identity"]) != _identity(expected_identity)):
            raise ValueError("training allocation identity differs")
        _allocation(record["allocation"])
        if expected_allocation is not None and record["allocation"] != _allocation(expected_allocation):
            raise ValueError("training allocation differs from the prospective limits")
        return cls(root, header)

    def inspect(self):
        """Verify all attempts, reservations and checkpoint bytes without Torch."""
        if _record(self.root / "allocation.json") != self.header:
            raise ValueError("training allocation changed after opening")
        charged = dict.fromkeys(_COUNTERS, 0)
        wall, committed, presentations, batches = 0.0, 0, 0, 0
        latest, parent, last_status = None, None, None
        summaries = []
        active = unresolved = False
        folder = self.root / "attempts"
        paths = sorted(folder.iterdir()) if folder.exists() else []
        if [p.name for p in paths] != [f"attempt-{i:04d}" for i in range(1, len(paths) + 1)]:
            raise ValueError("training attempt sequence differs")
        for number, attempt in enumerate(paths, 1):
            if attempt.is_symlink() or not attempt.is_dir():
                raise ValueError("training attempt must be a directory")
            if not (attempt / "start.json").exists():
                active = unresolved = True
                break
            start = _record(attempt / "start.json")
            expected_start = {
                "schema": SCHEMA, "attempt": number, "allocation_sha256": self.header["sha256"],
                "parent_sha256": parent, "prior_charged": charged,
                "prior_wall_seconds": wall, "prior_checkpoint": latest,
            }
            if set(start["record"]) != {*expected_start, "created_utc"} or any(
                    start["record"].get(key) != value for key, value in expected_start.items()):
                raise ValueError("training attempt lineage or counters differ")
            if number > self.allocation["max_attempts"] or last_status == "completed":
                raise ValueError("training attempt exceeds its declared lifecycle")
            previous = start["sha256"]
            events = sorted(attempt.glob("event-*.json"))
            if [p.name for p in events] != [f"event-{i:06d}.json" for i in range(1, len(events) + 1)]:
                raise ValueError("training event sequence differs")
            pending = None
            for sequence, path in enumerate(events, 1):
                event = _record(path)
                record = event["record"]
                if (set(record) != {"sequence", "previous_sha256", "kind", "payload"}
                        or record["sequence"] != sequence or record["previous_sha256"] != previous):
                    raise ValueError("training event chain differs")
                payload = record["payload"]
                if record["kind"] == "reserve":
                    if pending is not None or not isinstance(payload, dict) or set(payload) != {
                            "id", "kind", "samples", "calls", "call_id", "cost"}:
                        raise ValueError("training reservation sequence differs")
                    if (payload["id"] != f"{number:04d}-{sequence:06d}"
                            or type(payload["call_id"]) is not str or not 1 <= len(payload["call_id"]) <= 200):
                        raise ValueError("training reservation identity differs")
                    cost = _cost(payload["kind"], payload["samples"], payload["calls"])
                    if payload["cost"] != cost:
                        raise ValueError("training reservation cost differs")
                    charged = _add(charged, cost, self.allocation)
                    if payload["kind"] == "optimizer":
                        presentations += payload["samples"]
                    if payload["kind"] in {"optimizer", "presentation_only"}:
                        batches += 1
                    pending = payload
                elif record["kind"] == "complete":
                    if (pending is None or not isinstance(payload, dict)
                            or set(payload) != {"reservation_id", "receipt"}
                            or payload["reservation_id"] != pending["id"]
                            or not isinstance(payload["receipt"], dict)):
                        raise ValueError("training completion has no matching reservation")
                    if pending["kind"] == "optimizer":
                        committed += 1
                    if pending["kind"] == "checkpoint":
                        latest = payload["receipt"]
                        _verify_checkpoint(self.root, latest, self.identity)
                        state = latest["state"]
                        if (state.get("charged") != charged
                                or state.get("completed_optimizer_steps") != committed
                                or state.get("optimizer_presentations") != presentations
                                or state.get("consumed_batches") != batches
                                or state.get("reservation_id") != pending["id"]):
                            raise ValueError("training checkpoint accounting differs")
                    pending = None
                else:
                    raise ValueError("unknown training event kind")
                previous = event["sha256"]
            finish_path = attempt / "finish.json"
            if not finish_path.exists():
                if number != len(paths):
                    raise ValueError("a later attempt follows an unsealed attempt")
                active = unresolved = True
                break
            finish = _record(finish_path)
            record = finish["record"]
            if set(record) != {
                    "attempt", "start_sha256", "last_event_sha256", "event_count", "status", "reason",
                    "elapsed_seconds", "charged", "completed_optimizer_steps", "optimizer_presentations",
                    "consumed_batches", "checkpoint", "pending_reservation", "inventory"}:
                raise ValueError("training terminal fields differ")
            elapsed = record["elapsed_seconds"]
            if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("training elapsed time is invalid")
            if (record["attempt"] != number or record["start_sha256"] != start["sha256"]
                    or record["last_event_sha256"] != previous or record["event_count"] != len(events)
                    or record["charged"] != charged
                    or record["completed_optimizer_steps"] != committed
                    or record["optimizer_presentations"] != presentations or record["consumed_batches"] != batches
                    or record["checkpoint"] != latest or record["pending_reservation"] != pending
                    or record["inventory"] != _inventory(attempt)):
                raise ValueError("training terminal inventory or accounting differs")
            status, reason = record["status"], record["reason"]
            if status not in _STATUSES or type(reason) is not str or not reason:
                raise ValueError("training terminal outcome is missing")
            if status in {"paused", "completed"} and pending is not None:
                raise ValueError("successful boundary retains an unresolved reservation")
            wall += elapsed
            if wall > self.allocation["max_wall_seconds"] and status != "capped":
                raise ValueError("wall overrun must remain a capped outcome")
            parent, last_status = finish["sha256"], status
            summaries.append({"attempt": number, "status": status, "reason": reason,
                              "elapsed_seconds": elapsed, "record_sha256": parent,
                              "pending_reservation": pending})
            unresolved = unresolved or pending is not None
            seal_path = attempt / "sealed.json"
            if not seal_path.exists():
                active = unresolved = True
                if number != len(paths):
                    raise ValueError("a later attempt follows an unsealed terminal")
            elif _read(seal_path) != {"finish_sha256": finish["sha256"]}:
                raise ValueError("training attempt seal differs")
        if latest is not None:
            checkpoint_matches = (
                latest["state"]["charged"] == charged
                and latest["state"]["completed_optimizer_steps"] == committed
                and latest["state"]["optimizer_presentations"] == presentations
                and latest["state"]["consumed_batches"] == batches)
        else:
            checkpoint_matches = not any(charged.values())
        remaining = max(0.0, self.allocation["max_wall_seconds"] - wall)
        can_resume = (not active and not unresolved and checkpoint_matches
                      and last_status != "completed" and len(paths) < self.allocation["max_attempts"]
                      and remaining > 0)
        return {
            "schema": SCHEMA, "identity": _copy(self.identity), "allocation": _copy(self.allocation),
            "attempt_count": len(paths), "attempts": summaries, "charged": charged,
            "charged_wall_seconds": wall, "completed_optimizer_steps": committed,
            "optimizer_presentations": presentations, "consumed_batches": batches,
            "latest_checkpoint": latest, "last_status": last_status, "active": active,
            "unresolved_work": unresolved or not checkpoint_matches,
            "remaining_wall_seconds": 0.0 if active or unresolved else remaining,
            "can_resume": can_resume, "parent_sha256": parent,
        }

    def begin_attempt(self, *, started=None):
        if self._attempt is not None and not self._attempt.closed:
            raise ValueError("this ledger already owns an active attempt")
        prior = self.inspect()
        # The measured supervisor writes outside attempts/ so it can retain
        # startup and shutdown uncertainty. Known failures also block callers
        # using this lower-level API; only the current pending start is allowed.
        if any(self.root.glob("supervisor-overrun-*.json")):
            raise ValueError("whole-process overrun blocks further attempts")
        for terminal in self.root.glob("supervisor-result-*.json"):
            if _read(terminal).get("verified_within_charged_wall") is not True:
                raise ValueError("uncertain supervisor result blocks further attempts")
        for start_path in self.root.glob("supervisor-start-*.json"):
            match = re.fullmatch(r"supervisor-start-([0-9]{4})\.json", start_path.name)
            if match is None:
                raise ValueError("malformed supervisor attempt identity")
            if int(match.group(1)) <= prior["attempt_count"]:
                terminal = start_path.with_name(start_path.name.replace("start", "result", 1))
                if not terminal.is_file() or _read(terminal).get("verified_within_charged_wall") is not True:
                    raise ValueError("missing verified supervisor outcome blocks continuation")
        if not prior["can_resume"]:
            raise ValueError("allocation is complete, exhausted or has unreconciled work")
        begin = time.monotonic() if started is None else started
        now = time.monotonic()
        if type(begin) not in (int, float) or not math.isfinite(begin) or not 0 <= begin <= now:
            raise ValueError("attempt start must be a past monotonic timestamp")
        number = prior["attempt_count"] + 1
        folder = self.root / "attempts" / f"attempt-{number:04d}"
        folder.mkdir(parents=True, exist_ok=False)
        start = _publish(folder / "start.json", {
            "schema": SCHEMA, "attempt": number, "allocation_sha256": self.header["sha256"],
            "parent_sha256": prior["parent_sha256"], "prior_charged": prior["charged"],
            "prior_wall_seconds": prior["charged_wall_seconds"],
            "prior_checkpoint": prior["latest_checkpoint"], "created_utc": utc_now(),
        })
        self._attempt = IKAttempt(self, number, folder, start, prior, float(begin))
        return self._attempt

    def resume(self, trainer, sampler):
        """Restore into fresh components before this attempt admits any operation."""
        attempt = self._attempt
        if attempt is None or attempt.closed or attempt.sequence != 0:
            raise ValueError("resume requires the newly opened attempt before any operation")
        latest = attempt.prior["latest_checkpoint"]
        if latest is None:
            return {}
        state = latest["state"]
        if (state["trainer_config_sha256"] != trainer.config_sha256
                or state["trainer_normalization_sha256"] != trainer.normalization_sha256
                or state["loss_weights_sha256"] != trainer.weights.sha256):
            raise ValueError("checkpoint trainer, normalization or loss differs")
        # Validate the sampler cursor before mutating model/optimizer/RNG state.
        sampler.load_state_dict(_copy(state["sampler"]))
        payload = load_checkpoint(
            _checkpoint_file(self.root, latest["path"]),
            expected_sha256=latest["reference"]["sha256"], expected_metadata=self.identity)
        restored, limitations = restore_checkpoint(payload, trainer.model, trainer.optimizer)
        if restored != state or limitations:
            raise ValueError("checkpoint restoration is incomplete or changes devices")
        trainer.attempted_steps = state["trainer"]["attempted_steps"]
        trainer.committed_steps = state["trainer"]["committed_steps"]
        trainer.sample_presentations = state["trainer"]["sample_presentations"]
        trainer.failed = False
        return _copy(state["extra_state"])


class IKAttempt:
    """Single writer with durable reservations before each requested operation."""

    def __init__(self, ledger, number, folder, start, prior, started):
        self.ledger, self.number, self.folder = ledger, number, folder
        self.start, self.prior, self.started = start, prior, started
        self.sequence, self.previous = 0, start["sha256"]
        self.charged = _copy(prior["charged"])
        self.committed = prior["completed_optimizer_steps"]
        self.optimizer_presentations = prior["optimizer_presentations"]
        self.batches = prior["consumed_batches"]
        self.latest = prior["latest_checkpoint"]
        self.pending = None
        self.closed = False

    @property
    def remaining_wall_seconds(self):
        return max(0.0, self.ledger.allocation["max_wall_seconds"]
                   - self.prior["charged_wall_seconds"] - (time.monotonic() - self.started))

    def _event(self, kind, payload):
        if self.closed:
            raise ValueError("training attempt is already sealed")
        sequence = self.sequence + 1
        event = _publish(self.folder / f"event-{sequence:06d}.json", {
            "sequence": sequence, "previous_sha256": self.previous, "kind": kind, "payload": payload})
        self.sequence, self.previous = sequence, event["sha256"]

    def reserve(self, kind, *, samples=0, calls=1, call_id):
        if self.closed or self.pending is not None:
            raise ValueError("training attempt is sealed or has a pending reservation")
        if self.remaining_wall_seconds <= 5.0:
            raise ValueError("whole-process wall allocation lacks its finalization reserve")
        if type(call_id) is not str or not 1 <= len(call_id) <= 200:
            raise ValueError("training call identity is required")
        cost = _cost(kind, samples, calls)
        charged = _add(self.charged, cost, self.ledger.allocation)
        reservation = {
            "id": f"{self.number:04d}-{self.sequence + 1:06d}", "kind": kind,
            "samples": samples, "calls": calls, "call_id": call_id, "cost": cost}
        self._event("reserve", reservation)
        self.charged = charged
        self.pending = reservation
        if kind == "optimizer":
            self.optimizer_presentations += samples
        if kind in {"optimizer", "presentation_only"}:
            self.batches += 1
        return _copy(reservation)

    def complete(self, reservation, receipt):
        if reservation != self.pending or self.pending is None:
            raise ValueError("completion does not match the active reservation")
        if not isinstance(receipt, Mapping):
            raise ValueError("completion receipt must be a mapping")
        receipt = _copy(dict(receipt))
        if self.pending["kind"] == "checkpoint":
            _verify_checkpoint(self.ledger.root, receipt, self.ledger.identity)
        self._event("complete", {"reservation_id": self.pending["id"], "receipt": receipt})
        if self.pending["kind"] == "optimizer":
            self.committed += 1
        if self.pending["kind"] == "checkpoint":
            self.latest = receipt
        self.pending = None

    def checkpoint(self, trainer, sampler_state, *, extra_state=None):
        if self.pending is not None or trainer.failed:
            raise ValueError("checkpoint requires reconciled work and a healthy trainer")
        sampler_state = _copy(sampler_state)
        counters = {
            "attempted_steps": trainer.attempted_steps, "committed_steps": trainer.committed_steps,
            "sample_presentations": trainer.sample_presentations}
        if (any(type(v) is not int or v < 0 for v in counters.values())
                or counters["attempted_steps"] != self.charged["optimizer_steps"]
                or counters["committed_steps"] != self.committed
                or counters["sample_presentations"] != self.optimizer_presentations
                or sampler_state.get("cursor") != self.batches
                or sampler_state.get("sample_presentations") != self.charged["sample_presentations"]):
            raise ValueError("trainer or sampler counters differ from durable work")
        extra_state = {} if extra_state is None else _copy(extra_state)
        if not isinstance(extra_state, dict):
            raise ValueError("extra training state must be a JSON object")
        reservation = self.reserve("checkpoint", call_id=f"checkpoint-{self.charged['checkpoints'] + 1:06d}")
        state = {
            "schema": "nca-ik-training-state/v1", "boundary": "optimizer_step",
            "reservation_id": reservation["id"], "charged": _copy(self.charged),
            "completed_optimizer_steps": self.committed,
            "optimizer_presentations": self.optimizer_presentations, "consumed_batches": self.batches,
            "trainer": counters, "trainer_config_sha256": trainer.config_sha256,
            "trainer_normalization_sha256": trainer.normalization_sha256,
            "loss_weights_sha256": trainer.weights.sha256, "sampler": sampler_state,
            "extra_state": extra_state,
        }
        path = self.folder / "checkpoints" / f"checkpoint-{self.charged['checkpoints']:06d}.pt"
        reference = write_checkpoint(
            path, trainer.model, trainer.optimizer, metadata=self.ledger.identity,
            training_state=state, max_bytes=MAX_BYTES)
        receipt = {"path": path.relative_to(self.ledger.root).as_posix(),
                   "reference": reference, "state": state}
        self.complete(reservation, receipt)
        return _copy(receipt)

    def finish(self, status, *, reason, publication_reserve_seconds=5.0):
        if self.closed or status not in _STATUSES or type(reason) is not str or not reason:
            raise ValueError("valid unsealed training outcome and reason required")
        if status in {"paused", "completed"} and self.pending is not None:
            raise ValueError("successful terminal status cannot hide a pending operation")
        if (type(publication_reserve_seconds) not in (int, float)
                or not math.isfinite(publication_reserve_seconds)
                or not 5.0 <= publication_reserve_seconds <= 60.0):
            raise ValueError("publication reserve must be finite and between five and sixty seconds")
        inventory = _inventory(self.folder)
        observed = time.monotonic() - self.started
        remaining = max(0.0, self.ledger.allocation["max_wall_seconds"]
                        - self.prior["charged_wall_seconds"] - observed)
        # Charge publication/readback before performing it. Missing or late
        # finalization never grants a cheaper continuation.
        elapsed = observed + min(publication_reserve_seconds, remaining)
        if self.prior["charged_wall_seconds"] + elapsed >= self.ledger.allocation["max_wall_seconds"]:
            status, reason = "capped", "whole_process_wall_allocation_exhausted"
        finish = _publish(self.folder / "finish.json", {
            "attempt": self.number, "start_sha256": self.start["sha256"],
            "last_event_sha256": self.previous, "event_count": self.sequence,
            "status": status, "reason": reason, "elapsed_seconds": elapsed,
            "charged": self.charged, "completed_optimizer_steps": self.committed,
            "optimizer_presentations": self.optimizer_presentations, "consumed_batches": self.batches,
            "checkpoint": self.latest, "pending_reservation": self.pending, "inventory": inventory})
        self.closed = True
        self.ledger.inspect()  # Read back all bytes before publishing the seal.
        if time.monotonic() - self.started > elapsed:
            raise ValueError("training finalization exceeded its charged reserve; attempt remains unsealed")
        immutable_json(self.folder / "sealed.json", {"finish_sha256": finish["sha256"]})
        result = self.ledger.inspect()
        if time.monotonic() - self.started > elapsed:
            # A late readback is explicitly retained and invalidates the sealed
            # inventory; the supervisor also checks whole-process elapsed time.
            immutable_json(self.folder / "finalization-overrun.json", {"reason": "charged_reserve_exceeded"})
            raise ValueError("training finalization exceeded its charged reserve")
        return result


# Copyright 2026 techrote. All rights reserved.
