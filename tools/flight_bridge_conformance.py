#!/usr/bin/env python3
"""Finite 4LY-004 actual-host bridge qualification; never trains or opens finals."""

from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nca.adapters.flight import (  # noqa: E402
    FlightSession,
    UnknownConsumptionError,
)
from nca.adapters.flight_transport import FlightGodotTransport  # noqa: E402

PLAN = {
    "schema": "nca-4ly-flight-conformance-plan/v1",
    "host_starts": 1,
    "resets": 4,
    "host_calls": 14,
    "step_calls": 7,
    "reserved_physical_ticks": 20,
    "fresh_decisions": 4,
    "null_continuations": 3,
    "cancel_calls": 1,
    "takeover_calls": 1,
    "shutdown_calls": 1,
    "cleanup_operations": 1,
    "teacher_capture": False,
    "training": False,
    "cuda": False,
    "final_access": False,
    "automatic_replay": False,
    "sequence": [
        "reset-a", "full-6", "fresh-partial-3", "continue-3",
        "continue-boundary-5", "lease-expiry-1",
        "reset-b", "fresh-partial-1", "cancel",
        "reset-c", "fresh-partial-1", "takeover",
        "reset-d", "lost-close-reply", "post-exit-cleanup",
    ],
}


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


class ConformanceAccounting:
    """Fsynced qualification ledger, structurally separate from training admission."""

    MAXIMUM = {
        "host_starts": 1,
        "resets": 4,
        "host_calls": 14,
        "physical_ticks": 20,
        "decisions": 4,
        "startup_operations": 1,
        "cleanup_operations": 1,
    }

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("xb")
        self.sequence = 0
        self.next_id = 0
        self.charged = {name: 0 for name in self.MAXIMUM}
        self.pending: dict[str, dict[str, int]] = {}
        self.unknown = False
        self.aborted = False
        self.fsync_seconds = 0.0

    def _append(self, kind: str, payload: dict[str, Any]) -> None:
        self.sequence += 1
        row = canonical({
            "schema": "nca-4ly-flight-conformance-accounting/v1",
            "sequence": self.sequence,
            "kind": kind,
            "payload": payload,
        })
        started = time.perf_counter()
        self.stream.write(row)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.fsync_seconds += time.perf_counter() - started

    def _reserve(self, increments: dict[str, int], scope: str) -> str:
        for name, amount in increments.items():
            if (
                name not in self.MAXIMUM
                or type(amount) is not int
                or amount < 0
                or self.charged[name] + amount > self.MAXIMUM[name]
            ):
                raise RuntimeError(
                    f"conformance plan ceiling exceeded: {name}")
        self.next_id += 1
        reservation = f"q{self.next_id}"
        for name, amount in increments.items():
            self.charged[name] += amount
        self.pending[reservation] = dict(increments)
        self._append("reserve", {
            "reservation_id": reservation,
            "scope": scope,
            "increments": increments,
        })
        return reservation

    def reserve_host_start(self) -> str:
        return self._reserve({
            "host_starts": 1, "startup_operations": 1,
        }, "host_start")

    def reserve_reset(self, **_fields) -> str:
        return self._reserve({
            "resets": 1, "host_calls": 1,
        }, "reset")

    def reserve_step(
        self, *, maximum_ticks: int, capture_labels: int = 0,
        decision: bool = True, **_fields,
    ) -> str:
        if capture_labels:
            raise RuntimeError(
                "conformance may not reserve capture labels")
        increments = {
            "host_calls": 1,
            "physical_ticks": maximum_ticks,
        }
        if decision:
            increments["decisions"] = 1
        return self._reserve(
            increments, "step" if decision else "null_continuation")

    def reserve_host_call(self, *, scope: str, **_fields) -> str:
        return self._reserve({"host_calls": 1}, scope)

    def reserve_cleanup(self, *, host_call: bool = False) -> str:
        increments = {"cleanup_operations": 1}
        if host_call:
            increments["host_calls"] = 1
        return self._reserve(increments, "cleanup")

    def complete(
        self, reservation: str, *, confirmed_physical_ticks: int = 0,
    ) -> None:
        self.pending.pop(reservation)
        self._append("complete", {
            "reservation_id": reservation,
            "confirmed_physical_ticks": confirmed_physical_ticks,
        })

    def fail(
        self, reservation: str, reason: str,
        *, confirmed_physical_ticks: int = 0,
    ) -> None:
        self.pending.pop(reservation)
        self._append("failure", {
            "reservation_id": reservation,
            "reason": reason,
            "confirmed_physical_ticks": confirmed_physical_ticks,
        })

    def mark_unknown(
        self, reservation: str, reason: str,
        *, confirmed_physical_ticks: int = 0,
    ) -> None:
        self.pending.pop(reservation, None)
        self.unknown = True
        self._append("unknown", {
            "reservation_id": reservation,
            "reason": reason,
            "confirmed_physical_ticks": confirmed_physical_ticks,
        })

    def abort(self, reason: str) -> None:
        self.aborted = True
        self._append("attempt_failure", {"reason": reason})

    def close(self) -> None:
        if self.stream.closed:
            return
        for reservation in list(self.pending):
            self.mark_unknown(
                reservation,
                "qualification_closed_with_pending_reservation",
            )
        self._append("close", {
            "unknown_consumption": self.unknown,
            "aborted": self.aborted,
            "charged": self.charged,
        })
        self.stream.close()


def process_tree_rss(root: int) -> int:
    pending = [root]
    seen: set[int] = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
            for child_file in Path(f"/proc/{pid}/task").glob("*/children"):
                try:
                    pending.extend(
                        int(value)
                        for value in child_file.read_text().split())
                except (FileNotFoundError, ProcessLookupError, ValueError):
                    pass
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return total


class MemorySampler:
    def __init__(self):
        self.stop_event = threading.Event()
        self.peak = 0
        self.samples = 0
        self.thread = threading.Thread(
            target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.peak = max(
                self.peak, process_tree_rss(os.getpid()))
            self.samples += 1
            self.stop_event.wait(0.01)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.peak = max(
            self.peak, process_tree_rss(os.getpid()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_sequence(
    checkout: Path, godot: Path, output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    ledger = ConformanceAccounting(output / "accounting.jsonl")
    events: list[dict[str, Any]] = []
    latencies: list[dict[str, Any]] = []
    confirmed_ticks = 0
    receipt_bytes = 0
    sampler = MemorySampler()
    sampler.start()
    transport: FlightGodotTransport | None = None

    def timed(name: str, function):
        started = time.perf_counter_ns()
        value = function()
        latencies.append({
            "operation": name,
            "seconds": (time.perf_counter_ns() - started) / 1e9,
        })
        return value

    def record_step(name: str, reply: dict[str, Any]) -> None:
        nonlocal confirmed_ticks, receipt_bytes
        receipts = reply.get("owner_receipts")
        ticks = reply["transition"]["ticks_consumed"]
        if type(receipts) is not list or len(receipts) != ticks:
            raise RuntimeError(
                "owner receipt count differs from confirmed ticks")
        confirmed_ticks += ticks
        receipt_bytes += len(canonical(receipts))
        events.append({
            "operation": name,
            "reply": deepcopy(reply),
        })

    try:
        startup_started = time.perf_counter()
        transport = FlightGodotTransport(
            checkout, godot, accounting=ledger, timeout=60.0)
        startup_seconds = time.perf_counter() - startup_started
        session = FlightSession(transport)
        identity = deepcopy(transport.identity)

        observation = timed(
            "reset-a",
            lambda: session.reset(
                scenario="horizontal_reach",
                seed=101,
                split="training",
            ),
        )
        events.append({
            "operation": "reset-a",
            "observation": observation,
            "exchange": deepcopy(session.last_exchange),
        })

        record_step("full-6", timed(
            "full-6",
            lambda: session.step(
                [0.25, 0.0],
                observation_id=session.observation["observation_id"],
                repeat_ticks=6,
            ),
        ))
        record_step("fresh-partial-3", timed(
            "fresh-partial-3",
            lambda: session.step(
                [0.25, 0.0],
                observation_id=session.observation["observation_id"],
                repeat_ticks=3,
            ),
        ))
        record_step("continue-3", timed(
            "continue-3",
            lambda: session.continue_step(
                observation_id=session.observation["observation_id"],
                repeat_ticks=3,
            ),
        ))
        record_step("continue-boundary-5", timed(
            "continue-boundary-5",
            lambda: session.continue_step(
                observation_id=session.observation["observation_id"],
                repeat_ticks=5,
            ),
        ))
        generation_before_expiry = (
            session.observation["control_generation"])
        expiry_reply = timed(
            "lease-expiry-1",
            lambda: session.continue_step(
                observation_id=session.observation["observation_id"],
                repeat_ticks=1,
            ),
        )
        record_step("lease-expiry-1", expiry_reply)
        expiry = expiry_reply["transition"]["observation"]
        if not (
            expiry_reply["transition"]["physical_ticks_total"] == 18
            and expiry["control_generation"]
                == generation_before_expiry + 1
            and expiry["command_live"] is False
            and expiry["lease_remaining_ticks"] == 0
            and expiry["actuator_mode"] == "passive_no_lease"
            and expiry["assistance_valid"] is False
        ):
            raise RuntimeError(
                "real lease-expiry receipt differs from accepted #249")

        observation = timed(
            "reset-b",
            lambda: session.reset(
                scenario="horizontal_reach",
                seed=101,
                split="training",
            ),
        )
        events.append({
            "operation": "reset-b",
            "observation": observation,
            "exchange": deepcopy(session.last_exchange),
        })
        record_step("cancel-pre-step", timed(
            "cancel-pre-step",
            lambda: session.step(
                [0.0, 0.0],
                observation_id=session.observation["observation_id"],
                repeat_ticks=1,
            ),
        ))
        authority = timed("cancel", session.cancel)
        events.append({
            "operation": "cancel",
            "authority": authority,
            "exchange": deepcopy(session.last_exchange),
        })

        observation = timed(
            "reset-c",
            lambda: session.reset(
                scenario="horizontal_reach",
                seed=101,
                split="training",
            ),
        )
        events.append({
            "operation": "reset-c",
            "observation": observation,
            "exchange": deepcopy(session.last_exchange),
        })
        record_step("takeover-pre-step", timed(
            "takeover-pre-step",
            lambda: session.step(
                [0.0, 0.0],
                observation_id=session.observation["observation_id"],
                repeat_ticks=1,
            ),
        ))
        authority = timed("takeover", session.takeover)
        events.append({
            "operation": "takeover",
            "authority": authority,
            "exchange": deepcopy(session.last_exchange),
        })

        observation = timed(
            "reset-d",
            lambda: session.reset(
                scenario="horizontal_reach",
                seed=101,
                split="training",
            ),
        )
        final_observation = deepcopy(observation)
        events.append({
            "operation": "reset-d",
            "observation": final_observation,
            "exchange": deepcopy(session.last_exchange),
        })

        transport.arm_reply_loss_for_conformance()
        lost_started = time.perf_counter()
        try:
            session._request({"command": "close"})
            raise RuntimeError(
                "deliberately lost close reply unexpectedly returned")
        except UnknownConsumptionError as exc:
            if exc.reason != "conformance_deliberate_reply_loss":
                raise
            events.append({
                "operation": "lost-close-reply",
                "final_observation_before_close": final_observation,
                "unknown_consumption":
                    deepcopy(session.unknown_consumption),
                "seconds": time.perf_counter() - lost_started,
            })

        shutdown_started = time.perf_counter()
        graceful_exit_code = transport.wait_for_process_exit(
            timeout=5.0)
        shutdown_seconds = time.perf_counter() - shutdown_started
        cleanup_reservation = ledger.reserve_cleanup(
            host_call=False)
        cleanup_started = time.perf_counter()
        transport.close()
        cleanup_seconds = time.perf_counter() - cleanup_started
        ledger.complete(cleanup_reservation)
        transport = None
        sampler.stop()

        if graceful_exit_code != 0:
            raise RuntimeError(
                "close was dispatched but host did not exit cleanly")
        if confirmed_ticks != 20:
            raise RuntimeError(
                f"expected 20 confirmed physical ticks, got {confirmed_ticks}")
        if not ledger.unknown:
            raise RuntimeError(
                "lost close reply did not preserve unknown consumption")
        if ledger.aborted:
            raise RuntimeError(
                "post-receipt semantic validation failed")

        ledger.close()
        accounting_bytes = ledger.path.stat().st_size
        step_latencies = [
            item["seconds"] for item in latencies
            if item["operation"] in {
                "full-6", "fresh-partial-3", "continue-3",
                "continue-boundary-5", "lease-expiry-1",
                "cancel-pre-step", "takeover-pre-step",
            }
        ]
        reset_latencies = [
            item["seconds"] for item in latencies
            if item["operation"].startswith("reset-")
        ]
        serializations = [
            event["reply"]["owner_timing"][
                "observation_serialization_microseconds"]
            for event in events
            if "reply" in event
        ]
        event_payload_bytes = len(canonical(events))
        measurements = {
            "startup_seconds": startup_seconds,
            "shutdown_seconds": shutdown_seconds,
            "cleanup_seconds": cleanup_seconds,
            "call_latencies": latencies,
            "reset_latency_max_seconds": max(reset_latencies),
            "reset_latency_median_seconds":
                statistics.median(reset_latencies),
            "step_latency_max_seconds": max(step_latencies),
            "step_latency_median_seconds":
                statistics.median(step_latencies),
            "confirmed_physical_ticks": confirmed_ticks,
            "confirmed_step_seconds": sum(step_latencies),
            "confirmed_ticks_per_second":
                confirmed_ticks / sum(step_latencies),
            "owner_serialization_microseconds_max":
                max(serializations),
            "process_tree_peak_resident_bytes": sampler.peak,
            "memory_samples": sampler.samples,
            "owner_receipt_bytes": receipt_bytes,
            "event_payload_bytes": event_payload_bytes,
            "accounting_journal_bytes": accounting_bytes,
            "accounting_fsync_seconds": ledger.fsync_seconds,
            "durable_accounting_bytes_per_second":
                accounting_bytes / ledger.fsync_seconds
                if ledger.fsync_seconds > 0 else None,
        }
        evidence = {
            "schema": "nca-4ly-flight-actual-host-conformance/v1",
            "plan": PLAN,
            "identity": identity,
            "events": events,
            "measurements": measurements,
            "accounting": {
                "charged": deepcopy(ledger.charged),
                "unknown_consumption": ledger.unknown,
                "pending": sorted(ledger.pending),
            },
            "prohibitions": {
                "teacher_capture": False,
                "training": False,
                "cuda": False,
                "training_allocation": False,
                "teacher_final_access": False,
                "learner_final_access": False,
                "ik_final_access": False,
            },
        }
        evidence_path = output / "evidence.json"
        evidence_path.write_bytes(canonical(evidence))
        summary = {
            "schema": evidence["schema"],
            "status":
                "physical_sequence_complete_pending_independent_readback",
            "evidence_sha256": sha256_file(evidence_path),
            "evidence_file_bytes": evidence_path.stat().st_size,
            **measurements,
        }
        (output / "run-summary.json").write_bytes(
            canonical(summary))
        print(
            "NCA_4LY_CONFORMANCE_RESULT "
            + json.dumps(summary, sort_keys=True))
        return summary
    finally:
        if sampler.thread.is_alive():
            sampler.stop()
        if transport is not None:
            transport.close()
        ledger.close()


def verify(output: Path) -> dict[str, Any]:
    evidence_path = output / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    if (
        evidence.get("schema")
            != "nca-4ly-flight-actual-host-conformance/v1"
        or evidence.get("plan") != PLAN
    ):
        raise RuntimeError(
            "evidence differs from reviewed finite invocation")
    if any(evidence["prohibitions"].values()):
        raise RuntimeError(
            "conformance evidence crossed a prohibited boundary")

    confirmed = 0
    expiry_seen = False
    tick_by_run: dict[str, int] = {}
    for event in evidence["events"]:
        reply = event.get("reply")
        if reply is None:
            continue
        transition = reply["transition"]
        ticks = transition["ticks_consumed"]
        receipts = reply["owner_receipts"]
        if len(receipts) != ticks:
            raise RuntimeError(
                "independent owner-receipt cardinality mismatch")
        run_id = transition["observation"]["run_id"]
        if receipts:
            previous = tick_by_run.get(
                run_id, receipts[0]["tick"] - 1)
            for receipt in receipts:
                if (
                    receipt["tick"] != previous + 1
                    or type(receipt.get("physical_receipt")) is not dict
                ):
                    raise RuntimeError(
                        "owner physical receipt discontinuity")
                previous = receipt["tick"]
            tick_by_run[run_id] = previous
            if previous != transition["physical_ticks_total"]:
                raise RuntimeError(
                    "owner receipt total differs from transition")
        confirmed += ticks

        if event["operation"] == "lease-expiry-1":
            expiry_seen = True
            authority = receipts[-1]["authority"]
            observation = transition["observation"]
            if not (
                authority["command_live"] is False
                and authority["lease_remaining_ticks"] == 0
                and authority["actuator_mode"] == "passive_no_lease"
                and authority["fallback_reason"] == "expired"
                and observation["assistance_valid"] is False
            ):
                raise RuntimeError(
                    "independent lease-expiry readback failed")

    lost = [
        event for event in evidence["events"]
        if event["operation"] == "lost-close-reply"
    ]
    if (
        confirmed != 20
        or not expiry_seen
        or len(lost) != 1
        or lost[0]["unknown_consumption"]["state"] != "unknown"
        or evidence["accounting"]["unknown_consumption"] is not True
        or evidence["accounting"]["pending"]
    ):
        raise RuntimeError(
            "independent conformance readback failed")

    result = {
        "schema": "nca-4ly-flight-conformance-readback/v1",
        "status": "PASS",
        "evidence_sha256": sha256_file(evidence_path),
        "evidence_file_bytes": evidence_path.stat().st_size,
        "confirmed_physical_ticks": confirmed,
        "unknown_close_consumption_preserved": True,
        "lease_expiry_verified": True,
    }
    (output / "readback.json").write_bytes(canonical(result))
    print(
        "NCA_4LY_CONFORMANCE_READBACK "
        + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(
        dest="command", required=True)
    commands.add_parser("plan")
    run_parser = commands.add_parser("run")
    run_parser.add_argument(
        "--checkout", type=Path, required=True)
    run_parser.add_argument(
        "--godot", type=Path, required=True)
    run_parser.add_argument(
        "--output", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument(
        "--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(PLAN, indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        run_sequence(
            args.checkout, args.godot, args.output)
        return 0
    verify(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
