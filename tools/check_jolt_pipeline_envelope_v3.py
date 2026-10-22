#!/usr/bin/env python3
"""Fixed synthetic, typed-event storage planning for LEARN-JOLT-003.

No host, learner or checkpoint is opened.  This complements, rather than
relabels, the original storage/CPU reports.  Its timing specimens do not prove
native-service, episode-semantic replay or full-worker memory feasibility.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import time
ENTRY_MONOTONIC = time.monotonic()

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile

try:
    import resource
except ImportError:  # Windows can inspect/test the helper; measurement is Linux.
    resource = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nca.adapters.godot import physical_config
from nca.contracts import canonical_json, sha256_json, strict_json_loads
from nca.evaluation.jolt_journal_v3 import journal_identity_v3
from nca.train.jolt_budget_v3 import _encoded, _header, reservation_records
from nca.train.jolt_execution_v3 import COUNTERS, STAGES, RESERVATION_RECORDS_MAX
from nca.train.jolt_shape_v3 import (
    FLOAT, INTEGER, command_shape, observation_shape, response_shape,
)
from nca.train.jolt_storage_v3 import (
    BoundedVerifiedJournalV3, CHUNK_BYTES, MAX_AUTHORED_EPISODES,
    MAX_JOURNAL_RECORDS, MAX_JOURNAL_RECORD_BYTES, _write_episode,
    file_identity_v3, storage_bounds_v3,
)

MAXIMUM_SECONDS = 180
FACTOR = 4
JOURNAL_ROWS_PER_TYPE = 16
READ_REPETITIONS = 4
RESERVATION_SAMPLE_ROWS = 25000
EPISODE_LENGTHS = (1, 40)
EPISODE_REPETITIONS = 3
HASH_SAMPLE_BYTES = 8 * 1024 * 1024


def fixed_design():
    return {"maximum_sampling_seconds_from_script_entry": MAXIMUM_SECONDS,
        "whole_process_hard_deadline_enforced": False,
        "deadline_scope": "sampling; final environment/report publication and observed exit are outside this guard",
        "planning_factor": FACTOR,
        "journal_rows_per_type": JOURNAL_ROWS_PER_TYPE, "read_repetitions": READ_REPETITIONS,
        "reservation_sample_rows": RESERVATION_SAMPLE_ROWS,
        "episode_lengths": list(EPISODE_LENGTHS), "episode_repetitions": EPISODE_REPETITIONS,
        "hash_sample_bytes": HASH_SAMPLE_BYTES, "adaptive_sampling": False,
        "native_host_or_trained_checkpoint_permitted": False}


def event_counts():
    """Loose, source-derived count maxima; no observed outcome chooses a count."""
    counts = {}
    for operation in ("step", "reset", "profile", "cancel", "close"):
        for direction in ("request", "response"):
            counts[f"owner_{direction}_{operation}"] = COUNTERS["owner_" + operation]
    starts = sum(stage["episodes_started"] for stage in STAGES.values())
    counts.update(decision_proposed=COUNTERS["owner_step"], episode_started=starts,
        episode_finished_authored=MAX_AUTHORED_EPISODES,
        episode_finished_other=starts-MAX_AUTHORED_EPISODES,
        safe_retirement_verified=starts)
    # 36 checkpoint publications; 6 BC + 48 PPO fit boundaries; host start,
    # host started, host terminated, worker closed, dataset shape, final freeze,
    # sealed opening, and at most one retained allocation-stop note.
    counts["fixed_notes"] = COUNTERS["checkpoint_writes"] + 6 + 48 + 8
    if sum(counts.values()) != MAX_JOURNAL_RECORDS:
        raise ValueError("typed event inventory differs from source journal envelope")
    if sum(COUNTERS["owner_" + name] for name in ("step", "reset", "profile", "cancel", "close")) != COUNTERS["owner_requests"]:
        raise ValueError("typed owner inventory differs from source operation caps")
    return counts


def pass_inventory():
    """Independent source walk; partial-prefix sizes conservatively use full caps."""
    return {
        "active": {"journal_appends": 1, "episode_family_writes": 2},
        "closure": {"journal_chain_passes": 1, "reservation_passes": 2},
        "audit": {"journal_chain_passes": 1, "journal_syntax_passes": 1,
                  "reservation_passes": 3, "episode_family_decode_hash_passes": 2},
        "publication": {"reservation_passes": 2, "whole_artifact_hash_passes": 1},
        "verification": {"journal_chain_passes": 1, "reservation_passes": 2,
                         "whole_artifact_hash_passes": 1},
    }


def _metadata_specimen(event, target_bytes):
    """Explicit size surrogate for bounded metadata, not a valid event record.

    Small integers give substantial parser/canonical-walk work per byte.  The
    retained report labels this as a surrogate; structure-dependent semantics
    remain unmeasured.  Leave room for writer-owned chain fields and envelope.
    """
    value = {"event": event, "synthetic_bounded_metadata": []}
    overhead = len(_encoded(value))
    value["synthetic_bounded_metadata"] = [0] * max(0, (target_bytes-overhead-512)//2)
    return value


def templates():
    view = observation_shape()
    config = physical_config("jev", ticks=240, seed=2147483647)
    config["actors"][0]["position"] = [FLOAT, FLOAT]
    profile_file = ROOT / "configs/morphology/profiles-v1.json"
    profiles = strict_json_loads(profile_file.read_bytes(), max_bytes=1024*1024)
    profile = profiles["profiles"]["jev"]
    config_file = ROOT / "configs/learning/jolt-learn-003-v1.json"
    current = strict_json_loads(config_file.read_bytes(), max_bytes=1024*1024)
    if (profiles["host_revision"] != current["host_revision"]
            or sha256_json(profile["definition"]) != current["profile_sha256"]):
        raise ValueError("historical profile shape does not match the frozen prerequisite")
    fields = {"step": {"commands": [command_shape()], "ticks": 6},
        "reset": {"config": config, "actor_id": INTEGER}, "profile": {},
        "cancel": {"run_id": "f"*32, "epoch": INTEGER}, "close": {}}
    result = {f"owner_request_{operation}": {"event": "owner_request", "serial": 38338,
        "stage": "teacher", "operation": operation, "fields": value,
        "elapsed_seconds": 1.2345678901234567e-100} for operation, value in fields.items()}
    reset = response_shape()
    reset["traces"] = []
    responses = {"step": response_shape(), "reset": reset,
        "profile": {"profile": profile["definition"], "id": INTEGER, "protocol": "nca-host/1", "ok": True},
        "cancel": {"cancelled": True, "observation": view, "id": INTEGER, "protocol": "nca-host/1", "ok": True},
        "close": {"closed": True, "id": INTEGER, "protocol": "nca-host/1", "ok": True}}
    result.update({f"owner_response_{operation}": {"event": "owner_response",
        "serial": 38338, "response": value} for operation, value in responses.items()})
    result.update({
        "decision_proposed": _metadata_specimen("decision_proposed", 8192),
        "episode_started": {"event": "episode_started", "id": "x"*128,
            "case_id": "x"*128, "controller_id": "x"*128, "stage": "teacher",
            "case_stream_entry": {"seed": 190301, "index": 2047, "update": 7}},
        "episode_finished_authored": _metadata_specimen("episode_finished", 48*1024+512),
        "episode_finished_other": _metadata_specimen("episode_finished", 8*1024+512),
        "safe_retirement_verified": _metadata_specimen("safe_retirement_verified", 4*1024+512),
        "fixed_notes": _metadata_specimen("synthetic_fixed_note", MAX_JOURNAL_RECORD_BYTES-512),
    })
    if set(result) != set(event_counts()):
        raise ValueError("typed timing specimen inventory is incomplete")
    return result


def projection(samples, units, sample_units=1):
    if (not samples or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                           for value in samples)
            or type(units) is not int or units < 0 or type(sample_units) is not int or sample_units <= 0):
        raise ValueError("finite nonnegative observations and integer source counts are required")
    return {"samples_seconds": samples, "maximum_sample_seconds": max(samples),
        "source_units": units, "sample_units": sample_units, "planning_factor": FACTOR,
        "projected_seconds": max(samples)*units/sample_units*FACTOR}


def _timed(check, function):
    check()
    begin = time.monotonic()
    value = function()
    duration = time.monotonic()-begin
    check()
    return duration, value


def journal_conformance(folder, check):
    counts, result = event_counts(), {}
    for name, record in templates().items():
        print("synthetic typed journal: " + name, flush=True)
        path = folder / (name + ".jsonl")
        writer = BoundedVerifiedJournalV3(path)
        payload = (canonical_json(record)+"\n").encode()
        appends, serializations = [], []
        try:
            for _ in range(JOURNAL_ROWS_PER_TYPE):
                duration, candidate = _timed(check, lambda: (canonical_json(record)+"\n").encode())
                serializations.append(duration)
                if candidate != payload:
                    raise ValueError("synthetic serialization changed")
                duration, _ = _timed(check, lambda: writer.write(payload))
                appends.append(duration)
            close_seconds, _ = _timed(check, writer.close)
        finally:
            if not writer.closed:
                writer.close()
        chains, syntaxes = [], []
        for _ in range(READ_REPETITIONS):
            duration, identity = _timed(check, lambda: journal_identity_v3(path, check=check))
            chains.append(duration)
            if identity["records"] != JOURNAL_ROWS_PER_TYPE:
                raise ValueError("synthetic journal did not retain every row")
            def parse():
                with path.open("rb") as stream:
                    for line in stream:
                        check()
                        strict_json_loads(line, max_bytes=MAX_JOURNAL_RECORD_BYTES)
            duration, _ = _timed(check, parse)
            syntaxes.append(duration)
        with path.open("rb") as stream:
            rowsize = max(len(line) for line in stream)
        result[name] = {"source_count": counts[name], "sample_linked_record_bytes": rowsize,
            "synthetic_size_surrogate": "synthetic_bounded_metadata" in record,
            "serialization": projection(serializations, counts[name]),
            "append_fsync_readback": projection(appends, counts[name]),
            "chain_read": projection(chains, counts[name], JOURNAL_ROWS_PER_TYPE),
            "syntax_read_only": projection(syntaxes, counts[name], JOURNAL_ROWS_PER_TYPE),
            "sample_writer_close_seconds": close_seconds,
            "sample_file": file_identity_v3(path),
            "sample_closed_verified": writer.durability_receipt()["closed_verified"]}
        path.unlink()
    return result


def reservation_conformance(folder, check):
    path = folder / "synthetic-reservations.json"
    count = RESERVATION_SAMPLE_ROWS
    row = {"index": 0, "elapsed_seconds": 1.2345678901234567e-100,
        "stage": "teacher", "phase": "active", "cleanup": False,
        "increments": {"owner_requests": 1, "owner_step": 1, "decisions": 1, "requested_ticks": 6}}
    with path.open("xb") as stream:
        stream.write(_header(count))
        for index in range(count):
            if index % 512 == 0:
                check()
            row["index"] = index
            stream.write(_encoded(row)+(b",\n" if index+1<count else b"\n"))
        stream.write(b"]}\n")
        stream.flush()
        os.fsync(stream.fileno())
    samples = []
    for _ in range(READ_REPETITIONS):
        def read():
            digest = hashlib.sha256()
            for item in reservation_records(path):
                if item["index"] % 512 == 0:
                    check()
                digest.update(canonical_json(item).encode())
                sum(item["increments"].values())
            return digest.hexdigest()
        duration, digest = _timed(check, read)
        samples.append(duration)
    result = projection(samples, RESERVATION_RECORDS_MAX, count)
    result.update(sample_file=file_identity_v3(path), canonical_rows_sha256=digest,
        reservations_are_synthetic_not_a_valid_operation_ledger=True,
        publication_uses_same_conservative_parse_hash_rate=True,
        publication_write_fsync_and_small_file_latency_not_established=True)
    path.unlink()
    return result


def _episode_specimen(length):
    # Source-shaped nested native objects, but deliberately no valid experiment
    # identity, outcomes, policy record or physical semantics are constructed.
    decision = {"native": response_shape(), "before": observation_shape(),
        "after": observation_shape(), "issued": observation_shape(),
        "packet": command_shape(), "per_decision_metadata": [0]*4096}
    return {"synthetic_storage_specimen": True, "fixed_metadata": [0]*16384,
            "decisions": [decision for _ in range(length)]}


def episode_conformance(folder, check):
    results = {}
    family_bytes = storage_bounds_v3()["raw_per_family_bytes"]
    for length in EPISODE_LENGTHS:
        specimen = _episode_specimen(length)
        for level in (1, 9):
            writes, reads, sizes = [], [], []
            for repeat in range(EPISODE_REPETITIONS):
                path = folder / f"synthetic-episode-{length}-{level}-{repeat}.gz"
                duration, identity = _timed(check, lambda: _write_episode(path, specimen,
                    decisions=length, compresslevel=level, check=check))
                writes.append(duration)
                def decode():
                    with gzip.open(path, "rb") as stream:
                        raw = stream.read()
                    value = strict_json_loads(raw, max_bytes=8*1024*1024)
                    if sha256_json(value["episode"]) != value["sha256"]:
                        raise ValueError("synthetic episode hash changed")
                    return len(raw)
                duration, size = _timed(check, decode)
                reads.append(duration)
                sizes.append(size)
                path.unlink()
            if len(set(sizes)) != 1:
                raise ValueError("synthetic episode size changed across fixed trials")
            results[f"decisions_{length}_gzip_{level}"] = {
                "specimen_bytes": sizes[0], "last_compressed_identity": identity,
                "streaming_write_gzip_fsync_hash_and_eof_readback": projection(writes, family_bytes, sizes[0]),
                "decode_and_canonical_hash_only": projection(reads, family_bytes, sizes[0]),
                "valid_episode_or_dataclass_semantic_replay": False}
    return results


def hash_conformance(folder, check):
    path = folder / "synthetic-hash-bytes"
    block = hashlib.sha256(b"jolt-pipeline-envelope-v3-synthetic-only").digest() * (CHUNK_BYTES//32)
    with path.open("xb") as stream:
        for _ in range(HASH_SAMPLE_BYTES//len(block)):
            check()
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
    samples = []
    for _ in range(READ_REPETITIONS):
        duration, identity = _timed(check, lambda: file_identity_v3(path, check=check))
        samples.append(duration)
    path.unlink()
    result = projection(samples, storage_bounds_v3()["total_artifact_bytes"], HASH_SAMPLE_BYTES)
    result.update(sample_identity=identity, filesystem_cache_not_evicted=True,
                  small_file_open_stat_overhead_excluded=True)
    return result


def planning_components(journal, reservations, episodes, hashes):
    """Measured component projections, never a complete admission verdict."""
    appends = sum(row["append_fsync_readback"]["projected_seconds"]
                  + row["serialization"]["projected_seconds"] for row in journal.values())
    chain = sum(row["chain_read"]["projected_seconds"] for row in journal.values())
    syntax = sum(row["syntax_read_only"]["projected_seconds"] for row in journal.values())
    reservation = reservations["projected_seconds"]
    decode = max(row["decode_and_canonical_hash_only"]["projected_seconds"] for row in episodes.values())
    write = {level: max(row["streaming_write_gzip_fsync_hash_and_eof_readback"]["projected_seconds"]
        for key, row in episodes.items() if key.endswith("gzip_"+str(level))) for level in (1, 9)}
    hashed = hashes["projected_seconds"]
    return {
        "active": {"journal_serialization_and_durable_appends": appends,
                   "two_episode_families_write_readback": sum(write.values())},
        "closure": {"journal_complete_chain": chain, "reservation_publication_two_passes": 2*reservation},
        "audit": {"reservations_three_passes": 3*reservation, "journal_complete_chain": chain,
                  "journal_syntax_only": syntax, "two_episode_families_decode_hash_only": 2*decode},
        "publication": {"closed_reservations_two_passes": 2*reservation, "whole_artifact_hash": hashed},
        "verification": {"journal_complete_chain": chain, "two_reservation_prefixes": 2*reservation,
                         "whole_artifact_hash": hashed},
    }


UNRESOLVED = [
    "Native host startup, reset, stepping, transport, cancellation and process memory are unmeasured here.",
    "Episode dataclass construction, history/retirement/metric validation and semantic journal replay are not timed by storage specimens.",
    "Neural fitting, inference, authentication and audit forwards require separate source/runtime-bound CPU evidence.",
    "Full-worker peak memory includes retained PPO transitions, metrics/results, auditor copies, models, native process and allocator overhead; specimen RSS is not that bound.",
    "Complete reservation publication writes/fsync, small-file open/stat latency, source admission and fixed JSON publication are not established by read throughput.",
    "Page-cache behavior and type/shape surrogates make these factor-four observations planning estimates, not universal worst-case execution guarantees.",
    "Final process reaping, stream fsync, receipts and observed outer exit must still fit their original shared-clock phases.",
]


def environment(folder):
    memory = {}
    for name in ("memory.max", "memory.current", "memory.peak", "cpu.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            memory[name] = path.read_text().strip()
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, value, _ = line.split()
                memory[key.rstrip(":")+"_bytes"] = int(value)*1024
    cpuinfo = Path("/proc/cpuinfo")
    cpu_model = "unavailable"
    if cpuinfo.exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    disk = shutil.disk_usage(folder)
    return {"runtime": {"python": platform.python_version(), "system": platform.system(),
        "machine": platform.machine(), "cpu_model": cpu_model},
        "available_disk_bytes": disk.free, "filesystem_total_bytes": disk.total,
        "shared_memory_snapshot_not_attributable_to_this_process": memory,
        "process_peak_rss_bytes": None if resource is None else
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


def source_binding():
    paths = set((ROOT/"src/nca").rglob("*.py")) | set((ROOT/"src/nca").rglob("*.gd"))
    paths |= set((ROOT/"tools").glob("*.py")) | set((ROOT/"tests").rglob("*.py"))
    paths |= set((ROOT/"configs/learning").glob("*.json")) | {ROOT/"pyproject.toml"}
    paths.add(ROOT/"configs/morphology/profiles-v1.json")
    files = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(paths) if not path.name.startswith("jolt-execution-")}
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip())
    return {"source_revision": revision, "source_files": files, "source_sha256": sha256_json(files),
            "source_checkout_dirty": dirty,
            "closure_includes_additional_historical_profile_shape": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise SystemExit("synthetic runtime measurement requires Linux; static tests remain portable")
    if args.output.exists():
        raise SystemExit("synthetic output exists; use a fresh output identity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = ENTRY_MONOTONIC+MAXIMUM_SECONDS
    def check(*_):
        if time.monotonic() >= deadline:
            raise TimeoutError("fixed synthetic wall budget exhausted")
    report = {"schema": "nca-jolt-003-typed-pipeline-conformance/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "fixed_design": fixed_design(),
        "physical_host_started": False, "model_or_checkpoint_executed": False,
        "experiment_allocation_consumed": False, "offline_scope_only": True,
        "prospective_experiment_admitted": False, "complete_pipeline_feasibility_established": False,
        "event_counts": event_counts(), "phase_pass_inventory": pass_inventory(),
        "storage_bounds": storage_bounds_v3(), "environment_before": environment(args.output.parent),
        **source_binding(), "completed": False, "unresolved_terms": list(UNRESOLVED)}
    previous_handler = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError("fixed synthetic wall budget exhausted")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
    try:
        with tempfile.TemporaryDirectory(prefix="jolt-pipeline-synthetic-", dir=args.output.parent) as folder:
            folder = Path(folder)
            report["typed_journal"] = journal_conformance(folder, check)
            report["reservations"] = reservation_conformance(folder, check)
            report["episode_storage"] = episode_conformance(folder, check)
            report["artifact_hash"] = hash_conformance(folder, check)
        report["planning_components_seconds"] = planning_components(report["typed_journal"],
            report["reservations"], report["episode_storage"], report["artifact_hash"])
        report["measured_component_sums_seconds_excluding_unresolved"] = {
            phase: sum(parts.values()) for phase, parts in report["planning_components_seconds"].items()}
        report["requested_audit_reallocation_arithmetic"] = {
            "reservation_seconds": 1200, "journal_seconds": 900,
            "episode_seconds": 900, "neural_seconds": 900,
            "requested_combined_audit_seconds": 3900,
            "unchanged_whole_ceiling_seconds": 7200,
            "unchanged_other_tail_seconds": 600,
            "remaining_active_seconds": 2700,
            "arithmetic_only_not_an_authorized_budget_revision": True}
        check()
        report["completed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "code": getattr(exc, "code", None)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    report["elapsed_before_report_publication_seconds"] = time.monotonic()-ENTRY_MONOTONIC
    report["environment_after"] = environment(args.output.parent)
    with args.output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"completed": report["completed"], "output_basename": args.output.name,
        "elapsed_seconds": time.monotonic()-ENTRY_MONOTONIC,
        "component_sums_excluding_unresolved": report.get("measured_component_sums_seconds_excluding_unresolved"),
        "complete_pipeline_feasibility_established": False,
        "prospective_experiment_admitted": False}), flush=True)
    return 0 if report["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
