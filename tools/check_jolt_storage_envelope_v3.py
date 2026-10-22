#!/usr/bin/env python3
"""Fixed synthetic storage conformance; never opens a host/model/allocation.

The complete loose reservation count is streamed once. Journal and gzip checks
use fixed-size batches; projections use the maximum observed block time times a
predeclared factor four, not historical means. These observations do not assert
an admitted complete pipeline or a physical/native service-time guarantee.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nca.contracts import canonical_json, strict_json_loads
from nca.data.capture import source_identity
from nca.evaluation.durable_journal import validate_record_chain
from nca.train.jolt_budget_v3 import reservation_records, _header, _encoded
from nca.train.jolt_execution_v3 import RESERVATION_RECORDS_MAX
from nca.train.jolt_shape_v3 import response_shape, shape_bounds_v3
from nca.train.jolt_storage_v3 import (BoundedVerifiedJournalV3, MAX_JOURNAL_RECORDS,
    MAX_JOURNAL_BYTES, CHUNK_BYTES, file_identity_v3, storage_bounds_v3, episode_byte_limit)

FACTOR = 4
BLOCK = 25000
JOURNAL_BATCH = 256
GZIP_REPEATS = 4


def summary(samples, *, projected_units, units_per_sample):
    maximum = max(samples)
    return {"samples_seconds": samples, "maximum_sample_seconds": maximum,
        "units_per_sample": units_per_sample, "projected_units": projected_units,
        "planning_factor": FACTOR,
        "projected_seconds_at_maximum_times_factor": maximum * projected_units / units_per_sample * FACTOR}


def reservations(folder):
    path = folder / "synthetic-reservations.json"
    costs = {"owner_requests": 1, "owner_step": 1, "decisions": 1, "requested_ticks": 6}
    row = {"index": 0, "elapsed_seconds": 1.2345678901234567e-100, "stage": "teacher",
           "phase": "active", "cleanup": False, "increments": costs}
    count = RESERVATION_RECORDS_MAX
    started = time.perf_counter()
    with path.open("xb") as target:
        target.write(_header(count))
        for index in range(count):
            row["index"] = index
            target.write(_encoded(row) + (b",\n" if index+1<count else b"\n"))
        target.write(b"]}\n")
        target.flush()
        os.fsync(target.fileno())
    write_seconds = time.perf_counter()-started
    samples, digest, counter = [], hashlib.sha256(), 0
    started = boundary = time.perf_counter()
    for record in reservation_records(path):
        # Independent auditor snapshot does this additional canonical walk and
        # digest beyond the strict streaming reader's own shape validation.
        digest.update(canonical_json(record).encode())
        counter += sum(record["increments"].values())
        if (record["index"]+1) % BLOCK == 0:
            now = time.perf_counter()
            samples.append(now-boundary)
            boundary=now
    elapsed = time.perf_counter()-started
    tail = count % BLOCK
    tail_seconds = time.perf_counter()-boundary
    # Scale the partial block to one full block before selecting the maximum.
    if tail:
        samples.append(tail_seconds * BLOCK/tail)
    result = summary(samples, projected_units=count, units_per_sample=BLOCK)
    result.update(count=count, serialized_bytes=path.stat().st_size,
        maximum_row_bytes=len(_encoded(row)), write_and_fsync_seconds=write_seconds,
        actual_full_reader_and_hash_seconds=elapsed, sum_increments=counter,
        canonical_rows_sha256=digest.hexdigest(), synthetic_operation_totals_are_not_an_admitted_ledger=True,
        audit_reader_passes_in_source=3,
        three_audit_pass_projection_seconds=result["projected_seconds_at_maximum_times_factor"]*3)
    path.unlink()
    return result


def journals(folder):
    path = folder / "synthetic-journal.jsonl"
    response = response_shape()
    writer = BoundedVerifiedJournalV3(path)
    writes = []
    for index in range(JOURNAL_BATCH):
        payload = (canonical_json({"event": "owner_response", "serial": index+1, "response": response})+"\n").encode()
        started = time.perf_counter()
        writer.write(payload)
        writes.append(time.perf_counter()-started)
    started = time.perf_counter()
    writer.close()
    close_seconds = time.perf_counter()-started
    reads = []
    for _ in range(4):
        started = time.perf_counter()
        validate_record_chain(path)
        reads.append(time.perf_counter()-started)
    result = {"batch_records": JOURNAL_BATCH, "batch_bytes": path.stat().st_size,
        "record_append": summary(writes, projected_units=MAX_JOURNAL_RECORDS, units_per_sample=1),
        "complete_chain_read": summary(reads, projected_units=MAX_JOURNAL_RECORDS, units_per_sample=JOURNAL_BATCH),
        "writer_close_seconds": close_seconds,
        "reader_projection_uses_every_row_as_largest_native_response": True,
        "writer_durability_receipt": writer.durability_receipt()}
    path.unlink()
    return result


def compressed(folder):
    size = episode_byte_limit(40)
    # An exact ceiling-sized deterministic high-entropy input complements the
    # deeply nested source-shaped journal. It exercises zlib expansion/CPU work,
    # not an admitted observation or a claimed valid EpisodeRecord.
    payload = bytearray()
    index = 0
    while len(payload)<size:
        payload.extend(hashlib.sha256(str(index).encode()).digest())
        index += 1
    raw = bytes(payload[:size])
    del payload
    results={}
    for level in (1,9):
        writes, reads=[],[]
        for repeat in range(GZIP_REPEATS):
            path = folder / f"synthetic-gzip-{level}-{repeat}.gz"
            started = time.perf_counter()
            with path.open("xb") as target:
                with gzip.GzipFile(fileobj=target,mode="wb",filename="",mtime=0,compresslevel=level) as compressed:
                    for offset in range(0,len(raw),CHUNK_BYTES):
                        compressed.write(raw[offset:offset+CHUNK_BYTES])
                target.flush()
                os.fsync(target.fileno())
            writes.append(time.perf_counter()-started)
            started=time.perf_counter()
            digest=hashlib.sha256()
            with gzip.open(path,"rb") as source:
                for block in iter(lambda:source.read(CHUNK_BYTES),b""):
                    digest.update(block)
            reads.append(time.perf_counter()-started)
            assert digest.digest()==hashlib.sha256(raw).digest()
            path.unlink()
        results[str(level)]={"bytes_per_sample":size,
            "compression_and_fsync":summary(writes,projected_units=storage_bounds_v3()["raw_per_family_bytes"],units_per_sample=size),
            "decompression_and_hash":summary(reads,projected_units=storage_bounds_v3()["raw_per_family_bytes"],units_per_sample=size)}
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit("synthetic evidence already exists; choose a fresh output")
    revision, manifest, manifest_sha=source_identity(ROOT)
    fixed_source_files={"tools/check_jolt_storage_envelope_v3.py":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    cpu_lines=Path("/proc/cpuinfo").read_text().splitlines()
    cpu_model=next((line.split(":",1)[1].strip() for line in cpu_lines if line.startswith("model name")),"unavailable")
    evidence={"schema":"nca-jolt-synthetic-storage-envelope/v3","created_utc":datetime.now(timezone.utc).isoformat(),
        "physical_host_started":False,"model_or_checkpoint_executed":False,"experiment_allocation_consumed":False,
        "offline_scope_only":True,"prospective_experiment_admitted":False,
        "source_revision":revision,"source_files":manifest,"source_sha256":manifest_sha,"harness_source_files":fixed_source_files,
        "runtime":{"python":platform.python_version(),"platform":platform.system(),"cpu_model":cpu_model},
        "predeclared_planning_factor":FACTOR,"block_rows":BLOCK,"fixed_journal_batch":JOURNAL_BATCH,
        "fixed_gzip_repeats":GZIP_REPEATS,"source_shape":shape_bounds_v3(),"storage":storage_bounds_v3(),
        "source_manifest_bytes":len(canonical_json(manifest).encode())}
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="jolt-storage-synthetic-",dir=args.output.parent) as folder:
        print("synthetic reservation stream: start",flush=True)
        evidence["reservations"]=reservations(Path(folder))
        print("synthetic source-shaped journal: start",flush=True)
        evidence["journal"]=journals(Path(folder))
        print("synthetic gzip ceilings: start",flush=True)
        evidence["gzip"]=compressed(Path(folder))
    evidence["elapsed_seconds"]=time.perf_counter()-started
    evidence["maximum_resident_set_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    evidence["complete_pipeline_feasibility_established"]=False
    evidence["limitations"]=["No native simulation, per-episode capture/replay or model math was measured.",
        "Maximum measured fixed-block timings with factor4 are prospective planning observations, not universal WCET guarantees.",
        "Three audit reservation passes and journal chain+semantic passes must fit the same600-second interval together with full neural replay.",
        "Per-file fixed metadata and every required final publication/readback need source review; this report cannot independently admit a run."]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as target:
        json.dump(evidence,target,sort_keys=True,indent=2,allow_nan=False)
        target.write("\n")
    print(json.dumps({"output_basename":args.output.name,"elapsed_seconds":evidence["elapsed_seconds"],
        "reservations_three_pass_projected_seconds":evidence["reservations"]["three_audit_pass_projection_seconds"],
        "journal_single_chain_projected_seconds":evidence["journal"]["complete_chain_read"]["projected_seconds_at_maximum_times_factor"],
        "complete_pipeline_feasibility_established":False}),flush=True)


if __name__=="__main__":
    main()
