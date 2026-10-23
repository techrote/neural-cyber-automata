"""One-use CPU custodian for the fixed fresh final suite; never imports Torch.

The public receipt contains geometry and byte identities, never labels. The
archive is read only by the separately admitted, tuning-gated evaluator.
"""
from __future__ import annotations

import time
STARTED = time.monotonic()
import argparse
import ctypes
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import uuid

from nca.contracts import canonical_json, sha256_json, strict_json_loads
from nca.ik.data import iter_tasks, generate_sample, validate_sample
from nca.ik.fresh_final import FreshFinalConfig
from nca.ik.process_limits import memory_limit
from nca.ik.declaration import runtime_expectation, _project
from nca.runs import immutable_json, immutable_bytes, sha256_file, utc_now
from nca.train.ik_experiment import _source, _read

ISOLATION_SHA256 = "08d1b4402cea8c2dd09cd1acafa9f2cca29abbe1ba05ae18050750082356d3db"


def process_peak_bytes():
    """OS process peak working set, independent of Python object estimates."""
    if os.name == "nt":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
        handle = kernel.GetCurrentProcess()
        if not query(handle, ctypes.byref(counters), counters.cb):
            raise ValueError("OS process memory measurement unavailable")
        return counters.PeakWorkingSetSize
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def geometry_identity():
    config = FreshFinalConfig()
    tasks = tuple(iter_tasks(config, "final"))
    identity = {"config_sha256": config.sha256, "task_count": len(tasks),
                "ordered_task_ids_sha256": sha256_json([t.id for t in tasks]),
                "ordered_geometry_sha256": sha256_json([t.to_dict() for t in tasks])}
    if (identity["task_count"] != 2940
            or identity["ordered_task_ids_sha256"] != "aab6fb764202204f397b5a73a0b0f6aa25e7ec5ca4fc841e4dba5891f0984b72"
            or identity["ordered_geometry_sha256"] != "57f7008cbaf6d776ad36e2aeec97019a68a8929e697a2734ea3fde2d7abe64c2"):
        raise ValueError("fixed independently reviewed geometry differs")
    return config, tasks, identity


def declare(output, isolation):
    source = _source(None)
    if sha256_file(isolation) != ISOLATION_SHA256:
        raise ValueError("exact independent whole-original geometry isolation receipt required")
    config, _, geometry = geometry_identity()
    root = Path(output).resolve()
    checkout = _project(None)
    if root == checkout or checkout in root.parents:
        raise ValueError("custodian output must remain outside checkout")
    root.mkdir(parents=True, exist_ok=False)
    declaration = {"schema": "nca-ik-fresh-seal-declaration/v2", "id": uuid.uuid4().hex,
        "created_utc": utc_now(), "source": source, "runtime": runtime_expectation(),
        "config": config.to_dict(), "geometry": geometry,
        "allocation": config.budget.to_dict(), "attempts": 1, "retry_or_resume": False,
        "independent_isolation_file_sha256": ISOLATION_SHA256}
    declaration["sha256"] = sha256_json(declaration)
    immutable_json(root / "declaration.json", declaration)
    immutable_bytes(root / "isolation.json", Path(isolation).read_bytes())
    return declaration


def load(root):
    value = _read(Path(root) / "declaration.json")
    config, _, geometry = geometry_identity()
    if (value.get("schema") != "nca-ik-fresh-seal-declaration/v2"
            or value.get("sha256") != sha256_json({k:v for k,v in value.items() if k != "sha256"})
            or value.get("config") != config.to_dict() or value.get("geometry") != geometry
            or value.get("allocation") != config.budget.to_dict()
            or value.get("attempts") != 1 or value.get("retry_or_resume") is not False
            or value.get("independent_isolation_file_sha256") != ISOLATION_SHA256
            or sha256_file(Path(root) / "isolation.json") != ISOLATION_SHA256
            or value.get("runtime") != runtime_expectation()
            or value.get("source") != _source(None)):
        raise ValueError("fresh seal declaration/source differs")
    return value


def worker(root):
    root = Path(root)
    declaration = load(root)
    config, tasks, _ = geometry_identity()
    memory_method = memory_limit(config.budget.max_memory_mib)
    if not (root / "started.json").is_file():
        raise ValueError("custodian supervisor must own the single attempt")
    rows, iterations, peak = [], 0, 0
    # Exclusive journal creation prevents an independent second teacher pass.
    with (root / "attempts.jsonl").open("xb") as journal:
        for task in tasks:
            if time.monotonic() - STARTED > config.budget.max_wall_seconds - 120:
                raise ValueError("generation stop reserve reached")
            reservation = {"task_id": task.id, "teacher_attempts": 1,
                           "reserved_iterations": config.teacher.max_iterations}
            journal.write((canonical_json(reservation) + "\n").encode())
            journal.flush()
            os.fsync(journal.fileno())
            sample = generate_sample(task, config)
            validate_sample(sample, config)
            iterations += sample.label.teacher.iterations
            rows.append((canonical_json(sample.to_dict()) + "\n").encode())
            peak = max(peak, process_peak_bytes())
            if peak > config.budget.max_memory_mib * 1024**2 or iterations > config.budget.max_teacher_iterations:
                raise ValueError("generation resource allocation exceeded")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for index in range(0, len(rows), 64):
            raw = b"".join(rows[index:index+64])
            info = tarfile.TarInfo(f"final-{index//64:05d}.jsonl")
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    archive_bytes = stream.getvalue()
    digest = immutable_bytes(root / "sealed-final.tar.gz", archive_bytes)
    peak = max(peak, process_peak_bytes())
    if peak > config.budget.max_memory_mib * 1024**2:
        raise ValueError("archive publication exceeded generation memory ceiling")
    receipt = {"schema": "nca-ik-fresh-final-seal/v2", "declaration_sha256": declaration["sha256"],
        "registry_id": declaration["id"], "geometry": declaration["geometry"],
        "independent_isolation_file_sha256": ISOLATION_SHA256,
        "archive_sha256": digest, "archive_bytes": len(archive_bytes), "rows": len(rows),
        "teacher_attempts": len(rows), "teacher_iterations": iterations,
        "reserved_teacher_iterations": len(rows) * config.teacher.max_iterations,
        "attempt_journal_sha256": sha256_file(root / "attempts.jsonl"),
        "peak_process_bytes": peak, "memory_limit_method": memory_method,
        "hard_memory_limit_bytes": config.budget.max_memory_mib * 1024**2,
        "worker_seconds": time.monotonic() - STARTED,
        "final_state": "sealed", "candidate_conditioned_filtering": False}
    receipt["sha256"] = sha256_json(receipt)
    immutable_json(root / "seal.json", receipt)
    return receipt


def supervise(root):
    root = Path(root)
    declaration = load(root)
    immutable_json(root / "started.json", {"declaration_sha256": declaration["sha256"],
        "attempt": 1, "retry_or_resume": False, "started_utc": utc_now()})
    with (root / "worker.log").open("xb") as log:
        process = subprocess.Popen([sys.executable, "-m", __spec__.name, "_worker", "--root", str(root)],
                                   stdout=log, stderr=log)
    timeout = False
    try:
        process.wait(timeout=max(.001, 895 - (time.monotonic() - STARTED)))
    except subprocess.TimeoutExpired:
        timeout = True
        process.kill()
        process.wait(timeout=5)
    elapsed = time.monotonic() - STARTED
    seal = _read(root / "seal.json") if (root / "seal.json").exists() else None
    complete = (not timeout and process.returncode == 0 and elapsed < 900 and seal is not None
                and seal["peak_process_bytes"] <= 512 * 1024**2
                and sha256_file(root / "sealed-final.tar.gz") == seal["archive_sha256"])
    receipt = {"schema": "nca-ik-fresh-seal-supervisor/v2", "declaration_sha256": declaration["sha256"],
        "returncode": process.returncode, "timed_out": timeout, "whole_process_seconds": elapsed,
        "complete": complete, "seal_sha256": None if seal is None else seal["sha256"],
        "retry_or_resume": False}
    immutable_json(root / "supervisor.json", receipt)
    if time.monotonic() - STARTED >= 900:
        immutable_json(root / "supervisor-overrun.json", {
            "whole_process_seconds_through_publication": time.monotonic() - STARTED,
            "complete": False, "retry_or_resume": False})
        raise ValueError("custodian publication exceeded allocation; preserve and stop")
    if not complete:
        raise ValueError("custodian failed; preserve consumed generation allocation, no retry")
    return receipt


def verify_seal(root):
    """Metadata and compressed-byte authentication only; never read final rows."""
    root = Path(root)
    if (root / "supervisor-overrun.json").exists():
        raise ValueError("custodian publication overrun blocks sealing admission")
    declaration, seal, supervisor = load(root), _read(root / "seal.json"), _read(root / "supervisor.json")
    if (seal.get("sha256") != sha256_json({k:v for k,v in seal.items() if k != "sha256"})
            or seal.get("declaration_sha256") != declaration["sha256"]
            or seal.get("geometry") != declaration["geometry"] or seal.get("rows") != 2940
            or seal.get("independent_isolation_file_sha256") != ISOLATION_SHA256
            or seal.get("teacher_attempts") != 2940 or seal.get("reserved_teacher_iterations") != 188160
            or not 0 <= seal.get("teacher_iterations", -1) <= 188160
            or seal.get("peak_process_bytes", float("inf")) > 512*1024**2
            or seal.get("hard_memory_limit_bytes") != 512*1024**2
            or seal.get("memory_limit_method") not in ("RLIMIT_AS", "Windows Job Object process committed-memory limit")
            or seal.get("final_state") != "sealed" or (root / "opened.json").exists()
            or supervisor.get("complete") is not True or supervisor.get("returncode") != 0
            or supervisor.get("timed_out") is not False or supervisor.get("whole_process_seconds", 900) >= 900
            or supervisor.get("declaration_sha256") != declaration["sha256"]
            or supervisor.get("seal_sha256") != seal["sha256"]
            or sha256_file(root / "attempts.jsonl") != seal["attempt_journal_sha256"]
            or (root / "sealed-final.tar.gz").stat().st_size != seal["archive_bytes"]
            or sha256_file(root / "sealed-final.tar.gz") != seal["archive_sha256"]):
        raise ValueError("fresh suite seal/allocation/supervisor differs")
    return seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "generate", "inspect", "_worker"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--isolation")
    args = parser.parse_args()
    result = declare(args.root, args.isolation) if args.command == "declare" else {
        "generate": supervise, "inspect": verify_seal, "_worker": worker}[args.command](args.root)
    print(canonical_json(result))


if __name__ == "__main__":
    main()

# Copyright 2026 techrote. All rights reserved.
