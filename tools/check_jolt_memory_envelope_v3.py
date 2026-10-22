#!/usr/bin/env python3
"""LEARN-JOLT-003 independent-of-allocation native/learner memory prepass.

No scheduled Jolt episode, sealed case, physical step or experimental
allocation is opened. A real pinned Godot process is used for capability
handshake *only*, alongside synthetic worst-shape retained worker data.
This is an observation/planning diagnostic, NOT an admitted full-worker peak.
© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
try:
    import resource
except ImportError:  # Windows may inspect the script; measurement is Linux-only.
    resource = None
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

STRESS_SEED = 991003
RESULT_ROWS = 6464
AUTHORED_ROWS = 84
MAX_PPO_DECISIONS_PER_UPDATE = 256
CPU_RUNNER_ADVERTISED_BYTES = 16_000_000_000
SOURCE_SCHEMA = "nca-jolt-003-memory-diagnostic/v1"


def _read_kib_stats(path):
    """Read only well-defined numeric Linux status fields; no PID inference."""
    try:
        rows = Path(path).read_text().splitlines()
    except OSError:
        return {}
    result = {}
    for row in rows:
        if ":" not in row:
            continue
        key, value = row.split(":", 1)
        if key in {"VmRSS", "VmHWM", "RssAnon", "RssFile", "VmSize"}:
            fields = value.strip().split()
            if len(fields) == 2 and fields[0].isdigit() and fields[1] == "kB":
                result[key] = int(fields[0]) * 1024
    return result


def _smaps_rollup(pid):
    """Read PSS when the kernel permits it, without treating failure as zero."""
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                parts = line.split()
                if len(parts) == 3 and parts[1].isdigit() and parts[2] == "kB":
                    return int(parts[1]) * 1024
    except OSError:
        pass
    return None


def _meminfo():
    data = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            number = value.strip().split()
            if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"} and len(number) == 2:
                if number[0].isdigit() and number[1] == "kB":
                    data[name] = int(number[0]) * 1024
    except OSError:
        pass
    return data


def _cgroup():
    # A sample is never elevated to a proved runner/job quota. v1 and v2
    # containers have different mount locations; report missing, not zero.
    paths = {
        "v2_limit": "/sys/fs/cgroup/memory.max",
        "v2_current": "/sys/fs/cgroup/memory.current",
        "v2_peak": "/sys/fs/cgroup/memory.peak",
        "v1_limit": "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "v1_current": "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "v1_peak": "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
    }
    results = {}
    for name, path in paths.items():
        try:
            raw = Path(path).read_text().strip()
            results[name] = int(raw) if raw.isdecimal() else None
        except OSError:
            results[name] = None
    return results


class ProcessMemorySampler:
    """Concurrently sample both worker and original native process identities."""
    def __init__(self, interval=0.05):
        self.interval = interval
        self.child_pid = None
        self.running = False
        self.samples = 0
        self.samples_with_child = 0
        self.maximum_self_rss = None
        self.maximum_child_rss = None
        self.maximum_combined_rss = None
        self.maximum_combined_pss = None
        self.child_exit_seen = False
        self.errors = []
        self.thread = None

    def take(self):
        own = _read_kib_stats(f"/proc/{os.getpid()}/status").get("VmRSS")
        native = None
        if self.child_pid is not None:
            native = _read_kib_stats(f"/proc/{self.child_pid}/status").get("VmRSS")
            self.samples_with_child += int(native is not None)
            self.child_exit_seen = self.child_exit_seen or native is None
        if own is not None:
            self.maximum_self_rss = max(self.maximum_self_rss or 0, own)
        if native is not None:
            self.maximum_child_rss = max(self.maximum_child_rss or 0, native)
        if own is not None and native is not None:
            self.maximum_combined_rss = max(self.maximum_combined_rss or 0, own + native)
            own_pss = _smaps_rollup(os.getpid())
            child_pss = _smaps_rollup(self.child_pid)
            if own_pss is not None and child_pss is not None:
                self.maximum_combined_pss = max(self.maximum_combined_pss or 0, own_pss + child_pss)
        self.samples += 1

    def _loop(self):
        try:
            while self.running:
                self.take()
                time.sleep(self.interval)
            self.take()
        except Exception as exc:
            self.errors.append(type(exc).__name__)

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.running = False
        self.thread.join()
        self.child_pid = None

    def record(self):
        self.take()
        return {
            "samples": self.samples,
            "native_samples": self.samples_with_child,
            "maximum_worker_rss_bytes": self.maximum_self_rss,
            "maximum_native_rss_bytes": self.maximum_child_rss,
            "maximum_overlapping_worker_plus_native_rss_bytes": self.maximum_combined_rss,
            "maximum_overlapping_pss_bytes": self.maximum_combined_pss,
            "errors": self.errors,
        }


def _synthetic_retention():
    """Create bounded maximum counts, with NO true/held-out physical outcome."""
    import torch
    from nca.controllers.jolt_learned import FEATURE_NAMES, JoltPolicy
    from nca.train.jolt_learning import LearningRow, ordered_window
    torch.manual_seed(STRESS_SEED)
    n_features = len(FEATURE_NAMES)
    row = LearningRow(tuple([.25] * n_features), reset=True,
                      label=(.0, .0, .3, .0, -1., 0., 0.),
                      eligible=True, advance=True, reset_after=False)
    compact_teacher = [
        tuple(deepcopy(row) for _ in range(40)) for _ in range(64)
    ]
    # Authored metrics are bounded to 48 KiB and the other 6,380 to 8 KiB.
    # Explicitly exceed semantic field sizes; this padded shape is a memory
    # stress proxy, not a valid result JSON or independent completeness audit.
    results = []
    for i in range(RESULT_ROWS):
        width = 48 * 1024 if i < AUTHORED_ROWS else 8 * 1024
        blob = "x" * (width - 512 + i % 19)
        results.append({
            "id": "synthetic-%05d" % i, "stage": "teacher" if i < 64 else "ppo",
            "metrics": {"bounded_synthetic_payload": blob},
            "outcome": {"complete": True, "success": False},
            "supervisor_interventions": 0, "owner_closure_confirmed": True,
        })
    # Six initial/final BC models + six initial/final residual models and
    # their separately loaded BC bases + three retained BC finals + one
    # actively fitting residual. A source over-approximation (22 models).
    bc = [JoltPolicy("bc") for _ in range(6 + 6 + 3)]
    residual = [JoltPolicy("residual") for _ in range(6 + 1)]
    optimizer = torch.optim.Adam(residual[-1].parameters(), lr=.0003)
    optimizer_bc = torch.optim.Adam(bc[-1].parameters(), lr=.001)
    reserve = lambda *_args, **_kwargs: None
    for policy, optim, stage, base in (
        (bc[-1], optimizer_bc, "bc", None),
        (residual[-1], optimizer, "ppo", bc[-2]),
    ):
        policy.train()
        # Real GRU32 forward/backward graph and Adam allocation on synthetic
        # feature rows. No lesson, policy training cycle or checkpoint.
        kwargs = {"reserve": reserve, "stage": stage, "length": 16}
        if base is not None:
            base.eval().requires_grad_(False)
            kwargs["base_policy"] = base
        output = ordered_window(policy, compact_teacher[0], 0, **kwargs)
        loss = output.means.sum() + output.button_logits.sum() + output.values.sum()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    # Same original fixed 256-decision upper bound, plus the 256-episode
    # structural extreme. These are synthetic Python proxies, NOT authentic
    # rollout dataclasses; full episode-object memory remains unproved.
    ppo_episode_proxies = [
        {"id": f"synthetic-rollout-{i}", "rows": [deepcopy(row)]}
        for i in range(MAX_PPO_DECISIONS_PER_UPDATE)
    ]
    # Include the 64MiB v3 worker-result reader and a second JSON buffer
    # concurrently as a deliberately conservative synthetic memory stress.
    audit_buffers = [bytearray(64 * 1024 * 1024) for _ in range(2)]
    return {
        "objects": (results, compact_teacher, bc, residual, optimizer,
                    optimizer_bc, ppo_episode_proxies, audit_buffers),
        "shape": {
            "results": len(results), "authored_rows": AUTHORED_ROWS,
            "teacher_rows": sum(len(ep) for ep in compact_teacher),
            "ppo_proxy_episodes": len(ppo_episode_proxies),
            "ppo_proxy_decisions": MAX_PPO_DECISIONS_PER_UPDATE,
            "simultaneous_policy_instances": len(bc) + len(residual),
            "worker_result_reader_buffers_bytes": sum(len(b) for b in audit_buffers),
            "metrics_payload_bytes": sum(len(r["metrics"]["bounded_synthetic_payload"]) for r in results),
            "actual_native_case_or_sealed_data_used": False,
            "actual_training_or_benchmark_performed": False,
        },
    }


def measure(args):
    import numpy
    import torch
    from nca.contracts import sha256_json
    from nca.adapters.godot import inspect_host
    from nca.train.jolt_admission_v3 import validate_config_v3
    from nca.train.jolt_screen import source_inventory, runtime_identity
    from nca.train.jolt_execution_v3 import OriginalClockV3
    from nca.train.jolt_native_v3 import BoundedGodotHostV3
    from nca.train.jolt_storage_v3 import storage_bounds_v3

    if platform.system() != "Linux":
        raise RuntimeError("process-tree RSS measurement requires Linux")
    if str(torch.__version__) != "2.8.0+cpu" or numpy.__version__ != "2.3.5":
        raise RuntimeError("pinned CPU Torch/NumPy required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = json.loads((ROOT / "configs/learning/jolt-learn-003-v2.json").read_text())
    validate_config_v3(ROOT, config)
    source_files = source_inventory(ROOT)
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("immutable memory evidence output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    host_identity = inspect_host(args.checkout, args.godot, timeout=None)
    started = time.monotonic()
    frozen = None
    native = None
    actual_native_pid = None
    native_cleaned = False
    with ProcessMemorySampler() as sampler:
        try:
            frozen = _synthetic_retention()
            sampler.take()
            origin = OriginalClockV3(started, maximum_wall_seconds=None,
                                     monotonic=time.monotonic)
            native = BoundedGodotHostV3(args.checkout, args.godot,
                original_clock=origin, verified_identity=host_identity, timeout=None)
            actual_native_pid = native.pid
            sampler.child_pid = actual_native_pid
            # Capability handshake/idle residence only; do NOT send reset,
            # step, train, inspect sealed data or issue an experiment command.
            time.sleep(1.0)
            sampler.take()
        finally:
            if native is not None:
                native.close(graceful=True)
                native_cleaned = native._process.poll() is not None
        evidence = sampler.record()
    if evidence["maximum_overlapping_worker_plus_native_rss_bytes"] is None or not native_cleaned:
        raise RuntimeError("native/worker coexistence measurement or closure missing")
    peak = evidence["maximum_overlapping_worker_plus_native_rss_bytes"]
    report = {
        "schema": SOURCE_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "allocation_id": config["allocation_id"],
        "source_revision": subprocess.check_output(["git", "-C", str(ROOT),
            "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": sha256_json(source_files),
        "source_files": source_files,
        "config_sha256": sha256_json(config),
        "runtime": runtime_identity(),
        "host_identity": host_identity,
        "storage_bounds": storage_bounds_v3(),
        "physical_experiment_started": False, "allocation_consumed": False,
        "native_host_started_for_diagnostic": True,
        "native_handshake_only": True,
        "sealed_case_inspection_or_training": False,
        "synthetic_shape": frozen["shape"],
        "memory": {
            **evidence, "worker_ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cgroup": _cgroup(), "proc_meminfo": _meminfo(),
            "runner_label": os.environ.get("RUNNER_OS") and "ubuntu-24.04",
            "advertised_public_runner_bytes": CPU_RUNNER_ADVERTISED_BYTES,
            "illustrative_fourfold_peak_bytes": 4 * peak,
            "illustrative_fourfold_peak_below_runner_spec": 4 * peak < CPU_RUNNER_ADVERTISED_BYTES,
            "actual_full_worker_and_native_peak_measured": False,
        },
        "elapsed_seconds": time.monotonic() - started,
        "admission": {
            "accepted": False, "offline_scope_only": True,
            "prospective_experiment_admitted": False,
            "peak_resident_memory_gate_established": False,
            "reason": "Synthetic retained records and native handshake do not exercise a full real worker, every original episode, native stepped physics, full audit and publication.",
        },
        "limitations": [
            "A real pinned Godot process is started only for a capability handshake; no case, native reset, physical step, learner or sealed evaluation is opened.",
            "Synthetic retained results are deliberately byte-padded but not source-complete semantic rows.",
            "PPO proxies do not provide the entire validated JoltTransition/episode object graph.",
            "Linux VmRSS and /proc/self VmHWM are observed values, not hard memory ceilings or a guarantee for all future host/native allocations.",
            "Sampling can miss transients; simultaneous process RSS double-counts shared pages and is conservative compared with total unique PSS.",
            "GitHub public ubuntu-24.04 advertises 16 GB; /proc/meminfo and v1/v2 cgroup values may refer to a shared host. Missing quotas are not assumed infinite.",
            "No source/runtime-bound accepted prospective independent human/agent reviews are supplied by this measurement.",
        ],
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"schema": SOURCE_SCHEMA, "completed": True,
        "synthetic_plus_handshake_peak_bytes": peak,
        "native_host_closed": native_cleaned, "admission": False,
        "physical_experiment_started": False, "allocation_consumed": False}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--godot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return measure(parser.parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(2)
