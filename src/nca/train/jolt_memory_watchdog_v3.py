"""Source-bound, non-hard process-tree RSS watchdog for admitted Jolt v3.

The original Linux observer is the subreaper. Sample its own RSS plus every
identity-checked descendant (CLI, worker, Godot and verifier) on the original
process tree; a breach or unavailable live RSS stops the current invocation.
The observer retains original streams and independently reaps every child.
This is an operational soft threshold, NOT a cgroup or kernel memory ceiling.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import time

from nca.contracts import ContractError, sha256_json

WATCHDOG_SCHEMA = "nca-jolt-003-process-tree-memory-watchdog/v1"
RECEIPT_SCHEMA = "nca-jolt-003-process-tree-memory-receipt/v1"
SOFT_THRESHOLD_BYTES = 8 * 1024 ** 3
SAMPLE_INTERVAL_SECONDS = 0.25


def watchdog_policy_v3():
    """Exact reviewable policy. None of these fields imply hard OOM safety."""
    return {
        "schema": WATCHDOG_SCHEMA,
        "soft_threshold_bytes": SOFT_THRESHOLD_BYTES,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "scope": "observer_self_and_subreaper_owned_descendants",
        "measurement": "sum_identity_checked_live_VmRSS_bytes",
        "on_exceedance": "abort_original_invocation_and_revoke_owned_tree",
        "on_unreadable_live_process": "abort_original_invocation",
        "kernel_hard_limit_enforced": False,
        "peak_between_samples_guaranteed": False,
        "automatic_retry": False,
    }


def validate_watchdog_policy_v3(value):
    """Do not accept bool-as-int or weaker substituted watchdog policies."""
    if not isinstance(value, dict) or sha256_json(value) != sha256_json(watchdog_policy_v3()):
        raise ContractError("jolt_v3_memory_policy", "an exact 8 GiB original-tree watchdog is required")
    return value


def _rss_bytes(pid, identity, helper):
    """Reject PID reuse and inaccessible *live* /proc status, without guessing.

    An exited PID may vanish after identity discovery; it is not a live load.
    Zombies are zero-RSS evidence, but still belong to the reaping obligation.
    """
    try:
        first = helper._process(pid)
    except FileNotFoundError:
        return None
    if first["start"] != identity["start"]:
        raise ContractError("jolt_v3_memory_identity", "process identity changed before RSS read")
    if first["state"] == "Z":
        return 0
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except FileNotFoundError:
        try:
            helper._process(pid)
        except FileNotFoundError:
            return None
        raise ContractError("jolt_v3_memory_unreadable", "live process RSS is inaccessible")
    except OSError as exc:
        raise ContractError("jolt_v3_memory_unreadable", "live process RSS is inaccessible") from exc
    result = None
    for line in lines:
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) != 3 or not fields[1].isdecimal() or fields[2] != "kB":
                raise ContractError("jolt_v3_memory_unreadable", "invalid Linux VmRSS reading")
            result = int(fields[1]) * 1024
            break
    try:
        second = helper._process(pid)
    except FileNotFoundError:
        return None
    if second["start"] != identity["start"]:
        raise ContractError("jolt_v3_memory_identity", "process identity changed during RSS read")
    if second["state"] == "Z":
        return 0
    if result is None or result < 0:
        raise ContractError("jolt_v3_memory_unreadable", "live process has no readable VmRSS")
    return result


class ProcessTreeWatchdogV3:
    """Sampler, invoked in the observer's actual nonblocking child-wait loop.

    Descendants are identified by the original observer's Linux subreaper
    helper and their /proc start times. The complete worker is unmodified.
    Missing samples do not become zeros; a monitoring error invalidates the
    current invocation before any positive completion can be declared.
    """

    def __init__(self, *, owner_start, helper, monotonic=time.monotonic,
                 policy=None, rss_reader=_rss_bytes):
        validate_watchdog_policy_v3(watchdog_policy_v3() if policy is None else policy)
        if type(owner_start) is not int or owner_start <= 0:
            raise ContractError("jolt_v3_memory_identity", "the original subreaper identity is required")
        self.owner_start = owner_start
        self.helper = helper
        self.monotonic = monotonic
        self.rss_reader = rss_reader
        self.owner_pid = os.getpid()
        self.started_monotonic = monotonic()
        self.last_monotonic = None
        self.sample_count = 0
        self.peak_total_rss_bytes = 0
        self.peak_owned_descendants = 0
        self.tripped = False
        self.failure_code = None
        self.monitoring_error = False
        self.max_overage_bytes = 0

    def check(self, *, force=False):
        if self.tripped:
            raise ContractError(self.failure_code or "jolt_v3_memory_limit",
                                "original invocation was already stopped by memory watchdog")
        now = self.monotonic()
        if not math.isfinite(now) or now < self.started_monotonic:
            self._fail("jolt_v3_memory_clock")
        if (not force and self.last_monotonic is not None
                and now - self.last_monotonic < SAMPLE_INTERVAL_SECONDS):
            return
        try:
            descendants = self.helper.owned_processes(self.owner_start)
            own_identity = self.helper._process(self.owner_pid)
            if own_identity["start"] != self.owner_start or own_identity["state"] == "Z":
                raise ContractError("jolt_v3_memory_identity", "original observer identity differs")
            participants = {self.owner_pid: own_identity, **descendants}
            total = 0
            for pid, identity in sorted(participants.items()):
                current = self.rss_reader(pid, identity, self.helper)
                if current is None:
                    if pid == self.owner_pid:
                        raise ContractError("jolt_v3_memory_unreadable", "observer RSS disappeared")
                    # A process that actually exited is no longer resident.
                    # A *live* descendant whose VmRSS cannot be read must
                    # refuse, rather than falsely counting zero memory.
                    try:
                        still = self.helper._process(pid)
                    except FileNotFoundError:
                        continue
                    if still["start"] != identity["start"]:
                        raise ContractError("jolt_v3_memory_identity", "descendant PID changed")
                    if still["state"] == "Z":
                        continue
                    raise ContractError("jolt_v3_memory_unreadable", "live descendant has no RSS")
                if type(current) is not int or current < 0:
                    raise ContractError("jolt_v3_memory_unreadable", "noninteger process RSS")
                total += current
            self.sample_count += 1
            self.last_monotonic = now
            self.peak_total_rss_bytes = max(self.peak_total_rss_bytes, total)
            self.peak_owned_descendants = max(self.peak_owned_descendants, len(descendants))
            if total >= SOFT_THRESHOLD_BYTES:
                self.max_overage_bytes = max(self.max_overage_bytes, total - SOFT_THRESHOLD_BYTES)
                self._fail("jolt_v3_memory_limit")
        except BaseException as exc:
            if not self.tripped:
                self._fail(getattr(exc, "code", None) or "jolt_v3_memory_unreadable")
            raise

    def _fail(self, code):
        self.tripped = True
        self.failure_code = code
        self.monitoring_error = code != "jolt_v3_memory_limit"
        raise ContractError(code, "one original execution must stop; never retry or raise the threshold")

    def receipt(self, *, cleanup_confirmed, declaration_identity):
        """An original readbackable receipt, not an after-the-fact success gate."""
        return {
            "schema": RECEIPT_SCHEMA, "policy": watchdog_policy_v3(),
            "declaration_identity": declaration_identity,
            "allocation_execution_started_monotonic": self.started_monotonic,
            "sample_count": self.sample_count,
            "peak_sampled_process_tree_rss_bytes": self.peak_total_rss_bytes,
            "peak_owned_descendants": self.peak_owned_descendants,
            "limit_reached": self.tripped and self.failure_code == "jolt_v3_memory_limit",
            "sampling_failure": self.monitoring_error,
            "failure_code": self.failure_code,
            "sampled_overage_bytes": self.max_overage_bytes,
            "owned_tree_cleanup_confirmed": cleanup_confirmed,
            "kernel_hard_limit_enforced": False,
            "unobserved_transient_peaks_possible": True,
            "declared_software_resource_gate_accepted_by_this_receipt": False,
            "automatic_retry": False,
        }
