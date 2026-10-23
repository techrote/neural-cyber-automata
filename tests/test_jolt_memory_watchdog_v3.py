"""Nonexperimental RSS watchdog and original child-wait regression controls.

No Godot environment, frozen case, learner or single-use allocation is run.
© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

from copy import deepcopy
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from nca.contracts import ContractError
from nca.train.jolt_memory_watchdog_v3 import (
    SOFT_THRESHOLD_BYTES, SAMPLE_INTERVAL_SECONDS, WATCHDOG_SCHEMA,
    ProcessTreeWatchdogV3, _rss_bytes, validate_watchdog_policy_v3,
    watchdog_policy_v3,
)
from nca.train.jolt_streams_v3 import wait_bounded_child


class FakeTree:
    def __init__(self, owner, parent=0):
        self.owner = owner
        self.parent = parent
        self.descendants = {}
        self.identities = {os.getpid(): {"parent": 1, "start": owner, "state": "R"}}

    def owned_processes(self, identity):
        if identity != self.owner:
            raise RuntimeError("wrong original observer")
        return self.descendants.copy()

    def _process(self, pid):
        return self.identities[pid]


def sampler(*, owner_bytes, child_bytes=0):
    tree = FakeTree(1234)
    if child_bytes:
        child = 8127
        tree.identities[child] = {"parent": os.getpid(), "start": 4321, "state": "R"}
        tree.descendants[child] = tree.identities[child]
    values = {os.getpid(): owner_bytes}
    if child_bytes:
        values[child] = child_bytes
    now = [300.]
    guard = ProcessTreeWatchdogV3(
        owner_start=1234, helper=tree, monotonic=lambda: now[0],
        rss_reader=lambda pid, identity, helper: values[pid],
    )
    return guard, tree, values, now


class MemoryPolicyTests(unittest.TestCase):
    def test_exact_soft_limit_and_no_hard_guarantee(self):
        policy = watchdog_policy_v3()
        self.assertEqual(policy["schema"], WATCHDOG_SCHEMA)
        self.assertEqual(policy["soft_threshold_bytes"], 8 * 1024 ** 3)
        self.assertEqual(policy["sample_interval_seconds"], .25)
        self.assertIs(policy["kernel_hard_limit_enforced"], False)
        self.assertIs(policy["peak_between_samples_guaranteed"], False)
        self.assertEqual(validate_watchdog_policy_v3(policy), policy)
        for field, altered in [
            ("soft_threshold_bytes", 12 * 1024 ** 3),
            ("sample_interval_seconds", 2.0),
            ("on_exceedance", "warn_and_continue"),
            ("kernel_hard_limit_enforced", True),
            ("automatic_retry", True),
            ("soft_threshold_bytes", True),
        ]:
            changed = deepcopy(policy)
            changed[field] = altered
            with self.subTest(field=field, replacement=altered), self.assertRaises(ContractError):
                validate_watchdog_policy_v3(changed)

    def test_bounded_interval_and_original_descendant_aggregation(self):
        guard, _tree, values, clock = sampler(owner_bytes=400_000_000, child_bytes=300_000_000)
        guard.check(force=True)
        self.assertEqual(guard.sample_count, 1)
        self.assertEqual(guard.peak_total_rss_bytes, 700_000_000)
        self.assertEqual(guard.peak_owned_descendants, 1)
        clock[0] += .1
        guard.check()
        self.assertEqual(guard.sample_count, 1)
        clock[0] += SAMPLE_INTERVAL_SECONDS
        values[os.getpid()] += 200_000_000
        guard.check()
        self.assertEqual(guard.sample_count, 2)
        self.assertEqual(guard.peak_total_rss_bytes, 900_000_000)
        record = guard.receipt(cleanup_confirmed=True, declaration_identity={"sha256":"0"*64})
        self.assertFalse(record["limit_reached"])
        self.assertEqual(record["sample_count"], 2)
        self.assertFalse(record["kernel_hard_limit_enforced"])
        self.assertTrue(record["owned_tree_cleanup_confirmed"])

    def test_sampled_overage_permanently_refuses_completion(self):
        guard, _tree, _values, clock = sampler(
            owner_bytes=SOFT_THRESHOLD_BYTES - 128, child_bytes=256)
        with self.assertRaisesRegex(ContractError, "jolt_v3_memory_limit"):
            guard.check(force=True)
        self.assertTrue(guard.tripped)
        self.assertEqual(guard.peak_total_rss_bytes, SOFT_THRESHOLD_BYTES + 128)
        with self.assertRaises(ContractError):
            clock[0] += 3.
            guard.check()
        record = guard.receipt(cleanup_confirmed=True, declaration_identity={"sha256":"0"*64})
        self.assertTrue(record["limit_reached"])
        self.assertFalse(record["sampling_failure"])
        self.assertEqual(record["sampled_overage_bytes"], 128)

    def test_missing_live_memory_is_a_failure_not_zero(self):
        guard, _tree, values, _now = sampler(owner_bytes=1, child_bytes=1)
        child = next(pid for pid in values if pid != os.getpid())
        values[child] = None
        with self.assertRaisesRegex(ContractError, "jolt_v3_memory_unreadable"):
            guard.check(force=True)
        self.assertTrue(guard.monitoring_error)
        self.assertIsNotNone(guard.failure_code)

    def test_changed_observer_start_time_is_a_failure(self):
        guard, tree, _values, _now = sampler(owner_bytes=100)
        tree.identities[os.getpid()]["start"] += 1
        with self.assertRaisesRegex(ContractError, "jolt_v3_memory_identity"):
            guard.check(force=True)
        self.assertTrue(guard.tripped)

    def test_linux_live_status_and_zombie_have_distinct_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "status"
            path.write_text("Name:\tsynthetic\nVmRSS:\t2048 kB\n")
            helper = FakeTree(99)
            helper.identities[8127] = {"parent": 1, "start": 77, "state": "R"}
            identity = helper.identities[8127]
            with mock.patch("nca.train.jolt_memory_watchdog_v3.Path",
                            side_effect=lambda source: path if source == "/proc/8127/status" else Path(source)):
                self.assertEqual(_rss_bytes(8127, identity, helper), 2048 * 1024)
            helper.identities[8127] = {"parent": 1, "start": 77, "state": "Z"}
            self.assertEqual(_rss_bytes(8127, identity, helper), 0)

    def test_sampler_cannot_succeed_with_nonfinite_clock(self):
        guard, _tree, _values, clock = sampler(owner_bytes=100)
        clock[0] = float("nan")
        with self.assertRaises(ContractError):
            guard.check(force=True)
        self.assertTrue(guard.tripped)


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux original PIPE and process tree")
class OriginalWaitLoopTests(unittest.TestCase):
    def test_soft_guard_is_polled_while_original_child_is_active(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(12)"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, start_new_session=True)
        count = [0]
        try:
            with tempfile.TemporaryDirectory() as folder:
                with (Path(folder) / "stdout").open("wb") as out, \
                        (Path(folder) / "stderr").open("wb") as err:
                    def stop():
                        count[0] += 1
                        if count[0] >= 2:
                            raise ContractError("jolt_v3_memory_limit", "synthetic overage")
                    with self.assertRaises(ContractError):
                        wait_bounded_child(process, out, err, deadline=None, check=stop)
            self.assertGreaterEqual(count[0], 2)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)

    def test_original_wait_without_watchdog_preserves_complete_stdout(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "print('original')"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, start_new_session=True)
        with tempfile.TemporaryDirectory() as folder:
            stdout = Path(folder) / "stdout"
            stderr = Path(folder) / "stderr"
            with stdout.open("wb") as out, stderr.open("wb") as err:
                self.assertEqual(wait_bounded_child(process, out, err, deadline=None), 0)
            self.assertEqual(stdout.read_bytes(), b"original\n")
        process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
