"""Nonphysical memory envelope and Linux process telemetry checks.

No test launches Godot, trains a policy or opens a physical task.
© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "jolt_memory_prepass", ROOT / "tools/check_jolt_memory_envelope_v3.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MemoryTelemetryTests(unittest.TestCase):
    def test_proc_values_use_kilobytes_and_refuse_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status"
            path.write_text(
                "Name:\tpython\nVmRSS:\t 2048 kB\nVmHWM:\t 8192 kB\n"
                "VmSize:\t10240 kB\nVmRSS:\tunknown kB\nRssAnon:\t4 kB\n"
            )
            actual = MODULE._read_kib_stats(path)
            self.assertEqual(actual["VmRSS"], 2048 * 1024)
            self.assertEqual(actual["VmHWM"], 8192 * 1024)
            self.assertEqual(actual["RssAnon"], 4 * 1024)
            self.assertNotIn("VmPeak", actual)
            self.assertEqual(MODULE._read_kib_stats(Path(directory) / "missing"), {})

    def test_no_linux_quota_is_never_zero(self):
        with mock.patch.object(MODULE.Path, "read_text", side_effect=FileNotFoundError):
            result = MODULE._cgroup()
        self.assertEqual(len(result), 6)
        self.assertTrue(all(v is None for v in result.values()))

    def test_sampler_reports_missing_child_separately(self):
        sampler = MODULE.ProcessMemorySampler()
        sampler.child_pid = -1
        sampler.take()
        result = sampler.record()
        self.assertGreaterEqual(result["samples"], 2)
        self.assertEqual(result["native_samples"], 0)
        self.assertIsNone(result["maximum_native_rss_bytes"])
        self.assertIsNone(result["maximum_overlapping_worker_plus_native_rss_bytes"])
        self.assertTrue(result["child_exit_seen"] or sampler.child_exit_seen)

    def test_memory_report_is_explicitly_nonadmitting(self):
        source = (ROOT / "tools/check_jolt_memory_envelope_v3.py").read_text()
        self.assertIn('"peak_resident_memory_gate_established": False', source)
        self.assertIn('"actual_full_worker_and_native_peak_measured": False', source)
        self.assertIn('"physical_experiment_started": False', source)
        self.assertIn('"sealed_case_inspection_or_training": False', source)
        self.assertIn("source_inventory(ROOT)", source)
        self.assertIn("inspect_host(args.checkout, args.godot, timeout=None)", source)


if __name__ == "__main__":
    unittest.main()
