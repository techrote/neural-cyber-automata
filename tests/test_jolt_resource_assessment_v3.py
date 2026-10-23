"""Source-bound, *negative* Jolt resource assessment tests.

No training, native host, policy checkpoint or admitted experiment is opened.
Measured inputs here are synthetic fixtures and cannot certify full memory.
© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from nca.contracts import ContractError, sha256_json
from nca.train.jolt_admission_v3 import validate_resource_admission
from nca.train.jolt_memory_watchdog_v3 import watchdog_policy_v3
from nca.train.jolt_storage_v3 import storage_bounds_v3

ROOT = Path(__file__).resolve().parents[1]
# The ordinary minimal CPU conformance lane intentionally installs neither
# Torch nor NumPy. The original full-worker resource tool imports the
# optional learning runtime; discover it only in the pinned optional lane.
OPTIONAL_RUNTIME_AVAILABLE = (importlib.util.find_spec("torch") is not None
                              and importlib.util.find_spec("numpy") is not None)
ASSESS = None
if OPTIONAL_RUNTIME_AVAILABLE:
    SPEC = importlib.util.spec_from_file_location(
        "jolt_resources_v3", ROOT / "tools/assess_jolt_resources_v3.py")
    ASSESS = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(ASSESS)


@unittest.skipUnless(OPTIONAL_RUNTIME_AVAILABLE, "optional pinned CPU learning runtime required")
class ResourceAssessmentTests(unittest.TestCase):
    def fixture(self, folder):
        config = json.loads((ROOT / "configs/learning/jolt-learn-003-v2.json").read_text())
        source = {"src/nca/production.py": "f"*64}
        runtime = {
            "torch":"2.8.0+cpu", "numpy":"2.3.5", "device":"cpu",
            "threads":1, "interop_threads":1, "python":"3.12.10",
            "system":"Linux", "kernel":"6.17", "machine":"x86_64",
            "libc":["glibc","2.39"], "cpu_model":"Synthetic EPYC",
            "logical_cpus_available":4, "deterministic_algorithms":True,
        }
        from nca.train.jolt_runtime_v3 import capability_sha256
        record = {
            "schema":"nca-jolt-003-memory-diagnostic/v1",
            "source_revision":"a"*40, "source_files":source,
            "source_sha256":sha256_json(source),
            "config_sha256":sha256_json(config),
            "runtime":runtime,
            "runtime_software_contract_sha256":capability_sha256(runtime),
            "host_identity":{"host_revision":config["host_revision"]},
            "storage_bounds":storage_bounds_v3(),
            "physical_experiment_started":False,
            "allocation_consumed":False,
            "native_handshake_only":True,
            "sealed_case_inspection_or_training":False,
            "admission":{"accepted":False},
            "synthetic_shape":{
                "results":6464, "ppo_proxy_decisions":256,
                "actual_training_or_benchmark_performed":False,
            },
            "memory":{
                "maximum_overlapping_worker_plus_native_rss_bytes":640_000_000,
                "worker_ru_maxrss_bytes":500_000_000,
                "native_samples":19,
            }
        }
        report = Path(folder) / "original-memory.json"
        report.write_text(json.dumps(record))
        return config, runtime, source, record, report

    def assess_with_fixture(self, folder, mutate=None, disk_bytes=80_000_000_000):
        config, runtime, source, record, report = self.fixture(folder)
        if mutate:
            mutate(record)
            report.write_text(json.dumps(record))
        with mock.patch.object(ASSESS, "validate_config_v3"), \
                mock.patch.object(ASSESS, "source_inventory", return_value=source), \
                mock.patch.object(ASSESS, "runtime_identity", return_value=runtime), \
                mock.patch.object(ASSESS.subprocess, "check_output", return_value="a"*40), \
                mock.patch.object(ASSESS.shutil, "disk_usage",
                                  return_value=SimpleNamespace(free=disk_bytes)):
            result = ASSESS.assess(root=ROOT,
                config_path=ROOT/"configs/learning/jolt-learn-003-v2.json",
                measured_report_path=report, output_parent=Path(folder))
        return result

    def test_measured_headroom_does_not_approve_full_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            record=self.assess_with_fixture(folder)
        self.assertFalse(record["accepted"])
        self.assertFalse(record["prospective_experiment_admitted"])
        self.assertFalse(record["gates"]["peak_resident_memory"])
        self.assertTrue(record["gates"]["total_artifact_bytes"])
        self.assertTrue(record["gates"]["available_disk"])
        self.assertEqual(record["memory_watchdog"], watchdog_policy_v3())
        self.assertEqual(record["memory"]["illustrative_fourfold_rss_bytes"], 2_560_000_000)
        self.assertGreater(len(record["unresolved_blocking_findings"]), 2)
        self.assertFalse(record["experiment_allocation_consumed"])
        with self.assertRaises(Exception):
            validate_resource_admission({"resource_admission":record})

    def test_changed_or_physical_memory_evidence_fails_closed(self):
        for field, replacement in [
            ("source_sha256", "b"*64),
            ("config_sha256", "c"*64),
            ("native_handshake_only", False),
            ("physical_experiment_started", True),
            ("allocation_consumed", True),
        ]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                def mutate(data):
                    data[field]=replacement
                with self.assertRaises((ContractError, KeyError)):
                    self.assess_with_fixture(folder, mutate)

    def test_missing_or_excessive_memory_refuses_or_remains_unadmitted(self):
        with tempfile.TemporaryDirectory() as folder:
            def mutate(data):
                data["memory"]["native_samples"]=0
            with self.assertRaises(ContractError):
                self.assess_with_fixture(folder, mutate)
        with tempfile.TemporaryDirectory() as folder:
            def mutate(data):
                data["memory"]["maximum_overlapping_worker_plus_native_rss_bytes"]=12*1024**3
            with self.assertRaises(ContractError):
                self.assess_with_fixture(folder, mutate)

    def test_disk_shortfall_keeps_explicit_gate_false(self):
        with tempfile.TemporaryDirectory() as folder:
            record=self.assess_with_fixture(folder, disk_bytes=1_000_000_000)
        self.assertFalse(record["gates"]["available_disk"])
        self.assertFalse(record["accepted"])
        self.assertIn("RESOURCE-DISK-01", [item["id"] for item in record["unresolved_blocking_findings"]])


if __name__ == "__main__":
    unittest.main()
