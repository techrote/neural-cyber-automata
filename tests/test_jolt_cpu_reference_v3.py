"""CPU diagnostic normalization CLI: synthetic records only, zero training."""
import importlib.util
from pathlib import Path
import unittest

from nca.contracts import ContractError
from nca.train.jolt_runtime_v3 import CALIBRATION_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("jolt_compare_cpu_v3", ROOT / "tools/compare_jolt_cpu_v3.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def original(model, revision, rate):
    return {
        "source_revision": revision,
        "runtime": {
            "torch":"2.8.0+cpu", "numpy":"2.3.5", "device":"cpu",
            "threads":1, "interop_threads":1, "python":"3.12.10",
            "system":"Linux", "kernel":"6.17", "machine":"x86_64",
            "libc":["glibc","2.39"], "cpu_model":model,
            "logical_cpus_available":4, "deterministic_algorithms":True,
        },
        "cpu_calibration": {
            "schema":CALIBRATION_SCHEMA,
            "method":"sha256_64kib_fixed_buffer_v1",
            "bytes_per_second":rate
        },
    }


class CpuReferenceComparisonTests(unittest.TestCase):
    def test_cpu_identity_can_differ_without_changing_pinned_software(self):
        reference = original("EPYC 9V74", "a"*40, 2e9)
        observed = original("EPYC 7763", "b"*40, 1e9)
        result = CLI.compare(reference, observed, duration_seconds=20)
        self.assertAlmostEqual(result["normalization"]["reference_seconds_approximate"], 10)
        self.assertEqual(result["reference_cpu_model"], "EPYC 9V74")
        self.assertEqual(result["observed_cpu_model"], "EPYC 7763")
        self.assertTrue(result["not_a_hard_resource_bound"])
        self.assertFalse(result["experimental_admission_inferred"])

    def test_missing_calibration_or_software_drifts_refuse(self):
        reference=original("EPYC 9V74","a"*40,1e9)
        observed=original("EPYC 7763","b"*40,1e9)
        observed["cpu_calibration"] = None
        with self.assertRaises(ValueError):
            CLI.compare(reference, observed, duration_seconds=20)
        observed=original("EPYC 7763","b"*40,1e9)
        observed["runtime"]["torch"] = "2.8.0+cuda"
        with self.assertRaises(ContractError):
            CLI.compare(reference, observed, duration_seconds=20)
        observed=original("EPYC 7763","b"*40,1e9)
        with self.assertRaises(ContractError):
            CLI.compare(reference, observed, duration_seconds=-1)


if __name__ == "__main__":
    unittest.main()
