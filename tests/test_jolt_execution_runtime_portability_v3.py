"""Integration contract for hardware-advisory Jolt v3 execution admission.

Uses synthetic declarations and mocked source/host discovery; never starts
training, a native host or the one-time experimental allocation.
© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

from nca.contracts import sha256_json

HAS_TORCH = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_TORCH, "actual v3 module requires optional Torch CPU")
class PortableExecutorAdmissionTests(unittest.TestCase):
    def fixture(self):
        from test_jolt_runtime_portability_v3 import (
            admitted_synthetic_declaration, runtime, synth_reviews,
        )
        from nca.train.jolt_admission_v3 import ALLOCATION_ID, AUTHORIZATION_REF
        conf = json.loads((ROOT / "configs/learning/jolt-learn-003-v2.json").read_text())
        declared = admitted_synthetic_declaration()
        files = {"src/nca/fake-source.py": "f" * 64}
        declared.update(
            schema="nca-learn-jolt-003-execution/v3", attempts=1,
            allocation_id=ALLOCATION_ID, authorization_ref=AUTHORIZATION_REF,
            source_revision="a" * 40, source_files=files,
            source_sha256=sha256_json(files),
            config_sha256=sha256_json(conf),
            teacher_prerequisite=conf["prerequisite"],
            opened_inventory=conf["opened_inventory"], split=conf["split"],
            ppo_case_stream_sha256=sha256_json(conf["ppo_case_stream"]),
            comparison_order_sha256=sha256_json(conf["comparison_order"]),
            output_identity={"basename":ALLOCATION_ID,
                "lifecycle_basename":ALLOCATION_ID+"-lifecycle", "reuse_forbidden":True},
            automatic_retry=False, automatic_extension=False,
        )
        declared["resource_admission"]["source_sha256"] = declared["source_sha256"]
        declared["resource_admission"]["config_sha256"] = declared["config_sha256"]
        synth_reviews(declared)
        return declared, conf, files, runtime

    def inspect(self, declared, conf, files, actual_runtime, *, actual_host=None):
        from nca.train.jolt_admission_v3 import validate_execution_v3
        import nca.train.jolt_admission_v3 as admission

        def subprocess_reply(command, **_kwargs):
            if command[-2:] == ["rev-parse", "HEAD"]:
                return declared["source_revision"]
            if command[-2:] == ["status", "--porcelain"]:
                return ""
            raise AssertionError("unexpected Git command")
        current_host = declared["host_identity"] if actual_host is None else actual_host
        with mock.patch.object(admission, "validate_config_v3"), \
                mock.patch.object(admission, "validate_v3_allocation_identity"), \
                mock.patch.object(admission, "_read", return_value={"consumed_allocations":[]}), \
                mock.patch.object(admission.subprocess, "check_output", side_effect=subprocess_reply), \
                mock.patch("nca.train.jolt_screen.source_inventory", return_value=files), \
                mock.patch("nca.train.jolt_screen.runtime_identity", return_value=actual_runtime):
            return validate_execution_v3(ROOT, declared, config=conf, host_identity=current_host)

    def test_cpu_models_kernels_cores_and_python_patch_are_not_execution_gates(self):
        declared, conf, files, runtime = self.fixture()
        actual = runtime(model="AMD EPYC 7763", kernel="6.18.55-azure", cpus=9)
        actual["python"] = "3.12.14"
        actual["libc"] = ["glibc", "2.40"]
        self.assertIsNone(self.inspect(declared, conf, files, actual))

    def test_software_pins_are_still_required(self):
        declared, conf, files, runtime = self.fixture()
        for key, value in (("torch", "2.8.0+cu128"), ("numpy", "2.4.0"),
                           ("machine", "aarch64"), ("deterministic_algorithms", False)):
            actual = runtime()
            actual[key] = value
            with self.subTest(key=key), self.assertRaises(Exception):
                self.inspect(declared, conf, files, actual)

    def test_source_host_budget_and_reviews_remain_exact(self):
        declared, conf, files, runtime = self.fixture()
        with self.assertRaises(Exception):
            self.inspect(declared, conf, files, runtime(),
                         actual_host={"synthetic_test_host":"tampered"})
        bad_source = deepcopy(declared)
        bad_source["source_files"] = {"src/nca/fake-source.py":"a"*64}
        with self.assertRaises(Exception):
            self.inspect(bad_source, conf, files, runtime())
        bad_review = deepcopy(declared)
        bad_review["independent_reviews"].pop()
        with self.assertRaises(Exception):
            self.inspect(bad_review, conf, files, runtime())
        bad_budget = deepcopy(declared)
        bad_budget["budget"]["counters"]["attempts"] = 2
        with self.assertRaises(Exception):
            self.inspect(bad_budget, conf, files, runtime())


if __name__ == "__main__":
    unittest.main()
