"""Jolt v3 CPU identity is advisory; software/seed/host gates stay strict.

All records are synthetic, no native host, training or physical allocation.
© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import unittest

from nca.contracts import ContractError, sha256_json
from nca.train.jolt_runtime_v3 import (
    CALIBRATION_SCHEMA, POLICY, capability_contract_v3, capability_sha256,
    normalize_diagnostic_seconds, cpu_calibration_v3,
)
from nca.train.jolt_admission_v3 import (
    validate_resource_admission, validate_declaration_reviews, prospective_binding,
    PORTABLE_RESOURCE_SCHEMA, PORTABLE_REVIEW_SCHEMA,
)
from nca.train.jolt_storage_v3 import storage_bounds_v3

ROOT = Path(__file__).resolve().parents[1]


def runtime(model="AMD EPYC 9V74", kernel="6.17.0-1022-azure", cpus=4):
    return dict(torch="2.8.0+cpu", numpy="2.3.5", device="cpu",
                threads=1, interop_threads=1, python="3.12.10", system="Linux",
                kernel=kernel, machine="x86_64", libc=["glibc", "2.39"],
                cpu_model=model, logical_cpus_available=cpus,
                deterministic_algorithms=True)


def admitted_synthetic_declaration():
    conf = json.loads((ROOT / "configs/learning/jolt-learn-003-v2.json").read_text())
    budget = conf["budget"]
    host = {"synthetic_test_host": "fixed_hash"}
    declaration = dict(
        source_revision="a" * 40, source_sha256="a" * 64,
        config_sha256="b" * 64, host_identity=host,
        runtime=runtime(), budget=budget,
    )
    gate_names = {"canonical_reservation_maximum", "reader_limits", "journal_aggregate_bytes",
                  "total_artifact_bytes", "total_file_count", "peak_resident_memory",
                  "available_disk"}
    gates = {name: True for name in gate_names}
    gates["original_clock_phase_enforcement"] = None
    gates["complete_pipeline_time_feasibility"] = None
    bounds = storage_bounds_v3()
    declaration["resource_admission"] = dict(
        schema=PORTABLE_RESOURCE_SCHEMA, accepted=True,
        unresolved_blocking_findings=[], source_sha256=declaration["source_sha256"],
        config_sha256=declaration["config_sha256"],
        host_identity_sha256=sha256_json(host),
        runtime_contract_sha256=capability_sha256(declaration["runtime"]),
        cpu_comparison_policy=POLICY, gates=gates,
        serialization=bounds,
        required_free_disk_bytes=bounds["required_disk_bytes_including_1_gib_reserve"],
    )
    return declaration


def review(declaration, reviewer, role):
    record = dict(
        schema=PORTABLE_REVIEW_SCHEMA, reviewer=reviewer, review_roles=[role],
        accepted=True, offline_scope_only=False, prospective_experiment_admitted=True,
        source_revision=declaration["source_revision"],
        source_sha256=declaration["source_sha256"],
        config_sha256=declaration["config_sha256"],
        host_identity_sha256=sha256_json(declaration["host_identity"]),
        runtime_contract_sha256=capability_sha256(declaration["runtime"]),
        cpu_comparison_policy=POLICY,
        prospective_binding_sha256=sha256_json(prospective_binding(declaration)),
        unresolved_blocking_findings=[],
    )
    original = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return {"path": "docs/evidence/" + reviewer + ".json",
            "content": original, "file_sha256": hashlib.sha256(original.encode()).hexdigest()}


def synth_reviews(declaration):
    declaration["independent_reviews"] = [
        review(declaration, "independent-terminal", "terminal_learning_qualification"),
        review(declaration, "independent-envelope", "operation_serialization_deadline_feasibility"),
    ]


class PortableCpuContractTests(unittest.TestCase):
    def test_different_models_and_kernels_have_same_software_contract(self):
        left, right = runtime(), runtime("AMD EPYC 7763", "6.18.55-azure", 9)
        right["libc"] = ["glibc", "2.40"]
        right["python"] = "3.12.14"
        self.assertEqual(capability_sha256(left), capability_sha256(right))
        self.assertEqual(capability_contract_v3(left)["machine"], "x86_64")
        self.assertEqual(capability_contract_v3(left)["python_minor"], "3.12")

    def test_hardware_facts_are_still_recorded_and_not_omitted(self):
        for key, value in [("cpu_model", ""), ("kernel", None),
                           ("logical_cpus_available", 0), ("logical_cpus_available", True)]:
            with self.subTest(key=key, value=value):
                record = runtime()
                record[key] = value
                with self.assertRaises(ContractError):
                    capability_contract_v3(record)

    def test_changed_runtime_software_or_semantics_refuses(self):
        for key, value in [
            ("torch", "2.8.0+cu128"), ("numpy", "2.4.0"), ("device", "cuda"),
            ("threads", 2), ("interop_threads", 2), ("machine", "aarch64"),
            ("system", "Windows"), ("python", "3.13.1"),
            ("deterministic_algorithms", False),
        ]:
            with self.subTest(key=key):
                record = runtime()
                record[key] = value
                with self.assertRaises(ContractError):
                    capability_contract_v3(record)


class NormalizationTests(unittest.TestCase):
    def test_reference_cpu_seconds_are_scaled_for_ballpark_comparisons(self):
        measured = dict(schema=CALIBRATION_SCHEMA, method="sha256_64kib_fixed_buffer_v1",
                        bytes_per_second=2_000_000_000.)
        reference = {**measured, "bytes_per_second":1_000_000_000.}
        mapped = normalize_diagnostic_seconds(7.5, measured, reference)
        self.assertEqual(mapped["raw_seconds"], 7.5)
        self.assertAlmostEqual(mapped["reference_seconds_approximate"], 15.)
        self.assertTrue(mapped["planning_only_not_execution_admission_proof"])
        self.assertFalse(mapped["quality_thresholds_or_physical_results_normalized"])

    def test_nonfinite_missing_or_negative_calibration_is_refused(self):
        correct = dict(schema=CALIBRATION_SCHEMA, method="sha256_64kib_fixed_buffer_v1",
                       bytes_per_second=1_000_000_000.)
        for value in (float("nan"), float("inf"), 0, -1, "fast"):
            with self.subTest(value=value), self.assertRaises(ContractError):
                normalize_diagnostic_seconds(3.0, {**correct, "bytes_per_second":value}, correct)
        for raw in (-1, float("nan"), float("inf"), True):
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                normalize_diagnostic_seconds(raw, correct, correct)

    def test_small_fixed_calibration_is_positive_and_nonphysical(self):
        c = cpu_calibration_v3(repetitions=2, iterations=2)
        self.assertEqual(c["schema"], CALIBRATION_SCHEMA)
        self.assertEqual(c["repetitions"], 2)
        self.assertGreater(c["bytes_per_second"], 0)
        self.assertTrue(c["not_a_host_or_neural_workload"])


class PortableAdmissionTests(unittest.TestCase):
    def test_portable_resource_admission_refuses_unresolved_gates(self):
        declaration = admitted_synthetic_declaration()
        validate_resource_admission(declaration)
        for field in ("peak_resident_memory", "available_disk", "reader_limits"):
            rejected = deepcopy(declaration)
            rejected["resource_admission"]["gates"][field] = None
            with self.subTest(field=field), self.assertRaises(Exception):
                validate_resource_admission(rejected)

    def test_two_distinct_versioned_reviews_bind_software_not_cpu_model(self):
        declaration = admitted_synthetic_declaration()
        synth_reviews(declaration)
        validate_declaration_reviews(declaration)
        binding = sha256_json(prospective_binding(declaration))
        declaration["runtime"] = runtime("AMD EPYC 7763", "6.18.55-azure", 9)
        self.assertEqual(sha256_json(prospective_binding(declaration)), binding)
        validate_resource_admission(declaration)
        validate_declaration_reviews(declaration)

    def test_bad_host_code_or_calibration_policy_cannot_be_hardware_normalized(self):
        declaration = admitted_synthetic_declaration()
        synth_reviews(declaration)
        for mutation in ("host", "runtime_contract", "resource_cpu_policy", "review_cpu_policy"):
            test = deepcopy(declaration)
            if mutation == "host":
                test["host_identity"]["synthetic_test_host"] = "other"
            elif mutation == "runtime_contract":
                test["runtime"]["torch"] = "2.8.0+cuda"
            elif mutation == "resource_cpu_policy":
                test["resource_admission"]["cpu_comparison_policy"] = "none"
            else:
                record = json.loads(test["independent_reviews"][0]["content"])
                record["cpu_comparison_policy"] = "none"
                raw = json.dumps(record, sort_keys=True, separators=(",", ":"))
                test["independent_reviews"][0]["content"] = raw
                test["independent_reviews"][0]["file_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
            with self.subTest(mutation=mutation), self.assertRaises(Exception):
                if mutation == "review_cpu_policy":
                    validate_declaration_reviews(test)
                else:
                    validate_resource_admission(test)

    def test_unreviewed_previous_schema_is_not_promoted_to_portable(self):
        declaration = admitted_synthetic_declaration()
        synth_reviews(declaration)
        old = json.loads(declaration["independent_reviews"][0]["content"])
        old["schema"] = "nca-learn-jolt-003-source-admission-review/v1"
        raw = json.dumps(old, sort_keys=True, separators=(",", ":"))
        declaration["independent_reviews"][0]["content"] = raw
        declaration["independent_reviews"][0]["file_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
        with self.assertRaises(Exception):
            validate_declaration_reviews(declaration)


if __name__ == "__main__":
    unittest.main()
