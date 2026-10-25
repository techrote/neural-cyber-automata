"""Software-only conformance for the 4LY-004 experiment-admission package."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from nca.contracts import ContractError, sha256_json
from nca.train.flight_accounting import FlightAccountingLedger, execution_gate
from nca.train.flight_admission import (
    CUSTODY_SCHEMA,
    FINAL_COMPOSITION,
    FINAL_EVALUATOR_CONTRACT_SHA256,
    FINAL_SCENARIO_COUNTS,
    FINAL_SUITE_SCHEMA,
    MAX_PUBLICATION_BYTES,
    MAX_RESIDENT_BYTES,
    MAX_STORAGE_BYTES,
    MAX_WALL_SECONDS,
    REVIEW_SCHEMA,
    RUNTIME_PLATFORM,
    RUNTIME_PYTHON,
    RUNTIME_THREADS,
    RUNTIME_TORCH,
    FlightResourceGuard,
    admission_candidate,
    build_split_manifest,
    prospective_declaration,
    resource_ceilings,
    review_binding,
    runtime_identity,
    validate_custody_record,
)


DEV_SHA = "8b68faa6d3bf6172d0ba1dcf6d2752aae6ff9e83d521cfe33cf537cbbad9e5e7"
TUNING_SHA = "0295a669549bc7cf852fd12934d85506ef1f55ceec07340531f801084e08b3df"


def source_fixture():
    return {
        "learner_revision": "a" * 40,
        "learner_source_sha256": "b" * 64,
    }


def custody_fixture():
    return {
        "schema": CUSTODY_SCHEMA,
        "suite_schema": FINAL_SUITE_SCHEMA,
        "suite_sha256": "c" * 64,
        "suite_bytes": 123456,
        "case_count": 60,
        "family_composition": dict(FINAL_COMPOSITION),
        "scenario_composition": dict(FINAL_SCENARIO_COUNTS),
        "host_revision": "9935d5696721d2c72a8273ea70454ab0c5b11361",
        "host_build_sha256": "55e5ba334b6683adf2ba0d8fc93fb7defba109cd4a319d2f366a2a86e4dfdf84",
        "evaluator_contract_sha256": FINAL_EVALUATOR_CONTRACT_SHA256,
        "teacher_final_suite_sha256": "4bd18d728581f64fa770c4dc29f765eff956f61999b96d03d8e378d54efc3401",
        "teacher_final_reused": False,
        "custodian_identity": "external-custodian-a",
        "reviewer_identity": "external-private-reviewer-b",
        "custodian_independent": True,
        "reviewer_independent": True,
        "repository_payload_present": False,
        "executor_opened_payload": False,
        "private_disjointness_verified": True,
        "private_feasibility_verified": True,
    }


def review_fixture(binding):
    return {
        "schema": REVIEW_SCHEMA,
        "review_binding_sha256": binding["review_binding_sha256"],
        "accepted": True,
        "reviewer_identity": "independent-execution-reviewer-c",
        "independent": True,
        "source_clean": True,
        "runtime_accepted": True,
        "resource_policy_accepted": True,
        "stage_accounting_accepted": True,
        "custody_accepted": True,
        "final_access_unopened": True,
        "training_not_executed": True,
        "findings": [],
    }


class SplitAndCustodyTests(unittest.TestCase):
    def test_public_split_manifests_are_exact_and_frozen(self):
        development = build_split_manifest("development")
        tuning = build_split_manifest("tuning")
        self.assertEqual(development["case_count"], 44)
        self.assertEqual(development["max_ticks"], 68400)
        self.assertEqual(development["max_decisions"], 11400)
        self.assertEqual(tuning["case_count"], 40)
        self.assertEqual(tuning["max_ticks"], 67200)
        self.assertEqual(tuning["max_decisions"], 11200)
        self.assertEqual(sha256_json(development), DEV_SHA)
        self.assertEqual(sha256_json(tuning), TUNING_SHA)

    def test_custody_requires_external_unopened_non_teacher_final(self):
        valid = validate_custody_record(custody_fixture())
        self.assertEqual(valid["case_count"], 60)
        self.assertFalse(valid["executor_opened_payload"])
        self.assertFalse(valid["teacher_final_reused"])

        changed = custody_fixture()
        changed["teacher_final_reused"] = True
        with self.assertRaisesRegex(ContractError, "teacher final"):
            validate_custody_record(changed)

        changed = custody_fixture()
        changed["executor_opened_payload"] = True
        with self.assertRaisesRegex(ContractError, "unopened"):
            validate_custody_record(changed)

        changed = custody_fixture()
        changed["reviewer_identity"] = changed["custodian_identity"]
        with self.assertRaisesRegex(ContractError, "distinct"):
            validate_custody_record(changed)

    def test_candidate_never_fabricates_missing_external_evidence(self):
        packet = admission_candidate(
            source=source_fixture(),
            development_sha256=DEV_SHA,
            tuning_sha256=TUNING_SHA,
        )
        self.assertEqual(packet["status"], "READY_FOR_EXTERNAL_CUSTODY")
        self.assertFalse(packet["admitted"])
        self.assertFalse(packet["final_access"])
        self.assertFalse(packet["training_executed"])
        self.assertEqual(packet["reasons"], ["learner_final_custody_missing"])

        packet = admission_candidate(
            source=source_fixture(),
            development_sha256=DEV_SHA,
            tuning_sha256=TUNING_SHA,
            custody=custody_fixture(),
        )
        self.assertEqual(
            packet["status"], "READY_FOR_INDEPENDENT_EXECUTION_REVIEW")
        self.assertEqual(
            packet["reasons"], ["independent_execution_review_missing"])
        self.assertIsNotNone(packet["review_binding"])
        self.assertIsNone(packet["declaration"])


class ProspectiveDeclarationTests(unittest.TestCase):
    def admitted(self):
        binding = review_binding(
            source=source_fixture(),
            development_sha256=DEV_SHA,
            tuning_sha256=TUNING_SHA,
            custody=custody_fixture(),
        )
        declaration, evidence, _ = prospective_declaration(
            source=source_fixture(),
            development_sha256=DEV_SHA,
            tuning_sha256=TUNING_SHA,
            custody=custody_fixture(),
            independent_review=review_fixture(binding),
        )
        return declaration, evidence

    def test_independent_review_closes_only_the_review_gate(self):
        declaration, evidence = self.admitted()
        value = declaration.to_dict()
        self.assertEqual(value["runtime"], {
            "device": "cpu",
            "platform": RUNTIME_PLATFORM,
            "python": RUNTIME_PYTHON,
            "torch": RUNTIME_TORCH,
            "threads": RUNTIME_THREADS,
        })
        self.assertEqual(value["splits"]["development_sha256"], DEV_SHA)
        self.assertEqual(value["splits"]["tuning_sha256"], TUNING_SHA)
        self.assertEqual(value["splits"]["final_commitment_sha256"], "c" * 64)
        self.assertIsNotNone(value["splits"]["final_custody_sha256"])
        self.assertFalse(value["splits"]["opened_final"])
        self.assertEqual(value["resource_ceilings"], resource_ceilings())
        gate = execution_gate(declaration, current_evidence=evidence)
        self.assertTrue(gate["admitted"])
        self.assertTrue(gate["training_enabled"])
        self.assertFalse(gate["final_access"])

    def test_production_declaration_requires_live_resource_guard(self):
        declaration, evidence = self.admitted()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ContractError, "resource guard"):
                FlightAccountingLedger.create(
                    root / "unguarded.jsonl",
                    declaration,
                    attempt_id="unguarded",
                    current_evidence=evidence,
                )

            guard = FlightResourceGuard(
                root,
                resource_ceilings(),
                clock=lambda: 0.0,
                rss_reader=lambda: 1024,
                storage_reader=lambda _root: 0,
            ).start()
            try:
                ledger = FlightAccountingLedger.create(
                    root / "guarded.jsonl",
                    declaration,
                    attempt_id="guarded",
                    current_evidence=evidence,
                    resource_guard=guard,
                )
                ledger.close("fixture_complete")
            finally:
                guard.close()


class ResourceGuardTests(unittest.TestCase):
    def guard(self, root, clock, rss=1024, storage=0):
        return FlightResourceGuard(
            Path(root),
            resource_ceilings(),
            clock=lambda: clock[0],
            rss_reader=lambda: rss[0] if isinstance(rss, list) else rss,
            storage_reader=lambda _root: (
                storage[0] if isinstance(storage, list) else storage
            ),
        )

    def test_resource_policy_has_finite_whole_attempt_caps(self):
        ceilings = resource_ceilings()
        self.assertEqual(ceilings["maximum_wall_seconds"], MAX_WALL_SECONDS)
        self.assertEqual(ceilings["maximum_resident_bytes"], MAX_RESIDENT_BYTES)
        self.assertEqual(ceilings["maximum_storage_bytes"], MAX_STORAGE_BYTES)
        self.assertEqual(
            ceilings["maximum_publication_bytes"], MAX_PUBLICATION_BYTES)
        self.assertGreater(MAX_WALL_SECONDS, 0)
        self.assertGreater(MAX_RESIDENT_BYTES, 0)

    def test_wall_rss_storage_and_publication_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            clock = [0.0]
            guard = self.guard(temporary, clock)
            guard.checkpoint("initial")
            clock[0] = MAX_WALL_SECONDS + 0.001
            with self.assertRaisesRegex(ContractError, "maximum_wall_seconds"):
                guard.checkpoint("late")

            clock = [0.0]
            rss = [MAX_RESIDENT_BYTES + 1]
            guard = self.guard(temporary, clock, rss=rss)
            with self.assertRaisesRegex(ContractError, "maximum_resident_bytes"):
                guard.checkpoint("rss")

            storage = [MAX_STORAGE_BYTES + 1]
            guard = self.guard(temporary, [0.0], storage=storage)
            with self.assertRaisesRegex(ContractError, "maximum_storage_bytes"):
                guard.checkpoint("storage")

            guard = self.guard(temporary, [0.0])
            with self.assertRaisesRegex(
                ContractError, "maximum_publication_bytes"
            ):
                guard.charge_publication_bytes(MAX_PUBLICATION_BYTES + 1)


@unittest.skipUnless(
    sys.platform.startswith("linux"),
    "admitted 4LY-004 runtime is Linux only",
)
class RuntimeIdentityTests(unittest.TestCase):
    def test_runtime_identity_accepts_only_pinned_cpu_runtime(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch unavailable")
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            identity = runtime_identity(enforce=True)
            self.assertEqual(identity["python"], RUNTIME_PYTHON)
            self.assertEqual(identity["torch"], RUNTIME_TORCH)
            self.assertEqual(identity["threads"], 1)
            self.assertFalse(torch.cuda.is_available())
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()
