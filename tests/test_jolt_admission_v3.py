"""Independent negative controls for the fresh allocation's static admission."""
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from nca.contracts import ContractError, sha256_json
from nca.train.jolt_admission import AdmissionError, load_config, validate_config
from nca.train.jolt_admission_v3 import (
    ALLOCATION_ID, AUTHORIZATION_REF, CLOCK_AUTHORIZATION_REF, RETIREMENT_CONTRACT, comparison_order,
    make_ppo_case_stream, validate_config_v3, validate_splits_v3,
    validate_declaration_reviews, validate_resource_admission,
)
from nca.train.jolt_budget_v3 import BudgetLedgerV3, reservation_records, _header
from nca.train.jolt_storage_v3 import storage_bounds_v3
from nca.train.jolt_execution_contract import validate_v3_allocation_identity

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/learning/jolt-learn-003-v1.json").read_text())


def unlimited_budget():
    value = deepcopy(config()["budget"])
    for name in ("maximum_owner_transport_seconds", "maximum_wall_seconds", "cleanup_reserve_seconds",
                 "executor_verification_reserve_seconds", "executor_finalization_reserve_seconds"):
        value[name] = None
    return value


class AdmissionV3Tests(unittest.TestCase):
    def test_v3_registry_discovery_and_cross_version_refusal_do_not_import_torch(self):
        script = """
import sys
from nca.builtins import builtin_registry
from nca.contracts import ContractError
registry = builtin_registry()
assert registry.describe('trainer','learn-jolt',2).version == 2
assert 'safe-retirement' in registry.describe('trainer','learn-jolt',3).capabilities
try:
    registry.create('trainer','learn-jolt',3,execution_version=2)
except ContractError as error:
    assert error.code == 'jolt_execution_version'
else:
    raise AssertionError('cross-version execution was admitted')
assert 'torch' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_static_configuration_binds_complete_opened_lineage_and_unopened_proof(self):
        value = load_config(ROOT / "configs/learning/jolt-learn-003-v1.json", root=ROOT)
        self.assertEqual(len(value["budget"]["counters"]), 39)
        self.assertEqual(sum(row["decisions"] for row in value["budget"]["stages"].values()), 18944)
        opened = json.loads((ROOT / value["opened_inventory"]["path"]).read_text())
        self.assertEqual(len(opened["learn_jolt_002_opened_cases"]), 68)
        self.assertTrue(all(case["contact_sha256"] in opened["opened_contact_sha256"]
                            for case in value["cases"]["teacher"]))
        self.assertTrue(all(case["contact_sha256"] not in opened["opened_contact_sha256"]
                            for case in value["cases"]["sealed"]))

    def test_full_stream_matches_continuous_rng_without_update_reset(self):
        value = config()
        stream = make_ppo_case_stream(value["cases"]["teacher"])
        ids = [case["id"] for case in value["cases"]["teacher"]]
        for seed in value["training_seeds"]:
            rng = random.Random(seed + 300000)
            expected = [rng.choice(ids) for _ in range(2048)]
            self.assertEqual(stream["seeds"][str(seed)], expected)
        self.assertEqual(sum(len(rows) for rows in stream["seeds"].values()), 6144)
        changed = deepcopy(value)
        changed["ppo_case_stream"]["seeds"]["190103"][256] = "outcome-chosen-case"
        with self.assertRaisesRegex(AdmissionError, "case stream"):
            validate_config(changed)

    def test_comparison_order_is_case_major_and_preserves_all_controller_pairs(self):
        value = config()
        order = comparison_order(value["cases"])
        self.assertEqual((len(order["open"]), len(order["sealed"])), (32, 224))
        self.assertEqual([row["controller_id"] for row in order["sealed"][:2]], ["authored", "idle"])
        value["comparison_order"]["sealed"].reverse()
        with self.assertRaisesRegex(AdmissionError, "comparator/case order"):
            validate_config(value)

    def test_fixed_learning_reward_teacher_and_gate_changes_are_rejected(self):
        for section, key, replacement in (("ppo", "learning_rate", .0004),
                ("bc", "epochs", 41), ("goal", "deadline_ticks", 288),
                ("learned_gate", "minimum_mean_success", .5)):
            with self.subTest(section=section, key=key):
                value = config()
                value[section][key] = replacement
                with self.assertRaisesRegex(AdmissionError, "fixed learning"):
                    validate_config(value)

    def test_no_missing_changed_or_boolean_operation_ceiling(self):
        for counter in config()["budget"]["counters"]:
            value = config()
            value["budget"]["counters"][counter] += 1
            with self.subTest(counter=counter), self.assertRaises(ContractError):
                validate_config(value)
        value = config()
        value["budget"]["counters"]["attempts"] = True
        with self.assertRaises(ContractError):
            validate_config(value)

    def test_retirement_has_narrow_reason_zero_bootstrap_and_no_new_penalty(self):
        value = config()
        self.assertEqual(value["retirement_contract"], RETIREMENT_CONTRACT)
        for key, replacement in (("eligible_reason", "unknown"), ("emergency_consumed_ticks", 54),
                                 ("new_terminal_penalty", True), ("success", 0)):
            value = config()
            value["retirement_contract"][key] = replacement
            with self.subTest(key=key), self.assertRaisesRegex(AdmissionError, "retirement semantics"):
                validate_config(value)

    def test_old_consumed_declaration_cannot_enter_v3(self):
        old = json.loads((ROOT / "docs/evidence/jolt-execution-learn-002-20261021.json").read_text())
        with self.assertRaises(ContractError):
            validate_v3_allocation_identity(old)
        current = {"schema": "nca-learn-jolt-003-execution/v3", "allocation_id": ALLOCATION_ID,
                   "authorization_ref": AUTHORIZATION_REF}
        with self.assertRaises(ContractError):
            validate_v3_allocation_identity(current, consumed_allocation_ids=[ALLOCATION_ID])

    def test_omitting_original_training_cases_cannot_manufacture_sealed_freshness(self):
        value = config()
        split = json.loads((ROOT / value["split"]["path"]).read_text())
        opened = json.loads((ROOT / value["opened_inventory"]["path"]).read_text())
        opened["opened_contact_sha256"].remove(value["cases"]["teacher"][0]["contact_sha256"])
        with self.assertRaisesRegex(AdmissionError, "complete opened inventory"):
            validate_splits_v3(ROOT, split, opened)

    def test_old_planning_flags_and_unresolved_feasibility_cannot_admit(self):
        with self.assertRaises(AdmissionError):
            validate_declaration_reviews({"independent_reviews": []})
        with self.assertRaisesRegex(AdmissionError, "feasibility is unresolved"):
            validate_resource_admission({"resource_admission": {
                "schema": "nca-jolt-003-resource-admission/v1", "accepted": False,
                "unresolved_blocking_findings": ["reader/time envelope unresolved"]}})


class NoWallClockAdmissionV3Tests(unittest.TestCase):
    def test_configuration_versions_select_explicit_clock_modes_without_changing_physics(self):
        old = config()
        validate_config_v3(old)
        current = deepcopy(old)
        current.update(schema="nca-jolt-learn-003/v2", id="learn-jolt-003-v2", budget=unlimited_budget(),
                       clock_authorization_ref=CLOCK_AUTHORIZATION_REF)
        validate_config_v3(current)
        self.assertEqual(current["goal"], old["goal"])
        self.assertEqual(current["budget"]["counters"], old["budget"]["counters"])
        self.assertEqual(current["budget"]["stages"], old["budget"]["stages"])
        for value in ({**old, "budget": current["budget"]}, {**current, "budget": old["budget"]}):
            with self.assertRaisesRegex(AdmissionError, "wall-clock mode"):
                validate_config_v3(value)
        for authorization in (None, AUTHORIZATION_REF, "unreviewed timing change"):
            with self.subTest(authorization=authorization), self.assertRaisesRegex(AdmissionError, "wall-clock revision"):
                validate_config_v3({**current, "clock_authorization_ref": authorization})

    def test_published_v2_configuration_loads_with_unchanged_prerequisite_and_lineage(self):
        value = load_config(ROOT / "configs/learning/jolt-learn-003-v2.json", root=ROOT)
        self.assertEqual(value["schema"], "nca-jolt-learn-003/v2")
        self.assertEqual(value["id"], "learn-jolt-003-v2")
        self.assertEqual(value["authorization_ref"], AUTHORIZATION_REF)
        self.assertEqual(value["clock_authorization_ref"], CLOCK_AUTHORIZATION_REF)
        self.assertEqual(value["budget"], unlimited_budget())
        for field in ("prerequisite", "opened_inventory", "split", "cases", "training_seeds"):
            self.assertEqual(value[field], config()[field])

    def test_late_unlimited_ledger_preserves_original_elapsed_phases_and_all_counters(self):
        budget, now = unlimited_budget(), [8000.]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ledger = BudgetLedgerV3(budget, started=0., spool_path=root / "live.jsonl",
                                   monotonic=lambda: now[0])
            try:
                self.assertIsNone(ledger.seconds)
                self.assertIsNone(ledger.reserve_seconds)
                self.assertIsNone(ledger.deadline)
                self.assertIsNone(ledger.active_deadline)
                self.assertIsNone(ledger.remaining_seconds())
                ledger.reserve("attempts", allowance_seconds=None)
                before = ledger.snapshot()
                with self.assertRaisesRegex(ContractError, "counter exhausted"):
                    ledger.reserve("attempts")
                self.assertEqual(ledger.snapshot(), before)
                with self.assertRaisesRegex(ContractError, "declared order"):
                    ledger.enter_phase("audit")
                now[0] = 9000.
                ledger.enter_phase("closure")
                with self.assertRaisesRegex(ContractError, "native revocation"):
                    ledger.reserve("bc_forward", cleanup=True)
                ledger.reserve("owner_close", cleanup=True, allowance_seconds=None)
                now[0] = 10000.
                ledger.enter_phase("audit")
                ledger.reserve("audit_forward", cleanup=True)
                now[0] = 11000.
                ledger.enter_phase("publication")
                target = root / "reservations.json"
                ledger.publish_reservations(target)
                final = ledger.close()
                self.assertIsNone(final["maximum_wall_seconds"])
                self.assertIsNone(final["cleanup_reserve_seconds"])
                self.assertEqual(final["wall_seconds"], 11000.)
                self.assertEqual([row["elapsed_seconds"] for row in final["phase_history"]],
                                 [8000., 9000., 10000., 11000.])
                self.assertEqual(len(final["counts"]), 39)
                self.assertEqual(final["counts"]["attempts"], 1)
                self.assertTrue(target.read_bytes().startswith(_header(3)))
                rows = list(reservation_records(target, budget=budget))
                self.assertEqual([row["elapsed_seconds"] for row in rows], [8000., 9000., 10000.])
                self.assertEqual(sha256_json(rows), final["reservations_sha256"])
                for selected in (None, config()["budget"]):
                    with self.assertRaisesRegex(ContractError, "original time"):
                        list(reservation_records(target, budget=selected))
            finally:
                ledger._stream.close()

    def test_unlimited_reader_still_rejects_invalid_original_elapsed_values(self):
        budget = unlimited_budget()
        for elapsed in (-1., True, None, float("nan"), float("inf")):
            with self.subTest(elapsed=elapsed), tempfile.TemporaryDirectory() as folder:
                row = {"index": 0, "elapsed_seconds": elapsed, "stage": None, "cleanup": False,
                       "increments": {"attempts": 1}, "phase": "active"}
                path = Path(folder) / "invalid.json"
                path.write_bytes(_header(1) + json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                                 + b"\n]}\n")
                with self.assertRaises(ContractError):
                    list(reservation_records(path, budget=budget))

    def resource_declaration(self, *, unlimited):
        budget = unlimited_budget() if unlimited else config()["budget"]
        declaration = {"source_sha256": "a"*64, "config_sha256": "b"*64, "budget": budget,
                       "host_identity": {"synthetic": True}, "runtime": {"synthetic": True}}
        gates = {name: True for name in ("canonical_reservation_maximum", "reader_limits",
                 "journal_aggregate_bytes", "total_artifact_bytes", "total_file_count",
                 "peak_resident_memory", "available_disk", "original_clock_phase_enforcement",
                 "complete_pipeline_time_feasibility")}
        if unlimited:
            gates.update(original_clock_phase_enforcement=None, complete_pipeline_time_feasibility=None)
        bounds = storage_bounds_v3()
        declaration["resource_admission"] = {
            "schema": "nca-jolt-003-resource-admission/v1", "accepted": True,
            "unresolved_blocking_findings": [], "source_sha256": declaration["source_sha256"],
            "config_sha256": declaration["config_sha256"],
            "host_identity_sha256": sha256_json(declaration["host_identity"]),
            "runtime_sha256": sha256_json(declaration["runtime"]), "gates": gates,
            "serialization": bounds,
            "required_free_disk_bytes": bounds["required_disk_bytes_including_1_gib_reserve"]}
        return declaration

    def test_resource_admission_marks_time_inapplicable_only_for_explicit_unlimited_budget(self):
        for unlimited in (False, True):
            value = self.resource_declaration(unlimited=unlimited)
            validate_resource_admission(value)
            for name in ("original_clock_phase_enforcement", "complete_pipeline_time_feasibility"):
                changed = deepcopy(value)
                changed["resource_admission"]["gates"][name] = True if unlimited else None
                with self.subTest(unlimited=unlimited, name=name), self.assertRaisesRegex(AdmissionError, "wall-clock mode"):
                    validate_resource_admission(changed)
        value = self.resource_declaration(unlimited=True)
        value.pop("budget")
        with self.assertRaisesRegex(AdmissionError, "wall-clock mode"):
            validate_resource_admission(value)

    def test_unlimited_time_does_not_waive_memory_storage_or_source_admission(self):
        original = self.resource_declaration(unlimited=True)
        for name in ("canonical_reservation_maximum", "reader_limits", "journal_aggregate_bytes",
                     "total_artifact_bytes", "total_file_count", "peak_resident_memory", "available_disk"):
            for replacement in (None, False, 1):
                changed = deepcopy(original)
                changed["resource_admission"]["gates"][name] = replacement
                with self.subTest(name=name, replacement=replacement), self.assertRaises(AdmissionError):
                    validate_resource_admission(changed)
        changed = deepcopy(original)
        changed["resource_admission"]["source_sha256"] = "c"*64
        with self.assertRaisesRegex(AdmissionError, "different execution"):
            validate_resource_admission(changed)
        changed = deepcopy(original)
        changed["budget"]["maximum_owner_transport_seconds"] = 15
        with self.assertRaises(ContractError):
            validate_resource_admission(changed)


if __name__ == "__main__":
    unittest.main()
