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
    ALLOCATION_ID, AUTHORIZATION_REF, RETIREMENT_CONTRACT, comparison_order,
    make_ppo_case_stream, validate_config_v3, validate_splits_v3,
    validate_declaration_reviews, validate_resource_admission,
)
from nca.train.jolt_execution_contract import validate_v3_allocation_identity

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/learning/jolt-learn-003-v1.json").read_text())


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


if __name__ == "__main__":
    unittest.main()
