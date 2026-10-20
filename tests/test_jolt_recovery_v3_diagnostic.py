"""V3 source/allocation admission and unchanged teacher rules; no physics."""

import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from nca.contracts import ContractError, sha256_json
from nca.data import read_episode
from nca.evaluation import jolt_arrival as arrival
from nca.evaluation.jolt_recovery_acceptance import (
    CANDIDATE, CANDIDATE_V2, CANDIDATE_V3, acceptance_specification,
    assess_teacher,
)
from test_jolt_decision_refusals import SyntheticController, SyntheticOwner
from test_jolt_recovery_acceptance import fixture as acceptance_fixture


ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "configs/baselines/jolt-recovery-diagnostic-v3.json"


class RecoveryV3DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.declaration = json.loads(DECLARATION.read_bytes())

    def assert_rejected_before_host(self, value):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "changed.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(arrival, "GodotHost") as host, \
                    self.assertRaises(ContractError):
                arrival.run_diagnostic(root=ROOT, checkout=root / "unused-host",
                    godot=root / "unused-godot", declaration_path=path,
                    output=root / "must-not-start", expected_revision="a" * 40,
                    expected_declaration_sha256=sha256_json(value))
            host.assert_not_called()
            self.assertFalse((root / "must-not-start").exists())

    def test_declared_specification_cases_physics_and_numeric_gates_are_bound(self):
        declaration = arrival.read_declaration(DECLARATION,
            expected_sha256=sha256_json(self.declaration))
        earlier = json.loads((ROOT / "configs/baselines/jolt-recovery-diagnostic-v2.json").read_bytes())
        self.assertEqual(declaration["case_source"], earlier["case_source"])
        self.assertEqual(declaration["physics"], earlier["physics"])
        self.assertEqual(declaration["goal"], earlier["goal"])
        self.assertEqual(declaration["families"], list(arrival.RECOVERY_V3_FAMILIES))
        for family, specification in declaration["candidate_specifications"].items():
            self.assertEqual(arrival._controller(family).specification(), specification)
        strip_identity = lambda value: {key: val for key, val in value.items()
                                       if key not in ("schema", "candidate")}
        self.assertEqual(strip_identity(acceptance_specification(CANDIDATE_V2)),
                         strip_identity(declaration["teacher_acceptance"]))
        original = json.loads((ROOT / declaration["case_source"]["path"]).read_bytes())
        cases = arrival.diagnostic_cases(original)
        self.assertEqual(len(cases), 32)
        self.assertEqual(sum(case["origin"] == "previously_open_evaluation" for case in cases), 8)

    def test_old_declaration_and_controller_identities_stay_replayable(self):
        for version, canonical in (
            (1, "2e5d4e3c554ec2382d10a95d347c1f933252050f2a18dff320711c2cc3c7fff7"),
            (2, "d7cc13335b453a755e9b7d950ac52d982aff553c025cbe0fbd47b81b5b0729eb"),
        ):
            path = ROOT / f"configs/baselines/jolt-recovery-diagnostic-v{version}.json"
            value = arrival.read_declaration(path, expected_sha256=canonical)
            family = f"jolt-recovery-v{version}"
            self.assertEqual(arrival._controller(family).specification(),
                             value["candidate_specifications"][family])
        for path, digest in self.declaration["preserved_sources"].items():
            if path.startswith("src/nca/controllers/"):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)

    def test_missing_or_inconsistent_v3_ceilings_refuse_before_runtime(self):
        for key in self.declaration["budget"]:
            with self.subTest(missing=key):
                value = copy.deepcopy(self.declaration)
                value["budget"].pop(key)
                self.assert_rejected_before_host(value)
        for key, changed in (("maximum_episodes", 95), ("maximum_decisions", 3839),
                             ("maximum_requested_ticks", 23046), ("maximum_host_requests", 4227),
                             ("maximum_wall_seconds", 901), ("maximum_attempts", 2),
                             ("maximum_host_starts", True), ("learned_policy_calls", 1),
                             ("optimizer_updates", 1), ("checkpoint_executions", 1),
                             ("automatic_retry", True), ("automatic_budget_extension", True)):
            with self.subTest(key=key, value=changed):
                value = copy.deepcopy(self.declaration)
                value["budget"][key] = changed
                self.assert_rejected_before_host(value)

    def test_v3_cannot_change_the_teacher_rule_or_relabel_earlier_families(self):
        for key, changed in (("minimum_extra_arrivals", 0), ("minimum_extra_arrivals", True),
                             ("minimum_historical_rescues", 2),
                             ("maximum_regressions", 2), ("required_unsupported_rescue", "other"),
                             ("minimum_toward_speed_ratio", .6), ("maximum_requested_primary_edges", 3)):
            with self.subTest(key=key):
                value = copy.deepcopy(self.declaration)
                value["teacher_acceptance"][key] = changed
                self.assert_rejected_before_host(value)
        for schema, families in (("nca-jolt-recovery-diagnostic/v3", arrival.RECOVERY_V2_FAMILIES),
                                 ("nca-jolt-recovery-diagnostic/v2", arrival.RECOVERY_V3_FAMILIES)):
            value = copy.deepcopy(self.declaration)
            value.update(schema=schema, families=list(families))
            self.assert_rejected_before_host(value)

    def test_v3_rejects_changed_preserved_source_profile_and_candidate(self):
        for section, key in (("preserved_sources", "src/nca/controllers/jolt_recovery_v2.py"),
                             ("preserved_sources", "src/nca/controllers/recovery_supervisor.py"),
                             ("physics", "profile_sha256"), ("physics", "host_revision")):
            with self.subTest(section=section, key=key):
                value = copy.deepcopy(self.declaration)
                value[section][key] = "f" * len(value[section][key])
                self.assert_rejected_before_host(value)
        value = copy.deepcopy(self.declaration)
        value["candidate_specifications"][CANDIDATE_V3] = {}
        self.assert_rejected_before_host(value)

    def test_v3_acceptance_does_not_treat_v2_rows_as_v3_evidence(self):
        rows = acceptance_fixture()
        for row in rows:
            if row["family"] == CANDIDATE:
                row["family"] = CANDIDATE_V2
        with self.assertRaises(ContractError):
            assess_teacher(rows, pulse_authority_verified=True, candidate_family=CANDIDATE_V3)
        for row in rows:
            if row["family"] == CANDIDATE_V2:
                row["family"] = CANDIDATE_V3
        result = assess_teacher(rows, pulse_authority_verified=True, candidate_family=CANDIDATE_V3)
        self.assertTrue(result["teacher_accepted"])
        self.assertEqual(result["schema"], "nca-jolt-recovery-behavior-review/v3")
        self.assertFalse(result["learned_skill_qualified"])
        self.assertFalse(result["morphology_admitted"])

    def test_v3_never_dispatches_an_invalid_supervisor_lease(self):
        for lease in (0, 13, True):
            with self.subTest(lease=lease), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                owner = SyntheticOwner()
                controller = SyntheticController()
                controller.diagnostic_state = lambda: {"tick": controller.tick,
                    "action_lease_ticks": lease, "supervisor_intervention": False,
                    "supervisor_reason": None}
                case = {"id": "synthetic-invalid-lease", "family": "synthetic", "host_seed": 1,
                    "spawn": owner.view["position"],
                    "target": [owner.view["position"][0] + 100., owner.view["position"][1]],
                    "origin": "previously_open_training"}
                budget = arrival.DiagnosticBudget(self.declaration["budget"], io.BytesIO())
                metered = arrival.MeteredHost(owner, budget)
                with mock.patch.object(arrival, "_controller", return_value=controller), \
                        mock.patch.object(arrival, "GodotHost", side_effect=AssertionError("no physics")), \
                        mock.patch("nca.data.capture.source_identity",
                                   return_value=("b" * 40, {}, sha256_json({}))):
                    row = arrival._capture(metered, budget, case, CANDIDATE_V3, output,
                        {"profile_sha256": sha256_json(owner.profile)}, self.declaration, ROOT)
                self.assertEqual(owner.steps, 0)
                self.assertEqual(row["error"]["code"], "arrival_supervisor_lease")
                self.assertTrue(row["error_cleanup"]["confirmed"])
                episode = read_episode(output / row["artifact"])
                self.assertEqual(episode.decisions, 0)
                failures = episode.evaluation["controller_decision_failures"]
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0]["disposition"], "refused_before_transport")
                self.assertEqual(failures[0]["owner_requests_before"], failures[0]["owner_requests_after"])


if __name__ == "__main__":
    unittest.main()
