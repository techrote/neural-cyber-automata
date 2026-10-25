"""Software-only failure controls for 4LY-004 accounting preparation."""

from pathlib import Path
import tempfile
import unittest

from nca.adapters.flight import (
    CONTROLLER_SHA256 as ADAPTER_CONTROLLER_SHA256,
    DEVELOPMENT_SEEDS as ADAPTER_DEVELOPMENT_SEEDS,
    SCENARIO_MAX_TICKS as ADAPTER_SCENARIO_MAX_TICKS,
    TRAINER_PROFILE_SHA256 as ADAPTER_TRAINER_PROFILE_SHA256,
    TEACHER_SHA256 as ADAPTER_TEACHER_SHA256,
    TUNING_SEEDS as ADAPTER_TUNING_SEEDS,
)
from nca.contracts import ContractError, sha256_json
from nca.train.flight_accounting import (
    ACCEPTED_DEPENDENCY_EVIDENCE,
    DEPENDENCIES,
    DESIGN_WORK_MAXIMA,
    ACTUAL_HOST_CONFORMANCE_EVIDENCE_SHA256,
    CONTROLLER_SHA256,
    HOST_BUILD_SHA256,
    HOST_CONTRACT_HEAD,
    SCENARIO_MAX_TICKS,
    STAGE_CEILINGS,
    TRAINER_PROFILE_SHA256,
    TEACHER_SHA256,
    FlightAccountingLedger,
    FlightDeclaration,
    execution_gate,
    preparation_declaration,
    required_measurements,
)


def admitted_fixture():
    """Complete synthetic identities for ledger mechanics only; never experiment evidence."""
    value = preparation_declaration().to_dict()
    value["status"] = "prospective"
    value["identities"] = {
        "learner_revision": "a" * 40,
        "learner_source_sha256": sha256_json(["fixture", "learner"]),
        "host_revision": HOST_CONTRACT_HEAD,
        "host_build_sha256": sha256_json(["fixture", "host-build"]),
        "profile_sha256": TRAINER_PROFILE_SHA256,
        "controller_sha256": CONTROLLER_SHA256,
        "teacher_sha256": TEACHER_SHA256,
    }
    value["splits"].update({
        "development_sha256": sha256_json(["fixture", "development"]),
        "tuning_sha256": sha256_json(["fixture", "tuning"]),
        "final_commitment_sha256": sha256_json(["fixture", "sealed-final-commitment"]),
    })
    value["runtime"].update({
        "platform": "fixture-linux-x86_64",
        "python": "fixture-python-3.12",
        "torch": "fixture-torch-cpu",
        "threads": 1,
    })
    value["work_ceilings"].update({
        "host_calls": DESIGN_WORK_MAXIMA["host_calls"],
        "startup_operations": DESIGN_WORK_MAXIMA["startup_operations"],
        "cleanup_operations": DESIGN_WORK_MAXIMA["cleanup_operations"],
    })
    value["resource_ceilings"].update({
        "maximum_wall_seconds": 60.0,
        "maximum_resident_bytes": 1024 * 1024,
        "maximum_storage_bytes": 16 * 1024 * 1024,
        "maximum_publication_bytes": 4 * 1024 * 1024,
    })
    evidence = {}
    for name in DEPENDENCIES:
        digest = sha256_json(["fixture-evidence", name])
        value["dependencies"][name] = {"status": "accepted", "evidence_sha256": digest}
        evidence[name] = digest
    return FlightDeclaration.from_mapping(value), evidence


class DeclarationGateTests(unittest.TestCase):
    def test_preparation_template_is_immutable_and_fail_closed(self):
        declaration = preparation_declaration()
        changed = declaration.to_dict()
        changed["seeds"][0] = 1
        self.assertEqual(declaration.to_dict()["seeds"], [74011, 74013, 74017])
        value = declaration.to_dict()
        self.assertEqual(
            {name: value["work_ceilings"][name] for name in DESIGN_WORK_MAXIMA},
            DESIGN_WORK_MAXIMA,
        )
        self.assertEqual(value["stage_ceilings"], STAGE_CEILINGS)
        self.assertEqual(value["source_contract"]["revision"], HOST_CONTRACT_HEAD)
        self.assertEqual(
            value["source_contract"]["qualification_status"],
            "accepted_lease_repair_host_with_qualified_teacher_lineage")
        self.assertEqual(value["identities"]["host_revision"], HOST_CONTRACT_HEAD)
        self.assertEqual(value["identities"]["host_build_sha256"], HOST_BUILD_SHA256)
        self.assertEqual(value["identities"]["teacher_sha256"], TEACHER_SHA256)
        self.assertFalse(
            value["source_contract"]["teacher_final_may_be_reused_as_learner_final"]
        )
        self.assertEqual(value["dependencies"]["host_4ly003"]["status"], "accepted")
        self.assertEqual(value["dependencies"]["teacher_qualification"]["status"], "accepted")
        self.assertEqual(value["dependencies"]["actual_host_conformance"]["status"], "accepted")
        self.assertEqual(
            value["dependencies"]["actual_host_conformance"]["evidence_sha256"],
            ACTUAL_HOST_CONFORMANCE_EVIDENCE_SHA256,
        )
        self.assertEqual(value["dependencies"]["throughput_measurement"]["status"], "accepted")
        self.assertEqual(value["scenario"]["scenario_max_ticks"], SCENARIO_MAX_TICKS)
        self.assertEqual(value["scenario"]["scenario_max_ticks"]["slalom"], 2400)
        self.assertIn("development_sha256", value["splits"])
        self.assertNotIn("training_sha256", value["splits"])
        self.assertIsNone(value["runtime"]["threads"])
        self.assertEqual(value["work_ceilings"]["host_calls"], 246_765)
        self.assertEqual(value["work_ceilings"]["startup_operations"], 1)
        self.assertEqual(value["work_ceilings"]["cleanup_operations"], 1)

        gate = execution_gate(
            declaration, current_evidence=ACCEPTED_DEPENDENCY_EVIDENCE
        )
        self.assertFalse(gate["admitted"])
        self.assertFalse(gate["training_enabled"])
        self.assertFalse(gate["final_access"])
        self.assertIn("declaration_not_prospective", gate["reasons"])
        self.assertIn("runtime_unresolved:threads", gate["reasons"])
        self.assertIn("resource_ceiling_unmeasured:maximum_wall_seconds", gate["reasons"])
        self.assertIn("dependency_unfinished:stage_accounting_integration", gate["reasons"])
        self.assertGreaterEqual(len(required_measurements()), 10)

    def test_adapter_contract_matches_accounting_source_binding(self):
        self.assertEqual(ADAPTER_TRAINER_PROFILE_SHA256, TRAINER_PROFILE_SHA256)
        self.assertEqual(ADAPTER_CONTROLLER_SHA256, CONTROLLER_SHA256)
        self.assertEqual(ADAPTER_TEACHER_SHA256, TEACHER_SHA256)
        self.assertEqual(dict(ADAPTER_SCENARIO_MAX_TICKS), SCENARIO_MAX_TICKS)
        self.assertEqual(set(ADAPTER_DEVELOPMENT_SEEDS), {101, 103, 107, 109})
        self.assertEqual(set(ADAPTER_TUNING_SEEDS), {211, 223, 227, 229})

    def test_superseded_960_tick_and_generic_training_split_are_rejected(self):
        value = preparation_declaration().to_dict()
        value["scenario"]["scenario_max_ticks"]["slalom"] = 960
        with self.assertRaisesRegex(ContractError, "scenario"):
            FlightDeclaration.from_mapping(value)

        value = preparation_declaration().to_dict()
        value["splits"]["training_sha256"] = value["splits"].pop("development_sha256")
        with self.assertRaisesRegex(ContractError, "splits"):
            FlightDeclaration.from_mapping(value)

    def test_known_preregistered_work_and_stage_ceilings_are_exact(self):
        value = preparation_declaration().to_dict()
        value["work_ceilings"]["physical_ticks"] -= 6
        with self.assertRaisesRegex(ContractError, "must equal"):
            FlightDeclaration.from_mapping(value)

        value = preparation_declaration().to_dict()
        value["stage_ceilings"]["capture"]["physical_ticks"] = 960
        with self.assertRaisesRegex(ContractError, "stage ceilings"):
            FlightDeclaration.from_mapping(value)

    def test_stale_evidence_refuses_synthetic_complete_declaration(self):
        declaration, evidence = admitted_fixture()
        self.assertTrue(execution_gate(declaration, current_evidence=evidence)["admitted"])
        evidence["teacher_qualification"] = "0" * 64
        gate = execution_gate(declaration, current_evidence=evidence)
        self.assertFalse(gate["admitted"])
        self.assertIn("evidence_stale_or_missing:teacher_qualification", gate["reasons"])


class DurableAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.declaration, self.evidence = admitted_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def create(self, name="attempt-a1"):
        return FlightAccountingLedger.create(
            self.root / f"{name}.jsonl", self.declaration,
            attempt_id=name, current_evidence=self.evidence)

    def test_reservation_is_durable_before_confirmation_and_never_refunded(self):
        ledger = self.create()
        reservation = ledger.reserve_step(maximum_ticks=6, capture_labels=1)
        durable = FlightAccountingLedger.read_snapshot(
            ledger.path, self.declaration, attempt_id="attempt-a1")
        self.assertEqual(durable["pending_reservations"], [reservation])
        self.assertEqual(durable["charged"]["physical_ticks"], 6)
        self.assertEqual(durable["charged"]["capture_labels"], 1)
        self.assertEqual(durable["stage_counts"]["capture"]["physical_ticks"], 6)
        self.assertEqual(durable["stage_counts"]["capture"]["decisions"], 1)
        self.assertEqual(durable["stage_counts"]["capture"]["capture_labels"], 1)
        self.assertEqual(durable["confirmed_physical_ticks"], 0)

        ledger.complete(reservation, confirmed_physical_ticks=3)
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot["charged"]["physical_ticks"], 6)
        self.assertEqual(snapshot["confirmed_physical_ticks"], 3)
        self.assertFalse(snapshot["refunded_work"])
        ledger.close()

    def test_failed_startup_is_charged_and_blocks_ordinary_work(self):
        ledger = self.create()
        reservation = ledger.reserve_host_start()
        ledger.fail(reservation, "fixture_startup_failed")
        snapshot = ledger.snapshot()
        self.assertTrue(snapshot["failed"])
        self.assertEqual(snapshot["charged"]["host_starts"], 1)
        self.assertEqual(snapshot["charged"]["startup_operations"], 1)
        with self.assertRaisesRegex(ContractError, "cleanup only"):
            ledger.reserve_reset()

        cleanup = ledger.reserve_cleanup()
        ledger.complete(cleanup)
        final = ledger.close("failed_startup")
        self.assertTrue(final["failed"])
        self.assertEqual(final["failures"][0]["reason"], "fixture_startup_failed")

    def test_post_receipt_semantic_failure_preserves_confirmed_cost_and_blocks_work(self):
        ledger = self.create()
        reservation = ledger.reserve_step(maximum_ticks=6, stage="capture")
        ledger.complete(reservation, confirmed_physical_ticks=6)
        ledger.abort("semantic_reply_validation:stale_observation")
        snapshot = ledger.snapshot()
        self.assertTrue(snapshot["failed"])
        self.assertEqual(snapshot["charged"]["physical_ticks"], 6)
        self.assertEqual(snapshot["confirmed_physical_ticks"], 6)
        self.assertEqual(
            snapshot["failures"][-1]["reason"],
            "semantic_reply_validation:stale_observation",
        )
        self.assertIsNone(snapshot["failures"][-1]["reservation_id"])
        with self.assertRaisesRegex(ContractError, "cleanup only"):
            ledger.reserve_step(maximum_ticks=6, stage="capture")
        cleanup = ledger.reserve_cleanup()
        ledger.complete(cleanup)
        ledger.close("post_receipt_semantic_failure")

    def test_lost_reply_retains_unknown_consumption_and_refuses_replay(self):
        ledger = self.create()
        reservation = ledger.reserve_step(maximum_ticks=6, stage="capture")
        ledger.mark_unknown(reservation, "fixture_reply_lost_after_send")
        snapshot = ledger.snapshot()
        self.assertTrue(snapshot["unknown_consumption"])
        self.assertEqual(snapshot["charged"]["physical_ticks"], 6)
        self.assertEqual(snapshot["confirmed_physical_ticks"], 0)
        with self.assertRaisesRegex(ContractError, "cleanup only"):
            ledger.reserve_step()
        ledger.close("unknown_consumption")

    def test_failed_partial_step_retains_confirmed_ticks_and_reserved_capacity(self):
        ledger = self.create()
        reservation = ledger.reserve_step(maximum_ticks=6, stage="capture")
        ledger.fail(
            reservation, "fixture_terminal_integrity_failure",
            confirmed_physical_ticks=2)
        snapshot = ledger.snapshot()
        self.assertTrue(snapshot["failed"])
        self.assertEqual(snapshot["charged"]["physical_ticks"], 6)
        self.assertEqual(snapshot["confirmed_physical_ticks"], 2)
        self.assertEqual(snapshot["failures"][0]["confirmed_physical_ticks"], 2)
        with self.assertRaisesRegex(ContractError, "cleanup only"):
            ledger.reserve_reset()
        ledger.close("partial_failure")

    def test_partial_checkpoint_publication_becomes_unknown_on_resume(self):
        ledger = self.create()
        ledger.reserve_checkpoint_publication(seed=74011)
        ledger._stream.close()  # fixture: process dies after durable reservation

        resumed = FlightAccountingLedger.resume(
            ledger.path, self.declaration, attempt_id="attempt-a1",
            current_evidence=self.evidence)
        snapshot = resumed.snapshot()
        self.assertTrue(snapshot["unknown_consumption"])
        self.assertTrue(snapshot["interrupted"])
        self.assertEqual(snapshot["charged"]["checkpoint_publications"], 1)
        self.assertEqual(snapshot["checkpoints"], {})
        self.assertEqual(snapshot["pending_reservations"], [])
        with self.assertRaises(ContractError):
            resumed.reserve_checkpoint_publication()
        resumed.close("partial_publication")

    def test_checkpoint_lineage_and_costs_survive_clean_same_attempt_resume(self):
        ledger = self.create()
        first = ledger.reserve_checkpoint_publication(seed=74011)
        ledger.publish_checkpoint(
            first, checkpoint_id="epoch-05",
            sha256=sha256_json(["fixture", "epoch-05"]), parent_id=None)
        second = ledger.reserve_checkpoint_publication(seed=74011)
        ledger.publish_checkpoint(
            second, checkpoint_id="epoch-10",
            sha256=sha256_json(["fixture", "epoch-10"]), parent_id="epoch-05")
        before = ledger.snapshot()
        ledger._stream.close()  # fixture: process exits between settled operations

        resumed = FlightAccountingLedger.resume(
            ledger.path, self.declaration, attempt_id="attempt-a1",
            current_evidence=self.evidence)
        snapshot = resumed.snapshot()
        self.assertEqual(
            snapshot["checkpoints"]["epoch-10"]["parent_id"], "epoch-05")
        self.assertEqual(
            snapshot["checkpoints"]["epoch-10"]["stage"], "clone:74011")
        self.assertTrue(snapshot["interrupted"])
        self.assertEqual(
            snapshot["charged"]["checkpoint_publications"],
            before["charged"]["checkpoint_publications"],
        )
        # This is the same durable attempt, not a new attempt or replay: settled
        # work and lineage remain charged while new budgeted work may continue.
        reset = resumed.reserve_reset(stage="capture")
        resumed.complete(reset)
        self.assertEqual(resumed.snapshot()["charged"]["resets"], 1)
        resumed.close("fixture_clean_resume")

    def test_checkpoint_parent_cannot_cross_policy_seed(self):
        ledger = self.create()
        first = ledger.reserve_checkpoint_publication(seed=74011)
        ledger.publish_checkpoint(
            first, checkpoint_id="seed74011-epoch05",
            sha256=sha256_json(["fixture", 74011, 5]), parent_id=None)
        second = ledger.reserve_checkpoint_publication(seed=74013)
        with self.assertRaisesRegex(ContractError, "another policy seed"):
            ledger.publish_checkpoint(
                second, checkpoint_id="seed74013-epoch10",
                sha256=sha256_json(["fixture", 74013, 10]),
                parent_id="seed74011-epoch05")
        ledger.mark_unknown(second, "invalid_cross_seed_publication_attempt")
        ledger.close("fixture_cross_seed_rejected")

    def test_interrupted_jsonl_write_is_not_repaired_or_replayed(self):
        ledger = self.create()
        reservation = ledger.reserve_reset(stage="capture")
        ledger.complete(reservation)
        ledger._stream.close()
        with ledger.path.open("ab") as target:
            target.write(b'{"schema":"nca-flight-accounting-record/v1"')
            target.flush()
        before = ledger.path.read_bytes()
        with self.assertRaisesRegex(ContractError, "interrupted"):
            FlightAccountingLedger.resume(
                ledger.path, self.declaration, attempt_id="attempt-a1",
                current_evidence=self.evidence)
        self.assertEqual(ledger.path.read_bytes(), before)

    def test_duplicate_invocation_and_consumed_attempt_are_refused(self):
        ledger = self.create()
        with self.assertRaisesRegex(ContractError, "already exists"):
            self.create()
        ledger.close()
        with self.assertRaisesRegex(ContractError, "cannot be resumed"):
            FlightAccountingLedger.resume(
                ledger.path, self.declaration, attempt_id="attempt-a1",
                current_evidence=self.evidence)

    def test_null_continuation_charges_ticks_without_decision(self):
        ledger = self.create()
        reservation = ledger.reserve_step(
            maximum_ticks=3, capture_labels=0, decision=False)
        durable = FlightAccountingLedger.read_snapshot(
            ledger.path, self.declaration, attempt_id="attempt-a1")
        self.assertEqual(durable["charged"]["host_calls"], 1)
        self.assertEqual(durable["charged"]["physical_ticks"], 3)
        self.assertEqual(durable["charged"]["decisions"], 0)
        ledger.complete(reservation, confirmed_physical_ticks=3)
        ledger.close()

    def test_tuning_reset_charges_reset_call_and_tuning_episode_atomically(self):
        ledger = self.create()
        reservation = ledger.reserve_reset(tuning=True)
        durable = FlightAccountingLedger.read_snapshot(
            ledger.path, self.declaration, attempt_id="attempt-a1")
        self.assertEqual(durable["charged"]["resets"], 1)
        self.assertEqual(durable["charged"]["host_calls"], 1)
        self.assertEqual(durable["charged"]["tuning_episodes"], 1)
        self.assertEqual(durable["stage_counts"]["tuning"]["resets"], 1)
        self.assertEqual(durable["stage_counts"]["tuning"]["tuning_episodes"], 1)
        self.assertEqual(durable["pending_reservations"], [reservation])
        ledger.complete(reservation)
        ledger.close()

    def test_capture_and_per_seed_clone_stage_ceilings_fail_closed(self):
        ledger = self.create()
        capture = ledger.reserve_many({
            "physical_ticks": 68_400,
            "decisions": 11_400,
            "capture_labels": 11_400,
        }, scope="fixture_capture_capacity", stage="capture")
        ledger.complete(capture)
        with self.assertRaisesRegex(ContractError, "capture.decisions"):
            ledger.reserve_many(
                {"decisions": 1}, scope="fixture_capture_overrun", stage="capture")

        clone = ledger.reserve_optimizer(
            updates=1_800, presentations=228_000, seed=74011)
        ledger.complete(clone)
        self.assertEqual(
            ledger.snapshot()["stage_counts"]["clone:74011"]["optimizer_updates"],
            1_800,
        )
        with self.assertRaisesRegex(ContractError, "clone:74011.optimizer_updates"):
            ledger.reserve_optimizer(updates=1, presentations=1, seed=74011)
        ledger.close()

    def test_stage_sensitive_work_cannot_be_mislabeled(self):
        ledger = self.create()
        with self.assertRaisesRegex(ContractError, "optimizer_updates.*capture"):
            ledger.reserve_many(
                {"optimizer_updates": 1, "sample_presentations": 1},
                scope="wrong_stage_optimizer",
                stage="capture",
            )
        with self.assertRaisesRegex(ContractError, "physical_ticks.*clone:74011"):
            ledger.reserve_many(
                {"physical_ticks": 6, "decisions": 1},
                scope="wrong_stage_physical",
                stage="clone:74011",
            )
        ledger.close()

    def test_final_stage_is_structurally_disabled(self):
        ledger = self.create()
        with self.assertRaisesRegex(ContractError, "final work is disabled"):
            ledger.reserve_reset(stage="final")
        ledger.close()

    def test_exhausted_counter_refuses_only_new_work_that_needs_it(self):
        ledger = self.create()
        start = ledger.reserve_host_start()
        ledger.complete(start)
        with self.assertRaisesRegex(ContractError, "counter exhausted: host_starts"):
            ledger.reserve_host_start()

        reset = ledger.reserve_reset(stage="capture")
        ledger.complete(reset)
        self.assertEqual(ledger.snapshot()["charged"]["resets"], 1)
        ledger.close()


if __name__ == "__main__":
    unittest.main()
