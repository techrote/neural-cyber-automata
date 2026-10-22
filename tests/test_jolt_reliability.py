"""Offline failure-boundary regression tests, with synthetic observations only."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from nca.contracts import ContractError, sha256_json
    from nca.controllers.baseline import SkillGoal
    from nca.controllers.jolt_learned import JoltController, JoltPolicy, SupervisorRetirement
    from nca.data import read_episode
    from nca.evaluation.verified_journal import VerifiedDurableJournal
    from nca.train.jolt_screen import capture, close_worker_v2, error_record, validate_execution
    from test_jolt_learned import observation
    from test_jolt_screen import CutoffFixtureOwner, Journal


def setUpModule():
    if TORCH_AVAILABLE:
        torch.set_num_threads(1)


if TORCH_AVAILABLE:
    class EmergencyOwner(CutoffFixtureOwner):
        """Always unstable; cancellation can revoke without suspension."""

        def __init__(self, *, invalid_cancel=False):
            super().__init__()
            self.invalid_cancel = invalid_cancel

        def observation(self, tick, action, **kwargs):
            return {**super().observation(tick, action, **kwargs), "rotation": 2.}

        def request(self, operation, **fields):
            result = super().request(operation, **fields)
            if operation == "cancel":
                self.view["suspended"] = False
                if self.invalid_cancel:
                    self.view["position"][0] += 1.
            return result


@unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning extra is not installed")
class JoltReliabilityTests(unittest.TestCase):
    def test_emergency_limit_retires_at_48_ticks_without_any_neural_forward(self):
        runtime = JoltController(JoltPolicy("bc"), structured_retirement=True)
        goal = SkillGoal("synthetic-retirement", "directed_hop", (48., 0.), timeout_seconds=4.)
        with patch.object(runtime.policy, "forward", side_effect=AssertionError("no model work during emergency")) as forward:
            for tick in range(0, 48, 6):
                decision = runtime.step(observation(tick, rotation=2.), goal)
                self.assertIsNone(decision.proposal)
                runtime.record_dispatch(decision, decision.action, admitted=True, intervention=True)
            with self.assertRaises(SupervisorRetirement) as captured:
                runtime.step(observation(48, rotation=2.), goal)
            self.assertEqual(error_record(captured.exception), {
                "type": "SupervisorRetirement", "code": "jolt_supervisor_retired",
                "reason": "emergency_recovery_deadline"})
            self.assertEqual(runtime.supervisor.reason, "emergency_recovery_deadline")
            self.assertEqual(runtime.context.deadline_tick, 240)
            with self.assertRaisesRegex(ValueError, "retired invocation"):
                runtime.step(observation(54, rotation=2.), goal)
            forward.assert_not_called()

    def test_legacy_retirement_diagnostic_keeps_its_default_behavior(self):
        runtime = JoltController(JoltPolicy("bc"))
        goal = SkillGoal("synthetic-legacy", "directed_hop", (48., 0.), timeout_seconds=4.)
        for tick in range(0, 48, 6):
            decision = runtime.step(observation(tick, rotation=2.), goal)
            runtime.record_dispatch(decision, decision.action, admitted=True, intervention=True)
        with self.assertRaises(ValueError) as captured:
            runtime.step(observation(48, rotation=2.), goal)
        self.assertIs(type(captured.exception), ValueError)
        self.assertEqual(error_record(captured.exception), {"type": "ValueError", "code": None})

    def capture_failure(self, output, *, invalid_cancel=False):
        owner, journal = EmergencyOwner(invalid_cancel=invalid_cancel), Journal()
        ledger = SimpleNamespace(check=lambda: None, reserve=lambda *args, **kwargs: None)
        with patch("nca.train.jolt_learning.rollout_sequence") as rollout:
            with self.assertRaises(SupervisorRetirement) as captured:
                capture(owner, ledger, journal,
                    {"id": "synthetic-emergency", "host_seed": 7, "spawn": [0., 0.], "target": [48., 0.]},
                    identifier="ppo-seed190103-u0-e0", controller_id="residual-seed190103-initial",
                    kind="residual", stage="ppo", output=output, root=ROOT,
                    config={"profile_sha256": sha256_json(owner.profile)}, source_sha256="c" * 64,
                    policy=JoltPolicy("residual"), base=JoltPolicy("bc"),
                    generator=torch.Generator().manual_seed(390103), execution_version=2)
            rollout.assert_not_called()
        return owner, captured.exception.row

    def test_failed_capture_retains_native_cancel_and_unattempted_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            owner, row = self.capture_failure(output)
            raw = read_episode(output / row["artifact"])
            retained = json.loads((output / "metrics" / f"{row['id']}.json").read_text())
        self.assertEqual(owner.calls, ["reset", "profile"] + ["step"] * 8 + ["cancel"])
        self.assertEqual(row["consumed_ticks"], 48)
        self.assertTrue(row["owner_closure_confirmed"])
        self.assertFalse(raw.events[-1].next_observation["suspended"])
        self.assertIsNone(row["cancellation_error"])
        self.assertEqual(row["error"]["reason"], "emergency_recovery_deadline")
        self.assertEqual(row["bootstrap_reason"], "not_attempted")
        self.assertFalse(row["bootstrap_observation_valid"])
        self.assertIsNone(row["bootstrap_observation_sha256"])
        self.assertEqual(row["bootstrap_value"], 0.)
        self.assertEqual(retained["error"], raw.evaluation["error"])
        self.assertTrue(row["metrics"]["integrity_failure"])
        self.assertEqual(raw.status, "failed")

    def test_failed_cancel_remains_separate_from_original_retirement(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _, row = self.capture_failure(output, invalid_cancel=True)
            raw = read_episode(output / row["artifact"])
        self.assertFalse(row["owner_closure_confirmed"])
        self.assertEqual(row["error"]["code"], "jolt_supervisor_retired")
        self.assertEqual(row["cancellation_error"]["code"], "morphology_conformance_cancel")
        self.assertEqual(row["cancellation_error"], raw.evaluation["cancellation_error"])

    def test_poisoned_recording_and_budget_failure_cannot_prevent_host_revocation(self):
        journal = SimpleNamespace(write=Mock(side_effect=OSError("private path must not escape")),
            close=Mock(), durability_receipt=lambda: {"poisoned": True, "closed_verified": False})
        native = SimpleNamespace(close=Mock(), _process=SimpleNamespace(poll=lambda: 0))
        owner = SimpleNamespace(closed_native=False, request=Mock(side_effect=OSError("synthetic close failure")))
        ledger = SimpleNamespace(reserve=Mock(side_effect=RuntimeError("synthetic exhausted budget")), snapshot=lambda: {})
        errors = []
        original = {"type": "SupervisorRetirement", "code": "jolt_supervisor_retired", "reason": "emergency_recovery_deadline"}
        status, failure = close_worker_v2(native, owner, ledger, journal, status="failed", failure=deepcopy(original),
                                          recording_errors=errors)
        native.close.assert_called_once_with(graceful=False)
        journal.close.assert_called_once()
        self.assertEqual(status, "failed")
        self.assertEqual(failure, original)
        self.assertEqual({item["stage"] for item in errors}, {
            "owner_close", "host_termination_reservation", "host_terminated_note", "worker_closed_note"})
        self.assertNotIn("private path", json.dumps(errors))

    def test_clean_close_readback_is_not_a_successful_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = VerifiedDurableJournal(Path(directory) / "owner-journal.jsonl")
            ledger = SimpleNamespace(snapshot=lambda: {})
            errors = []
            status, failure = close_worker_v2(None, None, ledger, journal, status="failed",
                failure={"type": "ValueError", "code": None}, recording_errors=errors)
        self.assertTrue(journal.closed_verified)
        self.assertEqual(status, "failed")
        self.assertEqual(failure["type"], "ValueError")
        self.assertEqual(errors, [])

    def test_old_declaration_cannot_admit_v2_worker(self):
        with self.assertRaisesRegex(ContractError, "prospective invocation"):
            validate_execution(ROOT, {"schema": "nca-learn-jolt-002-execution/v1", "attempts": 1},
                config={}, host_identity={}, execution_version=2)

    def test_boolean_attempt_count_cannot_admit_v2_worker(self):
        with self.assertRaisesRegex(ContractError, "prospective invocation"):
            validate_execution(ROOT, {"schema": "nca-learn-jolt-002-execution/v2", "attempts": True},
                config={}, host_identity={}, execution_version=2)


class JoltReliabilityCLITests(unittest.TestCase):
    def test_v2_freeze_requires_a_distinct_public_commission_before_reading_config(self):
        spec = importlib.util.spec_from_file_location("jolt_cli_v2_freeze", ROOT / "tools/learn_jolt.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        if not TORCH_AVAILABLE:
            self.skipTest("freeze validation imports optional runtime")
        with self.assertRaisesRegex(ValueError, "jolt_allocation_identity"):
            cli.freeze(SimpleNamespace(), execution_version=2)

    def test_run_v2_rejects_consumed_v1_declaration_without_spawning_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            declaration = root / "declaration.json"
            declaration.write_text(json.dumps({"schema": "nca-learn-jolt-002-execution/v1", "attempts": 1,
                "automatic_retry": False, "automatic_extension": False,
                "budget": {"maximum_wall_seconds": 1800}}))
            original = declaration.read_bytes()
            output = root / "output"
            result = subprocess.run([sys.executable, str(ROOT / "tools/learn_jolt.py"), "run-v2",
                "--declaration", str(declaration), "--checkout", str(root / "absent-host"),
                "--godot", str(root / "absent-runtime"), "--output", str(output)],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())
            self.assertFalse((root / "output-lifecycle" / "worker-started.json").exists())
            self.assertEqual(declaration.read_bytes(), original)
            verdict = json.loads(result.stdout)
            self.assertFalse(verdict["completed"])

    def test_offline_software_reviews_cannot_admit_a_prospective_experiment(self):
        from nca.contracts import ContractError
        from nca.train.jolt_execution_contract import require_prospective_review
        for record in ({"accepted": True}, {"offline_scope_only": True, "prospective_experiment_admitted": False},
                       {"offline_scope_only": False}, {"prospective_experiment_admitted": True}):
            with self.subTest(record=record), self.assertRaises(ContractError):
                require_prospective_review(record)
        require_prospective_review({"offline_scope_only": False, "prospective_experiment_admitted": True})

    def inventory_fixture(self, root):
        from nca.contracts import sha256_json
        from nca.evaluation.verified_journal import VerifiedDurableJournal
        from test_jolt_learning_executor import ClosedLearningInventoryTests
        fixture = ClosedLearningInventoryTests()
        fixture.setUp()
        with patch("test_jolt_learning_executor.DurableJournal", VerifiedDurableJournal):
            docs = fixture.fixture(root)
        declaration = docs["declaration.json"]
        declaration.update(schema="nca-learn-jolt-002-execution/v2", attempts=1,
            allocation_id="synthetic-software-only",
            authorization_ref="https://github.com/techrote/neural-cyber-automata/issues/62#issuecomment-1")
        manifest = docs["manifest.json"]
        manifest.update(schema="nca-learn-jolt-002-run/v2", declaration_sha256=sha256_json(declaration),
                        whole_process_started_monotonic=100.)
        docs["worker-result.json"].update(schema=manifest["schema"], manifest=deepcopy(manifest))
        docs["result.json"]["schema"] = manifest["schema"]
        docs["independent-readback.json"]["schema"] = "nca-learn-jolt-002-independent-readback/v2"
        fixture.publish(root, docs)
        return fixture, docs

    def test_v2_inventory_binds_exact_file_inventory_and_rejects_relabelled_schemas(self):
        from nca.evaluation.durable_journal import file_identity
        from nca.contracts import sha256_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, docs = self.inventory_fixture(root)
            result = fixture.cli.verify_inventory(root, execution_version=2)
            self.assertEqual(result["inventory_identity"], file_identity(root / "inventory.json"))
            for target in ("declaration.json", "manifest.json"):
                altered = deepcopy(docs)
                altered[target]["schema"] = altered[target]["schema"].replace("/v2", "/v1")
                altered["manifest.json"]["declaration_sha256"] = sha256_json(altered["declaration.json"])
                altered["worker-result.json"]["manifest"] = deepcopy(altered["manifest.json"])
                fixture.publish(root, altered)
                with self.subTest(target=target), self.assertRaises(ValueError):
                    fixture.cli.verify_inventory(root, execution_version=2)

    def test_lifecycle_binding_rejects_swapped_output_or_original_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, docs = self.inventory_fixture(root)
            docs["synthetic-extra.json"] = {"fixture": "original-output"}
            fixture.publish(root, docs)
            inventory = fixture.cli.verify_inventory(root, execution_version=2)
            lifecycle = {"completed": True, "verified": True, "worker_inventory": inventory,
                         "whole_process_started_monotonic": 100.}
            with patch("nca.train.jolt_executor.verify_lifecycle", return_value=lifecycle):
                result = fixture.cli.verify_execution_v2(root, terminal_stdout=b"synthetic", observed_exit_code=0)
                self.assertTrue(result["completed"])
                self.assertFalse(result["qualified"])
                # Same counts and chain, but changed independently inventoried
                # bytes cannot be substituted for the original worker output.
                docs["synthetic-extra.json"] = {"fixture": "different-output"}
                fixture.publish(root, docs)
                changed = fixture.cli.verify_inventory(root, execution_version=2)
                self.assertEqual({key: value for key, value in changed.items() if key != "inventory_identity"},
                                 {key: value for key, value in inventory.items() if key != "inventory_identity"})
                self.assertNotEqual(changed["inventory_identity"], inventory["inventory_identity"])
                with self.assertRaisesRegex(ValueError, "different worker inventory"):
                    fixture.cli.verify_execution_v2(root, terminal_stdout=b"synthetic", observed_exit_code=0)
                lifecycle["worker_inventory"] = fixture.cli.verify_inventory(root, execution_version=2)
                lifecycle["whole_process_started_monotonic"] = 101.
                with self.assertRaisesRegex(ValueError, "original invocation clock"):
                    fixture.cli.verify_execution_v2(root, terminal_stdout=b"synthetic", observed_exit_code=0)


if __name__ == "__main__":
    unittest.main()
