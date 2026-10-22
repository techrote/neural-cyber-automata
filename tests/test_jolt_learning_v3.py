"""Execution-v3 synthetic boundary tests; no physical experiment or trained weights."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from nca.contracts import ContractError, WireAction, sha256_json
    from nca.controllers.jolt_learned import JoltController, JoltPolicy, SupervisorRetirement
    from nca.controllers.jolt_recovery_v4 import JoltRecoveryControllerV4
    from nca.controllers.recovery_supervisor import PhysicalRecoverySupervisor
    from nca.data import read_episode
    from nca.data.jolt_samples import JoltEpisode, JoltEpisodeV3
    from nca.train.jolt_capture_v3 import read_sample_episode_v3, write_sample_episode_v3
    from nca.train.jolt_learning import (
        convert_clone_episode_v3, generalized_advantages, ppo_update_v3, rollout_sequence, rollout_sequence_v3,
    )
    from nca.train.jolt_screen import capture, validate_execution
    from test_jolt_screen import CutoffFixtureOwner, Journal, fixture


def setUpModule():
    if TORCH_AVAILABLE:
        torch.set_num_threads(1)


if TORCH_AVAILABLE:
    class SyntheticRetirementOwner(CutoffFixtureOwner):
        """Native-shaped cancellation without a physical host or a model pilot."""

        def __init__(self, *, start_emergency=0, invalid_cancel=None):
            super().__init__(unstable_tick=-1)
            self.start_emergency = start_emergency
            self.invalid_cancel = invalid_cancel

        def observation(self, tick, action, **kwargs):
            result = super().observation(tick, action, **kwargs)
            result.update(rotation=2. if tick >= self.start_emergency else 0.,
                          pending_input_edges=0, lease_remaining=0)
            return result

        def request(self, operation, **fields):
            if operation != "cancel":
                return super().request(operation, **fields)
            self.calls.append(operation)
            self.view = deepcopy(self.view)
            self.view.update(action={}, suspended=False, pending_input_edges=0, lease_remaining=0,
                             control_generation=self.view["control_generation"] + 1)
            self.view["observation_id"] = ":".join(str(self.view[key]) for key in
                ("run_id", "epoch", "actor_id", "control_generation", "tick"))
            if self.invalid_cancel == "pose":
                self.view["position"][0] += 1.
            elif self.invalid_cancel == "edges":
                self.view["pending_input_edges"] = 1
            elif self.invalid_cancel == "action":
                self.view["action"] = WireAction(effort=0.).to_dict()
            elif self.invalid_cancel == "lease":
                self.view["lease_remaining"] = 6
            return {"cancelled": True, "observation": self.view}


    class SyntheticLedger:
        def __init__(self):
            self.counts = Counter()
            self.phase = "active"

        def enter_phase(self, phase):
            if self.phase != "active" or phase != "closure":
                raise AssertionError("synthetic capture may only advance once to closure")
            self.phase = phase

        def check(self, **kwargs):
            pass

        def remaining_seconds(self, **kwargs):
            return 6000.

        def reserve(self, name, amount=1, **kwargs):
            self.counts[name] += amount


@unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning extra is not installed")
class JoltV3CaptureTests(unittest.TestCase):
    def capture(self, output, *, kind="residual", maximum_decisions=40,
                start_emergency=0, invalid_cancel=None, identifier="synthetic-v3"):
        owner = SyntheticRetirementOwner(start_emergency=start_emergency, invalid_cancel=invalid_cancel)
        ledger, journal = SyntheticLedger(), Journal()
        policy = JoltPolicy(kind) if kind in ("bc", "residual") else None
        base = JoltPolicy("bc") if kind == "residual" else None
        stage = "teacher" if kind == "authored" else "ppo" if kind == "residual" else "open"
        result = capture(owner, ledger, journal,
            {"id": "synthetic-emergency-v3", "host_seed": 7, "spawn": [0., 0.], "target": [48., 0.]},
            identifier=identifier, controller_id=kind, kind=kind, stage=stage,
            output=output, root=ROOT, config={"profile_sha256": sha256_json(owner.profile)},
            source_sha256="c" * 64, policy=policy, base=base,
            generator=torch.Generator().manual_seed(390103), maximum_decisions=maximum_decisions,
            execution_version=3)
        return owner, ledger, journal, result

    def test_shared_safe_rule_completes_unsuccessful_capture_for_every_comparator(self):
        for kind in ("authored", "idle", "bc", "residual"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                with patch.object(JoltPolicy, "forward", side_effect=AssertionError("no emergency model forward")):
                    owner, ledger, journal, (sample, row, sequence) = self.capture(Path(directory), kind=kind)
                raw = read_episode(Path(directory) / row["artifact"])
                readback = read_sample_episode_v3(Path(directory) / row["sample_artifact"])
                self.assertEqual(readback.sha256, sample.sha256)
                self.assertEqual(row["outcome"], dict(sample.outcome))
                self.assertEqual(row["outcome"], dict(raw.evaluation["outcome"]))
                self.assertEqual(sample.status, "completed")
                self.assertEqual(row["outcome"]["kind"], "safe_retirement")
                self.assertTrue(row["outcome"]["complete"])
                self.assertFalse(row["metrics"]["success"])
                self.assertFalse(row["metrics"]["integrity_failure"])
                self.assertFalse(row["metrics"]["administratively_capped"])
                self.assertEqual(row["metrics"]["capped_completion_seconds"], 4.)
                self.assertIsNone(row["error"])
                self.assertIsNone(row["cancellation_error"])
                self.assertEqual(row["decisions"], 8)
                self.assertEqual(row["consumed_ticks"], 48)
                self.assertEqual(owner.calls, ["reset", "profile"] + ["step"] * 8 + ["cancel"])
                self.assertTrue(row["owner_closure_confirmed"])
                self.assertFalse(raw.events[-1].next_observation["suspended"])
                self.assertEqual(ledger.counts["episodes_started"], 1)
                self.assertEqual(ledger.counts["episodes_completed"], 1)
                self.assertEqual(ledger.counts["episodes_aborted"], 0)
                self.assertEqual(ledger.counts["checkpoint_executions"], 9 if kind in ("bc", "residual") else 0)
                self.assertEqual(row["retirement"]["path"], "ordinary_attempt")
                self.assertEqual(row["bootstrap_reason"], "safe_retirement")
                self.assertIsNone(row["bootstrap_observation_sha256"])
                self.assertFalse(row["bootstrap_observation_valid"])
                self.assertEqual(len([event for event in journal.records() if event["event"] == "safe_retirement_verified"]), 1)
                if sequence is not None:
                    self.assertTrue(sequence.terminal_failure)
                    self.assertEqual(row["policy_values"], [None] * 8)
                    self.assertEqual(row["bootstrap_value"], 0.)
                    self.assertFalse(any(item.valid for item in sequence.transitions))
                    self.assertFalse(any(item.actor_eligible for item in sequence.transitions))
                    advantages, returns = generalized_advantages(sequence)
                    self.assertEqual(advantages.tolist(), [0.] * 8)
                    self.assertEqual(returns.tolist(), [0.] * 8)

    def test_retirement_at_cutoff_peek_wins_without_forward_or_extra_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            owner, ledger, _, (sample, row, sequence) = self.capture(Path(directory), maximum_decisions=8)
        self.assertEqual(row["retirement"]["path"], "cutoff_peek")
        self.assertEqual(ledger.counts["checkpoint_executions"], 8)
        self.assertEqual(ledger.counts["bootstrap_forward"], 0)
        self.assertTrue(sample.decisions[-1].event.truncated)
        self.assertFalse(sample.decisions[-1].event.terminated)
        self.assertTrue(sequence.transitions[-1].administrative_cutoff)
        self.assertFalse(sequence.transitions[-1].task_terminal)
        self.assertTrue(sequence.terminal_failure)
        self.assertEqual(sequence.bootstrap_value, 0.)
        self.assertEqual(owner.calls.count("step"), 8)

    def test_safe_failure_keeps_teacher_ordinary_labels_abstention_and_original_rewards(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, (sample, row, _) = self.capture(Path(directory), kind="authored", start_emergency=12)
            raw = read_episode(Path(directory) / row["artifact"])
        converted = convert_clone_episode_v3(sample,
            expected_teacher_sha256=JoltRecoveryControllerV4().specification_sha256,
            expected_supervisor_sha256=sha256_json(PhysicalRecoverySupervisor().specification()))
        self.assertEqual(len(converted.rows), 10)
        self.assertEqual(sum(item.eligible for item in converted.rows), 2)
        self.assertFalse(any(item.eligible for item in converted.rows[2:]))
        self.assertTrue(all(item.label is not None and item.label[-1] == 0. for item in converted.rows[:2]))
        self.assertEqual(row["metrics"]["elapsed_ticks"], 60)
        self.assertEqual(row["metrics"]["return"], sum(record.event.reward for record in sample.decisions))
        self.assertTrue(all(terms["deadline"] == 0. for terms in raw.evaluation["reward_terms"]))
        self.assertFalse(any(record.event.terminated for record in sample.decisions))

    def test_all_masked_teacher_episode_is_retained_without_fabricating_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, (sample, _, _) = self.capture(Path(directory), kind="authored")
        converted = convert_clone_episode_v3(sample,
            expected_teacher_sha256=JoltRecoveryControllerV4().specification_sha256,
            expected_supervisor_sha256=sha256_json(PhysicalRecoverySupervisor().specification()))
        self.assertEqual(len(converted.rows), 8)
        self.assertFalse(any(item.eligible for item in converted.rows))
        self.assertTrue(all(item.label is None for item in converted.rows))

    def test_failed_terminal_does_not_add_penalty_or_cross_original_override_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, (sample, row, sequence) = self.capture(Path(directory), start_emergency=12)
        original = JoltEpisode(sample.id, sample.split, sample.decisions, "completed", sample.reason)
        legacy = rollout_sequence(original, [0. if value is None else value for value in row["policy_values"]],
                                  bootstrap_value=0., final_observation_valid=False)
        advantages, returns = generalized_advantages(sequence)
        old_advantages, old_returns = generalized_advantages(legacy)
        self.assertTrue(torch.equal(advantages, old_advantages))
        self.assertTrue(torch.equal(returns, old_returns))
        self.assertEqual(sum(item.valid for item in sequence.transitions), 2)
        self.assertEqual(advantages.tolist()[2:], [0.] * 8)
        self.assertAlmostEqual(float(returns[1]), sequence.transitions[1].reward, places=6)

    def test_exact_256_all_masked_decisions_keep_zero_actual_optimizer_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, (_, _, original) = self.capture(Path(directory))
        # Repeated independent synthetic reset fixtures exercise the arithmetic;
        # these are neither captured experiments nor a new case distribution.
        sequences = [replace(original, episode_id=f"synthetic-masked-{index}") for index in range(32)]
        policy, base = JoltPolicy("residual"), JoltPolicy("bc")
        optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
        counters = Counter()
        with patch.object(optimizer, "step", side_effect=AssertionError("no masked optimization")), \
                patch.object(policy, "forward", side_effect=AssertionError("no masked model value")), \
                patch.object(base, "forward", side_effect=AssertionError("no masked base value")):
            result = ppo_update_v3(policy, base, optimizer, sequences, seed=190103,
                reserve=lambda name, amount=1: counters.update({name: amount}), source_sha256="c" * 64)
        self.assertEqual(result["confirmed_decisions"], 256)
        self.assertEqual(result["safe_retirements"], 32)
        self.assertEqual(result["optimizer_calls"], 0)
        self.assertEqual(result["actor_samples"], 0)
        self.assertEqual(result["excluded_actor_samples"], 256)
        self.assertIsNone(result["mean_loss"])
        self.assertFalse(result["new_terminal_penalty"])
        self.assertEqual(counters, {})
        with self.assertRaises(ValueError):
            ppo_update_v3(policy, base, optimizer, sequences[:-1], seed=190103,
                reserve=lambda *args: None, source_sha256="c" * 64)

    def test_bad_cancellation_never_becomes_safe_complete_outcome(self):
        for corruption in ("pose", "edges", "action", "lease"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(SupervisorRetirement) as captured:
                    self.capture(Path(directory), invalid_cancel=corruption)
                row = captured.exception.row
                self.assertFalse(row["outcome"]["complete"])
                self.assertEqual(row["outcome"]["kind"], "integrity_error")
                self.assertIsNone(row["retirement"])
                self.assertIsNotNone(row["cancellation_error"])
                self.assertTrue(row["metrics"]["integrity_failure"])

    def test_reason_string_cannot_forge_early_retirement_or_unknown_outcome(self):
        original = JoltController.step
        for reason, tick in (("emergency_recovery_deadline", 6), ("canceled", 48)):
            def forged(controller, observation, goal, **kwargs):
                if observation["tick"] >= tick:
                    raise SupervisorRetirement(reason)
                return original(controller, observation, goal, **kwargs)
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                with patch.object(JoltController, "step", forged), self.assertRaises(SupervisorRetirement) as caught:
                    self.capture(Path(directory))
                self.assertIsNone(caught.exception.row["retirement"])
                self.assertFalse(caught.exception.row["outcome"]["complete"])
                self.assertTrue(caught.exception.row["metrics"]["integrity_failure"])

    def test_version_boundary_and_noninvented_bootstrap_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, (sample, row, _) = self.capture(Path(directory))
            with self.assertRaises(ContractError):
                JoltEpisode.from_dict(sample.to_dict())
            with self.assertRaises(ContractError):
                replace(sample, outcome={**sample.outcome, "success": True})
            with self.assertRaises(ValueError):
                rollout_sequence_v3(sample, row["policy_values"], bootstrap_value=1., final_observation_valid=False)
            with self.assertRaises(ValueError):
                rollout_sequence_v3(sample, row["policy_values"], final_observation_valid=True)
            with self.assertRaises(ValueError):
                rollout_sequence_v3(sample, [0.] * 8, final_observation_valid=False)
            with patch("nca.train.jolt_capture_v3.MAX_SAMPLE_BYTES_V3", 128), self.assertRaises(ContractError):
                read_sample_episode_v3(Path(directory) / row["sample_artifact"])
            with patch("nca.train.jolt_storage_v3.episode_byte_limit", return_value=128), self.assertRaises(ContractError):
                write_sample_episode_v3(Path(directory) / "oversized.json.gz", sample)
            # Interrupted final-path bytes remain available as incomplete
            # evidence; no retry may overwrite them with a repaired success.
            self.assertTrue((Path(directory) / "oversized.json.gz").exists())

    def test_execution_v3_dispatch_preserves_keyword_only_admission_contract(self):
        declaration, config, identity = {"schema": "synthetic"}, {"budget": {}}, {"host": "synthetic"}
        with patch("nca.train.jolt_admission_v3.validate_execution_v3", return_value="admitted") as admission:
            result = validate_execution(ROOT, declaration, config=config, host_identity=identity,
                                        execution_version=3)
        self.assertEqual(result, "admitted")
        admission.assert_called_once_with(ROOT, declaration, config=config, host_identity=identity)

    def test_active_deadline_uses_reserved_closure_and_preserves_capped_episode(self):
        from nca.train.jolt_budget_v3 import BudgetLedgerV3
        clock = [5980.]

        class DeadlineOwner(SyntheticRetirementOwner):
            def request(self, operation, **fields):
                result = super().request(operation, **fields)
                if operation == "step":
                    clock[0] = 6000.
                return result

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            budget = json.loads((ROOT / "configs/learning/jolt-learn-003-v1.json").read_text())["budget"]
            ledger = BudgetLedgerV3(budget, started=0., spool_path=output / "ledger.jsonl",
                                    monotonic=lambda: clock[0])
            owner, journal = DeadlineOwner(start_emergency=0), Journal()
            try:
                with self.assertRaises(ContractError) as caught:
                    capture(owner, ledger, journal,
                        {"id": "synthetic-deadline-v3", "host_seed": 7, "spawn": [0., 0.], "target": [48., 0.]},
                        identifier="synthetic-deadline-v3", controller_id="idle", kind="idle", stage="open",
                        output=output, root=ROOT, config={"profile_sha256": sha256_json(owner.profile)},
                        source_sha256="c" * 64, execution_version=3)
                self.assertEqual(caught.exception.code, "jolt_v3_phase_deadline")
                row = caught.exception.row
                sample = read_sample_episode_v3(output / row["sample_artifact"])
                self.assertEqual(ledger.phase, "closure")
                self.assertEqual(sample.status, "capped")
                self.assertEqual(row["outcome"]["kind"], "integrity_error")
                self.assertFalse(row["outcome"]["complete"])
                self.assertFalse(row["metrics"]["success"])
                self.assertFalse(row["metrics"]["integrity_failure"])
                self.assertTrue(row["owner_closure_confirmed"])
                self.assertIsNone(row["cleanup_phase_error"])
                self.assertEqual(row["decisions"], 1)
                self.assertEqual(owner.calls, ["reset", "profile", "step", "cancel"])
                self.assertEqual(ledger.counts["episodes_started"], 1)
                self.assertEqual(ledger.counts["episodes_aborted"], 1)
                self.assertEqual(ledger.counts["episodes_completed"], 0)
            finally:
                ledger.enter_phase("audit")
                ledger.enter_phase("publication")
                ledger.close()

    def test_complete_native_exchange_deadline_includes_validation_and_durable_response(self):
        for delayed_phase in ("response", "journal"):
            with self.subTest(delayed_phase=delayed_phase):
                clock, ledger, journal, host, owner = fixture()
                owner.execution_version = 3
                original_request = host.request
                original_write = journal.write

                def delayed_request(operation, **fields):
                    response = original_request(operation, **fields)
                    if delayed_phase == "response":
                        clock[0] += 15.
                    return response

                def delayed_write(payload):
                    result = original_write(payload)
                    if delayed_phase == "journal" and b'"event":"owner_response"' in payload:
                        clock[0] += 15.
                    return result

                with patch("nca.train.jolt_screen.time.monotonic", side_effect=lambda: clock[0]), \
                        patch.object(host, "request", delayed_request), patch.object(journal, "write", delayed_write):
                    with self.assertRaises(ContractError) as caught:
                        owner.request("step", ticks=6)
                self.assertEqual(caught.exception.code, "jolt_owner_exchange_deadline")
                self.assertEqual(owner.pending, 1)
                self.assertEqual(owner.uncertain, [1])
                self.assertEqual(ledger.counts["requested_ticks"], 6)
                self.assertEqual(ledger.counts["confirmed_ticks"], 0)
                self.assertEqual(ledger.counts["uncertain_ticks"], 6)


if __name__ == "__main__":
    unittest.main()
