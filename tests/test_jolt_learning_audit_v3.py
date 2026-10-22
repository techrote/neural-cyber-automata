"""Independent v3 terminal, qualification and bounded-readback conformance.

All observations are synthetic native-shaped fixtures and all neural models
are untrained. These tests do not open an experiment or sample physical cases.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import gzip
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nca.contracts import ContractError, canonical_json, sha256_json, strict_json_loads
from nca.evaluation.jolt_learning import POLICY_SEEDS, qualify, qualify_v3
from nca.evaluation.jolt_retirement import (
    OUTCOME_SCHEMA, RETIREMENT_SCHEMA, native_cancellation_v3, validate_safe_retirement_v3,
)
from test_jolt_screen import rows


ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from nca.controllers.jolt_learned import JoltPolicy
    from nca.data import read_episode
    from nca.evaluation.durable_journal import publish_json
    from nca.evaluation.jolt_learning_audit import (
        RUN_SCHEMA, RUN_SCHEMA_V2, RUN_SCHEMA_V3, SCHEMA_V3, _Audit, _Reservations,
        _durability_v3, _episode_closure_v3, _execution_version, _journal_identity_v3,
        _raw_episode_v3, _sample, verify_output,
    )
    from nca.train.jolt_budget_v3 import BudgetLedgerV3, reservation_records
    from nca.train.jolt_execution_v3 import COUNTERS, STAGES
    from nca.train.jolt_screen import capture
    from nca.train.jolt_storage_v3 import BoundedVerifiedJournalV3, episode_byte_limit
    from test_jolt_learning_v3 import SyntheticBoundaryOwner, SyntheticRetirementOwner, SyntheticLedger
    from test_jolt_screen import FakeSocket, Journal


def setUpModule():
    if TORCH_AVAILABLE:
        torch.set_num_threads(1)


def paired_v3():
    result = rows()
    for row in result:
        reason = "supported_arrival" if row["metrics"]["success"] else "task_deadline"
        row["metrics"]["reason"] = reason
        row.update(cancellation_error=None, retirement_validation_error=None, cleanup_phase_error=None, retirement=None,
            outcome={"schema": OUTCOME_SCHEMA, "kind": "task_terminal", "complete": True,
                     "success": row["metrics"]["success"], "reason": reason})
    return result


def expected_failure(row):
    row["metrics"].update(success=False, reason="emergency_recovery_deadline", capped_completion_seconds=4.)
    row.update(supervisor_interventions=8,
        outcome={"schema": OUTCOME_SCHEMA, "kind": "safe_retirement", "complete": True,
                 "success": False, "reason": "emergency_recovery_deadline"},
        retirement={"schema": RETIREMENT_SCHEMA, "reason": "emergency_recovery_deadline",
                    "path": "ordinary_attempt"})


class JoltQualificationV3Tests(unittest.TestCase):
    def test_safe_failures_keep_identical_case_denominators_and_measured_metrics(self):
        evidence = paired_v3()
        legacy = qualify(evidence, [f"case{i}" for i in range(16)], evidence_complete=True)
        for row in evidence:
            if row["case_id"] == "case15":
                expected_failure(row)
                # Failure handling must not replace these actual observations
                # with zeros or remove the failed case from a seed mean.
                row["metrics"]["mean_absolute_angle_radians"] = .3
                row["metrics"]["ordinary_trim_reversals_per_second"] = .5
        result = qualify_v3(evidence, [f"case{i}" for i in range(16)], evidence_complete=True)
        self.assertEqual(result["schema"], "nca-jolt-learning-qualification/v3")
        self.assertTrue(result["global_integrity"])
        self.assertEqual(result["safe_retirements"], 14)
        self.assertEqual(result["expected_rows"], result["observed_rows"])
        self.assertEqual(result["authored"]["success"], .5)
        self.assertAlmostEqual(result["authored"]["mean_absolute_angle_radians"], (15 * .1 + .3) / 16)
        self.assertEqual(result["families"]["bc"]["mean_arrival"], legacy["families"]["bc"]["mean_arrival"])
        self.assertTrue(result["qualified"])
        self.assertEqual(result["selected_family"], "bc")
        self.assertFalse(result["terminal_penalty_added"])
        self.assertFalse(result["owner_feel_accepted"])
        self.assertFalse(result["morphology_transfer_admitted"])

    def test_one_shared_safe_failure_rule_covers_every_comparator(self):
        evidence = paired_v3()
        for row in evidence:
            if row["case_id"] == "case15":
                expected_failure(row)
        reference = qualify_v3(evidence, [f"case{i}" for i in range(16)], evidence_complete=True)
        for controller in {row["controller_id"] for row in evidence}:
            changed = deepcopy(evidence)
            row = next(item for item in changed if item["case_id"] == "case15" and item["controller_id"] == controller)
            row["metrics"]["capped_completion_seconds"] = .8
            with self.subTest(controller=controller), self.assertRaises(ContractError):
                qualify_v3(changed, [f"case{i}" for i in range(16)], evidence_complete=True)
        self.assertTrue(reference["global_integrity"])

    def test_missing_latency_is_retained_and_fails_original_every_case_gate(self):
        evidence = paired_v3()
        target = next(row for row in evidence if row["case_id"] == "case15"
                      and row["controller_id"] == "bc-seed190103-final")
        expected_failure(target)
        target["metrics"]["inference_p95_ms"] = None
        result = qualify_v3(evidence, [f"case{i}" for i in range(16)], evidence_complete=True)
        seed = result["families"]["bc"]["per_seed"][0]
        self.assertFalse(seed["gates"]["every_case_latency"])
        self.assertIsNone(seed["case_inference_p95_ms"][-1])
        self.assertEqual(seed["final"]["success"], 14 / 16)
        self.assertFalse(result["families"]["bc"]["qualified"])

    def test_safe_failure_does_not_waive_integrity_or_independent_evidence(self):
        for mutation in ("success", "incomplete", "missing_cancel", "proof_error", "integrity", "admission", "unknown"):
            evidence = paired_v3()
            target = evidence[-1]
            expected_failure(target)
            if mutation == "success":
                target["outcome"]["success"] = target["metrics"]["success"] = True
            elif mutation == "incomplete":
                target["outcome"]["complete"] = False
            elif mutation == "missing_cancel":
                target["owner_closure_confirmed"] = False
            elif mutation == "proof_error":
                target["retirement_validation_error"] = {"type": "ContractError", "code": "jolt_safe_retirement"}
            elif mutation == "integrity":
                target["metrics"]["integrity_failure"] = True
            elif mutation == "admission":
                target["metrics"]["failed_admissions"] = 1
            else:
                target["outcome"]["reason"] = target["metrics"]["reason"] = "other_retirement"
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                qualify_v3(evidence, [f"case{i}" for i in range(16)], evidence_complete=True)
        result = qualify_v3(paired_v3(), [f"case{i}" for i in range(16)], evidence_complete=False)
        self.assertFalse(result["qualified"])
        self.assertFalse(result["families"]["bc"]["gates"]["complete_evidence"])

    def test_incomplete_integrity_error_is_distinct_from_complete_safe_failure(self):
        evidence = paired_v3()
        row = evidence[-1]
        row.update(error={"type": "ContractError", "code": "missing_native_receipt"},
                   cancellation_error={"type": "TimeoutError", "code": None}, owner_closure_confirmed=False)
        row["metrics"]["integrity_failure"] = True
        row["outcome"].update(kind="integrity_error", complete=False)
        result = qualify_v3(evidence, [f"case{i}" for i in range(16)], evidence_complete=False)
        self.assertTrue(result["complete_pairs"])
        self.assertFalse(result["global_integrity"])
        self.assertEqual(result["safe_retirements"], 0)
        self.assertFalse(result["qualified"])

    def test_optional_learner_is_not_imported_by_terminal_or_qualifier(self):
        subprocess.run([sys.executable, "-c", "import sys; import nca.evaluation.jolt_retirement; "
            "import nca.evaluation.jolt_learning; import nca.evaluation.jolt_journal_v3; "
            "assert 'torch' not in sys.modules"], check=True, cwd=ROOT)


@unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning extra is not installed")
class JoltAuditV3Tests(unittest.TestCase):
    def captured(self, output, *, kind="residual", start_emergency=0, maximum_decisions=40, prior_decisions=0,
                 owner=None):
        owner = owner or SyntheticRetirementOwner(start_emergency=start_emergency)
        ledger, journal = SyntheticLedger(), Journal()
        torch.manual_seed(188)
        policy = JoltPolicy(kind) if kind in ("bc", "residual") else None
        base = JoltPolicy("bc") if kind == "residual" else None
        stage = "teacher" if kind == "authored" else "ppo" if kind == "residual" else "open"
        controller = f"{kind}-seed190103-initial" if policy is not None else kind
        identifier = f"ppo-seed190103-u0-e{int(bool(prior_decisions))}" if stage == "ppo" else "synthetic-v3-" + kind
        case = {"id": "synthetic-emergency-v3", "host_seed": 7, "spawn": [0., 0.], "target": [48., 0.]}
        sample, row, sequence = capture(owner, ledger, journal, case,
            identifier=identifier, controller_id=controller, kind=kind, stage=stage, output=output,
            root=ROOT, config={"profile_sha256": sha256_json(owner.profile)}, source_sha256="c" * 64,
            policy=policy, base=base, generator=torch.Generator().manual_seed(390103),
            maximum_decisions=maximum_decisions, execution_version=3,
            case_stream_entry={"seed": 190103, "index": int(bool(prior_decisions)), "update": 0} if stage == "ppo" else None)
        raw = read_episode(output / row["artifact"])
        costs = Counter()
        audit = _Audit.__new__(_Audit)
        audit.execution_version = 3
        audit.output, audit.reserve = output, lambda name, amount=1: costs.update({name: amount})
        audit.manifest = {"source_sha256": "c" * 64, "source_revision": raw.learner_revision,
                          "source_files": dict(raw.source_files)}
        audit.config = {"profile_sha256": sha256_json(owner.profile),
                        "cases": {"teacher": [case], "open": [case], "sealed": [case]}}
        audit.host_identity = owner.identity
        audit.models = {} if policy is None else {controller: (policy, base)}
        audit.forwards = audit.runtime_forwards = audit.bootstrap_forwards = audit.executions = 0
        audit.action_generators = {190103: torch.Generator().manual_seed(390103)}
        audit.ppo_groups = {(190103, 0): [(None, SimpleNamespace(decisions=tuple(range(prior_decisions))))]} if prior_decisions else {}
        audit.teacher_rows, audit.partial, audit.verified_rows = [], [], []
        bundle = audit._read_episode(row)
        bundle["proposals"] = [item for item in journal.records() if item["event"] == "decision_proposed"]
        return SimpleNamespace(audit=audit, bundle=bundle, row=row, sample=sample,
                               ledger=ledger, costs=costs, owner=owner, journal=journal, sequence=sequence)

    def test_original_arrival_and_native_deadline_roundtrip_without_safe_failure(self):
        for boundary, expected_reason, expected_ticks in (("arrival", "arrival", 12),
                                                          ("deadline", "task_deadline", 240)):
            for kind in ("authored", "idle", "bc", "residual"):
                with self.subTest(boundary=boundary, kind=kind), tempfile.TemporaryDirectory() as directory:
                    fixture = self.captured(Path(directory), kind=kind, owner=SyntheticBoundaryOwner(boundary))
                    fixture.audit.episode(fixture.bundle)
                self.assertEqual(fixture.row["outcome"]["kind"], "task_terminal")
                self.assertEqual(fixture.row["outcome"]["reason"], expected_reason)
                self.assertIsNone(fixture.row["retirement"])
                self.assertEqual(fixture.audit.verified_rows[0]["consumed_ticks"], expected_ticks)
                self.assertEqual(fixture.audit.verified_rows[0]["metrics_sha256"], sha256_json(fixture.row["metrics"]))
                self.assertEqual(fixture.audit.executions, fixture.ledger.counts["checkpoint_executions"])
                self.assertEqual(fixture.audit.runtime_forwards, fixture.ledger.counts["inference_forward"])
                self.assertEqual(fixture.audit.bootstrap_forwards, 0)

    def test_native_deadline_cannot_be_relabelled_as_coincident_emergency_failure(self):
        for kind in ("authored", "idle", "bc", "residual"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                fixture = self.captured(Path(directory), kind=kind,
                                        owner=SyntheticBoundaryOwner("deadline_emergency"))
                fixture.audit.episode(fixture.bundle)
                self.assertEqual(fixture.row["supervisor_interventions"], 8)
                self.assertEqual(fixture.row["outcome"]["reason"], "task_deadline")
                self.assertTrue(fixture.sample.decisions[-1].event.terminated)
                self.assertEqual(fixture.audit.executions, fixture.ledger.counts["checkpoint_executions"])
                # A complete consumed task terminal cannot be exchanged for
                # an unconsumed, would-be 48-tick supervisory attempt.
                from nca.data.jolt_samples import reconstructed_history
                final = fixture.sample.decisions[-1]
                with self.assertRaisesRegex(ContractError, "earlier task terminal"):
                    validate_safe_retirement_v3(fixture.sample.decisions, final.event.next_observation,
                        fixture.bundle["goal"], fixture.bundle["context"], reconstructed_history(final),
                        "ordinary_attempt", after_cancel=final.event.next_observation)

    def test_independent_replay_preserves_quiet_exit_before_emergency_expiry(self):
        for kind in ("authored", "idle", "bc", "residual"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                fixture = self.captured(Path(directory), kind=kind, owner=SyntheticBoundaryOwner("quiet_exit"))
                fixture.audit.episode(fixture.bundle)
            self.assertEqual(fixture.row["supervisor_interventions"], 8)
            self.assertEqual(fixture.sample.decisions[8].event.tick_before, 48)
            self.assertFalse(fixture.sample.decisions[8].overridden)
            self.assertIsNone(fixture.row["retirement"])
            self.assertEqual(fixture.audit.runtime_forwards, fixture.ledger.counts["inference_forward"])
            self.assertEqual(fixture.audit.executions, fixture.ledger.counts["checkpoint_executions"])
            self.assertEqual(fixture.audit.bootstrap_forwards, 0)
            self.assertEqual(fixture.audit.verified_rows[0]["consumed_ticks"], 240)

    def test_quiet_exit_cutoff_replays_actual_value_and_masked_gae_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.captured(Path(directory), maximum_decisions=8, prior_decisions=248,
                                    owner=SyntheticBoundaryOwner("quiet_exit"))
            fixture.audit.episode(fixture.bundle)
        self.assertEqual(fixture.row["outcome"]["kind"], "administrative_cutoff")
        self.assertNotEqual(fixture.row["bootstrap_value"], 0.)
        self.assertEqual(fixture.sequence.bootstrap_value, 0.)
        self.assertEqual(fixture.audit.bootstrap_forwards, 2)
        self.assertEqual(fixture.audit.runtime_forwards, 0)
        self.assertEqual(fixture.audit.executions, 8)
        self.assertEqual(fixture.costs["audit_forward"], 2)
        self.assertEqual(fixture.audit.verified_rows[0]["consumed_ticks"], 48)
        self.assertEqual(fixture.audit.verified_rows[0]["metrics_sha256"], sha256_json(fixture.row["metrics"]))

    def test_every_comparator_replays_complete_unsuccessful_native_retirement(self):
        for kind in ("authored", "idle", "bc", "residual"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                fixture = self.captured(Path(directory), kind=kind)
                with patch.object(JoltPolicy, "forward", side_effect=AssertionError("no emergency model work")):
                    fixture.audit.episode(fixture.bundle)
                self.assertEqual(fixture.audit.executions, 9 if kind in ("bc", "residual") else 0)
                self.assertEqual(fixture.audit.executions, fixture.ledger.counts["checkpoint_executions"])
                self.assertEqual(fixture.costs["audit_forward"], 0)
                self.assertEqual(fixture.audit.bootstrap_forwards, 0)
                self.assertEqual(len(fixture.audit.verified_rows), 1)
                self.assertEqual(fixture.audit.verified_rows[0]["consumed_ticks"], 48)
                self.assertEqual(fixture.audit.verified_rows[0]["metrics_sha256"], sha256_json(fixture.row["metrics"]))
                self.assertEqual(fixture.owner.calls[-1], "cancel")
                self.assertFalse(fixture.owner.view["suspended"])

    def test_cutoff_retirement_adds_neither_model_forward_nor_execution_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.captured(Path(directory), maximum_decisions=8, prior_decisions=248)
            fixture.audit.episode(fixture.bundle)
        self.assertEqual(fixture.row["retirement"]["path"], "cutoff_peek")
        self.assertEqual(fixture.audit.executions, 8)
        self.assertEqual(fixture.audit.executions, fixture.ledger.counts["checkpoint_executions"])
        self.assertEqual(fixture.costs["audit_forward"], 0)
        self.assertEqual(fixture.audit.bootstrap_forwards, 0)
        self.assertTrue(fixture.sample.decisions[-1].event.truncated)
        self.assertFalse(fixture.sample.decisions[-1].event.terminated)

    def test_only_actual_ordinary_recurrent_models_are_replayed_before_retirement(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.captured(Path(directory), start_emergency=12)
            fixture.audit.episode(fixture.bundle)
        self.assertEqual(len(fixture.sample.decisions), 10)
        self.assertEqual(fixture.audit.executions, 11)
        self.assertEqual(fixture.costs["audit_forward"], 4)
        self.assertEqual(fixture.audit.runtime_forwards, 4)
        self.assertEqual(fixture.audit.bootstrap_forwards, 0)
        self.assertEqual(fixture.row["policy_values"][2:], [None] * 8)
        self.assertAlmostEqual(fixture.row["metrics"]["return"], sum(item.event.reward for item in fixture.sample.decisions))
        retained = fixture.audit.ppo_groups[(190103, 0)][0][1]
        self.assertFalse(hasattr(retained.decisions[0].event, "ticks"))

    def test_non_neural_retirement_replay_cannot_consume_the_publication_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.captured(Path(directory), kind="authored")
            checks = []

            def original_audit_deadline():
                checks.append(True)
                if len(checks) == 4:
                    raise ContractError("jolt_v3_phase_deadline", "synthetic original audit interval exhausted")

            fixture.audit.check = original_audit_deadline
            with self.assertRaisesRegex(ContractError, "original audit interval"):
                fixture.audit.episode(fixture.bundle)
        self.assertEqual(fixture.costs["audit_forward"], 0)
        self.assertEqual(fixture.audit.verified_rows, [])
        self.assertEqual(len(checks), 4)

    def test_retirement_cannot_invent_bootstrap_masked_value_or_short_cost(self):
        for mutation in ("value", "observation", "valid", "reason", "masked_value", "short_cost", "failed_proof"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = self.captured(Path(directory))
                row = fixture.row
                if mutation == "value":
                    row["bootstrap_value"] = 1.
                elif mutation == "observation":
                    row["bootstrap_observation_sha256"] = "a" * 64
                elif mutation == "valid":
                    row["bootstrap_observation_valid"] = True
                elif mutation == "reason":
                    row["bootstrap_reason"] = "received_cutoff_observation"
                elif mutation == "masked_value":
                    row["policy_values"][0] = 0.
                elif mutation == "short_cost":
                    row["metrics"]["capped_completion_seconds"] = .8
                else:
                    row["retirement_validation_error"] = {"type": "ContractError", "code": "jolt_safe_retirement"}
                with self.assertRaises(ContractError):
                    fixture.audit.episode(fixture.bundle)

    def test_independent_replay_rejects_unbound_final_observation_and_fake_path(self):
        for mutation in ("pre_cancel", "reason", "path"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = self.captured(Path(directory))
                raw = fixture.bundle["raw"]
                last = raw.events[-1]
                if mutation == "pre_cancel":
                    changed = strict_json_loads(canonical_json(last.issued_observation))
                    changed["position"] = [3., 0.]
                    # This corruption is already rejected by the native
                    # record's independently validated policy projection.
                    with self.assertRaises(ContractError):
                        last = replace(last, issued_observation=changed)
                        fixture.bundle["raw"] = replace(raw, events=(*raw.events[:-1], last))
                        fixture.audit.episode(fixture.bundle)
                    continue
                elif mutation == "reason":
                    fixture.bundle["raw"] = replace(raw, events=(*raw.events[:-1], replace(last, reason="unknown_retirement")))
                else:
                    changed = deepcopy(fixture.row["retirement"])
                    changed["path"] = "cutoff_peek"
                    fixture.row["retirement"] = changed
                    fixture.bundle["raw"] = replace(raw, evaluation={**raw.evaluation, "retirement": changed})
                    fixture.bundle["sample"] = replace(fixture.sample, retirement=changed)
                with self.assertRaises(ContractError):
                    fixture.audit.episode(fixture.bundle)

    def test_native_cancel_requires_same_incarnation_and_cleared_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.captured(Path(directory), kind="authored")
        cancel = fixture.bundle["raw"].events[-1]
        receipt = native_cancellation_v3(cancel.issued_observation, cancel.next_observation)
        self.assertTrue(receipt["same_actor_incarnation"])
        for field, value in (("pending_input_edges", 1), ("lease_remaining", 1), ("action", {"effort": 0}),
                             ("run_id", "another-owner"), ("tick", 49), ("control_generation", 0),
                             ("position", [1., 2.]), ("fault", "controller_fault"), ("lease_remaining", False)):
            changed = strict_json_loads(canonical_json(cancel.next_observation))
            changed[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ContractError):
                native_cancellation_v3(cancel.issued_observation, changed)

    def test_history_requires_all_original_intervals_and_unchanged_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.captured(Path(directory), kind="authored")
        bundle, proof = fixture.bundle, fixture.row["retirement"]
        final = bundle["raw"].events[-1]
        for records, context, history in (
                (fixture.sample.decisions[1:], bundle["context"], proof["history"]),
                (fixture.sample.decisions[:-1], bundle["context"], proof["history"]),
                (fixture.sample.decisions, {**proof["invocation"], "deadline_tick": 288}, proof["history"]),
                (fixture.sample.decisions, bundle["context"], {**proof["history"], "intervention": False})):
            with self.assertRaises(ContractError):
                validate_safe_retirement_v3(records, final.issued_observation, bundle["goal"],
                    context, history, "ordinary_attempt", after_cancel=final.next_observation)

    def test_explicit_retirement_proof_failure_does_not_erase_valid_native_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.captured(Path(directory), kind="authored")
        row = deepcopy(fixture.row)
        row.update(error={"type": "SupervisorRetirement", "code": "jolt_supervisor_retired"},
                   retirement=None, retirement_validation_error={"type": "ContractError", "code": "jolt_safe_retirement"})
        row["outcome"].update(kind="integrity_error", complete=False)
        raw = replace(fixture.bundle["raw"], evaluation={**fixture.bundle["raw"].evaluation,
            "retirement_validation_error": row["retirement_validation_error"]})
        self.assertTrue(_episode_closure_v3(raw, row, deadline_missed=False))
        row["outcome"]["complete"] = True
        with self.assertRaises(ContractError):
            _episode_closure_v3(raw, row, deadline_missed=False)

    def test_cross_version_evidence_refuses_before_any_neural_readback(self):
        run = {"schema": RUN_SCHEMA_V3}
        self.assertEqual(_execution_version(run, run, {"schema": "nca-learn-jolt-003-execution/v3"}), 3)
        for old in (RUN_SCHEMA, RUN_SCHEMA_V2):
            with self.assertRaises(ContractError):
                _execution_version(run, {"schema": old}, {"schema": "nca-learn-jolt-003-execution/v3"})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name, content in (("worker-result.json", {**run, "status": "failed"}),
                    ("manifest.json", run), ("declaration.json", {"schema": "nca-learn-jolt-002-execution/v2"}),
                    ("config.json", {})):
                publish_json(output / name, content)
            costs = []
            result = verify_output(output, reserve=lambda *args: costs.append(args))
        self.assertEqual(result["schema"], SCHEMA_V3)
        self.assertFalse(result["verified"])
        self.assertEqual(result["checkpoint_loads"], 0)
        self.assertEqual(result["audit_forward"], 0)
        self.assertEqual(costs, [])

    def test_v3_sample_cannot_be_read_as_legacy_and_expansion_has_coupled_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            fixture = self.captured(output, kind="authored")
            path = output / fixture.row["sample_artifact"]
            with self.assertRaises(ContractError):
                _sample(path, execution_version=2)
            oversized = output / "oversized.json.gz"
            with gzip.open(path, "rb") as source:
                original = source.read()
            oversized.write_bytes(gzip.compress(original + b" " * episode_byte_limit(8), mtime=0))
            with self.assertRaisesRegex(ContractError, "per-decision"):
                _sample(oversized, execution_version=3)
            self.assertEqual(_raw_episode_v3(output / fixture.row["artifact"]).consumed_ticks, 48)


@unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning extra is not installed")
class JoltAuditEnvelopeV3Tests(unittest.TestCase):
    def test_streaming_reservations_reconstruct_prefixes_and_phase_costs(self):
        now = [0.]
        budget = strict_json_loads((ROOT / "configs/learning/jolt-learn-003-v1.json").read_bytes())["budget"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            ledger = BudgetLedgerV3(budget, started=0., spool_path=output / "live.jsonl", monotonic=lambda: now[0])
            for _ in range(19):
                ledger.reserve("inference_forward")
            prefix = ledger.snapshot()
            for _ in range(11):
                ledger.reserve("checkpoint_executions")
            ledger.enter_phase("closure")
            ledger.reserve("host_terminations", cleanup=True)
            ledger.enter_phase("audit")
            ledger.reserve("audit_forward", cleanup=True)
            final = ledger.snapshot()
            path = output / "budget-reservations.json"
            ledger.publish_reservations(path)
            stream = reservation_records(path)
            reconstructed = _Reservations(budget, stream, final)
            self.assertFalse(hasattr(stream, "__getitem__"))
            self.assertEqual(reconstructed.counts["inference_forward"], 19)
            self.assertEqual(reconstructed.counts["checkpoint_executions"], 11)
            self.assertEqual(reconstructed.snapshot(prefix)["inference_forward"], 19)
            self.assertEqual(reconstructed.snapshot(final)["audit_forward"], 1)
            changed = deepcopy(final)
            changed["phase_history"][1]["elapsed_seconds"] = 1.
            with self.assertRaises(ContractError):
                reconstructed.snapshot(changed)
            ledger.enter_phase("publication")
            ledger.close()

    def test_v3_durable_reader_requires_original_byte_and_count_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            journal = BoundedVerifiedJournalV3(path)
            journal.write((canonical_json({"event": "synthetic"}) + "\n").encode())
            journal.close()
            receipt, identity = journal.durability_receipt(), _journal_identity_v3(path)
            _durability_v3(receipt, identity, system="Linux")
            for mutation in ("schema", "maximum_record_bytes", "maximum_records", "maximum_stream_bytes", "closed_identity"):
                changed = deepcopy(receipt)
                if mutation == "schema":
                    changed[mutation] = "nca-owner-journal-durability/v2"
                elif mutation == "closed_identity":
                    changed[mutation]["sha256"] = "a" * 64
                else:
                    changed[mutation] += 1
                with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                    _durability_v3(changed, identity, system="Linux")
            path.write_bytes(path.read_bytes().replace(b"synthetic", b"synthetix"))
            with self.assertRaises(ContractError):
                _journal_identity_v3(path)


@unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning extra is not installed")
class JoltAuditJournalV3Tests(unittest.TestCase):
    def captured_journal(self, output, *, omit_retirement=False):
        """One complete safe episode inside a deliberately partial allocation."""
        from nca.train.jolt_screen import MeteredOwner, note

        class CorrelatedOwner(SyntheticRetirementOwner):
            identity = {"lane": "act004", "mode": "stepped", "schedule": "synthetic-cutoff-only"}

            def __init__(self):
                super().__init__()
                self._channel = SimpleNamespace(connection=FakeSocket())
                self.serial = 0

            def request(self, operation, **fields):
                self.serial += 1
                result = {"closed": True} if operation == "close" else super().request(operation, **fields)
                return {**result, "id": self.serial, "ok": True}

        class OriginalJournal(BoundedVerifiedJournalV3):
            def write(self, payload):
                # A valid chain alone must not excuse a missing original
                # retirement boundary, even with otherwise matching files.
                if omit_retirement and strict_json_loads(payload)["event"] == "safe_retirement_verified":
                    return len(payload)
                return super().write(payload)

        budget = strict_json_loads((ROOT / "configs/learning/jolt-learn-003-v1.json").read_bytes())["budget"]
        ledger = BudgetLedgerV3(budget, spool_path=output / "reservation-ledger.jsonl")
        journal = OriginalJournal(output / "owner-journal.jsonl")
        host = CorrelatedOwner()
        owner = MeteredOwner(host, ledger, journal)
        ledger.reserve("attempts")
        ledger.reserve_many({"host_starts": 1, "host_start_attempts": 1})
        note(journal, "host_start_reserved", counts=ledger.snapshot())
        note(journal, "host_started", identity=host.identity)
        case = {"id": "synthetic-emergency-v3", "host_seed": 7, "spawn": [0., 0.], "target": [48., 0.]}
        _, row, _ = capture(owner, ledger, journal, case, identifier="synthetic-v3-authored",
            controller_id="authored", kind="authored", stage="teacher", output=output, root=ROOT,
            config={"profile_sha256": sha256_json(host.profile)}, source_sha256="c" * 64, execution_version=3)
        ledger.enter_phase("closure")
        owner.request("close")
        ledger.reserve("host_terminations", cleanup=True)
        note(journal, "host_terminated", native_close_confirmed=True, process_exited=True)
        note(journal, "worker_closed", status="failed", counters=ledger.snapshot())
        journal.close()
        closure = {"native_close_confirmed": owner.closed_native, "pending_owner_request": owner.pending,
            "uncertain_owner_requests": owner.uncertain, "owner_requests": owner.requests,
            "requested_ticks": owner.requested_ticks, "confirmed_ticks": owner.confirmed_ticks,
            "refused_ticks": owner.refused_ticks, "uncertain_ticks": owner.uncertain_ticks,
            "durability": journal.durability_receipt()}
        worker = {"schema": RUN_SCHEMA_V3, "status": "failed", "results": [row],
                  "checkpoints": [], "learning": [], "closure": closure}
        raw = read_episode(output / row["artifact"])
        manifest = {"schema": RUN_SCHEMA_V3, "source_sha256": "c" * 64,
                    "source_revision": raw.learner_revision, "source_files": dict(raw.source_files),
                    "host_identity": host.identity, "runtime": {"system": "Linux"}}
        config = {"budget": budget, "profile_sha256": sha256_json(host.profile),
                  "cases": {"teacher": [case], "open": [], "sealed": []}}
        for name, content in (("worker-result.json", worker), ("manifest.json", manifest),
                              ("config.json", config), ("declaration.json", {})):
            publish_json(output / name, content)
        ledger.publish_reservations(output / "budget-reservations.json")
        audit = _Audit(output, reserve=lambda *_: None)
        audit.episodes = {row["id"]: row}
        audit.budget = _Reservations(budget, reservation_records(output / "budget-reservations.json"), ledger.snapshot())
        ledger.enter_phase("audit")
        ledger.enter_phase("publication")
        ledger.close()
        return audit

    def test_original_native_journal_replays_safe_episode_without_promoting_partial_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = self.captured_journal(Path(directory))
            identity = audit.journal()
        self.assertGreater(identity["records"], 20)
        self.assertEqual(len(audit.verified_rows), 1)
        self.assertEqual(audit.verified_rows[0]["consumed_ticks"], 48)
        self.assertFalse(audit.complete)
        self.assertEqual(audit.partial[-1]["reason"], "incomplete_allocation")
        self.assertEqual(audit.budget.counts["owner_cancel"], 1)
        self.assertEqual(audit.budget.counts["episodes_completed"], 1)
        self.assertEqual(audit.budget.counts["episodes_aborted"], 0)

    def test_complete_native_receipts_do_not_excuse_missing_durable_retirement_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = self.captured_journal(Path(directory), omit_retirement=True)
            with self.assertRaisesRegex(ContractError, "durable retirement"):
                audit.journal()

    def test_journal_rejects_unfrozen_case_order_before_reading_any_episode(self):
        config = strict_json_loads((ROOT / "configs/learning/jolt-learn-003-v1.json").read_bytes())
        original = config["ppo_case_stream"]["seeds"]["190103"][0]
        another = next(case["id"] for case in config["cases"]["teacher"] if case["id"] != original)
        for mutation in ("index", "case", "update", "episode_index"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                journal = BoundedVerifiedJournalV3(output / "owner-journal.jsonl")
                records = [{"event": "host_started", "identity": {"lane": "act004", "mode": "stepped"}},
                    {"event": "episode_started", "id": "ppo-seed190103-u0-e0", "case_id": original,
                     "stage": "ppo", "controller_id": "residual-seed190103-initial",
                     "case_stream_entry": {"seed": 190103, "index": 0, "update": 0}}]
                if mutation == "index":
                    records[1]["case_stream_entry"]["index"] = 1
                elif mutation == "case":
                    records[1]["case_id"] = another
                elif mutation == "update":
                    records[1]["case_stream_entry"]["update"] = 1
                else:
                    records[1]["id"] = "ppo-seed190103-u0-e1"
                for record in records:
                    journal.write((canonical_json(record) + "\n").encode())
                journal.close()
                for name, content in (("worker-result.json", {"schema": RUN_SCHEMA_V3, "status": "failed",
                        "closure": {"durability": journal.durability_receipt()}}),
                        ("manifest.json", {"host_identity": {}, "runtime": {"system": "Linux"}}),
                        ("config.json", config), ("declaration.json", {})):
                    publish_json(output / name, content)
                audit = _Audit(output, reserve=lambda *_: self.fail("case provenance must precede neural replay"))
                with self.assertRaisesRegex(ContractError, "PPO"):
                    audit.journal()


if __name__ == "__main__":
    unittest.main()
