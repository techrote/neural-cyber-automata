"""Versioned independent readback controls; synthetic owners/models only."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from nca.contracts import ContractError, WireAction, canonical_json, sha256_json
    from nca.controllers.jolt_learned import JoltPolicy, SupervisorRetirement
    from nca.evaluation.durable_journal import DurableJournal, publish_json, validate_record_chain
    from nca.evaluation.verified_journal import VerifiedDurableJournal
    from nca.evaluation.jolt_learning_audit import (
        RUN_SCHEMA, RUN_SCHEMA_V2, SCHEMA_V2, _Audit, _durability_v2,
        _episode_closure_v2, _execution_version, verify_output,
    )
    from test_jolt_learning_audit import journal_fixture, synthetic_episode
    from test_jolt_screen import CutoffFixtureOwner, Journal, audit_saved_episode
    from nca.train.jolt_screen import capture


def setUpModule():
    if TORCH_AVAILABLE:
        torch.set_num_threads(1)


def failed_bundle_v2():
    audit, bundle, costs = synthetic_episode()
    audit.execution_version = 2
    row = bundle["row"]
    row.update(bootstrap_observation_valid=False, bootstrap_reason="not_attempted",
               cancellation_error=None)
    raw = bundle["raw"]
    bundle["raw"] = replace(raw, evaluation={**raw.evaluation, "cancellation_error": None})
    return audit, bundle, costs


def cancellation_fixture():
    before = {"run_id": "synthetic-cancel", "epoch": 1, "actor_id": 0,
              "model": "jev", "profile_version": 1, "tick": 48,
              "control_generation": 0, "observation_id": "synthetic-cancel:1:0:0:48",
              "position": [1., 2.], "velocity": [0., 0.], "rotation": .2,
              "angular_velocity": .1, "parts": [{"id": "core"}],
              "joints": [{"index": 0, "target": .4, "angle": .2}], "suspended": False}
    after = deepcopy(before)
    after.update(control_generation=1, observation_id="synthetic-cancel:1:0:1:48")
    after["joints"][0]["target"] = 0.
    event = SimpleNamespace(kind="cancel", valid=True, next_observation=after,
                            issued_observation=before, tick_before=48, tick_after=48,
                            ticks=(), requested_packet=None)
    raw = SimpleNamespace(events=[event], evaluation={"cancellation_error": None})
    row = {"cancellation_error": None, "error": {"type": "SupervisorRetired", "code": "jolt_supervisor_retired"}}
    return raw, row


@unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning extra is not installed")
class JoltAuditV2Tests(unittest.TestCase):
    def test_versions_must_match_and_unknown_versions_refuse(self):
        for version in (1, 2):
            run = {"schema": f"nca-learn-jolt-002-run/v{version}"}
            declaration = {"schema": f"nca-learn-jolt-002-execution/v{version}"}
            self.assertEqual(_execution_version(run, run, declaration), version)
            with self.assertRaises(ContractError):
                _execution_version(run, {"schema": RUN_SCHEMA_V2 if version == 1 else RUN_SCHEMA}, declaration)
            with self.assertRaises(ContractError):
                _execution_version(run, run, {"schema": f"nca-learn-jolt-002-execution/v{3-version}"})
        with self.assertRaises(ContractError):
            _execution_version({"schema": "nca-learn-jolt-002-run/v3"}, {}, {})

    def test_mismatched_v2_declaration_returns_negative_without_neural_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("worker-result.json", {"schema": RUN_SCHEMA_V2, "status": "failed"}),
                                ("manifest.json", {"schema": RUN_SCHEMA_V2}),
                                ("declaration.json", {"schema": "nca-learn-jolt-002-execution/v1"}),
                                ("config.json", {})):
                publish_json(root / name, value)
            costs = []
            result = verify_output(root, reserve=lambda *args: costs.append(args))
        self.assertEqual(result["schema"], SCHEMA_V2)
        self.assertFalse(result["verified"])
        self.assertFalse(result["complete"])
        self.assertFalse(result["qualified"])
        self.assertEqual(result["checkpoint_loads"], 0)
        self.assertEqual(result["audit_forward"], 0)
        self.assertEqual(costs, [])

    def test_failed_capture_preserves_unattempted_bootstrap_without_extra_forward(self):
        audit, bundle, costs = failed_bundle_v2()
        audit.episode(bundle)
        self.assertEqual(costs["audit_forward"], 6)
        self.assertEqual(audit.bootstrap_forwards, 0)
        self.assertEqual(len(audit.verified_rows), 1)
        self.assertEqual(bundle["row"]["bootstrap_reason"], "not_attempted")

    def test_saved_emergency_failure_replays_with_unsuspended_cancel(self):
        class UnstableOwner(CutoffFixtureOwner):
            def observation(self, tick, action, **kwargs):
                observed = super().observation(tick, action, **kwargs)
                observed["rotation"] = 1.5
                return observed

            def request(self, operation, **fields):
                if operation == "cancel":
                    self.calls.append(operation)
                    self.view = self.observation(self.view["tick"], WireAction(effort=0.).to_dict(),
                                                 generation=1, suspended=False)
                    return {"cancelled": True, "observation": self.view}
                return super().request(operation, **fields)

        owner, journal = UnstableOwner(), Journal()
        ledger = SimpleNamespace(check=lambda: None, reserve=lambda *_args, **_kwargs: None)
        torch.manual_seed(188)
        policy, base = JoltPolicy("residual"), JoltPolicy("bc")
        case = {"id": "synthetic-emergency", "host_seed": 7, "spawn": [0., 0.], "target": [48., 0.]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(SupervisorRetirement) as failure:
                capture(owner, ledger, journal, case,
                    identifier="ppo-seed190103-u0-e0", controller_id="residual-seed190103-initial",
                    kind="residual", stage="ppo", output=output, root=Path(__file__).resolve().parents[1],
                    config={"profile_sha256": sha256_json(owner.profile)}, source_sha256="c" * 64,
                    policy=policy, base=base, generator=torch.Generator().manual_seed(390103), execution_version=2)
            row = failure.exception.row
            self.assertEqual(row["error"]["reason"], "emergency_recovery_deadline")
            self.assertEqual(row["decisions"], 8)
            self.assertEqual(row["consumed_ticks"], 48)
            self.assertEqual(row["bootstrap_reason"], "not_attempted")
            self.assertFalse(row["bootstrap_observation_valid"])
            self.assertIsNone(row["cancellation_error"])
            self.assertTrue(row["owner_closure_confirmed"])
            self.assertFalse(owner.view["suspended"])
            with patch.object(_Audit, "execution_version", 2, create=True):
                audit, costs = audit_saved_episode(output, row, journal, owner, case, policy=policy, base=base)
            self.assertEqual(costs["audit_forward"], 0)
            self.assertEqual(audit.bootstrap_forwards, 0)
            self.assertEqual(owner.calls.count("step"), 8)
            self.assertEqual(owner.calls[-1], "cancel")

    def test_v2_rejects_unattempted_bootstrap_relabelled_as_valid_or_terminal(self):
        for mutation in ("valid", "terminal", "value", "no_error"):
            audit, bundle, _ = failed_bundle_v2()
            row = bundle["row"]
            if mutation == "valid":
                row["bootstrap_observation_valid"] = True
            elif mutation == "terminal":
                row.update(bootstrap_reason="task_terminal", bootstrap_observation_valid=True)
            elif mutation == "value":
                row["bootstrap_value"] = 1.
            else:
                row["error"] = None
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ContractError, "v2 absent bootstrap"):
                audit.episode(bundle)

    def test_v1_default_reporting_is_not_reinterpreted(self):
        audit, bundle, costs = synthetic_episode()
        audit.episode(bundle)
        self.assertEqual(costs["audit_forward"], 6)
        self.assertEqual(bundle["row"]["bootstrap_reason"], "task_terminal")
        self.assertNotIn("cancellation_error", bundle["row"])

    def test_received_v2_bootstrap_still_replays_both_model_states(self):
        audit, bundle, costs = synthetic_episode(cutoff=True)
        audit.execution_version = 2
        bundle["row"]["cancellation_error"] = None
        raw = bundle["raw"]
        bundle["raw"] = replace(raw, evaluation={**raw.evaluation, "cancellation_error": None})
        audit.episode(bundle)
        self.assertEqual(costs["audit_forward"], 8)
        self.assertEqual(audit.bootstrap_forwards, 2)

    def test_failed_cutoff_bootstrap_is_explicit_without_inventing_a_value(self):
        audit, bundle, costs = synthetic_episode(cutoff=True)
        audit.execution_version = 2
        error = {"type": "AdmissionError", "code": "synthetic_readback_ceiling"}
        bundle["row"].update(error=error, cancellation_error=None, bootstrap_value=0.,
            bootstrap_observation_sha256=None, bootstrap_observation_valid=False,
            bootstrap_reason="attempt_failed")
        raw = bundle["raw"]
        bundle["raw"] = replace(raw, evaluation={**raw.evaluation, "cancellation_error": None, "error": error})
        audit.episode(bundle)
        self.assertEqual(costs["audit_forward"], 6)
        self.assertEqual(audit.bootstrap_forwards, 0)
        self.assertEqual(audit.partial[-1]["reason"], "bootstrap_attempt_failed")

    def test_failed_bootstrap_requires_a_received_administrative_cutoff(self):
        audit, bundle, _ = failed_bundle_v2()
        bundle["row"]["bootstrap_reason"] = "attempt_failed"
        with self.assertRaisesRegex(ContractError, "v2 absent bootstrap"):
            audit.episode(bundle)

    def test_native_cancel_can_close_without_suspending_actor(self):
        raw, row = cancellation_fixture()
        self.assertFalse(raw.events[0].next_observation["suspended"])
        self.assertTrue(_episode_closure_v2(raw, row, deadline_missed=False))

    def test_cancellation_requires_matching_generation_pose_targets_and_event(self):
        for mutation in ("generation", "position", "joint_target", "tick", "valid", "packet"):
            raw, row = cancellation_fixture()
            event = raw.events[0]
            if mutation == "generation":
                event.next_observation["control_generation"] = 0
            elif mutation == "position":
                event.next_observation["position"][0] += 1.
            elif mutation == "joint_target":
                event.next_observation["joints"][0]["target"] = .4
            elif mutation == "tick":
                event.tick_after += 1
            elif mutation == "valid":
                event.valid = False
            else:
                event.requested_packet = {"invented": "neutral_step"}
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                _episode_closure_v2(raw, row, deadline_missed=False)

    def test_cleanup_failure_is_separate_and_cannot_claim_closure(self):
        raw, row = cancellation_fixture()
        error = {"type": "ContractError", "code": "morphology_conformance_cancel"}
        raw.evaluation["cancellation_error"] = row["cancellation_error"] = error
        self.assertFalse(_episode_closure_v2(raw, row, deadline_missed=False))
        self.assertEqual(row["error"]["type"], "SupervisorRetired")
        raw.evaluation["cancellation_error"] = None
        with self.assertRaisesRegex(ContractError, "raw cancellation"):
            _episode_closure_v2(raw, row, deadline_missed=False)

    def test_v2_cancellation_diagnostics_are_explicit_and_structured(self):
        for error in ({"type": "ValueError", "code": None, "reason": "unbounded exception text"},
                      {"type": "ValueError", "code": "not a stable code"}):
            raw, row = cancellation_fixture()
            raw.evaluation["cancellation_error"] = row["cancellation_error"] = error
            with self.assertRaisesRegex(ContractError, "bounded structured"):
                _episode_closure_v2(raw, row, deadline_missed=False)
        raw, row = cancellation_fixture()
        del row["cancellation_error"]
        with self.assertRaisesRegex(ContractError, "omitted"):
            _episode_closure_v2(raw, row, deadline_missed=False)

    def test_suspension_only_closes_an_actual_deadline_terminal(self):
        raw, row = cancellation_fixture()
        event = raw.events[0]
        event.kind = "step"
        event.next_observation["suspended"] = True
        self.assertFalse(_episode_closure_v2(raw, row, deadline_missed=False))
        self.assertTrue(_episode_closure_v2(raw, row, deadline_missed=True))

    def test_verified_writer_receipt_is_checked_against_fresh_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "owner-journal.jsonl"
            journal = VerifiedDurableJournal(path)
            journal.write((canonical_json({"event": "synthetic"}) + "\n").encode())
            journal.close()
            receipt, identity = journal.durability_receipt(), validate_record_chain(path)
            system = "Windows" if os.name == "nt" else "Linux"
            _durability_v2(receipt, identity, system=system)
            for field, value in (("closed_verified", False), ("poisoned", True),
                                 ("readback_records", 2), ("record_fsyncs", True),
                                 ("total_fsyncs", 3), ("closed_identity", None)):
                changed = deepcopy(receipt)
                changed[field] = value
                with self.subTest(field=field), self.assertRaises(ContractError):
                    _durability_v2(changed, identity, system=system)
            changed = deepcopy(receipt)
            changed["closed_identity"]["bytes"] += 1
            with self.assertRaisesRegex(ContractError, "closed journal bytes"):
                _durability_v2(changed, identity, system=system)
            changed = deepcopy(receipt)
            changed["closed_identity"]["sha256"] = "a" * 64
            with self.assertRaisesRegex(ContractError, "closed journal bytes"):
                _durability_v2(changed, identity, system=system)
            changed = deepcopy(receipt)
            changed["directory_fsync"] = changed["final_directory_fsync"] = False
            with self.assertRaisesRegex(ContractError, "directory synchronization"):
                _durability_v2(changed, identity, system="Linux")
            _durability_v2(changed, identity, system="Windows")

    def test_legacy_durability_cannot_be_relabelled_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "owner-journal.jsonl"
            journal = DurableJournal(path)
            journal.write((canonical_json({"event": "synthetic"}) + "\n").encode())
            journal.close()
            with self.assertRaisesRegex(ContractError, "incomplete v2"):
                _durability_v2(journal.durability_receipt(), validate_record_chain(path), system="Linux")

    def test_v2_original_journal_replays_as_partial_without_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("test_jolt_learning_audit.DurableJournal", VerifiedDurableJournal):
                audit = journal_fixture(Path(directory))
            audit.execution_version = 2
            audit.manifest["runtime"] = {"system": "Windows" if os.name == "nt" else "Linux"}
            identity = audit.journal()
            self.assertGreater(identity["records"], 0)
            self.assertFalse(audit.complete)
            self.assertTrue(audit.closed)
            self.assertEqual(audit.partial[-1]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
