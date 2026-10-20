"""Offline regression for 39 physical steps followed by a refused local frame.

Synthetic receipts exercise the real recorder and evidence boundaries. This
file never starts a physical host and makes no movement-performance claim.
"""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from nca.adapters.godot import HostError
from nca.contracts import ContractError, WireAction, canonical_json, sha256_json
from nca.data import read_episode
from nca.data.records import json_copy
from nca.evaluation import jolt_arrival as arrival
from nca.evaluation.durable_journal import DurableJournal, seal_owner_journal, validate_record_chain
from nca.evaluation.jolt_character import character_metrics
from nca.evaluation.jolt_decisions import SCHEMA, replay_controller_frames
from test_jolt_evidence_durability import bound_fixture
from test_morphology import fixture


ROOT = Path(__file__).resolve().parents[1]


class SyntheticOwner:
    lane, mode = "act004", "stepped"
    identity = {"lane": "act004", "test_fixture": True}
    capabilities = {"schedule": "synthetic-offline-receipts/v1"}

    def __init__(self, *, uncertain_step=False):
        adapter, initial = fixture()
        self.profile, self.view = json_copy(adapter.profile), json_copy(initial)
        self.view.update(grounded=True, cooldown=0., energy=1., velocity=[0., 0.],
                         last_sequence=-1, action=WireAction(effort=0.).to_dict())
        self.serial, self.steps, self.uncertain_step = 0, 0, uncertain_step

    def request(self, operation, **fields):
        self.serial += 1
        result = {"protocol": "nca-host/1", "id": self.serial, "ok": True}
        if operation == "profile":
            return {**result, "profile": json_copy(self.profile)}
        if operation == "close":
            return {**result, "closed": True}
        if operation == "cancel":
            self.view["control_generation"] += 1
            self.view.update(action={}, lease_remaining=0)
            for joint in self.view["joints"]:
                joint["target"] = 0.
            self.stamp()
            return {**result, "cancelled": True, "observation": json_copy(self.view)}
        traces, start = [], self.view["tick"]
        if operation == "step":
            self.steps += 1
            if self.uncertain_step and self.steps == 40:
                raise HostError("synthetic uncertain transport; no physical host exists")
            packet = fields["commands"][0]
            for index in range(fields["ticks"]):
                tick = self.view["tick"]
                self.view.update(tick=tick + 1, action=packet["action"], last_sequence=packet["sequence"])
                self.stamp()
                traces.append({"tick_before": tick, "tick_after": tick + 1, "world_tick": tick + 1,
                    "received_monotonic_usec": tick * 100 + 1, "completed_monotonic_usec": tick * 100 + 2,
                    "requested": [packet] if index == 0 else [],
                    "admission": [{"accepted": True, "reason": "synthetic"}] if index == 0 else [],
                    "applied": {"action_available": True, "terminal_revocation": False,
                        "action": packet["action"], "linear_impulse": [0., 0.],
                        "angular_impulse": 0., "energy_used": 0.},
                    "passive": {"available": False, "reason": "synthetic"}, "actual_motion": {},
                    "observation": json_copy(self.view), "cellular_sample_tick": tick, "failed": False})
        elif operation != "reset":
            raise AssertionError(operation)
        return {**result, "observation": json_copy(self.view), "tick": self.view["tick"],
            "start_tick": start, "world_tick": self.view["tick"], "actual_ticks": len(traces), "traces": traces,
            "schedule": self.capabilities["schedule"], "reward_kind": "no_task_reward",
            "reason": "synthetic", "failed": False, "terminated": False, "truncated": False, "finished": False}

    def stamp(self):
        self.view["observation_id"] = ":".join(str(self.view[key]) for key in
            ("run_id", "epoch", "actor_id", "control_generation", "tick"))


class SyntheticController:
    controller_id, specification_sha256 = "synthetic-recovery", "a" * 64

    def __init__(self, *, refuse=True):
        self.calls, self.refuse = 0, refuse

    def step(self, observation, goal):
        self.calls += 1
        self.tick = observation.tick
        refused = self.refuse and self.calls == 40
        return SimpleNamespace(action=WireAction(effort=0.), phase="settled" if refused else "approach",
            status="succeeded" if refused else "running", reason="synthetic_local_estimate" if refused else "synthetic")

    def diagnostic_state(self):
        return {"tick": self.tick, "action_lease_ticks": 0 if self.refuse and self.calls == 40 else min(12, 240 - self.tick),
                "supervisor_intervention": False, "supervisor_reason": None}

    def cancel(self):
        pass


class DecisionRefusalTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def capture(self, *, uncertain_step=False, reservation_fsync_failure=False):
        declaration = arrival.read_declaration(ROOT / "configs/baselines/jolt-arrival-diagnostic-v1.json")
        owner = SyntheticOwner(uncertain_step=uncertain_step)
        controller = SyntheticController(refuse=not (uncertain_step or reservation_fsync_failure))
        initial = json_copy(owner.view)
        case = {"id": "synthetic-refusal", "family": "synthetic", "host_seed": 1,
                "spawn": initial["position"], "target": [initial["position"][0] + 100., initial["position"][1]],
                "origin": "previously_open_training"}
        raw = self.root / "owner-journal.jsonl"
        with DurableJournal(raw) as journal:
            budget = arrival.DiagnosticBudget(declaration["budget"], journal)
            budget.counts["host_start_attempts"] = 1
            budget.note("host_start_reserved")
            metered = arrival.MeteredHost(owner, budget)
            original_write = journal.write
            def write(payload):
                record = json.loads(payload)
                if (reservation_fsync_failure and record["event"] == "request_reserved"
                        and record["operation"] == "step" and record["serial"] == 42):
                    with mock.patch("nca.evaluation.durable_journal.os.fsync", side_effect=OSError("synthetic fsync failure")):
                        return original_write(payload)
                return original_write(payload)
            with mock.patch.object(arrival, "_controller", return_value=controller), \
                 mock.patch.object(arrival, "GodotHost", side_effect=AssertionError("no physical host permitted")), \
                 mock.patch.object(journal, "write", side_effect=write), \
                 mock.patch("nca.data.capture.source_identity", return_value=("b" * 40, {}, sha256_json({}))):
                def capture():
                    return arrival._capture(metered, budget, case, arrival.RECOVERY_V2_FAMILIES[2], self.root,
                        {"profile_sha256": sha256_json(owner.profile)}, declaration, ROOT)
                if reservation_fsync_failure:
                    # The failed episode is retained first; its completion
                    # note must still fail against the poisoned journal.
                    with self.assertRaisesRegex(ContractError, "journal_writer_closed"):
                        capture()
                    row = json.loads((self.root / "metrics" / "jolt-recovery-v2-synthetic-refusal.json").read_bytes())
                else:
                    row = capture()
            if not journal.poisoned:
                metered.request("close")
                budget.counts["host_terminations"] = 1
                budget.note("host_closed", termination_confirmed=True)
            counts = budget.snapshot()
        self.raw, self.durability, self.counts, self.row, self.case = raw, journal.durability_receipt(), counts, row, case
        self.episode = read_episode(self.root / row["artifact"])
        self.records = [json.loads(line) for line in raw.read_bytes().splitlines()]
        return controller, owner

    def test_39_steps_and_one_refusal_preserve_prefix_metrics_and_native_cleanup(self):
        controller, owner = self.capture()
        episode, row = self.episode, self.row
        self.assertEqual(controller.calls, 40)
        self.assertEqual(owner.steps, 39)
        self.assertEqual(episode.decisions, 39)
        self.assertEqual(episode.consumed_ticks, 234)
        self.assertEqual(len(replay_controller_frames(episode)), 39)
        self.assertEqual(row["controller_decision_schema"], SCHEMA)
        self.assertEqual(row["controller_decision_accounting"], {"controller_calls_with_frames": 40,
            "physical_step_frames": 39, "refused_before_transport": 1, "owner_response_unrecorded": 0})
        failure = episode.evaluation["controller_decision_failures"][0]
        self.assertEqual(failure["frame"]["tick"], 234)
        self.assertEqual(failure["frame"]["diagnostic_state"]["action_lease_ticks"], 0)
        self.assertEqual(failure["owner_requests_before"], failure["owner_requests_after"])
        self.assertEqual(failure["error"]["code"], "arrival_supervisor_lease")
        self.assertEqual(row["character_metrics"]["consumed_ticks"], 234)
        metrics, terms = arrival.replay_task_metrics(episode)
        self.assertEqual(sha256_json(metrics), sha256_json(row["metrics"]))
        self.assertEqual(sha256_json(terms), sha256_json(episode.evaluation["reward_terms"]))
        self.assertFalse(metrics["success"])
        self.assertTrue(metrics["administratively_capped"])
        self.assertFalse(row["owner_closure_confirmed"])
        self.assertTrue(row["error_cleanup"]["confirmed"])
        self.assertEqual(row["error_cleanup"]["tick"], 234)
        self.assertEqual(row["error_cleanup"]["request_serial"], 42)
        failed = next(i for i, record in enumerate(self.records) if record["event"] == "controller_decision_failed")
        self.assertEqual(self.records[failed + 1]["operation"], "cancel")
        self.assertEqual(validate_record_chain(self.raw)["records"], len(self.records))
        comparison = arrival.summarize([row], [self.case], families=arrival.RECOVERY_V2_FAMILIES)
        self.assertFalse(comparison["complete_valid_pairing"])
        self.assertFalse(comparison["learned_skill_qualified"])
        self.assertFalse(comparison["morphology_admitted"])
        with self.assertRaisesRegex(ContractError, "arrival_journal_episodes"):
            seal_owner_journal(self.raw, expected_counts=self.counts, expected_rows=[row],
                durability=self.durability, validate_owner=arrival.validate_closed_owner_journal,
                families=arrival.RECOVERY_V2_FAMILIES, artifact_root=self.root)
        self.assertFalse((self.root / "owner-journal-seal.json").exists())
        self.assertFalse((self.root / "owner-journal.jsonl.gz").exists())
        self.assertTrue((self.root / "owner-journal.jsonl.gz.pending").exists())

    def test_uncertain_reserved_step_is_not_reclassified_as_a_pretransport_refusal(self):
        self.capture(uncertain_step=True)
        self.assertEqual(len(replay_controller_frames(self.episode)), 39)
        self.assertEqual(self.counts["step_requests"], 40)
        self.assertEqual(self.counts["confirmed_step_ticks"], 234)
        self.assertTrue(self.counts["unconfirmed_tail"])
        self.assertEqual(self.row["controller_decision_accounting"]["refused_before_transport"], 0)
        self.assertEqual(self.row["controller_decision_accounting"]["owner_response_unrecorded"], 1)
        changed = json_copy(self.episode.evaluation)
        changed["controller_decision_failures"][0]["disposition"] = "refused_before_transport"
        with self.assertRaisesRegex(ContractError, "arrival_decision_audit"):
            replay_controller_frames(replace(self.episode, evaluation=changed))

    def test_same_identity_with_changed_refusal_support_or_velocity_is_rejected(self):
        self.capture()
        for field, value in (("grounded", False), ("velocity", [99., 0.])):
            changed = json_copy(self.episode.evaluation)
            changed["controller_decision_failures"][0]["source_observation"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ContractError, "arrival_decision_audit"):
                arrival.replay_task_metrics(replace(self.episode, evaluation=changed))

    def test_fsync_failure_does_not_prove_dispatch_or_permit_cleanup_through_poisoned_writer(self):
        controller, owner = self.capture(reservation_fsync_failure=True)
        self.assertEqual(controller.calls, 40)
        self.assertEqual(owner.steps, 39)
        self.assertEqual(owner.serial, 41)
        self.assertEqual(self.counts["step_requests"], 40)
        self.assertTrue(self.counts["unconfirmed_tail"])
        self.assertTrue(self.durability["poisoned"])
        failure = self.episode.evaluation["controller_decision_failures"][0]
        self.assertEqual(failure["disposition"], "owner_response_unrecorded")
        self.assertEqual(failure["error"]["type"], "OSError")
        self.assertEqual(self.episode.evaluation["decision_failure_journal_error"]["code"], "journal_writer_closed")
        self.assertFalse(self.row["error_cleanup"]["confirmed"])
        self.assertEqual(self.row["error_cleanup"]["error"]["code"], "journal_writer_closed")
        self.assertEqual(len(replay_controller_frames(self.episode)), 39)

    def test_rehashed_refusal_journal_still_needs_exact_physical_source(self):
        self.capture()
        altered = self.root / "altered-journal.jsonl"
        with DurableJournal(altered) as journal:
            for original in self.records:
                record = json_copy(original)
                record.pop("_journal")
                if record["event"] == "controller_decision_failed":
                    record["decision_failure"]["source_observation"]["grounded"] = False
                journal.write((canonical_json(record) + "\n").encode())
        with self.assertRaisesRegex(ContractError, "arrival_decision_audit"):
            seal_owner_journal(altered, expected_counts=self.counts, expected_rows=[self.row],
                durability=journal.durability_receipt(), validate_owner=arrival.validate_closed_owner_journal,
                families=arrival.RECOVERY_V2_FAMILIES, artifact_root=self.root)
        self.assertFalse((self.root / "owner-journal-seal.json").exists())

    def test_refusal_and_native_cleanup_cannot_be_copied_into_completed_episode(self):
        self.capture()
        changed = json_copy(self.episode.evaluation)
        changed.update(error=None, owner_closure_confirmed=True)
        with self.assertRaisesRegex(ContractError, "arrival_decision_audit"):
            replay_controller_frames(replace(self.episode, status="completed", evaluation=changed))
        changed = json_copy(self.episode.evaluation)
        changed["error_cleanup"]["after_observation_sha256"] = "a" * 64
        with self.assertRaisesRegex(ContractError, "arrival_decision_audit"):
            replay_controller_frames(replace(self.episode, evaluation=changed))

    def test_legacy_complete_frames_read_unchanged_and_extra_frame_is_not_normalized(self):
        _, _, rows = bound_fixture(self.root)
        episode = read_episode(self.root / rows[0]["artifact"])
        self.assertEqual(len(replay_controller_frames(episode)), 2)
        arrival.replay_task_metrics(episode)
        changed = {**episode.evaluation, "controller_frames": [*episode.evaluation["controller_frames"],
                    episode.evaluation["controller_frames"][-1]]}
        legacy = replace(episode, evaluation=changed)
        self.assertEqual(len(replay_controller_frames(legacy)), 3)
        with self.assertRaisesRegex(ContractError, "arrival_task_replay"):
            arrival.replay_task_metrics(legacy)
        with self.assertRaisesRegex(ContractError, "character_frames"):
            character_metrics(legacy.initial_observation, (50., 0.), legacy.events, replay_controller_frames(legacy))


if __name__ == "__main__":
    unittest.main()
