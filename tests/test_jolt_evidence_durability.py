"""Persistence, transport and independent physical-record binding controls."""

import copy
from dataclasses import replace
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from nca.contracts import ContractError, WireAction, canonical_json, sha256_json
from nca.data import (ActionSource, CaptureEvent, EpisodeRecord, PolicyContext, TickRecord,
                      policy_view, read_episode, write_episode)
from nca.evaluation.durable_journal import (DurableJournal, file_identity, publish_json,
    seal_owner_journal, validate_record_chain, verify_sealed_journal)
from nca.evaluation.jolt_arrival import (DiagnosticBudget, FAMILIES, MeteredHost,
    replay_task_metrics, validate_closed_owner_journal)
from nca.evaluation.jolt_character import character_metrics
from test_jolt_arrival_journal import closed_journal, observation


def bound_fixture(folder):
    """Tiny actual-format records; no physical game or policy is executed."""
    records, counts, rows = closed_journal()
    for family_index, row in enumerate(rows):
        source = ActionSource(row["family"], "scripted", "synthetic-reference", "a" * 64)
        context = PolicyContext("synthetic-goal", "directed_hop", (50., 0.), .8, 4., 4.)
        action = WireAction(move=(.5, 0.)).to_dict()
        def view(tick, generation=0, suspended=False):
            return {**observation(tick, generation=generation, suspended=suspended),
                    "version": 2, "representation": "articulated-rapier", "position": [float(tick), 0.],
                    "velocity": [1., 0.], "grounded": True, "cooldown": 0., "energy": 1.,
                    "blocked": False, "action": action}
        initial, events, frames = view(0), [], []
        for index, start in enumerate((0, 6)):
            terminal = family_index == 1 and start == 6
            before = view(start)
            packet = {"version": 2, "kind": "physical_wire_action", "run_id": "synthetic", "epoch": 1,
                      "actor_id": 0, "sequence": index, "observation_id": before["observation_id"],
                      "observed_tick": start, "lease_ticks": 12, "action": action}
            ticks = []
            for tick in range(start, start + 6):
                final = terminal and tick == start + 5
                after = view(tick + 1, generation=int(final), suspended=final)
                applied = {"action_available": not final, "terminal_revocation": final,
                           "action": None if final else action, "linear_impulse": [.1, 0.],
                           "angular_impulse": 0., "energy_used": 0.}
                ticks.append(TickRecord(tick, tick + 1, tick + 1, 100 + tick, 101 + tick,
                    (packet,) if tick == start else (), ({"accepted": True, "reason": ""},) if tick == start else (),
                    packet, source.id, action, applied, {"available": False, "reason": "unavailable"},
                    {"velocity": [1., 0.]}, after, tick))
            events.append(CaptureEvent("step", index, 0, start, start + 6, start * 1000, (start + 6) * 1000,
                source, before, policy_view(before), requested_packet=packet, source_observation=before,
                context=context, ticks=tuple(ticks), next_observation=ticks[-1].observation,
                truncated=terminal))
            frames.append({"tick": start, "phase": "approach", "status": "running", "reason": "synthetic",
                           "action": action, "inference_ns": 100, "diagnostic_state": None})
        character = character_metrics(initial, context.target, events, frames)
        episode = EpisodeRecord(f"synthetic-{family_index}", {"lane": "act004", "test_fixture": True}, "b" * 40,
            sha256_json({}), {"seed": 1}, {"synthetic": True}, initial, tuple(events), "completed", "synthetic",
            "synthetic-family", "evaluation", evaluation={"metrics": row["metrics"],
                "character_metrics": character, "controller_frames": frames})
        task_metrics, task_terms = replay_task_metrics(episode)
        row["metrics"] = task_metrics
        episode = replace(episode, evaluation={**episode.evaluation, "metrics": task_metrics,
            "reward_terms": task_terms, "task_reward_schema": "jolt-arrival-best-progress/v1"})
        path = folder / "episodes" / f"synthetic-{family_index}.json.gz"
        row.update(artifact=path.relative_to(folder).as_posix(), artifact_sha256=write_episode(path, episode),
                   record_sha256=episode.sha256, character_metrics=character)
        active = False
        next_step = 0
        previous_operation = None
        for record in records:
            if record["event"] == "episode_started":
                active = record["family"] == row["family"]
            if not active:
                continue
            if record["event"] == "request_reserved":
                previous_operation = record["operation"]
                if previous_operation == "step":
                    record["fields"]["commands"] = [events[next_step].requested_packet.to_dict()
                        if hasattr(events[next_step].requested_packet, "to_dict") else json.loads(canonical_json(events[next_step].requested_packet))]
            elif record["event"] == "response_received":
                response = record["response"]
                if previous_operation == "reset":
                    response["observation"] = initial
                elif previous_operation == "step":
                    event = events[next_step]
                    response["observation"] = json.loads(canonical_json(event.next_observation))
                    response["traces"] = [{key: value for key, value in tick.to_dict().items()
                        if key not in ("consumed_request", "source_id", "consumed_action", "valid")}
                        for tick in event.ticks]
                    next_step += 1
                elif previous_operation == "cancel":
                    response["observation"] = view(12, generation=1)
            elif record["event"] == "episode_completed":
                record["row"] = copy.deepcopy(row)
    return records, counts, rows


class JournalDurabilityTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.raw = self.root / "owner-journal.jsonl"

    def write_records(self, records):
        journal = DurableJournal(self.raw)
        with journal:
            for record in records:
                journal.write((canonical_json(record) + "\n").encode())
        return journal.durability_receipt()

    def sealed(self):
        records, counts, rows = bound_fixture(self.root)
        durability = self.write_records(records)
        result = seal_owner_journal(self.raw, expected_counts=counts, expected_rows=rows,
            durability=durability, validate_owner=validate_closed_owner_journal,
            families=FAMILIES, artifact_root=self.root)
        return result, counts, rows

    def verify(self, counts, rows):
        return verify_sealed_journal(self.root, expected_counts=counts, expected_rows=rows,
            validate_owner=validate_closed_owner_journal, families=FAMILIES)

    def test_reservation_is_fsynced_before_transport_and_response_before_confirmation(self):
        journal = DurableJournal(self.raw)
        with journal:
            limits = {"maximum_wall_seconds": 10, "maximum_host_requests": 4,
                      "maximum_decisions": 1, "maximum_requested_ticks": 6}
            budget = DiagnosticBudget(limits, journal)
            class Owner:
                def request(inner, operation, **fields):
                    self.assertEqual(validate_record_chain(self.raw)["records"], 1)
                    self.assertEqual(journal.fsyncs, 1)
                    self.assertEqual(budget.counts["confirmed_step_ticks"], 0)
                    return {"actual_ticks": 6}
            MeteredHost(Owner(), budget).request("step", ticks=6)
            self.assertEqual(journal.fsyncs, 2)
            self.assertEqual(validate_record_chain(self.raw)["records"], 2)
            self.assertEqual(budget.counts["confirmed_step_ticks"], 6)

    def test_failed_persistence_never_dispatches_and_poisoned_writer_cannot_retry(self):
        journal = DurableJournal(self.raw)
        host = mock.Mock()
        budget = DiagnosticBudget({"maximum_wall_seconds": 10, "maximum_host_requests": 3}, journal)
        try:
            with mock.patch("nca.evaluation.durable_journal.os.fsync", side_effect=OSError("synthetic storage failure")):
                with self.assertRaises(OSError):
                    MeteredHost(host, budget).request("cancel")
            self.assertTrue(journal.poisoned)
            self.assertTrue(budget.snapshot()["unconfirmed_tail"])
            with self.assertRaisesRegex(ContractError, "journal_writer_closed"):
                MeteredHost(host, budget).request("cancel")
            host.request.assert_not_called()
        finally:
            journal.close()

    def test_unclean_worker_exit_preserves_readable_prefix_without_completed_seal(self):
        code = ("import os,sys; from pathlib import Path; "
                "from nca.evaluation.durable_journal import DurableJournal; "
                "j=DurableJournal(Path(sys.argv[1])); "
                "j.write(b'{\"event\":\"request_reserved\",\"serial\":1}\\n'); os._exit(17)")
        result = subprocess.run([sys.executable, "-c", code, str(self.raw)], check=False)
        self.assertEqual(result.returncode, 17)
        self.assertEqual(validate_record_chain(self.raw)["records"], 1)
        self.assertFalse((self.root / "owner-journal-seal.json").exists())
        self.assertFalse((self.root / "owner-journal.jsonl.gz").exists())
        with self.assertRaises(FileExistsError):
            DurableJournal(self.raw)

    def test_record_reordering_and_changed_content_fail_even_if_json_is_valid(self):
        self.write_records([{"event": "a"}, {"event": "b"}])
        lines = self.raw.read_bytes().splitlines(keepends=True)
        for data in (b"".join(reversed(lines)), b"".join(lines).replace(b'"event":"a"', b'"event":"c"')):
            self.raw.write_bytes(data)
            with self.assertRaisesRegex(ContractError, "journal_chain"):
                validate_record_chain(self.raw)

    def test_seal_replays_native_journal_and_every_retained_physical_tick(self):
        result, counts, rows = self.sealed()
        verified = self.verify(counts, rows)
        self.assertEqual(result, verified)
        self.assertTrue(verified["owner_validation"]["episode_receipts_bound"])
        self.assertEqual(verified["owner_validation"]["bound_episode_ticks"], 24)
        raw = self.raw.read_bytes()
        self.assertEqual(gzip.decompress((self.root / "owner-journal.jsonl.gz").read_bytes()), raw)

    def test_truncated_transport_or_early_copied_raw_cannot_match_seal(self):
        _, counts, rows = self.sealed()
        path = self.root / "owner-journal.jsonl.gz"
        original = path.read_bytes()
        path.write_bytes(original[:-8])
        with self.assertRaises(ContractError):
            self.verify(counts, rows)
        path.write_bytes(original)
        self.raw.write_bytes(self.raw.read_bytes().splitlines(keepends=True)[0])
        with self.assertRaises(ContractError):
            self.verify(counts, rows)

    def test_changed_summary_or_episode_bytes_cannot_acquire_a_completed_seal(self):
        _, counts, rows = self.sealed()
        with self.assertRaisesRegex(ContractError, "journal_seal"):
            self.verify({**counts, "confirmed_step_ticks": 30}, rows)
        path = self.root / rows[0]["artifact"]
        path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaisesRegex(ContractError, "arrival_journal_artifact"):
            self.verify(counts, rows)

    def test_independently_rehashed_journal_cannot_replace_retained_physical_receipt(self):
        records, counts, rows = bound_fixture(self.root)
        step = next(record for record in records if record["event"] == "response_received"
                    and "traces" in record["response"])
        step["response"]["traces"][0]["applied"]["linear_impulse"] = [9., 0.]
        durability = self.write_records(records)
        with self.assertRaisesRegex(ContractError, "arrival_journal_artifact"):
            seal_owner_journal(self.raw, expected_counts=counts, expected_rows=rows,
                durability=durability, validate_owner=validate_closed_owner_journal,
                families=FAMILIES, artifact_root=self.root)
        self.assertFalse((self.root / "owner-journal-seal.json").exists())
        self.assertTrue((self.root / "owner-journal.jsonl.gz.pending").exists())

    def corrupt_copied_metric(self, key, field, value):
        records, counts, rows = bound_fixture(self.root)
        row = rows[0]
        episode = read_episode(self.root / row["artifact"])
        changed_metrics = {**json.loads(canonical_json(episode.evaluation[key])), field: value}
        changed = replace(episode, evaluation={**episode.evaluation, key: changed_metrics})
        path = self.root / "episodes" / "copied-changed.json.gz"
        row[key] = changed_metrics
        row.update(artifact=path.relative_to(self.root).as_posix(), artifact_sha256=write_episode(path, changed),
                   record_sha256=changed.sha256)
        for record in records:
            if record["event"] == "episode_completed" and record["family"] == row["family"]:
                record["row"] = copy.deepcopy(row)
        durability = self.write_records(records)
        with self.assertRaisesRegex(ContractError, "arrival_journal_artifact"):
            seal_owner_journal(self.raw, expected_counts=counts, expected_rows=rows,
                durability=durability, validate_owner=validate_closed_owner_journal,
                families=FAMILIES, artifact_root=self.root)

    def test_matching_rehashed_task_claims_still_need_receipt_recomputation(self):
        self.corrupt_copied_metric("metrics", "success", True)

    def test_matching_rehashed_character_claims_still_need_receipt_recomputation(self):
        self.corrupt_copied_metric("character_metrics", "airborne_ticks", 12)

    def corrupt_source_view(self, source_field, physical_field, value):
        records, counts, rows = bound_fixture(self.root)
        row = rows[0]
        episode = read_episode(self.root / row["artifact"])
        first = episode.events[0]
        altered_view = {**getattr(first, source_field), physical_field: value}
        self.assertEqual(altered_view["observation_id"], first.source_observation["observation_id"])
        replacement = {source_field: altered_view}
        if source_field == "source_observation":
            replacement["policy_observation"] = policy_view(altered_view)
        changed = replace(episode, events=(replace(first, **replacement), *episode.events[1:]))
        path = self.root / "episodes" / "changed-source-view.json.gz"
        row.update(artifact=path.relative_to(self.root).as_posix(), artifact_sha256=write_episode(path, changed),
                   record_sha256=changed.sha256)
        for record in records:
            if record["event"] == "episode_completed" and record["family"] == row["family"]:
                record["row"] = copy.deepcopy(row)
        durability = self.write_records(records)
        with self.assertRaisesRegex(ContractError, "arrival_journal_artifact: synchronous controller source"):
            seal_owner_journal(self.raw, expected_counts=counts, expected_rows=rows,
                durability=durability, validate_owner=validate_closed_owner_journal,
                families=FAMILIES, artifact_root=self.root)
        self.assertFalse((self.root / "owner-journal-seal.json").exists())

    def test_same_identity_source_with_changed_support_cannot_justify_a_controller_action(self):
        self.corrupt_source_view("source_observation", "grounded", False)

    def test_same_identity_issued_view_with_changed_velocity_cannot_replace_native_state(self):
        self.corrupt_source_view("issued_observation", "velocity", [42., -12.])

    def test_atomic_publication_preserves_previous_failure(self):
        path = self.root / "receipt.json"
        publish_json(path, {"status": "failed"})
        before = file_identity(path)
        with self.assertRaises(FileExistsError):
            publish_json(path, {"status": "completed"})
        self.assertEqual(before, file_identity(path))


if __name__ == "__main__":
    unittest.main()
