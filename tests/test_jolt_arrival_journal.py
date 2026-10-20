"""Offline rejection of incomplete owner evidence; no physical host is used."""

import copy
import gzip
from pathlib import Path
import tempfile
import unittest

from nca.contracts import ContractError, canonical_json
from nca.evaluation.jolt_arrival import FAMILIES, validate_closed_owner_journal


def observation(tick, *, suspended=False, generation=0):
    return {"tick": tick, "suspended": suspended, "control_generation": generation,
            "run_id": "synthetic", "epoch": 1, "actor_id": 0, "model": "jev", "profile_version": 1,
            "observation_id": f"synthetic:1:0:{generation}:{tick}", "joints": [], "parts": [],
            "position": [0., 0.], "velocity": [0., 0.], "rotation": 0., "angular_velocity": 0.}


def closed_journal():
    """Two small synthetic episodes exercise cancellation and terminal closure."""
    records, serial = [], 0
    rows = [{"case_id": "synthetic-case", "family": family, "decisions": 2, "consumed_ticks": 12,
             "owner_closure_confirmed": True, "error": None, "metrics": {"success": index == 0}}
            for index, family in enumerate(FAMILIES)]

    def note(event, **fields):
        records.append({"event": event, "elapsed_seconds": len(records) / 100, **fields})

    def exchange(operation, response, **fields):
        nonlocal serial
        serial += 1
        note("request_reserved", serial=serial, operation=operation,
             requested_ticks=6 if operation == "step" else 0, fields=fields)
        note("response_received", serial=serial,
             response={"protocol": "nca-host/1", "id": serial, "ok": True, **response})

    note("host_start_reserved")
    for index, row in enumerate(rows):
        note("episode_started", case=row["case_id"], family=row["family"])
        exchange("reset", {"observation": observation(0)})
        exchange("profile", {"profile": {"synthetic": True}})
        for start in (0, 6):
            terminal = index == 1 and start == 6
            exchange("step", {"actual_ticks": 6, "start_tick": start, "tick": start + 6,
                "observation": observation(start + 6, suspended=terminal, generation=int(terminal)),
                "terminated": False, "truncated": terminal, "finished": terminal,
                "traces": [{"tick_before": tick, "tick_after": tick + 1} for tick in range(start, start + 6)]},
                ticks=6, commands=[])
        if index == 0:
            exchange("cancel", {"cancelled": True, "observation": observation(12, generation=1)})
        note("episode_completed", case=row["case_id"], family=row["family"], row=copy.deepcopy(row))
    exchange("close", {"closed": True})
    note("host_closed", termination_confirmed=True)
    counts = {"host_requests": 10, "step_requests": 4, "requested_step_ticks": 24,
              "confirmed_step_ticks": 24, "episodes_started": 2, "host_start_attempts": 1,
              "host_terminations": 1, "pending_requests": [], "unconfirmed_tail": False,
              "wall_seconds": 1., "optimizer_updates": 0, "checkpoint_executions": 0, "learned_policy_calls": 0}
    return records, counts, rows


class ArrivalJournalTests(unittest.TestCase):
    def setUp(self):
        self.records, self.counts, self.rows = closed_journal()
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "owner-journal.jsonl.gz"

    def write(self, records=None):
        records = self.records if records is None else records
        payload = b"".join((canonical_json(record) + "\n").encode() for record in records)
        self.path.write_bytes(gzip.compress(payload, mtime=0))

    def validate(self, *, counts=None, rows=None):
        return validate_closed_owner_journal(self.path,
            expected_counts=self.counts if counts is None else counts,
            expected_rows=self.rows if rows is None else rows)

    def response_index(self, operation):
        return next(index + 1 for index, record in enumerate(self.records)
                    if record["event"] == "request_reserved" and record["operation"] == operation)

    def test_complete_native_cancellation_and_terminal_receipts_reconcile(self):
        self.write()
        result = self.validate()
        self.assertTrue(result["verified"])
        self.assertTrue(result["gzip_eof_verified"])
        self.assertTrue(result["native_close_confirmed"])
        self.assertEqual(result["records"], 26)
        self.assertEqual(result["episodes_completed"], 2)
        self.assertEqual(result["counts"]["confirmed_step_ticks"], 24)

    def test_missing_gzip_trailer_is_rejected_even_with_every_complete_record(self):
        self.write()
        original = self.path.read_bytes()
        self.path.write_bytes(original[:-8])
        with self.assertRaisesRegex(ContractError, "arrival_journal_stream"):
            self.validate()
        self.assertEqual(self.path.read_bytes(), original[:-8])

    def test_crc_damage_is_rejected_after_all_records_have_been_read(self):
        self.write()
        damaged = bytearray(self.path.read_bytes())
        damaged[-8] ^= 1
        self.path.write_bytes(damaged)
        with self.assertRaisesRegex(ContractError, "arrival_journal_stream"):
            self.validate()

    def test_a_prefix_with_a_valid_gzip_footer_is_still_incomplete(self):
        first_completion = next(index for index, record in enumerate(self.records)
                                if record["event"] == "episode_completed")
        for prefix in (self.records[:first_completion + 1], self.records[:-1]):
            with self.subTest(records=len(prefix)):
                self.write(prefix)
                self.assertTrue(gzip.decompress(self.path.read_bytes()).endswith(b"\n"))
                with self.assertRaisesRegex(ContractError, "arrival_journal_incomplete"):
                    self.validate()

    def test_pending_reservation_cannot_be_hidden_by_a_valid_gzip_footer(self):
        pending_index = self.response_index("step")
        self.write(self.records[:pending_index])
        with self.assertRaisesRegex(ContractError, "arrival_journal_incomplete"):
            self.validate()

    def test_replayed_counts_must_match_every_expected_counter(self):
        self.write()
        for key in ("host_requests", "step_requests", "requested_step_ticks", "confirmed_step_ticks",
                    "host_start_attempts", "host_terminations"):
            counts = copy.deepcopy(self.counts)
            counts[key] += 1
            with self.subTest(counter=key), self.assertRaisesRegex(ContractError, "arrival_journal_counts"):
                self.validate(counts=counts)

    def test_expected_uncertainty_or_model_activity_cannot_be_reclassified(self):
        self.write()
        for key, value in (("pending_requests", [{"serial": 11}]), ("unconfirmed_tail", True),
                           ("confirmed_step_ticks", 18), ("learned_policy_calls", 1)):
            counts = {**self.counts, key: value}
            with self.subTest(counter=key), self.assertRaisesRegex(ContractError, "arrival_journal_counts"):
                self.validate(counts=counts)

    def test_partial_native_tick_confirmation_is_not_a_complete_six_tick_step(self):
        for actual in (5, True, None):
            records = copy.deepcopy(self.records)
            records[self.response_index("step")]["response"]["actual_ticks"] = actual
            self.write(records)
            with self.subTest(actual=actual), self.assertRaisesRegex(ContractError, "arrival_journal_response"):
                self.validate()

    def test_missing_duplicate_and_uncorrelated_response_records_fail_closed(self):
        index = self.response_index("step")
        missing = copy.deepcopy(self.records)
        del missing[index]
        duplicate = copy.deepcopy(self.records)
        duplicate.insert(index, copy.deepcopy(duplicate[index]))
        uncorrelated = copy.deepcopy(self.records)
        uncorrelated[index]["response"]["id"] += 1
        for records in (missing, duplicate, uncorrelated):
            self.write(records)
            with self.subTest(records=len(records)), self.assertRaisesRegex(ContractError, "arrival_journal_(order|response)"):
                self.validate()

    def test_native_close_response_and_final_process_confirmation_are_both_required(self):
        index = self.response_index("close")
        missing = copy.deepcopy(self.records)
        del missing[index]
        refused = copy.deepcopy(self.records)
        refused[index]["response"]["closed"] = False
        unconfirmed = copy.deepcopy(self.records)
        unconfirmed[-1]["termination_confirmed"] = False
        for records in (missing, refused, unconfirmed):
            self.write(records)
            with self.subTest(records=len(records)), self.assertRaisesRegex(ContractError, "arrival_journal_close"):
                self.validate()

    def test_native_cancellation_must_revoke_generation_without_advancing_physics(self):
        index = self.response_index("cancel")
        for key, value in (("control_generation", 0), ("tick", 13), ("position", [1., 0.])):
            records = copy.deepcopy(self.records)
            records[index]["response"]["observation"][key] = value
            self.write(records)
            with self.subTest(field=key), self.assertRaisesRegex(ContractError, "arrival_journal_close"):
                self.validate()

    def test_arrival_in_a_terminal_interval_may_cancel_once_after_terminal_revocation(self):
        # _capture may first recognize arrival inside the repeat that also
        # reaches the native deadline, then explicitly cancel that owner.
        index = max(index for index, record in enumerate(self.records) if record["event"] == "episode_completed")
        records = copy.deepcopy(self.records)
        records[index:index] = [
            {"event": "request_reserved", "serial": 10, "operation": "cancel", "requested_ticks": 0, "fields": {}},
            {"event": "response_received", "serial": 10, "response": {"protocol": "nca-host/1", "id": 10,
                "ok": True, "cancelled": True, "observation": observation(12, suspended=True, generation=2)}}]
        for offset, record in enumerate(records):
            record["elapsed_seconds"] = offset / 100
            if offset > index + 1 and record["event"] in ("request_reserved", "response_received"):
                record["serial"] += 1
                if record["event"] == "response_received":
                    record["response"]["id"] += 1
        self.write(records)
        self.assertTrue(self.validate(counts={**self.counts, "host_requests": 11})["verified"])

        repeated = copy.deepcopy(records)
        repeated.insert(index + 2, {"event": "request_reserved", "elapsed_seconds": (index + 1) / 100,
            "serial": 11, "operation": "cancel", "requested_ticks": 0, "fields": {}})
        self.write(repeated)
        with self.assertRaisesRegex(ContractError, "arrival_journal_order"):
            self.validate(counts={**self.counts, "host_requests": 12})

    def test_completed_rows_are_bound_to_declared_results_and_native_counts(self):
        index = next(index for index, record in enumerate(self.records) if record["event"] == "episode_completed")
        for mutate_expected in (False, True):
            records, rows = copy.deepcopy(self.records), copy.deepcopy(self.rows)
            records[index]["row"]["decisions"] += 1
            if mutate_expected:
                rows[0]["decisions"] += 1
            self.write(records)
            with self.subTest(mutate_expected=mutate_expected), self.assertRaisesRegex(ContractError, "arrival_journal_episodes"):
                self.validate(rows=rows)

    def test_missing_episode_boundary_or_extra_post_close_record_is_rejected(self):
        index = next(index for index, record in enumerate(self.records) if record["event"] == "episode_completed")
        missing = copy.deepcopy(self.records)
        del missing[index]
        extra = self.records + [{"event": "host_closed", "elapsed_seconds": .5, "termination_confirmed": True}]
        for records in (missing, extra):
            self.write(records)
            with self.subTest(records=len(records)), self.assertRaisesRegex(ContractError, "arrival_journal_(episodes|order)"):
                self.validate()

    def test_malformed_or_unterminated_json_record_is_rejected(self):
        for payload in (b'{"event":"host_start_reserved", "event":"host_closed"}\n',
                        b'null\n', b'{"event":"host_start_reserved","elapsed_seconds":0}'):
            self.path.write_bytes(gzip.compress(payload, mtime=0))
            with self.subTest(payload=payload), self.assertRaisesRegex(ContractError, "arrival_journal_record"):
                self.validate()


if __name__ == "__main__":
    unittest.main()
