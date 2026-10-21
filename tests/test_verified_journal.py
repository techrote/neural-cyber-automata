"""Synthetic storage faults; no physical host, model or learning invocation."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from nca.contracts import ContractError
from nca.evaluation.durable_journal import validate_record_chain
from nca.evaluation.jolt_arrival import DiagnosticBudget, MeteredHost
from nca.evaluation.verified_journal import RECEIPT_SCHEMA, VerifiedDurableJournal


class VerifiedJournalTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.path = self.root / "owner-journal.jsonl"

    def writer(self):
        journal = VerifiedDurableJournal(self.path)
        self.addCleanup(journal.close)
        return journal

    def test_clean_receipt_binds_independent_complete_stream_and_never_verifies_early(self):
        journal = self.writer()
        journal.write(b'{"event":"first"}\n')
        self.assertFalse(journal.durability_receipt()["closed_verified"])
        self.assertIsNone(journal.durability_receipt()["closed_identity"])
        journal.write(b'{"event":"second"}\n')
        journal.close()
        identity = validate_record_chain(self.path)
        receipt = journal.durability_receipt()
        self.assertEqual(receipt["schema"], RECEIPT_SCHEMA)
        self.assertTrue(receipt["closed_verified"])
        self.assertFalse(receipt["poisoned"])
        self.assertEqual(receipt["closed_identity"], identity)
        self.assertEqual(receipt["records"], 2)
        self.assertEqual(receipt["record_fsyncs"], 2)
        self.assertEqual(receipt["readback_records"], 2)
        self.assertEqual(receipt["total_fsyncs"], 3)
        receipt["closed_identity"]["bytes"] = -1
        self.assertEqual(journal.durability_receipt()["closed_identity"], identity)
        with self.assertRaisesRegex(ContractError, "journal_writer_closed"):
            journal.write(b'{"event":"late"}\n')

    def test_truncated_prefix_is_rejected_before_another_owner_dispatch(self):
        journal = self.writer()
        journal.write(b'{"event":"retained_prefix"}\n')
        self.path.write_bytes(b"")
        host = mock.Mock()
        budget = DiagnosticBudget({"maximum_wall_seconds": 10, "maximum_host_requests": 3}, journal)
        with self.assertRaisesRegex(ContractError, "journal_length"):
            MeteredHost(host, budget).request("cancel")
        host.request.assert_not_called()
        self.assertTrue(journal.poisoned)
        self.assertEqual(journal.records, 1)
        self.assertEqual(self.path.read_bytes(), b"")
        with self.assertRaisesRegex(ContractError, "journal_writer_closed"):
            MeteredHost(host, budget).request("cancel")
        journal.close()
        self.assertFalse(journal.durability_receipt()["closed_verified"])

    def test_external_append_is_not_silently_adopted(self):
        journal = self.writer()
        journal.write(b'{"event":"first"}\n')
        with self.path.open("ab") as stream:
            stream.write(b'{"event":"foreign"}\n')
        with self.assertRaisesRegex(ContractError, "journal_length"):
            journal.write(b'{"event":"second"}\n')
        self.assertTrue(journal.poisoned)
        self.assertEqual(journal.records, 1)

    def test_path_replacement_cannot_divert_live_writer_from_readers(self):
        journal = self.writer()
        journal.write(b'{"event":"first"}\n')
        original = self.path.read_bytes()
        replacement = self.root / "replacement.jsonl"
        replacement.write_bytes(original)
        try:
            os.replace(replacement, self.path)
        except PermissionError:
            self.skipTest("this platform denies replacing an open file")
        with self.assertRaisesRegex(ContractError, "journal_identity"):
            journal.write(b'{"event":"second"}\n')
        self.assertEqual(self.path.read_bytes(), original)
        self.assertTrue(journal.poisoned)

    def test_previous_tail_same_length_corruption_stops_next_append(self):
        journal = self.writer()
        journal.write(b'{"event":"first"}\n')
        corrupted = self.path.read_bytes().replace(b'"first"', b'"other"')
        self.path.write_bytes(corrupted)
        with self.assertRaisesRegex(ContractError, "journal_readback"):
            journal.write(b'{"event":"second"}\n')
        self.assertEqual(self.path.read_bytes(), corrupted)
        self.assertEqual(journal.records, 1)
        self.assertTrue(journal.poisoned)

    def test_earlier_same_length_corruption_cannot_obtain_verified_closure(self):
        journal = self.writer()
        journal.write(b'{"event":"first"}\n')
        journal.write(b'{"event":"second"}\n')
        self.path.write_bytes(self.path.read_bytes().replace(b'"first"', b'"other"'))
        # Per-append verification is bounded to the previous and new record.
        journal.write(b'{"event":"third"}\n')
        with self.assertRaisesRegex(ContractError, "journal_chain"):
            journal.close()
        receipt = journal.durability_receipt()
        self.assertTrue(receipt["closed"])
        self.assertTrue(receipt["poisoned"])
        self.assertFalse(receipt["closed_verified"])
        self.assertIsNone(receipt["closed_identity"])

    def test_post_sync_readback_failure_prevents_dispatch_and_cannot_retry(self):
        journal = self.writer()
        host = mock.Mock()
        budget = DiagnosticBudget({"maximum_wall_seconds": 10, "maximum_host_requests": 3}, journal)
        with mock.patch.object(journal, "_read_span", side_effect=OSError("synthetic read failure")):
            with self.assertRaises(OSError):
                MeteredHost(host, budget).request("cancel")
        host.request.assert_not_called()
        self.assertEqual(journal.records, 0)
        self.assertEqual(journal.record_fsyncs, 1)
        self.assertEqual(journal.readback_records, 0)
        self.assertEqual(validate_record_chain(self.path)["records"], 1)
        self.assertTrue(journal.poisoned)
        journal.close()
        self.assertFalse(journal.durability_receipt()["closed_verified"])

    def test_fsync_failure_and_short_write_poison_without_claiming_record(self):
        for fault in ("fsync", "short"):
            with self.subTest(fault=fault):
                path = self.root / (fault + ".jsonl")
                journal = VerifiedDurableJournal(path)
                try:
                    if fault == "fsync":
                        context = mock.patch("nca.evaluation.verified_journal.os.fsync", side_effect=OSError("synthetic fsync failure"))
                    else:
                        context = mock.patch.object(journal._stream, "write", return_value=1)
                    with context, self.assertRaises(OSError):
                        journal.write(b'{"event":"first"}\n')
                    self.assertTrue(journal.poisoned)
                    self.assertEqual(journal.records, 0)
                    self.assertEqual(journal.record_fsyncs, 0)
                    with self.assertRaisesRegex(ContractError, "journal_writer_closed"):
                        journal.write(b'{"event":"retry"}\n')
                finally:
                    journal.close()

    def test_close_fsync_failure_retains_uncertain_receipt_and_releases_descriptor(self):
        journal = self.writer()
        journal.write(b'{"event":"first"}\n')
        with mock.patch("nca.evaluation.verified_journal.os.fsync", side_effect=OSError("synthetic close failure")):
            with self.assertRaises(OSError):
                journal.close()
        self.assertTrue(journal.closed)
        self.assertTrue(journal._stream.closed)
        self.assertTrue(journal.poisoned)
        self.assertFalse(journal.closed_verified)

    def test_complete_readback_rejects_a_self_consistent_rewritten_prefix(self):
        journal = self.writer()
        journal.write(b'{"event":"first"}\n')
        other = self.root / "other.jsonl"
        with VerifiedDurableJournal(other) as rewritten:
            rewritten.write(b'{"event":"other"}\n')
        self.assertEqual(other.stat().st_size, self.path.stat().st_size)
        self.path.write_bytes(other.read_bytes())
        with self.assertRaisesRegex(ContractError, "journal_readback"):
            journal.close()
        self.assertFalse(journal.closed_verified)
        self.assertTrue(journal.poisoned)

    def test_unclean_exit_retains_readable_prefix_and_cannot_resume(self):
        code = ("import os,sys; from pathlib import Path; "
                "from nca.evaluation.verified_journal import VerifiedDurableJournal; "
                "j=VerifiedDurableJournal(Path(sys.argv[1])); "
                "j.write(b'{\"event\":\"reserved\"}\\n'); os._exit(17)")
        process = subprocess.run([sys.executable, "-c", code, str(self.path)], check=False)
        self.assertEqual(process.returncode, 17)
        self.assertEqual(validate_record_chain(self.path)["records"], 1)
        self.assertEqual({entry.name for entry in self.root.iterdir()}, {self.path.name})
        with self.assertRaises(FileExistsError):
            VerifiedDurableJournal(self.path)

    def test_readback_is_visible_before_native_request_and_response_confirmation(self):
        journal = self.writer()
        limits = {"maximum_wall_seconds": 10, "maximum_host_requests": 4,
                  "maximum_decisions": 1, "maximum_requested_ticks": 6}
        budget = DiagnosticBudget(limits, journal)
        class Owner:
            def request(inner, operation, **fields):
                self.assertEqual(journal.readback_records, 1)
                self.assertEqual(validate_record_chain(self.path)["records"], 1)
                self.assertEqual(budget.counts["confirmed_step_ticks"], 0)
                return {"actual_ticks": 6}
        MeteredHost(Owner(), budget).request("step", ticks=6)
        self.assertEqual(journal.readback_records, 2)
        self.assertEqual(budget.counts["confirmed_step_ticks"], 6)
        journal.close()
        self.assertTrue(journal.closed_verified)


if __name__ == "__main__":
    unittest.main()
