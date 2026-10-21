"""Synthetic v2 sealing, interrupted writes and strict final inventory tests."""

import copy
import gzip
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from nca.contracts import ContractError, canonical_json
from nca.evaluation.durable_journal import DurableJournal, file_identity
from nca.evaluation.durable_publication import (
    diagnostic_artifact_inventory, publish_json_v2, seal_owner_journal_v2,
    strict_artifact_inventory, verify_sealed_journal_v2,
)
from nca.evaluation.jolt_arrival import FAMILIES, validate_closed_owner_journal
from test_jolt_evidence_durability import bound_fixture


class EvidencePublicationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.raw = self.root / "owner-journal.jsonl"
        self.compressed = self.root / "owner-journal.jsonl.gz"
        self.seal = self.root / "owner-journal-seal.json"
        self.pending = self.root / "owner-journal.jsonl.gz.pending"

    def fixture(self, mutate_records=None):
        records, counts, rows = bound_fixture(self.root)
        # The legacy tiny fixture has synthetic filenames; v2 binds the exact
        # family/case layout used by the diagnostic writer.
        for row in rows:
            artifact = f"episodes/{row['family']}-{row['case_id']}.json.gz"
            (self.root / row["artifact"]).rename(self.root / artifact)
            row["artifact"] = artifact
            for record in records:
                if record["event"] == "episode_completed" and record["family"] == row["family"]:
                    record["row"] = copy.deepcopy(row)
        if mutate_records is not None:
            mutate_records(records)
        publish_json_v2(self.root / "manifest.json", {"synthetic": True})
        for row in rows:
            publish_json_v2(self.root / "metrics" / f"{row['family']}-{row['case_id']}.json", row)
        with DurableJournal(self.raw) as journal:
            for record in records:
                journal.write((canonical_json(record) + "\n").encode())
        self.counts, self.rows, self.durability = counts, rows, journal.durability_receipt()

    def seal_fixture(self, *, validate_owner=validate_closed_owner_journal):
        return seal_owner_journal_v2(self.raw, expected_counts=self.counts, expected_rows=self.rows,
            durability=self.durability, validate_owner=validate_owner,
            families=FAMILIES, artifact_root=self.root)

    def verify(self, *, validate_owner=validate_closed_owner_journal):
        return verify_sealed_journal_v2(self.root, expected_counts=self.counts, expected_rows=self.rows,
            validate_owner=validate_owner, families=FAMILIES, summary=False)

    def test_closed_native_receipts_seal_and_verify_without_any_link_or_pending_file(self):
        self.fixture()
        with mock.patch("nca.evaluation.durable_publication.os.link",
                        side_effect=AssertionError("v2 must not publish through a hard link")):
            sealed = self.seal_fixture()
            verified = self.verify()
        self.assertEqual(sealed, verified)
        self.assertTrue(verified["owner_validation"]["native_close_confirmed"])
        self.assertTrue(verified["owner_validation"]["episode_receipts_bound"])
        self.assertEqual(verified["owner_validation"]["bound_episode_ticks"], 24)
        self.assertEqual(gzip.decompress(self.compressed.read_bytes()), self.raw.read_bytes())
        self.assertFalse(any(path.name.endswith(".pending") for path in self.root.rglob("*")))

    def test_surplus_pending_before_seal_is_retained_and_refused_not_inventoried(self):
        self.fixture()
        self.pending.write_bytes(b"retained previous failure")
        before = file_identity(self.pending)
        with self.assertRaisesRegex(ContractError, "journal_inventory"):
            self.seal_fixture()
        self.assertEqual(before, file_identity(self.pending))
        self.assertFalse(self.compressed.exists())
        self.assertFalse(self.seal.exists())

    def test_byte_identical_pending_created_during_replay_cannot_acquire_seal(self):
        self.fixture()
        def introduce_surplus(path, **kwargs):
            result = validate_closed_owner_journal(path, **kwargs)
            self.pending.write_bytes(path.read_bytes())
            return result
        with self.assertRaisesRegex(ContractError, "journal_inventory"):
            self.seal_fixture(validate_owner=introduce_surplus)
        self.assertEqual(file_identity(self.pending), file_identity(self.compressed))
        self.assertFalse(self.seal.exists())

    def test_identical_surplus_after_seal_fails_final_inventory_and_stays_preserved(self):
        self.fixture()
        self.seal_fixture()
        self.pending.write_bytes(self.compressed.read_bytes())
        original_seal = file_identity(self.seal)
        with self.assertRaisesRegex(ContractError, "journal_inventory"):
            self.verify()
        self.assertTrue(self.pending.exists())
        self.assertEqual(original_seal, file_identity(self.seal))

    def test_interrupted_compression_preserves_unsealed_bytes_and_cannot_resume(self):
        self.fixture()
        original = gzip.GzipFile.write
        def fail_after_prefix(stream, payload):
            original(stream, payload[:16])
            raise OSError("synthetic interrupted compression")
        with mock.patch("nca.evaluation.durable_publication.gzip.GzipFile.write", fail_after_prefix):
            with self.assertRaisesRegex(OSError, "synthetic interrupted compression"):
                self.seal_fixture()
        partial = file_identity(self.compressed)
        self.assertNotEqual(gzip.decompress(self.compressed.read_bytes()), self.raw.read_bytes())
        self.assertFalse(self.seal.exists())
        self.assertFalse(self.pending.exists())
        with self.assertRaisesRegex(ContractError, "journal_inventory"):
            self.seal_fixture()
        self.assertEqual(partial, file_identity(self.compressed))

    def test_failed_directory_flush_cannot_publish_completed_seal(self):
        self.fixture()
        with mock.patch("nca.evaluation.durable_publication.sync_directory",
                        side_effect=OSError("synthetic directory persistence failure")):
            with self.assertRaises(OSError):
                self.seal_fixture()
        self.assertTrue(self.compressed.exists())
        self.assertFalse(self.seal.exists())

    def test_missing_native_close_refuses_even_with_complete_gzip_and_chain(self):
        self.fixture(lambda records: records.pop())
        with self.assertRaisesRegex(ContractError, "arrival_journal_incomplete"):
            self.seal_fixture()
        self.assertTrue(self.compressed.exists())
        self.assertEqual(gzip.decompress(self.compressed.read_bytes()), self.raw.read_bytes())
        self.assertFalse(self.seal.exists())

    def test_outer_inventory_is_rechecked_after_independent_native_replay(self):
        self.fixture()
        self.seal_fixture()
        def change_after_replay(path, **kwargs):
            result = validate_closed_owner_journal(path, **kwargs)
            (self.root / "manifest.json").write_text('{"changed":true}\n')
            return result
        with self.assertRaisesRegex(ContractError, "journal_inventory"):
            self.verify(validate_owner=change_after_replay)

    def test_seal_binds_preseal_inventory_even_when_extra_is_added_to_result(self):
        self.fixture()
        self.seal_fixture()
        # A caller cannot legitimize an extra artifact by discovering it and
        # later adding it to a result; expected paths come from row identities.
        (self.root / "surplus.json").write_text('{}\n')
        with self.assertRaisesRegex(ContractError, "journal_inventory"):
            diagnostic_artifact_inventory(self.root, self.rows)

    def test_empty_surplus_directory_is_not_invisible_to_inventory(self):
        self.fixture()
        (self.root / "unexpected").mkdir()
        with self.assertRaisesRegex(ContractError, "journal_inventory"):
            diagnostic_artifact_inventory(self.root, self.rows, compressed=False, sealed=False)

    def test_symlinks_and_unsafe_declared_paths_are_refused(self):
        self.fixture()
        with self.assertRaisesRegex(ContractError, "journal_inventory"):
            strict_artifact_inventory(self.root, ["../elsewhere"])
        with self.assertRaisesRegex(ContractError, "journal_inventory"):
            strict_artifact_inventory(self.root, ["sub/../manifest.json"])
        with self.assertRaisesRegex(ContractError, "journal_inventory"):
            strict_artifact_inventory(self.root, ["C:/manifest.json"])
        changed = copy.deepcopy(self.rows)
        changed[0]["artifact"] = "episodes/unrelated-case.json.gz"
        with self.assertRaisesRegex(ContractError, "journal_inventory"):
            diagnostic_artifact_inventory(self.root, changed, compressed=False, sealed=False)
        if hasattr(os, "symlink"):
            try:
                (self.root / "extra-link").symlink_to(self.raw)
            except OSError:
                return  # Windows may not grant symlink creation to this user.
            with self.assertRaisesRegex(ContractError, "journal_inventory"):
                diagnostic_artifact_inventory(self.root, self.rows, compressed=False, sealed=False)

    def test_json_publication_preserves_original_bytes_after_duplicate_attempt(self):
        path = self.root / "result.json"
        publish_json_v2(path, {"status": "failed"})
        before = file_identity(path)
        with self.assertRaises(FileExistsError):
            publish_json_v2(path, {"status": "completed"})
        self.assertEqual(before, file_identity(path))
        self.assertEqual(set(self.root.iterdir()), {path})

    def test_changed_counts_or_rows_cannot_use_existing_seal(self):
        self.fixture()
        self.seal_fixture()
        self.counts = {**self.counts, "confirmed_step_ticks": 30}
        with self.assertRaisesRegex(ContractError, "journal_seal"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
