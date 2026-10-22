"""Adversarial independent readback of original v3 cost phases and prefixes."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock

from nca.contracts import ContractError, sha256_json
from nca.evaluation.durable_journal import file_identity
from nca.evaluation.jolt_journal_v3 import journal_identity_v3
from nca.train.jolt_budget_v3 import BudgetLedgerV3, reservation_records
from nca.train.jolt_cli_v3 import cost_snapshot_v3, verify_inventory, verify_execution
from nca.train.jolt_storage_v3 import MAX_JOURNAL_RECORD_BYTES, MAX_JOURNAL_RECORDS, MAX_JOURNAL_BYTES

ROOT = Path(__file__).resolve().parents[1]
BUDGET = json.loads((ROOT / "configs/learning/jolt-learn-003-v1.json").read_bytes())["budget"]


class ClosedCostsV3Tests(unittest.TestCase):
    def fixture(self, folder):
        ledger = BudgetLedgerV3(BUDGET, started=0, spool_path=folder / "live.jsonl", monotonic=lambda: 1.)
        ledger.reserve("attempts")
        ledger.reserve_many({"episodes_started": 1, "decisions": 1, "requested_ticks": 6}, stage="teacher")
        ledger.enter_phase("closure")
        ledger.reserve("episodes_completed", cleanup=True)
        before = ledger.snapshot()
        ledger.publish_reservations(folder / "before.json")
        ledger.enter_phase("audit")
        ledger.reserve("checkpoint_loads", cleanup=True)
        ledger.reserve("audit_forward", 2, cleanup=True)
        ledger.enter_phase("publication")
        ledger.publish_reservations(folder / "after.json")
        after = ledger.close()
        return before, after

    def test_actual_stream_prefix_and_exact_audit_deltas_reconcile(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            before, after = self.fixture(folder)
            first = cost_snapshot_v3(BUDGET, reservation_records(folder / "before.json"), before, closed=False)
            final = cost_snapshot_v3(BUDGET, reservation_records(folder / "after.json"), after, closed=True,
                                    prefix=before, audit={"checkpoint_loads": 1, "audit_forward": 2})
            self.assertEqual(final["audit_forward"] - first["audit_forward"], 2)
            self.assertEqual(final["decisions"], 1)

    def test_consistent_digest_cannot_hide_work_in_publication_or_reset_clock(self):
        for mutation in (lambda row: row.update(phase="publication"),
                         lambda row: row.update(elapsed_seconds=0.),
                         lambda row: row.update(increments={"decisions": 1}, stage="sealed"),
                         lambda row: row.update(increments={"audit_forward": -1})):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                before, after = self.fixture(folder)
                rows = list(reservation_records(folder / "after.json"))
                mutation(rows[-1])
                after["reservations_sha256"] = sha256_json(rows)
                with self.assertRaises(ContractError):
                    cost_snapshot_v3(BUDGET, rows, after, closed=True, prefix=before)

    def test_full_prefix_digest_and_phase_history_are_independently_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            before, after = self.fixture(folder)
            rows = list(reservation_records(folder / "after.json"))
            bad_prefix = deepcopy(before)
            bad_prefix["reservations_sha256"] = hashlib.sha256(b"different prefix").hexdigest()
            with self.assertRaisesRegex(ContractError, "prefix changed"):
                cost_snapshot_v3(BUDGET, rows, after, closed=True, prefix=bad_prefix)
            after["phase_history"][1]["elapsed_seconds"] = .5
            with self.assertRaisesRegex(ContractError, "phase reset"):
                cost_snapshot_v3(BUDGET, rows, after, closed=True, prefix=before)

    def test_reported_audit_counts_cannot_refund_or_hide_model_work(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            before, after = self.fixture(folder)
            with self.assertRaisesRegex(ContractError, "charged readback costs"):
                cost_snapshot_v3(BUDGET, reservation_records(folder / "after.json"), after, closed=True,
                                 prefix=before, audit={"checkpoint_loads": 0, "audit_forward": 2})

    def test_closed_snapshot_requires_all_four_original_worker_phases(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            before, after = self.fixture(folder)
            after["phase_history"].pop(1)
            with self.assertRaises(ContractError):
                cost_snapshot_v3(BUDGET, reservation_records(folder / "after.json"), after, closed=True)


class ClosedInventoryV3Tests(unittest.TestCase):
    """Exercise real tree/byte verification independently of neural cost replay."""

    def fixture(self, folder):
        (folder / "owner-journal.jsonl").write_bytes(b"")
        chain = journal_identity_v3(folder / "owner-journal.jsonl")
        durable = {"schema": "nca-owner-journal-durability/v3",
            "maximum_record_bytes": MAX_JOURNAL_RECORD_BYTES, "maximum_records": MAX_JOURNAL_RECORDS,
            "maximum_stream_bytes": MAX_JOURNAL_BYTES, "closed_verified": True, "poisoned": False,
            "closed_identity": chain, "readback_records": 0}
        (folder / "result.json").write_text(json.dumps({"closure": {"durability": durable},
            "status": "completed", "qualification": {"qualified": False}}))
        (folder / "independent-readback.json").write_text(json.dumps({"verified": True, "complete": True}))
        samples = folder / "episodes" / "teacher"
        samples.mkdir(parents=True)
        (samples / "sample.json").write_text("{}")
        files = {path.relative_to(folder).as_posix(): file_identity(path)
                 for path in folder.rglob("*") if path.is_file()}
        (folder / "inventory.json").write_text(json.dumps({"schema": "jolt-closed-inventory/v1", "files": files}))

    def verify(self, folder):
        # Cost counters have separate adversarial tests above. Keep this
        # fixture about actual original tree entries, hashes and journal EOF.
        with mock.patch("nca.train.jolt_cli_v3.verify_closed_costs", return_value={"synthetic": True}):
            return verify_inventory(folder)

    def test_exact_tree_allows_only_required_ancestor_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.fixture(folder)
            result = self.verify(folder)
            self.assertTrue(result["inventory_verified"])
            self.assertEqual(result["files"], 4)
            self.assertFalse(result["qualified"])

    def test_windows_zero_direntry_link_metadata_uses_real_nofollow_file_metadata(self):
        real_scandir = os.scandir
        @contextmanager
        def windows_scandir(path):
            with real_scandir(path) as entries:
                proxies = []
                for entry in entries:
                    values = list(entry.stat(follow_symlinks=False))
                    # Windows DirEntry.stat omits these fields; Path.lstat
                    # still supplies the actual identity and hard-link count.
                    values[1:4] = [0, 0, 0]
                    proxies.append(mock.Mock(path=entry.path,
                        stat=mock.Mock(return_value=os.stat_result(values))))
                yield iter(proxies)
        for linked in (False, True):
            with self.subTest(linked=linked), tempfile.TemporaryDirectory() as temp:
                parent = Path(temp)
                folder = parent / "output"
                folder.mkdir()
                self.fixture(folder)
                if linked:
                    os.link(folder / "episodes/teacher/sample.json", parent / "outside-alias")
                with mock.patch("nca.train.jolt_cli_v3.os.scandir", side_effect=windows_scandir):
                    if linked:
                        with self.assertRaisesRegex(ContractError, "hard links"):
                            self.verify(folder)
                    else:
                        self.assertTrue(self.verify(folder)["inventory_verified"])

    def test_unexpected_empty_directories_fail_at_root_and_beneath_declared_ancestors(self):
        for relative in ("extra", "episodes/teacher/extra", "episodes/extra/nested"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                self.fixture(folder)
                (folder / relative).mkdir(parents=True)
                with self.assertRaisesRegex(ContractError, "unexpected directory"):
                    self.verify(folder)

    def test_missing_extra_and_changed_regular_files_fail(self):
        for kind in ("missing", "extra", "changed"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                self.fixture(folder)
                sample = folder / "episodes" / "teacher" / "sample.json"
                if kind == "missing":
                    sample.unlink()
                elif kind == "extra":
                    (sample.parent / "extra.json").write_text("{}")
                else:
                    sample.write_text("{\"changed\":true}")
                with self.assertRaises(ContractError):
                    self.verify(folder)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO entries require POSIX")
    def test_unexpected_and_inventoried_fifo_entries_are_never_opened(self):
        for replace_sample in (False, True):
            with self.subTest(replace_sample=replace_sample), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                self.fixture(folder)
                path = folder / ("episodes/teacher/sample.json" if replace_sample else "extra-fifo")
                if replace_sample:
                    path.unlink()
                os.mkfifo(path)
                with mock.patch("nca.train.jolt_cli_v3.file_identity",
                                side_effect=AssertionError("tree must reject FIFO before byte reads")):
                    with self.assertRaisesRegex(ContractError, "special or nonregular"):
                        self.verify(folder)

    @unittest.skipUnless(os.name == "posix" and hasattr(socket, "AF_UNIX"), "UNIX sockets require POSIX")
    def test_socket_entry_is_rejected_before_byte_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.fixture(folder)
            with socket.socket(socket.AF_UNIX) as endpoint:
                try:
                    endpoint.bind(str(folder / "extra-socket"))
                except PermissionError:
                    self.skipTest("execution sandbox forbids creating a UNIX socket")
                with mock.patch("nca.train.jolt_cli_v3.file_identity",
                                side_effect=AssertionError("tree must reject socket before byte reads")):
                    with self.assertRaisesRegex(ContractError, "special or nonregular"):
                        self.verify(folder)

    @unittest.skipUnless(os.name == "posix", "symlink creation requires POSIX fixture permissions")
    def test_root_file_directory_and_dangling_symlinks_fail(self):
        for kind in ("root", "file", "directory", "dangling"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                parent = Path(temp)
                folder = parent / "output"
                folder.mkdir()
                self.fixture(folder)
                if kind == "root":
                    link = parent / "output-link"
                    link.symlink_to(folder, target_is_directory=True)
                    folder = link
                elif kind == "file":
                    path = folder / "episodes/teacher/sample.json"
                    path.unlink()
                    path.symlink_to(folder / "result.json")
                elif kind == "directory":
                    (folder / "extra-link").symlink_to(folder / "episodes", target_is_directory=True)
                else:
                    (folder / "dangling-link").symlink_to(folder / "missing")
                with self.assertRaises(ContractError):
                    self.verify(folder)

    @unittest.skipUnless(hasattr(os, "link"), "hard links require filesystem support")
    def test_byte_identical_outside_hardlinks_to_manifest_and_artifacts_fail(self):
        for relative in ("inventory.json", "episodes/teacher/sample.json"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temp:
                parent = Path(temp)
                folder = parent / "output"
                folder.mkdir()
                self.fixture(folder)
                os.link(folder / relative, parent / "outside-alias")
                # No file, directory or digest inside the output changed.
                # An outside alias must still revoke immutable-tree acceptance.
                with mock.patch("nca.train.jolt_cli_v3.file_identity",
                                side_effect=AssertionError("hardlink must fail before byte reads")):
                    with self.assertRaisesRegex(ContractError, "links"):
                        self.verify(folder)

    def test_noncanonical_and_self_referential_inventory_names_fail_before_hashing(self):
        for relative in ("", ".", "..", "../outside", "/outside", "episodes//sample.json",
                         "episodes/./sample.json", "episodes/", "episodes\\sample.json", "inventory.json"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                self.fixture(folder)
                path = folder / "inventory.json"
                value = json.loads(path.read_bytes())
                value["files"][relative] = {"bytes": 0, "sha256": "0" * 64}
                path.write_text(json.dumps(value))
                with mock.patch("nca.train.jolt_cli_v3.file_identity",
                                side_effect=AssertionError("invalid name must fail before byte reads")):
                    with self.assertRaisesRegex(ContractError, "canonical relative"):
                        self.verify(folder)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO entries require POSIX")
    def test_execution_binding_rejects_fifo_before_initial_declaration_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            os.mkfifo(folder / "declaration.json")
            with mock.patch.object(Path, "open", side_effect=AssertionError("FIFO must never be opened")):
                with self.assertRaises(ContractError):
                    verify_execution(folder, terminal_stdout=folder / "stdout.log",
                                     observed_exit_code=0, observed_exit_elapsed_seconds=1.)

    @unittest.skipUnless(hasattr(os, "link"), "hard links require filesystem support")
    def test_execution_binding_rejects_outside_alias_before_initial_declaration_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            folder = parent / "output"
            folder.mkdir()
            declaration = folder / "declaration.json"
            declaration.write_text("{}")
            os.link(declaration, parent / "outside-alias")
            with mock.patch.object(Path, "open", side_effect=AssertionError("linked declaration must not be opened")):
                with self.assertRaises(ContractError):
                    verify_execution(folder, terminal_stdout=folder / "stdout.log",
                                     observed_exit_code=0, observed_exit_elapsed_seconds=1.)


if __name__ == "__main__":
    unittest.main()
