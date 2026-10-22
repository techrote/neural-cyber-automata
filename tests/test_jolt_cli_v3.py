"""Adversarial independent readback of original v3 cost phases and prefixes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from nca.contracts import ContractError, sha256_json
from nca.train.jolt_budget_v3 import BudgetLedgerV3, reservation_records
from nca.train.jolt_cli_v3 import cost_snapshot_v3

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


if __name__ == "__main__":
    unittest.main()
