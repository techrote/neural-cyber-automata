"""Explicit null operational clocks; synthetic receipts only, no host launch."""

from jolt_source_fixture import synthetic_capture_provenance
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from nca.contracts import ContractError, sha256_json
from nca.train.jolt_budget_v3 import BudgetLedgerV3, reservation_records
from nca.train.jolt_cli_v3 import cost_snapshot_v3, verify_execution
from nca.train.jolt_execution_v3 import OriginalClockV3
import test_jolt_streams_v3 as stream_fixtures


ROOT = Path(__file__).resolve().parents[1]
BUDGET = json.loads((ROOT / "configs/learning/jolt-learn-003-v1.json").read_bytes())["budget"]
UNLIMITED = deepcopy(BUDGET)
for _field in ("maximum_wall_seconds", "maximum_owner_transport_seconds", "cleanup_reserve_seconds",
               "executor_verification_reserve_seconds", "executor_finalization_reserve_seconds"):
    UNLIMITED[_field] = None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    from nca.data import EpisodeRecorder
    from nca.evaluation.jolt_learning_audit import _Reservations
    from nca.train.jolt_capture_v3 import CaptureBudgetV3, capture_v3
    from nca.train.jolt_screen import MeteredOwner
    from test_jolt_learning_v3 import SyntheticBoundaryOwner, SyntheticLedger
    from test_jolt_screen import FakeHost, Journal


def close_ledger(ledger):
    for phase in ("closure", "audit", "publication"):
        ledger.enter_phase(phase)
    ledger.close()


class UnlimitedReadbackV3Tests(unittest.TestCase):
    def fixture(self, folder):
        now = [20_000.]
        ledger = BudgetLedgerV3(UNLIMITED, started=0., spool_path=folder / "live.jsonl",
                               monotonic=lambda: now[0])
        ledger.reserve("attempts")
        ledger.reserve_many({"episodes_started": 1, "decisions": 1, "requested_ticks": 6}, stage="teacher")
        now[0] += 10_000.
        ledger.enter_phase("closure")
        ledger.reserve("episodes_completed", cleanup=True)
        before = ledger.snapshot()
        ledger.publish_reservations(folder / "before.json")
        now[0] += 10_000.
        ledger.enter_phase("audit")
        ledger.reserve("checkpoint_loads", cleanup=True)
        ledger.reserve("audit_forward", 2, cleanup=True)
        now[0] += 10_000.
        ledger.enter_phase("publication")
        ledger.publish_reservations(folder / "after.json")
        return before, ledger.close()

    def test_late_complete_phase_history_requires_explicit_unlimited_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            before, after = self.fixture(folder)
            counts = cost_snapshot_v3(UNLIMITED,
                reservation_records(folder / "after.json", budget=UNLIMITED), after,
                closed=True, prefix=before, audit={"checkpoint_loads": 1, "audit_forward": 2})
            self.assertEqual(counts["decisions"], 1)
            self.assertEqual(counts["audit_forward"], 2)
            self.assertEqual(after["wall_seconds"], 50_000.)
            self.assertIsNone(after["maximum_wall_seconds"])
            with self.assertRaises(ContractError):
                list(reservation_records(folder / "after.json"))
            with self.assertRaises(ContractError):
                cost_snapshot_v3(BUDGET,
                    reservation_records(folder / "after.json", budget=UNLIMITED), after, closed=True)

    def test_null_snapshot_limits_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            _, after = self.fixture(folder)
            for field in ("maximum_wall_seconds", "cleanup_reserve_seconds"):
                changed = deepcopy(after)
                del changed[field]
                with self.subTest(field=field), self.assertRaises(ContractError):
                    cost_snapshot_v3(UNLIMITED,
                        reservation_records(folder / "after.json", budget=UNLIMITED), changed, closed=True)

    def test_unlimited_readback_still_refuses_nonfinite_or_reset_times(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            _, after = self.fixture(folder)
            original = list(reservation_records(folder / "after.json", budget=UNLIMITED))
            for elapsed in (float("inf"), float("nan"), -1.):
                rows = deepcopy(original)
                rows[-1]["elapsed_seconds"] = elapsed
                with self.subTest(elapsed=elapsed), self.assertRaises(ContractError):
                    cost_snapshot_v3(UNLIMITED, rows, after, closed=True)
            changed = deepcopy(after)
            changed["phase_history"][1]["elapsed_seconds"] = 19_999.
            with self.assertRaisesRegex(ContractError, "phase reset"):
                cost_snapshot_v3(UNLIMITED, original, changed, closed=True)

    def test_wrapper_passes_the_original_declaration_budget_to_lifecycle_reader(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "declaration.json").write_text(json.dumps({"budget": UNLIMITED}))
            with mock.patch("nca.train.jolt_executor_v3.verify_lifecycle",
                            return_value={"completed": False}) as verify:
                result = verify_execution(folder, terminal_stdout=folder / "stdout.log",
                    observed_exit_code=1, observed_exit_elapsed_seconds=50_000.)
            self.assertFalse(result["qualified"])
            self.assertEqual(verify.call_args.kwargs["expected_budget"], UNLIMITED)
            self.assertEqual(verify.call_args.kwargs["observed_exit_elapsed_seconds"], 50_000.)

    @unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning extra is not installed")
    def test_independent_auditor_replays_late_rows_and_retains_finite_time_rules(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            before, after = self.fixture(folder)
            rows = list(reservation_records(folder / "after.json", budget=UNLIMITED))
            replay = _Reservations(UNLIMITED, rows, after)
            self.assertEqual(replay.snapshot(before)["audit_forward"], 0)
            self.assertEqual(replay.snapshot(after)["audit_forward"], 2)
            changed = deepcopy(rows)
            changed[-1]["elapsed_seconds"] = float("inf")
            with self.assertRaises(ContractError):
                _Reservations(UNLIMITED, changed, after)


class UnlimitedNativeV3Tests(unittest.TestCase):
    def test_late_fragmented_startup_and_cleanup_have_no_hidden_timeouts(self):
        native, _, now, listener, connection, spawn = stream_fixtures.NativeStartupV3Tests.fixture(
            self, start=20_000., delay_per_fragment=4_000.)
        clock = OriginalClockV3(0., maximum_wall_seconds=None, monotonic=lambda: now[0])
        host = native.BoundedGodotHostV3("unused", "unused", original_clock=clock,
            verified_identity={"host_revision": native.PINNED_HOST_REVISION}, timeout=None)
        self.assertEqual(now[0], 28_000.)
        self.assertEqual(spawn.call_count, 1)
        self.assertIs(host._channel.connection, connection)
        listener.settimeout.assert_called_once_with(None)
        self.assertTrue(all(call.args == (None,) for call in connection.settimeout.call_args_list))
        host._process.poll.return_value = None
        with mock.patch.object(host, "request", return_value={"closed": True}) as request:
            host.close()
        request.assert_called_once_with("close")
        host._process.terminate.assert_called_once_with()
        host._process.wait.assert_called_once_with(timeout=None)
        host._reader.join.assert_called_once_with(timeout=None)
        connection.close.assert_called_once_with()


@unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning extra is not installed")
class UnlimitedCaptureAndTransportV3Tests(unittest.TestCase):
    def test_capture_budget_preserves_decision_limit_and_finite_measurements(self):
        budget = CaptureBudgetV3(40, None)
        self.assertFalse(budget.exhausted(39, 1_000_000.))
        self.assertTrue(budget.exhausted(40, 0.))
        self.assertTrue(CaptureBudgetV3(40, 1800.).exhausted(0, 1800.))
        for invalid in (float("inf"), float("nan"), -1.):
            with self.subTest(invalid=invalid), self.assertRaises(ContractError):
                budget.exhausted(0, invalid)

    def test_slow_capture_still_stops_at_native_tick240_and_forty_decisions(self):
        owner, ledger, journal = SyntheticBoundaryOwner("deadline"), SyntheticLedger(), Journal()
        ledger.remaining_seconds = lambda **kwargs: None
        case = {"id": "synthetic-unlimited", "host_seed": 7, "spawn": [0., 0.], "target": [48., 0.]}
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(EpisodeRecorder, "_wall", return_value=40_000_000_000_000):
            with synthetic_capture_provenance():
                episode, row, sequence = capture_v3(owner, ledger, journal, case,
                    identifier="unlimited-synthetic", controller_id="idle", kind="idle", stage="open",
                    output=Path(temp), root=ROOT, config={"profile_sha256": sha256_json(owner.profile)},
                    source_sha256="c" * 64)
        self.assertEqual(len(episode.decisions), 40)
        self.assertEqual(row["consumed_ticks"], 240)
        self.assertEqual(row["outcome"]["kind"], "task_terminal")
        self.assertEqual(row["outcome"]["reason"], "task_deadline")
        self.assertIsNone(sequence)

    def test_slow_owner_response_and_durability_keep_exact_consumption(self):
        with tempfile.TemporaryDirectory() as temp:
            now = [20_000.]
            ledger = BudgetLedgerV3(UNLIMITED, started=0., spool_path=Path(temp) / "live.jsonl",
                                   monotonic=lambda: now[0])
            journal = Journal()
            host = FakeHost(ledger, journal)
            owner = MeteredOwner(host, ledger, journal, transport_seconds=None, execution_version=3)
            request, write = host.request, journal.write
            def delayed_request(operation, **fields):
                self.assertIsNone(host._channel.connection.gettimeout())
                now[0] += 10_000.
                return request(operation, **fields)
            def delayed_write(payload):
                result = write(payload)
                if b'"event":"owner_response"' in payload:
                    now[0] += 10_000.
                return result
            try:
                with mock.patch("nca.train.jolt_screen.time.monotonic", side_effect=lambda: now[0]), \
                        mock.patch.object(host, "request", delayed_request), \
                        mock.patch.object(journal, "write", delayed_write):
                    owner.request("step", ticks=6)
                self.assertEqual(now[0], 40_000.)
                self.assertEqual(ledger.counts["requested_ticks"], 6)
                self.assertEqual(ledger.counts["confirmed_ticks"], 6)
                self.assertEqual(ledger.counts["uncertain_ticks"], 0)
                self.assertIsNone(owner.pending)
            finally:
                close_ledger(ledger)

    def test_unlimited_transport_does_not_reclassify_uncertain_native_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = BudgetLedgerV3(UNLIMITED, started=0., spool_path=Path(temp) / "live.jsonl",
                                   monotonic=lambda: 20_000.)
            journal = Journal()
            host = FakeHost(ledger, journal)
            host.fail = True
            owner = MeteredOwner(host, ledger, journal, transport_seconds=None, execution_version=3)
            try:
                with self.assertRaises(TimeoutError):
                    owner.request("step", ticks=6)
                self.assertEqual(ledger.counts["requested_ticks"], 6)
                self.assertEqual(ledger.counts["confirmed_ticks"], 0)
                self.assertEqual(ledger.counts["uncertain_ticks"], 6)
                self.assertEqual(owner.uncertain, [1])
                with self.assertRaisesRegex(ContractError, "uncertain prior work"):
                    owner.request("step", ticks=6)
            finally:
                close_ledger(ledger)


if __name__ == "__main__":
    unittest.main()
