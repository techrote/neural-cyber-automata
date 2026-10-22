"""Synthetic v3 budgets, bounded bytes and failure paths; no physical pilot."""
from copy import deepcopy
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from nca.contracts import ContractError, Record, WireAction, canonical_json, sha256_json
from nca.train.jolt_execution_v3 import (
    COUNTERS, STAGES, PHASE_OFFSETS, LEGACY_TIMING_SCALARS, OriginalClockV3,
    phase_offsets_v3, validate_budget_v3,
)
from nca.train.jolt_budget_v3 import (
    BudgetLedgerV3, MAX_PUBLISHED_BYTES, MAX_RECORD_BYTES, RESERVATION_RECORDS_MAX,
    reservation_records, _encoded,
)
from nca.train.jolt_storage_v3 import (
    BoundedVerifiedJournalV3, MAX_JOURNAL_RECORD_BYTES, canonical_chunks,
    publish_json_v3, _write_episode, episode_byte_limit, storage_bounds_v3,
    _BoundedCheckpointStream,
)
from nca.train.jolt_execution_contract import validate_v3_allocation_identity

ROOT = Path(__file__).resolve().parents[1]


def budget():
    return json.loads((ROOT / "docs/evidence/jolt-learn-003-allocation-plan-20261022.json").read_bytes())["proposed_budget"]


class OriginalClockTests(unittest.TestCase):
    def test_explicit_unlimited_clock_keeps_elapsed_and_named_phases_without_expiry(self):
        current = [0.]
        clock = OriginalClockV3(0., maximum_wall_seconds=None, monotonic=lambda: current[0])
        current[0] = 1_000_000.
        for phase in PHASE_OFFSETS:
            clock.check(phase, operation_seconds=None)
            clock.check(phase, operation_seconds=1_000_000.)
            self.assertIsNone(clock.remaining(phase))
        self.assertEqual(clock.elapsed(), 1_000_000.)
        self.assertEqual(clock.phase_offsets, dict.fromkeys(PHASE_OFFSETS))
        self.assertIsNone(clock.maximum_wall_seconds)
        with self.assertRaises(ContractError):
            clock.check("unknown")
        for allowance in (True, -1, float("nan"), float("inf")):
            with self.subTest(allowance=allowance), self.assertRaises(ContractError):
                clock.check("active", operation_seconds=allowance)
        current[0] = -1.
        with self.assertRaises(ContractError):
            clock.check("active", operation_seconds=None)
        with self.assertRaises(ContractError):
            clock.remaining("active")

    def test_unlimited_requires_explicit_null_not_a_large_or_nonfinite_limit(self):
        self.assertEqual(phase_offsets_v3(), PHASE_OFFSETS)
        self.assertEqual(phase_offsets_v3(None), dict.fromkeys(PHASE_OFFSETS))
        for limit in (True, 7200., 7201, 1_000_000, float("inf"), float("nan")):
            with self.subTest(limit=limit), self.assertRaises(ContractError):
                OriginalClockV3(0., maximum_wall_seconds=limit, monotonic=lambda: 0.)
        with self.assertRaises(ContractError):
            OriginalClockV3(0., monotonic=lambda: 0.).check("active", operation_seconds=None)

    def test_unlimited_budget_is_complete_and_keeps_operation_and_stage_caps(self):
        original = budget()
        unlimited = {**original, **dict.fromkeys(LEGACY_TIMING_SCALARS)}
        validate_budget_v3(unlimited)
        self.assertEqual(unlimited["counters"], original["counters"])
        self.assertEqual(unlimited["stages"], original["stages"])
        for field in LEGACY_TIMING_SCALARS:
            with self.subTest(field=field):
                mixed = {**unlimited, field: original[field]}
                with self.assertRaises(ContractError):
                    validate_budget_v3(mixed)
                missing = dict(unlimited)
                del missing[field]
                with self.assertRaises(ContractError):
                    validate_budget_v3(missing)
        changed = deepcopy(unlimited)
        changed["counters"]["owner_requests"] += 1
        with self.assertRaises(ContractError):
            validate_budget_v3(changed)

    def test_origin_zero_near_zero_and_arbitrary_epoch_share_exact_elapsed_cutoffs(self):
        for origin in (0., .00001, -10., 12345.):
            with self.subTest(origin=origin):
                current = [origin]
                clock = OriginalClockV3(origin, monotonic=lambda: current[0])
                for phase, cutoff in PHASE_OFFSETS.items():
                    current[0] = origin + cutoff - .25
                    clock.check(phase)
                    self.assertAlmostEqual(clock.remaining(phase), .25)
                    with self.assertRaises(ContractError):
                        clock.check(phase, operation_seconds=.25)
                    current[0] = origin + cutoff
                    with self.assertRaises(ContractError):
                        clock.check(phase)

    def test_nonfinite_future_and_backward_clock_are_refused(self):
        for origin in (True, None, float("nan"), float("inf"), 1.):
            with self.subTest(origin=origin), self.assertRaises(ContractError):
                OriginalClockV3(origin, monotonic=lambda: 0.)
        current = [0.]
        clock = OriginalClockV3(0., monotonic=lambda: current[0])
        current[0] = -1
        with self.assertRaises(ContractError):
            clock.remaining("active")

    def test_exact_thirty_nine_caps_and_integer_types(self):
        original = budget()
        self.assertEqual(original["counters"], COUNTERS)
        self.assertEqual(original["stages"], STAGES)
        self.assertEqual(sum(COUNTERS.values()), RESERVATION_RECORDS_MAX)
        validate_budget_v3(original)
        for change in ({"maximum_wall_seconds": 1800}, {"torch_threads": True}, {"new": 1}):
            with self.subTest(change=change), self.assertRaises(ContractError):
                validate_budget_v3({**original, **change})
        changed = deepcopy(original)
        changed["counters"]["attempts"] = True
        with self.assertRaises(ContractError):
            validate_budget_v3(changed)

    def test_v3_identity_never_reuses_old_or_consumed_declarations(self):
        value = {"schema": "nca-learn-jolt-003-execution/v3", "allocation_id": "learn-jolt-003-synthetic",
                 "authorization_ref": "https://github.com/techrote/neural-cyber-automata/issues/62#issuecomment-1"}
        validate_v3_allocation_identity(value)
        for changed in ({**value, "schema": "nca-learn-jolt-002-execution/v2"},
                        {**value, "allocation_id": "learn-jolt-002-consumed"}):
            with self.assertRaises(ContractError):
                validate_v3_allocation_identity(changed)
        with self.assertRaises(ContractError):
            validate_v3_allocation_identity(value, consumed_allocation_ids=(value["allocation_id"],))


class StreamingLedgerTests(unittest.TestCase):
    def ledger(self, root, clock=None):
        current = [0.] if clock is None else clock
        return BudgetLedgerV3(budget(), started=0., spool_path=root / "reservation-ledger.jsonl",
                              monotonic=lambda: current[0])

    def test_prefix_and_final_streams_bind_every_charge_without_whole_list(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ledger = self.ledger(root)
            ledger.reserve("attempts")
            ledger.reserve_many({"owner_requests": 1, "owner_step": 1, "decisions": 1,
                                 "requested_ticks": 6}, stage="teacher")
            snapshot = ledger.snapshot()
            ledger.publish_reservations(root / "before.json")
            rows = reservation_records(root / "before.json")
            self.assertEqual(len(rows), 2)
            self.assertEqual(sha256_json(list(rows)), snapshot["reservations_sha256"])
            self.assertEqual(list(rows), list(rows))
            self.assertFalse(hasattr(ledger, "_reservations"))
            ledger.enter_phase("closure")
            ledger.reserve("host_terminations", cleanup=True)
            ledger.enter_phase("audit")
            ledger.reserve("audit_forward", cleanup=True)
            ledger.enter_phase("publication")
            ledger.publish_reservations(root / "after.json")
            final = ledger.close()
            self.assertTrue(final["closed"])
            self.assertEqual(len(reservation_records(root / "after.json")), 4)
            self.assertEqual(len(rows), 2)
            self.assertEqual(sha256_json(list(reservation_records(root / "after.json"))), final["reservations_sha256"])
            with self.assertRaises(ContractError):
                ledger.reserve("audit_forward", cleanup=True)

    def test_atomic_stage_ceiling_and_refused_ticks_are_never_refunded(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = self.ledger(Path(folder))
            ledger.reserve_many({"owner_requests": 1, "owner_step": 1, "decisions": 1280,
                                 "requested_ticks": 7680}, stage="open")
            ledger.reserve("refused_ticks", 6, stage="open", cleanup=True)
            before = ledger.snapshot()
            with self.assertRaises(ContractError):
                ledger.reserve_many({"owner_requests": 1, "owner_step": 1, "decisions": 1,
                                     "requested_ticks": 6}, stage="open")
            self.assertEqual(before, ledger.snapshot())
            self.assertEqual(ledger.counts["requested_ticks"], 7680)
            self.assertEqual(ledger.counts["refused_ticks"], 6)
            ledger._stream.close()

    def test_active_and_cleanup_do_not_borrow_later_phase_time(self):
        with tempfile.TemporaryDirectory() as folder:
            clock = [5990.]
            ledger = self.ledger(Path(folder), clock)
            with self.assertRaises(ContractError):
                ledger.reserve("owner_close", cleanup=True, allowance_seconds=15)
            clock[0] = 6000.
            with self.assertRaises(ContractError):
                ledger.reserve("audit_forward", cleanup=True)
            ledger.enter_phase("closure")
            ledger.reserve("owner_close", cleanup=True)
            with self.assertRaises(ContractError):
                ledger.reserve("bc_forward", cleanup=True)
            ledger.enter_phase("audit")
            with self.assertRaises(ContractError):
                ledger.reserve("owner_cancel", cleanup=True)
            ledger.reserve("audit_forward", cleanup=True)
            with self.assertRaises(ContractError):
                ledger.enter_phase("active")
            ledger.enter_phase("publication")
            with self.assertRaises(ContractError):
                ledger.reserve("audit_forward", cleanup=True)
            ledger.close()

    def test_original_prefix_tampering_is_detected_without_repair(self):
        for kind in ("prefix", "tail", "suffix"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                ledger = self.ledger(root)
                ledger.reserve("bc_forward")
                ledger.reserve("bc_forward")
                ledger.snapshot()
                original = ledger.path.read_bytes()
                if kind == "prefix":
                    changed = original.replace(b'"bc_forward":1', b'"bc_burn_in":1', 1)
                elif kind == "tail":
                    changed = original[:-5]
                else:
                    changed = original + b"{}\n"
                ledger.path.write_bytes(changed)
                with self.assertRaises(ContractError):
                    ledger.publish_reservations(root / "reservations.json")
                self.assertEqual(ledger.path.read_bytes(), changed)
                ledger._stream.close()

    def test_readers_refuse_missing_tail_surplus_suffix_and_changed_count(self):
        mutations = (lambda b: b[:-3], lambda b: b + b" ",
                     lambda b: b.replace(b'"reservation_count":1', b'"reservation_count":2'))
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                ledger = self.ledger(root)
                ledger.reserve("attempts")
                target = root / "r.json"
                ledger.publish_reservations(target)
                target.write_bytes(mutate(target.read_bytes()))
                with self.assertRaises(ContractError):
                    list(reservation_records(target))
                ledger._stream.close()

    def test_worst_known_reservation_shapes_fit_enforced_record_bound(self):
        increments = [{key: maximum} for key, maximum in COUNTERS.items() if maximum]
        increments += [{"owner_requests": 1, "owner_step": 1, "decisions": 1, "requested_ticks": 6},
                       {"host_starts": 1, "host_start_attempts": 1}]
        for costs in increments:
            row = {"index": RESERVATION_RECORDS_MAX - 1, "elapsed_seconds": 0.00012345678901234567,
                   "stage": "teacher", "cleanup": False, "increments": costs, "phase": "publication"}
            self.assertLessEqual(len(_encoded(row)), MAX_RECORD_BYTES)
        self.assertLess(MAX_PUBLISHED_BYTES, 512 * 1024 * 1024)

    def test_streaming_publication_cannot_consume_next_phase(self):
        with tempfile.TemporaryDirectory() as folder:
            root, clock = Path(folder), [0.]
            ledger = self.ledger(root, clock)
            for _ in range(1024):
                ledger.reserve("bc_forward")
            ledger.enter_phase("closure")
            clock[0] = 6179.
            actual = ledger.check
            calls = [0]
            def bounded_check(**kwargs):
                calls[0] += 1
                if calls[0] == 3:
                    clock[0] = 6180.
                return actual(**kwargs)
            with mock.patch.object(ledger, "check", side_effect=bounded_check), self.assertRaises(ContractError):
                ledger.publish_reservations(root / "partial.json")
            self.assertTrue((root / "partial.json").exists())
            ledger._stream.close()


class StorageEnvelopeTests(unittest.TestCase):
    def test_streaming_json_preserves_existing_canonical_bytes(self):
        value = {"unicode": "£ λ", "nested": (None, False, 1, 1.25, WireAction(trim=-.3))}
        self.assertEqual(b"".join(canonical_chunks(value)), canonical_json(value).encode())
        for invalid in (float("nan"), float("inf"), {False: 1}, object()):
            with self.assertRaises(ContractError):
                b"".join(canonical_chunks(invalid))

    def test_size_interruption_preserves_partial_file_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "partial.json"
            with self.assertRaises(ContractError):
                publish_json_v3(target, {"value": "x" * 100}, maximum_bytes=32)
            self.assertTrue(target.exists())
            with self.assertRaises(FileExistsError):
                publish_json_v3(target, {})

    def test_gzip_envelope_has_exact_content_hash_and_complete_eof(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "synthetic.json.gz"
            record = WireAction(move=(.2, 0.), trim=.1)
            identity = _write_episode(target, record, decisions=1, compresslevel=1)
            raw = gzip.decompress(target.read_bytes())
            self.assertEqual(raw, (canonical_json({"episode": record, "sha256": record.sha256}) + "\n").encode())
            self.assertEqual(identity["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
            self.assertLessEqual(len(raw), episode_byte_limit(1))

    def test_journal_guard_includes_chain_metadata_and_preserves_prior_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "journal.jsonl"
            journal = BoundedVerifiedJournalV3(path)
            journal.write(b'{"event":"synthetic"}\n')
            before = path.read_bytes()
            with self.assertRaises(ContractError):
                journal.write((canonical_json({"event": "x" * (MAX_JOURNAL_RECORD_BYTES - 32)}) + "\n").encode())
            self.assertEqual(path.read_bytes(), before)
            journal.close()
            receipt = journal.durability_receipt()
            self.assertTrue(receipt["closed_verified"])
            self.assertEqual(receipt["schema"], "nca-owner-journal-durability/v3")
            self.assertEqual(receipt["maximum_record_bytes"], 65536)

    def test_checkpoint_write_is_bounded_before_an_oversized_chunk(self):
        output = io.BytesIO()
        stream = _BoundedCheckpointStream(output)
        stream.write(b"x" * 1024)
        with self.assertRaises(ContractError):
            stream.write(b"x" * 1024 ** 2)
        self.assertEqual(len(output.getvalue()), 1024)

    def test_bounded_checkpoint_keeps_unchanged_v2_untrained_bundle_reader(self):
        # Serialization conformance only: fresh random tensors, no model
        # forward, loss, optimizer, physical host or experiment checkpoint.
        try:
            import torch
        except ImportError:
            self.skipTest("optional CPU Torch is absent")
        from nca.controllers.jolt_learned import JoltPolicy, load_policy, policy_state_sha256
        from nca.train.jolt_storage_v3 import save_policy_v3
        with torch.random.fork_rng(), tempfile.TemporaryDirectory() as folder:
            torch.manual_seed(123)
            policy, base = JoltPolicy("residual"), JoltPolicy("bc")
            location = Path(folder) / "synthetic-untrained.pt"
            metadata = {"config_sha256": "f" * 64, "synthetic_only": True}
            receipt = save_policy_v3(location, policy, base_policy=base, metadata=metadata)
            restored, restored_base, retained = load_policy(location, expected_config_sha256="f" * 64,
                                                            expected_base_sha256=policy_state_sha256(base))
            self.assertEqual(policy_state_sha256(restored), receipt["policy_sha256"])
            self.assertEqual(policy_state_sha256(restored_base), receipt["base_sha256"])
            self.assertEqual(retained, metadata)
            self.assertLess(location.stat().st_size, 1024 * 1024)
            with self.assertRaises(FileExistsError):
                save_policy_v3(location, policy, base_policy=base, metadata=metadata)

    def test_aggregate_uses_decision_budget_not_maximum_length_for_every_start(self):
        bounds = storage_bounds_v3()
        self.assertEqual(bounds["raw_per_family_bytes"], 6464 * 65536 + 18944 * 131072)
        self.assertEqual(bounds["total_artifact_bytes"], sum(bounds["pieces"].values()))
        self.assertLess(bounds["required_disk_bytes_including_1_gib_reserve"], 17 * 1024 ** 3)
        self.assertLess(bounds["largest_episode_uncompressed_bytes"], 32 * 1024 ** 2)


if __name__ == "__main__":
    unittest.main()
