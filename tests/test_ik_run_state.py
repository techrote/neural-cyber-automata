"""CPU-only conformance for cumulative IK accounting and exact state resume."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nca.train.ik_run_state import IKRunLedger
from nca.contracts import canonical_json, sha256_json


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None


def identity():
    return {**{key: char * 64 for key, char in (
        ("declaration_sha256", "a"), ("config_sha256", "b"), ("source_sha256", "c"),
        ("profile_sha256", "d"), ("dataset_sha256", "e"), ("normalization_sha256", "f"),
        ("runtime_sha256", "1"))}, "seed": 41011}


def allocation(**changes):
    return {
        "max_optimizer_steps": 4, "max_sample_presentations": 20,
        "max_tuning_calls": 5, "max_evaluation_calls": 20,
        "max_checkpoints": 5, "max_attempts": 4,
        "max_wall_seconds": 60.0, "max_memory_bytes": 1024 * 1024, **changes}


class Sampler:
    """Tiny deterministic state fixture, independent of the corpus sampler."""

    def __init__(self):
        self.cursor = self.sample_presentations = 0

    def advance(self, count):
        self.cursor += 1
        self.sample_presentations += count

    def state_dict(self):
        return {"schema": "tiny-ledger-fixture/v1", "cursor": self.cursor,
                "sample_presentations": self.sample_presentations}

    def load_state_dict(self, value):
        if (set(value) != {"schema", "cursor", "sample_presentations"}
                or value["schema"] != "tiny-ledger-fixture/v1"):
            raise ValueError("fixture sampler identity differs")
        self.cursor = value["cursor"]
        self.sample_presentations = value["sample_presentations"]


class IKRunLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "run"
        self.ledger = IKRunLedger.create(self.path, identity=identity(), allocation=allocation())

    def test_dependency_free_import_does_not_initialize_tensor_runtime(self):
        program = """
import sys
sys.path.insert(0, sys.argv[1])
import nca.train.ik_run_state
assert "torch" not in sys.modules
assert "numpy" not in sys.modules
"""
        subprocess.run([sys.executable, "-S", "-c", program, str(ROOT / "src")],
                       check=True, capture_output=True)

    def test_publication_reserve_is_conservatively_charged_inside_same_wall_cap(self):
        for cap, expected, status in ((120., 61., "paused"), (20., 20., "capped")):
            ledger = IKRunLedger.create(self.path.parent / f"reserve-{int(cap)}",
                identity=identity(), allocation=allocation(max_wall_seconds=cap))
            # Exact synthetic timestamps isolate the publication-reserve rule
            # from cancellation in (real fractional start + 1) - start.
            with patch("nca.train.ik_run_state.time.monotonic", return_value=100.):
                attempt = ledger.begin_attempt()
            with patch("nca.train.ik_run_state.time.monotonic", return_value=101.):
                state = attempt.finish("paused", reason="bounded publication",
                                       publication_reserve_seconds=60.)
            self.assertEqual(state["charged_wall_seconds"], expected)
            self.assertEqual(state["last_status"], status)
            self.assertEqual(state["allocation"]["max_wall_seconds"], cap)
            self.assertLessEqual(state["charged_wall_seconds"], cap)

    def test_invalid_publication_reserve_cannot_publish_or_grant_time(self):
        attempt = self.ledger.begin_attempt()
        for value in (True, -1., 4.999, 60.001, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                attempt.finish("paused", reason="invalid reserve", publication_reserve_seconds=value)
            self.assertFalse((attempt.folder / "finish.json").exists())
            self.assertFalse(attempt.closed)
        attempt.finish("paused", reason="legacy five-second reserve")

    def test_identity_limits_and_two_hour_ceiling_are_strict(self):
        for changes in ({"max_wall_seconds": 7201}, {"max_wall_seconds": float("nan")},
                        {"max_optimizer_steps": True}, {"max_attempts": 0},
                        {"max_evaluation_calls": -1}, {"max_memory_bytes": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                IKRunLedger.create(self.path.parent / "invalid", identity=identity(),
                                   allocation=allocation(**changes))
        bad = identity() | {"source_sha256": "wrong"}
        with self.assertRaises(ValueError):
            IKRunLedger.open(self.path, expected_identity=bad)
        with self.assertRaises(ValueError):
            IKRunLedger.open(self.path, expected_identity=identity() | {"seed": 42013})
        with self.assertRaises(ValueError):
            IKRunLedger.open(self.path, expected_identity=identity(),
                             expected_allocation=allocation(max_optimizer_steps=5))

    def test_failed_reserved_work_is_charged_and_cannot_reset_on_resume(self):
        attempt = self.ledger.begin_attempt()
        reservation = attempt.reserve("optimizer", samples=4, call_id="batch-0")
        self.assertEqual(reservation["cost"]["optimizer_steps"], 1)
        with self.assertRaises(ValueError):
            attempt.finish("paused", reason="cannot conceal in-flight work")
        result = attempt.finish("failed", reason="synthetic optimizer failure")
        self.assertEqual(result["charged"]["optimizer_steps"], 1)
        self.assertEqual(result["charged"]["sample_presentations"], 4)
        self.assertEqual(result["completed_optimizer_steps"], 0)
        self.assertGreaterEqual(result["charged_wall_seconds"], 5.0)
        self.assertTrue(result["unresolved_work"])
        with self.assertRaises(ValueError):
            IKRunLedger.open(self.path, expected_identity=identity()).begin_attempt()

    def test_unsealed_attempt_and_parallel_writer_refuse_new_allowance(self):
        first = self.ledger.begin_attempt()
        first.reserve("tuning", calls=3, call_id="tuning-0")
        other = IKRunLedger.open(self.path, expected_identity=identity())
        self.assertTrue(other.inspect()["active"])
        self.assertFalse(other.inspect()["can_resume"])
        with self.assertRaises(ValueError):
            other.begin_attempt()

    def test_all_kinds_keep_denominators_and_refuse_budget_overflow(self):
        attempt = self.ledger.begin_attempt()
        for kind, values in (
                ("presentation_only", {"samples": 3}),
                ("tuning", {"calls": 4}), ("evaluation", {"calls": 20})):
            reservation = attempt.reserve(kind, call_id=kind, **values)
            attempt.complete(reservation, {"fixture": True})
        with self.assertRaises(ValueError):
            attempt.reserve("evaluation", calls=1, call_id="overflow")
        with self.assertRaises(ValueError):
            attempt.reserve("tuning", calls=2, call_id="overflow")
        result = attempt.finish("failed", reason="fixture ends without a checkpoint")
        self.assertEqual(result["charged"], {
            "optimizer_steps": 0, "sample_presentations": 3,
            "tuning_calls": 4, "evaluation_calls": 20, "checkpoints": 0})
        self.assertEqual(result["optimizer_presentations"], 0)
        self.assertEqual(result["consumed_batches"], 1)
        self.assertTrue(result["unresolved_work"])

    def test_completed_but_uncheckpointed_tail_is_not_replayed(self):
        attempt = self.ledger.begin_attempt()
        reservation = attempt.reserve("optimizer", samples=2, call_id="batch")
        attempt.complete(reservation, {"fixture": True})
        result = attempt.finish("failed", reason="publication failed")
        self.assertEqual(result["completed_optimizer_steps"], 1)
        self.assertFalse(result["can_resume"])

    def test_mismatched_or_duplicate_completions_are_refused(self):
        attempt = self.ledger.begin_attempt()
        reservation = attempt.reserve("tuning", call_id="tuning")
        with self.assertRaises(ValueError):
            attempt.complete(reservation | {"id": "wrong"}, {})
        with self.assertRaises(ValueError):
            attempt.reserve("tuning", call_id="parallel")
        attempt.complete(reservation, {})
        with self.assertRaises(ValueError):
            attempt.complete(reservation, {})

    def test_wall_reserve_refuses_work_before_whole_process_ceiling(self):
        attempt = self.ledger.begin_attempt()
        attempt.started = time.monotonic() - 58.0
        with self.assertRaisesRegex(ValueError, "reserve"):
            attempt.reserve("optimizer", samples=1, call_id="late")
        result = attempt.finish("capped", reason="synthetic clock")
        self.assertEqual(result["charged_wall_seconds"], 60.0)
        self.assertEqual(result["last_status"], "capped")
        self.assertFalse(result["can_resume"])

    def test_startup_failures_consume_attempts_and_wall_without_resetting(self):
        for index in range(4):
            attempt = self.ledger.begin_attempt()
            result = attempt.finish("failed", reason="fixture startup refusal")
            self.assertEqual(result["attempt_count"], index + 1)
            self.assertGreaterEqual(result["charged_wall_seconds"], 5.0 * (index + 1))
        self.assertFalse(result["can_resume"])
        with self.assertRaises(ValueError):
            self.ledger.begin_attempt()

    def test_terminal_inventory_catches_byte_changes_and_unlisted_files(self):
        attempt = self.ledger.begin_attempt()
        attempt.finish("failed", reason="fixture")
        start = self.path / "attempts/attempt-0001/start.json"
        start.write_bytes(start.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.ledger.inspect()

    def test_supervisor_unknown_outcome_cannot_be_bypassed_by_low_level_resume(self):
        for filename, value in (
                ("supervisor-overrun-0001.json", {"resume_forbidden": True}),
                ("supervisor-result-0001.json", {"verified_within_charged_wall": False})):
            with self.subTest(filename=filename):
                path = self.path / filename
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "supervisor|overrun"):
                    self.ledger.begin_attempt()
                path.unlink()
        attempt = self.ledger.begin_attempt()
        attempt.finish("failed", reason="fixture startup refusal")
        (self.path / "supervisor-start-0001.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "supervisor"):
            self.ledger.begin_attempt()

    def test_missing_terminal_seal_cannot_be_resumed(self):
        attempt = self.ledger.begin_attempt()
        attempt.finish("failed", reason="fixture")
        (self.path / "attempts/attempt-0001/sealed.json").unlink()
        result = self.ledger.inspect()
        self.assertTrue(result["active"])
        self.assertFalse(result["can_resume"])


@unittest.skipUnless(OPTIONAL, "optional pinned IK Torch/NumPy runtime is not installed")
class IKTrainingResumeCPUConformanceTests(unittest.TestCase):
    def setUp(self):
        import numpy
        import torch
        self.numpy, self.torch = numpy, torch
        torch.set_num_threads(1)
        torch.manual_seed(41011)
        numpy.random.seed(41011)
        random.seed(41011)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "run"
        self.ledger = IKRunLedger.create(self.path, identity=identity(), allocation=allocation())
        self.sampler = Sampler()
        self.trainer = self.make_trainer()

    def make_trainer(self):
        model = self.torch.nn.Linear(2, 2)
        return SimpleNamespace(
            model=model, optimizer=self.torch.optim.Adam(model.parameters(), lr=.001),
            config_sha256="2" * 64, normalization_sha256="3" * 64,
            weights=SimpleNamespace(sha256="4" * 64), attempted_steps=0,
            committed_steps=0, sample_presentations=0, failed=False)

    def update(self, trainer):
        trainer.attempted_steps += 1
        trainer.sample_presentations += 2
        trainer.optimizer.zero_grad(set_to_none=True)
        loss = trainer.model(self.torch.rand(2, 2)).square().mean()
        loss.backward()
        trainer.optimizer.step()
        trainer.committed_steps += 1
        return {"fixture_loss": float(loss.detach())}

    def recorded_update(self, attempt, trainer, sampler):
        reservation = attempt.reserve("optimizer", samples=2, call_id="fixture-batch")
        sampler.advance(2)
        attempt.complete(reservation, self.update(trainer))

    def test_exact_optimizer_sampler_global_rng_and_future_update_resume(self):
        attempt = self.ledger.begin_attempt()
        self.assertEqual(self.ledger.resume(self.trainer, self.sampler), {})
        initial = attempt.checkpoint(self.trainer, self.sampler.state_dict())
        self.assertEqual(initial["state"]["trainer"]["committed_steps"], 0)
        self.recorded_update(attempt, self.trainer, self.sampler)
        reference = attempt.checkpoint(self.trainer, self.sampler.state_dict(),
                                       extra_state={"tuning_history": [], "completed_epochs": 1})
        first = attempt.finish("paused", reason="fixture completed boundary")
        self.assertTrue(first["can_resume"])
        expected_random = (random.random(), float(self.numpy.random.rand()),
                           self.torch.rand(3).clone())
        self.update(self.trainer)
        expected_model = {k: v.detach().clone() for k, v in self.trainer.model.state_dict().items()}
        expected_optimizer = deepcopy(self.trainer.optimizer.state_dict())

        resumed = IKRunLedger.open(self.path, expected_identity=identity(),
                                   expected_allocation=allocation())
        second = resumed.begin_attempt()
        fresh, sampler = self.make_trainer(), Sampler()
        extras = resumed.resume(fresh, sampler)
        self.assertEqual(extras, {"tuning_history": [], "completed_epochs": 1})
        self.assertEqual(sampler.state_dict(), reference["state"]["sampler"])
        self.assertEqual(fresh.attempted_steps, 1)
        self.assertEqual(fresh.sample_presentations, 2)
        self.assertEqual(random.random(), expected_random[0])
        self.assertEqual(float(self.numpy.random.rand()), expected_random[1])
        self.assertTrue(self.torch.equal(self.torch.rand(3), expected_random[2]))
        self.recorded_update(second, fresh, sampler)
        for key, value in fresh.model.state_dict().items():
            self.assertTrue(self.torch.equal(value, expected_model[key]))
        for key, state in fresh.optimizer.state_dict()["state"].items():
            for name, value in state.items():
                self.assertTrue(self.torch.equal(value, expected_optimizer["state"][key][name]))
        second.checkpoint(fresh, sampler.state_dict())
        result = second.finish("paused", reason="fixture resumed boundary")
        self.assertEqual(result["charged"]["optimizer_steps"], 2)
        self.assertEqual(result["charged"]["sample_presentations"], 4)
        self.assertEqual(result["charged"]["checkpoints"], 3)
        self.assertGreaterEqual(result["charged_wall_seconds"], 10.0)

    def test_presentation_only_rows_are_retained_without_invented_updates(self):
        attempt = self.ledger.begin_attempt()
        reservation = attempt.reserve("presentation_only", samples=3, call_id="masked")
        self.sampler.advance(3)
        attempt.complete(reservation, {"no_eligible_supervision": True})
        attempt.checkpoint(self.trainer, self.sampler.state_dict())
        result = attempt.finish("paused", reason="retained masked batch")
        self.assertEqual(result["charged"]["sample_presentations"], 3)
        self.assertEqual(result["completed_optimizer_steps"], 0)
        self.assertTrue(result["can_resume"])

    def test_tuning_after_checkpoint_prevents_rewinding_selection_on_resume(self):
        attempt = self.ledger.begin_attempt()
        attempt.checkpoint(self.trainer, self.sampler.state_dict())
        reservation = attempt.reserve("tuning", calls=2, call_id="tuning-tail")
        attempt.complete(reservation, {"fixture_scores": [1.0, 2.0]})
        result = attempt.finish("failed", reason="selection checkpoint missing")
        self.assertEqual(result["charged"]["tuning_calls"], 2)
        self.assertFalse(result["can_resume"])

    def test_sampler_or_trainer_counter_changes_cannot_be_published(self):
        attempt = self.ledger.begin_attempt()
        with self.assertRaisesRegex(ValueError, "counters"):
            attempt.checkpoint(self.trainer, {"cursor": 1, "sample_presentations": 2})
        self.trainer.failed = True
        with self.assertRaisesRegex(ValueError, "healthy"):
            attempt.checkpoint(self.trainer, self.sampler.state_dict())
        self.assertEqual(attempt.charged["checkpoints"], 0)

    def test_normalization_or_loss_change_is_refused_before_restore(self):
        first = self.ledger.begin_attempt()
        first.checkpoint(self.trainer, self.sampler.state_dict())
        first.finish("paused", reason="fixture")
        second = self.ledger.begin_attempt()
        self.trainer.normalization_sha256 = "9" * 64
        with self.assertRaisesRegex(ValueError, "differs"):
            self.ledger.resume(self.trainer, self.sampler)
        second.finish("failed", reason="fixture identity refusal")

    def test_checkpoint_corruption_is_caught_by_dependency_free_inspection(self):
        attempt = self.ledger.begin_attempt()
        reference = attempt.checkpoint(self.trainer, self.sampler.state_dict())
        attempt.finish("paused", reason="fixture")
        target = self.path / reference["path"]
        contents = bytearray(target.read_bytes())
        contents[len(contents) // 2] ^= 1
        target.write_bytes(contents)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.ledger.inspect()

    def recovery_fixture(self, *, wall=1000.):
        path = self.path.parent / "recovery-run"
        ledger = IKRunLedger.create(path, identity=identity(), allocation=allocation(max_wall_seconds=wall))
        first = ledger.begin_attempt()
        cp = first.checkpoint(self.trainer, self.sampler.state_dict())
        prior = first.finish("paused", reason="complete checkpoint fixture")
        index = prior["attempt_count"] + 1
        failure = {"phase": "evaluate-all", "attempt": index, "seed": 41011, "returncode": 1,
            "verified_within_charged_wall": False, "unknown_work_blocks_resume": True,
            "charged_wall_seconds": 0., "whole_process_seconds": 121.734}
        files = {}
        for name, value in ((f"supervisor-result-{index:04d}.json", failure),
                            (f"supervisor-overrun-{index:04d}.json", {"resume_forbidden": True})):
            raw = (canonical_json(value)+"\n").encode()
            (path / name).write_bytes(raw)
            files[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        body = {"schema": "nca-ik-ledger-recovery/v1", "identity_sha256": sha256_json(identity()),
            "allocation_sha256": ledger.header["sha256"], "seed": 41011,
            "prior_parent_sha256": prior["parent_sha256"], "prior_attempt_count": prior["attempt_count"],
            "checkpoint_sha256": cp["reference"]["sha256"], "failed_attempt": index, "failure_files": files,
            "source_sha256": "5"*64, "authority_sha256": "6"*64,
            "frozen_candidates_sha256": "7"*64, "opening_sha256": "8"*64,
            "historical_elapsed_seconds": 130.984, "historical_charge_wall_seconds": 131.}
        proof = dict(body, sha256=sha256_json(body))
        return path, ledger, prior, proof

    def test_explicit_recovery_debits_history_preserves_failed_bytes_and_consumes_attempt(self):
        path, ledger, prior, proof = self.recovery_fixture()
        with self.assertRaises(ValueError):
            ledger.begin_attempt()
        opted = IKRunLedger.open(path, expected_identity=identity(), recovery_proof=proof)
        with self.assertRaises(ValueError):
            opted.begin_attempt()  # A Boolean/proof alone does not pay for failed work.
        state = opted.record_recovery(proof)
        self.assertEqual(state["attempt_count"], prior["attempt_count"] + 1)
        self.assertEqual(state["last_status"], "failed")
        self.assertEqual(state["charged"], prior["charged"])
        self.assertEqual(state["latest_checkpoint"], prior["latest_checkpoint"])
        self.assertGreaterEqual(state["charged_wall_seconds"] - prior["charged_wall_seconds"], 131.)
        self.assertLessEqual(state["charged_wall_seconds"], state["allocation"]["max_wall_seconds"])
        self.assertEqual(opted.record_recovery(proof)["charged_wall_seconds"], state["charged_wall_seconds"])
        for name, expected in proof["failure_files"].items():
            self.assertEqual(hashlib.sha256((path/name).read_bytes()).hexdigest(), expected["sha256"])
            self.assertTrue(opted.allows_recovered_failure(path/name))
        self.assertFalse(opted.allows_recovered_failure(path/"supervisor-result-0004.json"))
        with self.assertRaises(ValueError):
            IKRunLedger.open(path, expected_identity=identity()).begin_attempt()
        actual = opted.begin_attempt()
        self.assertEqual(actual.number, proof["failed_attempt"] + 1)
        actual.finish("failed", reason="fixture stops without evaluation or optimization")

    def test_recovery_refuses_undercharge_other_failure_and_changed_old_bytes(self):
        path, ledger, prior, proof = self.recovery_fixture()
        for field, value in (("historical_charge_wall_seconds", 130.), ("checkpoint_sha256", "0"*64)):
            bad = deepcopy(proof)
            bad[field] = value
            bad["sha256"] = sha256_json({k:v for k,v in bad.items() if k != "sha256"})
            with self.assertRaises(ValueError):
                ledger.record_recovery(bad)
        (path / "supervisor-overrun-0004.json").write_text('{"resume_forbidden":true}')
        with self.assertRaises(ValueError):
            ledger.record_recovery(proof)
        self.assertFalse((path / "attempts/attempt-0002").exists())
        terminal = path / "supervisor-result-0002.json"
        terminal.write_bytes(terminal.read_bytes()+b" ")
        with self.assertRaises(ValueError):
            IKRunLedger.open(path, expected_identity=identity(), recovery_proof=proof)

    def test_recovery_debit_can_exhaust_but_never_expand_original_wall_allowance(self):
        path, ledger, prior, proof = self.recovery_fixture(wall=150.)
        state = ledger.record_recovery(proof)
        self.assertEqual(state["allocation"]["max_wall_seconds"], 150.)
        self.assertEqual(state["charged_wall_seconds"], 150.)
        self.assertEqual(state["last_status"], "capped")
        self.assertFalse(state["can_resume"])
        with self.assertRaises(ValueError):
            ledger.begin_attempt()


if __name__ == "__main__":
    unittest.main()
