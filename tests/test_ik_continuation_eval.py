"""CPU-only final orchestration boundaries; no accepted final data or GPU."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from nca.contracts import canonical_json, sha256_json
from nca.evaluation import ik_continuation_eval as api


class IKContinuationEvaluationTests(unittest.TestCase):
    def histories(self, root):
        states, ledgers = {}, {}
        for seed in api.SEEDS:
            folder = root / f"seed-{seed}"
            events = folder / "attempts/attempt-0001"
            events.mkdir(parents=True)
            history = []
            for epoch in range(1, 41):
                metric = 0.1 if epoch in (10, 20) else 1.0
                report = {"schema": "nca-ik-direct-score/v1", "partition": "tuning",
                    "comparator": "trained_raw_reference", "complete": True,
                    "selection_metric": metric, "nominal": {"p95_error_cells": metric},
                    "config_sha256": "a" * 64, "schedule_sha256": "b" * 64}
                raw = canonical_json(report).encode()
                name = f"tuning-epoch-{epoch:04d}.json"
                (folder / name).write_bytes(raw)
                item = {"epoch": epoch, "step": epoch, "file": name,
                    "file_sha256": hashlib.sha256(raw).hexdigest(), "report_sha256": sha256_json(report)}
                history.append(item)
                receipt = {"path": f"epoch-{epoch}.pt", "reference": {"sha256": sha256_json([seed, epoch])},
                    "state": {"completed_optimizer_steps": epoch,
                        "extra_state": {"completed_epochs": epoch, "tuning_history": [item]}}}
                (events / f"event-{epoch:06d}.json").write_text(canonical_json({
                    "record": {"kind": "complete", "payload": {"receipt": receipt}}}))
            state = states[seed] = {"active": False, "unresolved_work": False,
                "latest_checkpoint": {"state": {"extra_state": {"tuning_history": history}}}}
            ledgers[seed] = SimpleNamespace(root=folder, inspect=lambda state=state: state)
        return states, ledgers

    def test_freeze_reconstructs_all_histories_and_earliest_tuning_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            states, ledgers = self.histories(root)
            corpus = SimpleNamespace(manifest_sha256="c" * 64,
                final_suite_identity_sha256="d" * 64, registry_id="fixture-registry")
            with patch.object(api, "_ledger", side_effect=lambda r, d, s: ledgers[s]), \
                 patch.object(api, "_verified_training") as verified:
                frozen, paths = api.freeze_training(root, {"id": "fixture-freeze"}, corpus)
                self.assertEqual([x["step"] for x in frozen["candidates"]], [10, 10, 10])
                for seed in api.SEEDS:
                    self.assertEqual(paths[(seed, sha256_json([seed, 10]))], "epoch-10.pt")
                verified.assert_called_once_with(root, {"id": "fixture-freeze"}, corpus, frozen)
                # A modified actual report cannot be rehashed by selection.
                (root / f"seed-{api.SEEDS[0]}" / "tuning-epoch-0001.json").write_bytes(b"{}")
                with self.assertRaisesRegex(ValueError, "tuning file changed"):
                    api.freeze_training(root, {"id": "fixture-freeze"}, corpus)
                states[api.SEEDS[0]]["latest_checkpoint"]["state"]["extra_state"]["tuning_history"].pop()
                with self.assertRaisesRegex(ValueError, "forty scheduled"):
                    api.freeze_training(root, {"id": "fixture-freeze"}, corpus)

    def test_failed_worker_retains_unknown_outcome_for_every_original_seed(self):
        import nca.train.ik_continuation as continuation
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            before = {"can_resume": True, "charged": {"evaluation_calls": 0},
                "remaining_wall_seconds": 100., "charged_wall_seconds": 0., "attempt_count": 2,
                "active": False, "unresolved_work": False, "last_status": "paused"}
            for seed in api.SEEDS:
                (run / f"seed-{seed}").mkdir(parents=True)
            fake = SimpleNamespace(returncode=7, wait=lambda **k: None)
            amendment = {"sha256": "a" * 64, "source": {"source_sha256": "b" * 64}}
            ledger = SimpleNamespace(inspect=lambda: deepcopy(before))
            with patch.object(continuation, "load_amendment", return_value=({}, amendment)), \
                 patch.object(continuation, "_review", return_value={"fixture": True}), \
                 patch.object(api, "_ledger", return_value=ledger), \
                 patch.object(api.subprocess, "Popen", return_value=fake):
                with self.assertRaisesRegex(ValueError, "failed or uncertain"):
                    api.supervise(run, root / "amendment.json", "c" * 64,
                                  root / "corpus", root / "registry", root / "review", root / "archive")
            terminal = api._read(root / "evaluation-supervisor-result.json")
            self.assertTrue(terminal["unknown_work_blocks_resume"])
            for seed in api.SEEDS:
                result = api._read(run / f"seed-{seed}" / "supervisor-result-0003.json")
                self.assertEqual(result["returncode"], 7)
                self.assertTrue(result["unknown_work_blocks_resume"])
                self.assertFalse(result["verified_within_charged_wall"])
                self.assertEqual(result["consumed_or_reserved_wall_seconds"], 100.)

    def test_all_evaluation_calls_reserved_before_first_comparator_and_failure_seals_attempts(self):
        import nca.train.ik_continuation as continuation
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            (run).mkdir()
            (run / "declaration.json").write_text("{}")
            execution = {"amendment_sha256": "a" * 64,
                         "source_sha256": "b" * 64, "source_commit": "fixture-head"}
            amendment = {"sha256": execution["amendment_sha256"],
                "source": {"source_sha256": execution["source_sha256"], "revision": "fixture-head"}}
            declaration = {"dataset": {"fixture": True}, "gpu_visibility_sha256": "c" * 64,
                "sha256": "d" * 64,
                "plan": {"qualified_environment": {"driver": "fixture"}, "max_memory_bytes": 2**31}}
            review = {"fixture": True}
            reserved, attempts, ledgers = [], {}, {}
            for seed in api.SEEDS:
                folder = run / f"seed-{seed}"
                folder.mkdir()
                attempt = attempts[seed] = MagicMock(closed=False)
                attempt.reserve.side_effect = lambda kind, *, calls, call_id, seed=seed: (
                    reserved.append((seed, kind, calls)), f"token-{seed}")[1]
                ledgers[seed] = SimpleNamespace(root=folder, identity={},
                    inspect=lambda: {"attempt_count": 1, "remaining_wall_seconds": 100.},
                    begin_attempt=lambda *, started, attempt=attempt: attempt,
                    resume=lambda *args: {"execution": execution,
                        "initial_checkpoint": {"path": "initial.pt", "reference": {"sha256": "f" * 64}}})
            frozen = {"candidates": [{"seed": s, "checkpoint_sha256": str(s)} for s in api.SEEDS]}
            paths = {(s, str(s)): "selected.pt" for s in api.SEEDS}
            corpus = SimpleNamespace(normalization=None, config=None, training=())
            context = SimpleNamespace(torch=SimpleNamespace(set_num_threads=lambda _: None), report={})
            sampler = SimpleNamespace(exhausted=True)
            model = MagicMock()
            def fail_comparison(*args, **kwargs):
                self.assertEqual(reserved, [(s, "evaluation", 65200) for s in api.SEEDS])
                raise ValueError("fixture comparator failed")
            with patch.object(continuation, "load_amendment", return_value=(declaration, amendment)), \
                 patch.object(continuation, "_review", return_value=review), \
                 patch.object(api, "_titan_visibility"), \
                 patch.object(api, "read_training_corpus", return_value=corpus), \
                 patch.object(api, "corpus_identity", return_value=declaration["dataset"]), \
                 patch.object(api, "freeze_training", return_value=(frozen, paths)), \
                 patch.object(api, "open_final_corpus", return_value=SimpleNamespace(samples=(), config=None, opening={})), \
                 patch.object(api, "_ledger", side_effect=lambda r, d, s: ledgers[s]), \
                 patch.object(api, "identity_for", return_value={"config_sha256": "e" * 64}), \
                 patch.object(api, "device_report", return_value={}), \
                 patch.object(api, "_qualified_titan_gpu_bytes", return_value=24*2**30), \
                 patch.object(api, "configure_device", return_value=context), \
                 patch.object(api, "IKTrainer", return_value=SimpleNamespace(context=context)), \
                 patch.object(api, "PairPreservingSampler", return_value=sampler), \
                 patch.object(api, "load_checkpoint", return_value={"model": {}}), \
                 patch.object(api, "build_model", return_value=model), \
                 patch.object(api, "model_predictor", return_value=lambda _: None), \
                 patch.object(api, "reference_predictors", return_value={}), \
                 patch.object(api, "evaluate_comparison", side_effect=fail_comparison):
                with self.assertRaisesRegex(ValueError, "fixture comparator failed"):
                    api.worker(run, root / "amendment.json", "f" * 64,
                        root / "corpus", root / "registry", root / "review", root / "archive",
                        started=0., review_sha256=sha256_json(review))
            for attempt in attempts.values():
                attempt.finish.assert_called_once_with("failed", reason="ValueError")
                attempt.complete.assert_not_called()
                attempt.checkpoint.assert_not_called()


if __name__ == "__main__":
    unittest.main()

# Copyright 2026 techrote. All rights reserved.
