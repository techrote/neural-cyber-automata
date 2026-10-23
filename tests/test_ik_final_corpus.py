"""Opened-fixture tests only: never access accepted final data."""
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

from nca.contracts import ContractError, canonical_json, sha256_json
from nca.ik.data import DatasetConfig, generate_sample, iter_tasks
from nca.ik.reference import SolverConfig
from nca.train.ik_corpus import make_fixture_corpus, TrainingCorpus
from nca.evaluation.ik_training import freeze_candidates
from nca.evaluation import ik_final_corpus as api
from nca.registry import Registry


class IKFinalCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = DatasetConfig(mode="fixture", points_per_bin=(1, 1, 1),
            trajectories_per_partition=1, trajectory_steps=3, stress_per_class=1,
            teacher=SolverConfig(tolerance_cells=1e-7, max_seconds=1.))
        cls.rows = {p: tuple(generate_sample(t, cls.config) for t in iter_tasks(cls.config, p))
                    for p in ("training", "tuning", "final")}
        cls.training = make_fixture_corpus(cls.config, cls.rows["training"], cls.rows["tuning"])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.corpus, self.registry = self.root / "corpus", self.root / "registry"
        self.corpus.mkdir(); self.registry.mkdir()
        self.shards = []
        rows = self.rows["final"]
        for index, start in enumerate(range(0, len(rows), 64)):
            chunk = rows[start:start+64]
            raw = ("\n".join(s.to_json() for s in chunk) + "\n").encode()
            name = f"final-{index:05d}.jsonl"
            (self.corpus / name).write_bytes(raw)
            self.shards.append({"path": name, "partition": "final", "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(chunk),
                "sample_ids_sha256": sha256_json([s.task.id for s in chunk])})
        self.manifest = {"sha256": "a"*64, "shards": self.shards,
                         "schedule": {"partitions": {"final": len(rows)}}}
        (self.corpus / "manifest.json").write_text(canonical_json(self.manifest))
        self.frozen = freeze_candidates({seed: {
            "step": 40, "checkpoint_sha256": f"{seed:064x}", "tuning_report_sha256": "a"*64,
            "selection_metric": 0.1, "config_sha256": self.config.sha256,
            "tuning_schedule_sha256": "b"*64} for seed in (41011, 42013, 43019)},
            experiment="opened-fixture", dataset_manifest_sha256=self.training.manifest_sha256,
            final_suite_identity_sha256=self.training.final_suite_identity_sha256,
            registry_id=self.training.registry_id)

    def call(self, archive=None):
        return api.open_final_corpus(self.corpus, self.registry, self.root / "run", "a"*64,
            self.frozen, training_corpus=self.training,
            amendment_sha256="c"*64, source_sha256="d"*64, archive=archive)

    def fixture_archive(self):
        target = self.root / "fixture.tar.gz"
        with tarfile.open(target, "w:gz") as archive:
            for item in self.shards:
                raw = (self.corpus / item["path"]).read_bytes()
                member = tarfile.TarInfo("corpus/" + item["path"])
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
            raw = b"this mixed journal must never be selected or parsed"
            member = tarfile.TarInfo("corpus/pending.jsonl")
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
        return target

    def test_archive_selected_final_members_read_only_after_durable_opening(self):
        self.fixture_gates()
        target = self.fixture_archive()
        marker = self.registry / (self.training.final_suite_identity_sha256 + ".final-opened.json")
        original = tarfile.TarFile.extractfile
        selected = []
        def guard(archive, member, *args, **kwargs):
            self.assertTrue(marker.is_file())
            self.assertTrue(member.name.startswith("corpus/final-"))
            selected.append(member.name)
            return original(archive, member, *args, **kwargs)
        with patch.object(api, "ACCEPTED_ARCHIVE_SHA256", hashlib.sha256(target.read_bytes()).hexdigest()), \
             patch.object(api, "ACCEPTED_ARCHIVE_BYTES", target.stat().st_size), \
             patch.object(tarfile.TarFile, "extractfile", guard):
            view = self.call(target)
        self.assertTrue(view.accepted)
        self.assertEqual(selected, ["corpus/" + item["path"] for item in self.shards])
        self.assertEqual(view.samples, self.rows["final"])

    def test_archive_hash_and_incomplete_training_refuse_before_archive_decoding(self):
        self.fixture_gates()
        target = self.fixture_archive()
        with patch.object(tarfile, "open", side_effect=AssertionError("archive decoded before gate")):
            with self.assertRaises(ContractError):
                self.call(target)
        self.assertEqual(list(self.registry.iterdir()), [])
        with patch.object(api, "ACCEPTED_ARCHIVE_SHA256", hashlib.sha256(target.read_bytes()).hexdigest()), \
             patch.object(api, "ACCEPTED_ARCHIVE_BYTES", target.stat().st_size), \
             patch.object(api, "_verified_training", side_effect=ContractError("fixture", "incomplete")), \
             patch.object(tarfile, "open", side_effect=AssertionError("archive decoded before gate")):
            with self.assertRaises(ContractError):
                self.call(target)
        self.assertEqual(list(self.registry.iterdir()), [])

    def fixture_gates(self):
        """Only tests replace production trust anchors with opened fixture ones."""
        stack = ExitStack()
        self.addCleanup(stack.close)
        for target, value in (
            ("nca.evaluation.ik_final_corpus.ACCEPTED_MANIFEST_FILE_SHA256", None),
            ("nca.evaluation.ik_final_corpus.ACCEPTED_MANIFEST_SHA256", "a"*64),
            ("nca.evaluation.ik_final_corpus.ACCEPTED_REGISTRY_ID", self.training.registry_id),
            ("nca.evaluation.ik_final_corpus._REGISTRY_FILES", {})):
            stack.enter_context(patch(target, value))
        stack.enter_context(patch.object(TrainingCorpus, "accepted", property(lambda _: True)))
        stack.enter_context(patch("nca.train.ik_experiment._source", return_value={"source_sha256": "d"*64}))
        stack.enter_context(patch("nca.train.ik_experiment.corpus_identity", return_value={"fixture": True}))
        stack.enter_context(patch("nca.train.ik_experiment.load_declaration", return_value={
            "dataset": {"fixture": True}, "sha256": "e"*64}))
        stack.enter_context(patch.object(api, "_verified_training"))
        return stack

    def test_fixture_cannot_acquire_production_admission(self):
        with patch.object(api, "_read_rows", side_effect=AssertionError("payload touched")):
            with self.assertRaises(ContractError):
                self.call()

    def test_gate_failure_precedes_marker_and_payload(self):
        self.fixture_gates()
        with patch.object(api, "_verified_training", side_effect=ContractError("fixture", "incomplete training")), \
                patch.object(api, "_read_rows", side_effect=AssertionError("payload touched")):
            with self.assertRaises(ContractError):
                self.call()
        self.assertEqual(list(self.registry.iterdir()), [])

    def test_late_training_overrun_blocks_opening_before_marker_or_payload(self):
        verified_training = api._verified_training
        self.fixture_gates()
        seed = self.root / "run" / "seed-41011"
        seed.mkdir(parents=True)
        # This is the real failure shape: a provisional successful terminal
        # remains on disk when the supervisor's publication check overruns.
        (seed / "supervisor-start-0001.json").write_text("{}")
        (seed / "supervisor-result-0001.json").write_text(canonical_json({
            "returncode": 0, "verified_within_charged_wall": True,
            "unknown_work_blocks_resume": False,
            "whole_process_seconds": 1., "charged_wall_seconds": 2.}))
        (seed / "supervisor-overrun-0001.json").write_text(canonical_json({
            "resume_forbidden": True}))
        with patch.object(api, "_verified_training", side_effect=verified_training), \
                patch.object(api, "_read_rows", side_effect=AssertionError("payload touched")), \
                patch.object(api.IKRunLedger, "open", side_effect=AssertionError("ledger touched")):
            with self.assertRaisesRegex(ContractError, "overrun blocks final opening"):
                self.call()
        self.assertEqual(list(self.registry.iterdir()), [])

    def test_original_bytes_open_after_durable_marker_never_pending(self):
        self.fixture_gates()
        original = Path.open
        opened = []
        marker = self.registry / (self.training.final_suite_identity_sha256 + ".final-opened.json")
        def guard(path, *args, **kwargs):
            if path.name.startswith("final-"):
                self.assertTrue(marker.is_file(), "payload read before durable opening")
                self.assertTrue(json.loads(marker.read_text())["sha256"])
            self.assertNotIn(path.name, ("pending.jsonl", "attempts.jsonl"))
            opened.append(path.name)
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guard):
            view = self.call()
        self.assertTrue(view.accepted)
        self.assertTrue(view.authenticates(view.samples, view.config, view.opening, self.frozen))
        self.assertFalse(view.authenticates(tuple(replace(s) for s in view.samples), view.config))
        self.assertFalse(view.authenticates(view.samples[:-1], view.config))
        self.assertFalse(view.authenticates(view.samples, view.config, {"changed": True}, self.frozen))
        self.assertEqual(view.tasks, tuple(s.task for s in self.rows["final"]))
        self.assertEqual([s.to_dict() for s in view.samples], [s.to_dict() for s in self.rows["final"]])
        self.assertTrue(any(name.startswith("final-") for name in opened))
        # Identical readback is allowed; a different freeze/source is not.
        again = self.call()
        self.assertEqual(again.opening, view.opening)
        marker.write_text("{}")
        with self.assertRaises(ContractError):
            self.call()

    def test_original_shard_tamper_refuses(self):
        self.fixture_gates()
        path = self.corpus / self.shards[0]["path"]
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ContractError):
            self.call()

    def test_geometry_exact_semantics_and_existing_tolerances(self):
        expected = self.rows["final"][0].task
        tiny = replace(expected, inputs=replace(expected.inputs,
            target_root=(math.nextafter(expected.inputs.target_root[0], math.inf), expected.inputs.target_root[1])))
        api._task_geometry(tiny, expected)
        for changed in (replace(expected, stage=(expected.stage+1)%4),
                        replace(expected, family_id="different-family"),
                        replace(expected, inputs=replace(expected.inputs,
                            target_root=(expected.inputs.target_root[0] + 1e-5, expected.inputs.target_root[1]))),
                        replace(expected, original_point=(100., 100.))):
            with self.assertRaises(ContractError):
                api._task_geometry(changed, expected)

    def test_complete_schedule_and_cross_partition_identity_guards(self):
        allrows = self.rows["training"] + self.rows["tuning"] + self.rows["final"]
        api._split_guards(allrows, self.config)
        with self.assertRaises(ContractError):
            api._split_guards(allrows[:-1], self.config)
        with self.assertRaises(ContractError):
            api._split_guards(allrows + allrows[:1], self.config)
        changed = replace(allrows[-1], task=replace(allrows[-1].task,
                                                  family_id=allrows[0].task.family_id))
        with self.assertRaises(ContractError):
            api._split_guards(allrows[:-1] + (changed,), self.config)

    def test_registry_extension_declares_v1_contract(self):
        registry = Registry()
        descriptor = api.register(registry)
        self.assertEqual(descriptor.id, "ik-authenticated-final-corpus")
        self.assertIn("no-pending-journal", descriptor.capabilities)

    def test_all_three_actual_tuning_histories_precede_freeze(self):
        """A valid-looking freeze alone cannot open incomplete training."""
        states, selected = {}, {}
        declaration = {"allocations": {str(s): {"max_optimizer_steps": 40}
                                       for s in (41011, 42013, 43019)}}
        for seed in (41011, 42013, 43019):
            root = self.root / f"seed-{seed}"
            attempt = root / "attempts/attempt-0001"
            attempt.mkdir(parents=True)
            (root / "supervisor-start-0001.json").write_text("{}")
            (root / "supervisor-result-0001.json").write_text(canonical_json({
                "returncode": 0, "verified_within_charged_wall": True,
                "unknown_work_blocks_resume": False,
                "whole_process_seconds": 1., "charged_wall_seconds": 2.}))
            history, records = [], []
            for epoch in range(1, 41):
                metric = 1. / epoch
                report = {"schema": "nca-ik-direct-score/v1", "partition": "tuning",
                    "comparator": "trained_raw_reference", "complete": True,
                    "selection_metric": metric, "nominal": {"p95_error_cells": metric},
                    "config_sha256": self.config.sha256,
                    "schedule_sha256": sha256_json([s.task.id for s in self.training.tuning]),
                    "scheduled_calls": len(self.training.tuning),
                    "received_samples": len(self.training.tuning),
                    "received_predictions": len(self.training.tuning)}
                raw = canonical_json(report).encode()
                name = f"tuning-epoch-{epoch:04d}.json"
                (root / name).write_bytes(raw)
                item = {"epoch": epoch, "step": epoch, "file": name,
                    "file_sha256": hashlib.sha256(raw).hexdigest(),
                    "report_sha256": sha256_json(report)}
                history.append(item)
                receipt = {"reference": {"sha256": sha256_json([seed, epoch])},
                    "state": {"completed_optimizer_steps": epoch,
                        "sampler": {"cursor": epoch},
                        "extra_state": {"completed_epochs": epoch, "tuning_history": [item]}}}
                (attempt / f"event-{epoch:06d}.json").write_text(canonical_json({
                    "record": {"kind": "complete", "payload": {"receipt": receipt}}}))
                records.append({"step": epoch, "checkpoint_sha256": receipt["reference"]["sha256"],
                                "tuning": report})
            states[seed] = {"active": False, "unresolved_work": False, "last_status": "paused",
                "charged": {"sample_presentations": 210400},
                "latest_checkpoint": {"state": {
                    "extra_state": {"completed_epochs": 40, "tuning_history": history},
                    "sampler": {"epochs": 40, "cursor": 40, "sample_presentations": 210400}}}}
            selected[seed] = api.select_tuning_checkpoint(records)
        frozen = freeze_candidates(selected, experiment="fixture-histories",
            dataset_manifest_sha256=self.training.manifest_sha256,
            final_suite_identity_sha256=self.training.final_suite_identity_sha256,
            registry_id=self.training.registry_id)
        with patch("nca.train.ik_experiment.identity_for", side_effect=lambda d, s: s), \
                patch.object(api.IKRunLedger, "open") as opened:
            opened.side_effect = lambda *a, **k: type("FixtureLedger", (), {
                "inspect": lambda _: states[k["expected_identity"]]})()
            api._verified_training(self.root, declaration, self.training, frozen)
            bad = dict(frozen, candidates=list(frozen["candidates"]))
            bad["candidates"][0] = dict(bad["candidates"][0], checkpoint_sha256="9"*64)
            with self.assertRaises(ContractError):
                api._verified_training(self.root, declaration, self.training, bad)
            states[43019]["latest_checkpoint"]["state"]["extra_state"]["completed_epochs"] = 39
            with self.assertRaises(ContractError):
                api._verified_training(self.root, declaration, self.training, frozen)


if __name__ == "__main__":
    unittest.main()

# Copyright 2026 techrote. All rights reserved.
