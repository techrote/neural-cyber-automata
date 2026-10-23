"""CPU-only corpus authentication and deterministic continuation conformance."""

from collections import Counter
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from nca.contracts import ContractError, canonical_json, sha256_json
from nca.ik.data import (DatasetConfig, build_split_plan, fit_normalization,
                         generate_sample, iter_tasks, validate_sample, TeacherLabel,
                         _authenticated_windows_label_replay, _windows_replay_inventory)
from nca.ik.profiles import runner_right_sole_profile
from nca.ik.reference import SolverConfig
from nca.train.ik_corpus import (
    ACCEPTED_CONFIG_SHA256, ACCEPTED_MANIFEST_SHA256, ACCEPTED_MANIFEST_FILE_SHA256,
    ACCEPTED_NORMALIZATION_FILE_SHA256, ACCEPTED_TRAINING_INPUTS_SHA256,
    PairPreservingSampler, TrainingCorpus, make_fixture_corpus,
    read_fixture_corpus, read_training_corpus,
)


ROOT = Path(__file__).resolve().parents[1]


def seal(value):
    return {**value, "sha256": sha256_json(value)}


def write_json(path, value):
    raw = (canonical_json(value) + "\n").encode()
    path.write_bytes(raw)
    return {"path": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class IKCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = DatasetConfig(mode="fixture", points_per_bin=(1, 1, 1),
                                   trajectories_per_partition=1, trajectory_steps=3,
                                   stress_per_class=1,
                                   teacher=SolverConfig(tolerance_cells=1e-7, max_seconds=1.0))
        cls.training = tuple(generate_sample(t, cls.config) for t in iter_tasks(cls.config, "training"))
        cls.tuning = tuple(generate_sample(t, cls.config) for t in iter_tasks(cls.config, "tuning"))
        cls.normalization = fit_normalization(cls.training, cls.config)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.corpus, self.registry = base / "corpus", base / "registry"
        self.corpus.mkdir()
        self.registry.mkdir()
        self.registry_id, self.suite_id = "1" * 32, "e" * 64
        write_json(self.registry / "registry.json",
                   {"schema": "nca-ik-dataset-registry/v1", "id": self.registry_id})
        counts = {"training": len(self.training), "tuning": len(self.tuning),
                  "final": 80 + self.config.trajectory_steps + 7}
        schedule = {"partitions": counts, "tasks": sum(counts.values()),
                    "final_suite_identity_sha256": self.suite_id}
        plan = build_split_plan(self.config)
        self.declaration = seal({
            "schema": "nca-ik-dataset-declaration/v1", "config": self.config.to_dict(),
            "config_sha256": self.config.sha256, "split_plan": plan,
            "split_plan_sha256": sha256_json(plan), "registry_id": self.registry_id,
            "run_id": "fixture-corpus", "schedule": schedule,
            "profile_sha256": runner_right_sole_profile().sha256,
        })
        artifacts = [write_json(self.corpus / "declaration.json", self.declaration),
                     write_json(self.corpus / "normalization.json", self.normalization.to_dict())]
        # The training view authenticates these entries through its manifest,
        # but never reads pending.jsonl because it would include final rows.
        artifacts += [{"path": name, "bytes": 1, "sha256": "a" * 64}
                      for name in ("pending.jsonl", "attempts.jsonl", "coverage.json")]
        shards = []
        for partition, rows in (("training", self.training), ("tuning", self.tuning)):
            for index, start in enumerate(range(0, len(rows), self.config.batch_size)):
                chunk = rows[start:start + self.config.batch_size]
                raw = "".join(s.to_json() + "\n" for s in chunk).encode()
                name = f"{partition}-{index:05d}.jsonl"
                (self.corpus / name).write_bytes(raw)
                shards.append({"path": name, "partition": partition, "rows": len(chunk),
                               "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                               "sample_ids_sha256": sha256_json([s.task.id for s in chunk])})
        # Deliberately absent final files prove the view never reads them.
        for index, start in enumerate(range(0, counts["final"], self.config.batch_size)):
            shards.append({"path": f"final-{index:05d}.jsonl", "partition": "final",
                           "rows": min(self.config.batch_size, counts["final"] - start),
                           "bytes": 1, "sha256": "a" * 64, "sample_ids_sha256": "b" * 64})
        self.manifest = {
            "schema": "nca-ik-dataset-manifest/v1", "run_id": "fixture-corpus",
            "status": "complete", "declaration_sha256": self.declaration["sha256"],
            "registry_id": self.registry_id, "config_sha256": self.config.sha256,
            "split_plan_sha256": sha256_json(plan), "schedule": schedule,
            "shards": shards, "artifacts": artifacts, "final_suite_identity_sha256": self.suite_id,
            "final_state": "development_fixture", "all_samples_sha256": "c" * 64,
            "counts": {}, "generation_cost": {}, "failures": [], "completed_utc": "fixture",
        }
        self._seal_manifest()

    def _seal_manifest(self):
        self.manifest.pop("sha256", None)
        self.manifest = seal(self.manifest)
        self.expected_hash = write_json(self.corpus / "manifest.json", self.manifest)["sha256"]

    def read(self):
        return read_fixture_corpus(self.corpus, self.registry, manifest_file_sha256=self.expected_hash)

    def _replace_shard(self, partition, transform):
        item = next(x for x in self.manifest["shards"] if x["partition"] == partition)
        path = self.corpus / item["path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        transform(rows)
        raw = "".join(canonical_json(row) + "\n" for row in rows).encode()
        path.write_bytes(raw)
        item.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                    sample_ids_sha256=sha256_json([row["task"]["id"] for row in rows]))
        self._seal_manifest()

    def test_trust_anchors_match_the_published_index(self):
        index = json.loads((ROOT / "docs/evidence/ik-dataset-20261022/index.json").read_text())
        self.assertEqual(index["manifest_sha256"], ACCEPTED_MANIFEST_SHA256)
        self.assertEqual(index["config_sha256"], ACCEPTED_CONFIG_SHA256)
        self.assertEqual(index["public_artifacts"]["manifest.json"]["sha256"], ACCEPTED_MANIFEST_FILE_SHA256)
        self.assertEqual(index["public_artifacts"]["normalization.json"]["sha256"],
                         ACCEPTED_NORMALIZATION_FILE_SHA256)
        normalizer = json.loads((ROOT / "docs/evidence/ik-dataset-20261022/normalization.json").read_text())
        self.assertEqual(normalizer["training_inputs_sha256"], ACCEPTED_TRAINING_INPUTS_SHA256)

    def test_fixture_read_preserves_labels_without_teacher_or_final_access(self):
        original_open = Path.open
        opened = []

        def guarded_open(path, *args, **kwargs):
            self.assertFalse(path.name.startswith("final-") or path.name in ("pending.jsonl", "attempts.jsonl"))
            opened.append(path.name)
            return original_open(path, *args, **kwargs)

        with patch("nca.ik.data.generate_sample", side_effect=AssertionError("teacher regeneration")), \
                patch("nca.ik.data.solve_numeric", side_effect=AssertionError("teacher regeneration")), \
                patch.object(Path, "open", guarded_open):
            corpus = self.read()
        self.assertEqual(corpus.training, self.training)
        self.assertEqual(corpus.tuning, self.tuning)
        self.assertEqual(corpus.normalization, self.normalization)
        self.assertFalse(corpus.accepted)
        self.assertFalse(corpus.identity["raw_final_rows_returned"])
        self.assertIn("normalization.json", opened)
        self.assertEqual(Counter(s.label.teacher.status for s in corpus.training),
                         Counter(s.label.teacher.status for s in self.training))

    def test_accepted_reader_refuses_self_authenticated_fixture(self):
        with self.assertRaises(ContractError):
            read_training_corpus(self.corpus, self.registry)

    def test_manifest_byte_tamper_is_rejected(self):
        path = self.corpus / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ContractError):
            self.read()

    def test_shard_byte_tamper_is_rejected(self):
        path = self.corpus / "training-00000.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"partition":"training"', b'"partition":"tuning"', 1))
        with self.assertRaises(ContractError):
            self.read()

    def test_rehashed_partition_swap_is_rejected_semantically(self):
        def swap(rows):
            rows[0]["task"]["partition"] = "tuning"
            rows[0]["task"]["state"]["partition"] = "tuning"
        self._replace_shard("training", swap)
        with self.assertRaises(ContractError):
            self.read()

    def test_rehashed_label_change_is_rejected(self):
        def corrupt(rows):
            rows[0]["label"]["teacher"]["status"] = "invalid_input"
        self._replace_shard("training", corrupt)
        with self.assertRaises(ContractError):
            self.read()

    def test_cross_platform_one_ulp_analytic_point_is_explicitly_scoped(self):
        # The original Linux label 7.1951845160020005 and Windows replay
        # 7.195184516002 are adjacent binary64 values, not equal values.
        self.assertEqual(math.nextafter(7.195184516002, math.inf), 7.1951845160020005)
        sample = next(s for s in self.training if s.label.analytic.points)
        original = sample.label.analytic
        saved = original.points[0][1]
        points = ((original.points[0][0], math.nextafter(saved, math.inf)),) + original.points[1:]
        modified = replace(sample, label=replace(
            sample.label, analytic=replace(original, points=points)))
        with self.assertRaises(ContractError):
            validate_sample(modified, self.config)
        # Only the pinned accepted archive reader may request this compatibility
        # branch after manifest, shard and original-registry verification.
        # A Boolean no longer grants a general one-ULP exception. Only an
        # enumerated original/Windows digest pair from the original corpus can
        # cross this boundary, after the reader authenticates its bytes.
        with self.assertRaises(ContractError):
            validate_sample(modified, self.config, _accepted_archive=True)
        further = ((original.points[0][0],
                    math.nextafter(math.nextafter(saved, math.inf), math.inf)),) + original.points[1:]
        with self.assertRaises(ContractError):
            validate_sample(replace(sample, label=replace(
                sample.label, analytic=replace(original, points=further))),
                self.config, _accepted_archive=True)
        changed_mask = replace(modified, label=replace(
            modified.label, pose_loss=not modified.label.pose_loss))
        with self.assertRaises(ContractError):
            validate_sample(changed_mask, self.config, _accepted_archive=True)

    def test_windows_replay_accepts_only_exact_original_pairs_and_semantics(self):
        cases = json.loads((ROOT / "tests/fixtures/ik_windows_replay_labels.json").read_text())
        with patch("nca.ik.data.platform.system", return_value="Windows"), \
                patch("nca.ik.data.platform.python_version", return_value="3.12.10"):
            for case in cases:
                retained = TeacherLabel.from_dict(case["retained"])
                replay = TeacherLabel.from_dict(case["windows_replay"])
                self.assertTrue(_authenticated_windows_label_replay(case["id"], replay, retained))
                # Any extra ULP, wrong residual, sample identity or semantic
                # decision is outside the exact inventory, even after auth.
                changed = replace(replay, endpoint_error_cells=math.nextafter(
                    replay.endpoint_error_cells, math.inf))
                self.assertFalse(_authenticated_windows_label_replay(case["id"], changed, retained))
                self.assertFalse(_authenticated_windows_label_replay("0" * 64, replay, retained))
                for key in ("pose_loss", "endpoint_loss", "verified"):
                    self.assertFalse(_authenticated_windows_label_replay(case["id"],
                                     replace(replay, **{key: not getattr(replay, key)}), retained))
                changed = replace(retained, teacher=replace(retained.teacher, iterations=0))
                self.assertFalse(_authenticated_windows_label_replay(case["id"], replay, changed))
            with patch("nca.ik.data.platform.system", return_value="Linux"):
                self.assertFalse(_authenticated_windows_label_replay(case["id"], replay, retained))

    def test_windows_inventory_tamper_refuses(self):
        _windows_replay_inventory.cache_clear()
        try:
            with patch.object(Path, "read_bytes", return_value=b"{}"):
                with self.assertRaisesRegex(ContractError, "inventory bytes changed"):
                    _windows_replay_inventory()
        finally:
            _windows_replay_inventory.cache_clear()

    def test_rehashed_one_ulp_fixture_label_still_refused(self):
        def edit(rows):
            value = rows[0]["label"]["analytic"]["points"][0][1]
            rows[0]["label"]["analytic"]["points"][0][1] = math.nextafter(value, math.inf)
        self._replace_shard("training", edit)
        with self.assertRaises(ContractError):
            self.read()

    def test_rehashed_row_duplicate_is_rejected(self):
        self._replace_shard("training", lambda rows: rows.__setitem__(1, rows[0]))
        with self.assertRaises(ContractError):
            self.read()

    def test_missing_inventory_shard_is_rejected(self):
        self.manifest["shards"].pop(0)
        self._seal_manifest()
        with self.assertRaises(ContractError):
            self.read()

    def test_rehashed_normalization_change_is_rejected_by_training_provenance(self):
        value = self.normalization.to_dict()
        value["mean"][0] += .1
        replacement = write_json(self.corpus / "normalization.json", value)
        self.manifest["artifacts"] = [replacement if x["path"] == "normalization.json" else x
                                      for x in self.manifest["artifacts"]]
        self._seal_manifest()
        with self.assertRaises(ContractError):
            self.read()

    def test_registry_substitution_is_rejected(self):
        write_json(self.registry / "registry.json",
                   {"schema": "nca-ik-dataset-registry/v1", "id": "2" * 32})
        with self.assertRaises(ContractError):
            self.read()

    def test_final_opening_refuses_further_training(self):
        (self.registry / (self.suite_id + ".final-opened.json")).write_text("{}")
        with self.assertRaises(ContractError):
            self.read()

    def test_path_traversal_is_rejected_before_file_access(self):
        self.manifest["shards"][0]["path"] = "../training-00000.jsonl"
        self._seal_manifest()
        with self.assertRaises(ContractError):
            self.read()

    def test_fixture_factory_never_authenticates_a_measured_corpus(self):
        corpus = make_fixture_corpus(self.config, self.training, self.tuning)
        self.assertFalse(corpus.accepted)
        self.assertEqual(corpus.identity["evidence_kind"], "development_fixture")
        counterfeit = replace(corpus, _authentication=True, manifest_sha256=ACCEPTED_MANIFEST_SHA256)
        self.assertFalse(counterfeit.accepted)
        with self.assertRaises(ContractError):
            make_fixture_corpus(DatasetConfig(), self.training)
        with self.assertRaises(ContractError):
            make_fixture_corpus(self.config, self.tuning)


class IKSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the same tiny generated fixture corpus across test classes.
        if not hasattr(IKCorpusTests, "training"):
            IKCorpusTests.setUpClass()
        cls.samples = IKCorpusTests.training

    def make(self, **kwargs):
        return PairPreservingSampler(self.samples, 41011, **({"batch_size": 16, "epochs": 3} | kwargs))

    def test_every_row_and_failed_label_is_retained_once_per_epoch(self):
        sampler = self.make()
        observed, counts = [], []
        for batch_count in sampler.batches_per_epoch:
            epoch = [sample for _ in range(batch_count) for sample in sampler.next_batch()]
            observed.extend(epoch)
            counts.append(Counter(s.task.id for s in epoch))
            self.assertEqual(counts[-1], Counter(s.task.id for s in self.samples))
            self.assertEqual([s.task.stage for s in epoch], sorted(s.task.stage for s in epoch))
        self.assertTrue(sampler.exhausted)
        self.assertEqual(sampler.total_presentations, len(observed))
        self.assertEqual(sampler.sample_presentations, sampler.total_presentations)
        with self.assertRaises(StopIteration):
            sampler.next_batch()

    def test_chronological_complete_trajectory_stays_in_one_batch(self):
        sampler = self.make(epochs=1)
        groups = {}
        for i in range(sampler.planned_batches):
            batch = sampler.next_batch()
            ids = [s.task.id for s in batch]
            self.assertEqual(len(ids), len(set(ids)))
            for sample in batch:
                if sample.task.state.kind == "trajectory":
                    groups.setdefault(sample.task.trajectory_id, []).append((i, sample.task.state.step))
        self.assertTrue(groups)
        for entries in groups.values():
            self.assertEqual(len({batch for batch, _ in entries}), 1)
            self.assertEqual([step for _, step in entries], [0, 1, 2])

    def test_exact_state_restoration_replays_all_future_batches(self):
        sampler = self.make()
        for _ in range(5):
            sampler.next_batch()
        saved = json.loads(json.dumps(sampler.state_dict()))
        restored = self.make()
        restored.load_state_dict(saved)
        self.assertEqual(restored.state_dict(), saved)
        while not sampler.exhausted:
            self.assertEqual(sampler.next_batch(), restored.next_batch())
        self.assertTrue(restored.exhausted)

    def test_changed_seed_order_config_or_counters_cannot_resume(self):
        sampler = self.make()
        sampler.next_batch()
        saved = sampler.state_dict()
        changes = ({"seed": 42013}, {"batch_size": 32}, {"epochs": 4},
                   {"sample_presentations": saved["sample_presentations"] - 1},
                   {"cursor": True}, {"cursor": sampler.planned_batches + 1},
                   {"unknown": 0}, {"schedule_sha256": "0" * 64})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ContractError):
                self.make().load_state_dict(saved | change)
        reordered = PairPreservingSampler(tuple(reversed(self.samples)), 41011, batch_size=16, epochs=3)
        with self.assertRaises(ContractError):
            reordered.load_state_dict(saved)

    def test_training_split_duplicates_and_partial_paths_are_rejected(self):
        with self.assertRaises(ContractError):
            PairPreservingSampler(IKCorpusTests.tuning, 41011)
        with self.assertRaises(ContractError):
            PairPreservingSampler(self.samples + self.samples[:1], 41011)
        with self.assertRaises(ContractError):
            self.make(batch_size=2)
        incomplete = tuple(s for s in self.samples
                           if not (s.task.state.kind == "trajectory" and s.task.state.step == 1))
        with self.assertRaises(ContractError):
            PairPreservingSampler(incomplete, 41011)
        for kwargs in ({"epochs": True}, {"batch_size": 0}, {"epochs": 1001}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ContractError):
                self.make(**kwargs)

    def test_deterministic_seeded_plan_does_not_import_torch(self):
        program = """
import sys
sys.path.insert(0, sys.argv[1])
import nca.train.ik_corpus
assert "torch" not in sys.modules
assert "numpy" not in sys.modules
"""
        subprocess.run([sys.executable, "-S", "-c", program, str(ROOT / "src")],
                       check=True, capture_output=True)
        first, same = self.make(), self.make()
        other = PairPreservingSampler(self.samples, 42013, batch_size=16, epochs=3)
        self.assertEqual(first.schedule_sha256, same.schedule_sha256)
        self.assertNotEqual(first.schedule_sha256, other.schedule_sha256)


if __name__ == "__main__":
    unittest.main()
