import json
import random
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from nca.contracts import ContractError, sha256_json
from nca.randomness import ScenarioSplit, derive_seed, random_stream
from nca.runs import ArtifactRef, RunBudget, RunManifest, RunStore, sha256_file


def manifest(**kwargs):
    value = RunManifest(
        run_id="fixture-run-001", experiment="contract-fixture",
        learner_revision="95d878" + "0" * 34,
        host_revision="f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497",
        profile_sha256="a" * 64, resolved_config={"cadence": {"base_hz": 60, "repeat": 6}},
        environment={"runtime": "test", "optional_gpu": "not_imported"},
        seeds=(1, 2, 3), scenario_split=ScenarioSplit(("train-flat",), ("tune-slope",), ("eval-gap",)),
        status="completed", runtime_seconds=.01, created_utc="2026-10-19T00:00:00Z",
        metrics={"physical_performance": None}, decisions=3)
    return replace(value, **kwargs)


class RunTests(unittest.TestCase):
    def test_hashes_are_order_stable_and_resolved_config_is_deeply_immutable(self):
        source = {"nested": {"values": [1, 2]}}
        record = manifest(resolved_config=source, config_sha256="")
        before = record.sha256
        source["nested"]["values"][0] = 99
        self.assertEqual(record.sha256, before)
        self.assertEqual(record.resolved_config["nested"]["values"], (1, 2))
        with self.assertRaises(TypeError):
            record.resolved_config["nested"]["new"] = 3
        self.assertEqual(sha256_json({"b": 2, "a": 1}), sha256_json({"a": 1, "b": 2}))
        with self.assertRaises(ContractError):
            replace(record, config_sha256="0" * 64)

    def test_complete_roundtrip_and_null_unmeasured_metric(self):
        record = manifest(artifacts=(ArtifactRef("fixtures/input.json", "b" * 64, 4, "trace"),))
        self.assertEqual(RunManifest.from_json(record.to_json()), record)
        self.assertIsNone(record.metrics["physical_performance"])
        with tempfile.TemporaryDirectory() as root:
            store = RunStore(root)
            path = store.write(record)
            self.assertTrue(path.exists())
            self.assertEqual(store.read(record.run_id), record)
            self.assertEqual(store.verify(), (record.run_id,))
            vector = Path(root) / "hash-vector.bin"
            vector.write_bytes(b"abc")
            self.assertEqual(sha256_file(vector),
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_atomic_duplicate_publish_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            store = RunStore(root)
            records = [manifest(metrics={"writer": i}) for i in range(12)]

            def publish(record):
                try:
                    store.write(record)
                    return True
                except ContractError as exc:
                    self.assertEqual(exc.code, "immutable_run")
                    return False

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(publish, records))
            self.assertEqual(sum(results), 1)
            winner = records[results.index(True)]
            self.assertEqual(store.read(winner.run_id), winner)
            self.assertEqual(len(tuple(Path(root).iterdir())), 1)

    def test_distinct_concurrent_runs_are_independent_and_indexed(self):
        with tempfile.TemporaryDirectory() as root:
            store = RunStore(root)
            records = [manifest(run_id=f"run-{i:02}") for i in range(12)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                paths = list(pool.map(store.write, records))
            self.assertEqual(len(set(paths)), 12)
            self.assertEqual(tuple(value.run_id for value in store.list()),
                             tuple(value.run_id for value in records))

    def test_tampered_content_and_substituted_identity_refused(self):
        with tempfile.TemporaryDirectory() as root:
            store = RunStore(root)
            path = store.write(manifest())
            data = json.loads(path.read_text())
            data["manifest"]["decisions"] = 99
            path.write_text(json.dumps(data))
            with self.assertRaises(ContractError) as error:
                store.read(manifest().run_id)
            self.assertEqual(error.exception.code, "hash_mismatch")
            path.write_text('{"manifest":{},"manifest":{}}')
            with self.assertRaises(ContractError) as error:
                store.read(manifest().run_id)
            self.assertEqual(error.exception.code, "duplicate_field")

    def test_fresh_episode_resume_keeps_parent_record(self):
        with tempfile.TemporaryDirectory() as root:
            store = RunStore(root)
            parent = manifest(status="capped", failures=("decision_ceiling",), decisions=2000)
            child = manifest(run_id="fixture-run-002", parent_run_id=parent.run_id,
                             continuation="fresh_episode")
            store.write(parent)
            store.write(child)
            self.assertEqual(store.read(parent.run_id).status, "capped")
            self.assertEqual(store.read(child.run_id).parent_run_id, parent.run_id)
        with self.assertRaises(ContractError):
            manifest(parent_run_id="fixture-run-001", continuation="fresh_episode")
        with self.assertRaises(ContractError):
            manifest(continuation="fresh_episode")

    def test_reasons_identity_budget_and_portable_paths(self):
        for status in ("failed", "capped", "canceled", "inconclusive"):
            with self.subTest(status=status), self.assertRaises(ContractError):
                manifest(status=status)
            self.assertEqual(manifest(status=status, failures=("recorded_reason",)).status, status)
        for kwargs in ({"run_id": "../escape"}, {"run_id": "run:ads"}, {"run_id": "CON"},
                       {"learner_revision": "main"}, {"runtime_seconds": None},
                       {"seeds": (True,)}, {"seeds": (1, 1)},
                       {"created_utc": "2026-10-19T00:00:00"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ContractError):
                manifest(**kwargs)
        for uri in ("/private/checkpoint", "C:\\private\\checkpoint", "file:///private"):
            with self.subTest(uri=uri), self.assertRaises(ContractError):
                ArtifactRef(uri, "c" * 64, 0, "checkpoint")
        with self.assertRaises(ContractError):
            RunBudget(0, 120)
        self.assertTrue(RunBudget(100, 120).exhausted(100, 1))
        self.assertTrue(RunBudget(100, 120).exhausted(1, 120))
        self.assertFalse(RunBudget(100, 120).exhausted(99, 119))


class RandomnessTests(unittest.TestCase):
    def test_named_seed_streams_are_stable_order_independent_and_local(self):
        before = random.getstate()
        a = derive_seed(42, "environment", 0)
        derive_seed(42, "policy", 19)
        self.assertEqual(a, derive_seed(42, "environment", 0))
        self.assertNotEqual(a, derive_seed(42, "environment", 1))
        self.assertEqual(random_stream(42, "environment").random(),
                         random_stream(42, "environment").random())
        self.assertEqual(random.getstate(), before)
        for seed in (-1, 2**64, True):
            with self.assertRaises(ContractError):
                derive_seed(seed, "environment")

    def test_family_splits_reject_leakage_and_unknown_families(self):
        split = ScenarioSplit(("rough",), ("slope",), ("gap",))
        self.assertEqual(split.partition("gap"), "evaluation")
        self.assertEqual(split.sha256, ScenarioSplit.from_json(split.to_json()).sha256)
        with self.assertRaises(ContractError):
            ScenarioSplit(("rough",), ("rough",), ("gap",))
        with self.assertRaises(ContractError):
            split.partition("unregistered")
        with self.assertRaises(ContractError):
            ScenarioSplit(("rough",), (), ())


if __name__ == "__main__":
    unittest.main()
