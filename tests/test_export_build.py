"""Conversion must verify referenced episode bytes, not only summary claims."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    from nca.contracts import ContractError, sha256_json
    from nca.export.build import convert, immutable_json, verify_artifact_files

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_NUMPY, "optional NumPy CPU export dependency unavailable")
class ExportArtifactTests(unittest.TestCase):
    def test_conversion_checks_manifest_contents_as_well_as_claimed_hash_fields(self):
        screen = json.loads((ROOT / "configs/export/cpu-bc-v1.json").read_text())
        for mutation in (None, "config", "revision", "sources", "dataset"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                training = json.loads((ROOT / "configs/first-skill/jolt-screen-v1.json").read_text())
                source_files = {"synthetic-fixture.py": "a"*64}
                dataset = {"id": "synthetic-only-dataset"}
                dataset["sha256"] = sha256_json(dataset)
                manifest = {"config": training, "learner_revision": screen["selection"]["learner_revision"],
                            "source_files": source_files, "learner_sources_sha256": sha256_json(source_files)}
                metadata = {"config_sha256": screen["selection"]["learner_config_sha256"],
                            "learner_sources_sha256": manifest["learner_sources_sha256"],
                            "dataset_sha256": dataset["sha256"], "phase": "final", "kind": "bc",
                            "training_seed": 1103, "epochs": 40, "training_privilege": "none"}
                if mutation == "config":
                    manifest["config"]["id"] = "changed-experiment"
                elif mutation == "revision":
                    manifest["learner_revision"] = "b"*40
                elif mutation == "sources":
                    manifest["source_files"]["synthetic-fixture.py"] = "b"*64
                elif mutation == "dataset":
                    metadata["dataset_sha256"] = "b"*64
                # Rehash the envelopes to establish that content checks, not
                # merely a stale outer self-hash, catch these substitutions.
                manifest["sha256"] = sha256_json(manifest)
                metadata["manifest_sha256"] = manifest["sha256"]
                summary = {"status": "completed", "evaluation": {"complete": True},
                           "manifest_sha256": manifest["sha256"], "config_sha256": metadata["config_sha256"],
                           "learner_sources_sha256": metadata["learner_sources_sha256"],
                           "results": [{"split": "evaluation", "policy_id": "bc-seed1103-final",
                                        "scenario_family": family["id"], "policy_sha256": "c"*64,
                                        "artifact": f"episodes/{family['id']}.json.gz", "artifact_sha256": "d"*64}
                                       for family in training["evaluation_families"]]}
                summary["sha256"] = sha256_json(summary)
                records = {"manifest.json": manifest, "summary.json": summary, "dataset.json": dataset,
                           "environment.json": {}, "exports/bc-seed1103-final.weights.json": {
                               "metadata": metadata, "policy_sha256": "c"*64}}
                for name, value in records.items():
                    immutable_json(root / name, value)
                reason = "artifact_unavailable" if mutation is None else "evaluation_gate"
                with self.assertRaisesRegex(ContractError, reason):
                    convert(root, {}, accepted_t105_commit="e"*40, output=root / "bundle.json",
                            screen_config=ROOT / "configs/export/cpu-bc-v1.json")
                self.assertFalse((root / "bundle.json").exists())

    def test_missing_or_changed_episode_cannot_inherit_unchanged_summary_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episodes/evaluation.json.gz"
            episode.parent.mkdir()
            episode.write_bytes(b"original episode bytes")
            files = {"episodes/evaluation.json.gz": hashlib.sha256(episode.read_bytes()).hexdigest()}
            verify_artifact_files(root, files)
            episode.write_bytes(b"changed episode bytes")
            with self.assertRaisesRegex(ContractError, "artifact_hash"):
                verify_artifact_files(root, files)
            episode.unlink()
            with self.assertRaisesRegex(ContractError, "artifact_unavailable"):
                verify_artifact_files(root, files)

    def test_external_or_ambiguous_paths_are_refused_even_with_matching_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "collection"
            root.mkdir()
            external = parent / "outside.json"
            external.write_bytes(b"external episode")
            expected = hashlib.sha256(external.read_bytes()).hexdigest()
            for name in ("../outside.json", str(external), "./outside.json", "sub//outside.json", "sub\\outside.json"):
                with self.subTest(name=name), self.assertRaisesRegex(ContractError, "artifact_path"):
                    verify_artifact_files(root, {name: expected})
            link = root / "linked.json"
            try:
                link.symlink_to(external)
            except OSError:
                return  # Windows hosts may lack local symlink creation rights.
            with self.assertRaisesRegex(ContractError, "artifact_path"):
                verify_artifact_files(root, {link.name: expected})


if __name__ == "__main__":
    unittest.main()
