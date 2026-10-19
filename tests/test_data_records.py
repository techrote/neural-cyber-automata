"""Synthetic data-boundary controls; actual owner replay is tested separately."""

import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from nca.contracts import ContractError, sha256_json
from nca.data import (ActionSource, CaptureEvent, Dataset, EpisodeRecord, PolicySample,
                      TickRecord, boundary_changes, policy_view, read_episode,
                      rewrite_packet_identity, write_episode)
from nca.randomness import ScenarioSplit
from nca.runs import RunStore


def view(tick=0, **kwargs):
    value = {"version": 2, "representation": "articulated-rapier", "run_id": "synthetic",
             "epoch": 1, "actor_id": 0, "tick": tick, "observation_id": f"synthetic:1:0:0:{tick}",
             "model": "jev", "profile_version": 1, "control_generation": 0,
             "position": [float(tick), 0.0], "velocity": [1.0, 0.0], "rotation": 0.0,
             "angular_velocity": 0.0, "grounded": True, "energy": None,
             "parts": [{"id": "body", "index": 0, "detached": False}]}
    value.update(kwargs)
    return value


def packet():
    return {"version": 2, "kind": "physical_wire_action", "run_id": "synthetic", "epoch": 1,
            "actor_id": 0, "sequence": 0, "observed_tick": 0, "observation_id": view()["observation_id"],
            "lease_ticks": 12, "action": {"move": [.5, 0]}}


def event(**kwargs):
    source = ActionSource("learner", "learned", "test-policy", "a" * 64)
    sample = PolicySample("wire_request", packet()["action"], "a" * 64, -.3, "test-distribution")
    tick = TickRecord(0, 1, 1, 10, 11, (packet(),), ({"accepted": True, "reason": ""},),
                      packet(), "learner", {"move": [.5, 0]}, {"linear_impulse": [.1, 0]},
                      {"available": False, "reason": "not exposed"}, {"velocity": [1, 0]}, view(1), 0)
    base = CaptureEvent("step", 0, 0, 0, 1, 0, 20, source, view(), policy_view(view()),
                        requested_packet=packet(), source_observation=view(), sample=sample,
                        ticks=(tick,), next_observation=view(1), reward=0)
    return replace(base, **kwargs)


def episode(**kwargs):
    files = {"test-fixture.py": "1" * 64}
    value = EpisodeRecord("synthetic-episode", {"lane": "act004", "test_fixture": True}, "b" * 40,
                          sha256_json(files), {"seed": 1}, {"id": "synthetic"}, view(), (event(),),
                          "completed", "fixture_complete", "heldout", "evaluation", source_files=files)
    return replace(value, **kwargs)


class DataRecordTests(unittest.TestCase):
    def test_privileged_fields_never_enter_policy_features(self):
        issued = view(evaluator={"hidden_target": [99, 99]}, reward=1e9, success=True,
                      actors=[{"secret": 1}], cell_patch={"materials": [1, 2]})
        allowed = policy_view(issued)
        self.assertNotIn("evaluator", allowed)
        self.assertNotIn("reward", allowed)
        self.assertNotIn("success", allowed)
        self.assertNotIn("actors", allowed)
        self.assertNotIn("cell_patch", allowed)
        self.assertIsNone(allowed["energy"])
        with self.assertRaises(ContractError):
            event(policy_observation={**policy_view(view()), "hidden_target": [1, 2]})

    def test_requested_consumed_applied_passive_remain_different_values(self):
        value = event().ticks[0]
        self.assertEqual(value.consumed_request["action"]["move"], (.5, 0))
        self.assertEqual(value.applied["linear_impulse"], (.1, 0))
        self.assertFalse(value.passive["available"])
        self.assertEqual(value.actual_motion["velocity"], (1, 0))
        with self.assertRaises(ContractError):
            replace(value, passive={"available": False})

    def test_ppo_accepts_only_unchanged_attributed_likelihood_samples(self):
        self.assertEqual(event().require_on_policy().log_probability, -.3)
        variants = [event(intervention="human_correction"), event(intervention="model_override"),
                    event(replaced=True), event(sample=None), event(discontinuities=("mount",)),
                    event(source=ActionSource("learner", "scripted", "test-policy", "a" * 64))]
        for item in variants:
            with self.subTest(event=item.intervention), self.assertRaises(ContractError):
                item.require_on_policy()
        corrected = event(intervention="human_correction")
        self.assertTrue(corrected.imitation_eligible)
        residual = event(sample=PolicySample("latent_residual", {"latent": [2.4, -.1]}, "a" * 64,
                                             -.8, "diagonal-normal-v1"))
        self.assertEqual(residual.require_on_policy().values["latent"], (2.4, -.1))
        self.assertNotEqual(residual.sample.values, residual.requested_packet["action"])

    def test_duplicate_denied_request_cannot_inherit_previous_consumption_as_ppo(self):
        original = event()
        tick = replace(original.ticks[0], admission=({"accepted": False, "reason": "sequence"},))
        duplicated = replace(original, ticks=(tick,))
        self.assertFalse(duplicated.imitation_eligible)
        with self.assertRaises(ContractError):
            duplicated.require_on_policy()

    def test_mixed_source_or_missing_tick_never_becomes_on_policy(self):
        original = event()
        mixed = replace(original, ticks=(replace(original.ticks[0], source_id="old-source"),))
        self.assertFalse(mixed.on_policy_eligible)
        with self.assertRaises(ContractError):
            replace(original, tick_after=2, next_observation=view(2))
        invalid = replace(original, valid=False, trace_complete=False)
        self.assertFalse(invalid.imitation_eligible)

    def test_reset_mount_profile_and_takeover_boundary_detection(self):
        before = view()
        self.assertEqual(boundary_changes(before, view(1)), ())
        for field, value, label in (("run_id", "new", "reset"), ("profile_version", 2, "profile"),
                                    ("control_generation", 1, "control_generation"),
                                    ("mount_generation", 1, "mount")):
            with self.subTest(field=field):
                self.assertIn(label, boundary_changes(before, view(**{field: value})))
        self.assertIn("part_topology", boundary_changes(before, view(parts=[{"id": "body", "index": 0, "detached": True}])))

    def test_identity_mapping_preserves_age_sequence_and_lease(self):
        original = packet()
        source = view(run_id="replay", epoch=2, observation_id="replay:2:0:0:0")
        rewritten, mapping = rewrite_packet_identity(original, {original["observation_id"]: source}, view(5))
        self.assertEqual(rewritten["observed_tick"], 0)
        self.assertEqual(mapping.consumed_tick, 5)
        self.assertEqual(rewritten["lease_ticks"], original["lease_ticks"])
        self.assertEqual(rewritten["sequence"], original["sequence"])
        self.assertEqual(rewritten["action"], original["action"])
        self.assertEqual(original["run_id"], "synthetic")
        with self.assertRaises(ContractError):
            rewrite_packet_identity(original, {original["observation_id"]: view(5)}, view(5))
        with self.assertRaises(ContractError):
            rewrite_packet_identity(original, {}, view())

    def test_unknown_or_incorrect_source_view_cannot_become_policy_data(self):
        self.assertFalse(event(source_observation=None).on_policy_eligible)
        with self.assertRaisesRegex(ContractError, "request_observation"):
            wrong = view(observation_id="not-the-request")
            event(source_observation=wrong, policy_observation=policy_view(wrong))

    def test_dataset_split_rejects_family_leakage(self):
        data = Dataset("synthetic-data", (episode(),), ScenarioSplit(("train",), ("tune",), ("heldout",)))
        self.assertEqual(len(list(data.imitation_samples())), 0)
        rows = list(data.imitation_samples("evaluation"))
        self.assertEqual(len(rows), 1)
        self.assertNotIn("issued_observation", rows[0])
        self.assertTrue(rows[0]["reset"])
        with self.assertRaises(ContractError):
            Dataset("leaky", (episode(split="training"),), data.splits)

    def test_published_fixture_manifest_and_failure_receipts(self):
        root = Path(__file__).resolve().parents[1]
        sample = read_episode(root / "tests/fixtures/data/synthetic-episode.json")
        self.assertTrue(sample.host_identity["test_fixture"])
        self.assertTrue(sample.evaluation["test_only"])
        row = list(sample.imitation_samples())[0]
        self.assertNotIn("evaluation", row)
        self.assertNotIn("privileged_check_value", row["observation"])
        manifest = json.loads((root / "configs/datasets/t104-capture-replay.json").read_text())
        self.assertEqual(manifest["total_consumed_ticks"], sum(item["consumed_ticks"] for item in manifest["episodes"]))
        splits = ScenarioSplit.from_dict(manifest["scenario_split"])
        store = RunStore(root / "configs/datasets/runs")
        for item in manifest["episodes"]:
            self.assertEqual(splits.partition(item["scenario_family"]), item["split"])
            self.assertEqual(item["on_policy_samples"], 0)
            run = store.read(item["id"])
            self.assertEqual(run.metrics["consumed_ticks"], item["consumed_ticks"])
            self.assertEqual(run.artifacts[0].content_sha256, item["episode"]["content_sha256"])
        failure = store.read("t104-first-capture")
        self.assertEqual(failure.status, "failed")
        self.assertEqual(failure.failures, ("TypeError",))

    def test_installed_package_source_identity_requires_matching_checkout(self):
        root = Path(__file__).resolve().parents[1]
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        with tempfile.TemporaryDirectory() as temporary:
            installed = Path(temporary) / "site-packages"
            shutil.copytree(root / "src/nca", installed / "nca", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            outside = Path(temporary) / "outside"
            outside.mkdir()
            environment = dict(os.environ, PYTHONPATH=str(installed))
            environment.pop("NCA_SOURCE_ROOT", None)
            code = '''
import json, sys
from pathlib import Path
from nca.data import capture
from nca.contracts import ContractError
assert Path(capture.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()), "expected installed-layout import"
try:
    revision, files, aggregate = capture.source_identity()
except ContractError as error:
    print(json.dumps({"error": error.code}))
else:
    print(json.dumps({"revision": revision, "aggregate": aggregate, "capture_hash": files["src/nca/data/capture.py"]}))
'''

            def run(cwd):
                value = subprocess.run([sys.executable, "-c", code, str(installed)], cwd=cwd,
                                       env=environment, capture_output=True, text=True)
                self.assertEqual(value.returncode, 0, value.stderr)
                return json.loads(value.stdout)

            matched = run(root)
            self.assertEqual(matched["revision"], revision)
            self.assertEqual(run(outside)["error"], "source_checkout")
            environment["NCA_SOURCE_ROOT"] = str(root)
            self.assertEqual(run(outside), matched)
            bridge = installed / "nca/adapters/host_bridge.gd"
            bridge_contents = bridge.read_bytes()
            bridge.write_bytes(bridge_contents + b"\n# Installed bridge mismatch fixture.\n")
            self.assertEqual(run(outside)["error"], "source_mismatch")
            bridge.write_bytes(bridge_contents)
            with (installed / "nca/contracts.py").open("a") as stream:
                stream.write("\n# Installed-source mismatch fixture.\n")
            self.assertEqual(run(outside)["error"], "source_mismatch")

    def test_immutable_compressed_roundtrip_and_tamper_detection(self):
        data = episode()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "tiny.json.gz"
            write_episode(path, data)
            self.assertEqual(read_episode(path), data)
            with self.assertRaises(ContractError):
                write_episode(path, data)
            contents = json.loads(gzip.decompress(path.read_bytes()))
            contents["episode"]["reason"] = "edited"
            path.write_bytes(gzip.compress(json.dumps(contents).encode()))
            with self.assertRaises(ContractError):
                read_episode(path)


if __name__ == "__main__":
    unittest.main()
