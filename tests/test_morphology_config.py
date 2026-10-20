"""Source admission, prospective placement and budget tests; no host work."""

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from nca.contracts import ContractError, sha256_json
from nca.train.morphology_config import (MorphologyBudget, MorphologyBudgetExceeded, PHASE_CAPS,
    PREREQUISITE_SELECTION_SHA256, file_receipt, read_json, read_prerequisite_selection,
    scenario, verify_direction_sources, verify_files, verify_primary_direction_identity)

ROOT = Path(__file__).resolve().parents[1]


class MorphologyAdmissionTests(unittest.TestCase):
    def test_import_is_closed_before_torch_and_host(self):
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        subprocess.run([sys.executable, "-c", "import sys; import nca.train.morphology_config; "
                        "assert 'torch' not in sys.modules; assert 'nca.train.morphology_capture' not in sys.modules"],
                       env=environment, check=True, capture_output=True)

    def test_designated_primary_cannot_be_replaced_by_another_valid_execution(self):
        selection = read_prerequisite_selection(ROOT)
        self.assertEqual(sha256_json(selection), PREREQUISITE_SELECTION_SHA256)
        primary = selection["primary"]
        summary = {key: primary[key] for key in ("learner_revision", "learner_sources_sha256", "manifest_sha256")}
        summary.update(sha256=primary["summary_content_sha256"], config_sha256=selection["followup_config_sha256"])
        receipt = {"file": "summary.json", "sha256": primary["summary_file_sha256"], "bytes": primary["summary_bytes"]}
        self.assertEqual(verify_primary_direction_identity(ROOT, summary, receipt), selection)
        for key in summary:
            changed = dict(summary, **{key: "a" * len(summary[key])})
            with self.subTest(field=key), self.assertRaises(ContractError):
                verify_primary_direction_identity(ROOT, changed, receipt)
        for changed in ({**receipt, "sha256": "a" * 64}, {**receipt, "bytes": receipt["bytes"] + 1}):
            with self.assertRaises(ContractError):
                verify_primary_direction_identity(ROOT, summary, changed)

    def test_rehashed_selection_does_not_reopen_the_prerequisite_choice(self):
        selection = deepcopy(read_prerequisite_selection(ROOT))
        selection["primary"]["learner_revision"] = "f0c4c1aa4ad30e972ebd406ca8eba78f96ef99ed"
        with patch("nca.train.morphology_config.read_json", return_value=selection):
            with self.assertRaises(ContractError) as caught:
                read_prerequisite_selection(ROOT)
        self.assertEqual(caught.exception.code, "morphology_prerequisite_selection")

    def test_original_source_closure_cannot_be_rehashed_after_omission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = ["src/nca/__init__.py", "src/nca/executed.py", "src/nca/bridge.gd", "pyproject.toml",
                     "tools/capture_replay.py", "tools/direction_frame.py",
                     "configs/first-skill/jolt-direction-frame-v1.json", "configs/first-skill/jolt-screen-v1.json",
                     "configs/morphology/entry-gate-v1.json", "docs/evidence/t106-prerequisite-result.json"]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((name + "\n").encode())
            for command in (["git", "init", "-q"], ["git", "config", "core.autocrlf", "false"],
                            ["git", "add", "."], ["git", "-c", "user.name=Source fixture", "-c",
                            "user.email=source-fixture@example.invalid", "commit", "-qm", "Synthetic source fixture"]):
                subprocess.run(command, cwd=root, check=True, capture_output=True)
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
            sources = {name: files[name] for name in names[:5]}
            manifest = {"learner_revision": revision, "source_files": sources,
                        "learner_sources_sha256": sha256_json(sources)}
            inputs = {"revision": revision, "files": files, "sha256": sha256_json(files)}
            verify_direction_sources(root, manifest, inputs)
            # New continuation files do not retroactively expand the old scope.
            (root / "src/nca/later.py").write_bytes(b"later source\n")
            verify_direction_sources(root, manifest, inputs)
            for mutation in ("omitted_module", "omitted_declaration", "aggregate", "blob"):
                changed_manifest, changed_inputs = deepcopy(manifest), deepcopy(inputs)
                if mutation == "omitted_module":
                    del changed_manifest["source_files"]["src/nca/executed.py"]
                    del changed_inputs["files"]["src/nca/executed.py"]
                    changed_manifest["learner_sources_sha256"] = sha256_json(changed_manifest["source_files"])
                elif mutation == "omitted_declaration":
                    del changed_inputs["files"]["configs/first-skill/jolt-direction-frame-v1.json"]
                elif mutation == "aggregate":
                    changed_manifest["learner_sources_sha256"] = "a" * 64
                else:
                    changed_inputs["files"]["tools/direction_frame.py"] = "a" * 64
                changed_inputs["sha256"] = sha256_json(changed_inputs["files"])
                with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                    verify_direction_sources(root, changed_manifest, changed_inputs)

    def test_portable_file_receipts_reject_changed_bytes_aliases_and_boolean_sizes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "receipt.json"
            path.write_bytes(b"x")
            receipt = file_receipt(path, root)
            self.assertEqual(verify_files(root, [receipt]), {"receipt.json"})
            for changed in ({**receipt, "file": "./receipt.json"}, {**receipt, "bytes": True},
                            {**receipt, "sha256": "a" * 64}):
                with self.assertRaises(ContractError):
                    verify_files(root, [changed])
            with self.assertRaises(ContractError):
                verify_files(root, [receipt, receipt])
            path.write_bytes(b"y")
            with self.assertRaises(ContractError):
                verify_files(root, [receipt])

    def test_actor_origin_and_root_goal_use_distinct_authored_offsets(self):
        profiles = read_json(ROOT / "configs/morphology/profiles-v1.json")
        row = scenario(profiles, "monowheel", "floor-right-30", 81100001)
        self.assertAlmostEqual(row["spawn"][1], 342.9289321899414)
        self.assertAlmostEqual(row["target"][1], 344.428924560547)
        self.assertNotEqual(row["spawn"][1], row["target"][1])
        drop = scenario(profiles, "monowheel", "drop16-left-54", 81100002)
        self.assertEqual(drop["target"][1], row["target"][1])
        self.assertEqual(drop["spawn"][1], row["spawn"][1] - 16)

    def test_phase_reservation_cannot_reset_or_exceed_original_allocation(self):
        self.assertEqual(sum(PHASE_CAPS.values()), 12864)
        budget = MorphologyBudget()
        budget.phases["qualification"] = 159
        budget.requests = 159
        budget.consume("qualification")
        with self.assertRaises(MorphologyBudgetExceeded):
            budget.consume("qualification")
        self.assertEqual(budget.requests, 160)

    def test_candidate_wall_is_cumulative_and_global_wall_never_resets(self):
        now = [0.]
        with patch("nca.train.morphology_config.time.monotonic", side_effect=lambda: now[0]):
            budget = MorphologyBudget()
            with budget.candidate("shared_graph", 6113):
                now[0] = 500.
            now[0] = 700.
            with budget.candidate("shared_graph", 6113):
                now[0] = 1100.
                with self.assertRaises(MorphologyBudgetExceeded):
                    budget.check()
            self.assertEqual(budget.candidate_seconds["shared_graph-seed6113"], 900.)
            now[0] = 1800.
            with self.assertRaises(MorphologyBudgetExceeded):
                budget.check()


if __name__ == "__main__":
    unittest.main()
