"""Prospective declaration conformance without optional libraries or execution."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import nca
from nca.contracts import canonical_json, sha256_json
from nca.ik.declaration import (load_declaration, metadata_for, prepare, read_config,
                                runtime_expectation, source_snapshot, validate_config, verify_source)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/ik/runtime-smoke-v1.json"


def snapshot():
    files = {"src/nca/ik/declaration.py": "a" * 64,
             "src/nca/ik/runner_right_sole_v1.json": "b" * 64}
    return {"revision": "c" * 40, "source_sha256": sha256_json(files), "files": files}


class DeclarationTests(unittest.TestCase):
    def test_bare_import_and_runtime_metadata_never_import_optional_libraries(self):
        program = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from nca.ik.declaration import runtime_expectation; runtime_expectation(); "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        )
        subprocess.run([sys.executable, "-S", "-c", program, str(ROOT / "src")], check=True, capture_output=True)

    def test_fixed_configuration_and_strict_fields(self):
        config = read_config(CONFIG)
        self.assertEqual(validate_config(config), config)
        for changed in ({**config, "extra": 1}, {key: value for key, value in config.items() if key != "seed"},
                        {**config, "workload": "different/v1"}, {**config, "profile": "other"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_config(changed)

    def test_configuration_rejects_boolean_nonfinite_and_overallocated_work(self):
        config = read_config(CONFIG)
        cases = {"seed": [True, -1], "batch_size": [False, 0, 257], "dataset_size": [256., 257],
                 "hidden_width": [True, 64], "steps": [True, 0, 257], "max_optimizer_steps": [True, 257],
                 "max_samples": [True, 1023, 8193], "learning_rate": [True, 0, float("nan")],
                 "memory_fraction": [False, 0, 1.1], "max_wall_seconds": [True, 0, 120.1, float("inf")],
                 "max_checkpoint_bytes": [True, 0, 33554433]}
        for key, values in cases.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_config({**config, key: value})

    def test_read_rejects_duplicate_json_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"schema":"one","schema":"two"}')
            with self.assertRaises(ValueError):
                read_config(path)

    def prepared(self, directory):
        destination = Path(directory) / "allocation"
        with patch("nca.ik.declaration.source_snapshot", return_value=snapshot()) as source:
            declaration = prepare(CONFIG, "cpu", destination)
            source.assert_called_once_with(None, require_clean=True)
        return destination, declaration

    def test_prepare_freezes_config_source_profile_dataset_runtime_and_atomic_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            destination, declaration = self.prepared(directory)
            raw = (destination / "declaration.json").read_bytes()
            self.assertEqual(load_declaration(destination, hashlib.sha256(raw).hexdigest()), declaration)
            self.assertEqual(declaration["source"], snapshot())
            self.assertEqual(declaration["runtime"], runtime_expectation())
            self.assertTrue(declaration["acceptance"]["selected_device_placement"])
            self.assertFalse(declaration["acceptance"]["quality_qualification"])
            with patch("nca.ik.declaration.source_snapshot", return_value=snapshot()):
                with self.assertRaises(FileExistsError):
                    prepare(CONFIG, "cpu", destination)
            self.assertEqual((destination / "declaration.json").read_bytes(), raw)
            with self.assertRaisesRegex(ValueError, "raw declaration SHA-256"):
                load_declaration(destination, "f" * 64)

    def test_loading_requires_semantic_hash_and_strict_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            destination, original = self.prepared(directory)
            path = destination / "declaration.json"
            changed = copy.deepcopy(original)
            changed["device"] = "cuda:0"
            path.write_text(canonical_json(changed))
            with self.assertRaisesRegex(ValueError, "semantic hash"):
                load_declaration(destination)
            for mutate in (lambda value: value["acceptance"].update(finite_loss=1),
                           lambda value: value["source"].update(source_sha256="f" * 64),
                           lambda value: value.update(config_sha256="f" * 64),
                           lambda value: value.update(profile_sha256="f" * 64),
                           lambda value: value.update(host_revision="f" * 40),
                           lambda value: value["config"].update(max_wall_seconds=121)):
                changed = copy.deepcopy(original)
                mutate(changed)
                changed["sha256"] = sha256_json({key: item for key, item in changed.items() if key != "sha256"})
                path.write_text(canonical_json(changed))
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    load_declaration(destination)

    def test_inspection_never_rechecks_current_source_and_metadata_binds_declared_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            destination, declaration = self.prepared(directory)
            with patch("nca.ik.declaration.source_snapshot", side_effect=AssertionError("inspection is read-only")):
                self.assertEqual(load_declaration(destination), declaration)
            values = metadata_for(declaration, "e" * 64)
            self.assertEqual(values["source_sha256"], snapshot()["source_sha256"])
            self.assertEqual(values["environment_sha256"], "e" * 64)
            self.assertEqual(values["dataset_sha256"], declaration["dataset"]["sha256"])
            with self.assertRaises(ValueError):
                metadata_for(declaration, "missing")

    def test_execution_requires_the_original_clean_source_revision_and_hash(self):
        declaration = {"source": snapshot()}
        with patch("nca.ik.declaration.source_snapshot", return_value=snapshot()) as fresh:
            verify_source(declaration)
            fresh.assert_called_once_with(None, require_clean=True)
        for difference in ({"revision": "d" * 40}, {"source_sha256": "d" * 64}):
            with patch("nca.ik.declaration.source_snapshot", return_value={**snapshot(), **difference}):
                with self.assertRaisesRegex(ValueError, "frozen source"):
                    verify_source(declaration)
        with patch("nca.ik.declaration.source_snapshot", side_effect=ValueError("dirty source")):
            with self.assertRaisesRegex(ValueError, "dirty source"):
                verify_source(declaration)

    def test_failed_clean_admission_creates_no_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "refused"
            with patch("nca.ik.declaration.source_snapshot", side_effect=ValueError("dirty source")):
                with self.assertRaisesRegex(ValueError, "dirty source"):
                    prepare(CONFIG, "cpu", destination)
            self.assertFalse(destination.exists())


@unittest.skipUnless(shutil.which("git"), "Git is required for source-admission conformance")
class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        package = Path(nca.__file__).resolve().parent
        # Reuse the existing source_identity boundary through a narrow mock;
        # exercise real Git cleanliness and all newly added JSON/setup binding.
        for source in package.rglob("*.json"):
            target = self.root / "src/nca" / source.relative_to(package)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for relative in ("tools/setup_ik.sh", "tools/setup_ik.ps1", "configs/ik/runtime-smoke-v1.json"):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        (self.root / "src/nca/ik/declaration.py").write_bytes(b'"""Synthetic source fixture."""\n')
        (self.root / ".gitignore").write_text("artifacts/\n.venv-ik/\n")
        self.git("init", "--quiet")
        self.git("config", "core.autocrlf", "false")
        self.commit()
        self.source = patch("nca.ik.declaration.source_identity", side_effect=self.source_identity)
        self.source.start()
        self.addCleanup(self.source.stop)

    def git(self, *arguments):
        return subprocess.check_output(["git", "-C", str(self.root), *arguments],
                                       stderr=subprocess.DEVNULL, text=True).strip()

    def commit(self):
        self.git("add", ".")
        self.git("-c", "user.name=Conformance", "-c", "user.email=conformance@example.invalid",
                 "commit", "--quiet", "-m", "Synthetic source fixture")

    def source_identity(self, root):
        self.assertEqual(root, self.root.resolve())
        files = {path.relative_to(self.root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (self.root / "src/nca").rglob("*.py")}
        return self.git("rev-parse", "HEAD"), files, sha256_json(files)

    def test_snapshot_binds_json_setup_config_and_ignores_declared_artifact_directories(self):
        initial = source_snapshot(self.root)
        self.assertIn("src/nca/ik/runner_right_sole_v1.json", initial["files"])
        self.assertIn("tools/setup_ik.ps1", initial["files"])
        self.assertIn("configs/ik/runtime-smoke-v1.json", initial["files"])
        (self.root / "artifacts").mkdir()
        (self.root / "artifacts/result.json").write_text("{}")
        self.assertEqual(source_snapshot(self.root), initial)

    def test_dirty_tracked_and_untracked_sources_are_refused(self):
        target = self.root / "tools/setup_ik.sh"
        original = target.read_bytes()
        target.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "clean source"):
            source_snapshot(self.root)
        target.write_bytes(original)
        (self.root / "unexpected.txt").write_text("untracked")
        with self.assertRaisesRegex(ValueError, "untracked"):
            source_snapshot(self.root)

    def test_clean_changed_json_cannot_impersonate_the_imported_profile(self):
        target = self.root / "src/nca/ik/runner_right_sole_v1.json"
        target.write_bytes(target.read_bytes() + b"\n")
        self.commit()
        self.assertFalse(self.git("status", "--porcelain"))
        with self.assertRaisesRegex(ValueError, "JSON profile/contract sources differ"):
            source_snapshot(self.root)

    def test_index_flags_cannot_hide_changed_tracked_source_bytes(self):
        relative = "tools/setup_ik.sh"
        target = self.root / relative
        original = target.read_bytes()
        for flag in ("assume-unchanged", "skip-worktree"):
            with self.subTest(flag=flag):
                self.git("update-index", "--" + flag, relative)
                target.write_bytes(original + b"# hidden edit\n")
                self.assertFalse(self.git("status", "--porcelain"))
                with self.assertRaisesRegex(ValueError, "committed source bytes differ"):
                    source_snapshot(self.root)
                target.write_bytes(original)
                self.git("update-index", "--no-" + flag, relative)

    def test_crlf_normalization_matches_head_but_raw_identity_is_preserved(self):
        # Copied files can already use CRLF on Windows. Keep their existing
        # committed-byte policy while changing only the synthetic source's
        # checkout policy; a global autocrlf flip would reinterpret their HEAD
        # blobs and correctly fail the production committed-source check.
        setup = self.root / "tools/setup_ik.ps1"
        setup_crlf = setup.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        setup.write_bytes(setup_crlf)
        (self.root / ".gitattributes").write_bytes(b"src/nca/ik/declaration.py text eol=crlf\n")
        self.commit()
        initial = source_snapshot(self.root)
        target = self.root / "src/nca/ik/declaration.py"
        self.assertNotIn(b"\r", target.read_bytes())
        target.unlink()
        self.git("checkout", "--", "src/nca/ik/declaration.py")
        self.assertIn(b"\r\n", target.read_bytes())
        self.assertFalse(self.git("status", "--porcelain"))
        converted = source_snapshot(self.root)
        self.assertEqual(setup.read_bytes(), setup_crlf)
        self.assertEqual(converted["files"]["tools/setup_ik.ps1"], initial["files"]["tools/setup_ik.ps1"])
        self.assertEqual(converted["revision"], initial["revision"])
        self.assertNotEqual(converted["source_sha256"], initial["source_sha256"])
        self.assertEqual(converted["files"]["src/nca/ik/declaration.py"], hashlib.sha256(target.read_bytes()).hexdigest())

    def test_ignored_untracked_source_cannot_enter_the_committed_closure(self):
        with (self.root / ".gitignore").open("a") as stream:
            stream.write("generated.py\n")
        self.commit()
        (self.root / "src/nca/generated.py").write_text("# untracked source\n")
        self.assertFalse(self.git("status", "--porcelain"))
        with self.assertRaisesRegex(ValueError, "missing or untracked committed file"):
            source_snapshot(self.root)


if __name__ == "__main__":
    unittest.main()
