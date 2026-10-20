"""Closed-entry and failed-prefix accounting tests; no host or policy execution."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from nca.contracts import ContractError
from nca.train.morphology_screen import prepare_continuation, run_comparison

ROOT = Path(__file__).resolve().parents[1]


class MorphologyScreenAdmissionTests(unittest.TestCase):
    def test_invalid_entry_writes_zero_work_without_importing_torch_or_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            declaration = directory / "not-a-declaration.json"
            declaration.write_text("{}\n")
            code = """
import json, pathlib, sys
from nca.train.morphology_screen import run_comparison
result = run_comparison(root=pathlib.Path(sys.argv[1]), prerequisite_directory=pathlib.Path(sys.argv[2]),
    declaration_path=pathlib.Path(sys.argv[3]), checkout=pathlib.Path('unopened-host'),
    godot=pathlib.Path('unopened-godot'), output=pathlib.Path(sys.argv[4]))
assert result['status'] == 'evidence_invalid'
assert result['host_start_attempts'] == result['model_construction_attempts'] == result['checkpoint_load_attempts'] == 0
assert result['attempted_game_requests'] == result['completed_optimizer_updates'] == 0
assert 'torch' not in sys.modules and 'nca.train.morphology_capture' not in sys.modules
print(json.dumps({'status': result['status'], 'work': 0}))
"""
            result = subprocess.run([sys.executable, "-c", code, str(ROOT), str(directory / "missing-original-evidence"),
                str(declaration), str(directory / "closed")], env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {"status": "evidence_invalid", "work": 0})
            admission = json.loads((directory / "closed/admission.json").read_bytes())
            self.assertFalse(admission["admitted"])
            self.assertFalse(admission["original_blocked_admission_changed"])

    def test_existing_output_is_never_reused_or_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "immutable"
            output.mkdir()
            sentinel = output / "admission.json"
            sentinel.write_bytes(b"original immutable receipt\n")
            with self.assertRaises(FileExistsError):
                prepare_continuation(root=ROOT, prerequisite_directory=Path("missing"), output=output)
            self.assertEqual(sentinel.read_bytes(), b"original immutable receipt\n")

    def test_unchanged_failed_prerequisite_never_reaches_optional_execution(self):
        # Only the gate's refusal is injected: no fake positive game evidence is
        # made, and no model, checkpoint, host or optimizer can be reached.
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            declaration = directory / "declaration.json"
            declaration.write_text("{}\n")
            with patch("nca.train.morphology_screen.validate_continuation",
                       side_effect=ContractError("morphology_prerequisite_failed", "synthetic gate refusal")), \
                 patch("nca.train.morphology_screen._execute") as execute:
                result = run_comparison(root=ROOT, prerequisite_directory=directory,
                    declaration_path=declaration, checkout=Path("unopened-host"), godot=Path("unopened-godot"),
                    output=directory / "blocked")
            execute.assert_not_called()
            self.assertEqual(result["status"], "training_blocked")
            self.assertEqual(result["attempted_game_requests"], 0)
            self.assertEqual(result["completed_optimizer_updates"], 0)


if __name__ == "__main__":
    unittest.main()
