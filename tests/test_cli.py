import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nca.cli import main
from nca.runs import RunStore


class CliTests(unittest.TestCase):
    def test_fixture_is_immutable_and_does_not_claim_game_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = ["fixture", "--output", temporary, "--run-id", "fixture-test",
                    "--learner-revision", "1" * 40]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
                self.assertEqual(main(args), 2)
            record = RunStore(temporary).read("fixture-test")
            self.assertEqual(record.gameplay_lane, "recorded_fixture")
            self.assertIsNone(record.metrics["gameplay_success"])

    def test_doctor_does_not_import_optional_runtime_or_emit_paths(self):
        script = ("import sys,json;from nca.environment import environment_report;"
                  "r=environment_report();print(json.dumps({'report':r,'torch_imported':'torch' in sys.modules}))")
        result = json.loads(subprocess.check_output([sys.executable, "-c", script], text=True))
        self.assertFalse(result["torch_imported"])
        self.assertEqual(result["report"]["core_runtime_dependencies"], [])
        self.assertNotIn("executable", result["report"])

    def test_unknown_command_fields_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            path.write_text('{"move":[0,0],"terrain_write":true}')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["validate", "wire-action", str(path)]), 2)


if __name__ == "__main__":
    unittest.main()
