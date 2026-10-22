"""Bounded normal setup and unbounded manual Jolt setup regressions.

No host runtime, network transfer, subprocess or model is launched here.
© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("nca_host_setup", ROOT / "tools/host_setup.py")
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)


class JoltLongSetupTests(unittest.TestCase):
    def test_regular_setup_retains_bounded_default(self):
        with mock.patch.object(SETUP.subprocess, "run") as fake:
            SETUP.run(["git", "--version"])
        fake.assert_called_once_with(["git", "--version"], check=True, env=None, timeout=300)

    def test_explicit_unbounded_setup_has_no_helper_deadline(self):
        with mock.patch.object(SETUP.subprocess, "run") as fake:
            SETUP.run(["git", "--version"], timeout=None)
        fake.assert_called_once_with(["git", "--version"], check=True, env=None, timeout=None)

    def test_manual_cli_flag_passes_unbounded_to_prepare(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder)
            runtime = {"host_revision": "revision", "godot_version": "4.7",
                       "native_artifact_sha256": "native", "rapier_artifact_sha256": "rapier"}
            with mock.patch.object(sys, "argv", ["host_setup.py",
                                                 "--destination", str(dest),
                                                 "--unbounded"]), \
                 mock.patch.object(SETUP, "prepare",
                                   return_value=(dest / "host", dest / "godot", runtime)) as prepare:
                self.assertEqual(SETUP.main(), 0)
            prepare.assert_called_once_with(dest, None, unbounded=True)

    def test_standard_cli_does_not_select_unbounded(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder)
            runtime = {"host_revision": "revision", "godot_version": "4.7",
                       "native_artifact_sha256": "native", "rapier_artifact_sha256": "rapier"}
            with mock.patch.object(sys, "argv", ["host_setup.py", "--destination", str(dest)]), \
                 mock.patch.object(SETUP, "prepare",
                                   return_value=(dest / "host", dest / "godot", runtime)) as prepare:
                self.assertEqual(SETUP.main(), 0)
            prepare.assert_called_once_with(dest, None, unbounded=False)


if __name__ == "__main__":
    unittest.main()
