"""Pure CPU interprocess-exclusive phase claims; no experiments."""
from pathlib import Path
import tempfile
import unittest
from nca.ik.phase_owner import serial_phase

class PhaseOwnerTests(unittest.TestCase):
    def owner(self,root,seed=41011,phase="train"):
        return serial_phase(root,phase=phase,seed=seed,declaration_sha256="a"*64)
    def test_other_seed_and_finish_refuse_while_claim_active(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.owner(folder):
                for seed,phase in ((42013,"train"),(43019,"finish")):
                    with self.assertRaises(FileExistsError):
                        with self.owner(folder,seed,phase):self.fail("concurrent owner")
            self.assertFalse((Path(folder)/"serial-phase-active.json").exists())
            self.assertEqual(len(list(Path(folder).glob("serial-phase-completed-*.json"))),1)
            with self.owner(folder,42013):pass
            self.assertEqual(len(list(Path(folder).glob("serial-phase-completed-*.json"))),2)
    def test_exception_retains_claim_and_prevents_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,"failure"):
                with self.owner(folder):raise ValueError("failure")
            self.assertTrue((Path(folder)/"serial-phase-active.json").is_file())
            with self.assertRaises(FileExistsError):
                with self.owner(folder,42013):pass
    def test_changed_claim_refuses_release(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,"ownership changed"):
                with self.owner(folder):(Path(folder)/"serial-phase-active.json").write_text("changed")
            self.assertTrue((Path(folder)/"serial-phase-active.json").exists())

# Copyright 2026 techrote. All rights reserved.
