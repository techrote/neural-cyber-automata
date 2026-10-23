"""Synthetic publication and byte-binding regressions, no real generation."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from nca.train import ik_followup as followup, ik_followup_seal as seal

class FollowupEnvelopeTests(unittest.TestCase):
    def test_publication_overrun_retains_hold_and_cannot_verify_seal(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"seal.json").write_text(json.dumps({"sha256":"a"*64,"peak_process_bytes":1,"archive_sha256":"a"*64}))
            process=SimpleNamespace(returncode=0,wait=lambda timeout:None)
            with patch.object(seal,"load",return_value={"sha256":"a"*64}), patch.object(seal.subprocess,"Popen",return_value=process), patch.object(seal,"STARTED",0), patch.object(seal.time,"monotonic",side_effect=[898,898,901,901]), patch.object(seal,"sha256_file",return_value="a"*64):
                with self.assertRaisesRegex(ValueError,"publication exceeded"):seal.supervise(root)
            self.assertTrue((root/"supervisor-overrun.json").exists())
            with patch.object(seal,"load") as load:
                with self.assertRaisesRegex(ValueError,"overrun blocks"):seal.verify_seal(root)
                load.assert_not_called()

    def test_equal_length_archive_replacement_refuses_before_opening(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"sealed-final.tar.gz").write_bytes(b"y")
            receipt={"archive_bytes":1,"archive_sha256":hashlib.sha256(b"x").hexdigest()}
            with patch.object(followup,"tuning_stop_gate",return_value={"final_opening_allowed":True}), patch.object(followup,"select_training",return_value=({},{},{})), patch.object(followup,"verify_seal",return_value=receipt), patch.object(followup.tarfile,"open") as archive:
                with self.assertRaisesRegex(ValueError,"archive bytes changed"):
                    followup.open_final(root,{"fresh_final":receipt},{},{},run="unused",ledgers={})
                archive.assert_not_called()
            self.assertFalse((root/"opened.json").exists())

# Copyright 2026 techrote. All rights reserved.
