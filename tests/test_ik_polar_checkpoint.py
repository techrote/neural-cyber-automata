"""Synthetic CPU checkpoint version isolation; no optimizer updates."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from nca.contracts import ContractError
OPTIONAL = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None

@unittest.skipUnless(OPTIONAL, "optional tensor runtime")
class PolarCheckpointTests(unittest.TestCase):
    def setUp(self):
        from nca.ik.device import configure_device
        from nca.ik.models import build_model
        from nca.ik.polar import build_model as polar, PolarNormalization
        from nca.ik.data import Normalization
        self.ctx=configure_device("cpu", 54011)
        self.v1=build_model(self.ctx, Normalization(1,(0.,)*14,(1.,)*14,"a"*64))
        self.v2=polar(self.ctx, PolarNormalization(1,(2.,)*14,(1.,)*14,"a"*64,"b"*64))

    def payload(self, model, directory):
        from nca.ik.checkpoint import write_checkpoint, load_checkpoint
        torch=self.ctx.torch
        meta={k:"a"*64 for k in ("config_sha256","source_sha256","profile_sha256","dataset_sha256","environment_sha256")}
        state={"boundary":"optimizer_step","cursor":0,"committed_steps":0,
               "normalization":{},"split":"synthetic/v1","generator":"cpu-fixture/v1",
               "remaining_allocation":{"steps":1,"samples":1,"seconds":30.}}
        path=Path(directory)/"fixture.pt"
        ref=write_checkpoint(path,model,torch.optim.Adam(model.parameters()),metadata=meta,training_state=state)
        return load_checkpoint(path,expected_sha256=ref["sha256"],expected_metadata=meta)

    def test_cross_version_restore_refuses_before_mutating_buffers(self):
        from nca.ik.checkpoint import restore_checkpoint
        for source,target in ((self.v1,self.v2),(self.v2,self.v1)):
            with self.subTest(source=source.model_version), tempfile.TemporaryDirectory() as folder:
                before={k:v.clone() for k,v in target.state_dict().items()}
                payload=self.payload(source,folder)
                with self.assertRaises(ValueError):
                    restore_checkpoint(payload,target,self.ctx.torch.optim.Adam(target.parameters()))
                for key,value in before.items():self.ctx.torch.testing.assert_close(target.state_dict()[key],value,rtol=0,atol=0)
        self.assertFalse(self.ctx.torch.cuda.is_initialized())

    def test_same_version_restores_and_legacy_identity_stays_unchanged(self):
        from nca.ik.checkpoint import restore_checkpoint
        self.assertEqual(type(self.v1).__qualname__,"build_model.<locals>.RunnerIKMLP")
        self.assertNotIn("feature_transform_version",self.v1.state_dict())
        for model in (self.v1,self.v2):
            with tempfile.TemporaryDirectory() as folder:
                payload=self.payload(model,folder)
                restore_checkpoint(payload,model,self.ctx.torch.optim.Adam(model.parameters()))
        self.assertFalse(self.ctx.torch.cuda.is_initialized())

    def test_v2_direct_load_rejects_missing_or_wrong_transform_marker(self):
        with self.assertRaises(ContractError):self.v2.load_state_dict(self.v1.state_dict())
        saved=self.v2.state_dict();saved["feature_transform_version"]=self.ctx.torch.tensor(1)
        with self.assertRaises(ContractError):self.v2.load_state_dict(saved)

# Copyright 2026 techrote. All rights reserved.
