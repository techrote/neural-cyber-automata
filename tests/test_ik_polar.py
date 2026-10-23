"""Synthetic CPU conformance; no accepted-corpus optimization or CUDA."""
import importlib.util
import math
import unittest
from nca.contracts import ContractError
from nca.ik.polar import PolarNormalization, transform_numeric, build_model

OPTIONAL = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None

def norm():
    return PolarNormalization(1,(0.,)*14,(1.,)*14,'a'*64,'b'*64)

class PolarContractTests(unittest.TestCase):
    def test_transform_uses_permitted_target_and_preserves_other_channels(self):
        raw=(.1,.2,4.6,9.,.3,.1)+(1.,)*8
        got=transform_numeric(raw)
        self.assertAlmostEqual(got[2],39.44)
        self.assertAlmostEqual(got[3],math.atan2(6.2,1))
        self.assertEqual(got[:2]+got[4:],raw[:2]+raw[4:])

    def test_fit_requires_authenticated_original_corpus(self):
        from types import SimpleNamespace
        from nca.ik.polar import fit_training_normalization
        with self.assertRaises(ContractError):
            fit_training_normalization(SimpleNamespace(accepted=True))

@unittest.skipUnless(OPTIONAL,'optional tensor runtime')
class PolarTensorTests(unittest.TestCase):
    def setUp(self):
        from nca.ik.device import configure_device
        self.ctx=configure_device('cpu',54011)
        self.model=build_model(self.ctx,norm())

    def test_transformed_network_channels_and_parameter_count_raw_input_unchanged(self):
        torch=self.ctx.torch
        x=torch.tensor([[.1,.2,4.6,9.,.3,.1]+[1.]*8]);saved=x.clone();captured=[]
        hook=self.model.network[0].register_forward_pre_hook(lambda _,args:captured.append(args[0].detach().clone()))
        try:prediction=self.model(x)
        finally:hook.remove()
        self.assertTrue(prediction.valid.all())
        torch.testing.assert_close(captured[0],torch.tensor([transform_numeric(tuple(x[0].tolist()))]),rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(x,saved,rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),5250)
        self.assertEqual(self.model.model_version,2)

    def test_raw_cartesian_fk_loss_keeps_finite_gradients(self):
        from nca.ik.kinematics import TorchFK,chain_from_profile
        from nca.ik.profiles import runner_right_sole_profile
        torch=self.ctx.torch
        x=torch.tensor([[0.,0.,4.6,9.,0.,0.]+[1.]*8])
        out=self.model(x);fk=TorchFK(chain_from_profile(runner_right_sole_profile()),self.ctx)(out.reference)
        loss=((fk.point-x[:,2:4])/6.550781059358212).square().sum();loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
        self.assertTrue(any(p.grad.abs().sum()>0 for p in self.model.parameters()))

    def test_legacy_normalization_cannot_be_reinterpreted(self):
        from nca.ik.data import Normalization
        with self.assertRaises(ContractError):
            build_model(self.ctx,Normalization(1,(0.,)*14,(1.,)*14,'a'*64))
        from nca.ik.models import build_model as legacy
        with self.assertRaises(ContractError):legacy(self.ctx,norm())

    def test_invalid_input_refused_and_sanitized(self):
        torch=self.ctx.torch
        x=torch.tensor([[0.,0.,float('nan'),9.,0.,0.]+[1.]*8])
        p=self.model(x)
        self.assertFalse(p.valid.item());self.assertTrue(torch.isfinite(p.reference).all())

    def test_integrated_trainer_uses_raw_cartesian_loss_and_versioned_receipt(self):
        from nca.ik.polar import build_trainer
        from nca.ik.data import DatasetConfig,TaskState,generate_sample,replay_task
        from nca.ik.reference import SolverConfig
        from nca.train.ik import prepare_training_batch,LENGTH_CELLS
        config=DatasetConfig(mode='fixture',points_per_bin=(1,1,1),trajectories_per_partition=1,
                             stress_per_class=1,teacher=SolverConfig(tolerance_cells=1e-7,max_seconds=1.))
        samples=[generate_sample(replay_task(config,TaskState('point','training',0,0)),config)]
        trainer=build_trainer(self.ctx,norm(),config);batch=prepare_training_batch(samples,config,self.ctx)
        saved=batch.features.clone();total,terms,pred,geometry=trainer.loss(batch)
        expected=((geometry.point-batch.features[:,2:4])/LENGTH_CELLS).square().sum(-1).mean()
        self.ctx.torch.testing.assert_close(terms['endpoint'],expected,rtol=0,atol=0)
        receipt=trainer.step(batch)
        self.assertEqual(receipt['model'],'runner-compact-ik@2')
        self.assertEqual(receipt['trainer'],'runner-compact-ik-step@2')
        self.assertTrue(receipt['weights_changed'])
        self.ctx.torch.testing.assert_close(batch.features,saved,rtol=0,atol=0)

# Copyright 2026 techrote. All rights reserved.
