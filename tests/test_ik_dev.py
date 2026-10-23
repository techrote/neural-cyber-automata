"""CPU development-lane fixtures; accepted-data checks are opt-in, never CUDA."""
from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nca.contracts import ContractError, sha256_json
from nca.train.ik_dev import DevConfig, grid, split, sampler, epoch_batches, point_stage_rms, SELECTION_RULE

OPTIONAL = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None


class DevContracts(unittest.TestCase):
    def test_grid_and_lr_and_cadence(self):
        self.assertEqual(len(grid()), 15)
        self.assertEqual(len({c.id for c in grid()}), 15)
        self.assertEqual(len(DevConfig().evaluation_epochs()), 40)
        self.assertEqual(len(DevConfig(epochs=200).evaluation_epochs()), 40)
        c = DevConfig(lr_schedule="cosine")
        self.assertEqual(c.lr(0, 1720), .001)
        self.assertAlmostEqual(c.lr(1719, 1720), .00001)
        self.assertEqual(DevConfig(lr_schedule="constant_3e-4").lr(0, 10), .0003)
        with self.assertRaises(ValueError):
            DevConfig(epochs=0)

    def test_split_uses_families_and_canonical_hash(self):
        rows = tuple(SimpleNamespace(task=SimpleNamespace(partition="training", family_id=f"family-{i//3}"),
                                     sha256=sha256_json(i)) for i in range(210))
        train, dev, record = split(rows)
        self.assertEqual(len(dev), 30)
        self.assertEqual(len(train), 180)
        self.assertFalse({s.task.family_id for s in train} & {s.task.family_id for s in dev})
        expected = sorted({s.task.family_id for s in rows}, key=lambda f: (sha256_json(["ik54-dev-split/v1", f]), f))[:10]
        self.assertEqual(set(record["dev_families"]), set(expected))
        self.assertEqual(split(rows)[2], record)
        self.assertEqual(set(split(tuple(reversed(rows)))[2]["dev_families"]), set(expected))

    def test_receipt_rms_uses_last_five_point_batches(self):
        receipts = [{"stage": 0, "terms": {"endpoint": i/10000}} for i in range(1, 8)]
        receipts.append({"stage": 3, "terms": {"endpoint": 100.}})
        self.assertAlmostEqual(point_stage_rms(receipts), math.sqrt(.0005)*6.550781059358212)

    def test_ranking_requires_all_seeds_and_uses_median(self):
        from nca.train.ik_dev import rank_configs, SEEDS
        a,b,c=grid()[:3]
        rows=[]
        for config, values in ((a,(.001,.3,.4)),(b,(.1,.1,.9))):
            for seed,p95 in zip(SEEDS,values):
                rows.append({"config":config.to_dict(),"seed":seed,"status":"complete",
                             "selected":{"nominal":{"p95_error_cells":p95,"worst_error_cells":1.}}})
        rows.append({"config":c.to_dict(),"seed":41011,"status":"complete",
                     "selected":{"nominal":{"p95_error_cells":0.,"worst_error_cells":0.}}})
        ranked=rank_configs(rows)
        self.assertEqual([r[1] for r in ranked],[b,a])

    def test_review_replication_fixture(self):
        path = Path(__file__).parent/"fixtures"/"ik"/"dev-review-receipts-v1.json"
        data = json.loads(path.read_text())
        for (seed, row), expected_p95, expected_rms in zip(data["results"].items(),
                (.322219, .304911, .282027), (.141, .142, .145)):
            self.assertAlmostEqual(row["p95"], expected_p95, places=6)
            values = [point_stage_rms([{"stage": 0, "terms": {"endpoint": x}} for x in epoch])
                      for epoch in row["last_point_receipts"]]
            self.assertAlmostEqual(sum(values)/5, row["point_rms_mean"], places=12)
            self.assertAlmostEqual(sum(values)/5, expected_rms, delta=.001)


@unittest.skipUnless(OPTIONAL, "optional Torch runtime")
class DevTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nca.ik.device import configure_device
        from nca.ik.polar import PolarNormalization
        cls.ctx = configure_device("cpu", 41011)
        cls.torch = cls.ctx.torch
        cls.norm = PolarNormalization(1, (0.,)*14, (1.,)*14, "a"*64, "b"*64)

    def test_initial_parameters_and_head_checkpoint_identity(self):
        from nca.ik.polar import build_model as v2
        from nca.ik.models_v3 import build_model
        for seed in (41011, 42013, 43019):
            self.torch.manual_seed(seed); old = v2(self.ctx, self.norm)
            self.torch.manual_seed(seed); new = build_model(self.ctx, self.norm)
            self.assertEqual(new.model_version, 3)
            self.assertTrue(all(self.torch.equal(a,b) for a,b in zip(old.parameters(), new.parameters())))
            for strict in (True, False):
                with self.assertRaises(ContractError):
                    old.load_state_dict(new.state_dict(), strict=strict)
                with self.assertRaises(ContractError):
                    new.load_state_dict(old.state_dict(), strict=strict)
            new.load_state_dict(new.state_dict())
            other = build_model(self.ctx, self.norm, head="tanh")
            with self.assertRaises(ContractError):
                new.load_state_dict(other.state_dict(), strict=False)

    def test_interior_identity_monotonic_c1_bounds_extremes(self):
        from nca.ik.models_v3 import build_model, interior_linear
        torch = self.torch
        model = build_model(self.ctx, self.norm)
        radius = (model.command_upper-model.command_lower)/2
        knee = torch.tensor((.62,.22))/radius
        z = torch.linspace(-1,1,1001)[:,None]*knee
        self.assertTrue(torch.equal(interior_linear(torch,z,radius), z))
        z = torch.linspace(-10,10,20001)[:,None].repeat(1,2).requires_grad_()
        y = interior_linear(torch,z,radius)
        self.assertTrue(bool(((y[1:]-y[:-1])>=0).all()))
        y.sum().backward(); self.assertTrue(bool(torch.isfinite(z.grad).all()))
        for sign in (-1, 1):
            x = (sign*knee + torch.tensor([[-1e-6],[0.],[1e-6]])).requires_grad_()
            y = interior_linear(torch,x,radius); y.sum().backward()
            self.assertTrue(bool((abs(x.grad-1)<2e-5).all()))
        maxval = torch.finfo(torch.float32).max
        x = torch.tensor([-math.inf,-maxval,-1e30,-1.,0.,1.,1e30,maxval,math.inf])[:,None].repeat(1,2)
        q = radius*interior_linear(torch,x,radius)
        self.assertTrue(bool((q.abs()<=radius).all()))
        self.assertTrue(torch.equal(q[0], -radius)); self.assertTrue(torch.equal(q[-1],radius))
        x = x[1:-1].detach().requires_grad_(); interior_linear(torch,x,radius).sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_masks_polar_transform_and_nan_output(self):
        from nca.ik.models_v3 import build_model
        from nca.ik.polar import build_model as v2
        torch = self.torch
        a, b = build_model(self.ctx,self.norm), v2(self.ctx,self.norm)
        x = torch.tensor([[0.,0.,4.6,9.,0.,0.]+[1.]*8]*5)
        x[1,6]=.5; x[2,7]=0; x[3,2]=math.nan; x[4,0]=1
        captured = []
        for model in (a,b):
            hook=model.network[0].register_forward_pre_hook(lambda _,args:captured.append(args[0].detach().clone()))
            model(x); hook.remove()
        self.assertTrue(torch.equal(captured[0],captured[1]))
        self.assertTrue(torch.equal(a(x).invalid_code,b(x).invalid_code))
        with torch.no_grad():
            a.network[-1].bias.fill_(math.nan)
        prediction=a(x[:1]); self.assertFalse(bool(prediction.valid[0]))
        self.assertTrue(int(prediction.invalid_code[0]) & 32)

    def test_width128_and_batch_one_parity_on_every_fixture_row(self):
        from nca.ik.models_v3 import build_model
        from nca.ik.data import DatasetConfig, iter_tasks, generate_sample
        from nca.ik.reference import SolverConfig
        from nca.train.ik_dev import parity, batched_predictions, dev_score
        config=DatasetConfig(mode="fixture",points_per_bin=(1,1,1),trajectories_per_partition=1,
                             trajectory_steps=3,stress_per_class=1,
                             teacher=SolverConfig(tolerance_cells=1e-7,max_seconds=1.))
        rows=tuple(generate_sample(t,config) for t in iter_tasks(config,"training"))
        for head in ("tanh","interior_linear"):
            for width in (64,128):
                model=build_model(self.ctx,self.norm,head=head,hidden_width=width)
                check=parity(self.ctx,model,rows)
                self.assertEqual(check["rows"],len(rows))
                self.assertLessEqual(check["max_joint_difference_radians"],2e-5)
                self.assertTrue(dev_score(rows,batched_predictions(self.ctx,model,rows))["complete"])
        from nca.evaluation.ik_training import score_raw
        tuning=tuple(generate_sample(t,config) for t in iter_tasks(config,"tuning"))
        predictions=batched_predictions(self.ctx,model,tuning)
        official=score_raw(tuning,predictions,config)
        dev=dev_score(tuning,predictions)
        self.assertEqual(official["nominal"],dev["nominal"])
        self.assertEqual(official["continuity"],dev["continuity"])
        self.assertEqual(official["complete"],dev["complete"])


@unittest.skipUnless(os.environ.get("IK_DEV_CORPUS"), "explicit authenticated training/tuning local readback")
class DevAcceptedReadback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nca.train.ik_corpus import read_polar_followup_corpus
        cls.corpus=read_polar_followup_corpus(os.environ["IK_DEV_CORPUS"],os.environ["IK_DEV_REGISTRY"],source_sha256="a"*64)

    def test_defaults_reproduce_original_full_sampler_hashes(self):
        expected=("5bd631790cc5d70571c63f3740f71e0e7c73e2854ebd514285f19ef25474ff33",
                  "41a01cfc06ebbe47bdab479245d05405a00d6e993ac52f8305ad411472649914",
                  "2ee545e1589c5a36d6fd9ba89030ec00a5d9563dd30eb73b456faca355b84d14")
        for seed, digest in zip((41011,42013,43019),expected):
            plan=sampler(self.corpus.training,seed,DevConfig())
            self.assertEqual(plan.schedule_sha256,digest)
        train,dev,_=split(self.corpus.training)
        plan=sampler(train,41011,DevConfig(epochs=200,batch_order="shuffled"))
        for e in (0,39,40,199):
            stage=epoch_batches(plan,DevConfig(epochs=200),e)
            shuffled=epoch_batches(plan,DevConfig(epochs=200,batch_order="shuffled"),e)
            self.assertCountEqual(stage,shuffled)
            self.assertEqual(sorted(i for b in shuffled for i in b),list(range(len(train))))

    @unittest.skipUnless(OPTIONAL,"optional Torch runtime")
    def test_cached_batches_match_original_and_default_step(self):
        from nca.ik.device import configure_device
        from nca.ik.polar import fit_training_normalization, build_trainer
        from nca.train.ik import prepare_training_batch
        from nca.train.ik_dev_trainer import PreparedRows, DevTrainer
        ctx=configure_device("cpu",41011)
        train,_,_=split(self.corpus.training)
        cache=PreparedRows(ctx,self.corpus,train)
        plan=sampler(train,41011,DevConfig())
        batches=epoch_batches(plan,DevConfig(),0)
        for indexes in batches:
            original=prepare_training_batch(tuple(train[i] for i in indexes),self.corpus.config,ctx,authenticated_corpus=self.corpus)
            cached=cache.batch(indexes)
            for name in ("features","teacher_reference","pose_mask","endpoint_mask","continuity_pairs"):
                self.assertTrue(ctx.torch.equal(getattr(original,name),getattr(cached,name)),name)
            self.assertEqual(original.sample_ids,cached.sample_ids)
        norm=fit_training_normalization(self.corpus)
        ctx.torch.manual_seed(41011); old=build_trainer(ctx,norm,self.corpus.config)
        ctx.torch.manual_seed(41011); new=DevTrainer(ctx,norm,self.corpus.config,DevConfig())
        batch=cache.batch(batches[0]); old.step(batch); new.step(batch)
        for a,b in zip(old.model.parameters(),new.model.parameters()):
            self.assertTrue(ctx.torch.equal(a,b))

# Copyright 2026 techrote. All rights reserved.
