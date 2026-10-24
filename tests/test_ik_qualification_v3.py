"""Strict v3 preparation/admission CPU fixtures, never an actual allocation."""
from pathlib import Path
import tempfile
import importlib.util
import os
import time
import unittest
from unittest.mock import patch

from nca.runs import immutable_json, sha256_file
from nca.contracts import sha256_json
from nca.train.ik_dev import DevConfig, SEEDS
from nca.train import ik_qualification_v3 as q


class QualificationContracts(unittest.TestCase):
    @unittest.skipUnless(os.name=="nt","qualified Windows process metadata")
    def test_process_creation_includes_startup_and_is_monotonic(self):
        before=time.monotonic();start=q.invocation_start();after=time.monotonic()
        self.assertGreaterEqual(start,0)
        self.assertLessEqual(start,before)
        self.assertLess(after-before,5.)

    @classmethod
    def setUpClass(cls):
        from nca.ik.data import DatasetConfig, iter_tasks, generate_sample
        from nca.ik.reference import SolverConfig
        cls.config=DatasetConfig(mode="fixture",points_per_bin=(1,1,1),trajectories_per_partition=1,
                                 trajectory_steps=3,stress_per_class=1,
                                 teacher=SolverConfig(tolerance_cells=1e-7,max_seconds=1.))
        cls.training=tuple(generate_sample(t,cls.config) for t in iter_tasks(cls.config,"training"))

    def test_finite_ceilings_one_attempt_no_finals(self):
        for epochs in (40,200,1000):
            config=DevConfig(epochs=epochs)
            limits=q.ceilings(config,self.training)
            self.assertEqual(limits["max_attempts"],1)
            self.assertEqual(limits["max_evaluation_calls"],0)
            self.assertEqual(limits["max_sample_presentations"],len(self.training)*epochs)
            self.assertEqual(limits["max_tuning_calls"],2020*len(config.evaluation_epochs()))
            self.assertEqual(limits["max_checkpoints"],1+len(config.evaluation_epochs()))
            self.assertEqual(limits["max_wall_seconds"],7200)

    def test_shuffled_order_is_frozen_and_ledger_prefix_is_correct(self):
        config=DevConfig(epochs=200,batch_order="shuffled")
        plan=q.ordered_sampler(self.training,41011,config)
        same=q.ordered_sampler(self.training,41011,config)
        self.assertEqual(plan.schedule_sha256,same.schedule_sha256)
        count=0
        while not plan.exhausted:
            rows=plan.next_batch();count+=len(rows)
            self.assertEqual(plan.sample_presentations,count)
        self.assertEqual(count,len(self.training)*200)
        with self.assertRaises(StopIteration):plan.next_batch()

    def test_admission_requires_two_independent_records_and_new_owner_go(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source={"revision":"a"*40,"source_sha256":"b"*64}
            recipe={"final_calls":0,"execution_authorized":False,"executable_source_sha256":"b"*64,
                    "ceilings_per_seed":{"max_attempts":1},"schedules":{},"corpus_identity":{},
                    "normalization_sha256":sha256_json({})}
            declaration={"schema":"nca-ik-qualification-v3/v1","source":source,"final_calls":0,
                         "recipe":recipe,"allocations":recipe["ceilings_per_seed"],"schedules":{},
                         "corpus_identity":{},"normalization":{}}
            immutable_json(root/"declaration.json",declaration)
            digest=sha256_file(root/"declaration.json")
            auth={"schema":"nca-ik-v3-owner-execution/v1","declaration_file_sha256":digest,
                  "execute_full_training":True,"final_calls":0}
            immutable_json(root/"owner.json",auth)
            paths=[]
            for reviewer in ("fixture-independent-a","fixture-independent-b"):
                path=root/(reviewer+".json");paths.append(path)
                immutable_json(path,{"schema":"nca-ik-v3-admission/v1","head":source["revision"],
                   "declaration_file_sha256":digest,"reviewer":reviewer,"independent_non_author":True,
                   "checks":{k:True for k in q.CHECKS},"verdict":"admit_strict_training_only"})
            with patch.object(q,"source",return_value=source),patch.object(q,"recipe_record",return_value=recipe):
                self.assertEqual(q.check_admission(root,paths,root/"owner.json"),declaration)
                with self.assertRaises(ValueError):q.check_admission(root,paths[:1],root/"owner.json")
                with self.assertRaises(ValueError):q.check_admission(root,[paths[0],paths[0]],root/"owner.json")
                immutable_json(root/"wrong-owner.json",dict(auth,final_calls=1))
                with self.assertRaises(ValueError):q.check_admission(root,paths,root/"wrong-owner.json")
                immutable_json(root/"pending-owner.json",dict(auth,execute_full_training=False))
                with self.assertRaises(ValueError):q.check_admission(root,paths,root/"pending-owner.json")
                changed=dict(declaration,allocations={"max_attempts":2})
                with self.assertRaises(ValueError):q.validate_declaration(changed,source,recipe)

    @unittest.skipUnless(importlib.util.find_spec("torch"),"optional Torch CPU runtime")
    def test_synthetic_cpu_full_training_uses_original_durable_ledger(self):
        from types import SimpleNamespace
        from nca.ik.device import configure_device
        from nca.ik.polar import PolarNormalization
        from nca.ik.data import iter_tasks,generate_sample
        from nca.train.ik_corpus import make_fixture_corpus
        from nca.train.ik import prepare_training_batch
        from nca.evaluation.ik_training import score_raw
        tuning=tuple(generate_sample(t,self.config) for t in iter_tasks(self.config,"tuning"))
        corpus=make_fixture_corpus(self.config,self.training,tuning)
        context=configure_device("cpu",41011)
        normalization=PolarNormalization(1,(0.,)*14,(1.,)*14,"a"*64,"b"*64)
        config=DevConfig(epochs=1)
        class FixturePrepared:
            def __init__(self,ctx,fixture,samples):
                if fixture.accepted:raise AssertionError("synthetic fixture only")
                self.ctx,self.samples=ctx,samples
            def batch(self,indexes):
                return prepare_training_batch(tuple(self.samples[i] for i in indexes),corpus.config,self.ctx)
        def fixture_score(samples,predictions,config,**kwargs):
            if kwargs.pop("authenticated_corpus").accepted:raise AssertionError("synthetic fixture only")
            return score_raw(samples,predictions,config,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            ledger=q.IKRunLedger.create(Path(directory)/"seed",identity={**{k:"c"*64 for k in
                ("declaration_sha256","config_sha256","source_sha256","profile_sha256","dataset_sha256",
                 "normalization_sha256","runtime_sha256")},"seed":41011},allocation=q.ceilings(config,self.training))
            attempt=ledger.begin_attempt()
            with patch("nca.train.ik_dev_trainer.PreparedRows",FixturePrepared),patch("nca.evaluation.ik_training.score_raw",fixture_score):
                selected=q.strict_train(ledger,attempt,corpus,context,config,normalization)
            state=ledger.inspect()
            self.assertEqual(state["attempt_count"],1)
            self.assertFalse(state["active"])
            self.assertEqual(state["charged"]["evaluation_calls"],0)
            self.assertEqual(state["charged"]["tuning_calls"],len(tuning))
            self.assertEqual(state["charged"]["sample_presentations"],len(self.training))
            self.assertEqual(state["charged"]["checkpoints"],2)
            self.assertEqual(selected["epoch"],1)

# Copyright 2026 techrote. All rights reserved.
