"""Strict v4 preparation/admission CPU fixtures, never an actual allocation."""
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
from nca.train import ik_qualification_v4 as q


class QualificationContracts(unittest.TestCase):
    def test_frozen_recipe_retains_training_contract_and_finite_resource_room(self):
        import json
        recipe=q.recipe_record()
        root=Path(__file__).resolve().parents[1]
        old=json.loads((root/'configs/ik/experiments/runner-qualification-v3.json').read_text())
        for key in ('config','training_seeds','schedules','selection','tuning_epochs','normalization_sha256',
                    'corpus_identity','loss_weights','optimizer','runtime','final_calls'):
            self.assertEqual(recipe[key],old[key],key)
        self.assertEqual(recipe['resource_policy'],q.resource_policy())
        self.assertEqual(recipe['ceilings_per_seed'],dict(max_optimizer_steps=43000,
            max_sample_presentations=5260000,max_tuning_calls=404000,max_evaluation_calls=0,
            max_checkpoints=201,max_attempts=1,max_wall_seconds=86400,max_memory_bytes=1024**3))
        self.assertIs(recipe['execution_authorized'],False)
        self.assertGreater(q.WORK_RESERVE_SECONDS-q.PUBLICATION_RESERVE_SECONDS,3*464.)

    def test_new_paid_reserve_is_bounded_without_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=self.fixture_ledger(Path(directory)/'ledger');attempt=ledger.begin_attempt()
            with self.assertRaises(ValueError):
                attempt.finish('failed',reason='fixture_above_cap',publication_reserve_seconds=7201.)
            state=attempt.finish('failed',reason='fixture_paid_reserve',publication_reserve_seconds=7200.)
            self.assertGreaterEqual(state['charged_wall_seconds'],7200.)
            self.assertFalse(state['active']);self.assertFalse(state['can_resume'])

    def test_cleanup_is_bounded_by_earliest_seed_charge(self):
        states=[dict(charged_wall_seconds=x) for x in (100.,200.,300.)]
        q.validate_cleanup_budget(89.,states)
        with self.assertRaises(ValueError):q.validate_cleanup_budget(90.,states)

    def test_callable_deadline_can_tighten_the_hard_timer(self):
        from unittest.mock import Mock
        stop=Mock();stop.wait.return_value=False
        with patch.object(q.time,'monotonic',return_value=101.),patch.object(q.os,'_exit',side_effect=RuntimeError('fixture exit')):
            with self.assertRaisesRegex(RuntimeError,'fixture exit'):q.hard_wall_watchdog(stop,0,lambda:100.)

    def fixture_ledger(self, root):
        identity={**{k:'0'*64 for k in ('declaration_sha256','config_sha256','source_sha256','profile_sha256',
             'dataset_sha256','normalization_sha256','runtime_sha256')},'seed':1}
        return q.IKRunLedger.create(root,identity=identity,allocation=q.ceilings(DevConfig(epochs=1),self.training))

    def test_pending_reservation_retains_charge_and_blocks_success(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=self.fixture_ledger(Path(directory)/'ledger');attempt=ledger.begin_attempt()
            attempt.reserve('optimizer',samples=1,call_id='synthetic-unknown-work')
            with self.assertRaises(ValueError):attempt.finish('completed',reason='fixture')
            state=attempt.finish('failed',reason='fixture_unknown_work')
            self.assertEqual(state['charged']['optimizer_steps'],1)
            self.assertTrue(state['unresolved_work']);self.assertFalse(state['can_resume'])

    def test_late_publication_remains_unsealed_and_active(self):
        from nca.train import ik_run_state_v2 as ledger_module
        with tempfile.TemporaryDirectory() as directory:
            ledger=self.fixture_ledger(Path(directory)/'ledger');attempt=ledger.begin_attempt()
            with patch.object(ledger_module.time,'monotonic',side_effect=[attempt.started+.5,attempt.started+6.]):
                with self.assertRaisesRegex(ValueError,'charged reserve'):
                    attempt.finish('failed',reason='synthetic_late_publication',publication_reserve_seconds=5.)
            self.assertTrue((attempt.folder/'finish.json').exists())
            self.assertFalse((attempt.folder/'sealed.json').exists())
            self.assertTrue(ledger.inspect()['active'])

    def test_terminal_requires_exact_completed_counters(self):
        allocation=dict(max_optimizer_steps=1,max_sample_presentations=128,max_tuning_calls=2020,
                        max_evaluation_calls=0,max_checkpoints=2)
        state=dict(active=False,attempt_count=1,unresolved_work=False,last_status='completed',
                   completed_optimizer_steps=1,charged={k[4:]:v for k,v in allocation.items()})
        q.validate_terminal(state,allocation)
        for changed in (dict(last_status='capped'),dict(completed_optimizer_steps=0),
                        dict(charged={**state['charged'],'tuning_calls':0})):
            with self.assertRaises(ValueError):q.validate_terminal({**state,**changed},allocation)

    def test_hard_wall_exit_has_no_io_or_telemetry_dependency(self):
        from unittest.mock import Mock
        stop=Mock();stop.wait.return_value=False
        with patch.object(q.time,'monotonic',return_value=86400),patch.object(q.os,'_exit',side_effect=RuntimeError('fixture exit')) as exit_call:
            with self.assertRaisesRegex(RuntimeError,'fixture exit'):q.hard_wall_watchdog(stop,0)
            exit_call.assert_called_once_with(2)

    def test_versioned_ledger_large_inventory_preserves_old_limit(self):
        from nca.train import ik_run_state as old,ik_run_state_v2 as new
        inventory={f'event-{i:06d}.json':{'sha256':'0'*64,'bytes':1} for i in range(1,86803)}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(Exception):old._publish(root/'old.json',{'inventory':inventory})
            new._publish(root/'new.json',{'inventory':inventory})
            self.assertEqual(len(new._record(root/'new.json')['record']['inventory']),86802)
        self.assertEqual(old._JSON_LIMIT,8*1024**2)
        self.assertEqual(new.MAX_BYTES,1024**2)

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
            self.assertEqual(limits["max_wall_seconds"],86400)

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
            declaration={"schema":"nca-ik-qualification-v4/v1","source":source,"final_calls":0,
                         "recipe":recipe,"allocations":recipe["ceilings_per_seed"],"schedules":{},
                         "corpus_identity":{},"normalization":{}}
            immutable_json(root/"declaration.json",declaration)
            digest=sha256_file(root/"declaration.json")
            auth={"schema":"nca-ik-v4-owner-execution/v1","declaration_file_sha256":digest,
                  "execute_full_training":True,"final_calls":0}
            immutable_json(root/"owner.json",auth)
            paths=[]
            for reviewer in ("fixture-independent-a","fixture-independent-b"):
                path=root/(reviewer+".json");paths.append(path)
                immutable_json(path,{"schema":"nca-ik-v4-admission/v1","head":source["revision"],
                   "declaration_file_sha256":digest,"reviewer":reviewer,"independent_non_author":True,
                   "checks":{k:True for k in q.CHECKS},"verdict":"admit_strict_training_only"})
            with patch.object(q,"source",return_value=source),patch.object(q,"recipe_record",return_value=recipe):
                self.assertEqual(q.check_reviews(root,paths),declaration)
                self.assertEqual(q.check_admission(root,paths,root/"owner.json"),declaration)
                with self.assertRaises(ValueError):q.check_admission(root,paths[:1],root/"owner.json")
                with self.assertRaises(ValueError):q.check_admission(root,[paths[0],paths[0]],root/"owner.json")
                immutable_json(root/"wrong-owner.json",dict(auth,final_calls=1))
                with self.assertRaises(ValueError):q.check_admission(root,paths,root/"wrong-owner.json")
                immutable_json(root/"pending-owner.json",dict(auth,execute_full_training=False))
                with self.assertRaises(ValueError):q.check_admission(root,paths,root/"pending-owner.json")
                changed=dict(declaration,allocations={"max_attempts":2})
                with self.assertRaises(ValueError):q.validate_declaration(changed,source,recipe)
                with self.assertRaises(ValueError):q.validate_declaration(dict(declaration,resource_policy={'workers':99}),source,recipe)

    @unittest.skipUnless(importlib.util.find_spec("torch"),"optional Torch CPU runtime")
    def test_synthetic_cpu_recipe_uses_versioned_durable_ledger(self):
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
