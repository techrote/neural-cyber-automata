"""CPU-only negative controls for the bounded polar experiment boundaries."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nca.train.ik_followup import plan, identity_for, open_final, load, select_training
from nca.train.ik_followup_seal import geometry_identity, verify_seal, process_peak_bytes
from nca.train.ik_experiment import train_attempt
from nca.train.ik_corpus import read_polar_followup_corpus
from nca.train.ik_run_state import IKRunLedger


def fixture_ledger(root):
    identity={k:'a'*64 for k in ('declaration_sha256','config_sha256','source_sha256',
        'profile_sha256','dataset_sha256','normalization_sha256','runtime_sha256')}
    identity['seed']=41011
    allocation=dict(max_optimizer_steps=10,max_sample_presentations=100,max_tuning_calls=100,
        max_evaluation_calls=100,max_checkpoints=10,max_attempts=4,max_wall_seconds=120,max_memory_bytes=1024**3)
    return IKRunLedger.create(root,identity=identity,allocation=allocation)


class FollowupContractTests(unittest.TestCase):
    def test_installed_wheel_resolves_plan_from_verified_checkout(self):
        expected=plan()
        with patch('nca.train.ik_followup.__file__','/unrelated/site-packages/nca/train/ik_followup.py'):
            self.assertEqual(plan(),expected)

    def test_fixed_budget_and_geometry_without_labels(self):
        schedule = plan()
        self.assertEqual(schedule["training_seeds"], [41011,42013,43019])
        self.assertEqual((schedule["epochs"],schedule["batch_size"]), (40,128))
        self.assertEqual(schedule["max_wall_seconds_per_seed"],7200)
        self.assertEqual(schedule["final_comparators"],6)
        config,tasks,identity = geometry_identity()
        self.assertEqual(len(tasks),2940)
        self.assertEqual(identity["ordered_geometry_sha256"],
            "57f7008cbaf6d776ad36e2aeec97019a68a8929e697a2734ea3fde2d7abe64c2")
        self.assertTrue(all(t.partition == "final" for t in tasks))
        self.assertEqual(config.budget.max_teacher_iterations,188160)
        self.assertGreater(process_peak_bytes(),0)

    def test_no_final_archive_read_before_all_seed_tuning_gate(self):
        with patch("nca.train.ik_followup.verify_seal") as seal, \
                patch("nca.train.ik_followup.tarfile.open") as archive:
            with self.assertRaises(ValueError):
                open_final("unused", {}, {}, {}, run="unused", ledgers={})
            seal.assert_not_called()
            archive.assert_not_called()
        with patch("nca.train.ik_followup.tuning_stop_gate", return_value={"final_opening_allowed":False}), \
                patch("nca.train.ik_followup.verify_seal") as seal:
            with self.assertRaisesRegex(ValueError,"tuning stop gate"):
                open_final("unused", {}, {}, {}, run="unused", ledgers={})
            seal.assert_not_called()

    def test_legacy_training_refuses_new_corpus_before_runtime(self):
        with self.assertRaisesRegex(ValueError,"source-bound polar"):
            train_attempt(None,None,SimpleNamespace(followup_only=True,recovery_only=False),None,
                          epochs=40,batch_size=128)

    def test_untyped_or_absent_original_source_cannot_get_view(self):
        for source in (None, True, "a"*63):
            with self.subTest(source=source), self.assertRaises(ValueError):
                read_polar_followup_corpus("unused","unused",source_sha256=source)

    def test_modified_declaration_bytes_refuse_before_source_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,"declaration.json").write_text("{}",encoding="utf-8")
            with patch("nca.train.ik_followup._source") as source, \
                    patch("nca.train.ik_followup._titan_visibility") as gpu:
                with self.assertRaisesRegex(ValueError,"file SHA-256"):
                    load(directory,"a"*64)
                source.assert_not_called()
                gpu.assert_not_called()

    def test_no_selection_from_partial_training(self):
        partial=SimpleNamespace(inspect=lambda:{"active":False,"unresolved_work":False,"latest_checkpoint":None})
        with patch("nca.train.ik_followup.ledger_at",return_value=partial), self.assertRaisesRegex(ValueError,"complete reconciled"):
            select_training("unused",{})

    def test_real_empty_active_ledger_reaches_completion_check_and_pending_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=fixture_ledger(Path(directory)/'seed')
            attempt=ledger.begin_attempt()
            self.assertTrue(ledger.inspect()['active'])
            self.assertTrue(ledger.inspect()['unresolved_work'])
            with self.assertRaisesRegex(ValueError,'complete reconciled forty-epoch'):
                select_training('unused',{},active_ledgers={41011:ledger})
            attempt.reserve('evaluation',calls=1,call_id='fixture-pending')
            with self.assertRaisesRegex(ValueError,'authenticated empty active'):
                select_training('unused',{},active_ledgers={41011:ledger})
            attempt.finish('failed',reason='synthetic_pending_negative_control')

    def test_run_wide_claim_blocks_different_seed_before_spawn_and_preserves_receipt(self):
        from nca.train.ik_followup import supervise
        import time
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            states={}
            class Ledger:
                def __init__(self,seed):
                    self.root=root/f'seed-{seed}'
                    self.root.mkdir()
                    self.state=dict(active=False,unresolved_work=False,can_resume=True,
                        attempt_count=0,charged_wall_seconds=0.,remaining_wall_seconds=120.)
                def inspect(self):
                    return dict(self.state)
            for seed in (41011,42013,43019):
                states[seed]=Ledger(seed)
            args=SimpleNamespace(run=directory,declaration_sha256='a'*64,review='unused',phase='train',
                seed=41011,corpus='unused',registry='unused',fresh='unused',archived_run='unused')
            calls=[]
            class Process:
                returncode=0
                def wait(self,timeout=None):
                    states[41011].state.update(charged_wall_seconds=20.,attempt_count=1)
            def launch(*a,**kw):
                calls.append(1)
                concurrent=SimpleNamespace(**vars(args))
                concurrent.seed=42013
                with self.assertRaises(FileExistsError):
                    supervise(concurrent)
                return Process()
            with patch('nca.train.ik_followup.load',return_value={'sha256':'b'*64}), \
                    patch('nca.train.ik_followup._review',return_value={}), \
                    patch('nca.train.ik_followup.ledger_at',side_effect=lambda r,d,s:states[s]), \
                    patch('nca.train.ik_followup.ENTRY_STARTED',time.monotonic()), \
                    patch('nca.train.ik_followup.subprocess.Popen',side_effect=launch):
                result=supervise(args)
            self.assertEqual(len(calls),1)
            self.assertTrue(result['41011']['verified_within_charged_wall'])
            self.assertFalse((root/'active-phase.json').exists())
            self.assertEqual(len(list((root/'phase-claims').glob('*-start.json'))),1)
            self.assertEqual(len(list((root/'phase-claims').glob('*-result.json'))),1)

    def test_imports_do_not_initialize_optional_runtime(self):
        code = "import sys; import nca.train.ik_followup; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'), 'optional tensor runtime')
class FollowupSyntheticResumeTests(unittest.TestCase):
    def test_actual_polar_trainer_loop_resumes_identically_and_closes_evaluation_charge(self):
        from nca.contracts import sha256_json
        from nca.ik.data import DatasetConfig,generate_sample,iter_tasks
        from nca.ik.device import configure_device
        from nca.ik.polar import PolarNormalization,build_trainer
        from nca.ik.checkpoint import load_checkpoint
        from nca.train.ik_corpus import make_fixture_corpus,PairPreservingSampler
        config=DatasetConfig(mode='fixture',points_per_bin=(1,1,1),
            trajectories_per_partition=1,trajectory_steps=3,stress_per_class=1)
        training=tuple(generate_sample(t,config) for t in iter_tasks(config,'training'))
        tuning=tuple(generate_sample(t,config) for t in iter_tasks(config,'tuning'))
        fixture=make_fixture_corpus(config,training,tuning)
        corpus=replace(fixture,_followup_source_sha256='a'*64)
        normalization=PolarNormalization(1,(0.,)*14,(1.,)*14,
            fixture.training_inputs_sha256,fixture.normalization.sha256)
        def run(root,pause=False):
            identity={k:'a'*64 for k in ('declaration_sha256','config_sha256','source_sha256',
                'profile_sha256','dataset_sha256','normalization_sha256','runtime_sha256')}
            identity.update(seed=41011,normalization_sha256=normalization.sha256)
            allocation=dict(max_optimizer_steps=20,max_sample_presentations=1000,max_tuning_calls=1000,
                max_evaluation_calls=100,max_checkpoints=10,max_attempts=4,max_wall_seconds=120,max_memory_bytes=1024**3)
            ledger=IKRunLedger.create(root,identity=identity,allocation=allocation)
            context=configure_device('cpu',41011)
            attempt=ledger.begin_attempt()
            with patch('nca.ik.polar.fit_training_normalization',return_value=normalization):
                result=train_attempt(ledger,attempt,corpus,context,epochs=1,batch_size=32,
                    pause_after_batches=1 if pause else None,polar_normalization=normalization)
                if pause:
                    self.assertFalse(result['training_complete'])
                    context=configure_device('cpu',41011)
                    attempt=ledger.begin_attempt()
                    result=train_attempt(ledger,attempt,corpus,context,epochs=1,batch_size=32,
                        polar_normalization=normalization)
            self.assertTrue(result['training_complete'])
            cp=ledger.inspect()['latest_checkpoint']
            payload=load_checkpoint(root/cp['path'],expected_sha256=cp['reference']['sha256'],expected_metadata=identity)
            return ledger,payload
        with tempfile.TemporaryDirectory() as directory:
            first,payload=run(Path(directory)/'full')
            resumed,restored=run(Path(directory)/'resume',True)
            for key,tensor in payload['model'].items():
                self.assertTrue(tensor.equal(restored['model'][key]),key)
            self.assertEqual(first.inspect()['completed_optimizer_steps'],resumed.inspect()['completed_optimizer_steps'])
            # A successful evaluation charge must be followed by an unchanged
            # trainer checkpoint; otherwise the real ledger reports unresolved.
            attempt=resumed.begin_attempt()
            context=configure_device('cpu',41011)
            trainer=build_trainer(context,normalization,config)
            sampler=PairPreservingSampler(training,41011,batch_size=32,epochs=1)
            extra=resumed.resume(trainer,sampler)
            token=attempt.reserve('evaluation',calls=100,call_id='synthetic-evaluation-boundary')
            attempt.complete(token,{'synthetic':True})
            attempt.checkpoint(trainer,sampler.state_dict(),extra_state=extra)
            attempt.finish('completed',reason='synthetic_evaluation_checkpoint_closure')
            self.assertFalse(resumed.inspect()['unresolved_work'])
            self.assertEqual(resumed.inspect()['charged']['evaluation_calls'],100)
            self.assertEqual(resumed.inspect()['completed_optimizer_steps'],first.inspect()['completed_optimizer_steps'])


if __name__ == "__main__":
    unittest.main()

# Copyright 2026 techrote. All rights reserved.
