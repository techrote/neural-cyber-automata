"""CPU-only recovery phase preparation, provenance and execution boundaries."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from nca.contracts import canonical_json, sha256_json
from nca.evaluation import ik_recovery as api


class IKRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def write(self, name, value):
        path = self.base/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(value)+'\n', encoding='utf-8')
        return path

    def source_pair(self):
        old = {'source_sha256': api.FAILED_SOURCE_SHA256,
               'files': {'src/nca/ik/data.py': '1'*64,
                         **{p: '2'*64 for p in api.SOURCE_DELTA if not p.endswith('/ik_recovery.py')}}}
        new = deepcopy(old)
        new['source_sha256'] = '3'*64
        new['files']['src/nca/evaluation/ik_recovery.py'] = '4'*64
        return old, new

    def test_source_delta_is_only_registered_frozen_recovery_extension(self):
        old,new=self.source_pair()
        self.assertEqual(api._source_delta(old,new),['src/nca/evaluation/ik_recovery.py'])
        for mutation in ('model','missing','extra','lineage'):
            a,b=deepcopy(old),deepcopy(new)
            if mutation=='model':b['files']['configs/ik/experiments/runner-compact-v1.json']='9'*64
            if mutation=='missing':del b['files']['src/nca/ik/data.py']
            if mutation=='extra':b['files']['src/nca/evaluation/unreviewed.py']='9'*64
            if mutation=='lineage':a['source_sha256']='0'*64
            with self.assertRaises(ValueError):api._source_delta(a,b)

    def test_metadata_preflight_uses_actual_prefixed_layout_and_never_extracts(self):
        corpus=self.base/'corpus';corpus.mkdir()
        self.write('corpus/manifest.json',{'shards':[{'partition':'final','path':'final-00000.jsonl','bytes':7}]})
        archive=self.base/'fixture.tar.gz'
        with tarfile.open(archive,'w:gz') as stream:
            m=tarfile.TarInfo('runner-ik-corpus-v1-20261022/corpus/final-00000.jsonl')
            m.size=7;stream.addfile(m,io.BytesIO(b'private'))
            m=tarfile.TarInfo('runner-ik-corpus-v1-20261022/corpus/pending.jsonl')
            m.size=5;stream.addfile(m,io.BytesIO(b'final'))
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        with patch.object(api,'ARCHIVE_SHA256',digest),patch.object(tarfile.TarFile,'extractfile',side_effect=AssertionError('no payload')):
            result=api._archive_metadata(archive,corpus)
        self.assertFalse(result['payloads_extracted'])
        self.assertEqual(result['final_members']['final-00000.jsonl']['name'],
            'runner-ik-corpus-v1-20261022/corpus/final-00000.jsonl')
        with self.assertRaises(ValueError):api._archive_metadata(archive,corpus)

    def test_metadata_wrong_member_size_refuses_without_payload(self):
        corpus=self.base/'corpus';corpus.mkdir()
        self.write('corpus/manifest.json',{'shards':[{'partition':'final','path':'final-00000.jsonl','bytes':8}]})
        archive=self.base/'fixture.tar.gz'
        with tarfile.open(archive,'w:gz') as stream:
            m=tarfile.TarInfo('runner-ik-corpus-v1-20261022/corpus/final-00000.jsonl');m.size=1
            stream.addfile(m,io.BytesIO(b'x'))
        with patch.object(api,'ARCHIVE_SHA256',hashlib.sha256(archive.read_bytes()).hexdigest()),patch.object(tarfile.TarFile,'extractfile',side_effect=AssertionError('no payload')):
            with self.assertRaises(ValueError):api._archive_metadata(archive,corpus)

    def test_recovery_json_has_explicit_realistic_bounded_size(self):
        path=self.write('large.json',{'retained_inventory':{str(i):'a'*160 for i in range(12000)}})
        self.assertGreater(path.stat().st_size,1_048_576)
        self.assertEqual(len(api._read(path)['retained_inventory']),12000)
        path.write_bytes(b' '*(api.MAX_JSON_BYTES+1))
        with self.assertRaises(ValueError):api._read(path)

    def valid_review(self):
        a={'sha256':'a'*64,'source':{'source_sha256':'b'*64}}
        r={'schema':api.REVIEW_SCHEMA,'amendment_sha256':'a'*64,'source_sha256':'b'*64,
           'reviewers':['actual-reviewer-one','actual-reviewer-two'],
           'checks':{k:True for k in api.CHECKS},'verdict':api.VERDICT}
        return a,r

    def test_new_actual_two_reviewer_admission_cannot_transfer_old_scope(self):
        a,r=self.valid_review();path=self.write('review.json',r)
        self.assertEqual(api.review(path,a),r)
        for key,value in [('schema','nca-ik-evaluation-admission/v1'),('verdict','admit_declared_final_evaluation'),
                          ('source_sha256','0'*64),('amendment_sha256','0'*64),
                          ('reviewers',['same-reviewer','same-reviewer']),('reviewers',[[],[]])]:
            bad=deepcopy(r);bad[key]=value;self.write('review.json',bad)
            with self.assertRaises(ValueError):api.review(path,a)
        r['checks']['allocation_and_resume']=1;self.write('review.json',r)
        with self.assertRaises(ValueError):api.review(path,a)

    def test_context_cannot_be_forged_by_boolean_or_changed_paths(self):
        a,_=self.valid_review()
        args=({},a,None,str(self.base),str(self.base/'amendment.json'),'a'*64,
              str(self.base/'registry'),str(self.base/'failed.json'))
        untrusted=api.RecoveryContext(*args)
        with self.assertRaises(ValueError):untrusted.validate_arguments(self.base,self.base/'amendment.json','a'*64,self.base/'registry')
        trusted=api.RecoveryContext(*args,api._TOKEN)
        trusted.validate_arguments(self.base,self.base/'amendment.json','a'*64,self.base/'registry')
        with self.assertRaises(ValueError):trusted.validate_arguments(self.base,self.base/'amendment.json','b'*64,self.base/'registry')
        self.assertEqual(trusted.worker_arguments()[0],'--failed-amendment')

    def test_prepare_existing_output_refuses_before_accounting(self):
        output=self.base/'exists';output.mkdir()
        with patch.object(api,'_project',return_value=self.base/'source'),patch.object(api,'_failed') as read:
            with self.assertRaises(ValueError):api.prepare(self.base/'run',self.base/'failed/a.json','0'*64,
                self.base/'corpus',self.base/'registry',self.base/'archive',output)
            read.assert_not_called()

    def test_prepare_immutably_binds_paid_history_before_freeze_without_gpu(self):
        run=self.base/'run';run.mkdir()
        failedpath=self.base/'failed'/'amendment.json';failedpath.parent.mkdir()
        output=self.base/'new-recovery'
        old,actual=self.source_pair()
        d={'sha256':'d'*64,'plan':{'qualified_environment':{'driver':'581.80'}},
           'dataset':{'same':True},'allocations':{str(s):{'same':True} for s in api.SEEDS}}
        states={};ledgers={};paid={}
        for seed in api.SEEDS:
            index=3 if seed==41011 else 2
            root=run/f'seed-{seed}';root.mkdir()
            for stem in ('supervisor-result','supervisor-overrun'):
                (root/f'{stem}-{index:04d}.json').write_text('{}')
            cp={'reference':{'sha256':str(seed).zfill(64)}}
            state={'attempt_count':index-1,'parent_sha256':'c'*64,'latest_checkpoint':cp}
            states[str(seed)]=state
            ledger=ledgers[seed]=Mock(identity={'seed':seed},header={'sha256':'h'*64})
            paid[seed]=False
            ledger.inspect.side_effect=lambda seed=seed,state=state: (
                {'historical_paid':True,'seed':seed} if paid[seed] else deepcopy(state))
        failed={'source':old,'runtime':{'same':True},'gpu_visibility_sha256':'e'*64,
            'baseline_ledgers':states,'baseline_files':{'old':{}},'training_execution':{'old':True},
            'original_declaration_file_sha256':'f'*64,'frozen_candidates':{'sha256':api.FROZEN_SHA256}}
        failed.update({k:d[k] for k in ('plan','dataset','allocations')})
        historical={'original': 'a'*64}
        opening={'sha256':'o'*64};binding={'sha256':'p'*64}
        proof=SimpleNamespace()
        training=SimpleNamespace()
        events=[]
        def record(proof):
            self.assertTrue((output/'recovery-intent.json').is_file())
            self.assertFalse((output/'amendment.json').exists())
            preflight.assert_called_once()
            events.append(proof)
            paid[proof['seed']]=True
        for ledger in ledgers.values():ledger.record_recovery.side_effect=record
        with patch.object(api,'_project',return_value=self.base/'source'), \
             patch.object(api,'_failed',return_value=(d,failed,{'original':True},'r'*64)), \
             patch.object(api,'_source',return_value=actual), \
             patch.object(api,'runtime_expectation',return_value=failed['runtime']), \
             patch.object(api,'_titan_visibility',return_value='e'*64), \
             patch.object(api,'_failure_facts',return_value=(130.984,131,historical)), \
             patch('nca.evaluation.ik_final_corpus.RECOVERY_HISTORICAL_RECEIPTS',historical), \
             patch.object(api,'_registry',return_value=(opening,binding,{})), \
             patch.object(api,'_archive_metadata',return_value={'payloads_extracted':False}), \
             patch.object(api,'_inventory',return_value={'original.bin':{'bytes':1,'sha256':'0'*64}}), \
             patch.object(api,'_baseline_unchanged'), \
             patch.object(api,'identity_for',side_effect=lambda d,s:{'seed':s}), \
             patch.object(api.IKRunLedger,'open',side_effect=lambda root,**kw:ledgers[int(Path(root).name.split('-')[1])]), \
             patch('nca.evaluation.ik_final_corpus.validate_recovery_proof',return_value=proof), \
             patch('nca.evaluation.ik_final_corpus.read_recovery_training_corpus',return_value=training), \
             patch('nca.evaluation.ik_final_corpus.verify_recovery_training_preflight') as preflight, \
             patch('nca.evaluation.ik_continuation_eval.freeze_training',return_value=(failed['frozen_candidates'],{})), \
             patch.object(api,'corpus_identity',return_value=d['dataset']), \
             patch.object(api,'_proof_document',return_value={'fixture':True}):
            preflight.side_effect=ValueError('retained report mismatch')
            with self.assertRaisesRegex(ValueError,'retained report mismatch'):
                api.prepare(run,failedpath,api.FAILED_AMENDMENT_FILE_SHA256,self.base/'corpus',
                    self.base/'registry',self.base/'archive',output)
            self.assertFalse(output.exists())
            self.assertFalse(events)
            for ledger in ledgers.values():ledger.record_recovery.assert_not_called()
            preflight.side_effect=None;preflight.reset_mock()
            result=api.prepare(run,failedpath,api.FAILED_AMENDMENT_FILE_SHA256,self.base/'corpus',
                self.base/'registry',self.base/'archive',output)
            context=api.load(run,output/'amendment.json',result['amendment_file_sha256'],
                failedpath,api.FAILED_AMENDMENT_FILE_SHA256,self.base/'registry')
            context.validate_arguments(run,output/'amendment.json',result['amendment_file_sha256'],
                self.base/'registry')
            with self.assertRaises(ValueError):
                api.load(run,output/'amendment.json','0'*64,failedpath,
                    api.FAILED_AMENDMENT_FILE_SHA256,self.base/'registry')
            with patch.object(api,'_source',return_value={'different':'source'}):
                with self.assertRaisesRegex(ValueError,'source or runtime differs'):
                    api.load(run,output/'amendment.json',result['amendment_file_sha256'],
                        failedpath,api.FAILED_AMENDMENT_FILE_SHA256,self.base/'registry')
        self.assertEqual(len(events),3)
        preflight.assert_called_once()
        self.assertTrue(all(p['historical_charge_wall_seconds']==131 for p in events))
        self.assertEqual(result['amendment']['frozen_candidates'],failed['frozen_candidates'])
        self.assertEqual(set(result['amendment']),api._FIELDS)
        self.assertFalse(result['cuda_initialized'])
        self.assertFalse(result['new_final_opening'])
        self.assertEqual(api._sha(output/'amendment.json'),result['amendment_file_sha256'])

    def test_supervisor_rejects_invalid_review_before_target_or_execution(self):
        context=Mock()
        context.review.side_effect=ValueError('fresh admission missing')
        with patch.object(api,'load',return_value=context),patch.object(api,'_titan_visibility') as target, \
             patch('nca.evaluation.ik_continuation_eval.supervise') as execute:
            with self.assertRaises(ValueError):api.supervise('run','amend','hash','failed','oldhash','corpus',
                'registry','archive','review')
            target.assert_not_called();execute.assert_not_called()

    def test_import_and_help_are_cpu_only_no_torch(self):
        result=subprocess.run([sys.executable,'-c',
            "import sys; import nca.evaluation.ik_recovery as x; assert 'torch' not in sys.modules; x.main(['--help'])"],
            capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()

# Copyright 2026 techrote. All rights reserved.
