"""CPU-only evaluation amendment phase separation and immutable lineage tests."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from nca.train import ik_continuation as api

class EvaluationAmendmentTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.parent_source={'revision':'a'*40,'source_sha256':'a'*64,'files':{'src/nca/ik/data.py':'1'*64,**{p:'2'*64 for p in api.EVALUATION_SOURCE_DELTA}}}
  self.current=deepcopy(self.parent_source);self.current['revision']='b'*40;self.current['source_sha256']='b'*64;self.current['files']['src/nca/train/ik_continuation.py']='3'*64
  body={k:None for k in api._AMENDMENT_FIELDS if k!='sha256'}
  body.update(schema=api.SCHEMA,created_utc='fixture',original_declaration_file_sha256=api.ORIGINAL_DECLARATION_FILE_SHA256,original_declaration_sha256='c'*64,original_admission_sha256='d'*64,source=self.parent_source,changed_source_files=['src/nca/train/ik_continuation.py'],runtime={'fixture':True},gpu_visibility_sha256='e'*64,dataset={'fixture':True},plan={'fixture':True},allocations={str(s):{'max_checkpoints':45} for s in api.SEEDS},baseline_ledgers={},baseline_files={'old.json':{'bytes':1,'sha256':'f'*64}},final_opened=False,scope='historical_training',no_new_allocations=True,integrated_upstream_revision=api.UPSTREAM_REVISION,integrated_upstream_source_pins=api.UPSTREAM_SOURCE_PINS)
  self.parent=dict(body,sha256=api.sha256_json(body))
  self.admission={'schema':'nca-ik-continuation-admission/v1','amendment_sha256':self.parent['sha256'],'source_sha256':self.parent_source['source_sha256'],'reviewers':['reviewer-one','reviewer-two'],'checks':{k:True for k in api.CHECKS},'verdict':'admit_declared_CUDA_continuation'}
 def child(self):
  body={k:v for k,v in self.parent.items() if k!='sha256'}
  body.update(schema=api.EVALUATION_SCHEMA,parent_amendment=self.parent,parent_amendment_file_sha256=api._canonical_file_hash(self.parent),parent_admission=self.admission,parent_admission_file_sha256=api._canonical_file_hash(self.admission),source=self.current,changed_source_files=api._evaluation_source_delta(self.parent_source,self.current),training_execution=api.execution_identity(self.parent),evaluation_only=True,frozen_candidates={'fixture':True},scope='evaluate_existing_frozen_three_seed_candidates_only_no_training')
  return dict(body,sha256=api.sha256_json(body))
 def test_only_three_existing_source_modules_can_change(self):
  self.assertEqual(api._evaluation_source_delta(self.parent_source,self.current),['src/nca/train/ik_continuation.py'])
  for key in ('src/nca/ik/data.py','configs/ik/experiments/runner-compact-v1.json'):
   value=deepcopy(self.current);value['files'][key]='9'*64
   with self.assertRaises(ValueError):api._evaluation_source_delta(self.parent_source,value)
  value=deepcopy(self.current);del value['files']['src/nca/ik/data.py']
  with self.assertRaises(ValueError):api._evaluation_source_delta(self.parent_source,value)
 def test_old_admission_cannot_be_rebound_to_evaluation_child(self):
  child=self.child();old=deepcopy(self.admission);old['amendment_sha256']=child['sha256'];old['source_sha256']=child['source']['source_sha256']
  with self.assertRaises(ValueError):api._review_value(old,child)
  fresh=deepcopy(old);fresh.update(schema='nca-ik-evaluation-admission/v1',verdict='admit_declared_final_evaluation')
  self.assertEqual(api._review_value(fresh,child),fresh)
  fresh['reviewers']=['one-reviewer','one-reviewer']
  with self.assertRaises(ValueError):api._review_value(fresh,child)
 def test_parent_authenticates_historical_source_without_claiming_it_is_current(self):
  with patch.object(api,'_declaration_binding',return_value={'fixture':True,'source':{}}),patch.object(api,'_source_delta',return_value=self.parent['changed_source_files']),patch.object(api,'_baseline_unchanged') as baseline,patch.object(api,'_source',side_effect=AssertionError('historical source cannot equal current')):
   api._validate_parent(self.root,self.parent,api._canonical_file_hash(self.parent),self.admission,api._canonical_file_hash(self.admission));baseline.assert_called_once()
   for parent_hash,review_hash in [('0'*64,api._canonical_file_hash(self.admission)),(api._canonical_file_hash(self.parent),'0'*64)]:
    with self.assertRaises(ValueError):api._validate_parent(self.root,self.parent,parent_hash,self.admission,review_hash)
 def test_unknown_child_fields_and_rehashed_training_mutation_refuse(self):
  child=self.child();api._amendment_record(child,evaluation=True)
  child['new_allocation']=1;child['sha256']=api.sha256_json({k:v for k,v in child.items() if k!='sha256'})
  with self.assertRaises(ValueError):api._amendment_record(child,evaluation=True)
  parent=deepcopy(self.parent);parent['schema']=api.EVALUATION_SCHEMA
  with self.assertRaises(ValueError):api._amendment_record(parent)
 def test_evaluation_child_cannot_begin_training_or_touch_device(self):
  child=self.child()
  with patch.object(api,'load_amendment',return_value=({},child)),patch.object(api.IKRunLedger,'open') as ledger,patch.object(api,'device_report') as gpu,patch.object(api,'read_training_corpus') as dataset:
   for fn in (api.worker,api.supervise):
    kw={'started':0.,'review_sha256':'0'*64} if fn is api.worker else {}
    with self.assertRaises(ValueError):fn(self.root,'unused','0'*64,41011,'unused','unused','unused',**kw)
   ledger.assert_not_called();gpu.assert_not_called();dataset.assert_not_called()
 def test_incomplete_or_active_training_refuses_before_freeze(self):
  state={'active':True,'latest_checkpoint':None}
  with patch.object(api,'_states',return_value={str(s):state for s in api.SEEDS}),patch('nca.evaluation.ik_continuation_eval.freeze_training',side_effect=AssertionError('freeze touched')):
   with self.assertRaises(ValueError):api._completed_training_baseline(self.root,{},self.parent,object())
 def test_checkpoint_and_same_budget_proofs_required_for_each_completed_seed(self):
  child=self.child();decl={'allocations':{str(s):{'max_checkpoints':45} for s in api.SEEDS}}
  child['baseline_ledgers']={};child['baseline_files']={}
  for seed in api.SEEDS:
   charged={'sample_presentations':210400,'tuning_calls':80800,'evaluation_calls':0,'checkpoints':42}
   state={'charged':charged,'extra_state':{'execution':child['training_execution'],'completed_epochs':40,'tuning_history':[{}]*40},'sampler':{'cursor':1720}}
   ref={'training_state_sha256':api.sha256_json(state),'bytes':10,'sha256':f'{seed:064x}'}
   cp={'path':'checkpoints/final.pt','reference':ref,'state':state}
   child['baseline_ledgers'][str(seed)]={'identity':{'seed':seed},'allocation':decl['allocations'][str(seed)],'active':False,'unresolved_work':False,'can_resume':True,'last_status':'paused','remaining_wall_seconds':100.,'charged':charged,'latest_checkpoint':cp}
   child['baseline_files'][f'seed-{seed}/checkpoints/final.pt']={'bytes':10,'sha256':ref['sha256']}
  with patch.object(api,'identity_for',side_effect=lambda d,s:{'seed':s}):
   api._evaluation_baselines(child,decl)
   for mutation in ('epoch','evaluation','execution','inventory'):
    bad=deepcopy(child);s=bad['baseline_ledgers']['43019']
    if mutation=='epoch':s['latest_checkpoint']['state']['extra_state']['completed_epochs']=39
    if mutation=='evaluation':s['charged']['evaluation_calls']=1
    if mutation=='execution':s['latest_checkpoint']['state']['extra_state']['execution']={'wrong':'proof'}
    if mutation=='inventory':bad['baseline_files'].pop('seed-43019/checkpoints/final.pt')
    with self.assertRaises(ValueError):api._evaluation_baselines(bad,decl)

 def test_parent_and_admission_body_tampering_cannot_reuse_original_file_hashes(self):
  original_parent_hash=api._canonical_file_hash(self.parent);original_review_hash=api._canonical_file_hash(self.admission)
  parent=deepcopy(self.parent);parent['source']['revision']='9'*40;parent['sha256']=api.sha256_json({k:v for k,v in parent.items() if k!='sha256'})
  review=deepcopy(self.admission);review['reviewers']=['different-one','different-two']
  with patch.object(api,'_declaration_binding',return_value={'source':{}}),patch.object(api,'_source_delta',return_value=self.parent['changed_source_files']),patch.object(api,'_baseline_unchanged'):
   with self.assertRaises(ValueError):api._validate_parent(self.root,parent,original_parent_hash,self.admission,original_review_hash)
   with self.assertRaises(ValueError):api._validate_parent(self.root,self.parent,original_parent_hash,review,original_review_hash)
 def test_frozen_candidate_proof_is_part_of_child_semantic_seal(self):
  child=self.child();child['frozen_candidates']={'schema':'nca-ik-frozen-candidates/v1','candidates':[{'seed':41011,'checkpoint_sha256':'1'*64}], 'sha256':'2'*64}
  child['sha256']=api.sha256_json({k:v for k,v in child.items() if k!='sha256'})
  api._amendment_record(child,evaluation=True)
  tampered=deepcopy(child);tampered['frozen_candidates']['candidates'][0]['checkpoint_sha256']='3'*64
  with self.assertRaises(ValueError):api._amendment_record(tampered,evaluation=True)

if __name__=='__main__':unittest.main()


# Copyright 2026 techrote. All rights reserved.
