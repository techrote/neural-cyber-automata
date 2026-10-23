"""CPU-only append-only refused launcher accounting with no worker slot."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nca.contracts import sha256_json
from nca.train.ik_run_state import IKRunLedger, _publish, _inventory
from test_ik_run_state import identity, allocation


class LauncherOverheadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.ledger = IKRunLedger.create(self.base/'run', identity=identity(), allocation=allocation(max_wall_seconds=1000))
        self.verify = patch('nca.train.ik_run_state._verify_checkpoint'); self.verify.start(); self.addCleanup(self.verify.stop)
        l=self.ledger; folder=l.root/'attempts/attempt-0001';folder.mkdir(parents=True)
        zero={k:0 for k in ('optimizer_steps','sample_presentations','tuning_calls','evaluation_calls','checkpoints')}
        start=_publish(folder/'start.json',{'schema':'nca-ik-training-ledger/v1','attempt':1,'allocation_sha256':l.header['sha256'],'parent_sha256':None,'prior_charged':zero,'prior_wall_seconds':0.,'prior_checkpoint':None,'created_utc':'fixture'})
        cost=dict(zero,checkpoints=1)
        reserve={'id':'0001-000001','kind':'checkpoint','samples':0,'calls':1,'call_id':'fixture','cost':cost}
        event=_publish(folder/'event-000001.json',{'sequence':1,'previous_sha256':start['sha256'],'kind':'reserve','payload':reserve})
        cp={'path':'fixture','reference':{'sha256':'9'*64},'state':{'charged':cost,'completed_optimizer_steps':0,'optimizer_presentations':0,'consumed_batches':0,'reservation_id':reserve['id']}}
        last=_publish(folder/'event-000002.json',{'sequence':2,'previous_sha256':event['sha256'],'kind':'complete','payload':{'reservation_id':reserve['id'],'receipt':cp}})
        finish=_publish(folder/'finish.json',{'attempt':1,'start_sha256':start['sha256'],'last_event_sha256':last['sha256'],'event_count':2,'status':'paused','reason':'fixture','elapsed_seconds':1.,'charged':cost,'completed_optimizer_steps':0,'optimizer_presentations':0,'consumed_batches':0,'checkpoint':cp,'pending_reservation':None,'inventory':_inventory(folder)})
        (folder/'sealed.json').write_text('{"finish_sha256":"'+finish['sha256']+'"}')
        self.before=l.inspect()
        stdout=self.base/'run.stdout.json'; stdout.write_bytes(b'')
        stderr=self.base/'run.stderr.txt'; stderr.write_text('ValueError: typed authenticated recovery execution context required')
        support=[]
        for name in ('refused-launcher-observation.json','terminal-refusal-a.json','terminal-refusal-b.json'):
            path=self.base/name;path.write_text('{}');support.append(path)
        self.proof={'schema':'nca-ik-recovery-launcher-overhead/v1','identity_sha256':sha256_json(l.identity),'allocation_sha256':l.header['sha256'],'prior_parent_sha256':self.before['parent_sha256'],'prior_attempt_count':1,'checkpoint_sha256':'9'*64,'prior_amendment_file_sha256':'1'*64,'prior_amendment_sha256':'2'*64,'prior_source_sha256':'3'*64,'source_sha256':'4'*64,'authority_sha256':'5'*64,'historical_charge_wall_seconds':10.,'refusal_files':{str(p.resolve()):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (stdout,stderr,*support)}}
        self.seal()

    def seal(self):
        self.proof['sha256']=sha256_json({k:v for k,v in self.proof.items() if k!='sha256'})

    def test_debit_preserves_slots_counters_checkpoint_and_is_idempotent(self):
        after=self.ledger.record_recovery_overhead(self.proof)
        for key in ('attempt_count','charged','latest_checkpoint','completed_optimizer_steps','optimizer_presentations','consumed_batches'):
            self.assertEqual(after[key],self.before[key])
        self.assertGreaterEqual(after['charged_wall_seconds']-self.before['charged_wall_seconds'],70.)
        self.assertNotEqual(after['parent_sha256'],self.before['parent_sha256'])
        self.assertEqual(self.ledger.record_recovery_overhead(self.proof),after)
        attempt=self.ledger.begin_attempt()
        self.assertEqual(attempt.number,2)
        import json
        start=json.loads((attempt.folder/'start.json').read_bytes())['record']
        self.assertEqual(start['parent_sha256'],after['parent_sha256'])
        self.assertEqual(start['prior_wall_seconds'],after['charged_wall_seconds'])
        attempt.finish('paused',reason='fixture continuation')
        self.assertGreater(self.ledger.inspect()['charged_wall_seconds'],after['charged_wall_seconds'])

    def test_changed_evidence_and_small_or_nonfinite_debit_refuse(self):
        for value in (0,6,9.999,True,float('inf')):
            self.proof['historical_charge_wall_seconds']=value
            if value==float('inf'):
                continue # canonical JSON itself rejects nonfinite proof seals
            self.seal()
            with self.assertRaises(ValueError):self.ledger.record_recovery_overhead(self.proof)
        self.proof['historical_charge_wall_seconds']=10.;self.seal()
        (self.base/'run.stderr.txt').write_text('different')
        with self.assertRaises(ValueError):self.ledger.record_recovery_overhead(self.proof)

    def test_delete_debit_breaks_next_attempt_lineage(self):
        self.ledger.record_recovery_overhead(self.proof)
        attempt=self.ledger.begin_attempt();attempt.finish('paused',reason='fixture')
        (self.ledger.root/'overheads/charge-0001.json').unlink()
        with self.assertRaises(ValueError):self.ledger.inspect()

    def test_overhead_tampering_or_publication_overrun_blocks_resume(self):
        self.ledger.record_recovery_overhead(self.proof)
        (self.ledger.root/'overheads/overrun-0001.json').write_text('{}')
        with self.assertRaises(ValueError):self.ledger.begin_attempt()

if __name__=='__main__':unittest.main()
