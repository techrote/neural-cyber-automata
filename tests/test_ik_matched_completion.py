"""Independent completion/lifecycle CPU mocks; no hardware or learner calls."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from nca.runs import immutable_json
from nca.train import ik_matched_supervisor as supervisor
from nca.train.ik_matched_compare import compare, state_hash


class CompletionReviewTests(unittest.TestCase):
    def test_comparison_rejects_every_required_fingerprint_divergence(self):
        evidence={str(s):{'checkpoints':['initial','selected','final'],
            'tuning':'tuning-hash','sampler_schedule':{'seed':s,'cursor':0},
            'ordered_receipts':'receipt-hash'} for s in supervisor.SEEDS}
        self.assertTrue(compare(evidence,deepcopy(evidence))['qualified_for_consistent_parallel'])
        for seed in evidence:
            for field in evidence[seed]:
                modified=deepcopy(evidence);modified[seed][field]='changed'
                with self.subTest(seed=seed,field=field):
                    verdict=compare(evidence,modified)
                    self.assertFalse(verdict['exact_match'])
                    self.assertFalse(verdict['qualified_for_consistent_parallel'])

    def test_state_hash_preserves_types_and_ignores_mapping_insertion_order(self):
        a={'optimizer':{0:{'step':1720,'moment':[.2,.3]}},'rng':(1,[2,3]),'sampler':{'cursor':1720}}
        self.assertEqual(state_hash(a),state_hash(dict(reversed(list(a.items())))))
        for changed in [dict(a,rng=[1,[2,3]]),dict(a,optimizer={'0':{'step':1720,'moment':[.2,.3]}}),dict(a,sampler={'cursor':True})]:
            self.assertNotEqual(state_hash(a),state_hash(changed))
        self.assertNotEqual(state_hash(True),state_hash(1))

    def test_peer_completed_allows_other_untouched_mode_in_both_directions(self):
        for mode in ('serial','parallel'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);peer='parallel' if mode=='serial' else 'serial'
                fingerprints={'verified':'same'}
                def read(path,**kw):
                    path=Path(path)
                    if path.name=='packet.json':return {'sha256':'a'*64}
                    if path.name=='declaration.json':return {'mode':path.parent.name}
                    if path.name=='group-result.json':return {'verified':True,'final_opened':False,'packet_sha256':'a'*64,'fingerprints':fingerprints}
                    raise AssertionError(path)
                def ledger(run,declaration,seed):
                    return SimpleNamespace(inspect=lambda:{'active':False,'unresolved_work':False,'attempt_count':0 if declaration['mode']==mode else 1})
                with patch.object(supervisor,'_read',side_effect=read),patch.object(supervisor,'ledger',side_effect=ledger),patch('nca.train.ik_matched_compare.evidence',return_value=fingerprints):
                    supervisor.validate_lifecycle(root,mode)
                with patch.object(supervisor,'_read',side_effect=read),patch.object(supervisor,'ledger',side_effect=ledger),patch('nca.train.ik_matched_compare.evidence',return_value={'changed':True}):
                    with self.assertRaisesRegex(ValueError,'peer completion'):
                        supervisor.validate_lifecycle(root,mode)

    def test_lifecycle_refusal_after_group_claim_pays_each_unstarted_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'serial').mkdir()
            immutable_json(root/'authorization.json',{'schema':'nca-ik-matched-owner-execution/v1','packet_sha256':'a'*64,'modes':['serial','parallel'],'execute':True})
            immutable_json(root/'serial/declaration.json',{'topology':{'max_workers':1}})
            ledgers={}
            for seed in supervisor.SEEDS:
                seedroot=root/f'serial/seed-{seed}';seedroot.mkdir()
                obj=Mock(root=seedroot);obj.inspect.return_value={'attempt_count':0};ledgers[seed]=obj
            with patch.object(supervisor,'admission',return_value={'sha256':'a'*64}),patch.object(supervisor,'board_sample',return_value={}),patch.object(supervisor,'validate_lifecycle',side_effect=ValueError('prebegin-failure')),patch.object(supervisor,'ledger',side_effect=lambda r,d,s:ledgers[s]),patch.object(supervisor.subprocess,'Popen') as spawn,patch('nca.train.ik_gpu_ownership.acquire'),patch('nca.train.ik_gpu_ownership.release'),patch('nca.ik.process_limits.memory_limit'):
                with self.assertRaisesRegex(ValueError,'prebegin-failure'):
                    supervisor.supervise(root,'serial','unused','unused',[],root/'authorization.json')
                spawn.assert_not_called()
            self.assertTrue((root/'group-active.json').exists())
            self.assertEqual(len(list(root.glob('group-failure-*.json'))),1)
            for ledger in ledgers.values():
                ledger.begin_attempt.assert_called_once()
                ledger.begin_attempt.return_value.finish.assert_called_once_with('failed',reason='group_startup_failure',matched_coordination_charge=True)



    def test_one_broken_ledger_does_not_skip_other_failure_charges_or_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'serial').mkdir()
            immutable_json(root/'authorization.json',{'schema':'nca-ik-matched-owner-execution/v1','packet_sha256':'a'*64,'modes':['serial','parallel'],'execute':True})
            immutable_json(root/'serial/declaration.json',{'topology':{'max_workers':1}})
            ledgers={}
            for seed in supervisor.SEEDS:
                obj=Mock();obj.inspect.return_value={'attempt_count':0};ledgers[seed]=obj
            ledgers[41011].inspect.side_effect=ValueError('damaged-ledger')
            with patch.object(supervisor,'admission',return_value={'sha256':'a'*64}),patch.object(supervisor,'board_sample',return_value={}),patch.object(supervisor,'validate_lifecycle',side_effect=ValueError('prebegin-failure')),patch.object(supervisor,'ledger',side_effect=lambda r,d,s:ledgers[s]),patch.object(supervisor.subprocess,'Popen') as spawn,patch('nca.train.ik_gpu_ownership.acquire'),patch('nca.train.ik_gpu_ownership.release'),patch('nca.ik.process_limits.memory_limit'):
                with self.assertRaises(ValueError):
                    supervisor.supervise(root,'serial','unused','unused',[],root/'authorization.json')
                spawn.assert_not_called()
            self.assertTrue((root/'group-active.json').exists())
            failures=list(root.glob('group-failure-*.json'))
            self.assertEqual(len(failures),1)
            import json
            receipt=json.loads(failures[0].read_bytes())
            self.assertTrue(receipt['unknown_work_blocks_reuse'])
            self.assertTrue(receipt['accounting_errors'])
            ledgers[41011].begin_attempt.assert_not_called()
            for seed in (42013,43019):
                ledgers[seed].begin_attempt.assert_called_once()
                ledgers[seed].begin_attempt.return_value.finish.assert_called_once()



    def report_history_fixture(self,root):
        from nca.contracts import sha256_json
        from nca.runs import sha256_file
        reports=[];history=[]
        for epoch in range(1,41):
            report={'selection_metric':float(41-epoch),'epoch_fixture':epoch}
            name=f'tuning-epoch-{epoch:04d}.json'
            immutable_json(root/name,report)
            reports.append(report)
            history.append({'epoch':epoch,'step':epoch*43,'file':name,
                'file_sha256':sha256_file(root/name),'report_sha256':sha256_json(report)})
        status={'latest_checkpoint':{'state':{'extra_state':{'tuning_history':history}}}}
        return SimpleNamespace(root=root,inspect=lambda:status),reports,history

    def test_all_forty_report_identities_match_authenticated_checkpoint_history(self):
        from nca.train.ik_matched_compare import authenticated_reports
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bound,reports,history=self.report_history_fixture(root)
            with patch.object(supervisor,'ledger',return_value=bound):
                self.assertEqual(authenticated_reports(root,{},41011),reports)
                for field,value in [('epoch',39),('step',1719),('file','other.json')]:
                    old=history[-1][field];history[-1][field]=value
                    with self.subTest(field=field),self.assertRaisesRegex(ValueError,'cadence'):
                        authenticated_reports(root,{},41011)
                    history[-1][field]=old

    def test_report_substitution_cannot_become_authentic_by_rehashing_peer_fingerprint(self):
        import json
        from nca.contracts import sha256_json
        from nca.runs import sha256_file
        from nca.train.ik_matched_compare import authenticated_reports
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bound,reports,history=self.report_history_fixture(root)
            path=root/'tuning-epoch-0040.json';original=path.read_bytes()
            modified=dict(reports[-1],selection_metric=0.)
            path.write_text(json.dumps(modified),encoding='utf-8')
            recomputed_peer_fingerprint=state_hash(reports[:-1]+[modified])
            self.assertNotEqual(recomputed_peer_fingerprint,state_hash(reports))
            with patch.object(supervisor,'ledger',return_value=bound):
                with self.assertRaisesRegex(ValueError,'checkpoint history'):
                    authenticated_reports(root,{},41011)
                history[-1]['file_sha256']=sha256_file(path)
                with self.assertRaisesRegex(ValueError,'checkpoint history'):
                    authenticated_reports(root,{},41011)
                path.write_bytes(original)
                history[-1]['file_sha256']=sha256_file(path)
                history[-1]['report_sha256']=sha256_json(modified)
                with self.assertRaisesRegex(ValueError,'checkpoint history'):
                    authenticated_reports(root,{},41011)


if __name__=='__main__':unittest.main()
