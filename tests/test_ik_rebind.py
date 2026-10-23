"""Recovery rebind lineage and fail-closed preflight tests; no real allocation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from nca.evaluation import ik_recovery as api
from nca.contracts import canonical_json


class RebindTests(unittest.TestCase):
    def test_refusal_pin_rejects_unrelated_observation_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'observation.json'
            p.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'exact independently audited'):
                api._refusal(p)
            self.assertEqual(list(Path(folder).iterdir()), [p])

    def test_rebind_new_directory_and_exact_parent_required_before_debit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); existing = root/'exists'; existing.mkdir()
            with patch.object(api, '_project', return_value=root/'source'), patch.object(api, 'load') as load:
                with self.assertRaisesRegex(ValueError, 'new external'):
                    api.rebind(root/'run','parent','wrong','failed','hash','corpus','registry','refusal',existing)
                with self.assertRaisesRegex(ValueError, 'exact refused parent'):
                    api.rebind(root/'run','parent','wrong','failed','hash','corpus','registry','refusal',root/'new')
                load.assert_not_called()
                self.assertFalse((root/'new').exists())

    def test_historical_source_exception_is_only_exact_refused_parent(self):
        with patch.object(api, '_sha') as read:
            with self.assertRaisesRegex(ValueError, 'exact refused parent'):
                api.load('run','parent','0'*64,'failed','hash','registry',_historical=True)
            read.assert_not_called()

    def test_original_paid_intent_is_copied_byte_exact_before_debit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); old=root/'old';old.mkdir();new=root/'new';new.mkdir()
            raw=b'{ "original": true }\r\n'
            (old/'recovery-intent.json').write_bytes(raw)
            api._copy_paid_intent(old/'amendment.json', {'recovery_intent':{'original':True}}, new)
            self.assertEqual((new/'recovery-intent.json').read_bytes(),raw)
            with self.assertRaises(FileExistsError):
                api._copy_paid_intent(old/'amendment.json', {'recovery_intent':{'original':True}}, new)
            with self.assertRaisesRegex(ValueError,'paid authority changed'):
                api._copy_paid_intent(old/'amendment.json', {'recovery_intent':{'changed':True}}, new)

    def fixture(self, root):
        parent = {k:{} for k in api._FIELDS}
        parent.update(sha256='a'*64, source={'source_sha256': api.REBIND_PARENT_SOURCE_SHA256})
        parent['baseline_ledgers'] = {str(s):{'parent_sha256':'b'*64,'attempt_count':2,
            'charged':{'evaluation_calls':0},'charged_wall_seconds':200.,'latest_checkpoint':{'same':True}}
            for s in api.SEEDS}
        a = deepcopy(parent); a['source']={'source_sha256':'c'*64}
        intent=api._sealed({'schema':'nca-ik-recovery-rebind-intent/v1','source':a['source'],
            'parent_file_sha256':api.REBIND_PARENT_FILE_SHA256,'refusal_files':{},'debit_seconds':10})
        r={'parent_amendment':'parent','parent_file_sha256':api.REBIND_PARENT_FILE_SHA256,
            'parent_sha256':parent['sha256'],'parent_source_sha256':api.REBIND_PARENT_SOURCE_SHA256,
            'refusal_observation':'refusal','intent':intent,'overhead_proofs':{}}
        a['rebind']=r
        for s in api.SEEDS:
            proof=api._sealed({'source_sha256':'c'*64,'prior_source_sha256':api.REBIND_PARENT_SOURCE_SHA256,
                'prior_amendment_file_sha256':api.REBIND_PARENT_FILE_SHA256,'prior_amendment_sha256':parent['sha256'],
                'authority_sha256':intent['sha256'],'prior_parent_sha256':'b'*64,
                'historical_charge_wall_seconds':10,'refusal_files':{}})
            r['overhead_proofs'][str(s)]=proof
            a['baseline_ledgers'][str(s)]['charged_wall_seconds']+=70
            p=root/f'seed-{s}'/'overheads'/'charge-0001.json';p.parent.mkdir(parents=True)
            p.write_text(canonical_json({'record':{'proof':proof}}))
        return parent,a

    def test_rebind_preserves_paid_authority_and_requires_durable_debit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); parent,a=self.fixture(root)
            with patch.object(api,'load',return_value=SimpleNamespace(amendment=parent)), patch.object(api,'_refusal',return_value={}):
                api._validate_rebind(root,'registry',a,'failed','hash')
                for key in ('allocations','ledger_recoveries','recovery_intent','frozen_candidates','opening'):
                    bad=deepcopy(a);bad[key]={'tampered':True}
                    with self.assertRaisesRegex(ValueError,'paid history'):
                        api._validate_rebind(root,'registry',bad,'failed','hash')
                bad=deepcopy(a);bad['baseline_ledgers']['41011']['charged']['evaluation_calls']=1
                with self.assertRaisesRegex(ValueError,'reset accounting'):
                    api._validate_rebind(root,'registry',bad,'failed','hash')
                p=root/'seed-41011/overheads/charge-0001.json';p.write_text('{}')
                with self.assertRaisesRegex(ValueError,'durable launcher charge'):
                    api._validate_rebind(root,'registry',a,'failed','hash')

    def test_rebind_cannot_recurse_or_replace_refused_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);parent,a=self.fixture(root)
            with patch.object(api,'load',return_value=SimpleNamespace(amendment=parent)), patch.object(api,'_refusal',return_value={}):
                bad=deepcopy(a);bad['rebind']['parent_source_sha256']='d'*64
                with self.assertRaisesRegex(ValueError,'exact refused'):
                    api._validate_rebind(root,'registry',bad,'failed','hash')
                parent['rebind']={}
                with self.assertRaisesRegex(ValueError,'recursively'):
                    api._validate_rebind(root,'registry',a,'failed','hash')

if __name__=='__main__': unittest.main()

# Copyright 2026 techrote. All rights reserved.
