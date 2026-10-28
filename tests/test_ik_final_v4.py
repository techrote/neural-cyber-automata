"""CPU-only checks for the separate final qualification custody boundary."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from nca.runs import immutable_json,sha256_file
from nca.evaluation import ik54_final_v1 as final


class FinalAdmissionTests(unittest.TestCase):
    def test_report_size_limit_is_enforced_before_creation(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'report.json'
            with patch.object(final,'REPORT_MAX_BYTES',8):
                with self.assertRaises(ValueError):final._write_report(path,{'text':'too large'})
            self.assertFalse(path.exists())

    def fixture(self, root):
        from nca.ik.declaration import runtime_expectation
        source={'revision':'a'*40,'source_sha256':'b'*64}
        frozen={'schema':'fixture','candidates':[{'checkpoint_sha256':str(i)*64} for i in (1,2,3)]}
        d={'schema':'nca-ik54-final-declaration/v1','id':'ik54-final-v1','source':source,'execution_authorized':False,
           'retry_or_resume':False,'frozen_candidates':frozen,'allocation':final.allocation(),
           'comparators':list(final.COMPARATOR_NAMES),'gpu_visibility_sha256':final.TITAN_FINGERPRINT,
           'acceptance_file_sha256':final.ACCEPTANCE_SHA,'final_state':'sealed_unverified_until_admitted_execution',
           'selection':'original_tuning_only_no_reselection','runtime':runtime_expectation()}
        immutable_json(root/'declaration.json',d);immutable_json(root/'frozen-candidates.json',frozen)
        digest=sha256_file(root/'declaration.json');reviews=[]
        for name in ('reviewer-a','reviewer-b'):
            p=root/(name+'.json');reviews.append(p)
            immutable_json(p,{'schema':'nca-ik54-final-admission/v1','head':source['revision'],
               'declaration_file_sha256':digest,'independent_non_author':True,'reviewer':name,
               'checks':{k:True for k in final.CHECKS},'verdict':'admit_final_qualification'})
        owner=root/'owner.json'
        immutable_json(owner,{'schema':'nca-ik54-final-owner/v1','declaration_file_sha256':digest,'execute_final':True})
        return d,reviews,owner,source

    def test_two_distinct_reviews_owner_and_exact_source_are_required(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);d,reviews,owner,source=self.fixture(root)
            with patch.object(final,'_source',return_value=source):
                self.assertEqual(final.check_admissions(root,reviews,owner),d)
                for bad in (reviews[:1],[reviews[0],reviews[0]]):
                    with self.assertRaises(ValueError):final.check_admissions(root,bad,owner)
                wrong=root/'wrong-owner.json';immutable_json(wrong,{'schema':'nca-ik54-final-owner/v1',
                    'declaration_file_sha256':'0'*64,'execute_final':True})
                with self.assertRaises(ValueError):final.check_admissions(root,reviews,wrong)
            with patch.object(final,'_source',return_value={**source,'revision':'c'*40}):
                with self.assertRaises(ValueError):final.check_admissions(root,reviews,owner)
            with patch.object(final,'_source',return_value=source), patch('nca.ik.declaration.runtime_expectation',return_value={}):
                with self.assertRaises(ValueError):final.check_admissions(root,reviews,owner)

    def test_allocation_and_comparator_tampering_refuses_before_reviews(self):
        for changed in ({'allocation':{**final.allocation(),'attempts':2}},
                        {'allocation':{**final.allocation(),'optimizer_calls':False}},
                        {'comparators':list(final.COMPARATOR_NAMES[:-1])}):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as name:
                root=Path(name);d,reviews,owner,source=self.fixture(root)
                original=final._bounded_json
                with patch.object(final,'_source',return_value=source), patch.object(final,'_bounded_json',wraps=original) as read:
                    def replace(path, maximum=final.REPORT_MAX_BYTES):
                        value=original(path,maximum)
                        return {**value,**changed} if Path(path).name=='declaration.json' else value
                    read.side_effect=replace
                    with self.assertRaises(ValueError):final.check_admissions(root,reviews,owner)
                    self.assertFalse(any(Path(c.args[0]).name=='reviewer-a.json' for c in read.call_args_list))

    def test_incomplete_reservations_refuse_before_suite_metadata(self):
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);suite=base/'suite';suite.mkdir();root=base/'evaluation';root.mkdir()
            declaration={'suite_root':str(suite),'suite_seal':{},'frozen_candidates':{}}
            immutable_json(root/'declaration.json',declaration);digest=sha256_file(root/'declaration.json')
            immutable_json(root/'started.json',{'declaration_file_sha256':digest})
            with patch.object(final,'check_admissions',return_value=declaration), patch.object(final,'verify_seal_metadata') as verify:
                with self.assertRaises((ValueError,FileNotFoundError)):final.open_suite(suite,root,[],None,{})
                verify.assert_not_called();self.assertEqual(list(suite.iterdir()),[])

    def test_unadmitted_metadata_never_touches_suite(self):
        with tempfile.TemporaryDirectory() as name:
            suite=Path(name)/'sealed';suite.mkdir()
            with patch.object(final,'check_admissions',side_effect=ValueError('denied')):
                with self.assertRaisesRegex(ValueError,'denied'):
                    final.verify_seal_metadata(suite,{},evaluation_root=Path(name)/'evaluation',reviews=[],owner=None)
            self.assertEqual(list(suite.iterdir()),[])

    def test_unadmitted_opener_never_touches_suite(self):
        with tempfile.TemporaryDirectory() as name:
            suite=Path(name)/'sealed';suite.mkdir()
            with patch.object(final,'check_admissions',side_effect=ValueError('denied')):
                with self.assertRaisesRegex(ValueError,'denied'):
                    final.open_suite(suite,Path(name)/'evaluation',[],None,{})
            self.assertEqual(list(suite.iterdir()),[])

    def test_opening_is_durable_before_archive_read_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);suite=base/'suite';suite.mkdir();root=base/'evaluation';root.mkdir()
            digest=sha256_file(root/'declaration.json') if (root/'declaration.json').exists() else None
            # Only a synthetic invalid archive is used; its deliberately bad hash
            # makes the first attempt stop after the durable single-use marker.
            (suite/'sealed-final.tar.gz').write_bytes(b'fixture')
            frozen={'experiment':'fixture','candidates':[{'checkpoint_sha256':str(i)*64} for i in (1,2,3)]}
            seal={'archive_sha256':'0'*64,'sha256':'a'*64,'registry_id':'fixture','archive_bytes':7}
            declaration={'suite_root':str(suite),'suite_seal':seal,'frozen_candidates':frozen}
            immutable_json(root/'declaration.json',declaration);digest=sha256_file(root/'declaration.json')
            immutable_json(root/'started.json',{'declaration_file_sha256':digest})
            for seed in final.SEEDS:
                immutable_json(root/f'reservation-{seed}.json',{'seed':seed,'declaration_file_sha256':digest,
                    'evaluation_calls':final.EVALUATIONS_PER_SEED,'optimizer_calls':0,
                    'wall_seconds':final.WALL_SECONDS,'attempt':1})
            with patch.object(final,'check_admissions',return_value=declaration), patch.object(final,'verify_seal_metadata',return_value=seal):
                with self.assertRaisesRegex(ValueError,'archive bytes differ'):
                    final.open_suite(suite,root,[],None,seal)
                self.assertTrue((suite/'opened.json').is_file())
                with self.assertRaises(ValueError):final.open_suite(suite,root,[],None,seal)

    def test_predictor_cap_prevents_excess_underlying_work(self):
        actual=[]
        wrapped,counts=final._counter_predictors({'fixture':lambda x:actual.append(x)},1)
        wrapped['fixture']('first')
        with self.assertRaisesRegex(ValueError,'cap exceeded'):wrapped['fixture']('second')
        self.assertEqual(actual,['first']);self.assertEqual(counts['total'],2)

    def test_output_must_be_new_and_disjoint_from_inputs(self):
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);source=base/'source';source.mkdir()
            with self.assertRaises(ValueError):final._external(source/'child',source)
            with self.assertRaises(ValueError):final._external(base,source)
            self.assertEqual(final._external(base/'new',source), (base/'new').resolve())


if __name__=='__main__':unittest.main()

# Copyright 2026 techrote. All rights reserved.
