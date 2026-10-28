"""CPU-only checks for the separate final qualification custody boundary."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from nca.runs import immutable_json,sha256_file
from nca.evaluation import ik54_frozen_final_v1 as final


class FrozenRecipeTests(unittest.TestCase):
    def test_new_recipe_dispatches_exact_final_schedule_and_namespaces(self):
        from nca.ik.data import iter_tasks,build_split_plan
        from nca.contracts import ContractError
        config=final.FrozenFinalConfig();tasks=tuple(iter_tasks(config))
        self.assertEqual(len(tasks),2940)
        self.assertEqual(sum(t.state.kind=='point' for t in tasks),2000)
        self.assertEqual(sum(t.state.kind=='trajectory' for t in tasks),800)
        self.assertEqual({t.partition for t in tasks},{'final'})
        self.assertTrue(all(t.family_id.startswith(config.family_namespace+':') for t in tasks))
        self.assertEqual(set(build_split_plan(config)['family_assignments']),{'final'})
        for task in tasks:
            if task.state.kind=='stress':continue
            bounds=config.bounds('final',task.state.bin_index)
            self.assertTrue(all(a<=q<=b for q,(a,b) in zip(task.target_configuration,bounds)))
        with self.assertRaises(ContractError):tuple(iter_tasks(config,'training'))

    def test_historical_recipe_geometry_identity_is_unchanged(self):
        from nca.train.ik_followup_seal import geometry_identity
        _,_,identity=geometry_identity()
        self.assertEqual(identity['ordered_geometry_sha256'],'57f7008cbaf6d776ad36e2aeec97019a68a8929e697a2734ea3fde2d7abe64c2')

    def test_continuous_guard_rejects_crossing_between_endpoints(self):
        from nca.train.ik_frozen_seal_v1 import separated
        with self.assertRaises(ValueError):separated([((-1.,0.),(1.,0.))],[((0.,-1.),(0.,1.))],.002)
        self.assertTrue(separated([((-1.,0.),(1.,0.))],[((2.,0.),(3.,0.))],.002)['pass'])

    def test_isolation_receipt_requires_both_complete_guards_and_original_identity(self):
        import copy
        from nca.train.ik_frozen_seal_v1 import validate_isolation,TRAIN_DECLARATION
        identity={'ordered_geometry_sha256':'a'*64};corpus={'original':'identity'}
        good={'schema':'nca-ik-frozen-final-isolation/v1','training_declaration_file_sha256':TRAIN_DECLARATION,
            'corpus_identity':corpus,'original_rows':7280,'new_rows':2940,'ordered_new_geometry_sha256':'a'*64,
            'family_trajectory_input_and_id_disjoint':True,'prior_suites_read':False,'models_or_candidate_outputs_used':False,
            'root_point_and_complete_path_guard':{'left_segments':8800,'right_segments':3700,'near_pairs_checked':0,
                'clearance_strictly_greater_than':.0005,'pass':True},
            'joint_point_and_complete_path_guard':{'left_segments':8760,'right_segments':3680,'near_pairs_checked':0,
                'clearance_strictly_greater_than':.002,'pass':True}}
        validate_isolation(good,identity,corpus)
        for key in good:
            bad=copy.deepcopy(good);bad.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):validate_isolation(bad,identity,corpus)
        bad=copy.deepcopy(good);bad['root_point_and_complete_path_guard']['clearance_strictly_greater_than']=.0001
        with self.assertRaises(ValueError):validate_isolation(bad,identity,corpus)

    def test_metadata_inspector_never_hashes_archive(self):
        from nca.train import ik_frozen_seal_v1 as custodian
        from nca.contracts import sha256_json
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'sealed-final.tar.gz').write_bytes(b'abc')
            d={'id':'fixture','sha256':'a'*64};geometry={'fixed':'fixture'}
            seal={'schema':'nca-ik-frozen-final-seal/v1','declaration_sha256':d['sha256'],'registry_id':d['id'],
                'geometry':geometry,'isolation_file_sha256':'b'*64,'rows':2940,'teacher_attempts':2940,
                'reserved_teacher_iterations':188160,'teacher_iterations':100,'peak_process_bytes':1024,
                'hard_memory_limit_bytes':512*1024**2,'memory_limit_method':'RLIMIT_AS','final_state':'sealed',
                'candidate_conditioned_filtering':False,'attempt_journal_sha256':'c'*64,'archive_bytes':3,'archive_sha256':'d'*64}
            seal['sha256']=sha256_json(seal)
            supervisor={'schema':'nca-ik-frozen-seal-supervisor/v1','complete':True,'returncode':0,'timed_out':False,
                'whole_process_seconds':10,'declaration_sha256':d['sha256'],'seal_sha256':seal['sha256'],
                'archive_file_sha256':seal['archive_sha256'],'retry_or_resume':False}
            observed={'exit_code':0,'seal_file_sha256':'e'*64,'supervisor_file_sha256':'f'*64,'failure_marker_absent':True}
            records={'seal.json':seal,'supervisor.json':supervisor,'geometry.json':geometry,'process-exit.json':observed}
            def file_hash(path):
                if Path(path).name=='sealed-final.tar.gz':raise AssertionError('archive read before opening')
                return {'isolation.json':'b'*64,'attempts.jsonl':'c'*64,'seal.json':'e'*64,'supervisor.json':'f'*64}[Path(path).name]
            with patch.object(custodian,'load',return_value=d),patch.object(custodian,'_read',side_effect=lambda p:records[Path(p).name]),patch.object(custodian,'sha256_file',side_effect=file_hash):
                self.assertEqual(custodian.inspect(root),seal)


class FrozenFinalAdmissionTests(unittest.TestCase):
    def test_report_size_limit_is_enforced_before_creation(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'report.json'
            with patch.object(final.historical,'REPORT_MAX_BYTES',8):
                with self.assertRaises(ValueError):final._write_report(path,{'text':'too large'})
            self.assertFalse(path.exists())

    def fixture(self, root):
        from nca.ik.declaration import runtime_expectation
        source={'revision':'a'*40,'source_sha256':'b'*64}
        frozen={'schema':'fixture','candidates':[{'checkpoint_sha256':str(i)*64} for i in (1,2,3)]}
        d={'schema':'nca-ik54-frozen-final-declaration/v1','id':'ik54-frozen-final-v1','source':source,'execution_authorized':False,
           'retry_or_resume':False,'frozen_candidates':frozen,'allocation':final.allocation(),
           'comparators':list(final.COMPARATOR_NAMES),'gpu_visibility_sha256':final.TITAN_FINGERPRINT,
           'acceptance_file_sha256':final.ACCEPTANCE_SHA,'final_state':'new_sealed_suite',
           'selection':'original_tuning_only_no_reselection','runtime':runtime_expectation(),
           'inference_route':final.inference_route(),'config':final.FrozenFinalConfig().to_dict()}
        immutable_json(root/'declaration.json',d);immutable_json(root/'frozen-candidates.json',frozen)
        digest=sha256_file(root/'declaration.json');reviews=[]
        for name in ('reviewer-a','reviewer-b'):
            p=root/(name+'.json');reviews.append(p)
            immutable_json(p,{'schema':'nca-ik54-frozen-final-admission/v1','head':source['revision'],
               'declaration_file_sha256':digest,'independent_non_author':True,'reviewer':name,
               'checks':{k:True for k in final.CHECKS},'verdict':'admit_frozen_final_qualification'})
        owner=root/'owner.json'
        immutable_json(owner,{'schema':'nca-ik54-frozen-final-owner/v1','declaration_file_sha256':digest,'execute_final':True})
        return d,reviews,owner,source

    def test_two_distinct_reviews_owner_and_exact_source_are_required(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);d,reviews,owner,source=self.fixture(root)
            with patch.object(final,'_source',return_value=source):
                self.assertEqual(final.check_admissions(root,reviews,owner),d)
                for bad in (reviews[:1],[reviews[0],reviews[0]]):
                    with self.assertRaises(ValueError):final.check_admissions(root,bad,owner)
                wrong=root/'wrong-owner.json';immutable_json(wrong,{'schema':'nca-ik54-frozen-final-owner/v1',
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
                    'wall_seconds':final.WALL_SECONDS,'attempt':1,'graph_construction_calls':16})
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
