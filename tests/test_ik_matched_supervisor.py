"""Independent adversarial CPU mocks; never launch training or query hardware."""
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from nca.train import ik_matched_supervisor as supervisor
from nca.runs import immutable_json
from nca.contracts import sha256_json
from nca.train.ik_run_state import IKRunLedger
from nca.train.ik_matched_prepare import candidate_config, allocation


class MatchedSupervisorReviewTests(unittest.TestCase):
    def setUp(self):
        for target in ('nca.train.ik_gpu_ownership.acquire',
                       'nca.train.ik_gpu_ownership.release',
                       'nca.train.ik_gpu_ownership.verify',
                       'nca.ik.process_limits.memory_limit'):
            mocked=patch(target);mocked.start();self.addCleanup(mocked.stop)

    def test_unadmitted_cli_run_and_private_worker_cannot_launch(self):
        with patch.object(sys,'argv',['matched','run','--packet','unused']), \
                patch.object(supervisor,'admission',side_effect=ValueError('fresh admissions required')), \
                patch.object(supervisor.subprocess,'Popen') as launch, \
                patch.object(supervisor,'board_sample') as board:
            with self.assertRaisesRegex(ValueError,'fresh admissions required'):
                supervisor.main()
            launch.assert_not_called();board.assert_not_called()
        with patch.object(sys,'argv',['matched','_worker','--packet','unused','--mode','parallel','--claim','fixture']), \
                patch.object(supervisor,'_read',return_value={'claim':'other','mode':'parallel'}), \
                patch.object(supervisor,'_titan_visibility') as device:
            with self.assertRaisesRegex(ValueError,'live group ownership'):
                supervisor.main()
            device.assert_not_called()

    def test_import_is_runtime_free_in_separate_process(self):
        result = subprocess.run([sys.executable, '-c',
            "import sys; import nca.train.ik_matched_supervisor; "
            "assert 'torch' not in sys.modules and 'numpy' not in sys.modules"],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_watchdog_rejects_unavailable_nonfinite_and_over_budget_metadata(self):
        for raw in ('N/A, 1, 0, 30', 'nan, 1, 0, 30', '-1, 1, 0, 30',
                    '1, 1, 101, 30', '20481, 1, 0, 30', '1, 1, 0'):
            with self.subTest(raw=raw), \
                    patch.object(supervisor, '_titan_visibility'), \
                    patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-fixture'), \
                    patch.object(supervisor.subprocess, 'run', return_value=SimpleNamespace(stdout=raw)):
                with self.assertRaises(ValueError):
                    supervisor.board_sample()

    def test_watchdog_queries_only_the_isolated_target(self):
        with patch.object(supervisor, '_titan_visibility') as visibility, \
                patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-fixture'), \
                patch.object(supervisor.subprocess, 'run', return_value=SimpleNamespace(
                    stdout='3072, 21504, 10, 30')) as query:
            sample = supervisor.board_sample()
        visibility.assert_called_once()
        argv = query.call_args.args[0]
        self.assertEqual(argv[argv.index('-i') + 1], 'GPU-fixture')
        self.assertEqual(sample['used_bytes'], 3 * supervisor.GIB)

    def test_missing_owner_execution_instruction_cannot_launch(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(supervisor, 'admission', return_value={'sha256':'a'*64}), \
                patch.object(supervisor, '_read', return_value={'execute':False}), \
                patch.object(supervisor, 'board_sample') as board, \
                patch.object(supervisor.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'owner execution instruction'):
                supervisor.supervise(directory, 'parallel', 'unused', 'unused', [], 'unused')
            board.assert_not_called()
            launch.assert_not_called()

    def test_stale_worker_claim_refuses_before_source_and_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            immutable_json(root/'group-active.json', {'claim':'other', 'mode':'parallel'})
            args = SimpleNamespace(packet=directory, mode='parallel', claim='expected')
            with patch.object(supervisor, 'prepared_source') as source, \
                    patch.object(supervisor, '_titan_visibility') as target:
                with self.assertRaisesRegex(ValueError, 'live group ownership'):
                    supervisor.worker(args)
                source.assert_not_called()
                target.assert_not_called()

    def test_cli_children_use_canonical_module_and_launch_failure_retains_hold(self):
        class LaunchRefused(RuntimeError):
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'serial').mkdir()
            for seed in supervisor.SEEDS:
                (root/'serial'/f'seed-{seed}').mkdir()
            packet = {'sha256':'a'*64}
            auth = dict(schema='nca-ik-matched-owner-execution/v1',
                        packet_sha256=packet['sha256'],modes=['serial','parallel'],execute=True)
            immutable_json(root/'authorization.json',auth)
            immutable_json(root/'serial'/'declaration.json',{'topology':{'max_workers':1}})
            state = dict(attempt_count=0,active=False,unresolved_work=False)
            with patch.object(supervisor,'__name__','__main__'), \
                    patch.object(supervisor,'admission',return_value=packet), \
                    patch.object(supervisor,'board_sample',return_value={'used_bytes':0}), \
                    patch.object(supervisor,'validate_lifecycle'), \
                    patch.object(supervisor,'ledger',return_value=Mock(inspect=lambda:state)), \
                    patch.object(supervisor.subprocess,'Popen',side_effect=LaunchRefused('fixture')) as launch:
                with self.assertRaises(LaunchRefused):
                    supervisor.supervise(root,'serial','unused','unused',[],root/'authorization.json')
            argv=launch.call_args.args[0]
            self.assertEqual(argv[argv.index('-m')+1],'nca.train.ik_matched_supervisor')
            self.assertTrue((root/'group-active.json').exists())
            self.assertEqual(len(list(root.glob('group-failure-*.json'))),1)

    def test_private_worker_cannot_turn_false_owner_authorization_into_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'parallel').mkdir()
            immutable_json(root/'authorization.json',{'execute':False})
            from nca.runs import sha256_file
            immutable_json(root/'group-active.json',dict(claim='fixture',mode='parallel',
                packet_sha256='a'*64,reviews=[],authorization=str(root/'authorization.json'),
                authorization_sha256=sha256_file(root/'authorization.json')))
            immutable_json(root/'parallel'/'declaration.json',{'source':{'fixture':True}})
            args=SimpleNamespace(packet=directory,mode='parallel',claim='fixture')
            with patch.object(supervisor,'admission',return_value={'sha256':'a'*64}), \
                    patch.object(supervisor,'prepared_source',return_value={'fixture':True}), \
                    patch.object(supervisor,'_titan_visibility') as hardware:
                with self.assertRaisesRegex(ValueError,'owner execution instruction'):
                    supervisor.worker(args)
                hardware.assert_not_called()

    def test_fixed_coordination_debit_is_explicit_and_legacy_allocation_refuses(self):
        identity={key:'a'*64 for key in ('declaration_sha256','config_sha256',
            'source_sha256','profile_sha256','dataset_sha256','normalization_sha256','runtime_sha256')}
        identity['seed']=41011
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            legacy=IKRunLedger.create(root/'legacy',identity=identity,allocation=allocation())
            attempt=legacy.begin_attempt()
            with self.assertRaisesRegex(ValueError,'matched coordination'):
                attempt.finish('failed',reason='fixture',matched_coordination_charge=True)
            self.assertFalse((attempt.folder/'finish.json').exists())
            attempt.finish('failed',reason='fixture_ordinary_failure')
            matched=IKRunLedger.create(root/'matched',
                identity=identity|{'config_sha256':sha256_json(candidate_config())},allocation=allocation())
            current=matched.begin_attempt()
            result=current.finish('failed',reason='fixture_conservative_group_charge',
                                  matched_coordination_charge=True)
            self.assertGreaterEqual(result['charged_wall_seconds'],7195.)
            self.assertLess(result['charged_wall_seconds'],7200.)
            self.assertEqual(result['attempt_count'],1)
            self.assertEqual(result['charged']['optimizer_steps'],0)

    def test_numeric_one_is_not_an_affirmative_independent_review(self):
        from nca.runs import sha256_file
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=dict(revision='b'*40,source_sha256='a'*64)
            hashes={}
            for mode in ('serial','parallel'):
                (root/mode).mkdir()
                path=root/mode/'declaration.json'
                immutable_json(path,dict(source=source,candidate=candidate_config(),
                    allocations={str(s):allocation() for s in supervisor.SEEDS}))
                hashes[mode]=sha256_file(path)
            qualification={'root':'fixture','result_file_sha256':'c'*64}
            packet=dict(source_sha256=source['source_sha256'],declaration_file_sha256=hashes,
                        resource_qualification=qualification)
            packet['sha256']=sha256_json(packet)
            immutable_json(root/'packet.json',packet)
            checks=dict.fromkeys(('exact_head_ci','corpus_and_final_exclusion',
                'accounting_and_stop','supervisor_and_watchdog','aggregate_resource_feasibility'),True)
            reviews=[]
            for index in range(2):
                path=root/f'reviewer-{index}.json'
                bad=dict(checks);bad['supervisor_and_watchdog']=1
                immutable_json(path,dict(schema='nca-ik-matched-admission/v1',
                    head=source['revision'],packet_sha256=packet['sha256'],
                    source_sha256=source['source_sha256'],modes=['serial','parallel'],
                    gpu_visibility_sha256=supervisor.TITAN_FINGERPRINT,
                    verdict='admit_both_matched_trials',independent_non_author=True,
                    reviewer=f'fixture-{index}',checks=bad))
                reviews.append(path)
            with patch.object(supervisor,'prepared_source',return_value=source), \
                    patch('nca.train.ik_memory_probe.qualification_record',return_value=qualification):
                with self.assertRaisesRegex(ValueError,'independent TITAN admission'):
                    supervisor.admission(root,reviews)

    def test_numeric_one_is_not_owner_execution_authority(self):
        auth=dict(schema='nca-ik-matched-owner-execution/v1',packet_sha256='a'*64,
                  modes=['serial','parallel'],execute=1)
        with self.assertRaisesRegex(ValueError,'owner execution instruction'):
            supervisor.validate_authorization(auth,'a'*64)

    def test_watchdog_failure_stops_every_child_even_when_first_cleanup_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'parallel';run.mkdir()
            for seed in supervisor.SEEDS:(run/f'seed-{seed}').mkdir()
            packet={'sha256':'a'*64}
            immutable_json(root/'authorization.json',dict(schema='nca-ik-matched-owner-execution/v1',
                packet_sha256=packet['sha256'],modes=['serial','parallel'],execute=True))
            immutable_json(run/'declaration.json',{'topology':{'max_workers':3}})
            children=[Mock(returncode=None,poll=lambda:None) for _ in supervisor.SEEDS]
            board=[{'used_bytes':0},{'used_bytes':0},ValueError('lost board telemetry')]
            with patch.object(supervisor,'admission',return_value=packet), \
                    patch.object(supervisor,'board_sample',side_effect=board), \
                    patch.object(supervisor,'validate_lifecycle'), \
                    patch.object(supervisor,'ledger',return_value=Mock(inspect=lambda:dict(
                        attempt_count=0,active=False,unresolved_work=False))), \
                    patch.object(supervisor.subprocess,'Popen',side_effect=children), \
                    patch.object(supervisor.time,'sleep'), \
                    patch.object(supervisor,'stop_tree',side_effect=[TimeoutError('fixture'),None,None]) as stop:
                with self.assertRaisesRegex(ValueError,'lost board telemetry'):
                    supervisor.supervise(root,'parallel','unused','unused',[],root/'authorization.json')
            self.assertEqual([x.args[0] for x in stop.call_args_list],children)
            self.assertTrue((root/'group-active.json').exists())
            failure=supervisor._read(next(root.glob('group-failure-*.json')))
            self.assertTrue(failure['unknown_work_blocks_reuse'])
            self.assertEqual(failure['cleanup_errors'],['TimeoutError'])


if __name__ == '__main__':
    unittest.main()

# Copyright 2026 techrote. All rights reserved.
