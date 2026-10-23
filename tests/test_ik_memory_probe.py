"""Independent allocation-probe adversarial CPU mocks; no CUDA operations."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import contextlib
import io
import runpy

from nca.runs import immutable_json,sha256_file
from nca.contracts import sha256_json,canonical_json
from nca.train import ik_memory_probe as probe


class ProbeReviewTests(unittest.TestCase):
    def setUp(self):
        # Production now has real host containment and a global board claim.
        # CPU mocks must never modify either on the review machine.
        for target in ('nca.train.ik_gpu_ownership.acquire',
                       'nca.train.ik_gpu_ownership.release',
                       'nca.train.ik_gpu_ownership.verify',
                       'nca.ik.process_limits.memory_limit'):
            mocked=patch(target);mocked.start();self.addCleanup(mocked.stop)

    def test_import_does_not_load_tensor_runtime(self):
        result=subprocess.run([sys.executable,'-c',
            "import sys; import nca.train.ik_memory_probe; "
            "assert 'torch' not in sys.modules and 'numpy' not in sys.modules"],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_fixed_allocation_arithmetic_excludes_training(self):
        limits=probe.limits()
        self.assertEqual(limits['workers'],3)
        self.assertEqual(limits['chunk_bytes']*limits['chunks'],5*probe.GIB)
        self.assertEqual(limits['workers']*limits['allocator_bytes_per_worker'],15*probe.GIB)
        self.assertEqual(limits['board_bytes'],20*probe.GIB)
        self.assertTrue(limits['no_training'] and limits['no_data'] and limits['one_attempt'])
        self.assertLess(limits['active_seconds'],limits['wall_seconds'])

    def test_module_cli_prepares_once_without_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'prepared'
            with patch.object(sys,'argv',['probe','prepare','--root',str(root)]), \
                    patch('nca.train.ik_matched_prepare.prepared_source',return_value={
                        'revision':'a'*40,'source_sha256':'b'*64}), \
                    patch('nca.train.ik_experiment._titan_visibility'), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                runpy.run_path(probe.__file__,run_name='__main__')
            self.assertTrue((root/'declaration.json').is_file())
            self.assertEqual(len(output.getvalue().splitlines()),1)

    def test_claimed_pass_without_authenticated_full_protocol_is_not_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source={'source_sha256':'a'*64}
            immutable_json(root/'declaration.json',dict(source=source,sha256='b'*64))
            immutable_json(root/'result.json',dict(outcome='pass',declaration_sha256='b'*64,
                whole_seconds=1,cleanup_errors=[],training=False,artifacts={},qualification='claimed'))
            for index in range(3):immutable_json(root/f'worker-{index}-20.json',{'claimed':True})
            with self.assertRaises(ValueError):probe.qualification_record(root,source)

    def test_complete_cpu_protocol_fixture_binds_then_changed_receipt_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=dict(revision='a'*40,source_sha256='b'*64)
            declaration=dict(schema='nca-ik-memory-probe/v1',source=source,limits=probe.limits(),
                             gpu_visibility_sha256=probe.TITAN_FINGERPRINT)
            declaration['sha256']=sha256_json(declaration)
            immutable_json(root/'declaration.json',declaration)
            immutable_json(root/'claim.json',dict(declaration_sha256=declaration['sha256'],
                started=0.,review_sha256=['c'*64,'d'*64]))
            resources=dict(peak_resident_bytes=128*1024**2,cpu_user_seconds=1.,
                           cpu_system_seconds=1.,read_bytes=0,write_bytes=0)
            for index in range(3):
                pid=6100+index
                immutable_json(root/f'worker-{index}-claim.json',dict(index=index,
                    declaration_sha256=declaration['sha256']))
                immutable_json(root/f'worker-{index}-ready.json',dict(pid=pid,monotonic=0.,
                    resources=resources,context={'status':'ready','requested_device':'cuda:0'}))
                for step in range(1,21):
                    immutable_json(root/f'approve-{index}-{step}.json',dict(step=step,index=index,
                        declaration_sha256=declaration['sha256']))
                    immutable_json(root/f'worker-{index}-{step}.json',dict(pid=pid,monotonic=float(step),
                        resources=resources,memory=dict(allocated_bytes=step*256*1024**2,
                            peak_reserved_bytes=step*256*1024**2)))
                immutable_json(root/f'worker-{index}-exit.json',dict(pid=pid,monotonic=55.))
            immutable_json(root/'release.json',{'release':True})
            rows=[dict(monotonic=float(t),used_bytes=(2 if t<20 or t==56 else 17)*probe.GIB,
                       free_bytes=7*probe.GIB,temperature=35,utilization=0)
                  for t in range(0,57,2)]
            (root/'telemetry.jsonl').write_text(''.join(canonical_json(x)+'\n' for x in rows))
            result=dict(schema='nca-ik-memory-probe-result/v1',outcome='pass',
                declaration_sha256=declaration['sha256'],whole_seconds=56.,
                charged_wall_ceiling_seconds=300,cleanup_errors=[],training=False,
                context_baseline=rows[0],released_board=rows[-1],
                qualification='three_contexts_and_touched_buffers_only_not_training_performance',
                artifacts={p.relative_to(root).as_posix():sha256_file(p)
                           for p in root.rglob('*') if p.is_file()})
            immutable_json(root/'result.json',result)
            bound=probe.qualification_record(root,source)
            self.assertEqual(bound['result_file_sha256'],sha256_file(root/'result.json'))
            receipt=root/'worker-1-20.json'
            receipt.write_bytes(receipt.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'evidence changed'):
                probe.qualification_record(root,source)

    def test_preexisting_parent_claim_blocks_every_child_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            immutable_json(root/'claim.json',{'first':'preserved'})
            before=(root/'claim.json').read_bytes()
            with patch.object(probe,'verify',return_value={'sha256':'a'*64}), \
                    patch.object(probe.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(ValueError,'immutable artifact already exists'):
                    probe.run(root,[])
                launch.assert_not_called()
            self.assertEqual((root/'claim.json').read_bytes(),before)

    def test_duplicate_index_refuses_before_any_cuda_report(self):
        from nca.ik.declaration import runtime_expectation
        from nca.ik import device
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            declaration={'sha256':'a'*64,'runtime':runtime_expectation()}
            immutable_json(root/'claim.json',dict(declaration_sha256='a'*64,started=0))
            immutable_json(root/'worker-0-claim.json',{'first':'preserved'})
            with patch.object(probe,'verify',return_value=declaration), \
                    patch.object(device,'device_report',side_effect=AssertionError('CUDA reached')) as report, \
                    patch.object(device,'configure_device',side_effect=AssertionError('CUDA reached')):
                with self.assertRaises(ValueError):
                    probe.worker(root,0,[])
                report.assert_not_called()

    def mock_children(self,root):
        class Process:
            returncode=None
            def poll(self):return 0 if (root/'release.json').exists() else None
            def wait(self,timeout=None):self.returncode=0
        def launch(command,**kwargs):
            index=int(command[command.index('--index')+1])
            immutable_json(root/f'worker-{index}-ready.json',{'synthetic':True})
            return Process()
        return launch

    def test_no_allocation_permit_without_board_headroom(self):
        class UnexpectedPermit(RuntimeError):pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);published=[]
            def publish(path,value):
                if path.name.startswith('approve-'):
                    raise UnexpectedPermit('allocation was permitted')
                published.append(path.name)
                return immutable_json(path,value)
            row=dict(used_bytes=20*probe.GIB,free_bytes=4*probe.GIB,
                     utilization=0,temperature=30,monotonic=0)
            with patch.object(probe,'verify',return_value={'sha256':'a'*64}), \
                    patch.object(probe,'board_sample',return_value=row), \
                    patch.object(probe,'immutable_json',side_effect=publish), \
                    patch.object(probe.subprocess,'Popen',side_effect=self.mock_children(root)), \
                    patch.object(probe,'stop_tree'),patch.object(probe.time,'sleep'):
                with self.assertRaisesRegex(ValueError,'headroom|memory|envelope'):
                    probe.run(root,[])
            self.assertFalse(list(root.glob('approve-*.json')))

    def test_success_publication_cannot_outlive_wall_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);clock=[0.]
            def now():clock[0]+=.1;return clock[0]
            def publish(path,value):
                digest=immutable_json(path,value)
                if path.name.startswith('approve-'):
                    _,index,step=path.stem.split('-')
                    immutable_json(root/f'worker-{index}-{step}.json',{'synthetic':True})
                if path.name=='result.json':clock[0]=400.
                return digest
            row=dict(used_bytes=0,free_bytes=24*probe.GIB,
                     utilization=0,temperature=30,monotonic=0)
            with patch.object(probe,'verify',return_value={'sha256':'a'*64}), \
                    patch.object(probe,'board_sample',return_value=row), \
                    patch.object(probe,'immutable_json',side_effect=publish), \
                    patch.object(probe.subprocess,'Popen',side_effect=self.mock_children(root)), \
                    patch.object(probe,'stop_tree'),patch.object(probe.time,'sleep'), \
                    patch.object(probe,'ENTRY_STARTED',0.), \
                    patch.object(probe.time,'monotonic',side_effect=now):
                with self.assertRaisesRegex((ValueError,TimeoutError),'publication|wall|overrun'):
                    probe.run(root,[])
            self.assertTrue((root/'publication-overrun.json').exists())


if __name__=='__main__':unittest.main()

# Copyright 2026 techrote. All rights reserved.
