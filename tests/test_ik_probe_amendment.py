"""Independent CPU-only host-limit amendment controls; never retire real holds."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from nca.runs import immutable_json,sha256_file
from nca.train import ik_memory_probe as probe
from nca.train import ik_matched_supervisor as supervisor


class ProbeAmendmentReviewTests(unittest.TestCase):
    def test_only_host_envelope_expands_while_gpu_work_and_wall_remain_fixed(self):
        limits=probe.limits()
        self.assertEqual(limits['host_peak_bytes_per_process'],8*probe.GIB)
        self.assertEqual((limits['workers'],limits['chunks'],limits['chunk_bytes']),(3,20,256*1024**2))
        self.assertEqual(limits['allocator_bytes_per_worker'],5*probe.GIB)
        self.assertEqual(limits['board_bytes'],20*probe.GIB)
        self.assertEqual((limits['wall_seconds'],limits['active_seconds'],limits['hold_seconds']),(300,240,30))
        self.assertIs(limits['no_training'],True);self.assertIs(limits['no_data'],True);self.assertIs(limits['one_attempt'],True)
        self.assertEqual(limits['prior_failed_result_sha256'],'77220c2773c6f9e36a4288639c7cb5c384a0355c07f21c476e60722d05827e35')

    def test_private_commit_and_pagefile_peak_are_not_substituted_by_resident_peak(self):
        kernel=Mock();psapi=Mock()
        kernel.GetCurrentProcess.return_value=123
        def io(handle,ref):
            ref._obj.read_bytes=17;ref._obj.write_bytes=29;return 1
        def memory(handle,ref,size):
            ref._obj.peak_working_set=512*1024**2
            ref._obj.private=3*probe.GIB
            ref._obj.pagefile=4*probe.GIB
            ref._obj.peak_pagefile=5*probe.GIB
            return 1
        kernel.GetProcessIoCounters.side_effect=io;psapi.GetProcessMemoryInfo.side_effect=memory
        with patch.object(supervisor.os,'name','nt'),patch('ctypes.WinDLL',side_effect=lambda name,**kw:kernel if name=='kernel32' else psapi,create=True),patch.object(supervisor.os,'times',return_value=SimpleNamespace(user=1.,system=2.)):
            result=supervisor.process_metrics()
        self.assertEqual(result['peak_resident_bytes'],512*1024**2)
        self.assertEqual(result['private_commit_bytes'],3*probe.GIB)
        self.assertEqual(result['pagefile_bytes'],4*probe.GIB)
        self.assertEqual(result['peak_pagefile_bytes'],5*probe.GIB)
        self.assertEqual((result['read_bytes'],result['write_bytes']),(17,29))

    def test_wrong_failed_history_never_changes_global_hold_or_declares_success(self):
        with tempfile.TemporaryDirectory() as directory:
            previous=Path(directory)/'failed';previous.mkdir()
            immutable_json(previous/'result.json',{'outcome':'failed','cleanup_errors':[]})
            with patch('nca.train.ik_gpu_ownership.path') as held:
                with self.assertRaisesRegex(ValueError,'exact retained failed'):
                    probe.retire_predecessor(Path(directory),{'previous_probe':str(previous),'sha256':'a'*64})
                held.assert_not_called()

    def test_live_predecessor_PID_refuses_before_ownership_retirement(self):
        kernel=Mock();kernel.OpenProcess.return_value=321
        with patch.object(probe,'validate_predecessor',return_value={'sha256':'old'}),patch.object(probe,'_read',return_value={'pid':6101}),patch('ctypes.WinDLL',return_value=kernel,create=True),patch('ctypes.set_last_error',create=True),patch('nca.train.ik_gpu_ownership.path') as held:
            with self.assertRaisesRegex(ValueError,'PID still exists'):
                probe.retire_predecessor('unused',{'previous_probe':'unused','sha256':'a'*64})
            kernel.CloseHandle.assert_called_once_with(321)
            held.assert_not_called()



    def test_exact_settled_hold_is_preserved_before_retirement_receipt_and_removal(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);previous=base/'old';previous.mkdir();root=base/'new';root.mkdir()
            immutable_json(previous/'result.json',{'outcome':'failed'})
            immutable_json(root/'claim.json',{'review_sha256':['a'*64,'b'*64]})
            held=base/'global-hold.json';raw=b'{ "kind": "allocation_probe", "declaration_sha256": "old" }\n'
            held.write_bytes(raw)
            actual_read=probe._read
            def read(path,**kw):
                if Path(path).name.endswith('-ready.json'):return {'pid':6101}
                return actual_read(path,**kw)
            kernel=Mock();kernel.OpenProcess.return_value=0
            with patch.object(probe,'validate_predecessor',return_value={'sha256':'old'}),patch.object(probe,'_read',side_effect=read),patch('ctypes.WinDLL',return_value=kernel,create=True),patch('ctypes.set_last_error',create=True),patch('ctypes.get_last_error',return_value=87,create=True),patch('nca.train.ik_gpu_ownership.path',return_value=held):
                probe.retire_predecessor(root,{'previous_probe':str(previous),'sha256':'new'})
            self.assertFalse(held.exists())
            self.assertEqual((root/'retired-predecessor-ownership.json').read_bytes(),raw)
            receipt=json.loads((root/'ownership-retirement.json').read_bytes())
            self.assertIs(receipt['all_predecessor_pids_absent'],True)
            self.assertEqual(receipt['successor_declaration_sha256'],'new')
            self.assertEqual(receipt['previous_result_sha256'],sha256_file(previous/'result.json'))
            self.assertEqual(kernel.OpenProcess.call_count,3)

    def test_access_denied_is_not_treated_as_dead_predecessor(self):
        kernel=Mock();kernel.OpenProcess.return_value=0
        with patch.object(probe,'validate_predecessor',return_value={'sha256':'old'}),patch.object(probe,'_read',return_value={'pid':6101}),patch('ctypes.WinDLL',return_value=kernel,create=True),patch('ctypes.set_last_error',create=True),patch('ctypes.get_last_error',return_value=5,create=True),patch('nca.train.ik_gpu_ownership.path') as held:
            with self.assertRaisesRegex(ValueError,'termination unverified'):
                probe.retire_predecessor('unused',{'previous_probe':'unused','sha256':'new'})
            held.assert_not_called()



    def test_host_headroom_requires_both_physical_and_commit_not_either(self):
        import ctypes
        def query(physical,commit,success=True):
            def fill(ref):
                ref._obj.available_physical=physical*probe.GIB
                ref._obj.available_commit=commit*probe.GIB
                return int(success)
            fake=SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=fill))
            with patch('ctypes.windll',fake,create=True):return probe.host_headroom()
        self.assertEqual(query(39,68),{'available_physical_bytes':39*probe.GIB,'available_commit_bytes':68*probe.GIB})
        for physical,commit,success in [(31,68,True),(39,35,True),(39,68,False)]:
            with self.subTest(physical=physical,commit=commit),self.assertRaises(ValueError):
                query(physical,commit,success)


if __name__=='__main__':unittest.main()
