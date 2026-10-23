"""CPU-only preparation/refusal tests; no accepted-corpus optimization."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nca.contracts import sha256_json
from nca.runs import immutable_json
from nca.train import ik_matched_prepare as matched

ROOT = Path(__file__).resolve().parents[1]


class MatchedPreparationTests(unittest.TestCase):
    def test_fixed_configuration_refuses_expansion_and_type_coercion(self):
        config = json.loads((ROOT / matched.CONFIG_PATH).read_text())
        self.assertEqual(matched.validate_config(config), matched.candidate_config())
        self.assertEqual(config['loss_weights']['endpoint'], 4.0)
        self.assertEqual(config['gpu_allocator_bytes_per_worker'] * 3
                         + config['gpu_nonallocator_and_monitor_margin_bytes'], 20 * 1024**3)
        for key, value in [('epochs', 41), ('max_evaluation_calls', 1),
                           ('execution_admitted', True), ('cpu_threads_per_worker', True),
                           ('excluded_final_suites', []), ('automatic_retry', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                matched.validate_config(config | {key: value})
        bad = deepcopy(config)
        bad['trials']['parallel']['max_workers'] = 4
        with self.assertRaises(ValueError):
            matched.validate_config(bad)
        bad = deepcopy(config)
        bad['loss_weights']['joint'] = 0.2
        with self.assertRaises(ValueError):
            matched.validate_config(bad)

    def test_cli_has_no_execution_verb_and_imports_no_numerical_runtime(self):
        program = '''
import sys
sys.path.insert(0,sys.argv[1])
from nca.train.ik_matched_prepare import main
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
try: main(['run'])
except SystemExit as e: assert e.code == 2
else: raise AssertionError('execution command accepted')
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
'''
        subprocess.run([sys.executable, '-S', '-c', program, str(ROOT / 'src')],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                       capture_output=True, check=True)

    def prepare_fixture(self, root):
        source = {'revision': 'b'*40, 'source_sha256': 'a'*64,
                  'files': {matched.ACCEPTANCE_PATH: 'c'*64}}
        runtime = dict(python='3.12.10', os='Windows', os_release='11', architecture='AMD64',
                       packages={'torch': '2.8.0+cu126', 'numpy': '2.3.5'})
        corpus = SimpleNamespace(accepted=True, training=[None]*5260, tuning=[None]*2020)
        normalization = SimpleNamespace(to_dict=lambda: {'fixture': 'normalization'})
        with patch.object(matched, 'prepared_source', return_value=source), \
                patch.object(matched, '_titan_visibility', return_value=matched.TITAN_FINGERPRINT), \
                patch.object(matched, 'runtime_expectation', return_value=runtime), \
                patch.object(matched, 'read_polar_followup_corpus', return_value=corpus), \
                patch.object(matched, 'fit_training_normalization', return_value=normalization), \
                patch.object(matched, 'schedules', return_value={str(s): {'seed': s} for s in matched.SEEDS}), \
                patch.object(matched, 'corpus_identity', return_value={'fixture': 'original'}):
            return matched.prepare(root/'corpus', root/'registry', root/'prepared')

    def test_complete_packet_reopens_six_zero_ledgers_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet = self.prepare_fixture(root)
            self.assertFalse(packet['execution_admitted'])
            self.assertEqual(packet['reviews'], [])
            self.assertTrue(matched.inspect(root/'prepared')['zero_consumption'])
            before = {p.relative_to(root).as_posix(): p.read_bytes()
                      for p in root.rglob('*') if p.is_file()}
            with patch.object(matched, 'prepared_source') as source:
                with self.assertRaisesRegex(ValueError, 'new output'):
                    matched.prepare(root/'corpus', root/'registry', root/'prepared')
                source.assert_not_called()
            self.assertEqual(before, {p.relative_to(root).as_posix(): p.read_bytes()
                                     for p in root.rglob('*') if p.is_file()})

    def test_metadata_or_corpus_failure_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(matched, 'prepared_source', return_value={}), \
                    patch.object(matched, '_titan_visibility', side_effect=ValueError('wrong GPU')), \
                    patch.object(matched, 'read_polar_followup_corpus') as reader:
                with self.assertRaisesRegex(ValueError, 'wrong GPU'):
                    matched.prepare(root/'corpus', root/'registry', root/'prepared')
                reader.assert_not_called()
            self.assertFalse((root/'prepared').exists())
            for output in (root/'corpus', root/'corpus'/'inside', root):
                with self.assertRaisesRegex(ValueError, 'new output'):
                    matched.prepare(root/'corpus', root/'registry', output)

    def test_ledger_consumption_and_modified_declaration_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare_fixture(root)
            path = root/'prepared'/'serial'/'declaration.json'
            before = path.read_bytes()
            path.write_bytes(before + b' ')
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                matched.inspect(root/'prepared')
            path.write_bytes(before)
            value = json.loads(before)
            ledger = matched.IKRunLedger.open(root/'prepared'/'serial'/'seed-41011',
                expected_identity=matched.identity_for(value, 41011), expected_allocation=matched.allocation())
            attempt = ledger.begin_attempt()
            try:
                with self.assertRaisesRegex(ValueError, 'consumed work'):
                    matched.inspect(root/'prepared')
            finally:
                attempt.finish('failed', reason='synthetic_preparation_negative_control')


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'),
                     'optional CPU tensor runtime')
class MatchedCandidateTests(unittest.TestCase):
    def test_only_loss_weight_changes_and_initial_tensors_are_matched(self):
        from nca.ik.data import DatasetConfig
        from nca.ik.device import configure_device
        from nca.ik.polar import PolarNormalization, build_trainer
        normalization = PolarNormalization(1, (0.,)*14, (1.,)*14, 'a'*64, 'b'*64)
        context = configure_device('cpu', 41011)
        baseline = build_trainer(context, normalization, DatasetConfig())
        context = configure_device('cpu', 41011)
        candidate = matched.build_candidate_trainer(context, normalization, DatasetConfig())
        self.assertEqual(candidate.weights.to_dict(), baseline.weights.to_dict() | {'endpoint': 4.0})
        self.assertEqual(candidate.optimizer.defaults, baseline.optimizer.defaults)
        for name, tensor in baseline.model.state_dict().items():
            context.torch.testing.assert_close(candidate.model.state_dict()[name], tensor, rtol=0, atol=0)
        self.assertFalse(context.torch.cuda.is_initialized())
        self.assertEqual(candidate.committed_steps, 0)

# Copyright 2026 techrote. All rights reserved.
