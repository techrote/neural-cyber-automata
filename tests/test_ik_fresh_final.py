"""Unlabeled CPU recipe conformance; no teacher calls or final scoring."""
from dataclasses import replace
import unittest
from nca.contracts import ContractError,sha256_json
from nca.ik.data import DatasetConfig,iter_tasks
from nca.ik.fresh_final import FreshFinalConfig

class FreshFinalTests(unittest.TestCase):
    def test_recipe_binds_geometry_and_preserves_counts(self):
        config=FreshFinalConfig();tasks=list(iter_tasks(config))
        self.assertEqual(config.sha256,'b7c59d037bffb00ecaba6537f865dad7303949cdaf5c5dcf4e89b7126c93b5b8')
        self.assertEqual(sha256_json([t.id for t in tasks]),'aab6fb764202204f397b5a73a0b0f6aa25e7ec5ca4fc841e4dba5891f0984b72')
        self.assertEqual(len(tasks),2940)
        self.assertTrue(all(t.partition=='final' and t.family_id.startswith(config.family_namespace+':') for t in tasks))
        self.assertEqual(sum(t.state.kind=='point' for t in tasks),2000)
        self.assertEqual(sum(t.state.kind=='trajectory' for t in tasks),800)

    def test_mutated_lane_or_withheld_regions_refuse(self):
        for changes in ({'withheld_bins':(0,1,2,3,4,5,6,7)}, {'withheld_hip_lane':(.08,.92)},
                        {'generation_seed':53024},{'mode':'fixture'}):
            with self.assertRaises(ContractError):replace(FreshFinalConfig(),**changes)
        with self.assertRaises(ContractError):list(iter_tasks(FreshFinalConfig(),'training'))

    def test_original_configuration_identity_unchanged(self):
        self.assertEqual(DatasetConfig().sha256,'fb8614412a62f7495519f463e3052ab060b0701701a3dfdb7265f0ad64488b51')

# Copyright 2026 techrote. All rights reserved.
