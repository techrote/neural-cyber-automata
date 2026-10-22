"""Independent arithmetic controls for the offline pipeline planning helper."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("jolt_pipeline_envelope", ROOT/"tools/check_jolt_pipeline_envelope_v3.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PipelineEnvelopeV3Tests(unittest.TestCase):
    def test_typed_owner_and_lifecycle_count_closure(self):
        counts = MODULE.event_counts()
        self.assertEqual(sum(counts.values()), 115110)
        self.assertEqual(sum(v for k, v in counts.items() if k.startswith("owner_request_")), 38338)
        self.assertEqual(sum(v for k, v in counts.items() if k.startswith("owner_response_")), 38338)
        self.assertEqual(counts["owner_response_step"], 18944)
        self.assertEqual(counts["episode_started"], 6464)
        self.assertEqual(counts["episode_finished_authored"], 84)
        self.assertEqual(counts["episode_finished_other"], 6380)
        self.assertEqual(counts["safe_retirement_verified"], 6464)
        self.assertEqual(counts["fixed_notes"], 98)

    def test_complete_tail_counts_all_repeated_readbacks(self):
        passes = MODULE.pass_inventory()
        self.assertEqual(sum(row.get("journal_chain_passes", 0) for row in passes.values()), 3)
        self.assertEqual(sum(row.get("reservation_passes", 0) for row in passes.values()), 9)
        self.assertEqual(passes["closure"]["reservation_passes"], 2)
        self.assertEqual(passes["publication"]["reservation_passes"], 2)
        self.assertEqual(passes["verification"]["reservation_passes"], 2)
        self.assertEqual(passes["audit"]["episode_family_decode_hash_passes"], 2)

    def test_projection_uses_maximum_times_predeclared_factor_and_never_negative(self):
        result = MODULE.projection([1., 4., 2.], 300, 10)
        self.assertEqual(result["projected_seconds"], 480.)
        for samples, units, sample_units in (([], 1, 1), ([float("nan")], 1, 1),
                ([-1.], 1, 1), ([1.], -1, 1), ([1.], 1, 0), ([True], 1, 1)):
            with self.subTest(samples=samples, units=units, sample_units=sample_units), self.assertRaises(ValueError):
                MODULE.projection(samples, units, sample_units)

    def test_all_typed_specimens_fit_the_linked_journal_bound(self):
        for name, value in MODULE.templates().items():
            value = dict(value)
            value["_journal"] = {"sequence": 115110, "previous_sha256": "f"*64, "record_sha256": "f"*64}
            self.assertLessEqual(len((MODULE.canonical_json(value)+"\n").encode()), 65536, name)

    def test_phase_components_do_not_drop_closure_or_external_verification(self):
        journal = {"example": {"append_fsync_readback": {"projected_seconds": 10.},
            "serialization": {"projected_seconds": 2.}, "chain_read": {"projected_seconds": 3.},
            "syntax_read_only": {"projected_seconds": 1.}}}
        episodes = {"decisions_1_gzip_"+str(level): {
            "decode_and_canonical_hash_only": {"projected_seconds": 6.},
            "streaming_write_gzip_fsync_hash_and_eof_readback": {"projected_seconds": 7.}}
            for level in (1, 9)}
        parts = MODULE.planning_components(journal, {"projected_seconds": 5.}, episodes,
                                          {"projected_seconds": 8.})
        self.assertEqual({phase: sum(values.values()) for phase, values in parts.items()},
                         {"active": 26., "closure": 13., "audit": 31., "publication": 18., "verification": 21.})
        self.assertEqual(len(MODULE.UNRESOLVED), 7)
        self.assertFalse(MODULE.fixed_design()["native_host_or_trained_checkpoint_permitted"])

    def test_import_is_host_and_optional_learner_free(self):
        code = """
import importlib.util, pathlib, sys
path = pathlib.Path('tools/check_jolt_pipeline_envelope_v3.py')
spec = importlib.util.spec_from_file_location('pipeline', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert 'nca.controllers.jolt_learned' not in sys.modules
assert module.fixed_design()['maximum_sampling_seconds_from_script_entry'] is None
assert module.fixed_design()['whole_process_hard_deadline_enforced'] is False
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
