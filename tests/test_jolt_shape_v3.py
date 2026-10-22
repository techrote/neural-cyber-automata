"""Source-shaped byte proof and fixed artifact envelope checks; no host I/O."""
from pathlib import Path
import tempfile
from unittest import TestCase, main

from nca.contracts import ContractError
from nca.train.jolt_shape_v3 import shape_bounds_v3
from nca.train.jolt_storage_v3 import (EPISODE_BASE_BYTES, EPISODE_DECISION_BYTES,
    MAX_JOURNAL_RECORD_BYTES, MAX_ORDINARY_JSON_BYTES, metrics_byte_limit_v3,
    storage_bounds_v3, validate_worker_result_v3, file_identity_v3, CHUNK_BYTES)
from nca.train.jolt_budget_v3 import budget_exhaustion_v3


class SourceShapeV3Tests(TestCase):
    def test_source_native_response_and_decision_shapes_fit_declared_readers(self):
        bounds=shape_bounds_v3()
        self.assertEqual(bounds["native_observation_bytes"],6021)
        self.assertEqual(bounds["linked_owner_response_bytes"],51120)
        self.assertLess(bounds["linked_owner_response_bytes"],MAX_JOURNAL_RECORD_BYTES)
        self.assertLess(bounds["capture_decision_bytes"],EPISODE_DECISION_BYTES)
        self.assertLess(bounds["episode_fixed_bytes"],EPISODE_BASE_BYTES)
        self.assertEqual(bounds["per_decision_metadata_allowance_bytes"],8192)
        self.assertFalse(bounds["timing_or_physical_feasibility_proved"])

    def test_authored_recovery_events_fit_distinct_composite_worker_bound(self):
        self.assertEqual(metrics_byte_limit_v3("authored"),48*1024)
        for kind in ("idle","bc","residual"):
            self.assertEqual(metrics_byte_limit_v3(kind),8*1024)
        bounds=storage_bounds_v3()
        self.assertEqual(bounds["pieces"]["metrics"],84*48*1024+6380*8*1024)
        self.assertLess(bounds["maximum_worker_result_bytes"],MAX_ORDINARY_JSON_BYTES)
        self.assertEqual(bounds["lifecycle_files"],17)
        self.assertEqual(bounds["complete_lifecycle_files"],16)
        self.assertEqual(bounds["outer_stdout_stderr_and_exit_files"],3)

    def test_worker_guard_rejects_extra_multiplicity_and_oversized_fixed_tail(self):
        value={"results":[{"kind":"authored","recovery":"x"*16000}],"schema":"synthetic"}
        calls=[]
        self.assertLess(validate_worker_result_v3(value,check=lambda:calls.append(1))["total_upper_bytes"],MAX_ORDINARY_JSON_BYTES)
        self.assertGreaterEqual(len(calls),2)
        with self.assertRaises(ContractError):
            validate_worker_result_v3({"results":[{"kind":"authored"}]*85})
        with self.assertRaises(ContractError):
            validate_worker_result_v3({"results":[{"kind":"residual","payload":"x"*9000}]})
        with self.assertRaises(ContractError):
            validate_worker_result_v3({"results":[],"fixed":["x"*65000]*130})

    def test_hash_checks_each_chunk_and_refuses_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"bytes"
            path.write_bytes(b"x"*(CHUNK_BYTES*3))
            calls=[]
            identity=file_identity_v3(path,check=lambda:calls.append(1))
            self.assertEqual(identity["bytes"],CHUNK_BYTES*3)
            self.assertGreaterEqual(len(calls),4)
            with self.assertRaises(ContractError):
                file_identity_v3(path,maximum_bytes=1)
            def alter():
                path.write_bytes(b"x")
            with self.assertRaises(ContractError):
                file_identity_v3(path,check=alter)

    def test_only_declared_counter_or_clock_exhaustion_is_capped(self):
        for code in ("budget_counter_bc_forward","budget_stage_ppo_decisions","jolt_v3_phase_deadline"):
            self.assertTrue(budget_exhaustion_v3(ContractError(code,"synthetic")))
        for code in ("jolt_storage_bytes","jolt_v3_reservations","jolt_v3_clock","jolt_v3_phase"):
            self.assertFalse(budget_exhaustion_v3(ContractError(code,"synthetic")))
        self.assertFalse(budget_exhaustion_v3(ValueError("not a declared budget")))


if __name__=="__main__":
    main()
