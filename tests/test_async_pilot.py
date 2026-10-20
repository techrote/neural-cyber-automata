"""Recorder coverage/accounting checks use explicit synthetic hosts, not gameplay evidence."""

from contextlib import redirect_stdout
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class AsyncPilotRecorderTests(unittest.TestCase):
    def record(self, *, stepped_calls=1, fail_step=False, subtest_failure=False, expire_on_close=None):
        script = Path(__file__).resolve().parents[1] / "tools" / "async_pilot.py"
        spec = importlib.util.spec_from_file_location("async_pilot_test_subject", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        hosts = []
        clock = [0.0]

        class FakeHost:
            def __init__(self, checkout, godot, *, mode):
                self.mode, self.identity, self.closed = mode, {"mode": mode, "synthetic": True}, False
                hosts.append(self)

            def request(self, operation, **fields):
                if fail_step and self.mode == "stepped":
                    raise RuntimeError("synthetic unknown completion")
                return {"actual_ticks": 3}

            def close(self):
                self.closed = True
                if self.mode == expire_on_close:
                    clock[0] = 121.0

        class Stepped(unittest.TestCase):
            mode = "stepped"

            @classmethod
            def tearDownClass(cls):
                cls.host.close()

            def test_recorded_operation(self):
                for _ in range(stepped_calls):
                    self.host.request("step", ticks=3)
                if subtest_failure:
                    with self.subTest(delay=240):
                        self.fail("synthetic expired-source subtest")

        class Realtime(Stepped):
            mode = "realtime"

            def test_recorded_operation(self):
                self.host.request("step", ticks=3)

        cases = SimpleNamespace(SteppedDecisionTests=Stepped, ContinuousDecisionTests=Realtime)
        fake_spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda value: None))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "synthetic.json.gz"
            arguments = [str(script), "--checkout", "synthetic-checkout", "--godot", "synthetic-godot",
                         "--output", str(output)]
            with (patch.object(module, "GodotHost", FakeHost),
                  patch.object(module, "source_identity", return_value=("a" * 40, {}, "b" * 64)),
                  patch.object(module.importlib.util, "spec_from_file_location", return_value=fake_spec),
                  patch.object(module.importlib.util, "module_from_spec", return_value=cases),
                  patch.object(module.time, "monotonic", side_effect=lambda: clock[0]),
                  patch.object(sys, "argv", arguments), patch.dict(os.environ), redirect_stdout(io.StringIO())):
                status = module.main()
            record = json.loads(gzip.decompress(output.read_bytes()))
            index = json.loads(output.with_suffix(".index.json").read_text())
        self.assertTrue(all(host.closed for host in hosts))
        return status, record, index

    def test_both_modes_and_same_named_cases_keep_all_host_operations(self):
        status, record, index = self.record()
        self.assertEqual(status, 0)
        self.assertEqual(record["schema"], "nca-async-host-pilot/v2")
        self.assertEqual(record["mode"], "mixed-stepped-realtime")
        self.assertEqual([row["mode"] for row in record["hosts"]], ["stepped", "realtime"])
        self.assertEqual([row["mode"] for row in record["cases"]], ["stepped", "realtime"])
        self.assertEqual(len({row["id"] for row in record["cases"]}), 2)
        self.assertEqual([row["host_id"] for row in record["operations"]], ["host-1", "host-2"])
        self.assertEqual(index["step_calls"], 2)
        self.assertEqual(index["attempted_step_calls"], 2)
        self.assertEqual(index["actual_ticks"], 6)

    def test_two_hosts_share_one_original_step_budget(self):
        status, record, index = self.record(stepped_calls=2000)
        self.assertEqual(status, 1)
        self.assertEqual(index["attempted_step_calls"], 2000)
        self.assertEqual(index["step_calls"], 2000)
        self.assertEqual(len(record["operations"]), 2000)
        self.assertTrue(record["cases"][0]["passed"])
        self.assertFalse(record["cases"][1]["passed"])
        self.assertIn("pilot budget exhausted", record["cases"][1]["failure"])

    def test_uncertain_operation_stays_charged_and_cannot_claim_actual_ticks(self):
        status, record, index = self.record(fail_step=True)
        self.assertEqual(status, 1)
        self.assertEqual(index["attempted_step_calls"], 2)
        self.assertEqual(index["step_calls"], 1)
        self.assertEqual(index["actual_ticks"], 3)
        self.assertEqual(record["operations"][0]["status"], "incomplete")
        self.assertEqual(record["operations"][0]["error_type"], "RuntimeError")
        self.assertIsNone(record["operations"][0]["response"])
        self.assertFalse(record["cases"][0]["passed"])
        self.assertTrue(record["cases"][1]["passed"])

    def test_subtest_failure_marks_its_parent_case_failed_and_keeps_detail(self):
        status, record, _ = self.record(subtest_failure=True)
        self.assertEqual(status, 1)
        self.assertFalse(record["cases"][0]["passed"])
        self.assertTrue(record["cases"][1]["passed"])
        self.assertIn("delay=240", record["suite_failures"][0]["id"])
        self.assertEqual(record["suite_failures"][0]["case_id"], record["cases"][0]["id"])

    def test_final_cleanup_crossing_wall_budget_cannot_publish_a_pass(self):
        status, record, _ = self.record(expire_on_close="realtime")
        self.assertEqual(status, 1)
        self.assertTrue(all(row["passed"] for row in record["cases"]))
        self.assertFalse(record["within_budget"])
        self.assertEqual(record["wall_seconds"], 121)
        self.assertTrue(all(row["cleanup"]["status"] == "completed" for row in record["hosts"]))

    def test_second_host_never_starts_after_wall_budget_is_exhausted(self):
        status, record, _ = self.record(expire_on_close="stepped")
        self.assertEqual(status, 1)
        self.assertEqual([row["mode"] for row in record["hosts"]], ["stepped"])
        self.assertIn("before host start", record["suite_failures"][0]["detail"])
        self.assertFalse(record["within_budget"])


if __name__ == "__main__":
    unittest.main()
