"""Software evidence for the authored diagnostic's admission and accounting."""

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from nca.contracts import ContractError, sha256_json
from nca.evaluation.jolt_arrival import (DiagnosticBudget, FAMILIES, RECOVERY_FAMILIES, RECOVERY_V2_FAMILIES, diagnostic_cases,
                                       read_declaration, summarize, validate_owner_interval, write_json)
from test_morphology import fixture

ROOT = Path(__file__).resolve().parents[1]


class ArrivalDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.config = read_declaration(ROOT / "configs/baselines/jolt-arrival-diagnostic-v1.json")
        original = json.loads((ROOT / self.config["case_source"]["path"]).read_text())
        self.cases = diagnostic_cases(original)

    def test_existing_case_seed_pairs_are_complete_and_not_relabelled_as_sealed(self):
        self.assertEqual(len(self.cases), 32)
        self.assertEqual(sum(row["origin"] == "previously_open_training" for row in self.cases), 24)
        self.assertEqual(sum(row["origin"] == "previously_open_evaluation" for row in self.cases), 8)
        self.assertEqual(len({(tuple(row["spawn"]), tuple(row["target"]), row["host_seed"])
                              for row in self.cases}), 32)

    def test_changed_declaration_refuses_before_any_runtime_work(self):
        changed = copy.deepcopy(self.config)
        changed["budget"]["maximum_decisions"] += 1
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "changed.json"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ContractError, "arrival_declaration"):
                read_declaration(path)

    def test_separately_pinned_recovery_declaration_is_additive_and_bounded(self):
        recovery = copy.deepcopy(self.config)
        recovery["schema"] = "nca-jolt-recovery-diagnostic/v1"
        recovery["families"] = list(RECOVERY_FAMILIES)
        recovery["budget"].update(maximum_episodes=96, maximum_decisions=3840,
            maximum_requested_ticks=23040, maximum_host_requests=4226, maximum_wall_seconds=900)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recovery.json"
            path.write_text(json.dumps(recovery))
            self.assertEqual(read_declaration(path, expected_sha256=sha256_json(recovery)), recovery)
            with self.assertRaisesRegex(ContractError, "arrival_declaration"):
                read_declaration(path)
            for mutate in (lambda x: x["budget"].update(maximum_wall_seconds=901),
                           lambda x: x["budget"].update(automatic_retry=True),
                           lambda x: x["budget"].update(optimizer_updates=1),
                           lambda x: x["goal"].update(require_support=False),
                           lambda x: x["physics"].update(decision_ticks=12),
                           lambda x: x["case_source"].update(sha256="a" * 64),
                           lambda x: x["preserved_sources"].pop("src/nca/train/task.py")):
                changed = copy.deepcopy(recovery)
                mutate(changed)
                path.write_text(json.dumps(changed))
                with self.subTest(changed=sha256_json(changed)), self.assertRaisesRegex(ContractError, "arrival_declaration"):
                    read_declaration(path, expected_sha256=sha256_json(changed))

    def test_uncertain_request_consumes_budget_and_remains_visible(self):
        limits = {**self.config["budget"], "maximum_decisions": 1, "maximum_requested_ticks": 6}
        journal = io.BytesIO()
        budget = DiagnosticBudget(limits, journal)
        serial = budget.reserve("step", {"ticks": 6})
        with self.assertRaisesRegex(ContractError, "arrival_step_budget"):
            budget.reserve("step", {"ticks": 6})
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["step_requests"], 1)
        self.assertEqual(snapshot["requested_step_ticks"], 6)
        self.assertEqual(snapshot["confirmed_step_ticks"], 0)
        self.assertTrue(snapshot["unconfirmed_tail"])
        self.assertEqual(snapshot["pending_requests"][0]["serial"], serial)
        self.assertIn(b'request_reserved', journal.getvalue())

    def test_v2_declaration_cannot_reinterpret_the_frozen_v1_family(self):
        recovery = copy.deepcopy(self.config)
        recovery.update(schema="nca-jolt-recovery-diagnostic/v2", families=list(RECOVERY_V2_FAMILIES))
        recovery["budget"].update(maximum_episodes=96, maximum_decisions=3840,
            maximum_requested_ticks=23040, maximum_host_requests=4226, maximum_wall_seconds=900)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v2.json"
            path.write_text(json.dumps(recovery))
            self.assertEqual(read_declaration(path, expected_sha256=sha256_json(recovery)), recovery)
            with self.assertRaisesRegex(ContractError, "arrival_declaration"):
                read_declaration(path)
            for schema, families in (("nca-jolt-recovery-diagnostic/v1", RECOVERY_V2_FAMILIES),
                                     ("nca-jolt-recovery-diagnostic/v2", RECOVERY_FAMILIES)):
                changed = {**recovery, "schema": schema, "families": list(families)}
                path.write_text(json.dumps(changed))
                with self.subTest(schema=schema), self.assertRaisesRegex(ContractError, "arrival_declaration"):
                    read_declaration(path, expected_sha256=sha256_json(changed))

    def test_malformed_confirmation_cannot_erase_uncertain_tail(self):
        budget = DiagnosticBudget(self.config["budget"], io.BytesIO())
        serial = budget.reserve("step", {"ticks": 6})
        for value in (None, True, 7, -1, "6"):
            with self.subTest(value=value), self.assertRaisesRegex(ContractError, "arrival_tick_receipt"):
                budget.confirm(serial, {"actual_ticks": value})
            self.assertTrue(budget.snapshot()["unconfirmed_tail"])
        budget.confirm(serial, {"actual_ticks": 6})
        self.assertFalse(budget.snapshot()["unconfirmed_tail"])
        self.assertEqual(budget.snapshot()["confirmed_step_ticks"], 6)

    def test_cancellation_is_available_after_wall_expiry_without_extra_steps(self):
        now = [0.]
        budget = DiagnosticBudget(self.config["budget"], io.BytesIO(), clock=lambda: now[0])
        now[0] = 600.
        with self.assertRaisesRegex(ContractError, "arrival_wall_budget"):
            budget.reserve("step", {"ticks": 6})
        serial = budget.reserve("cancel", {})
        budget.confirm(serial, {})
        self.assertEqual(budget.snapshot()["requested_step_ticks"], 0)

    def test_partial_or_duplicate_outcomes_cannot_become_complete(self):
        self.assertFalse(summarize([], self.cases)["complete_valid_pairing"])
        metrics = {"success": False, "administratively_capped": False, "integrity_failure": False,
            "failed_admissions": 0, "overshoot_cells": 1., "final_distance_cells": 2.,
            "capped_completion_seconds": 4., "peak_toward_speed_cells_per_second": 20.,
            "mean_absolute_angle_radians": .1, "ordinary_trim_reversals_per_second": 0.,
            "purposeful_launch": True, "primary_edges": 1}
        rows = [{"case_id": case["id"], "family": family, "origin": case["origin"],
                 "metrics": dict(metrics), "error": None, "owner_closure_confirmed": True}
                for case in self.cases for family in FAMILIES]
        result = summarize(rows, self.cases)
        self.assertTrue(result["complete_valid_pairing"])
        self.assertFalse(result["morphology_admitted"])
        self.assertFalse(result["learned_skill_qualified"])
        self.assertEqual(result["families"][FAMILIES[1]]["successes"], 0)
        self.assertFalse(summarize(rows[:-1], self.cases)["complete_valid_pairing"])
        with self.assertRaisesRegex(ContractError, "arrival_pairing"):
            summarize(rows + [rows[0]], self.cases)
        rows[0]["owner_closure_confirmed"] = False
        self.assertFalse(summarize(rows, self.cases)["complete_valid_pairing"])

    def test_failure_records_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            write_json(path, {"status": "failed"})
            with self.assertRaises(FileExistsError):
                write_json(path, {"status": "completed"})
            self.assertEqual(json.loads(path.read_bytes()), {"status": "failed"})

    def test_terminal_suspension_requires_exactly_one_owner_revocation(self):
        adapter, source = fixture(tick=0, control=0)
        for generation in (0, 1, 2):
            _, after = fixture(tick=6, control=generation)
            after["suspended"] = True
            with self.subTest(generation=generation):
                if generation == 1:
                    validate_owner_interval(adapter, source, after, terminal=True)
                else:
                    with self.assertRaisesRegex(ContractError, "arrival_owner_interval"):
                        validate_owner_interval(adapter, source, after, terminal=True)

    def test_terminal_does_not_allow_changed_mount_or_detached_part(self):
        adapter, source = fixture(tick=0, control=0)
        for change in ("mount", "part"):
            _, after = fixture(tick=6, control=1)
            after["suspended"] = True
            if change == "mount":
                after["mount_generation"] = 1
            else:
                after["parts"][1]["detached"] = True
            with self.subTest(change=change), self.assertRaisesRegex(ContractError, "arrival_owner_interval"):
                validate_owner_interval(adapter, source, after, terminal=True)

    def test_import_does_not_load_neural_runtime_or_start_host(self):
        code = "import sys; import nca.evaluation.jolt_arrival; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)

    def test_offline_final_verifier_shares_original_wall_cap_and_cannot_auto_retry(self):
        spec = importlib.util.spec_from_file_location("arrival_cli_budget", ROOT / "tools/jolt_arrival.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "single-invocation"
            args = SimpleNamespace(output=output, declaration=ROOT / "configs/baselines/jolt-arrival-diagnostic-v1.json",
                expected_declaration_sha256="269cef7a852fe8d690bb95ab9187576a3babe0f1fa5d1e8ee28fafc2730c474a",
                expected_revision="a" * 40, host=Path("synthetic-host"), godot=Path("synthetic-godot"))
            worker = mock.Mock(pid=123456, returncode=0)
            worker.poll.return_value = 0
            worker.wait.return_value = 0
            with mock.patch.object(cli.subprocess, "Popen", return_value=worker) as spawn, \
                 mock.patch.object(cli, "process_group_exists", return_value=False), \
                 mock.patch.object(cli.subprocess, "run", side_effect=subprocess.TimeoutExpired("verifier", .1)) as verifier:
                self.assertEqual(cli.supervise(args), 2)
            self.assertEqual(spawn.call_count, 1)
            self.assertEqual(verifier.call_count, 1)
            self.assertLessEqual(verifier.call_args.kwargs["timeout"], 594)
            self.assertIn("--verify-only", verifier.call_args.args[0])
            receipt = json.loads(output.with_name(output.name + "-supervisor.json").read_text())
            self.assertFalse(receipt["completed"])
            self.assertTrue(receipt["timeout"])
            self.assertEqual(receipt["attempts"], 1)
            self.assertEqual(receipt["error_type"], "OfflineVerificationTimeout")
            with self.assertRaisesRegex(ValueError, "automatic retry"):
                cli.supervise(args)

    @unittest.skipUnless(sys.platform == "linux", "process-state inspection uses Linux procfs")
    def test_supervisor_kills_descendant_after_worker_has_already_exited(self):
        spec = importlib.util.spec_from_file_location("arrival_cli", ROOT / "tools/jolt_arrival.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        # The leader exits successfully while its child remains in the owned group.
        code = ("import subprocess,sys; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],"
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print(child.pid,flush=True)")
        process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True,
                                   stdout=subprocess.PIPE, text=True)
        child_pid = int(process.stdout.readline())
        try:
            self.assertEqual(process.wait(timeout=3), 0)
            self.assertTrue(cli.process_group_exists(process.pid))
            confirmed = cli.terminate_owned_tree(process, grace_seconds=.3)
            child_state = Path(f"/proc/{child_pid}/stat")
            # Orphan zombies may await the container init's reaper. They are not
            # active workers, but the receipt must not claim an absent group.
            if child_state.exists():
                self.assertEqual(child_state.read_text().split(") ", 1)[1].split()[0], "Z")
                self.assertFalse(confirmed)
            else:
                self.assertTrue(confirmed)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
            process.stdout.close()


if __name__ == "__main__":
    unittest.main()
