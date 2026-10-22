"""Tiny CPU orchestration checks; no accepted corpus or CUDA experiment."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from nca.contracts import sha256_json
from nca.ik.data import DatasetConfig, generate_sample, iter_tasks
from nca.ik.device import configure_device
from nca.ik.reference import SolverConfig
from nca.train.ik_corpus import make_fixture_corpus
from nca.train.ik_experiment import (PLAN_PATH, allocation_for, supervise, train_attempt,
                                     validate_plan)
from nca.train.ik_run_state import IKRunLedger


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None


class IKExperimentContractTests(unittest.TestCase):
    def test_plan_is_exact_and_finite_without_admission(self):
        plan = json.loads((ROOT / PLAN_PATH).read_text())
        self.assertEqual(validate_plan(plan), plan)
        self.assertFalse(plan["measured_execution_admitted"])
        self.assertEqual(plan["training_seeds"], [41011, 42013, 43019])
        self.assertEqual(plan["epochs"] * plan["expected_dataset"]["training_rows"], 210400)
        self.assertEqual(plan["epochs"] * plan["expected_dataset"]["tuning_rows"], 80800)
        for key, value in (("epochs", 41), ("batch_size", True),
                           ("max_wall_seconds_per_seed", None), ("mode", "hybrid"),
                           ("measured_execution_admitted", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_plan(plan | {key: value})

    def test_cli_import_and_help_do_not_import_optional_runtime(self):
        program = """
import sys
sys.path.insert(0, sys.argv[1])
import nca.train.ik_experiment as experiment
assert "torch" not in sys.modules
assert "numpy" not in sys.modules
try:
    experiment.main(["--help"])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("argparse did not return help")
assert "torch" not in sys.modules
"""
        subprocess.run([sys.executable, "-S", "-c", program, str(ROOT / "src")],
                       check=True, capture_output=True)



class IKExperimentSupervisorTests(unittest.TestCase):
    def test_timeout_and_keyboard_interrupt_close_owned_disposable_child(self):
        # A real child verifies process ownership/exit. No optional runtime,
        # corpus or CUDA worker executes in this fixture.
        real_popen = subprocess.Popen
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                seed_root = root / "seed-41011"
                seed_root.mkdir()
                children = []
                info = {"active": False, "unresolved_work": False, "last_status": None,
                        "can_resume": True,
                        "remaining_wall_seconds": 4.0, "attempt_count": 0,
                        "charged_wall_seconds": 0.0}
                class Ledger:
                    def inspect(self):
                        return dict(info)
                def launch(*args, **kwargs):
                    child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    children.append(child)
                    if interrupted:
                        wait = child.wait
                        calls = [0]
                        def interrupt_once(*args, **kwargs):
                            calls[0] += 1
                            if calls[0] == 1:
                                raise KeyboardInterrupt()
                            return wait(*args, **kwargs)
                        child.wait = interrupt_once
                    return child
                declaration = {"sha256": "a" * 64, "allocations": {"41011": {}}}
                with patch("nca.train.ik_experiment.load_declaration", return_value=declaration), \
                     patch("nca.train.ik_experiment.identity_for", return_value={}), \
                     patch("nca.train.ik_experiment._review", return_value={}), \
                     patch("nca.train.ik_experiment.IKRunLedger.open", return_value=Ledger()), \
                     patch("nca.train.ik_experiment.subprocess.Popen", side_effect=launch):
                    try:
                        with self.assertRaises(ValueError):
                            supervise(root, "b" * 64, 41011, root, root, root)
                    finally:
                        for child in children:
                            if child.poll() is None:
                                child.kill()
                                child.wait(timeout=5)
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].poll())
                terminal = json.loads((seed_root / "supervisor-result-0001.json").read_text())
                self.assertEqual(terminal["interrupted"], interrupted)
                self.assertEqual(terminal["timed_out"], not interrupted)
                self.assertFalse(terminal["verified_within_charged_wall"])
                self.assertTrue(terminal["unknown_work_blocks_resume"])
                self.assertEqual(terminal["consumed_or_reserved_wall_seconds"], 4.0)

@unittest.skipUnless(OPTIONAL, "optional pinned IK Torch/NumPy runtime is not installed")
class IKExperimentCPUConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = DatasetConfig(mode="fixture", points_per_bin=(1, 1, 1),
                                   trajectories_per_partition=1, trajectory_steps=3,
                                   stress_per_class=1,
                                   teacher=SolverConfig(tolerance_cells=1e-7, max_seconds=1.0))
        cls.training = tuple(generate_sample(task, cls.config)
                             for task in iter_tasks(cls.config, "training"))
        cls.tuning = tuple(generate_sample(task, cls.config)
                           for task in iter_tasks(cls.config, "tuning"))
        cls.corpus = make_fixture_corpus(cls.config, cls.training, cls.tuning)

    def identity(self):
        return {"config_sha256": sha256_json({"fixture": "ik-training-attempt"}),
                "source_sha256": "a" * 64,
                "profile_sha256": "b" * 64,
                "dataset_sha256": self.corpus.manifest_sha256,
                "normalization_sha256": self.corpus.normalization.sha256,
                "runtime_sha256": "c" * 64,
                "declaration_sha256": "d" * 64, "seed": 41011}

    def allocation(self):
        return {"max_optimizer_steps": 32, "max_sample_presentations": len(self.training) * 2,
                "max_tuning_calls": len(self.tuning) * 2, "max_evaluation_calls": 1,
                "max_checkpoints": 10, "max_attempts": 3,
                "max_wall_seconds": 120, "max_memory_bytes": 256 * 1024**2}

    def run_fixture(self, root, *, pause=None):
        ledger = IKRunLedger.create(root, identity=self.identity(), allocation=self.allocation())
        context = configure_device("cpu", 41011)
        attempt = ledger.begin_attempt()
        result = train_attempt(ledger, attempt, self.corpus, context,
                               epochs=2, batch_size=32, pause_after_batches=pause)
        return ledger, result

    def test_actual_cpu_loop_accounts_rows_tuning_and_keeps_final_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger, result = self.run_fixture(Path(directory) / "fixture")
            info = ledger.inspect()
            self.assertTrue(result["training_complete"])
            self.assertEqual(result["completed_epochs"], 2)
            self.assertFalse(result["final_opened"])
            self.assertFalse(result["learned_quality_qualified"])
            self.assertEqual(info["charged"]["sample_presentations"], len(self.training) * 2)
            self.assertEqual(info["charged"]["tuning_calls"], len(self.tuning) * 2)
            self.assertEqual(info["charged"]["evaluation_calls"], 0)
            self.assertEqual(info["last_status"], "paused")
            self.assertFalse(info["active"])
            self.assertFalse(info["unresolved_work"])
            self.assertEqual(len(list(ledger.root.glob("tuning-epoch-*.json"))), 2)
            self.assertIsNone(result["memory"]["peak_reserved_bytes"])
            self.assertLessEqual(result["geometry_check"]["max_error_cells"], 2e-4)
            self.assertGreater(result["geometry_check"]["checked_rows"], 0)

    def test_pause_resume_consumes_same_finite_allocation_and_matches_uninterrupted(self):
        from nca.ik.checkpoint import load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted, original = self.run_fixture(root / "whole")
            ledger, first = self.run_fixture(root / "continued", pause=1)
            self.assertFalse(first["training_complete"])
            before = ledger.inspect()
            context = configure_device("cpu", 41011)
            attempt = ledger.begin_attempt()
            resumed = train_attempt(ledger, attempt, self.corpus, context, epochs=2, batch_size=32)
            after = ledger.inspect()
            self.assertTrue(resumed["training_complete"])
            self.assertEqual(after["attempt_count"], 2)
            self.assertGreater(after["charged_wall_seconds"], before["charged_wall_seconds"])
            self.assertEqual(after["charged"]["sample_presentations"],
                             uninterrupted.inspect()["charged"]["sample_presentations"])
            self.assertEqual(after["charged"]["optimizer_steps"],
                             uninterrupted.inspect()["charged"]["optimizer_steps"])
            # Verify the actual restored optimizer trajectory, not only counters.
            def payload(run_ledger):
                latest = run_ledger.inspect()["latest_checkpoint"]
                ref = latest["reference"]
                return load_checkpoint(run_ledger.root / latest["path"],
                                       expected_sha256=ref["sha256"],
                                       expected_metadata=ref["metadata"])
            whole_state, resumed_state = payload(uninterrupted), payload(ledger)
            for name in whole_state["model"]:
                self.assertTrue(context.torch.equal(whole_state["model"][name],
                                                     resumed_state["model"][name]), name)
            self.assertEqual(resumed["sampler"], original["sampler"])

    def test_accepted_marker_cannot_route_comparison_onto_cpu(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            ledger = IKRunLedger.create(Path(directory) / "fixture",
                                        identity=self.identity(), allocation=self.allocation())
            context = configure_device("cpu", 41011)
            attempt = ledger.begin_attempt()
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                train_attempt(ledger, attempt, SimpleNamespace(accepted=True), context,
                              epochs=2, batch_size=32)
            attempt.finish("failed", reason="fixture_CPU_route_refused")
            self.assertEqual(ledger.inspect()["charged"]["optimizer_steps"], 0)

    def test_allocation_derives_sampler_work_and_retains_final_timing_cost(self):
        plan = json.loads((ROOT / PLAN_PATH).read_text())
        budget = allocation_for(self.corpus, plan, 41011)
        self.assertEqual(budget["max_sample_presentations"], 40 * len(self.training))
        self.assertEqual(budget["max_tuning_calls"], 40 * len(self.tuning))
        self.assertEqual(budget["max_evaluation_calls"], 5 * (2940 + 100 + 10000))
        self.assertLessEqual(budget["max_wall_seconds"], 7200)
