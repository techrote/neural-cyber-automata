"""Terminal no-go has no training/final authority and preserves failure costs."""
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from copy import deepcopy

from nca.train import ik_followup_terminal as terminal
from nca.train.ik_run_state import IKRunLedger
from test_ik_run_state import identity, allocation, OPTIONAL, Sampler
from types import SimpleNamespace


class TerminalTests(unittest.TestCase):
    def test_duplicate_worker_refuses_before_source_or_selection_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            declaration = terminal.signed({"scope":"fixture"})
            terminal.immutable_json(root / "declaration.json", declaration)
            terminal.immutable_json(root / "worker-started.json", {"first":"preserved"})
            with patch.object(sys, "argv", ["terminal", "_derive", "--declaration", str(root / "declaration.json")]), \
                    patch.object(terminal, "_source") as source, patch.object(terminal, "derive") as derive:
                with self.assertRaisesRegex(ValueError, "immutable artifact already exists"):
                    terminal.main()
                source.assert_not_called()
                derive.assert_not_called()
            self.assertEqual(terminal.read(root / "worker-started.json"), {"first":"preserved"})

    def test_signed_but_substituted_worker_results_are_refused(self):
        declaration = {"sha256":"d"*64,"prepass":"unused"}
        gate = {"schema":"nca-ik-polar-tuning-stop/v1", "per_seed":{str(s):"no_go" for s in terminal.SEEDS},
                "final_opening_allowed":False,"reason":"all_seed_raw_quality_gate"}
        selections = {str(s):{"step":1720} for s in terminal.SEEDS}
        result = {"gate":gate,"selections":selections,"debits":{str(s):"a"*64 for s in terminal.SEEDS},
                  "terminal_declaration_sha256":"d"*64,"admission_file_sha256":"a"*64,
                  "training_authority":False,"fresh_final":"sealed_unmeasured","historical_finish_status":"failed_preserved"}
        ledgers = {s:Mock(root=Path("unused")) for s in terminal.SEEDS}
        with patch.object(terminal, "read", return_value={"seeds":{s:{"selected":v} for s,v in selections.items()}}), \
                patch.object(terminal, "sha256_file", return_value="a"*64):
            terminal.validate_result(terminal.signed(result), declaration, "review", ledgers)
            for key, value in (("training_authority",True),("fresh_final","opened"),
                               ("selections",{}),("debits",{}),("gate",dict(gate,final_opening_allowed=True)),
                               ("terminal_declaration_sha256","b"*64),("admission_file_sha256","b"*64)):
                modified = deepcopy(result); modified[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    terminal.validate_result(terminal.signed(modified), declaration, "review", ledgers)

    def test_final_publication_overrun_is_retained_without_deleting_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terminal.immutable_json(root / "terminal.json", {"no_go":True})
            with patch.object(terminal.time, "monotonic", side_effect=[899.,901.]):
                with self.assertRaisesRegex(ValueError, "final supervisor publication"):
                    terminal.publish_supervisor(root, 0.)
            self.assertTrue((root / "supervisor.json").exists())
            self.assertEqual(terminal.read(root / "overrun.json"), {"terminal_unverified":True,"resume_forbidden":True})

    def test_import_has_no_tensor_runtime(self):
        subprocess.run([sys.executable, "-c", "import sys; import nca.train.ik_followup_terminal; "
                        "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
                       check=True, capture_output=True)

    def test_passing_gate_refuses_terminal_only_authority(self):
        with patch.object(terminal, "select_training", return_value=({}, {}, {})), \
                patch.object(terminal, "tuning_stop_gate", return_value={
                    "final_opening_allowed":True,"per_seed":{"41011":"pass"}}):
            with self.assertRaisesRegex(ValueError, "all-seed tuning no-go"):
                terminal.derive("unused", {})

    def test_no_go_terminal_result_grants_no_resume_or_opening(self):
        gate = {"final_opening_allowed":False,"per_seed":{str(s):"no_go" for s in terminal.SEEDS}}
        with patch.object(terminal, "select_training", return_value=({41011:{"step":1591}}, {}, {})), \
                patch.object(terminal, "tuning_stop_gate", return_value=gate), \
                patch.dict(sys.modules, {}, clear=False):
            # Other conformance modules can have imported Torch in this process;
            # production child is separately checked by the import test above.
            with patch.object(terminal.sys, "modules", {}):
                result = terminal.derive("unused", {})
        self.assertFalse(result["training_authority"])
        self.assertEqual(result["fresh_final"], "sealed_unmeasured")
        self.assertEqual(result["historical_finish_status"], "failed_preserved")

    def test_legacy_overhead_method_cannot_accept_terminal_proof(self):
        ledger = Mock()
        with self.assertRaisesRegex(ValueError, "legacy launcher refusal"):
            IKRunLedger.record_recovery_overhead(ledger, {"schema":"nca-ik-polar-terminal-debit/v1"})
        ledger._append_overhead.assert_not_called()

    def test_permanent_marker_blocks_even_without_old_supervisor_holds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = IKRunLedger.create(root / "seed-41011", identity=identity(), allocation=allocation())
            self.assertTrue(ledger.inspect()["can_resume"])
            (root / "terminal-no-go.json").write_text("{}", encoding="utf-8")
            self.assertFalse(ledger.inspect()["can_resume"])
            with self.assertRaisesRegex(ValueError, "permanent tuning no-go"):
                ledger.begin_attempt()
            self.assertEqual(ledger.inspect()["attempt_count"], 0)

    def test_arbitrary_terminal_debit_is_refused_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = IKRunLedger.create(Path(directory) / "seed", identity=identity(), allocation=allocation())
            before = ledger.inspect()
            with self.assertRaises(ValueError):
                ledger.record_terminal_overhead({"schema":"nca-ik-polar-terminal-debit/v1"})
            self.assertEqual(ledger.inspect(), before)
            self.assertFalse((ledger.root / "overheads").exists())

    @unittest.skipUnless(OPTIONAL, "optional pinned tensor runtime for checkpoint fixture")
    def test_successful_terminal_real_ledger_debits_preserve_failed_histories(self):
        import torch
        from test_ik_followup_gate import reports
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root / "run"; run.mkdir()
            fresh = root / "fresh"; fresh.mkdir()
            terminal.immutable_json(fresh / "seal.json", {"sealed":"fixture"})
            original = terminal.signed({"source":{"source_sha256":"c"*64,"files":{}}})
            terminal.immutable_json(run / "declaration.json", original)
            ledgers = {}
            for seed in terminal.SEEDS:
                ident = dict(identity(), seed=seed, declaration_sha256=original["sha256"])
                ledger = ledgers[seed] = IKRunLedger.create(run / f"seed-{seed}", identity=ident,
                    allocation=allocation(max_wall_seconds=7200.))
                trainer = SimpleNamespace(model=torch.nn.Linear(2,2), config_sha256="2"*64,
                    normalization_sha256="3"*64, weights=SimpleNamespace(sha256="4"*64),
                    attempted_steps=0, committed_steps=0, sample_presentations=0, failed=False)
                trainer.optimizer = torch.optim.Adam(trainer.model.parameters())
                first = ledger.begin_attempt(); first.checkpoint(trainer, Sampler().state_dict())
                first.finish("paused", reason="fixture checkpoint")
                ledger.begin_attempt().finish("failed", reason="ContractError")
            terminal.immutable_json(run / "active-phase.json", {"hold":True})
            before_files = {p.relative_to(run).as_posix():terminal.pin(p) for p in run.rglob("*") if p.is_file()}
            prepass = root / "prepass.json"
            terminal.immutable_json(prepass, {"failure_files":before_files,"absent_terminal_files":["missing-result.json"]})
            source = {"revision":"f"*40,"source_sha256":"e"*64,"files":{p:"a"*64 for p in terminal.CHANGES}}
            packet = root / "terminal"; packet.mkdir(); declaration_path = packet / "declaration.json"
            patches = [patch.object(terminal,"ORIGINAL_FILE",terminal.sha256_file(run / "declaration.json")),
                       patch.object(terminal,"ORIGINAL_SOURCE","c"*64),
                       patch.object(terminal,"PREPASS_FILE",terminal.sha256_file(prepass)),
                       patch.object(terminal,"_source",return_value=source),
                       patch.object(terminal,"ledger_at",side_effect=lambda r,d,s:ledgers[s]),
                       patch.object(terminal,"ENTRY_STARTED",terminal.time.monotonic())]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for p in patches: stack.enter_context(p)
                terminal.declare(run,fresh,prepass,declaration_path)
                declaration = terminal.read(declaration_path)
                scores = reports()
                for report in scores.values(): report["nominal"].update(p95_error_cells=.3,worst_error_cells=.5,feasible_success_fraction=.2)
                selections = {s:{"step":1720} for s in terminal.SEEDS}
                with patch.object(terminal,"select_training",return_value=(selections,scores,{})), \
                        patch.object(terminal.sys,"modules",{}):
                    result = terminal.derive(run,original)
                # Independent prepass selections are separately pinned in production.
                pre = terminal.read(prepass); pre["seeds"]={str(s):{"selected":c} for s,c in selections.items()}
                # Rebuild fixture before admission, never mutate a runtime declaration.
                prepass2=root / "prepass-selected.json"; terminal.immutable_json(prepass2,pre)
                terminal.PREPASS_FILE=terminal.sha256_file(prepass2)
                declaration["prepass"]=str(prepass2)
                declaration=terminal.signed({k:v for k,v in declaration.items() if k!="sha256"})
                declaration_path=packet / "admitted-declaration.json";terminal.immutable_json(declaration_path,declaration)
                reviewers=[]
                for name in ("independent-a","independent-b"):
                    path=packet / (name+".json")
                    terminal.immutable_json(path,terminal.signed({"reviewer":name,"verdict":"approve",
                        "declaration_sha256":declaration["sha256"],"source_sha256":source["source_sha256"],
                        "checks":dict.fromkeys(terminal.CHECKS,True)}))
                    reviewers.append({"id":name,"artifact":str(path),"artifact_file_sha256":terminal.sha256_file(path)})
                review_path=packet / "review.json"
                terminal.immutable_json(review_path,terminal.signed({"schema":"nca-ik-polar-terminal-admission/v1",
                    "declaration_sha256":declaration["sha256"],"source_sha256":source["source_sha256"],
                    "reviewers":reviewers,"verdict":"admit_cpu_terminal_no_go_only"}))
                def cpu_child(*args,**kwargs):
                    result.update(terminal_declaration_sha256=declaration["sha256"],
                        admission_file_sha256=terminal.sha256_file(review_path),
                        debits={str(s):terminal.sha256_file(l.root / "overheads/charge-0001.json") for s,l in ledgers.items()})
                    terminal.immutable_json(packet / "worker-result.json",terminal.signed(result))
                    return Mock(returncode=0)
                with patch.object(terminal.subprocess,"Popen",side_effect=cpu_child):
                    receipt=terminal.execute(declaration_path,review_path)
                self.assertTrue(receipt["resume_forbidden"])
                self.assertEqual(terminal.read(packet / "terminal.json"),receipt)
                for seed,ledger in ledgers.items():
                    state=ledger.inspect(); old=declaration["boundaries"][str(seed)]
                    self.assertEqual(state["charged"],old["charged"])
                    self.assertEqual(state["latest_checkpoint"],old["latest_checkpoint"])
                    self.assertEqual(state["attempt_count"],2)
                    self.assertEqual(state["last_status"],"failed")
                    self.assertFalse(state["can_resume"])
                    self.assertGreaterEqual(state["charged_wall_seconds"]-old["charged_wall_seconds"],960.)
                for name,pin in before_files.items():self.assertEqual(terminal.pin(run / name),pin)

# Copyright 2026 techrote. All rights reserved.
