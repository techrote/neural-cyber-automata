"""Synthetic external-supervisor checks: no model fitting or host process."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from nca.contracts import canonical_json, sha256_json
from nca.evaluation.durable_journal import DurableJournal, file_identity, publish_json


ROOT = Path(__file__).resolve().parents[1]


class LearningExecutorTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("jolt_learning_cli", ROOT / "tools/learn_jolt.py")
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)
        self.cli.ENTRY_MONOTONIC = 100.

    def execute(self, root, *, cleanup=True, pending=False, late_publication=False,
                status="completed", verified=True, readback_complete=True, costs_verified=True, worker_running=False,
                worker_wait_error=None, stream_fsync_error=None, scan_error=None,
                verifier_error=None, cleanup_error=None):
        clock = [101.]
        declaration = root / "declaration.json"
        declaration.write_text(json.dumps({"budget": {"maximum_wall_seconds": 1800}}))
        args = SimpleNamespace(output=root / "output", declaration=declaration,
                               config=root / "config.json", checkout=root / "host", godot=root / "godot")
        worker = SimpleNamespace(returncode=0,
            poll=lambda: None if worker_running or worker_wait_error is not None else 0,
            wait=mock.Mock(side_effect=worker_wait_error), kill=mock.Mock())
        if worker_running:
            clock[0] = 1871.
        helper = SimpleNamespace(prepare_process_tree=mock.Mock(return_value=987),
                                 owned_processes=mock.Mock(), terminate_owned_tree=mock.Mock(), worker=worker)
        scans = 0

        def owned(identity):
            nonlocal scans
            self.assertEqual(identity, 987)
            scans += 1
            if scan_error is not None and scans == 1:
                raise scan_error
            return {123: {}} if pending else {}

        def terminate(process, identity, **kwargs):
            self.assertIs(process, worker)
            self.assertEqual(identity, 987)
            if cleanup_error is not None:
                raise cleanup_error
            return cleanup

        def publish(path, value):
            result = publish_json(path, value)
            if late_publication and path.name == "finalization.json":
                clock[0] = 1901.
            return result

        original_fsync = os.fsync

        def fsync(descriptor):
            # Only the post-worker stream synchronization fails. Admission and
            # the final durable negative receipts still use the real fsync.
            if stream_fsync_error is not None:
                path = args.output.with_name("output-lifecycle") / "stdout.log"
                if path.is_file() and os.fstat(descriptor).st_ino == path.stat().st_ino:
                    raise stream_fsync_error
            return original_fsync(descriptor)

        helper.owned_processes.side_effect = owned
        helper.terminate_owned_tree.side_effect = terminate
        verifier = {"inventory_verified": True, "readback_verified": verified,
                    "readback_complete": readback_complete, "costs_verified": costs_verified,
                    "status": status, "qualified": False}
        stream = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict("sys.modules", {"execute_jolt_diagnostic": helper}))
            stack.enter_context(mock.patch.object(self.cli.time, "monotonic", side_effect=lambda: clock[0]))
            stack.enter_context(mock.patch.object(self.cli.subprocess, "Popen", return_value=worker))
            verify = stack.enter_context(mock.patch.object(self.cli.subprocess, "run",
                return_value=SimpleNamespace(stdout=json.dumps(verifier)), side_effect=verifier_error))
            stack.enter_context(mock.patch.object(self.cli.os, "fsync", side_effect=fsync))
            stack.enter_context(mock.patch("nca.evaluation.durable_journal.publish_json", side_effect=publish))
            stack.enter_context(redirect_stdout(stream))
            code = self.cli.execute(args)
        return code, json.loads(stream.getvalue()), args, helper, verify

    def test_original_process_identity_and_complete_negative_result(self):
        with tempfile.TemporaryDirectory() as directory:
            code, verdict, args, helper, verify = self.execute(Path(directory))
            self.assertEqual(code, 0)
            self.assertTrue(verdict["completed"])
            self.assertTrue(verdict["cleanup_confirmed"])
            helper.prepare_process_tree.assert_called_once()
            verify.assert_called_once()
            self.assertTrue(args.output.with_name("output-lifecycle").joinpath("finalization.json").is_file())

    def test_descendant_cleanup_and_verification_are_completion_gates(self):
        for kwargs in ({"cleanup": False}, {"pending": True}, {"verified": False}, {"status": "capped"},
                       {"readback_complete": False}, {"costs_verified": False}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                code, verdict, _, _, _ = self.execute(Path(directory), **kwargs)
                self.assertEqual(code, 2)
                self.assertFalse(verdict["completed"])

    def test_final_publication_uses_the_same_original_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            code, verdict, _, _, _ = self.execute(Path(directory), late_publication=True)
            self.assertEqual(code, 2)
            self.assertFalse(verdict["completed"])
            self.assertEqual(verdict["wall_seconds"], 1801.)

    def test_worker_timeout_preserves_cleanup_reserve_and_never_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            code, verdict, _, helper, verify = self.execute(Path(directory), worker_running=True)
            self.assertEqual(code, 2)
            self.assertFalse(verdict["completed"])
            self.assertTrue(helper.terminate_owned_tree.called)
            verify.assert_not_called()

    def test_used_invocation_cannot_spawn_a_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, args, _, _ = self.execute(Path(directory))
            with mock.patch.object(self.cli.subprocess, "Popen") as spawn:
                with self.assertRaises(ValueError):
                    self.cli.execute(args)
                spawn.assert_not_called()

    def test_worker_baseexception_revokes_owned_tree_and_retains_failed_receipt(self):
        for interruption in (KeyboardInterrupt(), SystemExit(7)):
            with self.subTest(interruption=type(interruption).__name__), tempfile.TemporaryDirectory() as directory:
                code, verdict, args, helper, verify = self.execute(
                    Path(directory), worker_wait_error=interruption)
                self.assertEqual(code, 2)
                self.assertFalse(verdict["completed"])
                self.assertTrue(verdict["cleanup_confirmed"])
                helper.terminate_owned_tree.assert_called_once()
                verify.assert_not_called()
                receipt = json.loads(args.output.with_name("output-lifecycle").joinpath("executor.json").read_text())
                self.assertEqual(receipt["error"], {"type": type(interruption).__name__})
                self.assertFalse(receipt["completion_candidate"])
                self.assertEqual(receipt["owned_processes_remaining"], 0)

    def test_durability_scan_and_verifier_failures_always_revoke_tree(self):
        failures = ({"stream_fsync_error": OSError("synthetic stream failure")},
                    {"scan_error": OSError("synthetic process scan failure")},
                    {"verifier_error": KeyboardInterrupt()})
        for kwargs in failures:
            with self.subTest(failure=next(iter(kwargs))), tempfile.TemporaryDirectory() as directory:
                code, verdict, args, helper, _ = self.execute(Path(directory), **kwargs)
                self.assertEqual(code, 2)
                self.assertFalse(verdict["completed"])
                self.assertTrue(verdict["cleanup_confirmed"])
                helper.terminate_owned_tree.assert_called_once()
                receipt = json.loads(args.output.with_name("output-lifecycle").joinpath("executor.json").read_text())
                self.assertEqual(receipt["error"], {"type": type(next(iter(kwargs.values()))).__name__})
                self.assertFalse(receipt["completion_candidate"])

    def test_interrupted_cleanup_cannot_claim_verified_process_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            code, verdict, args, helper, _ = self.execute(Path(directory), cleanup_error=KeyboardInterrupt())
            self.assertEqual(code, 2)
            self.assertFalse(verdict["completed"])
            self.assertFalse(verdict["cleanup_confirmed"])
            helper.terminate_owned_tree.assert_called_once()
            receipt = json.loads(args.output.with_name("output-lifecycle").joinpath("executor.json").read_text())
            self.assertEqual(receipt["error"], {"type": "KeyboardInterrupt"})

    def test_live_worker_has_direct_kill_fallback_when_tree_inspection_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            code, verdict, args, helper, verify = self.execute(Path(directory),
                worker_wait_error=KeyboardInterrupt(), cleanup_error=OSError("unavailable process namespace"))
            self.assertEqual(code, 2)
            self.assertFalse(verdict["completed"])
            self.assertFalse(verdict["cleanup_confirmed"])
            helper.terminate_owned_tree.assert_called_once()
            helper.worker.kill.assert_called_once()
            self.assertEqual(helper.worker.wait.call_count, 2)
            self.assertEqual(helper.worker.wait.call_args, mock.call(timeout=3))
            verify.assert_not_called()
            receipt = json.loads(args.output.with_name("output-lifecycle").joinpath("executor.json").read_text())
            self.assertEqual(receipt["error"], {"type": "KeyboardInterrupt"})
            self.assertFalse(receipt["cleanup_confirmed"])


class ClosedLearningInventoryTests(unittest.TestCase):
    """Resealed synthetic corruptions must fail beyond simple byte hashing."""

    def setUp(self):
        spec = importlib.util.spec_from_file_location("jolt_inventory_cli", ROOT / "tools/learn_jolt.py")
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)

    @staticmethod
    def row(index, elapsed, increments, *, stage=None, cleanup=False):
        return {"index": index, "elapsed_seconds": elapsed, "stage": stage,
                "cleanup": cleanup, "increments": increments}

    @staticmethod
    def snapshot(budget, rows, *, closed, wall):
        counters = {name: sum(row["increments"].get(name, 0) for row in rows)
                    for name in budget["counters"]}
        stages = {stage: {name: sum(row["increments"].get(name, 0) for row in rows if row["stage"] == stage)
                          for name in limits} for stage, limits in budget["stages"].items()}
        return {"schema": "nca-jolt-learn-budget/v1", "counts": counters, "stage_counts": stages,
                "wall_seconds": wall, "maximum_wall_seconds": budget["maximum_wall_seconds"],
                "cleanup_reserve_seconds": budget["cleanup_reserve_seconds"],
                "reservation_count": len(rows), "reservations_sha256": sha256_json(rows),
                "closed": closed, "refused_or_uncertain_work_refunded": False}

    def fixture(self, root, *, status="completed", verified=True, complete=None):
        # These are fixture receipts, never outputs of a physical or neural run.
        budget = json.loads((ROOT / "configs/learning/jolt-learn-002-v1.json").read_text())["budget"]
        config = {"schema": "synthetic-closed-inventory-test/v1", "budget": budget}
        declaration = {"schema": "nca-learn-jolt-002-execution/v1", "budget": deepcopy(budget),
                       "config_sha256": sha256_json(config), "source_sha256": "a" * 64,
                       "automatic_retry": False, "automatic_extension": False}
        manifest = {"schema": "nca-learn-jolt-002-run/v1", "config_sha256": sha256_json(config),
                    "declaration_sha256": sha256_json(declaration), "source_sha256": declaration["source_sha256"]}
        before = [self.row(0, 1., {"attempts": 1}),
                  self.row(1, 2., {"episodes_started": 1}, stage="teacher"),
                  self.row(2, 3., {"decisions": 1, "requested_ticks": 6, "owner_step": 1,
                                   "owner_requests": 1}, stage="teacher"),
                  self.row(3, 4., {"confirmed_ticks": 6, "episodes_completed": 1}, stage="teacher", cleanup=True),
                  self.row(4, 4.5, {"host_terminations": 1}, cleanup=True)]
        after = deepcopy(before) + [self.row(5, 6., {"checkpoint_loads": 1}, cleanup=True),
                                    self.row(6, 7., {"audit_forward": 2}, cleanup=True)]
        with DurableJournal(root / "owner-journal.jsonl") as journal:
            journal.write(b'{"event":"synthetic_closed_inventory_fixture"}\n')
        qualification = {"schema": "nca-jolt-learning-qualification/v1", "qualified": False,
                         "evidence_complete": False, "synthetic_fixture": True}
        worker = {"schema": manifest["schema"], "status": status, "error": None,
                  "manifest": deepcopy(manifest), "closure": {"durability": journal.durability_receipt()},
                  "costs_before_readback": self.snapshot(budget, before, closed=False, wall=5.),
                  "qualification": deepcopy(qualification)}
        complete = verified and status == "completed" if complete is None else complete
        audit = {"schema": "nca-learn-jolt-002-independent-readback/v1", "verified": verified,
                 "complete": complete, "qualified": False, "audit_forward": 2, "checkpoint_loads": 1}
        if verified:
            audit.update(worker_status=status, config_sha256=sha256_json(config),
                         source_sha256=declaration["source_sha256"],
                         costs_reconstructed=deepcopy(worker["costs_before_readback"]["counts"]),
                         qualification={**qualification, "evidence_complete": complete})
        result = {"schema": manifest["schema"], "status": status, "error": None,
                  "closure": deepcopy(worker["closure"]),
                  "costs": self.snapshot(budget, after, closed=True, wall=8.),
                  "qualification": deepcopy(audit["qualification"] if complete else qualification)}
        docs = {"config.json": config, "declaration.json": declaration, "manifest.json": manifest,
                "worker-result.json": worker, "independent-readback.json": audit, "result.json": result,
                "budget-reservations.json": {"schema": "jolt-budget-reservations/v1", "reservations": before},
                "closed-budget-reservations.json": {"schema": "jolt-budget-reservations/v1", "reservations": after}}
        self.publish(root, docs)
        return docs

    @staticmethod
    def publish(root, docs, *, rebind=True):
        # A deliberate corruption is rehashed into its inventory so each test
        # exercises semantic reconciliation, not merely a stale file digest.
        ordered = [name for name in docs if name not in {"worker-result.json", "independent-readback.json", "result.json"}]
        for name in ordered + ["worker-result.json", "independent-readback.json", "result.json"]:
            if rebind and name == "independent-readback.json" and docs[name]["verified"]:
                docs[name]["worker_result_sha256"] = file_identity(root / "worker-result.json")["sha256"]
            if rebind and name == "result.json":
                docs[name]["worker_result_sha256"] = file_identity(root / "worker-result.json")["sha256"]
                docs[name]["audit_sha256"] = file_identity(root / "independent-readback.json")["sha256"]
            (root / name).write_text(canonical_json(docs[name]) + "\n")
        files = {path.name: file_identity(path) for path in root.iterdir() if path.name != "inventory.json"}
        (root / "inventory.json").write_text(canonical_json({"schema": "jolt-closed-inventory/v1", "files": files}) + "\n")

    def recount(self, docs):
        budget = docs["config.json"]["budget"]
        before = docs["budget-reservations.json"]["reservations"]
        after = docs["closed-budget-reservations.json"]["reservations"]
        worker, result, audit = (docs[name] for name in ("worker-result.json", "result.json", "independent-readback.json"))
        worker["costs_before_readback"] = self.snapshot(budget, before, closed=False,
                                                     wall=worker["costs_before_readback"]["wall_seconds"])
        result["costs"] = self.snapshot(budget, after, closed=True, wall=result["costs"]["wall_seconds"])
        if audit["verified"]:
            audit["costs_reconstructed"] = deepcopy(worker["costs_before_readback"]["counts"])

    def test_complete_negative_and_retained_partial_costs_reconcile(self):
        for kwargs in ({}, {"status": "capped"}, {"verified": False}, {"complete": False}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                docs = self.fixture(root, **kwargs)
                actual = self.cli.verify_inventory(root)
                self.assertTrue(actual["inventory_verified"])
                self.assertTrue(actual["costs_verified"])
                self.assertFalse(actual["qualified"])
                self.assertEqual(actual["readback_complete"], docs["independent-readback.json"]["complete"])
                self.assertEqual(actual["closed_costs"]["readback_deltas"], {"audit_forward": 2, "checkpoint_loads": 1})

    def test_final_prefix_cannot_change_even_when_all_files_are_rehashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = self.fixture(root)
            docs["closed-budget-reservations.json"]["reservations"][0]["elapsed_seconds"] = 1.5
            self.recount(docs)
            self.publish(root, docs)
            with self.assertRaisesRegex(ValueError, "immutable pre-audit"):
                self.cli.verify_inventory(root)

    def test_snapshot_hash_count_totals_stages_and_closed_state_are_reconstructed(self):
        mutations = [lambda value: value.update(reservations_sha256="0" * 64),
                     lambda value: value.update(reservation_count=6),
                     lambda value: value["counts"].update(audit_forward=1),
                     lambda value: value["stage_counts"]["teacher"].update(decisions=0),
                     lambda value: value.update(closed=False),
                     lambda value: value.update(refused_or_uncertain_work_refunded=True)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                docs = self.fixture(root)
                mutate(docs["result.json"]["costs"])
                self.publish(root, docs)
                with self.assertRaisesRegex(ValueError, "snapshot"):
                    self.cli.verify_inventory(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = self.fixture(root)
            docs["worker-result.json"]["costs_before_readback"]["closed"] = True
            self.publish(root, docs)
            with self.assertRaisesRegex(ValueError, "snapshot"):
                self.cli.verify_inventory(root)

    def test_post_prefix_allows_only_unchanged_cleanup_readback_operations(self):
        mutations = [lambda row: row.update(cleanup=False), lambda row: row.update(stage="sealed"),
                     lambda row: row.update(increments={"checkpoint_writes": 1})]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                docs = self.fixture(root)
                mutate(docs["closed-budget-reservations.json"]["reservations"][-1])
                self.recount(docs)
                self.publish(root, docs)
                with self.assertRaisesRegex(ValueError, "new ordinary work"):
                    self.cli.verify_inventory(root)

    def test_readback_delta_and_pre_readback_totals_match_independent_audit(self):
        for field in ("audit_forward", "checkpoint_loads", "costs_reconstructed"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                docs = self.fixture(root)
                if field == "costs_reconstructed":
                    docs["independent-readback.json"][field]["decisions"] = 0
                else:
                    docs["independent-readback.json"][field] += 1
                self.publish(root, docs)
                with self.assertRaisesRegex(ValueError, "independently reported|cost allocation"):
                    self.cli.verify_inventory(root)

    def test_counter_and_stage_caps_cannot_be_hidden_by_recounting_snapshots(self):
        for amount, reason in ((20001, "counter exceeded"), (2561, "stage exceeded")):
            with self.subTest(amount=amount), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                docs = self.fixture(root)
                for name in ("budget-reservations.json", "closed-budget-reservations.json"):
                    docs[name]["reservations"][2]["increments"]["decisions"] = amount
                self.recount(docs)
                self.publish(root, docs)
                with self.assertRaisesRegex(ValueError, reason):
                    self.cli.verify_inventory(root)

    def test_original_active_cleanup_and_monotonic_time_bounds_remain_closed(self):
        changes = [("prefix", 2, 1680., "monotonic deadline"),
                   ("prefix", 2, 1., "monotonic deadline"),
                   ("full", 6, 1800., "monotonic deadline"),
                   ("full", 5, 4.75, "predates")]
        for target, index, elapsed, reason in changes:
            with self.subTest(target=target, elapsed=elapsed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                docs = self.fixture(root)
                if target == "prefix":
                    docs["budget-reservations.json"]["reservations"][index]["elapsed_seconds"] = elapsed
                docs["closed-budget-reservations.json"]["reservations"][index]["elapsed_seconds"] = elapsed
                self.recount(docs)
                self.publish(root, docs)
                with self.assertRaisesRegex(ValueError, reason):
                    self.cli.verify_inventory(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = self.fixture(root)
            docs["closed-budget-reservations.json"]["reservations"] = deepcopy(docs["budget-reservations.json"]["reservations"])
            docs["result.json"]["costs"]["wall_seconds"] = 4.75
            docs["independent-readback.json"].update(audit_forward=0, checkpoint_loads=0)
            self.recount(docs)
            self.publish(root, docs)
            with self.assertRaisesRegex(ValueError, "renewed"):
                self.cli.verify_inventory(root)

    def test_final_receipt_and_audit_bind_exact_worker_and_config(self):
        cases = [("result.json", "worker_result_sha256"), ("result.json", "audit_sha256"),
                 ("independent-readback.json", "worker_result_sha256"),
                 ("independent-readback.json", "config_sha256"),
                 ("independent-readback.json", "source_sha256")]
        for name, field in cases:
            with self.subTest(name=name, field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                docs = self.fixture(root)
                docs[name][field] = "0" * 64
                # Keep the corrupted binding, but authenticate the altered audit
                # bytes in the final receipt to reach the worker-binding check.
                if name == "independent-readback.json":
                    (root / name).write_text(canonical_json(docs[name]) + "\n")
                    docs["result.json"]["audit_sha256"] = file_identity(root / name)["sha256"]
                self.publish(root, docs, rebind=False)
                with self.assertRaisesRegex(ValueError, "binding|another worker"):
                    self.cli.verify_inventory(root)

    def test_partial_allocation_and_failed_audit_cannot_be_promoted(self):
        changes = [lambda docs: docs["result.json"].update(status="completed"),
                   lambda docs: docs["independent-readback.json"].update(complete=True),
                   lambda docs: docs["independent-readback.json"].update(qualified=True),
                   lambda docs: docs["result.json"]["qualification"].update(qualified=True)]
        for index, mutate in enumerate(changes):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                docs = self.fixture(root, status="capped")
                mutate(docs)
                self.publish(root, docs)
                with self.assertRaisesRegex(ValueError, "outcome|complete evidence|qualification"):
                    self.cli.verify_inventory(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = self.fixture(root)
            docs["result.json"]["qualification"]["synthetic_fixture"] = False
            self.publish(root, docs)
            with self.assertRaisesRegex(ValueError, "qualification differs"):
                self.cli.verify_inventory(root)

    def test_reservations_over_one_mebibyte_verify_in_a_fresh_stdlib_only_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = self.fixture(root)
            rows = docs["closed-budget-reservations.json"]["reservations"][:6]
            rows.extend(self.row(len(rows), 7., {"audit_forward": 1}, cleanup=True) for _ in range(12000))
            docs["closed-budget-reservations.json"]["reservations"] = rows
            docs["independent-readback.json"]["audit_forward"] = 12000
            self.recount(docs)
            self.publish(root, docs)
            self.assertGreater((root / "closed-budget-reservations.json").stat().st_size, 1024 * 1024)
            script = ("import importlib.util,sys; "
                      "spec=importlib.util.spec_from_file_location('inventory',sys.argv[1]); "
                      "cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli); "
                      "result=cli.verify_inventory(sys.argv[2]); "
                      "assert result['costs_verified'] and result['readback_complete']; "
                      "assert 'torch' not in sys.modules and 'numpy' not in sys.modules")
            checked = subprocess.run([sys.executable, "-c", script, str(ROOT / "tools/learn_jolt.py"), str(root)],
                                     capture_output=True, text=True, timeout=15, check=False)
            self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_oversized_artifact_is_refused_before_loading_its_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bounded.json"
            path.write_text("{}")
            with mock.patch.object(Path, "read_bytes") as read:
                with self.assertRaisesRegex(ValueError, "bounded reader"):
                    self.cli._read_json(path, limit=1)
                read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
