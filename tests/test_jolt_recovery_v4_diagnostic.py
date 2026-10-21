"""Offline v4 source admission, fixed allocation and absolute transport limits."""

import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest import mock

from nca.adapters.protocol import JsonChannel, ProtocolError
from nca.contracts import ContractError, sha256_json
from nca.data import read_episode
from nca.evaluation import jolt_arrival as arrival
from nca.evaluation import jolt_recovery_acceptance as acceptance
from nca.evaluation.jolt_recovery_acceptance import (
    CANDIDATE, CANDIDATE_V3, CANDIDATE_V4, acceptance_specification, assess_teacher,
)
from test_jolt_decision_refusals import SyntheticController, SyntheticOwner
from test_jolt_recovery_acceptance import fixture as acceptance_fixture


ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "configs/baselines/jolt-recovery-diagnostic-v4.json"


class RecoveryV4DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.declaration = json.loads(DECLARATION.read_bytes())
        self.now = [100.0]

    def budget(self, journal=None):
        return arrival.DiagnosticBudget(self.declaration["budget"], io.BytesIO() if journal is None else journal,
            clock=lambda: self.now[0], whole_process_started=100.0)

    def reject_before_host(self, value, *, started=None, missing_start=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "declaration.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            supplied = None if missing_start else (time.monotonic() if started is None else started)
            with mock.patch.object(arrival, "GodotHost") as host, self.assertRaises(ContractError):
                arrival.run_diagnostic(root=ROOT, checkout=root / "unused", godot=root / "unused",
                    declaration_path=path, output=root / "must-not-start", expected_revision="a" * 40,
                    expected_declaration_sha256=sha256_json(value), whole_process_started=supplied)
            host.assert_not_called()
            self.assertFalse((root / "must-not-start").exists())

    def test_declaration_keeps_all_cases_references_and_acceptance_except_candidate_identity(self):
        value = arrival.read_declaration(DECLARATION, expected_sha256=sha256_json(self.declaration))
        previous = json.loads((ROOT / "configs/baselines/jolt-recovery-diagnostic-v3.json").read_bytes())
        for key in ("case_source", "goal", "physics"):
            self.assertEqual(value[key], previous[key])
        stripped = lambda spec: {key: val for key, val in spec.items() if key not in ("schema", "candidate")}
        self.assertEqual(stripped(value["teacher_acceptance"]), stripped(previous["teacher_acceptance"]))
        self.assertEqual(value["budget"], arrival.V4_BUDGET)
        original = json.loads((ROOT / value["case_source"]["path"]).read_bytes())
        self.assertEqual(value["case_inventory"], arrival.diagnostic_cases(original))
        self.assertEqual(len(value["case_inventory"]), 32)
        self.assertEqual(value["case_inventory_sha256"], sha256_json(value["case_inventory"]))
        self.assertEqual(value["prior_invocations"][:-1], previous["prior_invocations"])
        self.assertFalse(value["prior_invocations"][-1]["whole_invocation_qualified"])
        for family, spec in value["candidate_specifications"].items():
            self.assertEqual(arrival._controller(family).specification(), spec)

    def test_every_declared_budget_key_is_required_and_cannot_expand_or_change_type(self):
        for key, expected in arrival.V4_BUDGET.items():
            with self.subTest(missing=key):
                value = copy.deepcopy(self.declaration)
                value["budget"].pop(key)
                self.reject_before_host(value)
            for changed in ((True if type(expected) is int else 1),
                            expected + 1 if type(expected) is int else not expected):
                with self.subTest(key=key, changed=changed):
                    value = copy.deepcopy(self.declaration)
                    value["budget"][key] = changed
                    self.reject_before_host(value)

    def test_changed_case_seed_history_acceptance_or_preserved_source_refuses(self):
        mutations = (
            lambda d: d["case_inventory"][0].update(host_seed=999),
            lambda d: d["prior_invocations"][-1].update(whole_invocation_qualified=True),
            lambda d: d["teacher_acceptance"].update(minimum_historical_rescues=2),
            lambda d: d["teacher_acceptance"].update(maximum_regressions=2),
            lambda d: d["goal"].update(deadline_ticks=246),
            lambda d: d["physics"].update(decision_ticks=12),
            lambda d: d["preserved_sources"].update({"src/nca/controllers/jolt_recovery_v3.py": "a" * 64}),
            lambda d: d["candidate_specifications"].update({CANDIDATE_V4: {}}),
            lambda d: d.update(families=list(arrival.RECOVERY_V3_FAMILIES)),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                value = copy.deepcopy(self.declaration)
                mutate(value)
                self.reject_before_host(value)

    def test_missing_future_nonfinite_or_nonpositive_original_clock_refuses_before_host(self):
        self.reject_before_host(self.declaration, missing_start=True)
        for started in (True, 0., -1., float("inf"), float("nan"), time.monotonic() + 1000):
            with self.subTest(started=started):
                self.reject_before_host(self.declaration, started=started)

    def test_prior_controller_and_declaration_bytes_remain_replayable(self):
        for version, digest in (
            (1, "2e5d4e3c554ec2382d10a95d347c1f933252050f2a18dff320711c2cc3c7fff7"),
            (2, "d7cc13335b453a755e9b7d950ac52d982aff553c025cbe0fbd47b81b5b0729eb"),
            (3, "03c58b439b808581a63541a8dbb005a4b73486acd9afd2a0fd635f1604abfb03"),
        ):
            value = arrival.read_declaration(ROOT / f"configs/baselines/jolt-recovery-diagnostic-v{version}.json",
                                             expected_sha256=digest)
            family = f"jolt-recovery-v{version}"
            self.assertEqual(arrival._controller(family).specification(), value["candidate_specifications"][family])
        for path, digest in self.declaration["preserved_sources"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)

    def test_effective_96_by_40_by_six_ceiling_cannot_use_larger_nominal_global_budget(self):
        budget = self.budget()
        budget.start_host()
        for index in range(96):
            budget.start_episode(f"synthetic-{index}", CANDIDATE_V4)
            for _ in range(40):
                serial = budget.reserve("step", {"ticks": 6})
                budget.confirm(serial, {"actual_ticks": 6})
            with self.assertRaisesRegex(ContractError, "arrival_step_budget"):
                budget.reserve("step", {"ticks": 6})
            budget.note("episode_completed", case=f"synthetic-{index}", family=CANDIDATE_V4)
        self.assertEqual(budget.counts["step_requests"], 3840)
        self.assertEqual(budget.counts["requested_step_ticks"], 23040)
        with self.assertRaisesRegex(ContractError, "arrival_episode_budget"):
            budget.start_episode("not-an-extra-attempt", CANDIDATE_V4)
        with self.assertRaisesRegex(ContractError, "arrival_host_budget"):
            budget.start_host()

    def test_exact_global_owner_limit_also_counts_native_cleanup(self):
        budget = self.budget()
        budget.counts["host_requests"] = 4499
        serial = budget.reserve("close", {})
        budget.confirm(serial, {"closed": True})
        self.assertEqual(budget.counts["host_requests"], 4500)
        with self.assertRaisesRegex(ContractError, "arrival_owner_budget"):
            budget.reserve("close", {})

    def test_normal_admission_reserves_15_seconds_before_780_and_cleanup_uses_total_clock(self):
        budget = self.budget()
        budget.start_episode("synthetic", CANDIDATE_V4)
        self.now[0] = 865.0  # 765 seconds after outer process start.
        serial = budget.reserve("step", {"ticks": 6})
        budget.confirm(serial, {"actual_ticks": 6})
        self.now[0] = 865.001
        before = copy.deepcopy(budget.counts)
        with self.assertRaisesRegex(ContractError, "arrival_shutdown_reserve"):
            budget.reserve("step", {"ticks": 6})
        self.assertEqual(before, budget.counts)
        self.now[0] = 890.0
        serial = budget.reserve("cancel", {})
        budget.confirm(serial, {"cancelled": True})
        self.now[0] = 999.0
        serial = budget.reserve("close", {})
        budget.confirm(serial, {"closed": True})
        self.now[0] = 1000.0
        with self.assertRaisesRegex(ContractError, "arrival_wall_budget"):
            budget.reserve("close", {})
        self.assertEqual(budget.snapshot()["whole_process_started_monotonic"], 100.0)

    def test_slow_durable_reservation_cannot_dispatch_after_active_boundary(self):
        class SlowJournal(io.BytesIO):
            delay = False
            def flush(inner):
                if inner.delay:
                    self.now[0] += 16.0
                return super().flush()
        journal = SlowJournal()
        budget = self.budget(journal)
        budget.start_episode("synthetic", CANDIDATE_V4)
        self.now[0] = 850.0
        journal.delay = True
        host = mock.Mock()
        with self.assertRaisesRegex(ContractError, "arrival_shutdown_reserve"):
            arrival.MeteredHost(host, budget).request("step", ticks=6)
        host.request.assert_not_called()
        self.assertEqual(budget.counts["step_requests"], 1)
        self.assertEqual(budget.counts["confirmed_step_ticks"], 0)
        self.assertTrue(budget.snapshot()["unconfirmed_tail"])

    def test_v4_rows_cannot_relabel_v3_and_retains_separate_learning_gate(self):
        rows = acceptance_fixture()
        for row in rows:
            if row["family"] == CANDIDATE:
                row["family"] = CANDIDATE_V3
        with self.assertRaises(ContractError):
            assess_teacher(rows, pulse_authority_verified=True, candidate_family=CANDIDATE_V4)
        for row in rows:
            if row["family"] == CANDIDATE_V3:
                row["family"] = CANDIDATE_V4
        result = assess_teacher(rows, pulse_authority_verified=True, candidate_family=CANDIDATE_V4)
        self.assertTrue(result["teacher_accepted"])
        self.assertFalse(result["learned_skill_qualified"])
        self.assertFalse(result["morphology_admitted"])

    def test_invalid_v4_supervisor_lease_never_reaches_synthetic_owner(self):
        for lease in (0, 13, True):
            with self.subTest(lease=lease), tempfile.TemporaryDirectory() as folder:
                owner, controller = SyntheticOwner(), SyntheticController()
                controller.diagnostic_state = lambda: {"tick": controller.tick,
                    "action_lease_ticks": lease, "supervisor_intervention": False, "supervisor_reason": None}
                case = {"id": "synthetic-invalid-lease", "family": "synthetic", "host_seed": 1,
                    "spawn": owner.view["position"],
                    "target": [owner.view["position"][0] + 100., owner.view["position"][1]],
                    "origin": "previously_open_training"}
                budget = self.budget()
                with mock.patch.object(arrival, "_controller", return_value=controller), \
                     mock.patch.object(arrival, "GodotHost", side_effect=AssertionError("no physics")), \
                     mock.patch("nca.data.capture.source_identity", return_value=("b" * 40, {}, sha256_json({}))):
                    row = arrival._capture(arrival.MeteredHost(owner, budget), budget, case, CANDIDATE_V4,
                        Path(folder), {"profile_sha256": sha256_json(owner.profile)}, self.declaration, ROOT)
                self.assertEqual(owner.steps, 0)
                self.assertEqual(row["error"]["code"], "arrival_supervisor_lease")
                self.assertTrue(row["error_cleanup"]["confirmed"])
                self.assertEqual(read_episode(Path(folder) / row["artifact"]).decisions, 0)
                self.assertFalse(any(path.name.endswith(".pending") for path in Path(folder).rglob("*")))


class AbsoluteExchangeTests(unittest.TestCase):
    class TrickleSocket:
        def __init__(self, now, *, send_delay=3., receive_delays=(6., 6., 6.)):
            self.now, self.timeout = now, 15.
            self.send_delay, self.receive_delays = send_delay, list(receive_delays)
            self.chunks = [b'{"actual_', b'ticks":', b'6}\n']
            self.timeouts = []
        def gettimeout(self):
            return self.timeout
        def settimeout(self, value):
            self.timeout = value
            self.timeouts.append(value)
        def advance(self, duration):
            if duration >= self.timeout:
                self.now[0] += self.timeout
                raise TimeoutError("synthetic socket timeout")
            self.now[0] += duration
        def sendall(self, _):
            self.advance(self.send_delay)
        def recv(self, _):
            self.advance(self.receive_delays.pop(0))
            return self.chunks.pop(0)

    def exchange(self, *, elapsed=0., operation="step", send_delay=3., receive_delays=(6., 6., 6.)):
        now = [100. + elapsed]
        budget = arrival.DiagnosticBudget(arrival.V4_BUDGET, io.BytesIO(),
            clock=lambda: now[0], whole_process_started=100.)
        if operation == "step":
            budget.start_episode("synthetic", CANDIDATE_V4)
        connection = self.TrickleSocket(now, send_delay=send_delay, receive_delays=receive_delays)
        channel = JsonChannel(connection)
        class Owner:
            _channel = channel
            def request(inner, operation, **fields):
                channel.send({"op": operation, **fields})
                return channel.receive()
        self.now, self.budget, self.connection, self.channel = now, budget, connection, channel
        return arrival.MeteredHost(Owner(), budget).request(operation, **({"ticks": 6} if operation == "step" else {}))

    def test_trickled_replies_share_one_15_second_deadline_including_send(self):
        with self.assertRaises(ProtocolError):
            self.exchange()
        self.assertEqual(self.now[0], 115.)
        self.assertTrue(self.budget.snapshot()["unconfirmed_tail"])
        self.assertEqual(self.budget.counts["requested_step_ticks"], 6)
        self.assertEqual(self.budget.counts["confirmed_step_ticks"], 0)
        self.assertIs(self.channel.connection, self.connection)
        self.assertEqual(self.connection.gettimeout(), 15.)
        self.assertIn(6., self.connection.timeouts)

    def test_complete_fast_exchange_confirms_once_and_restores_original_socket(self):
        self.assertEqual(self.exchange(send_delay=1., receive_delays=(1., 1., 1.)), {"actual_ticks": 6})
        self.assertEqual(self.budget.counts["confirmed_step_ticks"], 6)
        self.assertFalse(self.budget.snapshot()["unconfirmed_tail"])
        self.assertIs(self.channel.connection, self.connection)
        self.assertEqual(self.connection.gettimeout(), 15.)

    def test_cleanup_exchange_is_capped_by_remaining_whole_process_time(self):
        with self.assertRaises(ProtocolError):
            self.exchange(elapsed=897., operation="close", send_delay=1., receive_delays=(2., 2., 2.))
        self.assertEqual(self.now[0], 1000.)
        self.assertTrue(self.budget.snapshot()["unconfirmed_tail"])
        self.assertEqual(self.budget.counts["host_requests"], 1)


class V4OuterReceiptBindingTests(unittest.TestCase):
    """Exercise final audit wiring with explicit synthetic upstream verdicts."""

    def test_exact_reserves_finalization_and_original_external_exit_are_required(self):
        declaration = json.loads(DECLARATION.read_bytes())
        revision, declaration_sha = "b" * 40, sha256_json(declaration)
        rows = acceptance_fixture()
        for row in rows:
            if row["family"] == CANDIDATE:
                row["family"] = CANDIDATE_V4
        verified = {"verified": True, "synthetic_upstream_verdict": True}
        manifest = {"declaration": declaration, "whole_process_started_monotonic": 100.,
            "source_files": {"tools/execute_jolt_diagnostic.py": "e" * 64}}
        summary = {"rows": rows, "counts": {"whole_process_started_monotonic": 100.}}
        supervisor = {"completed": True, "source_revision": revision,
            "declaration_sha256": declaration_sha, "attempts": 1, "timeout": False,
            "worker_terminated": True, "owned_process_group_terminated": True,
            "final_output_verification": verified, "wall_seconds": 300.,
            "whole_process_started_monotonic": 100., "cleanup_reserve_seconds": 120,
            "independent_verification_reserve_seconds": 60, "supervisor_finalization_reserve_seconds": 6}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "synthetic-run"
            output.mkdir()
            (output / "manifest.json").write_text(json.dumps(manifest))
            (output / "summary.json").write_text(json.dumps(summary))
            supervisor_path = output.with_name(output.name + "-supervisor.json")
            finalization_path = output.with_name(output.name + "-finalization.json")
            executor_path = output.with_name(output.name + "-executor.json")
            stream_paths = [output.with_name(output.name + suffix)
                for suffix in ("-executor-stdout.log", "-executor-stderr.log")]
            def write_receipts(outer=supervisor, *, wall=300.1, completed=True, linked_sha=None,
                               executor_change=None, terminal_change=None):
                supervisor_path.write_text(json.dumps(outer))
                finalization_path.write_text(json.dumps({
                    "schema": "nca-jolt-arrival-finalization/v1", "completed": completed,
                    "source_revision": revision, "declaration_sha256": declaration_sha,
                    "whole_process_started_monotonic": 100., "wall_seconds": wall,
                    "supervisor_sha256": linked_sha or hashlib.sha256(supervisor_path.read_bytes()).hexdigest()}))
                terminal = {"completed": True, "returncode": 0, "timeout": False,
                    "wall_seconds": 300.2,
                    "finalization_sha256": hashlib.sha256(finalization_path.read_bytes()).hexdigest()}
                terminal.update(terminal_change or {})
                stream_paths[0].write_text(json.dumps(terminal) + "\n")
                stream_paths[1].write_bytes(b"")
                executor = {"schema": "nca-jolt-arrival-executor/v1", "completed": True,
                    "source_revision": revision, "declaration_sha256": declaration_sha,
                    "executor_source_sha256": "e" * 64, "maximum_wall_seconds": 900,
                    "attempts": 1, "returncode": 0, "timeout": False, "error_type": None,
                    "supervisor_terminated": True, "owned_process_tree_terminated": True,
                    "whole_process_started_monotonic": 100., "process_exited_monotonic": 400.3,
                    "wall_seconds": 400.3 - 100., "terminal_report": terminal,
                    "supervisor_sha256": hashlib.sha256(supervisor_path.read_bytes()).hexdigest(),
                    "finalization_sha256": hashlib.sha256(finalization_path.read_bytes()).hexdigest(),
                    "streams": {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                            "bytes": path.stat().st_size} for path in stream_paths}}
                executor.update(executor_change or {})
                executor_path.write_text(json.dumps(executor))
            def audit():
                return acceptance.audit_recovery(output, expected_revision=revision,
                    expected_declaration_sha256=declaration_sha)
            with mock.patch.object(arrival, "verify_diagnostic_output", return_value=verified), \
                 mock.patch.object(acceptance, "verify_replay_source", return_value={"verified": True}), \
                 mock.patch.object(acceptance, "verify_controller_replay", return_value={"verified": True}):
                write_receipts()
                result = audit()
                self.assertTrue(result["teacher_accepted"])
                self.assertEqual(result["finalization_sha256"], hashlib.sha256(finalization_path.read_bytes()).hexdigest())
                self.assertEqual(result["executor_sha256"], hashlib.sha256(executor_path.read_bytes()).hexdigest())
                wrong_key = {**supervisor, "finalization_reserve_seconds": 6}
                wrong_key.pop("supervisor_finalization_reserve_seconds")
                write_receipts(wrong_key)
                with self.assertRaisesRegex(ContractError, "prospectively divided reserve"):
                    audit()
                for change in ({"wall": 900.1}, {"wall": 299.9}, {"completed": False}, {"linked_sha": "f" * 64}):
                    with self.subTest(change=change):
                        write_receipts(**change)
                        with self.assertRaisesRegex(ContractError, "post-receipt finalization"):
                            audit()
                for change in ({"wall_seconds": 900.1, "process_exited_monotonic": 1000.1},
                               {"completed": False}, {"returncode": 2}, {"returncode": False},
                               {"attempts": 2}, {"attempts": True}, {"timeout": True},
                               {"supervisor_terminated": False}, {"owned_process_tree_terminated": False},
                               {"error_type": "LateExit"}, {"whole_process_started_monotonic": 101.},
                               {"executor_source_sha256": "f" * 64}, {"supervisor_sha256": "f" * 64},
                               {"finalization_sha256": "f" * 64}, {"wall_seconds": 300.0}):
                    with self.subTest(executor=change):
                        write_receipts(executor_change=change)
                        with self.assertRaisesRegex(ContractError, "original external executor"):
                            audit()
                for change in ({"completed": False}, {"returncode": 2}, {"timeout": True},
                               {"wall_seconds": 300.0}, {"wall_seconds": 301.},
                               {"finalization_sha256": "f" * 64}):
                    with self.subTest(terminal=change):
                        write_receipts(terminal_change=change)
                        with self.assertRaisesRegex(ContractError, "original external executor"):
                            audit()
                write_receipts(executor_change={"streams": {}})
                with self.assertRaisesRegex(ContractError, "stdout/stderr inventory"):
                    audit()
                for path in stream_paths:
                    with self.subTest(changed_stream=path.name):
                        write_receipts()
                        path.write_bytes(path.read_bytes() + b"changed")
                        with self.assertRaisesRegex(ContractError, "stream bytes differ"):
                            audit()
                write_receipts()
                stream_paths[0].write_bytes(b"synthetic supervisor diagnostic prefix\n" + stream_paths[0].read_bytes())
                executor = json.loads(executor_path.read_bytes())
                executor["streams"][stream_paths[0].name] = {
                    "sha256": hashlib.sha256(stream_paths[0].read_bytes()).hexdigest(),
                    "bytes": stream_paths[0].stat().st_size}
                executor_path.write_text(json.dumps(executor))
                self.assertTrue(audit()["teacher_accepted"])
                write_receipts()
                stream_paths[0].write_text(json.dumps({"completed": False}) + "\n")
                executor = json.loads(executor_path.read_bytes())
                executor["streams"][stream_paths[0].name] = {
                    "sha256": hashlib.sha256(stream_paths[0].read_bytes()).hexdigest(),
                    "bytes": stream_paths[0].stat().st_size}
                executor_path.write_text(json.dumps(executor))
                with self.assertRaisesRegex(ContractError, "terminal report differs"):
                    audit()


if __name__ == "__main__":
    unittest.main()
