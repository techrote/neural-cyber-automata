"""Budget and duration predicates for the explicitly separate deployment lane."""

from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch"))
if HAS_RUNTIME:
    from nca.contracts import ContractError, WireAction, canonical_json
    from nca.bench.torch_screen import WorkBudget, BudgetHost, load_declaration, throughput_complete, require_consumed_decision

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_RUNTIME, "optional Torch and NumPy dependencies unavailable")
class TorchScreenTests(unittest.TestCase):
    def setUp(self):
        self.declaration, _ = load_declaration(ROOT / "configs/export/torch-cpu-followup-v1.json")

    def test_count_only_short_run_cannot_pass_the_full_ten_minute_duration(self):
        setting = self.declaration["throughput"]
        rows = [{"configuration": c, "decisions": 500, "active_window_seconds": seconds, "completed": True}
                for c, seconds in zip(setting["actor_configurations"], (45., 32.))]
        self.assertFalse(throughput_complete(rows, setting, phase_seconds=79.))
        for row in rows:
            row["active_window_seconds"] = 300.01
        self.assertTrue(throughput_complete(rows, setting, phase_seconds=605.))
        rows[1]["active_window_seconds"] = 299.99
        self.assertFalse(throughput_complete(rows, setting, phase_seconds=605.))

    def test_duration_requires_both_declared_configs_completed_within_all_caps(self):
        setting = self.declaration["throughput"]
        rows = [{"configuration": c, "decisions": 500, "active_window_seconds": 300., "completed": True}
                for c in setting["actor_configurations"]]
        for mutate in (lambda r: r.pop(), lambda r: r.reverse(),
                       lambda r: r[0].update(decisions=0), lambda r: r[1].update(completed=False),
                       lambda r: r[1].update(decisions=20000),
                       lambda r: r[1].update(active_window_seconds=float("inf"))):
            changed = deepcopy(rows)
            mutate(changed)
            self.assertFalse(throughput_complete(changed, setting, phase_seconds=605.))
        self.assertFalse(throughput_complete(rows, setting, phase_seconds=661.))

    def test_request_is_reserved_and_uncertain_tail_retained_if_transport_fails(self):
        clock = [0.]
        budget = WorkBudget(self.declaration["budget"], io.StringIO(), clock=lambda: clock[0])
        class BrokenTransport:
            def request(self, operation, **fields):
                self.assert_reserved = bool(budget.pending)
                raise OSError("bounded synthetic transport failure")
        transport = BrokenTransport()
        with self.assertRaises(OSError):
            BudgetHost(transport, budget).request("step", ticks=6, commands=[])
        self.assertTrue(transport.assert_reserved)
        result = budget.snapshot()
        self.assertEqual(result["step_requests"], 1)
        self.assertEqual(result["requested_step_ticks"], 6)
        self.assertEqual(result["confirmed_step_ticks"], 0)
        self.assertTrue(result["unconfirmed_tail"])

    def receipt(self, start=0):
        from nca.adapters.protocol import make_physical_command
        from test_export_runtime import view
        source = view(start)
        source["last_sequence"] = -1
        action = WireAction(aim=(1., 0), effort=.5).to_dict()
        packet = make_physical_command(source, 0, action, 12)
        traces = []
        terminal = start + 6 == 240
        for offset in range(6):
            observation = view(start + offset + 1)
            last = terminal and offset == 5
            observation.update(suspended=last, control_generation=int(last))
            consumed = deepcopy(action)
            consumed["aim"][0] += 1e-8
            traces.append({"tick_before": start + offset, "tick_after": start + offset + 1,
                "world_tick": start + offset + 1, "cellular_sample_tick": start + offset, "failed": False,
                "observation": observation, "requested": [deepcopy(packet)] if offset == 0 else [],
                "admission": [{"accepted": True, "reason": ""}] if offset == 0 else [],
                "applied": {"terminal_revocation": last, "action_available": not last,
                            "action": None if last else consumed}})
        return source, action, traces[-1]["observation"], False, terminal, {
            "actual_ticks": 6, "trace": traces, "idle_trace": [], "consumed_trace_complete": True,
            "idle_trace_dropped": 0, "unconsumed_terminal_commands": []}

    def test_consumption_allows_only_exact_declared_horizon_revocation(self):
        require_consumed_decision(*self.receipt())
        require_consumed_decision(*self.receipt(234))
        for mutate in (lambda r: r[5]["trace"][2]["observation"].update(suspended=True),
                       lambda r: r[5]["trace"][2]["observation"].update(control_generation=1),
                       lambda r: r[5]["trace"][-1]["applied"].update(terminal_revocation=True),
                       lambda r: r[5]["trace"][0]["applied"]["action"].update(primary=True)):
            row = self.receipt()
            mutate(row)
            with self.assertRaises(ContractError):
                require_consumed_decision(*row)
        terminal = self.receipt(234)
        terminal[5]["trace"][-1]["applied"]["action_available"] = True
        with self.assertRaises(ContractError):
            require_consumed_decision(*terminal)

    def test_locally_accepted_but_refused_incomplete_or_replaced_work_cannot_count(self):
        for mutate in (lambda r: r[5]["trace"][0]["admission"][0].update(accepted=False),
                       lambda r: r[5]["trace"][0]["requested"][0].update(observed_tick=1),
                       lambda r: r[5]["trace"][0]["requested"][0]["action"].update(trim=.5),
                       lambda r: r[5]["trace"][1]["requested"].append({}),
                       lambda r: r[5]["trace"].pop(),
                       lambda r: r[5].update(consumed_trace_complete=False),
                       lambda r: r[5].update(idle_trace_dropped=1),
                       lambda r: r[5].update(unconsumed_terminal_commands=[{}]),
                       lambda r: r[5]["trace"][0].update(failed=True)):
            row = self.receipt()
            mutate(row)
            with self.assertRaises(ContractError):
                require_consumed_decision(*row)

    def test_complete_request_response_journal_preserves_partial_phase_evidence(self):
        exchange = io.BytesIO()
        budget = WorkBudget(self.declaration["budget"], io.StringIO(), clock=lambda: 0., exchanges=exchange)
        command = {"commands": [{"action": {"move": [1, 0]}, "observation_id": "original-source"}], "ticks": 6}
        serial = budget.reserve("step", command)
        response = {"actual_ticks": 6, "traces": [{"actual_motion": [1, 2]}]}
        budget.confirm(serial, response)
        lines = exchange.getvalue().decode().splitlines()
        import json
        self.assertEqual(json.loads(lines[0])["fields"], command)
        self.assertEqual(json.loads(lines[1])["response"], response)

    def test_refused_new_work_does_not_prevent_explicit_owner_cancellation(self):
        clock = [0.]
        budget = WorkBudget(self.declaration["budget"], io.StringIO(), clock=lambda: clock[0])
        serial = budget.reserve("step", {"ticks": 6})
        budget.confirm(serial, {"actual_ticks": 6, "observation": {"run_id": "fixture", "epoch": 1, "tick": 6}})
        self.assertEqual(budget.snapshot()["confirmed_step_ticks"], 6)
        clock[0] = 900.
        with self.assertRaisesRegex(ContractError, "followup_wall_budget"):
            budget.reserve("step", {"ticks": 6})
        serial = budget.reserve("cancel", {})
        budget.confirm(serial, {"observation": {"run_id": "fixture", "epoch": 1, "tick": 6}})
        self.assertFalse(budget.snapshot()["unconfirmed_tail"])
        self.assertEqual(budget.snapshot()["confirmed_observed_episode_ticks"], 6)

    def test_step_budget_cannot_be_exceeded_before_a_new_owner_operation(self):
        limits = dict(self.declaration["budget"], maximum_step_requests=1, maximum_requested_step_ticks=6)
        budget = WorkBudget(limits, io.StringIO(), clock=lambda: 0.)
        serial = budget.reserve("step", {"ticks": 6})
        budget.confirm(serial, {"actual_ticks": 6})
        with self.assertRaisesRegex(ContractError, "followup_step_budget"):
            budget.reserve("step", {"ticks": 1})
        self.assertEqual(budget.counts["step_requests"], 1)
        budget.counts["host_requests"] = limits["maximum_host_requests"]
        with self.assertRaisesRegex(ContractError, "followup_request_budget"):
            budget.reserve("close", {})

    def test_realtime_work_has_observed_lower_bound_and_reserved_horizon_upper_bound(self):
        budget = WorkBudget(self.declaration["budget"], io.StringIO(), clock=lambda: 0.)
        budget.realtime_reserved_ticks = self.declaration["budget"]["realtime_episode_maximum_ticks"]
        for mode, operation, fields, tick in (("stepped", "step", {"ticks": 6}, 6),
                                              ("realtime", "step", {"ticks": 6}, 12),
                                              ("realtime", "observe", {}, 17),
                                              ("realtime", "cancel", {}, 17)):
            serial = budget.reserve(operation, fields, mode=mode)
            budget.confirm(serial, {"observation": {"run_id": "fixture", "epoch": 1, "tick": tick}})
        # A close response does not reveal the final autonomous simulation tick.
        serial = budget.reserve("close", {}, mode="realtime")
        budget.confirm(serial, {"closed": True})
        result = budget.snapshot()
        self.assertEqual(result["observed_tick_lower_bounds_by_mode"], {"stepped": 6, "realtime": 17})
        self.assertEqual(result["requested_step_ticks_by_mode"], {"stepped": 6, "realtime": 6})
        self.assertEqual(result["confirmed_observed_episode_ticks"], 23)
        self.assertEqual(result["realtime_horizon_reserved_upper_bound_ticks"], 600)
        self.assertEqual(result["native_tick_upper_bound"], 606)
        self.assertFalse(result["unconfirmed_tail"])

    def test_both_future_refusal_captures_preserve_live_profile_and_reset_provenance(self):
        from nca.bench.torch_screen import _variant
        from nca.bench.export_postprocess import validate_frames
        from test_export_postprocess import fixture
        for model, cadence in (("jev", 12), ("astra", 6)):
            episode, _, exchanges = fixture(model, cadence)
            first = exchanges[1]["response"]["observation"]
            profile = exchanges[2]["response"]["profile"]
            final = episode["trace"][-1]["observation"]
            env = Mock(config=exchanges[1]["request"]["fields"]["config"])
            env.step_packet.return_value = (final, 0., False, True, {"trace": episode["trace"]})
            host = Mock(identity={})
            host.request.return_value = {"id": 17, "profile": profile}
            with patch("nca.bench.torch_screen._environment", return_value=(env, first, {}, 0)), \
                 patch("nca.bench.torch_screen.context_for", return_value=None), \
                 patch("nca.bench.torch_screen.TorchRuntimeBundle", side_effect=ContractError("context", "fixture refusal")), \
                 patch("nca.bench.torch_screen.dispatch") as cancel:
                result = _variant(host, Mock(value={}), {"trajectory": {"cases": [episode["case"]]}},
                                  model=model, decision_ticks=cadence)
            validate_frames(result, 24)
            self.assertEqual(result["profile"], profile)
            self.assertEqual(result["profile_provenance"]["response_id"], 17)
            self.assertEqual(result["profile_provenance"]["reset_observation_id"], first["observation_id"])
            self.assertEqual(result["initial"], first)
            self.assertEqual(result["config"], env.config)
            self.assertTrue(result["passed"])
            self.assertEqual(result["requested_actions"], 0)
            cancel.assert_called_once()

    def test_declaration_cannot_expand_work_or_lower_acceptance_after_results(self):
        for change in (lambda p: p["throughput"].update(minimum_active_window_seconds=79.),
                       lambda p: p["budget"].update(maximum_invocations=2),
                       lambda p: p["original"].update(runtime_qualified=True)):
            declaration = deepcopy(self.declaration)
            change(declaration)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "altered.json"
                path.write_text(canonical_json(declaration))
                with self.assertRaisesRegex(ContractError, "followup_predeclaration"):
                    load_declaration(path)


if __name__ == "__main__":
    unittest.main()
