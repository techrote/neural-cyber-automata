"""Independent offline negative controls; no host, physical run or fitting."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from nca.contracts import ContractError, WireAction, canonical_json, sha256_json
    from nca.controllers.baseline import SkillGoal
    from nca.controllers.jolt_learned import ActionHistory, InvocationContext, JoltController, JoltPolicy, policy_state_sha256
    from nca.controllers.recovery_supervisor import PhysicalRecoverySupervisor
    from nca.data import ActionSource, EpisodeRecord
    from nca.data.records import json_copy
    from nca.data.jolt_samples import JoltEpisode, draw_from_decision
    from nca.evaluation.durable_journal import DurableJournal, publish_json
    from nca.evaluation.jolt_character import character_metrics
    from nca.evaluation.jolt_learning_audit import _Audit, _Reservations, _advantages, _native_step, _path, _window_counts, verify_output
    from nca.train.jolt_admission import BudgetLedger, REQUIRED_COUNTERS
    from nca.train.jolt_learning import generalized_advantages, rollout_sequence
    from nca.train.task import HopTask, REWARD_SCHEMA
    from test_jolt_learning_v2 import record as decision_record, view


def setUpModule():
    if TORCH_AVAILABLE:
        torch.set_num_threads(1)


def budget_fixture():
    budget = {"maximum_wall_seconds": 100, "cleanup_reserve_seconds": 20,
              "counters": {"attempts": 1, "owner_requests": 2, "owner_step": 2,
                           "decisions": 2, "requested_ticks": 12, "confirmed_ticks": 12},
              "stages": {"ppo": {"decisions": 2, "requested_ticks": 12}}}
    now = [10.]
    ledger = BudgetLedger(budget, started=10., monotonic=lambda: now[0])
    ledger.reserve("attempts")
    ledger.reserve_many({"owner_requests": 1, "owner_step": 1, "decisions": 1, "requested_ticks": 6}, stage="ppo")
    now[0] += 1
    ledger.reserve("confirmed_ticks", 6, stage="ppo")
    return budget, ledger.reservations, ledger.snapshot()


def synthetic_episode(*, cutoff=False, length=3):
    """Original-output fixture independent from the offline replay path."""
    torch.manual_seed(981)
    policy, base = JoltPolicy("residual"), JoltPolicy("bc")
    seed = 190103
    identifier = "ppo-seed190103-u0-e1" if cutoff else "ppo-seed190103-u0-e0"
    source_id = "residual-seed190103-initial"
    goal = SkillGoal(identifier, "directed_hop", (48., 0.), .8, 4., 4.)
    runtime = JoltController(policy, base_policy=base, generator=torch.Generator().manual_seed(seed + 200000))
    task = HopTask(goal, view(), deadline_ticks=240, quiet_ticks_required=12)
    decisions, raw_steps, frames, proposals, values, rewards = [], [], [], [], [], []
    for index in range(length):
        decision = runtime.step(view(index * 6), goal, deterministic=False)
        action = decision.action
        row = decision_record(index, decision.proposal, context=decision.context,
            history=decision.history.to_dict(), policy_sha=decision.policy_sha256, source_kind="learned",
            draw=draw_from_decision(decision, source_sha256="c" * 64))
        source = ActionSource(source_id, "learned", source_id, decision.policy_sha256)
        event = replace(row.event, source=source, ticks=tuple(replace(tick, source_id=source_id) for tick in row.event.ticks), reward=0.)
        raw_steps.append(event)
        terms = task.consume(action, [{**tick.to_dict(), "failed": not tick.valid} for tick in event.ticks], inference_ns=100)
        event = replace(event, reward=terms["total"], truncated=cutoff and index + 1 == length)
        row = replace(row, event=event, supervisor_sha256=sha256_json(PhysicalRecoverySupervisor().specification()),
                      supervision_reason=decision.supervision.reason)
        decisions.append(row)
        rewards.append(terms)
        values.append(decision.value)
        frames.append({"tick": index * 6, "action": action.to_dict(), "phase": "learned",
                       "reason": decision.supervision.reason, "status": "running", "inference_ns": 100,
                       "diagnostic_state": {"supervisor_intervention": False, "supervisor_reason": decision.supervision.reason}})
        proposals.append({"episode_id": identifier, "index": index,
            "observation_sha256": sha256_json(view(index * 6)), "invocation": decision.context.to_dict(),
            "history": decision.history.to_dict(), "proposal": decision.proposal.to_dict(),
            "supervision": asdict(decision.supervision), "dispatched_packet": event.requested_packet,
            "draw": row.draw.to_dict()})
        runtime.record_dispatch(decision, action, admitted=True)
    peek = runtime.clone_state().step(view(length * 6), goal, deterministic=True) if cutoff else None
    sample = JoltEpisode(identifier, "training", tuple(decisions), "capped", "administrative_cutoff")
    metrics = task.metrics(administratively_capped=True)
    character = character_metrics(view(), goal.target, raw_steps, frames)
    error = None if cutoff else {"type": "AdmissionError", "code": "synthetic_budget_end"}
    raw = EpisodeRecord(identifier, {"lane": "act004"}, "test-source", sha256_json({}), {}, {}, view(),
        tuple(raw_steps), "capped", "administrative_cutoff", "synthetic-case", "training",
        evaluation={"metrics": metrics, "character_metrics": character, "reward_terms": rewards,
                    "controller_frames": frames, "task_reward_schema": REWARD_SCHEMA,
                    "owner_closure_confirmed": False, "error": error})
    result = {"id": identifier, "controller_id": source_id, "kind": "residual", "stage": "ppo",
        "policy_sha256": decision.policy_sha256, "policy_values": values,
        "bootstrap_value": 0. if peek is None else peek.value,
        "bootstrap_observation_sha256": None if peek is None else sha256_json(view(length * 6)),
        "bootstrap_observation_valid": True,
        "bootstrap_reason": "task_terminal" if peek is None else "received_cutoff_observation",
        "error": error, "metrics": metrics, "character_metrics": character, "supervisor_interventions": 0,
        "owner_closure_confirmed": False, "artifact_sha256": "d" * 64}
    costs = Counter()
    audit = _Audit.__new__(_Audit)
    audit.reserve = lambda name, amount=1: costs.update({name: amount})
    audit.models = {source_id: (policy, base)}
    audit.manifest = {"source_sha256": "c" * 64}
    audit.forwards = audit.runtime_forwards = audit.bootstrap_forwards = audit.executions = 0
    audit.action_generators = {seed: torch.Generator().manual_seed(seed + 200000)}
    audit.ppo_groups = {(seed, 0): [(None, SimpleNamespace(decisions=tuple(range(256 - length))))]} if cutoff else {}
    audit.teacher_rows, audit.partial, audit.verified_rows = [], [], []
    bundle = {"row": result, "raw": raw, "sample": sample, "goal": goal,
              "context": InvocationContext.create(view(), goal), "proposals": proposals}
    return audit, bundle, costs


def journal_fixture(root, *, close_response=True, sealed_before_freeze=False):
    budget = {"maximum_wall_seconds": 100., "cleanup_reserve_seconds": 20.,
              "counters": {name: 100 for name in REQUIRED_COUNTERS},
              "stages": {stage: {"episodes_started": 5, "decisions": 5, "requested_ticks": 30}
                         for stage in ("teacher", "sealed")}}
    ledger = BudgetLedger(budget, started=0., monotonic=lambda: 1.)
    host = {"lane": "act004", "mode": "stepped", "schedule": "synthetic-schedule"}
    journal = DurableJournal(root / "owner-journal.jsonl")
    def note(event, **fields):
        journal.write((canonical_json({"event": event, **fields}) + "\n").encode())
    ledger.reserve("attempts")
    ledger.reserve_many({"host_starts": 1, "host_start_attempts": 1})
    note("host_start_reserved", counts=ledger.snapshot())
    note("host_started", identity=host)
    if sealed_before_freeze:
        ledger.reserve("episodes_started", stage="sealed")
        note("episode_started", id="sealed-reference", case_id="sealed-case", controller_id="authored", stage="sealed")
    ledger.reserve_many({"owner_requests": 1, "owner_close": 1}, stage="teacher", cleanup=True)
    note("owner_request", serial=1, stage="teacher", operation="close", fields={}, elapsed_seconds=1.)
    if close_response is not None:
        note("owner_response", serial=1, response={"id": 1, "ok": True, "closed": close_response})
    ledger.reserve("host_terminations", cleanup=True)
    note("host_terminated", native_close_confirmed=close_response is True, process_exited=True)
    note("worker_closed", status="failed", counters=ledger.snapshot())
    journal.close()
    closure = {"native_close_confirmed": close_response is True,
               "pending_owner_request": None if close_response is True else 1,
               "uncertain_owner_requests": [] if close_response is True else [1],
               "owner_requests": {name: int(name == "close") for name in ("reset", "profile", "configure", "cancel", "close", "step")},
               "requested_ticks": 0, "confirmed_ticks": 0, "refused_ticks": 0, "uncertain_ticks": 0,
               "durability": journal.durability_receipt()}
    worker = {"status": "failed", "results": [], "checkpoints": [], "learning": [], "closure": closure}
    for name, payload in (("worker-result.json", worker), ("manifest.json", {"host_identity": host}),
                          ("config.json", {"budget": budget}), ("declaration.json", {})):
        publish_json(root / name, payload)
    audit = _Audit(root, reserve=lambda *_: None)
    audit.budget = _Reservations(budget, ledger.reservations, ledger.snapshot())
    return audit


@unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning extra is not installed")
class JoltAuditTests(unittest.TestCase):
    def test_budget_reconstructs_all_counts_and_original_owner_reservation(self):
        budget, rows, snapshot = budget_fixture()
        audit = _Reservations(budget, rows, snapshot)
        request = {"operation": "step", "stage": "ppo", "elapsed_seconds": .1}
        audit.owner(request)
        self.assertEqual(audit.counts, snapshot["counts"])
        with self.assertRaisesRegex(ContractError, "no original atomic"):
            audit.owner(request)

    def test_incremental_reservation_digest_matches_every_canonical_prefix(self):
        budget, _, _ = budget_fixture()
        ledger = BudgetLedger(budget, started=0., monotonic=lambda: 1.25)
        self.assertEqual(ledger.snapshot()["reservations_sha256"], sha256_json([]))
        for increments, stage in (({"attempts": 1}, None),
                ({"owner_requests": 1, "owner_step": 1, "decisions": 1, "requested_ticks": 6}, "ppo"),
                ({"confirmed_ticks": 6}, "ppo")):
            ledger.reserve_many(increments, stage=stage)
            self.assertEqual(ledger.snapshot()["reservations_sha256"], sha256_json(ledger.reservations))
        exported = ledger.reservations
        exported[0]["increments"]["attempts"] = 100
        exported.append({"forged": True})
        self.assertEqual(len(ledger.reservations), 3)
        self.assertEqual(ledger.snapshot()["reservations_sha256"], sha256_json(ledger.reservations))

    def test_budget_rejects_refunds_reordering_stage_omission_and_changed_snapshot(self):
        budget, rows, snapshot = budget_fixture()
        mutations = []
        changed = deepcopy(rows)
        changed[1]["increments"]["requested_ticks"] = -6
        mutations.append((changed, snapshot))
        changed = deepcopy(rows)
        changed[1]["stage"] = None
        mutations.append((changed, snapshot))
        changed = deepcopy(rows)
        changed[1]["elapsed_seconds"] = 81.
        mutations.append((changed, snapshot))
        changed = deepcopy(snapshot)
        changed["counts"]["confirmed_ticks"] = 0
        mutations.append((rows, changed))
        for changed_rows, changed_snapshot in mutations:
            with self.subTest(rows=changed_rows), self.assertRaises(ContractError):
                _Reservations(budget, changed_rows, changed_snapshot)

    def test_owner_reservation_must_precede_native_dispatch(self):
        budget, rows, snapshot = budget_fixture()
        rows = deepcopy(rows)
        rows[1]["elapsed_seconds"] = .5
        snapshot = {**snapshot, "reservations_sha256": sha256_json(rows)}
        audit = _Reservations(budget, rows, snapshot)
        with self.assertRaisesRegex(ContractError, "preceded"):
            audit.owner({"operation": "step", "stage": "ppo", "elapsed_seconds": .4})

    def test_original_stochastic_values_actions_density_and_history_replay(self):
        audit, bundle, costs = synthetic_episode()
        audit.episode(bundle)
        self.assertEqual(len(audit.verified_rows), 1)
        self.assertEqual(costs["audit_forward"], 6)
        self.assertEqual(audit.executions, 3)

    def test_cutoff_bootstrap_replays_received_view_and_both_model_states(self):
        audit, bundle, costs = synthetic_episode(cutoff=True)
        audit.episode(bundle)
        self.assertEqual(costs["audit_forward"], 8)
        self.assertEqual(audit.bootstrap_forwards, 2)
        self.assertIsNotNone(audit.verified_rows[0]["gae_sha256"])

    def test_changed_value_density_latent_source_and_bootstrap_are_rejected(self):
        for mutation in ("value", "density", "latent", "source", "bootstrap", "bootstrap_view", "history"):
            audit, bundle, _ = synthetic_episode(cutoff=True)
            row, record = bundle["row"], bundle["sample"].decisions[0]
            if mutation == "value":
                row["policy_values"][0] += 1.
            elif mutation == "bootstrap":
                row["bootstrap_value"] += 1.
            elif mutation == "bootstrap_view":
                row["bootstrap_observation_sha256"] = "e" * 64
            elif mutation == "history":
                history = json_copy(record.history)
                history["intervention"] = True
                draw = replace(record.draw, history_sha256=sha256_json(history))
                record = replace(record, history=history, draw=draw)
            else:
                draw = record.draw
                if mutation == "density":
                    draw = replace(draw, log_probability=draw.log_probability + 1.)
                elif mutation == "source":
                    draw = replace(draw, source_sha256="e" * 64)
                else:
                    draw = replace(draw, latent=(draw.latent[0] + .1, *draw.latent[1:]))
                record = replace(record, draw=draw)
            if mutation in ("density", "latent", "source", "history"):
                bundle["sample"] = replace(bundle["sample"], decisions=(record, *bundle["sample"].decisions[1:]))
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                audit.episode(bundle)

    def test_readback_budget_refusal_precedes_any_forward(self):
        audit, bundle, _ = synthetic_episode()
        def denied(*_):
            raise RuntimeError("synthetic finite readback ceiling")
        audit.reserve = denied
        with self.assertRaisesRegex(RuntimeError, "finite readback"):
            audit.episode(bundle)
        self.assertEqual(audit.forwards, 0)

    def test_native_receipt_binds_actual_input_motion_and_original_clock(self):
        _, bundle, _ = synthetic_episode()
        event = bundle["raw"].events[0]
        traces = [{**tick.to_dict(), "failed": False} for tick in event.ticks]
        # A raw TickRecord stores the actual native post-step action view.
        event = replace(event, ticks=tuple(replace(tick, consumed_action=tick.observation.get("action", {}),
                                                 cellular_sample_tick=tick.tick_before) for tick in event.ticks))
        for trace in traces:
            trace["cellular_sample_tick"] = trace["tick_before"]
        request = {"fields": {"commands": [event.requested_packet], "ticks": 6}}
        response = {"schedule": "synthetic-schedule", "actual_ticks": 6, "start_tick": 0, "tick": 6,
            "failed": False, "observation": event.next_observation, "terminated": False, "truncated": False,
            "reason": event.reason, "traces": traces}
        _native_step(event, request, response, schedule="synthetic-schedule")
        for field in ("applied", "actual_motion", "observation", "requested"):
            changed = json_copy(response)
            changed["traces"][0][field] = {} if field != "requested" else []
            with self.subTest(field=field), self.assertRaises(ContractError):
                _native_step(event, request, changed, schedule="synthetic-schedule")
        changed = json_copy(request)
        changed["fields"]["ticks"] = 12
        with self.assertRaises(ContractError):
            _native_step(event, changed, response, schedule="synthetic-schedule")

    def test_gae_and_fixed_window_costs_use_complete_physical_sequence(self):
        _, bundle, _ = synthetic_episode(cutoff=True)
        records = bundle["sample"].decisions
        result = bundle["row"]
        independent = _advantages(records, result["policy_values"], result["bootstrap_value"], True)
        learned = rollout_sequence(bundle["sample"], result["policy_values"], bootstrap_value=result["bootstrap_value"])
        advantage, returns = generalized_advantages(learned)
        torch.testing.assert_close(torch.tensor(independent["advantages"]), advantage)
        torch.testing.assert_close(torch.tensor(independent["returns"]), returns)
        counters = _window_counts(records, "ppo")
        self.assertEqual(counters["ppo_optimizer"], 4)
        self.assertEqual(counters["ppo_forward"], 12)
        self.assertEqual(counters["ppo_replay_forward"], 3)
        self.assertEqual(counters["ppo_base_forward"], 12)

    def test_paths_and_missing_artifacts_cannot_become_verified_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ContractError):
                _path(root, "../outside.json")
            result = verify_output(root, reserve=lambda *_: None)
            self.assertFalse(result["verified"])
            self.assertFalse(result["qualified"])
            self.assertEqual(result["audit_forward"], 0)

    def test_closed_partial_journal_is_checked_without_becoming_qualified(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = journal_fixture(Path(directory))
            result = audit.journal()
            self.assertGreater(result["records"], 0)
            self.assertFalse(audit.complete)
            self.assertTrue(audit.closed)
            self.assertEqual(audit.partial[-1]["status"], "failed")

    def test_final_uncertain_or_refused_close_preserves_pending_identity(self):
        for response in (False, None):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory:
                audit = journal_fixture(Path(directory), close_response=response)
                audit.journal()
                self.assertFalse(audit.closed)
                self.assertEqual(audit.partial[-1]["pending_owner_request"], 1)
                self.assertEqual(audit.partial[-1]["uncertain_owner_requests"], [1])

    def test_any_sealed_reference_is_blocked_before_six_final_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = journal_fixture(Path(directory), sealed_before_freeze=True)
            with self.assertRaisesRegex(ContractError, "before all six"):
                audit.journal()

    def test_changed_owner_journal_cannot_pass_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = journal_fixture(root)
            path = root / "owner-journal.jsonl"
            path.write_bytes(path.read_bytes().replace(b'"closed":true', b'"closed":false'))
            with self.assertRaisesRegex(ContractError, "chain"):
                audit.journal()

    def test_non_ppo_capture_retains_absent_values_and_bootstrap(self):
        from nca.data import read_episode
        from nca.train.jolt_screen import capture
        from test_jolt_screen import CutoffFixtureOwner, Journal
        class StableOwner(CutoffFixtureOwner):
            def observation(self, tick, action, **kwargs):
                observed = super().observation(tick, action, **kwargs)
                observed["rotation"] = 0.
                return observed
            def transition(self, traces):
                response = super().transition(traces)
                response["start_tick"] = self.view["tick"] - len(traces)
                return response
            def request(self, operation, **fields):
                if operation == "cancel":
                    self.calls.append(operation)
                    self.view = self.observation(self.view["tick"], WireAction(effort=0.).to_dict(), generation=1, suspended=True)
                    return {"cancelled": True, "observation": self.view}
                if operation != "step":
                    return super().request(operation, **fields)
                self.calls.append(operation)
                packet, start, traces = fields["commands"][0], self.view["tick"], []
                for tick in range(start, start + 6):
                    self.view = self.observation(tick + 1, packet["action"])
                    traces.append({"tick_before": tick, "tick_after": tick + 1, "world_tick": tick + 1,
                        "cellular_sample_tick": tick, "received_monotonic_usec": tick * 10,
                        "completed_monotonic_usec": tick * 10 + 1, "requested": [packet] if tick == start else [],
                        "admission": [{"accepted": True, "reason": ""}] if tick == start else [],
                        "observation": self.view, "applied": {"action_available": True, "action": packet["action"],
                            "linear_impulse": [0., 0.], "angular_impulse": 0., "energy_used": 0.},
                        "passive": {"available": False, "reason": "synthetic_fixture"}})
                return self.transition(traces)
        owner, journal = StableOwner(), Journal()
        ledger = SimpleNamespace(check=lambda: None, reserve=lambda *_args, **_kwargs: None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample, row, _ = capture(owner, ledger, journal,
                {"id": "synthetic-case", "host_seed": 7, "spawn": [0., 0.], "target": [0., 0.]},
                identifier="open-synthetic-idle", controller_id="idle", kind="idle", stage="open",
                output=root, root=Path(__file__).resolve().parents[1],
                config={"profile_sha256": sha256_json(owner.profile)}, source_sha256="c" * 64)
            raw = read_episode(root / row["artifact"])
            self.assertIsNone(row["policy_values"])
            self.assertIsNone(row["bootstrap_value"])
            goal = SkillGoal(row["id"], "directed_hop", (0., 0.), .8, 4., 4.)
            audit, _, costs = synthetic_episode()
            audit.episode({"row": row, "raw": raw, "sample": sample, "goal": goal,
                "context": InvocationContext.create(raw.initial_observation, goal),
                "proposals": [item for item in journal.records() if item["event"] == "decision_proposed"]})
            self.assertEqual(costs["audit_forward"], 0)
            self.assertTrue(row["metrics"]["success"])


if __name__ == "__main__":
    unittest.main()
