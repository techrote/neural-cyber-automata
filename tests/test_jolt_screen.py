"""Independent fake-owner orchestration checks; no host or experiment runs."""

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


from nca.contracts import ContractError
from nca.evaluation.jolt_learning import POLICY_SEEDS, controller_ids, qualify, seed_interval
from nca.train.jolt_admission import AdmissionError, BudgetLedger

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from nca.contracts import WireAction, sha256_json
    from nca.controllers.jolt_learned import JoltPolicy
    from nca.train.jolt_learning import generalized_advantages
    from nca.train.jolt_screen import MeteredOwner, capture, freeze_candidates, runtime_identity


def rows():
    result = []
    for case in range(16):
        for controller in controller_ids():
            successes = 8 if controller == "authored" else 0 if controller == "idle" else (
                4 if controller.startswith("bc-") and controller.endswith("initial") else
                15 if controller.startswith("residual-") and controller.endswith("final") else 14)
            result.append({"split": "sealed", "case_id": f"case{case}", "controller_id": controller,
                "metrics": {"success": case < successes, "capped_completion_seconds": 2. if case < successes else 4.,
                    "ordinary_trim_reversals_per_second": .2, "mean_absolute_angle_radians": .1,
                    "peak_toward_speed_cells_per_second": 50., "purposeful_launch": controller != "idle",
                    "inference_p95_ms": 1., "integrity_failure": False,
                    "failed_admissions": 0, "administratively_capped": False},
                "supervisor_interventions": 0, "owner_closure_confirmed": True, "error": None})
    return result


class JoltQualificationTests(unittest.TestCase):
    def test_all_twenty_seven_seed_resamples_and_seed_mean_are_preserved(self):
        result = seed_interval([.25, .5, .75])
        self.assertEqual(result["resamples"], 27)
        self.assertEqual(result["mean"], .5)
        self.assertGreater(result["lower95"], 0.)
        with self.assertRaises(ContractError):
            seed_interval([.2, .4])

    def test_whole_bc_family_preferred_and_strong_base_does_not_waive_residual_gain(self):
        result = qualify(rows(), [f"case{i}" for i in range(16)], evidence_complete=True)
        self.assertTrue(result["qualified"])
        self.assertEqual(result["selected_family"], "bc")
        self.assertTrue(result["families"]["bc"]["qualified"])
        self.assertFalse(result["families"]["residual"]["gates"]["own_initial_gain"])
        self.assertFalse(result["morphology_transfer_admitted"])
        self.assertFalse(result["owner_feel_accepted"])

    def test_no_censoring_duplicate_pairs_or_unverified_evidence(self):
        original = rows()
        for selected in (original[:-1], original + [original[-1]]):
            result = qualify(selected, [f"case{i}" for i in range(16)], evidence_complete=True)
            self.assertFalse(result["complete_pairs"])
            self.assertFalse(result["qualified"])
        self.assertFalse(qualify(original, [f"case{i}" for i in range(16)])["qualified"])

    def test_integrity_failure_in_initial_or_reference_case_blocks_entire_result(self):
        original = rows()
        original[0]["metrics"]["failed_admissions"] = 1
        result = qualify(original, [f"case{i}" for i in range(16)], evidence_complete=True)
        self.assertFalse(result["global_integrity"])
        self.assertFalse(result["qualified"])

    def test_every_seed_and_every_case_latency_remain_hard_gates(self):
        original = rows()
        controller = f"bc-seed{POLICY_SEEDS[0]}-final"
        next(row for row in original if row["controller_id"] == controller)["metrics"]["inference_p95_ms"] = 5.01
        result = qualify(original, [f"case{i}" for i in range(16)], evidence_complete=True)
        self.assertFalse(result["families"]["bc"]["qualified"])
        self.assertFalse(result["qualified"])

    def test_invalid_metric_domains_and_nonboolean_evidence_refuse(self):
        for key, value in (("success", 2.), ("purposeful_launch", 2.),
                           ("success", 1), ("mean_absolute_angle_radians", -.1),
                           ("capped_completion_seconds", 5.), ("capped_completion_seconds", True),
                           ("peak_toward_speed_cells_per_second", float("inf")),
                           ("failed_admissions", False), ("failed_admissions", -1),
                           ("administratively_capped", 0), ("integrity_failure", "false"),
                           ("inference_p95_ms", -1.)):
            original = rows()
            original[0]["metrics"][key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                qualify(original, [f"case{i}" for i in range(16)], evidence_complete=True)
        with self.assertRaises(ContractError):
            qualify(rows(), [f"case{i}" for i in range(16)], evidence_complete="yes")
        for field, value in (("supervisor_interventions", -1), ("supervisor_interventions", False),
                             ("owner_closure_confirmed", "yes")):
            original = rows()
            original[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ContractError):
                qualify(original, [f"case{i}" for i in range(16)], evidence_complete=True)

    def test_versioned_extensions_discover_without_loading_optional_runtime(self):
        script = ("import sys; from nca.builtins import builtin_registry; r=builtin_registry(); "
                  "assert r.describe('policy', 'jolt-learned', 2).version == 2; "
                  "assert r.describe('trainer', 'learn-jolt', 2).version == 2; "
                  "assert 'torch' not in sys.modules and 'numpy' not in sys.modules; "
                  "assert 'nca.controllers.jolt_learned' not in sys.modules; "
                  "assert 'nca.train.jolt_screen' not in sys.modules")
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


class FakeSocket:
    def __init__(self):
        self.timeout = 15.

    def gettimeout(self):
        return self.timeout

    def settimeout(self, value):
        self.timeout = value


class Journal(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.fail = False

    def write(self, data):
        if self.fail:
            raise OSError("synthetic durable write failure")
        return super().write(data)

    def records(self):
        return [json.loads(line) for line in self.getvalue().decode().splitlines()]


class FakeHost:
    def __init__(self, ledger, journal):
        self._channel = SimpleNamespace(connection=FakeSocket())
        self.ledger, self.journal = ledger, journal
        self.calls = []
        self.fail = False
        self.actual = 6
        self.admissions = []
        self.close_confirmed = True

    def request(self, operation, **fields):
        # Evidence and cost reservation must precede even an uncertain send.
        assert self.journal.records()[-1]["event"] == "owner_request"
        assert self.ledger.counts["owner_requests"] > len(self.calls)
        self.calls.append((operation, fields))
        if self.fail and operation == "step":
            raise TimeoutError("synthetic uncertain native request")
        return ({"actual_ticks": self.actual, "traces": [{"admission": self.admissions}]}
                if operation == "step" else {"closed": operation == "close" and self.close_confirmed})


class CutoffFixtureOwner:
    """Explicitly synthetic receipts, optionally unstable on one final tick."""

    lane, mode = "act004", "stepped"
    identity = {"lane": "act004", "fixture": "synthetic-cutoff-only"}
    capabilities = {"schedule": "synthetic-cutoff-only"}
    profile = {"fixture": "synthetic-cutoff-only"}

    def __init__(self, *, unstable_tick=6):
        self.calls = []
        self.view = None
        self.unstable_tick = unstable_tick

    def observation(self, tick, action, *, generation=0, suspended=False):
        return {"version": 2, "representation": "articulated-rapier", "run_id": "synthetic-cutoff",
                "epoch": 1, "actor_id": 0, "observation_id": f"synthetic-cutoff:1:0:{generation}:{tick}",
                "profile_version": 1, "model": "jev", "rig_id": "jev", "tick": tick,
                "position": [0., 0.], "velocity": [0., 0.], "rotation": 3. if tick == self.unstable_tick else 0.,
                "angular_velocity": 0., "grounded": True, "cooldown": 0., "energy": 1.,
                "ground_distance": 0., "ceiling_distance": 96., "control_generation": generation,
                "parts": [{"id": "core", "index": 0, "detached": False}], "joints": [],
                "action": action, "suspended": suspended}

    def transition(self, traces):
        return {"schedule": self.capabilities["schedule"], "observation": self.view,
                "tick": self.view["tick"], "world_tick": self.view["tick"], "start_tick": self.view["tick"] - len(traces),
                "actual_ticks": len(traces), "traces": traces, "reason": "synthetic_fixture",
                "reward_kind": "no_task_reward", "terminated": False, "truncated": False,
                "finished": False}

    def request(self, operation, **fields):
        self.calls.append(operation)
        if operation == "profile":
            return {"profile": self.profile}
        if operation == "reset":
            self.view = self.observation(0, WireAction(effort=0.).to_dict())
            return self.transition([])
        if operation == "step":
            assert fields["ticks"] == 6
            packet = fields["commands"][0]
            start = self.view["tick"]
            traces = []
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
        if operation == "cancel":
            self.view = self.observation(self.view["tick"], WireAction(effort=0.).to_dict(), generation=1, suspended=True)
            return {"cancelled": True, "observation": self.view}
        raise AssertionError(f"unexpected synthetic operation {operation}")


def audit_saved_episode(output, row, journal, owner, case, *, policy=None, base=None, prior_decisions=0):
    """Check actual saved capture artifacts, without claiming a full allocation."""
    from collections import Counter
    from nca.data import read_episode
    from nca.evaluation.jolt_learning_audit import _Audit

    raw = read_episode(output / row["artifact"])
    costs = Counter()
    audit = _Audit.__new__(_Audit)
    audit.output = output
    audit.reserve = lambda name, amount=1: costs.update({name: amount})
    audit.manifest = {"source_sha256": "c" * 64, "source_revision": raw.learner_revision,
                      "source_files": dict(raw.source_files)}
    audit.config = {"profile_sha256": sha256_json(owner.profile),
                    "cases": {"teacher": [case], "open": [], "sealed": []}}
    audit.host_identity = owner.identity
    audit.models = {} if policy is None else {row["controller_id"]: (policy, base)}
    audit.forwards = audit.runtime_forwards = audit.bootstrap_forwards = audit.executions = 0
    audit.action_generators = {190103: torch.Generator().manual_seed(390103)}
    audit.ppo_groups = {(190103, 0): [(None, SimpleNamespace(decisions=tuple(range(prior_decisions))))]} if prior_decisions else {}
    audit.teacher_rows, audit.partial, audit.verified_rows = [], [], []
    bundle = audit._read_episode(row)
    bundle["proposals"] = [item for item in journal.records() if item["event"] == "decision_proposed"]
    audit.episode(bundle)
    return audit, costs


def fixture():
    now = [100.]
    counter = {"owner_requests": 20, "decisions": 2, "requested_ticks": 12, "confirmed_ticks": 12,
               "refused_ticks": 12, "uncertain_ticks": 12}
    counter.update({f"owner_{name}": 10 for name in ("reset", "configure", "profile", "cancel", "close", "step")})
    budget = {"maximum_wall_seconds": 100, "cleanup_reserve_seconds": 20, "counters": counter,
              "stages": {"teacher": {"decisions": 2, "requested_ticks": 12}}}
    ledger = BudgetLedger(budget, started=100., monotonic=lambda: now[0])
    journal = Journal()
    host = FakeHost(ledger, journal)
    return now, ledger, journal, host, MeteredOwner(host, ledger, journal)


@unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning extra is not installed")
class JoltOwnerTests(unittest.TestCase):
    def test_actual_runtime_identity_can_be_published_and_read_as_strict_json(self):
        from nca.contracts import strict_json_loads
        from nca.evaluation.durable_journal import publish_json
        identity = runtime_identity()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.json"
            publish_json(path, {"runtime": identity})
            restored = strict_json_loads(path.read_bytes())["runtime"]
        self.assertEqual(restored, identity)
        self.assertEqual(sha256_json(restored), sha256_json(identity))
        self.assertIs(type(restored["torch"]), str)
        self.assertEqual(restored["torch"], str(torch.__version__))

    def test_real_teacher_capture_artifacts_replay_in_independent_auditor(self):
        owner, journal = CutoffFixtureOwner(unstable_tick=None), Journal()
        ledger = SimpleNamespace(check=lambda: None, reserve=lambda *_args, **_kwargs: None)
        case = {"id": "synthetic-teacher-case", "host_seed": 7, "spawn": [0., 0.], "target": [0., 0.]}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            episode, row, sequence = capture(owner, ledger, journal, case,
                identifier="teacher-synthetic-teacher-case", controller_id="authored", kind="authored", stage="teacher",
                output=output, root=Path(__file__).resolve().parents[1],
                config={"profile_sha256": sha256_json(owner.profile)}, source_sha256="c" * 64)
            audit, costs = audit_saved_episode(output, row, journal, owner, case)
            self.assertEqual(owner.calls, ["reset", "profile", "step", "step", "cancel"])
            self.assertEqual(episode.status, "completed")
            self.assertTrue(row["metrics"]["success"])
            self.assertIsNone(sequence)
            self.assertIsNone(row["bootstrap_value"])
            self.assertEqual(costs["audit_forward"], 0)
            self.assertEqual(audit.verified_rows[0]["consumed_ticks"], 12)
            self.assertEqual(len(audit.teacher_rows), 1)

    def test_real_ppo_cutoff_artifacts_replay_without_invented_bootstrap(self):
        torch.manual_seed(917)
        torch.set_num_threads(1)
        owner, journal = CutoffFixtureOwner(), Journal()
        ledger = SimpleNamespace(check=lambda: None, reserve=lambda *_args, **_kwargs: None)
        policy, base = JoltPolicy("residual"), JoltPolicy("bc")
        case = {"id": "synthetic-ppo-case", "host_seed": 7, "spawn": [0., 0.], "target": [48., 0.]}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            episode, row, sequence = capture(owner, ledger, journal, case,
                identifier="ppo-seed190103-u0-e1", controller_id="residual-seed190103-initial", kind="residual", stage="ppo",
                output=output, root=Path(__file__).resolve().parents[1],
                config={"profile_sha256": sha256_json(owner.profile)}, source_sha256="c" * 64,
                policy=policy, base=base, generator=torch.Generator().manual_seed(390103), maximum_decisions=1)
            # This is the one remaining decision of a synthetic group, not a
            # claim that the other 255 decisions were collected by this test.
            audit, costs = audit_saved_episode(output, row, journal, owner, case,
                                               policy=policy, base=base, prior_decisions=255)
            self.assertEqual(episode.status, "capped")
            self.assertFalse(sequence.transitions[-1].final_observation_valid)
            self.assertEqual(costs["audit_forward"], 2)
            self.assertEqual(audit.bootstrap_forwards, 0)
            self.assertEqual(audit.verified_rows[0]["consumed_ticks"], 6)
            self.assertIsNotNone(audit.verified_rows[0]["gae_sha256"])

    def test_cutoff_with_supervisor_override_cannot_invent_a_zero_bootstrap_value(self):
        torch.manual_seed(917)
        torch.set_num_threads(1)
        reservations = []
        ledger = SimpleNamespace(check=lambda: None,
            reserve=lambda name, amount=1, **kwargs: reservations.append((name, amount, kwargs)))
        owner = CutoffFixtureOwner()
        with tempfile.TemporaryDirectory() as temporary:
            episode, row, sequence = capture(owner, ledger, Journal(),
                {"id": "synthetic-case", "host_seed": 7, "spawn": [0., 0.], "target": [48., 0.]},
                identifier="synthetic-cutoff", controller_id="synthetic-residual", kind="residual", stage="ppo",
                output=Path(temporary), root=Path(__file__).resolve().parents[1],
                config={"profile_sha256": sha256_json(owner.profile)}, source_sha256="c" * 64,
                policy=JoltPolicy("residual"), base=JoltPolicy("bc"),
                generator=torch.Generator().manual_seed(917), maximum_decisions=1)
            self.assertEqual(owner.calls, ["reset", "profile", "step", "cancel"])
            self.assertEqual(row["bootstrap_reason"], "supervisor_value_unavailable")
            self.assertFalse(row["bootstrap_observation_valid"])
            self.assertEqual(row["bootstrap_value"], 0.)
            self.assertEqual(row["bootstrap_observation_sha256"],
                             sha256_json(episode.decisions[-1].event.next_observation))
            self.assertEqual(sequence.bootstrap_value, 0.)
            self.assertTrue(sequence.transitions[-1].administrative_cutoff)
            self.assertTrue(sequence.transitions[-1].valid)
            self.assertFalse(sequence.transitions[-1].final_observation_valid)
            advantages, _ = generalized_advantages(sequence)
            self.assertAlmostEqual(advantages[0], sequence.transitions[0].reward - sequence.transitions[0].value)
            self.assertFalse(any(name == "bootstrap_forward" for name, _, _ in reservations))
            retained = json.loads((Path(temporary) / "metrics/synthetic-cutoff.json").read_text())
            self.assertEqual(retained["bootstrap_reason"], row["bootstrap_reason"])
            self.assertFalse(retained["bootstrap_observation_valid"])

    def test_native_reservation_journal_and_confirmed_consumption_order(self):
        _, ledger, journal, host, owner = fixture()
        original = host._channel.connection
        result = owner.request("step", ticks=6)
        self.assertEqual(result["actual_ticks"], 6)
        self.assertEqual(ledger.counts["decisions"], 1)
        self.assertEqual(ledger.counts["requested_ticks"], 6)
        self.assertEqual(ledger.counts["confirmed_ticks"], 6)
        self.assertEqual([row["event"] for row in journal.records()], ["owner_request", "owner_response"])
        self.assertIsNone(owner.pending)
        self.assertIs(host._channel.connection, original)

    def test_unknown_operation_bad_cadence_and_exhausted_stage_never_send(self):
        _, ledger, _, host, owner = fixture()
        for operation, arguments in (("arbitrary", {}), ("step", {"ticks": 12})):
            with self.assertRaises(ContractError):
                owner.request(operation, **arguments)
        self.assertEqual(host.calls, [])
        self.assertEqual(ledger.counts["decisions"], 0)
        owner.request("step", ticks=6)
        owner.request("step", ticks=6)
        with self.assertRaises(AdmissionError):
            owner.request("step", ticks=6)
        self.assertEqual(len(host.calls), 2)
        self.assertEqual(ledger.counts["requested_ticks"], 12)

    def test_uncertain_request_consumes_allocation_and_survives_cleanup(self):
        _, ledger, _, host, owner = fixture()
        host.fail = True
        with self.assertRaises(TimeoutError):
            owner.request("step", ticks=6)
        self.assertEqual(owner.pending, 1)
        with self.assertRaises(ContractError):
            owner.request("reset")
        owner.request("close")
        self.assertTrue(owner.closed_native)
        self.assertEqual(owner.uncertain, [1])
        self.assertEqual(ledger.counts["decisions"], 1)
        self.assertEqual(ledger.counts["requested_ticks"], 6)
        self.assertEqual(ledger.counts["confirmed_ticks"], 0)
        self.assertEqual(ledger.counts["uncertain_ticks"], 6)
        self.assertEqual(owner.uncertain_ticks, 6)

    def test_native_refusal_retains_requested_and_confirmed_costs_separately(self):
        _, ledger, _, host, owner = fixture()
        host.admissions = [{"accepted": False, "reason": "synthetic refusal"}]
        owner.request("step", ticks=6)
        self.assertEqual(ledger.counts["requested_ticks"], 6)
        self.assertEqual(ledger.counts["confirmed_ticks"], 6)
        self.assertEqual(ledger.counts["refused_ticks"], 6)
        self.assertEqual(ledger.counts["uncertain_ticks"], 0)
        self.assertEqual(owner.refused_ticks, 6)

    def test_durable_reservation_failure_prevents_transport_without_refund(self):
        _, ledger, journal, host, owner = fixture()
        journal.fail = True
        with self.assertRaises(OSError):
            owner.request("step", ticks=6)
        self.assertEqual(host.calls, [])
        self.assertEqual(ledger.counts["decisions"], 1)
        self.assertEqual(owner.pending, 1)

    def test_malformed_native_consumption_cannot_confirm_or_release_pending(self):
        _, ledger, _, host, owner = fixture()
        host.actual = 7
        with self.assertRaises(ContractError):
            owner.request("step", ticks=6)
        self.assertEqual(ledger.counts["confirmed_ticks"], 0)
        self.assertEqual(owner.pending, 1)
        self.assertEqual(ledger.counts["uncertain_ticks"], 6)

    def test_successful_transport_without_native_close_receipt_is_not_closure(self):
        _, ledger, _, host, owner = fixture()
        host.close_confirmed = False
        with self.assertRaises(ContractError):
            owner.request("close")
        self.assertFalse(owner.closed_native)
        self.assertEqual(owner.pending, 1)
        self.assertEqual(owner.uncertain, [1])
        self.assertEqual(ledger.counts["owner_close"], 1)
        self.assertEqual(ledger.counts["uncertain_ticks"], 0)

    def test_cleanup_reserve_blocks_new_work_but_keeps_original_close_deadline(self):
        clock, _, _, host, owner = fixture()
        clock[0] = 165.
        with self.assertRaises(AdmissionError):
            owner.request("step", ticks=6)
        owner.request("close")
        self.assertEqual([operation for operation, _ in host.calls], ["close"])

    def test_all_six_final_artifacts_must_exist_and_match_before_opening(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoints = []
            for family in ("bc", "residual"):
                for seed in POLICY_SEEDS:
                    name = f"{family}-seed{seed}-final"
                    path = root / f"{name}.pt"
                    path.write_bytes(name.encode())
                    checkpoints.append({"id": name, "path": path.name,
                                        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            with self.assertRaises(ContractError):
                freeze_candidates(checkpoints[:-1], destination=root, source_sha256="a" * 64, journal=Journal())
            with self.assertRaises(ContractError):
                freeze_candidates(checkpoints + [checkpoints[0]], destination=root, source_sha256="a" * 64, journal=Journal())
            corrupted = deepcopy(checkpoints)
            corrupted[0]["file_sha256"] = "b" * 64
            with self.assertRaises(ContractError):
                freeze_candidates(corrupted, destination=root, source_sha256="a" * 64, journal=Journal())
            journal = Journal()
            result = freeze_candidates(checkpoints, destination=root, source_sha256="a" * 64, journal=journal)
            self.assertEqual(len(result["finals"]), 6)
            self.assertTrue((root / "frozen-finals.json").exists())
            self.assertEqual(journal.records()[-1]["event"], "six_finals_frozen")


if __name__ == "__main__":
    unittest.main()
