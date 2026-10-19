"""Causal and lifecycle controls for the fixed graph; no synthetic physics claim."""

import copy
import io
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from nca.cellular.core import (FEATURE_NAMES, GraphIdentity, JOLT_PROFILE_SHA256,
                               PART_IDS, checked_parts, make_frame, mapped_request)
from nca.cellular.evaluation import ArrivalTask, bootstrap_three_seed, utility_gate
from nca.cellular.messages import FaultSpec, Message, MessageBus
from nca.contracts import ContractError, WireAction, sha256_json
from nca.controllers.baseline import SkillGoal
from nca.data import (ActionSource, CaptureEvent, Dataset, EpisodeRecord, PolicyContext,
                      TickRecord, policy_view)
from nca.randomness import ScenarioSplit

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "configs/cellular/jolt-three-part-hop-v1.json"


def view(tick=0, **changes):
    result = {"version": 2, "representation": "articulated-rapier", "run_id": "synthetic-graph",
              "epoch": 1, "actor_id": 0, "tick": tick, "observation_id": f"synthetic-graph:1:0:0:{tick}",
              "model": "jev", "profile_version": 1, "control_generation": 0,
              "position": [176., 346.5], "velocity": [0., 0.], "rotation": 0., "angular_velocity": 0.,
              "grounded": True, "energy": 1., "cooldown": 0., "ground_distance": 0., "ceiling_distance": 45.,
              "parts": [{"id": name, "index": index, "detached": False,
                         "position": [[176., 346.5], [171., 346.5], [181., 346.5]][index],
                         "velocity": [0., 0.], "rotation": 0., "angular_velocity": 0.,
                         "mass": 1.3 if index == 0 else .26, "size": [7., 7.] if index == 0 else [2., 6.]}
                        for index, name in enumerate(PART_IDS)],
              "joints": [{"id": name, "a": 0, "b": index + 1, "enabled": True, "type": "rotary"}
                         for index, name in enumerate(("pad_flex_-1", "pad_flex_1"))]}
    result.update(changes)
    return result


def goal(id="hop"):
    return SkillGoal(id, "directed_hop", (224., 346.5), .8, 4., 3.)


def sample_episode(*, id="synthetic-training", family="train", split="training", intervention="none", privilege="none"):
    source = ActionSource("authored", "scripted", "cellular-authored/v1", "a" * 64, privilege)
    context = PolicyContext("hop", "directed_hop", (224., 346.5), .8, 4., 3.)
    action = WireAction(move=(.2, 0), effort=.8).to_dict()
    packet = {"version": 2, "kind": "physical_wire_action", "run_id": "synthetic-graph", "epoch": 1,
              "actor_id": 0, "sequence": 0, "observed_tick": 0, "observation_id": view()["observation_id"],
              "lease_ticks": 12, "action": action}
    ticks = tuple(TickRecord(t, t + 1, t + 1, t * 10, t * 10 + 1,
                            (packet,) if t == 0 else (), ({"accepted": True, "reason": ""},) if t == 0 else (),
                            packet, "authored", action, {"linear_impulse": [0., 0.]},
                            {"available": False, "reason": "synthetic fixture has no passive wrench"},
                            {"velocity": [0., 0.]}, view(t + 1), t) for t in range(6))
    event = CaptureEvent("step", 0, 0, 0, 6, 0, 60, source, view(), policy_view(view()),
                         requested_packet=packet, source_observation=view(), context=context,
                         ticks=ticks, next_observation=view(6), intervention=intervention)
    files = {"synthetic-fixture": "b" * 64}
    return EpisodeRecord(id, {"lane": "act004", "synthetic_test_fixture": True}, "c" * 40,
                         sha256_json(files), {"seed": 7}, {"synthetic_profile": True}, view(), (event,),
                         "completed", "synthetic_complete", family, split, source_files=files)


class CellularCoreTests(unittest.TestCase):
    def test_node_sensors_are_local_and_exclude_evaluator(self):
        base = make_frame(view(), goal())
        changed = view(evaluator={"nearest_goal": [-1000, -1000]}, reward=1e9, actors=[{"secret": True}],
                       cell_patch={"materials": [99]})
        changed["parts"][2]["velocity"] = [80., -100.]
        frame = make_frame(changed, goal())
        self.assertEqual(len(FEATURE_NAMES), 30)
        self.assertEqual(base.nodes[:2], frame.nodes[:2])
        self.assertNotEqual(base.nodes[2], frame.nodes[2])
        for forbidden in ("evaluator", "reward", "actors", "cell_patch", "parts"):
            self.assertNotIn(forbidden, frame.root_view)
        self.assertEqual(frame.nodes[1][7:17], (0.,) * 10)
        self.assertEqual(frame.nodes[2][7:17], (0.,) * 10)

    def test_root_contact_and_validity_never_enter_pad_slots(self):
        known = make_frame(view(), goal())
        unknown = make_frame(view(grounded=None, energy=None, cooldown=None, ground_distance=None, ceiling_distance=None), goal())
        self.assertEqual(known.nodes[1:], unknown.nodes[1:])
        self.assertEqual(unknown.nodes[0][7:17], (0.,) * 10)
        self.assertEqual(known.nodes[0][8], 1.)
        self.assertEqual(known.nodes[0][12], 1.)

    def test_shared_goal_and_previous_request_are_explicit(self):
        previous = WireAction(move=(-.7, 0), trim=.2, effort=.4, primary=True)
        frame = make_frame(view(60), goal(), previous=previous)
        for node in frame.nodes:
            self.assertAlmostEqual(node[17], 2 / 3)
            self.assertEqual(node[20:26], (-.7, .2, .4, 1., 0., 0.))
        with self.assertRaises(TypeError):
            frame.root_view["tick"] = 7

    def test_observed_physical_topology_changes_are_refused(self):
        variants = []
        changed = view(); changed["parts"].reverse(); variants.append(changed)
        changed = view(); changed["parts"][1]["detached"] = True; variants.append(changed)
        changed = view(); changed["parts"][1]["index"] = True; variants.append(changed)
        changed = view(); changed["joints"][0]["enabled"] = False; variants.append(changed)
        changed = view(); changed["joints"][0]["id"] = "new_joint"; variants.append(changed)
        changed = view(); changed["joints"][1]["b"] = 1; variants.append(changed)
        variants.extend((view(parts=None), view(joints=[None, None])))
        for changed in variants:
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                checked_parts(changed)

    def test_invalid_profile_owner_and_nonfinite_sensor_are_refused(self):
        for changes in ({"profile_version": 2}, {"control_generation": True}, {"epoch": 0}, {"model": "astra"}):
            with self.subTest(changes=changes), self.assertRaises((ContractError, ValueError)):
                make_frame(view(**changes), goal())
        changed = view(); changed["parts"][2]["mass"] = float("nan")
        with self.assertRaises(ContractError):
            make_frame(changed, goal())

    def test_primary_requires_supported_known_ready_rising_edge(self):
        output = [2., -3., 2., 1., 1., 1.]
        action = mapped_request(output, view(), goal(), WireAction(effort=0))
        self.assertTrue(action.primary)
        self.assertEqual((action.move, action.trim, action.effort), ((1., 0.), -1., 1.))
        for changed, previous in ((view(grounded=False), WireAction()), (view(grounded=None), WireAction()),
                                  (view(cooldown=None), WireAction()), (view(cooldown=.1), WireAction()),
                                  (view(), WireAction(primary=True))):
            self.assertFalse(mapped_request(output, changed, goal(), previous).primary)
        self.assertEqual(action.reel, 0)
        self.assertFalse(action.tool or action.attach_usable or action.release_usable)
        with self.assertRaises(ContractError):
            mapped_request([float("nan")] * 6, view(), goal(), WireAction())

    def test_registration_does_not_import_optional_learning_runtime(self):
        code = ("import sys; from nca.registry import Registry; from nca.cellular import register; "
                "r=Registry(); register(r); assert len(r.list('cellular'))==2; "
                "assert 'torch' not in sys.modules; print('core-only')")
        child = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(child.stdout.strip(), "core-only")


class CellularMessageTests(unittest.TestCase):
    def setUp(self):
        self.owner = GraphIdentity.from_view(view(), goal())
        self.messages = [[.1] * 8, [.2] * 8, [.3] * 8]

    def test_normal_one_hop_routing_and_no_implicit_hold(self):
        bus = MessageBus()
        sums, counts, trace = bus.exchange(self.messages, 0, self.owner)
        self.assertEqual(counts, [2, 1, 1])
        self.assertEqual(sums[0], [.5] * 8)
        self.assertEqual(sums[1], [.1] * 8)
        self.assertEqual({(m.source, m.destination) for m in trace["arrived"]}, {(0, 1), (1, 0), (0, 2), (2, 0)})
        sums, counts, _ = bus.exchange([[0.] * 8] * 3, 6, self.owner)
        self.assertEqual(sums, [[0.] * 8] * 3)
        self.assertEqual(bus.pending, [])

    def test_delay_retains_original_source_ticks_and_exclusive_expiry(self):
        bus = MessageBus(FaultSpec("delay"))
        self.assertEqual(bus.exchange(self.messages, 0, self.owner)[1], [0, 0, 0])
        _, counts, trace = bus.exchange(self.messages, 6, self.owner)
        self.assertEqual(counts, [2, 1, 1])
        self.assertTrue(all(message.observed_tick == 0 and message.delivery_tick == 6 for message in trace["arrived"]))
        stale = MessageBus(FaultSpec("delay"))
        stale.exchange(self.messages, 0, self.owner)
        _, counts, trace = stale.exchange(self.messages, 12, self.owner)
        self.assertEqual(counts, [0, 0, 0])
        self.assertEqual(trace["totals"]["expired"], 4)

    def test_partition_is_communication_only_and_loss_has_own_seed(self):
        bus = MessageBus(FaultSpec("partition"))
        _, counts, trace = bus.exchange(self.messages, 0, self.owner)
        self.assertEqual(counts, [1, 1, 0])
        self.assertEqual(trace["totals"]["partitioned"], 2)
        before = view()
        make_frame(before, goal())
        self.assertTrue(before["joints"][1]["enabled"])
        left, right = MessageBus(FaultSpec("loss"), seed=77), MessageBus(FaultSpec("loss"), seed=77)
        for tick in range(0, 600, 6):
            self.assertEqual(left.exchange(self.messages, tick, self.owner), right.exchange(self.messages, tick, self.owner))
        self.assertGreater(left.totals["lost"], 0)
        self.assertLess(left.totals["lost"], left.totals["sent"])

    def test_new_owner_can_restart_clock_and_old_messages_do_not_arrive(self):
        bus = MessageBus(FaultSpec("delay"))
        bus.exchange(self.messages, 120, self.owner)
        fresh = replace(self.owner, run_id="fresh", epoch=2)
        _, counts, trace = bus.exchange(self.messages, 0, fresh)
        self.assertEqual(counts, [0, 0, 0])
        self.assertEqual(trace["pending"], 4)
        self.assertTrue(all(message.owner_sha256 == fresh.sha256 for message in bus.pending))
        for field in ("control_generation", "source_generation", "mount_generation"):
            fenced = replace(fresh, **{field: 1})
            self.assertEqual(bus.exchange(self.messages, 0, fenced)[1], [0, 0, 0])

    def test_cancel_node_reset_clock_and_capacity_are_bounded(self):
        bus = MessageBus(FaultSpec("delay"))
        bus.exchange(self.messages, 0, self.owner)
        bus.clear_nodes((2,))
        self.assertEqual(len(bus.pending), 2)
        with self.assertRaises(ContractError):
            bus.exchange(self.messages, 0, self.owner)
        crowded = MessageBus(FaultSpec("delay"))
        for tick in range(4):
            crowded.exchange(self.messages, tick, self.owner)
        with self.assertRaises(ContractError):
            crowded.exchange(self.messages, 4, self.owner)
        self.assertLessEqual(len(crowded.pending), 16)
        crowded.reset()
        self.assertEqual(crowded.pending, [])

    def test_no_message_control_and_invalid_packet_capacity(self):
        bus = MessageBus(enabled=False)
        self.assertEqual(bus.exchange(self.messages, 0, self.owner)[1], [0, 0, 0])
        self.assertEqual(bus.totals["sent"], 0)
        for payload in ((.1,) * 9, (2.,) * 8, (float("nan"),) * 8):
            with self.assertRaises(ContractError):
                Message(self.owner.sha256, 0, 1, 0, 0, payload)
        with self.assertRaises(ContractError):
            Message(self.owner.sha256, 1, 2, 0, 0, (.1,) * 8)


class CellularOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(DECLARATION.read_text())

    def outcomes(self, graph=True, control=False):
        result = []
        for seed, method, family, fault in itertools.product(self.config["training_seeds"], ("authored", "graph", "no_message"),
                [f["id"] for f in self.config["evaluation_families"]], [f["id"] for f in self.config["faults"]]):
            success = graph if method == "graph" else control if method == "no_message" else True
            result.append({"stage": "evaluation", "seed": seed, "method": method, "family": family, "fault": fault,
                "metrics": {"success": success, "integrity_failure": False, "failed_admissions": 0,
                            "capped_completion_seconds": 2. if success else 3., "mean_absolute_angle": .02,
                            "inference_p95_ms": .2}})
        return result

    def test_missing_duplicate_and_invalid_metrics_cannot_pass(self):
        rows = self.outcomes()
        self.assertEqual(utility_gate(rows[:-1], self.config)["decision"], "inconclusive")
        self.assertEqual(utility_gate(rows + rows[:1], self.config)["decision"], "inconclusive")
        rows[0]["metrics"]["inference_p95_ms"] = None
        self.assertEqual(utility_gate(rows, self.config)["decision"], "inconclusive")

    def test_equal_models_are_no_go_and_positive_all_seed_control_can_pass(self):
        equal = utility_gate(self.outcomes(control=True), self.config)
        self.assertEqual(equal["decision"], "no-go-at-this-screen")
        self.assertFalse(equal["checks"]["normal_benefit"])
        better = utility_gate(self.outcomes(), self.config)
        self.assertEqual(better["decision"], "conditional-utility")
        self.assertFalse(better["owner_feel_accepted"])
        self.assertEqual(better["success_gain"]["resamples"], 27)

    def test_one_seed_gain_is_not_positive_three_seed_lower_bound(self):
        result = bootstrap_three_seed((0., 0., 1.))
        self.assertEqual(result["lower95"], 0.)
        with self.assertRaises(ContractError):
            bootstrap_three_seed((1., 1.))

    def test_stress_or_admission_failure_blocks_advancement(self):
        for mode in ("stress", "admission"):
            rows = self.outcomes()
            for row in rows:
                if row["method"] == "graph" and row["fault"] == "partition":
                    if mode == "stress":
                        row["metrics"]["success"] = False
                    else:
                        row["metrics"]["failed_admissions"] = 1
            self.assertEqual(utility_gate(rows, self.config)["decision"], "no-go-at-this-screen")

    def test_arrival_uses_twelve_actual_ticks_not_one_decision_estimate(self):
        task = ArrivalTask(goal(), view(position=list(goal().target)))
        action = WireAction(effort=0)
        def receipts(start):
            return [{"tick_before": tick, "observation": view(tick + 1, position=list(goal().target)),
                     "admission": []} for tick in range(start, start + 6)]
        task.consume(action, receipts(0))
        self.assertFalse(task.done)
        task.consume(action, receipts(6))
        self.assertTrue(task.metrics()["success"])
        self.assertEqual(task.metrics()["capped_completion_seconds"], .2)
        with self.assertRaises(ContractError):
            task.consume(action, receipts(12))

    def test_receipt_gap_is_an_integrity_failure(self):
        task = ArrivalTask(goal(), view())
        with self.assertRaises(ContractError):
            task.consume(WireAction(), [{"tick_before": 1, "observation": view(2)}])
        self.assertTrue(task.metrics()["integrity_failure"])
        self.assertFalse(task.metrics()["success"])

    def test_unmeasured_failed_angle_and_inference_are_null(self):
        task = ArrivalTask(goal(), view())
        task.integrity_failure = True
        self.assertIsNone(task.metrics()["mean_absolute_angle"])
        self.assertIsNone(task.metrics()["inference_p95_ms"])
        self.assertFalse(task.metrics()["success"])


@unittest.skipUnless(torch is not None, "optional pinned Torch CPU environment required")
class CellularLearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(72)
        from nca.cellular.model import GraphController, GraphPolicy
        self.GraphController, self.GraphPolicy = GraphController, GraphPolicy

    def controller(self, **kwargs):
        return self.GraphController(self.GraphPolicy(), profile_sha256=JOLT_PROFILE_SHA256, **kwargs)

    def test_same_parameter_budget_and_no_message_root_cannot_see_pad_inputs(self):
        graph, control = self.GraphPolicy(), self.GraphPolicy("no_message")
        control.load_state_dict(graph.state_dict())
        self.assertEqual(graph.specification()["parameter_count"], 2878)
        self.assertEqual(control.specification()["parameter_count"], 2878)
        inputs = torch.tensor([make_frame(view(), goal()).nodes])
        changed = inputs.clone(); changed[:, 2, :7] += .4
        hidden = graph.zero_hidden()
        g0, gh0 = graph(inputs, hidden); g1, gh1 = graph(changed, hidden)
        self.assertFalse(torch.equal(g0, g1))
        self.assertTrue(torch.equal(gh0[:, 1], gh1[:, 1]))
        _, second0 = graph(inputs, gh0); _, second1 = graph(changed, gh1)
        self.assertFalse(torch.equal(second0[:, 1], second1[:, 1]))
        c0, ch0 = control(inputs, hidden); c1, ch1 = control(changed, hidden)
        self.assertTrue(torch.equal(c0, c1))
        c0, _ = control(inputs, ch0); c1, _ = control(changed, ch1)
        self.assertTrue(torch.equal(c0, c1))

    def test_self_message_parameters_remain_active_in_no_message_control(self):
        policy = self.GraphPolicy("no_message")
        _, hidden = policy(torch.randn(2, 3, 30), policy.zero_hidden(2))
        output, _ = policy(torch.randn(2, 3, 30), hidden)
        output.square().sum().backward()
        for name, parameter in policy.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)

    def test_duplicate_tick_does_not_advance_memory_or_send_twice(self):
        control = self.controller()
        first = control.step(view(), goal())
        hidden, sent = control.hidden.clone(), control.bus.totals["sent"]
        second = control.step(view(), goal())
        self.assertIs(first, second)
        self.assertTrue(torch.equal(hidden, control.hidden))
        self.assertEqual(control.bus.totals["sent"], sent)
        with self.assertRaises(ContractError):
            control.step(view(observation_id="changed-source"), goal())
        self.assertEqual(control.bus.pending, [])

    def test_generation_fences_retire_old_goal_and_clear_delayed_memory(self):
        for field in ("control_generation", "source_generation", "mount_generation"):
            control = self.controller(fault=FaultSpec("delay"))
            control.step(view(), goal())
            changed, options = view(6), {}
            if field == "control_generation":
                changed[field] = 1
            else:
                options[field] = 1
            frame = control.step(changed, goal(), **options)
            self.assertEqual(frame.reason, "retired_goal")
            self.assertEqual(frame.action, WireAction(effort=0))
            self.assertEqual(control.bus.pending, [])
            self.assertEqual(float(control.hidden.abs().sum()), 0.)
            fresh = control.step(changed, goal("new-hop"), **options)
            self.assertEqual(fresh.reason, "graph_root_readout")

    def test_old_goal_cannot_resume_after_new_goal_and_reset_releases_incarnation(self):
        control = self.controller()
        control.step(view(), goal())
        control.step(view(6), goal("next"))
        self.assertEqual(control.step(view(12), goal()).reason, "retired_goal")
        control.reset()
        self.assertEqual(control.step(view(), goal()).reason, "graph_root_readout")

    def test_unknown_topology_invalid_sensor_and_missed_cadence_disarm(self):
        for kind in ("topology", "sensor", "cadence"):
            control = self.controller(fault=FaultSpec("delay"))
            control.step(view(), goal())
            changed = view(12 if kind == "cadence" else 6)
            if kind == "topology":
                changed["parts"][1]["detached"] = True
            if kind == "sensor":
                changed["parts"][1]["mass"] = float("nan")
            with self.assertRaises((ContractError, ValueError)):
                control.step(changed, goal())
            self.assertEqual(control.bus.pending, [])
            self.assertEqual(float(control.hidden.abs().sum()), 0.)

    def test_memory_reset_keeps_public_goal_clock_and_original_physics(self):
        control = self.controller(fault=FaultSpec("memory_reset"))
        for tick in range(0, 60, 6):
            control.step(view(tick), goal())
        physical = view(60, velocity=[5., -3.])
        before = copy.deepcopy(physical)
        frame = control.step(physical, goal())
        self.assertEqual(frame.hidden_before, ((0.,) * 16,) * 3)
        self.assertAlmostEqual(frame.frame.nodes[0][17], 2 / 3)
        self.assertEqual(physical, before)
        self.assertEqual(control.start_tick, 0)

    def test_nonfinite_input_and_invalid_delivered_capacity_are_refused(self):
        policy = self.GraphPolicy()
        inputs, hidden = torch.zeros(1, 3, 30), policy.zero_hidden()
        inputs[0, 1, 2] = float("inf")
        with self.assertRaises(ContractError):
            policy(inputs, hidden)
        encoded, messages = policy.encode(torch.zeros_like(inputs), hidden)
        with self.assertRaises(ContractError):
            policy.update(encoded, messages, hidden, torch.zeros(1, 3, 8), torch.full((1, 3, 1), 3.))

    def test_padding_has_no_loss_or_hidden_state_transition(self):
        from nca.cellular.training import pad_sequences, sequence_loss
        short = ([make_frame(view(), goal()).nodes], [[.1, -.2, .8, 0., 0., 0.]])
        long = (short[0] * 3, short[1] * 3)
        policy, weights = self.GraphPolicy(), torch.ones(3)
        batch_loss, batch_hidden = sequence_loss(policy, *pad_sequences([short, long]), weights)
        short_loss, short_hidden = sequence_loss(policy, *pad_sequences([short]), weights)
        long_loss, long_hidden = sequence_loss(policy, *pad_sequences([long]), weights)
        torch.testing.assert_close(batch_loss, (short_loss + 3 * long_loss) / 4)
        torch.testing.assert_close(batch_hidden[0], short_hidden[0])
        torch.testing.assert_close(batch_hidden[1], long_hidden[0])

    def test_training_excludes_holdout_and_refuses_corrections_or_privileged_teachers(self):
        from nca.cellular.training import training_sequences
        train = sample_episode()
        evaluation = sample_episode(id="sealed", family="heldout", split="evaluation", privilege="privileged_teacher")
        split = ScenarioSplit(("train",), ("tune",), ("heldout",))
        sequences = training_sequences(Dataset("checked", (train, evaluation), split))
        self.assertEqual(len(sequences), 1)
        self.assertEqual(len(sequences[0][0]), 1)
        for changes in ({"intervention": "human_correction"}, {"privilege": "privileged_teacher"}):
            with self.assertRaises(ContractError):
                training_sequences(Dataset("invalid", (sample_episode(**changes),), split))
        self.assertFalse(train.events[0].on_policy_eligible)

    def test_weight_bundle_enforces_config_topology_and_tensor_digest(self):
        from nca.cellular.model import load_policy, save_policy, weight_sha256
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            policy = self.GraphPolicy()
            save_policy(policy, path, {"config_sha256": "d" * 64})
            loaded, _ = load_policy(path, expected_config_sha256="d" * 64)
            self.assertEqual(weight_sha256(policy), weight_sha256(loaded))
            with self.assertRaises(FileExistsError):
                save_policy(policy, path, {})
            with self.assertRaises(ContractError):
                load_policy(path, expected_config_sha256="e" * 64)
            payload = torch.load(path, weights_only=True)
            payload["state_dict"]["readout.bias"][0] += .1
            corrupt = Path(directory) / "corrupt.pt"
            torch.save(payload, corrupt)
            with self.assertRaises(ContractError):
                load_policy(corrupt)
            payload["specification"]["readout_node"] = 2
            torch.save(payload, corrupt)
            with self.assertRaises(ContractError):
                load_policy(corrupt)

    def test_fixed_bc_updates_parameters_reproducibly_without_holdout_selection(self):
        from nca.cellular.model import weight_sha256
        from nca.cellular.training import train_bc
        short = ([make_frame(view(), goal()).nodes], [[.4, -.2, .8, 0., 0., 0.]])
        sequences = [short, (short[0] * 2, short[1] * 2)]
        first, second = self.GraphPolicy(), self.GraphPolicy()
        second.load_state_dict(first.state_dict())
        initial_hash = weight_sha256(first)
        reports = [train_bc(policy, sequences, 17011) for policy in (first, second)]
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(weight_sha256(first), weight_sha256(second))
        self.assertNotEqual(weight_sha256(first), initial_hash)
        self.assertEqual(reports[0]["updates"], 30)
        self.assertEqual(len(reports[0]["epochs"]), 30)
        self.assertLess(reports[0]["final_training_loss"], reports[0]["initial_training_loss"])

    def test_preregistration_and_global_ceiling_are_fixed(self):
        from nca.cellular.screen import CONFIG_SHA256, ScreenBudget, load_declaration
        from nca.data.capture import CaptureBudgetExceeded
        self.assertEqual(sha256_json(load_declaration(DECLARATION)), CONFIG_SHA256)
        altered = json.loads(DECLARATION.read_text())
        altered["utility_gate"]["minimum_normal_success_gain_vs_no_message"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(altered))
            with self.assertRaises(ContractError):
                load_declaration(path)
        clock = [0.]
        budget = ScreenBudget(clock=lambda: clock[0])
        budget.decisions = 3999
        budget.claim_decision()
        with self.assertRaises(CaptureBudgetExceeded):
            budget.claim_decision()
        self.assertEqual(budget.decisions, 4000)
        budget = ScreenBudget(clock=lambda: clock[0])
        clock[0] = 900.
        with self.assertRaises(CaptureBudgetExceeded):
            budget.check()
        with self.assertRaises(ContractError):
            ScreenBudget(max_seconds=901)

    def test_prior_attempts_consume_the_original_decision_and_wall_ceiling(self):
        from nca.cellular.screen import BudgetReceipt, PriorBudget, ScreenBudget
        from nca.data.capture import CaptureBudgetExceeded
        prior = PriorBudget((BudgetReceipt("earlier-attempt", "e" * 64, 3999, 890.),))
        clock = [10.]
        budget = ScreenBudget(clock=lambda: clock[0], prior=prior)
        self.assertEqual((budget.decisions, budget.elapsed), (3999, 890.))
        budget.claim_decision()
        self.assertEqual(budget.attempt_decisions, 1)
        with self.assertRaises(CaptureBudgetExceeded):
            budget.claim_decision()
        budget = ScreenBudget(clock=lambda: clock[0], prior=prior)
        clock[0] += 10.
        self.assertEqual(budget.attempt_elapsed, 10.)
        with self.assertRaises(CaptureBudgetExceeded):
            budget.check()
        self.assertEqual(prior.game_decisions, 3999)
        self.assertEqual(prior.wall_seconds, 890.)

    def test_prior_receipts_are_strict_finite_and_cannot_extend_a_ceiling(self):
        from nca.cellular.screen import BudgetReceipt, PriorBudget
        valid = BudgetReceipt("earlier-attempt", "e" * 64, 17, 8.641717033)
        self.assertEqual(PriorBudget.from_json(PriorBudget((valid,)).to_json()), PriorBudget((valid,)))
        for changes in ({"game_decisions": True}, {"game_decisions": -1}, {"game_decisions": 4001},
                        {"wall_seconds": True}, {"wall_seconds": -1.}, {"wall_seconds": float("nan")},
                        {"wall_seconds": float("inf")}, {"wall_seconds": 901.}, {"content_sha256": "missing"}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                BudgetReceipt.from_dict({**valid.to_dict(), **changes})
        for receipts in ((valid, valid), (valid, BudgetReceipt("more-time", "d" * 64, 0, 900.)),
                         (valid, BudgetReceipt("more-decisions", "d" * 64, 4000, 0.))):
            with self.assertRaises(ContractError):
                PriorBudget(receipts)
        with self.assertRaises(ContractError):
            PriorBudget.from_dict({"receipts": [], "maximum_game_decisions": 8000})

    def test_run_ledger_is_published_before_host_and_retains_original_feature_schema(self):
        from nca.cellular.core import FEATURE_SCHEMA
        from nca.cellular.screen import CONFIG_SHA256, RUN_OBSERVATION_SCHEMA, run_screen
        from nca.runs import RunStore
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attempt"
            store = RunStore(output / "runs")

            def stopped_host(*args, **kwargs):
                prepared = store.read("t110-jolt-three-part-preflight")
                self.assertEqual(prepared.status, "prepared")
                self.assertEqual(prepared.decisions, 0)
                self.assertEqual(prepared.observation_schema, RUN_OBSERVATION_SCHEMA)
                self.assertEqual(prepared.resolved_config["sensors"]["schema"], FEATURE_SCHEMA)
                self.assertEqual(prepared.config_sha256, CONFIG_SHA256)
                raise ContractError("synthetic_host_stop", "no physical host is started in this ledger check")

            with patch("nca.cellular.screen.frozen_sources", return_value=("a" * 40, {}, "b" * 64)), \
                    patch("nca.cellular.screen.GodotHost", side_effect=stopped_host) as host, \
                    patch("nca.cellular.screen.torch.__version__", "2.8.0+cpu"), \
                    patch("nca.cellular.screen.np.__version__", "2.3.5"):
                summary = run_screen(config_path=DECLARATION, source_root=ROOT, output=output,
                                     host_checkout="synthetic", godot="synthetic")
            host.assert_called_once()
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["failure"], "ContractError:synthetic_host_stop")
            self.assertEqual(summary["attempt_game_decisions"], 0)
            finished = store.read("t110-jolt-three-part-screen")
            self.assertEqual(finished.observation_schema, RUN_OBSERVATION_SCHEMA)
            self.assertEqual(finished.resolved_config["sensors"]["schema"], FEATURE_SCHEMA)
            self.assertEqual(len(store.verify()), 2)
            with self.assertRaises(ContractError):
                store.write(finished)

    def test_cli_does_not_mislabel_escaped_recording_failure_as_preexperiment(self):
        from nca.cellular.screen import main
        output = io.StringIO()
        with patch("nca.cellular.screen.run_screen", side_effect=ContractError("invalid_identifier", "recording failed")), \
                redirect_stdout(output):
            code = main(["--config", "synthetic", "--source-root", "synthetic", "--output", "synthetic",
                         "--host", "synthetic", "--godot", "synthetic"])
        receipt = json.loads(output.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["phase"], "unknown_consult_attempt_artifacts")


@unittest.skipUnless(torch is not None and os.environ.get("NCA_HOST_CHECKOUT") and os.environ.get("NCA_GODOT"),
                     "actual pinned host plus Torch CPU required; no physical graph qualification")
class CellularActualHostTests(unittest.TestCase):
    def test_actual_owned_parts_and_cancel_clear_controller_only(self):
        from nca.adapters.godot import CybersandEnv, GodotHost, physical_config
        from nca.cellular.model import GraphController, GraphPolicy
        torch.set_num_threads(1)
        torch.manual_seed(73)
        with GodotHost(os.environ["NCA_HOST_CHECKOUT"], os.environ["NCA_GODOT"]) as host:
            env = CybersandEnv(host, physical_config("jev", ticks=24, seed=73))
            issued, _ = env.reset()
            self.assertEqual(sha256_json(host.request("profile")["profile"]), JOLT_PROFILE_SHA256)
            control = GraphController(GraphPolicy(), profile_sha256=JOLT_PROFILE_SHA256, fault=FaultSpec("delay"))
            request = control.step(issued, goal())
            observed, _, _, _, info = env.step(request.action.to_dict())
            self.assertEqual(len(info["trace"]), 6)
            self.assertTrue(all(tick["applied"].get("available", True) for tick in info["trace"]))
            frame = control.step(observed, goal())
            self.assertTrue(all(message.observed_tick == issued["tick"] for message in frame.communication["arrived"]))
            canceled = env.cancel()
            self.assertEqual(canceled["velocity"], observed["velocity"])
            self.assertEqual(canceled["parts"], observed["parts"])
            self.assertGreater(canceled["control_generation"], observed["control_generation"])
            stale = control.step(canceled, goal())
            self.assertEqual(stale.reason, "retired_goal")
            self.assertEqual(control.bus.pending, [])
            self.assertFalse(stale.action.primary)


if __name__ == "__main__":
    unittest.main()
