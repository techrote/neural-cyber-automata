"""Untrained numeric locality/reset checks; no optimization or game rollout."""

from dataclasses import replace
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

from nca.contracts import ContractError, strict_json_loads
from nca.controllers.morphology import information_reach
from test_morphology import ROOT, fixture, goal

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ReceiptOwner:
    """Synthetic native receipt boundary; this fixture does not run the game."""
    def __init__(self):
        self.calls = self.cancelled = 0
        self.view = None

    def command(self, action, observation, lease_ticks):
        from nca.adapters.protocol import make_physical_command
        self.calls += 1
        self.view = deepcopy(observation)
        return make_physical_command(observation, self.calls, action, lease_ticks)

    def step_packet(self, packet):
        before = deepcopy(self.view)
        self.view["tick"] += 6
        self._stamp()
        info = {"actual_ticks": 6, "elapsed_ticks": 6, "delay_ticks_before_repeat": 0,
                "consumed_trace_complete": True, "trace": []}
        for i in range(6):
            info["trace"].append({"tick_before": before["tick"] + i, "tick_after": before["tick"] + i + 1,
                                   "failed": False,
                                   "requested": [deepcopy(packet)] if i == 0 else [],
                                   "admission": [{"accepted": True}] if i == 0 else [],
                                   "applied": {"action_available": True, "terminal_revocation": False,
                                               "action": deepcopy(packet["action"])}})
        return deepcopy(self.view), 0., False, False, info

    def _stamp(self):
        self.view["observation_id"] = ":".join(str(self.view[key]) for key in
                                               ("run_id", "epoch", "actor_id", "control_generation", "tick"))

    def cancel(self):
        self.cancelled += 1
        if self.view is not None:
            self.view["control_generation"] += 1
            self._stamp()
            return deepcopy(self.view)


def permuted(frame, order):
    inverse = {old: new for new, old in enumerate(order)}
    return replace(frame, nodes=tuple(frame.nodes[i] for i in order),
                   node_valid=tuple(frame.node_valid[i] for i in order),
                   present=tuple(frame.present[i] for i in order), active=tuple(frame.active[i] for i in order),
                   root_index=inverse[frame.root_index],
                   edges=tuple(replace(edge, source=inverse[edge.source], destination=inverse[edge.destination])
                               for edge in frame.edges))


@unittest.skipUnless(HAS_TORCH, "optional CPU Torch needed for candidate arithmetic")
class MorphologyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.torch = torch
        from nca.controllers.morphology.models import MaskedRecurrentPolicy, SharedGraphPolicy, MorphologyRuntime
        cls.Padded, cls.Graph, cls.Runtime = MaskedRecurrentPolicy, SharedGraphPolicy, MorphologyRuntime

    def setUp(self):
        self.torch.manual_seed(6106)
        self.adapter, self.view = fixture("ripper")
        self.goal = goal(self.view)
        self.frame = self.adapter.frame(self.view, self.goal)

    def test_declared_parameter_and_compute_budgets_match_actual_modules(self):
        from nca.controllers.morphology.models import candidate_cost
        config = strict_json_loads((ROOT / "configs/morphology/comparison-v1.json").read_bytes())["candidates"]
        counts = []
        for model in (self.Padded(), self.Graph()):
            cost = candidate_cost(model, self.frame)
            self.assertEqual(cost["parameters"], config[model.family]["parameters"])
            self.assertLessEqual(cost["dense_multiply_accumulates_per_decision"], config["matched_dense_macs_ceiling_per_decision"])
            counts.append(cost["parameters"])
        self.assertLessEqual(max(counts) / min(counts), config["maximum_parameter_count_ratio"])

    def test_graph_current_input_has_only_declared_neighbor_hop_reach(self):
        for rounds in (1, 2):
            model = self.Graph(rounds)
            inputs = model.node_inputs(self.frame).requires_grad_(True)
            output = model.forward_frame(self.frame, model.zero_hidden(), node_inputs=inputs)
            output.logits.sum().backward()
            far_influence = float(inputs.grad[2].abs().sum())  # Ripper roller is two joints from the root.
            with self.subTest(rounds=rounds):
                if rounds == 1:
                    self.assertEqual(far_influence, 0.)
                    self.assertNotIn(2, information_reach(self.frame, rounds))
                else:
                    self.assertGreater(far_influence, 1e-8)
                    self.assertIn(2, information_reach(self.frame, rounds))

    def test_permutation_equivariance_has_no_rig_or_part_index_embedding(self):
        model = self.Graph()
        order = (4, 2, 0, 3, 1, 7, 5, 6)
        first = model.forward_frame(self.frame, model.zero_hidden())
        second = model.forward_frame(permuted(self.frame, order), model.zero_hidden())
        self.assertTrue(self.torch.allclose(first.logits, second.logits, atol=1e-7, rtol=1e-6))
        self.assertTrue(self.torch.allclose(first.hidden[list(order)], second.hidden, atol=1e-7, rtol=1e-6))

    def test_padding_neither_updates_memory_nor_enters_loss(self):
        for model in (self.Padded(), self.Graph()):
            hidden = self.torch.ones_like(model.zero_hidden())
            output = model.forward_frame(None, hidden)
            self.assertFalse(output.loss_mask)
            self.assertTrue(self.torch.equal(output.hidden, hidden))
            self.assertIsNot(output.hidden, hidden)

    def test_absent_or_disconnected_nodes_cannot_leak_hidden_values(self):
        model = self.Graph()
        _, detached = fixture("ripper")
        detached["parts"][2]["detached"] = True
        frame = self.adapter.frame(detached, self.goal)
        baseline = model.forward_frame(frame, model.zero_hidden())
        dirty = model.zero_hidden()
        dirty[2] = 1e6
        dirty[5:] = -1e6
        output = model.forward_frame(frame, dirty)
        self.assertTrue(self.torch.equal(output.logits, baseline.logits))
        self.assertTrue(self.torch.equal(output.hidden[2], self.torch.zeros(32)))
        self.assertTrue(self.torch.equal(output.hidden[5:], self.torch.zeros(3, 32)))

    def test_shared_weights_do_not_share_runtime_state(self):
        for model in (self.Padded(), self.Graph()):
            first, second = self.Runtime(model, self.adapter), self.Runtime(model, self.adapter)
            decision = first.step(self.view, self.goal)
            self.assertEqual(decision.status, "accepted_unqualified")
            self.assertIsNone(decision.log_probability)
            self.assertTrue(self.torch.equal(second.hidden, model.zero_hidden()))
            self.assertGreater(float(first.hidden.abs().sum()), 0)
            first.cancel()
            self.assertTrue(self.torch.equal(first.hidden, model.zero_hidden()))

    def test_actor_reset_and_mount_change_erase_hidden_and_history(self):
        model = self.Graph()
        runtime = self.Runtime(model, self.adapter)
        runtime.step(self.view, self.goal)
        _, next_view = fixture("ripper", tick=6)
        refused = runtime.step(next_view, self.goal, mount_generation=1)
        self.assertTrue(refused.cancel_owner)
        self.assertEqual(refused.reason, "retired_invocation")
        self.assertTrue(self.torch.equal(runtime.hidden, model.zero_hidden()))
        fresh_goal = replace(self.goal, id="fresh-mount")
        fresh = runtime.step(next_view, fresh_goal, mount_generation=1)
        reference = self.Runtime(model, self.adapter).step(next_view, fresh_goal, mount_generation=1)
        self.assertEqual(fresh.action, reference.action)
        _, reset_view = fixture("ripper", epoch=2)
        reset = runtime.step(reset_view, self.goal)
        self.assertIn("actor_lifetime", reset.reset_reasons)

    def test_unsupported_channel_output_is_labelled_and_material_commands_stay_off(self):
        model = self.Padded()
        with self.torch.no_grad():
            model.readout.bias.fill_(5.)
        result = self.Runtime(model, self.adapter).step(self.view, self.goal)
        self.assertIn("impulse", result.capability_masked)
        self.assertFalse(result.action.primary or result.action.tool or result.action.auxiliary)
        self.assertFalse(result.action.release_usable or result.action.attach_usable)

    def test_dispatch_preserves_source_and_cleans_up_command_construction_failure(self):
        from nca.controllers.morphology.models import dispatch
        model = self.Padded()
        runtime = self.Runtime(model, self.adapter)
        decision = runtime.step(self.view, self.goal)

        class Owner(ReceiptOwner):
            def __init__(self, fail=False):
                super().__init__()
                self.fail = fail
            def command(self, action, observation, lease_ticks):
                packet = super().command(action, observation, lease_ticks)
                if self.fail:
                    raise OSError("construct-failed")
                return packet

        owner = Owner()
        _, newer = fixture("ripper", tick=6)
        with self.assertRaisesRegex(ContractError, "morphology_source_mismatch"):
            dispatch(owner, runtime, decision, newer)
        self.assertEqual((owner.calls, owner.cancelled), (0, 1))
        with self.assertRaisesRegex(ContractError, "morphology_source_mismatch"):
            dispatch(owner, runtime, decision, self.view)
        self.assertEqual(owner.calls, 0)
        owner = Owner(fail=True)
        runtime = self.Runtime(model, self.adapter)
        decision = runtime.step(self.view, self.goal)
        with self.assertRaisesRegex(OSError, "construct-failed"):
            dispatch(owner, runtime, decision, self.view)
        self.assertEqual((owner.calls, owner.cancelled), (1, 1))

    def test_successful_dispatch_cannot_reissue_the_same_decision(self):
        from nca.controllers.morphology.models import dispatch
        runtime = self.Runtime(self.Padded(), self.adapter)
        decision = runtime.step(self.view, self.goal)

        owner = ReceiptOwner()
        dispatch(owner, runtime, decision, self.view)
        with self.assertRaisesRegex(ContractError, "morphology_source_mismatch"):
            dispatch(owner, runtime, decision, self.view)
        self.assertEqual((owner.calls, owner.cancelled), (1, 1))

    def test_delayed_goal_deadline_and_original_source_lease_are_both_bounded(self):
        from nca.controllers.morphology.models import dispatch
        runtime = self.Runtime(self.Padded(), self.adapter)
        context = replace(self.goal, deadline_ticks=12)
        runtime.step(self.view, context)
        _, following = fixture("ripper", tick=6)
        decision = runtime.step(following, context, current_tick=11)
        self.assertEqual(decision.status, "accepted_unqualified")
        self.assertEqual((decision.helper_lease_ticks, decision.goal_deadline_tick), (6, 12))

        class Owner(ReceiptOwner):
            lease = None
            observed_tick = None
            def command(self, action, observation, lease_ticks):
                self.lease, self.observed_tick = lease_ticks, observation["tick"]
                return super().command(action, observation, lease_ticks)
            def cancel(self):
                raise AssertionError("valid bounded dispatch should not need cleanup")

        owner = Owner()
        dispatch(owner, runtime, decision, following)
        self.assertEqual((owner.observed_tick, owner.lease), (6, 6))
        second = self.Runtime(self.Padded(), self.adapter)
        second.step(self.view, context)
        refused = second.step(following, context, current_tick=12)
        self.assertEqual(refused.reason, "morphology_deadline")
        self.assertTrue(refused.cancel_owner)

    def test_ambiguous_step_and_failed_cleanup_latch_new_goals_until_confirmed_cancel(self):
        from nca.controllers.morphology.models import dispatch
        runtime = self.Runtime(self.Padded(), self.adapter)
        decision = runtime.step(self.view, self.goal)

        class Owner(ReceiptOwner):
            def step_packet(self, packet):
                raise OSError("ambiguous-step")
            def cancel(self):
                raise OSError("cancel-unconfirmed")

        owner = Owner()
        with self.assertRaisesRegex(OSError, "ambiguous-step"):
            dispatch(owner, runtime, decision, self.view)
        self.assertTrue(runtime.owner_cleanup_unconfirmed)
        _, following = fixture("ripper", tick=6)
        new_goal = replace(self.goal, id="fresh")
        refused = runtime.step(following, new_goal, source_generation=2)
        self.assertEqual(refused.reason, "owner_cleanup_unconfirmed")
        with self.assertRaisesRegex(OSError, "cancel-unconfirmed"):
            dispatch(owner, runtime, refused, following)
        self.assertEqual(owner.calls, 1)
        with self.assertRaisesRegex(ContractError, "morphology_cancel_receipt"):
            runtime.confirm_owner_cancel(following)
        _, cancelled = fixture("ripper", tick=6, control=1)
        runtime.confirm_owner_cancel(cancelled)
        self.assertFalse(runtime.owner_cleanup_unconfirmed)
        stale = runtime.step(following, new_goal)
        self.assertEqual(stale.reason, "retired_owner")
        fresh = runtime.step(cancelled, new_goal, source_generation=2)
        self.assertEqual(fresh.status, "accepted_unqualified")

    def test_valid_tick_zero_new_incarnation_can_retire_uncertain_previous_owner(self):
        from nca.controllers.morphology.models import dispatch
        runtime = self.Runtime(self.Padded(), self.adapter)

        class Owner(ReceiptOwner):
            def step_packet(self, packet):
                raise OSError("ambiguous")
            def cancel(self):
                raise OSError("unconfirmed")

        decision = runtime.step(self.view, self.goal)
        with self.assertRaises(OSError):
            dispatch(Owner(), runtime, decision, self.view)
        _, same_world = fixture("ripper", tick=6, control=1)
        self.assertEqual(runtime.step(same_world, replace(self.goal, id="new-goal")).reason,
                         "owner_cleanup_unconfirmed")
        _, reset = fixture("ripper", epoch=2)
        fresh = runtime.step(reset, self.goal)
        self.assertEqual(fresh.status, "accepted_unqualified")
        self.assertIn("actor_lifetime", fresh.reset_reasons)
        self.assertFalse(runtime.owner_cleanup_unconfirmed)

    def test_live_source_goal_mount_and_topology_changes_require_real_owner_revocation(self):
        from nca.controllers.morphology.models import dispatch
        for boundary in ("goal", "source", "mount", "topology"):
            runtime = self.Runtime(self.Padded(), self.adapter)
            _, following = fixture("ripper", tick=6)
            _, cancelled = fixture("ripper", tick=6, control=1)

            class Owner(ReceiptOwner):
                commands = 0
                cancellations = 0
                def command(self, action, observation, lease_ticks):
                    self.commands += 1
                    return super().command(action, observation, lease_ticks)
                def cancel(self):
                    self.cancellations += 1
                    return cancelled

            owner = Owner()
            dispatch(owner, runtime, runtime.step(self.view, self.goal), self.view)
            context = replace(self.goal, id="new-goal") if boundary == "goal" else self.goal
            kwargs = {"source_generation": 2} if boundary == "source" else {}
            if boundary == "mount":
                kwargs["mount_generation"] = 2
            if boundary == "topology":
                following["parts"][2]["detached"] = True
                cancelled["parts"][2]["detached"] = True
            refusal = runtime.step(following, context, **kwargs)
            with self.subTest(boundary=boundary):
                self.assertEqual(refusal.reason, "owner_boundary_requires_cancel")
                self.assertTrue(runtime.owner_cleanup_unconfirmed)
                self.assertEqual((owner.commands, owner.cancellations), (1, 0))
                dispatch(owner, runtime, refusal, following)
                self.assertEqual((owner.commands, owner.cancellations), (1, 1))
                self.assertFalse(runtime.owner_cleanup_unconfirmed)
                fresh = runtime.step(cancelled, replace(self.goal, id="after-revocation"), **kwargs)
                self.assertEqual(fresh.status, "accepted_unqualified")
                dispatch(owner, runtime, fresh, cancelled)
                self.assertEqual(owner.commands, 2)

    def test_rejected_missing_and_replaced_native_receipts_cancel_and_erase_recurrence(self):
        from nca.controllers.morphology.models import dispatch
        for mutation in ("rejected", "missing", "replaced", "boundary", "malformed", "terminal_schema"):
            runtime = self.Runtime(self.Padded(), self.adapter)

            class Owner(ReceiptOwner):
                def step_packet(self, packet):
                    transition = super().step_packet(packet)
                    after, _, _, _, info = transition
                    if mutation == "rejected":
                        info["trace"][0]["admission"][0]["accepted"] = False
                    elif mutation == "missing":
                        info["trace"].pop()
                    elif mutation == "replaced":
                        info["trace"][2]["applied"]["action"]["move"][0] = 1.
                    elif mutation == "boundary":
                        after["parts"][2]["detached"] = True
                    elif mutation == "terminal_schema":
                        after["suspended"] = True
                        after["control_generation"] += 1
                        after["observation_id"] = ":".join(str(after[key]) for key in
                            ("run_id", "epoch", "actor_id", "control_generation", "tick"))
                        after["parts"] = []
                        info["trace"][-1]["applied"] = {"action_available": False, "terminal_revocation": True,
                                                       "action": None}
                        return after, 0., False, True, info
                    else:
                        return {"ok": True}
                    return transition

            owner = Owner()
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                dispatch(owner, runtime, runtime.step(self.view, self.goal), self.view)
            self.assertEqual((owner.calls, owner.cancelled), (1, 1))
            self.assertTrue(self.torch.equal(runtime.hidden, runtime.model.zero_hidden()))
            self.assertIsNone(runtime.pending)
            self.assertFalse(runtime.owner_cleanup_unconfirmed)
            self.assertIn(self.goal.id, runtime.fence.retired_goals)

    def test_terminal_readback_remains_unavailable_and_new_physical_reset_is_required(self):
        from nca.controllers.morphology.models import dispatch
        runtime = self.Runtime(self.Padded(), self.adapter)

        class Owner(ReceiptOwner):
            def step_packet(self, packet):
                after, reward, _, _, info = super().step_packet(packet)
                after["suspended"] = True
                after["control_generation"] += 1
                after["observation_id"] = ":".join(str(after[key]) for key in
                                                    ("run_id", "epoch", "actor_id", "control_generation", "tick"))
                info["trace"][-1]["applied"] = {"action_available": False, "terminal_revocation": True,
                                                   "action": None}
                return after, reward, False, True, info

        owner = Owner()
        transition = dispatch(owner, runtime, runtime.step(self.view, self.goal), self.view)
        self.assertTrue(transition[3])
        self.assertIsNone(transition[4]["trace"][-1]["applied"]["action"])
        self.assertTrue(self.torch.equal(runtime.hidden, runtime.model.zero_hidden()))
        self.assertTrue(runtime.owner_cleanup_unconfirmed)
        _, same_world = fixture("ripper", tick=6, control=1)
        self.assertEqual(runtime.step(same_world, replace(self.goal, id="fresh-goal")).reason,
                         "owner_cleanup_unconfirmed")
        _, reset = fixture("ripper", epoch=2)
        self.assertEqual(runtime.step(reset, self.goal).status, "accepted_unqualified")


if __name__ == "__main__":
    unittest.main()
