"""Synthetic v2 action/recurrence/provenance tests; no gameplay qualification."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from nca.contracts import WireAction
    from nca.controllers.baseline import FeedbackObservation, SkillGoal
    from nca.controllers.jolt_learned import (
        ACTION_SCHEMA, FEATURE_NAMES, SAMPLE_SCHEMA, ActionHistory, InvocationContext,
        JoltController, JoltPolicy, combine_residual, emergency_request, features,
        joint_log_probability, load_policy, map_action, mapped_continuous, normalized_aim,
        policy_state_sha256, save_policy,
    )
    from nca.controllers.learned import load_policy as load_old_policy


def observation(tick=0, **changes):
    value = {"version": 2, "representation": "articulated-rapier", "run_id": "synthetic-jolt-v2",
             "epoch": 1, "actor_id": 0, "observation_id": f"jolt-v2:1:0:{tick}",
             "profile_version": 1, "model": "jev", "rig_id": "jev", "tick": tick,
             "position": [0., 0.], "velocity": [0., 0.], "rotation": 0., "angular_velocity": 0.,
             "grounded": True, "cooldown": 0., "energy": 1., "ground_distance": 0.,
             "ceiling_distance": 96., "control_generation": 0,
             "parts": [{"id": "core", "index": 0, "detached": False}]}
    value.update(changes)
    return value


def setUpModule():
    if TORCH_AVAILABLE:
        torch.set_num_threads(1)


@unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning dependencies absent")
class FullJoltPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.goal = SkillGoal("v2-hop", "directed_hop", (48., 0.), timeout_seconds=4.)

    def test_full_aim_and_independent_buttons_reach_the_wire_without_goal_mapping(self):
        for aim in ((.001, -3.), (-.001, -3.), (3., .01), (-3., .01)):
            action = map_action((.2, -.3, .4, *aim), (True, False))
            roundtrip = WireAction.from_dict(action.to_dict())
            self.assertEqual(action, roundtrip)
            self.assertAlmostEqual(sum(v*v for v in action.aim), 1.)
            self.assertEqual(action.aim[0] > 0, aim[0] > 0)
        self.assertLess(map_action((0, 0, 0, .001, -3), (True, False)).aim[1], -.999)
        for primary, secondary in ((True, False), (False, True), (True, True), (False, False)):
            action = map_action((0, 0, 0, 0, 0), (primary, secondary))
            self.assertEqual((action.primary, action.secondary, action.auxiliary), (primary, secondary, False))
            self.assertEqual(action.aim, (0., -1.))

    def test_native_edge_semantics_are_not_replaced_by_a_scripted_pulse_schedule(self):
        desired = [True, True, False, False, True]
        outputs = [map_action((0, 0, 0, 0, -1), (button, False)).primary for button in desired]
        self.assertEqual(outputs, desired)
        edges = [value and (i == 0 or not outputs[i-1]) for i, value in enumerate(outputs)]
        self.assertEqual(edges, [True, False, False, False, True])

    def test_tensor_and_runtime_maps_agree_including_zero_aim(self):
        latent = torch.tensor([[.2, -.3, .4, .1, -2.], [0., 0., 0., 0., 0.]], dtype=torch.float64)
        mapped = mapped_continuous(latent)
        aims = normalized_aim(mapped[:, 3:5])
        for index in range(2):
            action = map_action(latent[index].tolist(), (True, False))
            torch.testing.assert_close(mapped[index, :3], torch.tensor([action.move[0], action.trim, action.effort], dtype=torch.float64))
            torch.testing.assert_close(aims[index], torch.tensor(action.aim, dtype=torch.float64))

    def test_whitelist_and_missing_contact_masks_are_explicit(self):
        context = InvocationContext.create(observation(), self.goal)
        expected = features(observation(), self.goal, context=context)
        polluted = observation(success=True, reward=9999, evaluator={"future_arrival": True}, terrain=[1, 2, 3])
        self.assertEqual(expected, features(polluted, self.goal, context=context))
        measured = features(observation(blocked=False, wall_left=False, wall_right=False, stuck_time=0.), self.goal, context=context)
        for name in ("blocked", "wall_left", "wall_right", "stuck_time"):
            self.assertEqual(expected[FEATURE_NAMES.index(name + "_valid")], 0.)
            self.assertEqual(measured[FEATURE_NAMES.index(name + "_valid")], 1.)
        self.assertEqual(len(expected), 48)

    def test_proposal_dispatch_and_absence_have_distinct_features(self):
        context = InvocationContext.create(observation(), self.goal)
        proposal = WireAction(primary=True, aim=(0., -1.))
        dispatch = WireAction(auxiliary=True, effort=.9)
        history = ActionHistory(proposal, dispatch, True, True)
        encoded = features(observation(6), self.goal, context=context, history=history)
        self.assertEqual(encoded[FEATURE_NAMES.index("previous_proposal_primary")], 1.)
        self.assertEqual(encoded[FEATURE_NAMES.index("previous_dispatched_primary")], 0.)
        self.assertEqual(encoded[FEATURE_NAMES.index("previous_dispatched_auxiliary")], 1.)
        self.assertEqual(encoded[FEATURE_NAMES.index("previous_intervention")], 1.)
        absent = features(observation(6), self.goal, context=context, history=ActionHistory(None, dispatch, True, True))
        self.assertEqual(absent[FEATURE_NAMES.index("previous_proposal_available")], 0.)
        self.assertEqual(ActionHistory.from_dict(history.to_dict()), history)
        self.assertEqual(InvocationContext.from_dict(context.to_dict()), context)

    def test_residual_starts_at_exact_bc_parity_across_entire_histories(self):
        bc = JoltPolicy("bc")
        direct = JoltController(bc)
        residual = JoltController(JoltPolicy("residual"), base_policy=bc)
        for tick in range(0, 240, 6):
            source = observation(tick, position=[tick / 10, 0.], velocity=[1., 0.])
            first = direct.step(source, self.goal)
            second = residual.step(source, self.goal)
            self.assertEqual(first.action, second.action)
            self.assertEqual(first.features, second.features)
            self.assertEqual(first.hidden_before, second.base_hidden_before)
            self.assertEqual(second.latent, (0.,) * 5)
            self.assertIsNone(second.log_probability)
            direct.record_dispatch(first, first.action, admitted=True)
            residual.record_dispatch(second, second.action, admitted=True)

    def test_residual_buttons_can_initiate_and_withhold_saturated_bc_buttons(self):
        base = torch.zeros((1, 5))
        base_logits = torch.tensor([[1000., -1000.]])
        residual_logits = torch.tensor([[-1000., 1000.]])
        combined, logits = combine_residual(base, base_logits, torch.zeros_like(base), residual_logits)
        self.assertTrue(torch.equal(combined, base))
        self.assertEqual((logits >= 0).tolist(), [[False, True]])

    def test_raw_joint_likelihood_matches_independent_distributions(self):
        latent = torch.tensor([[.2, -.3, 1., -2., .8]], dtype=torch.float64)
        mean = torch.tensor([[.1, .2, .3, -.4, .5]], dtype=torch.float64)
        log_std = torch.tensor([-.2, -.5, -.1, -.7, -.3], dtype=torch.float64)
        buttons = torch.tensor([[1., 0.]], dtype=torch.float64)
        logits = torch.tensor([[1.7, -.3]], dtype=torch.float64)
        expected = (torch.distributions.Normal(mean, log_std.exp()).log_prob(latent).sum(-1)
                    + torch.distributions.Bernoulli(logits=logits).log_prob(buttons).sum(-1))
        torch.testing.assert_close(joint_log_probability(latent, mean, log_std, buttons, logits), expected)
        alternative = latent.clone()
        alternative[:, -1] += .5
        self.assertNotEqual(float(expected), float(joint_log_probability(alternative, mean, log_std, buttons, logits)))

    def test_deterministic_requests_have_no_sampled_likelihood_or_rng_use(self):
        generator = torch.Generator().manual_seed(91)
        before = generator.get_state().clone()
        decision = JoltController(JoltPolicy("residual"), base_policy=JoltPolicy("bc"), generator=generator).step(observation(), self.goal)
        self.assertEqual(decision.sampling_mode, "deterministic_mean")
        self.assertIsNone(decision.log_probability)
        self.assertTrue(torch.equal(before, generator.get_state()))
        drawn = JoltController(JoltPolicy("residual"), base_policy=JoltPolicy("bc"), generator=generator).step(observation(), self.goal, deterministic=False)
        self.assertEqual(drawn.sampling_mode, "stochastic_draw")
        self.assertEqual(len(drawn.latent), 5)
        self.assertEqual(len(drawn.buttons), 2)
        self.assertIsNotNone(drawn.log_probability)
        self.assertFalse(torch.equal(before, generator.get_state()))

    def test_padding_neither_resets_nor_advances_recurrence(self):
        policy = JoltPolicy("bc")
        hidden = torch.randn((2, 32))
        inputs = torch.zeros((2, len(FEATURE_NAMES)))
        inputs[1] = float("nan")
        output = policy(inputs, hidden, reset=torch.tensor([True, True]), valid=torch.tensor([True, False]))
        self.assertTrue(torch.equal(output[3][1], hidden[1]))
        self.assertTrue(torch.isfinite(output[3]).all())
        fresh = policy(inputs[:1], policy.zero_hidden())
        torch.testing.assert_close(output[3][:1], fresh[3])

    def test_recurrent_segment_cut_never_renews_original_time_or_discards_history(self):
        controller = JoltController(JoltPolicy("bc"))
        for tick in range(0, 126, 6):
            if tick == 120:
                prior = controller.history
                controller.reset_recurrence()
            decision = controller.step(observation(tick), self.goal)
            if tick == 120:
                self.assertEqual(decision.context.start_tick, 0)
                self.assertEqual(decision.context.deadline_tick, 240)
                self.assertEqual(decision.features[FEATURE_NAMES.index("time_remaining_fraction")], .5)
                self.assertEqual(decision.history, prior)
                self.assertEqual(decision.hidden_before, (0.,) * 32)
            controller.record_dispatch(decision, decision.action, admitted=True)

    def test_overrides_supply_no_proposal_likelihood_or_forward_work(self):
        costs = []
        controller = JoltController(JoltPolicy("bc"), reserve=lambda name, count: costs.append((name, count)))
        source = observation(rotation=1.5)
        decision = controller.step(source, self.goal, deterministic=False)
        self.assertEqual(emergency_request(FeedbackObservation.from_host(source)), "unstable_attitude")
        self.assertTrue(decision.action.auxiliary)
        self.assertIsNone(decision.proposal)
        self.assertIsNone(decision.latent)
        self.assertIsNone(decision.log_probability)
        self.assertEqual(costs, [])
        with self.assertRaisesRegex(ValueError, "relabelled"):
            controller.record_dispatch(decision, decision.action, admitted=True, intervention=False)
        history = controller.record_dispatch(decision, decision.action, admitted=True)
        self.assertTrue(history.intervention)
        self.assertIsNone(history.proposal)

    def test_dispatch_history_must_bind_exact_pending_decision(self):
        controller = JoltController(JoltPolicy("bc"))
        first = controller.step(observation(), self.goal)
        with self.assertRaisesRegex(ValueError, "record the actual"):
            controller.step(observation(6), self.goal)
        with self.assertRaisesRegex(ValueError, "exact pending"):
            controller.record_dispatch(replace(first), first.action, admitted=True)
        with self.assertRaisesRegex(ValueError, "marked as intervention"):
            controller.record_dispatch(first, WireAction(effort=0.), admitted=True)
        controller.record_dispatch(first, first.action, admitted=True)
        with self.assertRaisesRegex(ValueError, "exact pending"):
            controller.record_dispatch(first, first.action, admitted=True)

    def test_delayed_dispatch_respects_original_lease_and_deadline(self):
        controller = JoltController(JoltPolicy("bc"))
        decision = controller.step(observation(), self.goal, current_tick=9)
        self.assertEqual(decision.repeat_ticks, 3)
        self.assertEqual(decision.supervision.action_lease_ticks, 12)
        controller = JoltController(JoltPolicy("bc"))
        tiny = replace(self.goal, timeout_seconds=.1)
        decision = controller.step(observation(), tiny, current_tick=5)
        self.assertEqual(decision.repeat_ticks, 1)
        self.assertEqual(decision.context.deadline_tick, 6)
        for delay in (6, 12):
            with self.assertRaisesRegex(ValueError, "expired"):
                JoltController(JoltPolicy("bc")).step(observation(), tiny, current_tick=delay)

    def test_retired_goal_lifetime_and_changed_owner_cannot_be_resurrected(self):
        controller = JoltController(JoltPolicy("bc"))
        first = controller.step(observation(), self.goal)
        controller.record_dispatch(first, first.action, admitted=True)
        with self.assertRaisesRegex(ValueError, "authority"):
            controller.step(observation(6, control_generation=1), self.goal)
        with self.assertRaisesRegex(ValueError, "retired"):
            controller.step(observation(12), self.goal)
        with self.assertRaisesRegex(ValueError, "retired"):
            controller.start(observation(12), self.goal)
        controller.start(observation(run_id="new-lifetime"), self.goal)
        with self.assertRaisesRegex(ValueError, "retired"):
            controller.start(observation(), replace(self.goal, id="another"))

    def test_goal_parameters_and_topology_are_bound_to_original_context(self):
        for source, goal in ((observation(6), replace(self.goal, target=(40., 0.))),
                             (observation(6, parts=[]), self.goal)):
            controller = JoltController(JoltPolicy("bc"))
            first = controller.step(observation(), self.goal)
            controller.record_dispatch(first, first.action, admitted=True)
            with self.assertRaisesRegex(ValueError, "authority or topology"):
                controller.step(source, goal)

    def test_first_step_cannot_delay_an_explicit_start_to_renew_supervisor_time(self):
        controller = JoltController(JoltPolicy("bc"))
        controller.start(observation(), self.goal)
        with self.assertRaisesRegex(ValueError, "original observation"):
            controller.step(observation(6), self.goal)

    def test_missing_decision_observation_cannot_bridge_recurrent_history(self):
        for gap in (12, 18, 60):
            controller = JoltController(JoltPolicy("bc"))
            first = controller.step(observation(), self.goal)
            controller.record_dispatch(first, first.action, admitted=True)
            with self.subTest(gap=gap), self.assertRaisesRegex(ValueError, "missing observation"):
                controller.step(observation(gap), self.goal)
            with self.assertRaisesRegex(ValueError, "retired"):
                controller.step(observation(6), self.goal)

    def test_runtime_snapshots_and_detects_changed_model_content(self):
        model = JoltPolicy("bc")
        controller = JoltController(model)
        frozen = controller.policy_sha256
        with torch.no_grad():
            model.actor.bias.add_(1.)
        self.assertEqual(policy_state_sha256(controller.policy), frozen)
        with torch.no_grad():
            controller.policy.actor.bias.add_(1.)
        with self.assertRaisesRegex(ValueError, "changed after admission"):
            controller.step(observation(), self.goal)

    def test_forward_costs_are_reserved_before_both_residual_models(self):
        calls = []
        controller = JoltController(JoltPolicy("residual"), base_policy=JoltPolicy("bc"),
                                    reserve=lambda name, count: calls.append((name, count)))
        first = controller.step(observation(), self.goal)
        self.assertEqual(calls, [("inference_forward", 1), ("inference_forward", 1)])
        controller.record_dispatch(first, first.action, admitted=True)
        copied = controller.clone_state()
        self.assertIs(copied.reserve, controller.reserve)
        copied.step(observation(6), self.goal)
        self.assertEqual(len(calls), 4)

    def test_new_checkpoints_reject_old_contract_and_changed_base_source_or_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bc, residual = JoltPolicy("bc"), JoltPolicy("residual")
            metadata = {"config_sha256": "a" * 64, "policy_seed": 190103}
            saved = save_policy(root / "new.pt", residual, base_policy=bc, metadata=metadata)
            policy, base, loaded = load_policy(root / "new.pt", expected_config_sha256="a" * 64,
                                               expected_base_sha256=policy_state_sha256(bc))
            self.assertEqual(policy_state_sha256(policy), saved["policy_sha256"])
            self.assertEqual(policy_state_sha256(base), saved["base_sha256"])
            self.assertEqual(loaded, metadata)
            self.assertFalse(any(p.requires_grad for p in base.parameters()))
            with self.assertRaises(ValueError):
                load_old_policy(root / "new.pt")
            with self.assertRaisesRegex(ValueError, "another frozen"):
                load_policy(root / "new.pt", expected_config_sha256="b" * 64)
            with self.assertRaisesRegex(ValueError, "replaced its frozen"):
                load_policy(root / "new.pt", expected_base_sha256="b" * 64)
            payload = torch.load(root / "new.pt", weights_only=True)
            for name, mutate in (
                ("old", lambda value: value.update(format="nca-first-skill/v1")),
                ("source", lambda value: value.update(source_sha256="b" * 64)),
                ("mapping", lambda value: value["specification"].update(action_schema="changed")),
                ("base", lambda value: value["base"]["state_dict"]["actor.bias"].add_(1.)),
            ):
                import copy
                changed = copy.deepcopy(payload)
                mutate(changed)
                torch.save(changed, root / f"{name}.pt")
                with self.assertRaises(ValueError):
                    load_policy(root / f"{name}.pt")
            with self.assertRaises(FileExistsError):
                save_policy(root / "new.pt", residual, base_policy=bc)
            self.assertEqual(policy.specification()["action_schema"], ACTION_SCHEMA)
            self.assertEqual(policy.specification()["sample_schema"], SAMPLE_SCHEMA)


if __name__ == "__main__":
    unittest.main()
