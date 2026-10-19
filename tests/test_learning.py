"""Tiny optional CPU conformance; these tests do not qualify gameplay."""

from dataclasses import replace
import hashlib
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from nca.contracts import ContractError, WireAction
    from nca.controllers.baseline import SkillGoal
    from nca.controllers.learned import (FEATURE_NAMES, RESIDUAL_SCALES, LearnedController, RecurrentPolicy,
                                        action_vector, combine_residual, features, load_policy,
                                        policy_state_sha256, residual_log_probability)
    from nca.data import ActionSource, CaptureEvent, PolicySample, TickRecord, policy_view
    from nca.train.artifacts import (export_weights, freeze_manifest, immutable_json, load_training_checkpoint,
                                    read_config, write_checkpoint)
    from nca.train.recurrent import (RolloutSequence, Transition, attributed_actor_sample,
                                    generalized_advantages, ordered_window, ppo_update, train_clone)
    from nca.train.screen import policy_seed_interval, recorded_policy_sample
    from nca.train.task import HopTask


def setUpModule():
    if TORCH_AVAILABLE:
        torch.set_num_threads(1)


def view(tick=0, **extra):
    value = {"version": 2, "representation": "articulated-rapier", "run_id": "synthetic-learning",
             "epoch": 1, "actor_id": 0, "observation_id": f"test:1:0:0:{tick}",
             "profile_version": 1, "model": "jev", "rig_id": "jev", "tick": tick,
             "position": [0., 0.], "velocity": [0., 0.], "rotation": 0., "angular_velocity": 0.,
             "grounded": True, "cooldown": 0., "energy": 1., "ground_distance": 0.,
             "ceiling_distance": 96., "control_generation": 0,
             "parts": [{"id": "core", "index": 0, "detached": False}]}
    value.update(extra)
    return value


def attributed_event(**overrides):
    action = WireAction(move=(.4, 0)).to_dict()
    packet = {"version": 2, "kind": "physical_wire_action", "run_id": "synthetic-learning", "epoch": 1,
              "actor_id": 0, "sequence": 0, "observed_tick": 0, "observation_id": view()["observation_id"],
              "lease_ticks": 12, "action": action}
    source = ActionSource("learner", "learned", "test-policy", "a" * 64)
    sample = PolicySample("latent_residual", {"latent": [.3, -.2, .1], "log_probability_space": "bounded_residual",
                                             "sampling_mode": "stochastic_draw"},
                          "a" * 64, -.3, "scaled-tanh-diagonal-normal/v1")
    tick = TickRecord(0, 1, 1, 10, 11, (packet,), ({"accepted": True, "reason": ""},), packet,
                      "learner", action, {"linear_impulse": [.1, 0.]},
                      {"available": False, "reason": "not exposed"}, {"velocity": [1., 0.]}, view(1), 0)
    event = CaptureEvent("step", 0, 0, 0, 1, 0, 20, source, view(), policy_view(view()),
                         requested_packet=packet, sample=sample, ticks=(tick,), next_observation=view(1),
                         source_observation=view(), reward=0)
    return replace(event, **overrides)


@unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning extra is not installed")
class LearnedBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.goal = SkillGoal("hop", "directed_hop", (48., 0.), timeout_seconds=4)

    def test_features_ignore_privileged_fields_and_preserve_missing_sensor_masks(self):
        expected = features(view(), self.goal)
        polluted = view(evaluator={"hidden_target": [1000, 400]}, reward=1e20, success=True,
                        actors=[{"position": [-10, 3]}], global_materials=list(range(20)))
        self.assertEqual(features(polluted, self.goal), expected)
        missing = features(view(energy=None, grounded=None), self.goal)
        zero = features(view(energy=0., grounded=False), self.goal)
        self.assertEqual(missing[FEATURE_NAMES.index("energy")], zero[FEATURE_NAMES.index("energy")])
        self.assertNotEqual(missing[FEATURE_NAMES.index("energy_valid")], zero[FEATURE_NAMES.index("energy_valid")])
        self.assertNotEqual(missing[FEATURE_NAMES.index("grounded_valid")], zero[FEATURE_NAMES.index("grounded_valid")])
        self.assertEqual(len(expected), 26)

    def test_deadline_and_previous_wire_history_are_explicit_features(self):
        before = features(view(), self.goal)
        later = features(view(120), self.goal, previous=WireAction(primary=True))
        self.assertEqual(before[FEATURE_NAMES.index("time_remaining_fraction")], 1.)
        self.assertEqual(later[FEATURE_NAMES.index("time_remaining_fraction")], .5)
        self.assertEqual(later[FEATURE_NAMES.index("previous_primary")], 1.)

    def test_transformed_density_matches_independent_torch_distribution(self):
        latent = torch.tensor([[.3, -.6, 1.2], [-1.5, 2.1, .7]], dtype=torch.float64)
        mean = torch.tensor([[.1, -.2, .3]], dtype=torch.float64)
        log_std = torch.tensor([-.3, -.4, -.5], dtype=torch.float64)
        base = torch.distributions.Normal(mean, log_std.exp())
        transformed = torch.distributions.TransformedDistribution(base, [
            torch.distributions.transforms.TanhTransform(),
            torch.distributions.transforms.AffineTransform(0., torch.tensor(RESIDUAL_SCALES, dtype=torch.float64))])
        sample = latent.tanh() * torch.tensor(RESIDUAL_SCALES, dtype=torch.float64)
        expected = transformed.log_prob(sample).sum(-1)
        torch.testing.assert_close(residual_log_probability(latent, mean, log_std), expected)

    def test_saturated_latent_has_finite_density_and_gradient(self):
        latent = torch.tensor([[20., -20., 0.]])
        mean = torch.zeros((1, 3), requires_grad=True)
        score = residual_log_probability(latent, mean, torch.zeros(3))
        self.assertTrue(torch.isfinite(score).all())
        score.sum().backward()
        self.assertTrue(torch.isfinite(mean.grad).all())

    def test_clipped_combined_wire_never_relabels_residual_likelihood(self):
        baseline = WireAction(move=(1., 0), trim=.99, effort=1., primary=True)
        first = combine_residual(baseline, (.1, .05, .1))
        second = combine_residual(baseline, (.2, .1, .12))
        self.assertEqual(first, second)
        latent1 = torch.atanh(torch.tensor([[.1, .05, .1]]) / torch.tensor(RESIDUAL_SCALES))
        latent2 = torch.atanh(torch.tensor([[.2, .1, .12]]) / torch.tensor(RESIDUAL_SCALES))
        self.assertNotEqual(float(residual_log_probability(latent1, torch.zeros(1, 3), torch.zeros(3))),
                            float(residual_log_probability(latent2, torch.zeros(1, 3), torch.zeros(3))))
        self.assertTrue(first.primary)
        self.assertEqual(first.aim, baseline.aim)

    def test_deterministic_residual_preserves_mean_wire_but_has_no_stochastic_likelihood(self):
        model = RecurrentPolicy("residual")
        generator = torch.Generator().manual_seed(73)
        before = generator.get_state().clone()
        mean = LearnedController(model, generator=generator).step(view(), self.goal, deterministic=True)
        self.assertEqual(mean.sampling_mode, "deterministic_mean")
        self.assertIsNone(mean.log_probability)
        self.assertTrue(torch.equal(before, generator.get_state()))
        with torch.no_grad():
            raw, _, _ = model(torch.tensor([mean.features]), model.zero_hidden())
        expected = (raw.tanh() * torch.tensor(RESIDUAL_SCALES))[0].tolist()
        self.assertEqual(mean.action, combine_residual(mean.baseline, expected))
        captured_mean = recorded_policy_sample(mean, "a" * 64)
        self.assertEqual(captured_mean.distribution, "deterministic-scaled-tanh-mean/v1")
        self.assertIsNone(captured_mean.log_probability)
        self.assertFalse(attributed_event(sample=captured_mean).on_policy_eligible)
        with self.assertRaises(ContractError):
            attributed_actor_sample(attributed_event(sample=captured_mean))
        with self.assertRaises(ValueError):
            recorded_policy_sample(replace(mean, log_probability=-1.), "a" * 64)
        drawn = LearnedController(model, generator=generator).step(view(), self.goal, deterministic=False)
        self.assertEqual(drawn.sampling_mode, "stochastic_draw")
        self.assertIsNotNone(drawn.log_probability)
        self.assertFalse(torch.equal(before, generator.get_state()))
        captured_draw = recorded_policy_sample(drawn, "a" * 64)
        self.assertTrue(attributed_event(sample=captured_draw).on_policy_eligible)
        self.assertEqual(attributed_actor_sample(attributed_event(sample=captured_draw)),
                         (drawn.latent, drawn.log_probability))

    def test_recurrent_reset_matches_fresh_inference_and_rejects_repeated_ticks(self):
        model = RecurrentPolicy("bc")
        controller = LearnedController(model)
        initial = controller.step(view(), self.goal)
        second = controller.step(view(6), self.goal)
        self.assertNotEqual(initial.hidden_before, second.hidden_before)
        with self.assertRaises(ValueError):
            controller.step(view(6), self.goal)
        fresh = controller.step(view(run_id="replacement"), self.goal)
        self.assertEqual(initial.action, fresh.action)
        self.assertEqual(initial.hidden_before, fresh.hidden_before)

    def test_explicit_gru_reset_mask_discards_prior_sequence(self):
        policy = RecurrentPolicy("residual")
        inputs = torch.ones((1, 32))
        arbitrary = torch.rand((1, 32))
        a = policy(inputs, arbitrary, torch.tensor([True]))
        b = policy(inputs, policy.zero_hidden())
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second)

    def test_canceled_goal_and_control_takeover_cannot_resume_old_invocation(self):
        controller = LearnedController(RecurrentPolicy("bc"))
        controller.step(view(), self.goal)
        controller.cancel()
        with self.assertRaises(ValueError):
            controller.step(view(6), self.goal)
        other = LearnedController(controller.policy)
        other.step(view(), self.goal)
        with self.assertRaises(ValueError):
            other.step(view(6, control_generation=1), self.goal)
        restarted = other.step(view(12, control_generation=1), replace(self.goal, id="fresh-hop"))
        self.assertEqual(restarted.hidden_before, (0.,) * 32)

    def test_same_goal_id_cannot_change_parameters(self):
        controller = LearnedController(RecurrentPolicy("bc"))
        controller.step(view(), self.goal)
        with self.assertRaises(ValueError):
            controller.step(view(6), replace(self.goal, target=(20., 0.)))

    def test_part_mount_boundary_fences_recurrence_and_old_goal(self):
        controller = LearnedController(RecurrentPolicy("bc"))
        controller.step(view(), self.goal)
        with self.assertRaises(ValueError):
            controller.step(view(6, parts=[{"id": "core", "index": 0, "detached": True}]), self.goal)

    def test_correction_replacement_mixed_and_ambiguous_scores_never_enter_actor_loss(self):
        self.assertEqual(attributed_actor_sample(attributed_event()), ((.3, -.2, .1), -.3))
        examples = [attributed_event(intervention="human_correction"), attributed_event(replaced=True),
                    attributed_event(discontinuities=("source_takeover",)),
                    attributed_event(ticks=(replace(attributed_event().ticks[0], source_id="other"),))]
        for event in examples:
            with self.subTest(event=event.intervention), self.assertRaises(ContractError):
                attributed_actor_sample(event)
        ambiguous = attributed_event(sample=replace(attributed_event().sample, values={"latent": [.1, .2, .3]}))
        with self.assertRaises(ValueError):
            attributed_actor_sample(ambiguous)
        for mode in (None, "deterministic_mean", "unknown"):
            original = attributed_event().sample
            mislabeled = replace(original, values={**original.values, "sampling_mode": mode})
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                attributed_actor_sample(attributed_event(sample=mislabeled))

    def test_complete_prefix_burn_in_matches_sequential_history_with_padding(self):
        model = RecurrentPolicy("residual")
        rows = [torch.randn(32).tolist() for _ in range(11)]
        hidden = model.zero_hidden()
        expected = []
        for row in rows:
            output, _, hidden = model(torch.tensor([row]), hidden)
            expected.append(output[0])
        actual, _, padding = ordered_window(model, rows, 8, 6)
        torch.testing.assert_close(actual[:3], torch.stack(expected[8:]))
        self.assertEqual(padding.tolist(), [True, True, True, False, False, False])
        self.assertFalse(actual[3:].any())

    def test_variable_duration_gae_and_terminal_mask_have_known_value(self):
        rows = (Transition((0.,) * 32, (.1, .2, .3), -.5, 1., .5, 6, True, episode_id="a"),
                Transition((0.,) * 32, (.1, .2, .3), -.4, 2., .75, 3, True,
                           task_terminal=True, episode_id="a", event_index=1))
        advantages, values = generalized_advantages(RolloutSequence(rows), gamma_per_tick=.9, lambda_per_tick=.8)
        final = .75 - 2.
        first = .5 + .9**6 * 2 - 1 + .9**6 * .8**6 * final
        torch.testing.assert_close(advantages, torch.tensor([first, final]))
        torch.testing.assert_close(values, torch.tensor([first + 1, .75]))

    def test_administrative_cutoff_bootstraps_only_its_valid_final_state(self):
        row = Transition((0.,) * 32, None, None, 1., .5, 3, False, administrative_cutoff=True,
                         episode_id="a")
        advantage, _ = generalized_advantages(RolloutSequence((row,), 4.), gamma_per_tick=.9)
        self.assertAlmostEqual(float(advantage[0]), .5 + .9**3 * 4 - 1, places=6)
        invalid = replace(row, final_observation_valid=False)
        with self.assertRaises(ValueError):
            RolloutSequence((invalid,), 4.)
        terminal = replace(row, administrative_cutoff=False, task_terminal=True)
        advantage, _ = generalized_advantages(RolloutSequence((terminal,)), gamma_per_tick=.9)
        self.assertEqual(float(advantage[0]), -.5)

    def test_sequence_cannot_reorder_or_cross_terminal_episode(self):
        row = Transition((0.,) * 32, None, None, 0., 0., 6, False, episode_id="a")
        with self.assertRaises(ValueError):
            RolloutSequence((row, replace(row, episode_id="b", event_index=1)))
        with self.assertRaises(ValueError):
            RolloutSequence((replace(row, task_terminal=True), replace(row, event_index=1)))
        with self.assertRaises(ValueError):
            RolloutSequence((replace(row, event_index=1),))

    def test_invalid_interior_final_observation_breaks_both_gae_paths(self):
        first = Transition((0.,) * 32, None, None, 0., 0., 6, False,
                           final_observation_valid=False, episode_id="a")
        following = replace(first, reward=1., final_observation_valid=True, event_index=1, task_terminal=True)
        advantages, _ = generalized_advantages(RolloutSequence((first, following)))
        self.assertEqual(float(advantages[0]), 0.)
        self.assertEqual(float(advantages[1]), 1.)

    def test_unmodified_policy_has_unit_ppo_ratio_under_reconstructed_history(self):
        model = RecurrentPolicy("residual")
        hidden, transitions = model.zero_hidden(), []
        for index in range(5):
            inputs = torch.randn((1, 32))
            with torch.no_grad():
                mean, value, hidden = model(inputs, hidden)
                latent = mean + .3 * torch.randn(1, 3)
                score = residual_log_probability(latent, mean, model.log_std)
            transitions.append(Transition(tuple(inputs[0].tolist()), tuple(latent[0].tolist()), float(score[0]),
                                          float(value[0]), .2, 6, True, task_terminal=index == 4,
                                          episode_id="a", event_index=index))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
        result = ppo_update(model, optimizer, [RolloutSequence(tuple(transitions))], seed=2, epochs=1, unroll=3)
        self.assertAlmostEqual(result["mean_ratio"], 1.0, places=6)
        self.assertEqual(result["clip_fraction"], 0.)

    def test_all_replaced_rollout_cannot_update_weights(self):
        model = RecurrentPolicy("residual")
        before = policy_state_sha256(model)
        row = Transition((0.,) * 32, None, None, 0., 10., 6, False, task_terminal=True, episode_id="a")
        result = ppo_update(model, torch.optim.Adam(model.parameters()), [RolloutSequence((row,))], seed=1)
        self.assertEqual(result["updates"], 0)
        self.assertEqual(policy_state_sha256(model), before)

    def test_tiny_clone_optimization_reduces_a_nontrivial_demonstration_loss(self):
        policy = RecurrentPolicy("bc")
        observations = [[(-1.) ** index / 2] + [0.] * 25 for index in range(8)]
        targets = [[row[0], -.2, .8, float(index % 2), 0., 0.] for index, row in enumerate(observations)]
        result = train_clone(policy, [(observations, targets)], seed=12, epochs=8, unroll=8, learning_rate=.02)
        self.assertLess(result["epoch_losses"][-1], result["epoch_losses"][0])

    def test_checkpoint_is_immutable_and_sequential_load_outputs_match(self):
        model = RecurrentPolicy("bc")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pt"
            write_checkpoint(path, model, {"config_sha256": "a" * 64})
            loaded, _ = load_policy(path, expected_config_sha256="a" * 64)
            left, right = LearnedController(model), LearnedController(loaded)
            for tick in (0, 6, 12, 18):
                self.assertEqual(left.step(view(tick), self.goal), right.step(view(tick), self.goal))
            with self.assertRaises(ValueError):
                write_checkpoint(path, model, {})
            with self.assertRaises(ValueError):
                load_policy(path, expected_config_sha256="b" * 64)

    def test_resume_recovers_rng_and_optimizer_only_at_fresh_episode_boundary(self):
        model = RecurrentPolicy("residual")
        optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
        generator = torch.Generator().manual_seed(42)
        torch.randn((7,), generator=generator)
        baseline_sha = hashlib.sha256((Path(__file__).parents[1] / "src/nca/controllers/baseline.py").read_bytes()).hexdigest()
        metadata = {"config_sha256": "a" * 64, "learner_sources_sha256": "b" * 64,
                    "baseline_source_sha256": baseline_sha}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundary.pt"
            write_checkpoint(path, model, metadata, optimizer=optimizer,
                             training_state={"next_episode_boundary": True, "torch_generator_state": generator.get_state()})
            restored, _, state, _ = load_training_checkpoint(path, expected_config_sha256="a" * 64,
                                                            expected_sources_sha256="b" * 64)
            resumed_generator = torch.Generator().set_state(state["torch_generator_state"])
            torch.testing.assert_close(torch.randn((6,), generator=generator), torch.randn((6,), generator=resumed_generator))
            self.assertEqual(policy_state_sha256(model), policy_state_sha256(restored))
            self.assertFalse(LearnedController(restored).hidden.any())

    def test_frozen_scenarios_are_family_disjoint_and_budget_math_fits(self):
        config, digest = read_config(Path(__file__).parents[1] / "configs/first-skill/jolt-screen-v1.json")
        families = [scenario["id"] for key in ("training", "tuning", "evaluation") for scenario in config[f"{key}_families"]]
        self.assertEqual(len(set(families)), len(families))
        b = config["budget"]
        self.assertEqual(sum(b[key] for key in ("demo_maximum_decisions", "ppo_maximum_decisions",
                                               "tuning_maximum_decisions", "evaluation_maximum_decisions")), 12864)
        self.assertLessEqual(12864, b["maximum_total_decisions"])
        self.assertEqual(len(digest), 64)

    def test_preregistration_is_written_before_any_host_and_rejects_missing_dependencies(self):
        config, digest = read_config(Path(__file__).parents[1] / "configs/first-skill/jolt-screen-v1.json")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                freeze_manifest(directory, config, {})
            manifest = freeze_manifest(directory, config, {"t102": "a" * 40, "t103": "b" * 40, "t104": "c" * 40})
            self.assertEqual(manifest["config_sha256"], digest)
            self.assertFalse(manifest["evaluation_opened"])
            self.assertEqual(manifest["phase"], "pre_registered_before_any_rollout")
            self.assertEqual(type(manifest["runtime"]["torch"]), str)
            with self.assertRaises(ValueError):
                freeze_manifest(directory, config, {"t102": "a" * 40, "t103": "b" * 40, "t104": "c" * 40})

    def test_uncertainty_resamples_policies_not_individual_transitions(self):
        interval = policy_seed_interval([-.25, 0., .25])
        self.assertEqual(interval["mean"], 0.)
        self.assertLess(interval["lower95"], 0.)
        self.assertGreater(interval["upper95"], 0.)
        with self.assertRaises(ValueError):
            policy_seed_interval([0.] * 1000)

    def test_reward_returning_to_previous_closest_distance_cannot_farm_progress(self):
        task = HopTask(self.goal, view())
        action = WireAction(move=(.5, 0))
        def receipt(tick, x):
            return {"tick_before": tick - 1, "observation": view(tick, position=[x, 0.]), "admission": []}
        first = task.consume(action, [receipt(1, 10)])
        retreat = task.consume(action, [receipt(2, 0)])
        return_again = task.consume(action, [receipt(3, 10)])
        self.assertGreater(first["progress"], 0)
        self.assertEqual(retreat["progress"], 0)
        self.assertEqual(return_again["progress"], 0)
        self.assertLess(retreat["total"] + return_again["total"], 0)

    def test_success_requires_actual_supported_quiet_dwell_and_scores_once(self):
        task = HopTask(self.goal, view())
        action = WireAction(effort=0)
        receipts = [{"tick_before": tick - 1, "observation": view(tick, position=[48., 0.]), "admission": []}
                    for tick in range(1, 13)]
        early = task.consume(action, receipts[:6])
        self.assertFalse(task.succeeded)
        final = task.consume(action, receipts[6:])
        self.assertEqual(early["success"], 0)
        self.assertEqual(final["success"], 2.)
        self.assertEqual(task.success_tick, 12)
        with self.assertRaises(ValueError):
            task.consume(action, receipts[-1:])
        airborne = HopTask(self.goal, view())
        airborne.consume(action, [{**row, "observation": {**row["observation"], "grounded": False}} for row in receipts])
        self.assertFalse(airborne.succeeded)

    def test_requested_pulse_during_passive_fall_cannot_prove_actual_launch(self):
        action = WireAction(primary=True)
        falling = HopTask(self.goal, view(grounded=False, cooldown=.5))
        falling.consume(action, [{"tick_before": 0, "observation": view(1, grounded=False, cooldown=.48),
                                  "admission": [], "applied": {"action": action.to_dict(), "action_available": True,
                                                               "linear_impulse": [0., 0.], "energy_used": 0.}}])
        self.assertEqual(falling.primary_edges, 1)
        self.assertFalse(falling.purposeful_launch)
        launched = HopTask(self.goal, view())
        launched.consume(action, [
            {"tick_before": 0, "observation": view(1, cooldown=.62), "admission": [],
             "applied": {"action": action.to_dict(), "action_available": True,
                         "linear_impulse": [10., -50.], "energy_used": .1}},
            {"tick_before": 1, "observation": view(2, cooldown=.603, grounded=False), "admission": [],
             "applied": {"action": action.to_dict(), "action_available": True,
                         "linear_impulse": [0., 0.], "energy_used": 0.}}])
        self.assertTrue(launched.purposeful_launch)

    def test_administrative_budget_and_intrinsic_deadline_have_distinct_metrics(self):
        task = HopTask(self.goal, view(), deadline_ticks=6)
        receipts = [{"tick_before": tick - 1, "observation": view(tick), "admission": []} for tick in range(1, 7)]
        terms = task.consume(WireAction(effort=0), receipts)
        self.assertEqual(terms["deadline"], -.5)
        self.assertEqual(task.metrics()["reason"], "task_deadline")
        capped = HopTask(self.goal, view())
        capped.consume(WireAction(effort=0), receipts)
        self.assertEqual(capped.metrics(administratively_capped=True)["reason"], "administrative_cutoff")
        self.assertFalse(capped.metrics(administratively_capped=True)["success"])


if __name__ == "__main__":
    unittest.main()
