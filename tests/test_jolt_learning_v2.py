"""Synthetic contract tests; no physical rollout or gameplay qualification."""

from collections import Counter
from dataclasses import replace
import importlib.util
import math
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from nca.contracts import ContractError, WireAction, sha256_json
    from nca.controllers.baseline import SkillGoal
    from nca.controllers.jolt_learned import (
        ActionHistory, FEATURE_NAMES, InvocationContext, JoltController, JoltPolicy,
        features, policy_state_sha256,
    )
    from nca.data import ActionSource, CaptureEvent, PolicyContext, TickRecord, policy_view
    from nca.data.jolt_samples import (
        JoltDecisionRecord, JoltDraw, JoltEpisode, draw_from_decision,
        reconstructed_history, validate_history,
    )
    from nca.train.jolt_learning import (
        LearningRow, authenticate_rollouts, clone_loss, clone_sequences,
        generalized_advantages, ordered_window, ppo_update, record_features,
        rollout_sequence, train_clone,
    )


def setUpModule():
    if TORCH_AVAILABLE:
        torch.set_num_threads(1)


def view(tick=0, **extra):
    result = {"version": 2, "representation": "articulated-rapier", "run_id": "synthetic-jolt-v2",
              "epoch": 1, "actor_id": 0, "observation_id": f"test:1:0:{tick}",
              "profile_version": 1, "model": "jev", "rig_id": "jev", "tick": tick,
              "position": [0., 0.], "velocity": [0., 0.], "rotation": 0., "angular_velocity": 0.,
              "grounded": True, "cooldown": 0., "energy": 1., "ground_distance": 0.,
              "ceiling_distance": 96., "control_generation": 0,
              "parts": [{"id": "core", "index": 0, "detached": False}]}
    result.update(extra)
    return result


def record(index, proposal, *, context, history, policy_sha="a" * 64, source_kind="scripted",
           draw=None, overridden=False, actual=6, reward=.1, terminal=False, cutoff=False):
    start = index * 6
    action = WireAction(effort=0., auxiliary=True) if overridden else proposal
    if action is None:
        action = WireAction(effort=0.)
    packet = {"version": 2, "kind": "physical_wire_action", "run_id": "synthetic-jolt-v2", "epoch": 1,
              "actor_id": 0, "sequence": index, "observed_tick": start,
              "observation_id": view(start)["observation_id"], "lease_ticks": 12, "action": action.to_dict()}
    source = ActionSource("source", source_kind, "synthetic-policy", policy_sha)
    ticks = tuple(TickRecord(tick, tick + 1, tick + 1, 10, 11,
        (packet,) if tick == start else (), ({"accepted": True, "reason": ""},) if tick == start else (),
        packet, "source", action.to_dict(), {"linear_impulse": [0., 0.], "action_available": True, "action": action.to_dict()},
        {"available": False, "reason": "not exposed"}, {"velocity": [0., 0.]}, view(tick + 1), 0)
        for tick in range(start, start + actual))
    goal = context.goal
    caller = PolicyContext(goal.id, goal.skill, goal.target, goal.pace, goal.tolerance, goal.timeout_seconds)
    event = CaptureEvent("step", index, 0, start, start + actual, 0, 20, source,
        view(start), policy_view(view(start)), requested_packet=packet, context=caller, ticks=ticks,
        next_observation=view(start + actual), reward=reward, terminated=terminal, truncated=cutoff,
        replaced=overridden, intervention="model_override" if overridden else "none",
        source_observation=view(start), valid=actual > 0, trace_complete=actual > 0)
    return JoltDecisionRecord(event, context.to_dict(), history, proposal, "b" * 64,
                              "observed emergency" if overridden else "", overridden, draw,
                              recurrent_reset=index == 0)


def teacher_episode(*, overridden_at=None):
    goal = SkillGoal("hop", "directed_hop", (48., 0.), timeout_seconds=4.)
    context = InvocationContext.create(view(), goal)
    rows = []
    for index in range(3):
        override = index == overridden_at
        proposal = None if override else WireAction(move=(.2, 0.), aim=(.6, -.8), primary=index == 1)
        rows.append(record(index, proposal, context=context,
                           history=reconstructed_history(rows[-1] if rows else None), overridden=override))
    return JoltEpisode("unsuccessful-teacher", "training", tuple(rows), "failed", "synthetic negative case")


def learned_episode(policy, base, *, length=3, deterministic=False, terminal=False, cutoff=False):
    goal = SkillGoal("hop", "directed_hop", (48., 0.), timeout_seconds=4.)
    runtime = JoltController(policy, base_policy=base, generator=torch.Generator().manual_seed(191))
    rows, values = [], []
    for index in range(length):
        decision = runtime.step(view(index * 6), goal, deterministic=deterministic)
        draw = draw_from_decision(decision, source_sha256="c" * 64)
        rows.append(record(index, decision.proposal, context=decision.context,
            history=decision.history.to_dict(), policy_sha=decision.policy_sha256, source_kind="learned", draw=draw,
            terminal=terminal and index + 1 == length, cutoff=cutoff and index + 1 == length))
        values.append(decision.value)
        runtime.record_dispatch(decision, decision.action, admitted=True)
    return JoltEpisode("residual-rollout", "training", tuple(rows), "capped" if cutoff else "completed"), values


@unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning extra is not installed")
class JoltSampleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(811)

    def test_versioned_records_roundtrip_and_keep_full_native_parts(self):
        episode = teacher_episode()
        restored = JoltEpisode.from_json(episode.to_json())
        self.assertEqual(restored.sha256, episode.sha256)
        self.assertEqual(restored.decisions[0].event.source_observation["parts"][0]["id"], "core")
        self.assertEqual(len(record_features(restored.decisions[0])), len(FEATURE_NAMES))
        with self.assertRaises(ContractError):
            JoltEpisode.from_dict({**episode.to_dict(), "schema": "jolt-invocation-episode/v1"})

    def test_conversion_preserves_failure_abstention_and_original_time(self):
        episode = teacher_episode(overridden_at=1)
        converted = clone_sequences([episode])[0]
        self.assertEqual(converted.status, "failed")
        self.assertEqual(len(converted.rows), 3)
        self.assertEqual([row.eligible for row in converted.rows], [True, False, True])
        self.assertIsNone(converted.rows[1].label)
        self.assertFalse(converted.rows[1].advance)
        remaining = FEATURE_NAMES.index("time_remaining_fraction")
        self.assertEqual(converted.rows[2].features[remaining], .95)
        self.assertTrue(episode.decisions[2].history["dispatched"]["auxiliary"])
        self.assertIsNone(episode.decisions[2].history["proposal"])

    def test_intervention_segment_cannot_renew_invocation_deadline(self):
        episode = teacher_episode()
        row = episode.decisions[-1]
        context = InvocationContext.from_dict(row.invocation)
        renewed = replace(context, start_tick=12, deadline_tick=252)
        with self.assertRaises(ContractError):
            replace(episode, decisions=(*episode.decisions[:-1], replace(row, invocation=renewed.to_dict(), recurrent_reset=True)))

    def test_converter_rejects_any_holdout_before_training(self):
        for split in ("open", "sealed"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                clone_sequences([replace(teacher_episode(), split=split)])

    def test_supervisor_replacement_cannot_be_imitation_or_policy_likelihood(self):
        episode = teacher_episode(overridden_at=1)
        row = episode.decisions[1]
        self.assertFalse(row.imitation_eligible)
        self.assertFalse(row.actor_eligible)
        self.assertIsNone(row.proposal)
        self.assertTrue(row.admitted)
        with self.assertRaises(ContractError):
            replace(row, overridden=False)

    def test_corrupt_previous_dispatch_or_proposal_is_rejected(self):
        episode = teacher_episode()
        row = episode.decisions[1]
        history = ActionHistory(WireAction(), WireAction(), False, True).to_dict()
        corrupted = replace(episode, decisions=(episode.decisions[0], replace(row, history=history), episode.decisions[2]))
        with self.assertRaises(ContractError):
            validate_history(corrupted)

    def test_native_mixed_or_refused_consumption_masks_actor(self):
        policy, base = JoltPolicy("residual"), JoltPolicy("bc")
        episode, _ = learned_episode(policy, base, length=1)
        row = episode.decisions[0]
        mixed = replace(row, event=replace(row.event, ticks=(replace(row.event.ticks[0], source_id="other"), *row.event.ticks[1:])))
        self.assertFalse(mixed.actor_eligible)
        refused = replace(row.event.ticks[0], admission=({"accepted": False, "reason": "refused"},))
        row = replace(row, event=replace(row.event, ticks=(refused, *row.event.ticks[1:])))
        self.assertFalse(row.actor_eligible)
        self.assertFalse(row.admitted)

    def test_native_action_or_packet_mutation_cannot_keep_actor_eligibility(self):
        policy, base = JoltPolicy("residual"), JoltPolicy("bc")
        episode, _ = learned_episode(policy, base, length=1)
        row = episode.decisions[0]
        first = row.event.ticks[0]
        wrong_action = WireAction(move=(-.99, 0.)).to_dict()
        changed = replace(first, consumed_action=wrong_action)
        mutated = replace(row, event=replace(row.event, ticks=(changed, *row.event.ticks[1:])))
        self.assertFalse(mutated.imitation_eligible)
        self.assertFalse(mutated.actor_eligible)
        packet = {**first.consumed_request, "action": wrong_action}
        changed = replace(first, consumed_request=packet)
        mutated = replace(row, event=replace(row.event, ticks=(changed, *row.event.ticks[1:])))
        self.assertFalse(mutated.actor_eligible)

    def test_native_roundoff_has_fixed_tolerance_and_applied_action_is_authoritative(self):
        policy, base = JoltPolicy("residual"), JoltPolicy("bc")
        episode, _ = learned_episode(policy, base, length=1)
        row = episode.decisions[0]
        updated = []
        for tick in row.event.ticks:
            action = row.proposal.to_dict()
            action["move"][0] += 5e-7
            packet = {**tick.consumed_request, "action": action}
            updated.append(replace(tick, requested=(packet,) if tick.requested else (),
                consumed_request=packet, consumed_action=action,
                applied={**tick.applied, "action": action}))
        rounded = replace(row, event=replace(row.event, ticks=tuple(updated)))
        self.assertTrue(rounded.admitted)
        self.assertTrue(rounded.actor_eligible)
        changed = replace(updated[0], applied={**updated[0].applied, "action_available": False})
        unavailable = replace(rounded, event=replace(rounded.event, ticks=(changed, *updated[1:])))
        self.assertFalse(unavailable.actor_eligible)

    def test_native_boolean_cannot_alias_integer_authority_metadata(self):
        policy, base = JoltPolicy("residual"), JoltPolicy("bc")
        episode, _ = learned_episode(policy, base, length=1)
        row = episode.decisions[0]
        first = row.event.ticks[0]
        for key, value in (("epoch", True), ("actor_id", False), ("sequence", False)):
            packet = {**first.consumed_request, key: value}
            changed = replace(first, requested=(packet,), consumed_request=packet)
            malformed = replace(row, event=replace(row.event, ticks=(changed, *row.event.ticks[1:])))
            with self.subTest(key=key):
                self.assertFalse(malformed.admitted)
                self.assertFalse(malformed.actor_eligible)
        packet = {**first.consumed_request, "epoch": 1., "actor_id": 0., "sequence": 0.}
        changed = replace(first, requested=(packet,), consumed_request=packet)
        numeric = replace(row, event=replace(row.event, ticks=(changed, *row.event.ticks[1:])))
        self.assertTrue(numeric.admitted)
        self.assertTrue(numeric.actor_eligible)

    def test_deterministic_sample_never_acquires_original_draw_density(self):
        policy, base = JoltPolicy("residual"), JoltPolicy("bc")
        episode, _ = learned_episode(policy, base, length=1, deterministic=True)
        self.assertFalse(episode.decisions[0].actor_eligible)
        with self.assertRaises(ContractError):
            replace(episode.decisions[0].draw, log_probability=0.)


@unittest.skipUnless(TORCH_AVAILABLE, "optional CPU learning extra is not installed")
class JoltLearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(977)
        self.counts = Counter()
        self.reserve = lambda name, amount=1: self.counts.update({name: amount})

    def test_prefix_burn_in_matches_recurrence_and_padding_does_not_execute(self):
        policy = JoltPolicy("bc")
        rows = [LearningRow(tuple(torch.randn(len(FEATURE_NAMES)).tolist()), reset=index == 0)
                for index in range(11)]
        rows[4] = replace(rows[4], advance=False, reset_after=True)
        expected, hidden = [], policy.zero_hidden()
        for row in rows:
            if not row.advance:
                hidden = policy.zero_hidden()
                expected.append(None)
                continue
            mean, _, _, hidden = policy(torch.tensor([row.features]), hidden)
            expected.append(mean[0])
        result = ordered_window(policy, rows, 8, reserve=self.reserve, stage="bc", length=6)
        torch.testing.assert_close(result.means[:3], torch.stack(expected[8:]))
        self.assertEqual(result.padding.tolist(), [False, False, False, True, True, True])
        self.assertEqual(self.counts["bc_burn_in"], 7)
        self.assertEqual(self.counts["bc_forward"], 3)

    def test_cost_reservation_precedes_every_model_forward(self):
        policy = JoltPolicy("bc")
        rows = [LearningRow((0.,) * len(FEATURE_NAMES))]
        calls = []
        handle = policy.register_forward_pre_hook(lambda *args: calls.append("forward"))
        def exhausted(name, amount):
            raise RuntimeError("bounded stop")
        try:
            with self.assertRaises(RuntimeError):
                ordered_window(policy, rows, 0, reserve=exhausted, stage="bc")
        finally:
            handle.remove()
        self.assertEqual(calls, [])

    def test_primary_conditioned_aim_uses_request_not_success(self):
        means = torch.tensor([[.1, -.2, .3, .4, -.5]], requires_grad=True)
        logits = torch.zeros(1, 2)
        labels = torch.tensor([[.2, .1, .7, 1., 0., 0., 0.]])
        first, detail = clone_loss(means, logits, labels, torch.ones(2))
        changed = labels.clone()
        changed[0, 3:5] = torch.tensor([0., -1.])
        second, _ = clone_loss(means, logits, changed, torch.ones(2))
        torch.testing.assert_close(first, second)
        self.assertEqual(detail["launch_aim"], 0.)
        changed[0, 5] = 1.
        _, launched = clone_loss(means, logits, changed, torch.ones(2))
        self.assertGreater(launched["launch_aim"], 0.)

    def test_fixed_bc_epochs_masks_and_zero_secondary_evidence(self):
        policy = JoltPolicy("bc")
        rows = clone_sequences([teacher_episode(overridden_at=1)])
        before = policy_state_sha256(policy)
        result = train_clone(policy, rows, seed=190103, reserve=self.reserve)
        self.assertNotEqual(before, policy_state_sha256(policy))
        self.assertEqual(result["optimizer_calls"], 40)
        self.assertEqual(result["resolved_shape"]["physical_rows"], 3)
        self.assertEqual(result["resolved_shape"]["eligible_rows"], 2)
        self.assertTrue(result["zero_positive_secondary_demonstrations"])
        self.assertEqual(self.counts["bc_forward"], 80)
        self.assertEqual(self.counts["bc_eligible_samples"], 80)
        self.assertEqual(self.counts["bc_optimizer"], 40)
        with self.assertRaises(ValueError):
            train_clone(policy, rows, seed=190103, reserve=self.reserve, epochs=1)

    def test_original_draw_replay_authenticates_all_five_latents_buttons_and_base(self):
        policy, base = JoltPolicy("residual"), JoltPolicy("bc")
        episode, values = learned_episode(policy, base)
        sequence = rollout_sequence(episode, values)
        authenticate_rollouts(policy, base, [sequence], reserve=self.reserve, source_sha256="c" * 64)
        for channel in range(5):
            row = episode.decisions[0]
            latent = list(row.draw.latent)
            latent[channel] += .25
            damaged = replace(row, draw=replace(row.draw, latent=tuple(latent)))
            broken = replace(episode, decisions=(damaged, *episode.decisions[1:]))
            with self.subTest(channel=channel), self.assertRaises(ValueError):
                authenticate_rollouts(policy, base, [rollout_sequence(broken, values)], reserve=self.reserve, source_sha256="c" * 64)
        for field, changed in (("buttons", (False, False)), ("base_sha256", "d" * 64), ("source_sha256", "e" * 64)):
            row = episode.decisions[0]
            if field == "buttons":
                changed = (not row.draw.buttons[0], row.draw.buttons[1])
            damaged = replace(row, draw=replace(row.draw, **{field: changed}))
            broken = replace(episode, decisions=(damaged, *episode.decisions[1:]))
            with self.subTest(field=field), self.assertRaises(ValueError):
                authenticate_rollouts(policy, base, [rollout_sequence(broken, values)], reserve=self.reserve, source_sha256="c" * 64)

    def test_changed_feature_history_and_invocation_hashes_refuse_actor_evidence(self):
        policy, base = JoltPolicy("residual"), JoltPolicy("bc")
        episode, _ = learned_episode(policy, base, length=1)
        row = episode.decisions[0]
        for name in ("history_sha256", "invocation_sha256", "proposal_sha256"):
            with self.subTest(name=name), self.assertRaises(ContractError):
                replace(row, draw=replace(row.draw, **{name: "d" * 64}))
        damaged = replace(row, draw=replace(row.draw, feature_sha256="d" * 64))
        with self.assertRaises(ContractError):
            record_features(damaged)

    def test_duration_gae_distinguishes_terminal_cutoff_invalid_and_reset(self):
        policy, base = JoltPolicy("residual"), JoltPolicy("bc")
        episode, values = learned_episode(policy, base, length=1, cutoff=True)
        sequence = rollout_sequence(episode, values, bootstrap_value=2.)
        advantage, _ = generalized_advantages(sequence, gamma_per_tick=.9, lambda_per_tick=1.)
        self.assertAlmostEqual(float(advantage[0]), .1 + .9 ** 6 * 2 - values[0], places=6)
        original = sequence.transitions[0]
        terminal_record = replace(original.record, event=replace(original.record.event, terminated=True, truncated=False))
        terminal_row = replace(original, record=terminal_record, task_terminal=True, administrative_cutoff=False)
        terminal = replace(sequence, transitions=(terminal_row,), bootstrap_value=0.)
        advantage, _ = generalized_advantages(terminal, gamma_per_tick=.9, lambda_per_tick=1.)
        self.assertAlmostEqual(float(advantage[0]), .1 - values[0], places=6)
        with self.assertRaises(ValueError):
            replace(terminal, bootstrap_value=2.)
        invalid_record = replace(original.record, event=replace(original.record.event, valid=False))
        invalid = replace(original, record=invalid_record, valid=False, final_observation_valid=False)
        invalid_sequence = replace(sequence, transitions=(invalid,), bootstrap_value=0.)
        advantage, _ = generalized_advantages(invalid_sequence)
        self.assertEqual(float(advantage[0]), 0.)

    def test_reward_terminal_cutoff_and_reset_cannot_detach_from_evidence(self):
        policy, base = JoltPolicy("residual"), JoltPolicy("bc")
        episode, values = learned_episode(policy, base, length=1, cutoff=True)
        sequence = rollout_sequence(episode, values)
        row = sequence.transitions[0]
        for changes in ({"reward": 1000.}, {"task_terminal": True, "administrative_cutoff": False},
                        {"administrative_cutoff": False}, {"valid": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(row, **changes)
        with self.assertRaises(ValueError):
            replace(sequence, transitions=(replace(row, reset=False),))

    def test_ppo_mutates_only_residual_after_authenticated_original_draws(self):
        policy, base = JoltPolicy("residual"), JoltPolicy("bc")
        episode, values = learned_episode(policy, base, terminal=True)
        sequence = rollout_sequence(episode, values)
        before, base_before = policy_state_sha256(policy), policy_state_sha256(base)
        optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
        result = ppo_update(policy, base, optimizer, [sequence], seed=191,
                            reserve=self.reserve, source_sha256="c" * 64)
        self.assertNotEqual(before, policy_state_sha256(policy))
        self.assertEqual(base_before, policy_state_sha256(base))
        self.assertEqual(result["optimizer_calls"], 4)
        self.assertEqual(self.counts["ppo_replay_forward"], 3)
        self.assertEqual(self.counts["ppo_replay_base_forward"], 3)
        self.assertEqual(self.counts["ppo_forward"], 12)
        self.assertEqual(self.counts["ppo_base_forward"], 12)
        self.assertEqual(self.counts["ppo_actor_samples"], 12)
        self.assertEqual(self.counts["ppo_optimizer"], 4)
        # The old rollout cannot be presented as on-policy after mutation.
        with self.assertRaises(ValueError):
            ppo_update(policy, base, optimizer, [sequence], seed=191,
                        reserve=self.reserve, source_sha256="c" * 64)

    def test_ppo_excludes_deterministic_actor_samples_but_preserves_value_work(self):
        policy, base = JoltPolicy("residual"), JoltPolicy("bc")
        episode, values = learned_episode(policy, base, deterministic=True, terminal=True)
        result = ppo_update(policy, base, torch.optim.Adam(policy.parameters(), lr=.0003),
            [rollout_sequence(episode, values)], seed=191, reserve=self.reserve, source_sha256="c" * 64)
        self.assertEqual(result["actor_samples"], 0)
        self.assertEqual(result["excluded_actor_samples"], 3)
        self.assertEqual(self.counts["ppo_actor_samples"], 0)
        self.assertEqual(self.counts["ppo_critic_samples"], 12)


if __name__ == "__main__":
    unittest.main()
