"""Synthetic arithmetic/schema tests; no host run or learned transfer result."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

from nca.contracts import ContractError
from nca.controllers.morphology.core import SemanticAction, map_action
from nca.data import ActionSource, CaptureEvent, EpisodeRecord, TickRecord
from nca.data.records import policy_view
from nca.data.morphology import MorphologyEpisode, capture_from_owner
from test_morphology import fixture, goal
from test_morphology_models import ReceiptOwner

HAS_TORCH = importlib.util.find_spec("torch") is not None


def source_view(tick=0):
    adapter, view = fixture("ripper", tick=tick)
    return adapter, view


def owner_episode():
    """A fabricated six-tick receipt fixture, never submitted to a host."""
    adapter, view = source_view()
    target = goal(view)
    semantic = SemanticAction(.2, 0., -.1, .8)
    wire, _ = map_action(adapter, semantic, view, target)
    owner = ReceiptOwner()
    packet = owner.command(wire.to_dict(), view, 12)
    after, _, _, _, _ = owner.step_packet(packet)
    source = ActionSource("synthetic-reference", "scripted", "synthetic-reference", "a" * 64)
    ticks = []
    for i in range(6):
        _, observed = source_view(i + 1)
        ticks.append(TickRecord(i, i + 1, i + 1, i, i + 1,
            (packet,) if i == 0 else (), ({"accepted": True, "reason": "synthetic"},) if i == 0 else (),
            packet, source.id, wire.to_dict(), {"action_available": True, "terminal_revocation": False,
                "action": wire.to_dict()}, {"available": False, "reason": "synthetic unavailable"}, {}, observed, None))
    event = CaptureEvent("step", 0, 0, 0, 6, 0, 100, source, view, policy_view(view),
                         requested_packet=packet, source_observation=view, ticks=tuple(ticks),
                         next_observation=after, reward=0.)
    record = EpisodeRecord("synthetic-morphology", {"lane": "act004"}, "synthetic-revision", "b" * 64,
        {}, adapter.profile, view, (event,), "capped", "synthetic-cutoff", "synthetic-family", "training")
    frame = adapter.frame(view, target)
    return record, target, {0: (frame, semantic, None, {"total": 0.})}


class MorphologyCaptureTests(unittest.TestCase):
    def test_new_schema_roundtrip_retains_full_frame_without_changing_old_episode(self):
        old, target, bindings = owner_episode()
        original = old.sha256
        new = capture_from_owner(old, goal=target, bindings=bindings)
        self.assertEqual(old.sha256, original)
        self.assertEqual(MorphologyEpisode.from_dict(new.to_dict()), new)
        self.assertEqual(new.events[0].policy_observation, bindings[0][0])
        self.assertEqual(new.owner_record(), old)
        self.assertEqual(list(new.imitation_samples()), [(bindings[0][0], bindings[0][1])])
        with self.assertRaises(ContractError):
            EpisodeRecord.from_dict(new.to_dict())

    def test_changed_frame_target_semantic_or_missing_source_is_refused(self):
        old, target, bindings = owner_episode()
        new = capture_from_owner(old, goal=target, bindings=bindings)
        for mutation in ("frame", "target", "semantic", "source"):
            value = new.to_dict()
            if mutation == "frame":
                value["events"][0]["policy_observation"]["nodes"][0][0] += .2
            elif mutation == "target":
                value["goal"]["target"][0] += 1
            elif mutation == "semantic":
                value["events"][0]["semantic_request"]["drive_x"] = .9
            else:
                value["events"][0]["source_observation"] = None
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                MorphologyEpisode.from_dict(value)


@unittest.skipUnless(HAS_TORCH, "optional CPU Torch required for synthetic learner arithmetic")
class MorphologyLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from nca.train import morphology_learning as learning
        cls.torch, cls.learning = torch, learning
        torch.set_num_threads(1)

    def setUp(self):
        self.adapter, self.view = source_view()
        self.goal = goal(self.view)
        self.frame = self.adapter.frame(self.view, self.goal)
        self.model = self.learning.new_policy("padded_recurrent", 6113)

    def frames(self, count):
        return [self.adapter.frame(source_view(i * 6)[1], self.goal, started_tick=0) for i in range(count)]

    def rollout(self, count=2):
        learner, torch = self.learning, self.torch
        hidden, rows = self.model.zero_hidden(), []
        generator = torch.Generator().manual_seed(33)
        for i, frame in enumerate(self.frames(count)):
            with torch.no_grad():
                output = self.model.forward_frame(frame, hidden)
            hidden = output.hidden
            sample = learner.sample_action(output, frame, policy_hash=learner.policy_sha256(self.model),
                                            stochastic=True, generator=generator)
            rows.append(learner.MorphologyTransition(frame, sample, .1, 6, "synthetic", i,
                                                     task_terminal=i == count - 1))
        return learner.MorphologyRollout(tuple(rows))

    def test_masked_latent_density_and_gradient_exclude_unavailable_channels(self):
        torch, learner = self.torch, self.learning
        logits = torch.zeros(7, requires_grad=True)
        latent = torch.zeros(4)
        buttons = torch.zeros(3)
        mask = torch.tensor(self.frame.action_mask, dtype=torch.float32)
        density = learner.latent_log_probability(logits, latent, buttons, mask)
        expected = int(mask[:4].sum()) * (-math.log(.35) - .5 * math.log(2 * math.pi))
        expected -= int(mask[4:].sum()) * math.log(2)
        self.assertAlmostEqual(float(density.detach()), expected, places=6)
        (density + learner.latent_entropy(logits, mask)).backward()
        self.assertEqual(logits.grad[~mask.bool()].tolist(), [0.] * int((mask == 0).sum()))
        large_inactive = logits.detach().clone()
        large_inactive[~mask.bool()] = 3e38
        self.assertEqual(float(learner.latent_log_probability(large_inactive, latent, buttons, mask)),
                         float(density.detach()))

    def test_stochastic_samples_replay_and_never_put_unavailable_actions_in_history(self):
        learner, torch = self.learning, self.torch
        output = self.model.forward_frame(self.frame, self.model.zero_hidden())
        samples = [learner.sample_action(output, self.frame, policy_hash=learner.policy_sha256(self.model),
            stochastic=True, generator=torch.Generator().manual_seed(7)) for _ in range(2)]
        self.assertEqual(samples[0], samples[1])
        for value, active in zip(samples[0].action.vector, self.frame.action_mask):
            if not active:
                self.assertEqual(value, 0)
        with self.assertRaises(ContractError):
            replace(samples[0], log_probability=None)
        with self.assertRaises(ContractError):
            replace(samples[0], action=replace(samples[0].action, drive_y=.5))

    def test_sampler_keeps_original_runtime_source_and_owner_fences(self):
        from nca.controllers.morphology.models import dispatch
        actor = self.learning.MorphologyActor(self.model, self.adapter, seed=7, stochastic=True)
        owner = ReceiptOwner()
        decision = actor.step(self.view, self.goal)
        self.assertFalse(decision.cancel_owner)
        self.assertEqual(decision.semantic_request, actor.last_sample.action)
        after, *_ = dispatch(owner, actor, decision, self.view)
        self.assertTrue(actor.step(self.view, self.goal).cancel_owner)
        self.assertEqual(owner.calls, 1)
        self.assertTrue(actor.step(after, self.goal).cancel_owner)

    def test_complete_prefix_burn_in_matches_direct_recurrence(self):
        frames = self.frames(4)
        hidden, expected = self.model.zero_hidden(), []
        with self.torch.no_grad():
            for frame in frames:
                output = self.model.forward_frame(frame, hidden)
                hidden = output.hidden
                expected.append(output.logits)
        outputs, _ = self.learning.ordered_window(self.model, frames, 2, 16)
        self.assertTrue(self.torch.equal(outputs.detach(), self.torch.stack(expected[2:])))
        with self.assertRaises(ContractError):
            self.learning.train_clone(self.model, [(frames[1:], [SemanticAction()] * 3)], seed=1, epochs=1)

    def test_clone_loss_mask_and_preupdate_budget_guard(self):
        learner = self.learning
        frames = self.frames(2)
        targets = [SemanticAction(.2, 0., -.1, .8)] * 2
        first = learner.new_policy("padded_recurrent", 6113)
        second = learner.new_policy("padded_recurrent", 6113)
        a = learner.train_clone(first, [(frames, targets)], seed=11, epochs=1)
        b = learner.train_clone(second, [(frames, [replace(t, drive_y=.9, impulse=True, rebound=True)
                                                  for t in targets])], seed=11, epochs=1)
        self.assertEqual(a["epoch_losses"], b["epoch_losses"])
        self.assertEqual(learner.policy_sha256(first), learner.policy_sha256(second))
        original = learner.policy_sha256(self.model)
        def exhausted():
            raise RuntimeError("synthetic budget ceiling")
        with self.assertRaises(RuntimeError):
            learner.train_clone(self.model, [(frames, targets)], seed=11, epochs=1, check_budget=exhausted)
        self.assertEqual(learner.policy_sha256(self.model), original)

    def test_ppo_requires_original_policy_density_value_and_order_before_updates(self):
        learner = self.learning
        rollout = self.rollout()
        original = learner.policy_sha256(self.model)
        for field, value in (("policy_sha256", "c" * 64), ("value", 7.), ("log_probability", 7.)):
            sample = replace(rollout.transitions[0].sample, **{field: value})
            changed = replace(rollout, transitions=(replace(rollout.transitions[0], sample=sample),
                                                   *rollout.transitions[1:]))
            optimizer = self.torch.optim.Adam(self.model.parameters(), lr=.0003)
            with self.subTest(field=field), self.assertRaises(ContractError):
                learner.ppo_update(self.model, optimizer, [changed], seed=4)
            self.assertEqual(learner.policy_sha256(self.model), original)
        with self.assertRaises(ContractError):
            replace(rollout, transitions=tuple(reversed(rollout.transitions)))
        critic_only = replace(rollout.transitions[0], actor_eligible=False,
                              sample=replace(rollout.transitions[0].sample, value=7.))
        changed = replace(rollout, transitions=(critic_only, rollout.transitions[1]))
        with self.assertRaises(ContractError):
            learner.ppo_update(self.model, self.torch.optim.Adam(self.model.parameters(), lr=.0003), [changed], seed=4)
        self.assertEqual(learner.policy_sha256(self.model), original)
        with self.assertRaises(ContractError):
            replace(rollout.transitions[0], actual_ticks=1)
        partial_terminal = replace(rollout.transitions[0], actual_ticks=1, task_terminal=True)
        with self.assertRaises(ContractError):
            replace(rollout, transitions=(partial_terminal, rollout.transitions[1]))
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=.0003)
        completed = []
        result = learner.ppo_update(self.model, optimizer, [rollout], seed=4, on_update=completed.append)
        self.assertEqual(result["optimizer_steps"], 4)
        self.assertEqual(completed, [1, 2, 3, 4])

    def test_transition_flags_indices_and_source_generations_are_strict(self):
        learner = self.learning
        rollout = self.rollout()
        row = rollout.transitions[0]
        for field in ("actor_eligible", "valid", "task_terminal", "administrative_cutoff", "final_observation_valid"):
            with self.subTest(field=field), self.assertRaises(ContractError):
                replace(row, **{field: 1})
        for field, value in (("event_index", True), ("actual_ticks", True), ("reward", True)):
            with self.subTest(field=field), self.assertRaises(ContractError):
                replace(row, **{field: value})
        second = rollout.transitions[1]
        identity = replace(second.frame.identity, source_generation=2)
        frame = replace(second.frame, identity=identity)
        with self.assertRaises(ContractError):
            learner.MorphologyRollout((row, replace(second, frame=frame)))

    def test_checkpoint_bytes_metadata_and_rng_are_bound_before_loading(self):
        from nca.train.morphology_artifacts import load_checkpoint, write_checkpoint
        metadata = {"family": "padded_recurrent", "training_seed": 6113, "phase": "initial",
                    "config_sha256": "a" * 64, "manifest_sha256": "b" * 64,
                    "learner_revision": "c" * 40, "learner_sources_sha256": "d" * 64,
                    "dataset_sha256": "e" * 64, "training_privilege": "none"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = write_checkpoint(root, "checkpoints/initial.pt", self.model, metadata,
                                       training_state={"next_episode_boundary": True})
            before = self.torch.get_rng_state().clone()
            loaded, payload = load_checkpoint(root, receipt, expected_metadata=metadata)
            self.assertTrue(self.torch.equal(before, self.torch.get_rng_state()))
            self.assertEqual(self.learning.policy_sha256(loaded), self.learning.policy_sha256(self.model))
            self.assertTrue(payload["training_state"]["next_episode_boundary"])
            with self.assertRaises(ContractError):
                load_checkpoint(root, receipt, expected_metadata={**metadata, "training_seed": 6229})
            original = (root / receipt["file"]).read_bytes()
            (root / receipt["file"]).write_bytes(original + b"changed")
            with self.assertRaises(ContractError):
                load_checkpoint(root, receipt, expected_metadata=metadata)

    def test_cutoff_bootstrap_requires_its_actual_final_input_and_original_value(self):
        learner = self.learning
        terminal = self.rollout()
        last = replace(terminal.transitions[-1], task_terminal=False, administrative_cutoff=True)
        transitions = (*terminal.transitions[:-1], last)
        frame = self.frames(3)[2]
        hidden = self.model.zero_hidden()
        with self.torch.no_grad():
            for row in transitions:
                hidden = self.model.forward_frame(row.frame, hidden).hidden
            value = float(self.model.forward_frame(frame, hidden).value)
        with self.assertRaises(ContractError):
            learner.MorphologyRollout(transitions, value)
        valid = learner.MorphologyRollout(transitions, value, frame)
        changed = replace(valid, bootstrap_value=value + 1)
        original = learner.policy_sha256(self.model)
        with self.assertRaises(ContractError):
            learner.ppo_update(self.model, self.torch.optim.Adam(self.model.parameters(), lr=.0003), [changed], seed=3)
        self.assertEqual(learner.policy_sha256(self.model), original)


if __name__ == "__main__":
    unittest.main()
