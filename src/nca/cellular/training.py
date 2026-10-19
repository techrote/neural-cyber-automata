"""Fixed small BC over eligible training-only owned-part demonstrations."""

from __future__ import annotations

import random

import torch
from torch.nn import functional as F

from nca.contracts import ContractError, WireAction
from nca.controllers.baseline import SkillGoal
from nca.data import Dataset
from nca.randomness import derive_seed
from .core import action_vector, make_frame
from .model import GraphPolicy


def training_sequences(dataset: Dataset):
    """The broader own-part schema is explicit; evaluator values are unused."""
    sequences = []
    for episode in dataset.episodes:
        if episode.split != "training":
            continue
        previous = WireAction(effort=0)
        inputs, targets = [], []
        for event in episode.events:
            if not event.imitation_eligible:
                if inputs:
                    sequences.append((inputs, targets))
                inputs, targets = [], []
                previous = WireAction(effort=0)
                continue
            if (event.intervention != "none" or event.source.training_privilege != "none"
                    or event.source.kind != "scripted"):
                raise ContractError("graph_training_privilege", "this screen excludes corrections and privileged teachers")
            if event.tick_after - event.tick_before != 6 or event.source_observation["tick"] != event.tick_before:
                raise ContractError("graph_training_cadence", "BC requires fresh six-tick authored decision intervals")
            context = event.context
            if context is None:
                raise ContractError("graph_goal", "BC needs an explicit caller-supplied goal")
            goal = SkillGoal(context.goal_id, context.skill, context.target, context.pace,
                             context.tolerance, context.timeout_seconds)
            frame = make_frame(event.source_observation, goal, previous=previous,
                               start_tick=int(episode.initial_observation["tick"]))
            action = WireAction.from_dict(dict(event.requested_packet["action"]))
            inputs.append(frame.nodes)
            targets.append(action_vector(action))
            previous = action
        if inputs:
            sequences.append((inputs, targets))
    if not sequences:
        raise ContractError("graph_training_data", "no eligible autonomous training episodes")
    return sequences


def pad_sequences(sequences):
    if not sequences:
        raise ContractError("graph_sequence", "a batch needs at least one nonempty sequence")
    maximum = max(len(sequence[0]) for sequence in sequences)
    features = torch.zeros((len(sequences), maximum, 3, 30))
    targets = torch.zeros((len(sequences), maximum, 6))
    mask = torch.zeros((len(sequences), maximum), dtype=torch.bool)
    for index, (inputs, actions) in enumerate(sequences):
        if not inputs or len(inputs) != len(actions):
            raise ContractError("graph_sequence", "feature and request lengths must agree")
        features[index, :len(inputs)] = torch.tensor(inputs)
        targets[index, :len(inputs)] = torch.tensor(actions)
        mask[index, :len(inputs)] = True
    return features, targets, mask


def sequence_loss(policy, features, targets, mask, positive_weight):
    hidden = policy.zero_hidden(len(features))
    total = features.new_zeros(())
    for tick in range(features.shape[1]):
        logits, candidate = policy(features[:, tick], hidden)
        continuous = (policy.outputs(logits)[:, :3] - targets[:, tick, :3]).square().mean(dim=-1)
        discrete = F.binary_cross_entropy_with_logits(logits[:, 3:], targets[:, tick, 3:],
                    pos_weight=positive_weight, reduction="none").mean(dim=-1)
        active = mask[:, tick]
        total = total + ((continuous + .5 * discrete) * active.to(features.dtype)).sum()
        # Padding has neither a loss nor a state transition.
        hidden = torch.where(active[:, None, None], candidate, hidden)
    return total / mask.sum().clamp_min(1), hidden


def train_bc(policy: GraphPolicy, sequences, seed: int, budget_check=lambda: None):
    if not sequences:
        raise ContractError("graph_training_data", "empty training set")
    order = random.Random(derive_seed(seed, "cellular.bc"))
    all_targets = torch.tensor([target for _, targets in sequences for target in targets])
    positives = all_targets[:, 3:].sum(dim=0)
    positive_weight = ((len(all_targets) - positives) / positives.clamp_min(1)).clamp(1, 10)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.003)
    packed = pad_sequences(sequences)
    with torch.no_grad():
        initial_loss = float(sequence_loss(policy, *packed, positive_weight)[0])
    epochs, updates = [], 0
    for epoch in range(30):
        indices = list(range(len(sequences)))
        order.shuffle(indices)
        losses = []
        for begin in range(0, len(indices), 2):
            budget_check()
            batch = [sequences[index] for index in indices[begin:begin + 2]]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = sequence_loss(policy, *pad_sequences(batch), positive_weight)
            if not bool(torch.isfinite(loss)):
                raise ContractError("graph_training_finite", "nonfinite imitation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            updates += 1
        epochs.append({"epoch": epoch + 1, "mean_batch_loss": sum(losses) / len(losses)})
    with torch.no_grad():
        final_loss = float(sequence_loss(policy, *packed, positive_weight)[0])
    policy.eval()
    return {"initial_training_loss": initial_loss, "final_training_loss": final_loss,
            "epochs": epochs, "updates": updates, "sequences": len(sequences),
            "samples": len(all_targets), "positive_weight": positive_weight.tolist(),
            "selection": "fixed-final-epoch; no heldout checkpoint selection"}
