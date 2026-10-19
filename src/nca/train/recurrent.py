"""Ordered recurrent BC/PPO updates with explicit timing and loss masks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from nca.controllers.learned import RecurrentPolicy, residual_log_probability
from nca.data import CaptureEvent


@dataclass(frozen=True)
class Transition:
    features: tuple[float, ...]
    latent: tuple[float, ...] | None
    old_log_probability: float | None
    value: float
    reward: float
    actual_ticks: int
    actor_eligible: bool
    valid: bool = True
    task_terminal: bool = False
    administrative_cutoff: bool = False
    final_observation_valid: bool = True
    episode_id: str = ""
    event_index: int = 0

    def __post_init__(self):
        if type(self.actual_ticks) is not int or not 1 <= self.actual_ticks <= 18:
            raise ValueError("a learning transition requires its actual positive base-tick duration")
        if self.task_terminal and self.administrative_cutoff:
            raise ValueError("a task terminal and an administrative cutoff are distinct")
        if any(not math.isfinite(value) for value in (*self.features, self.value, self.reward)):
            raise ValueError("nonfinite rollout value")
        if self.actor_eligible and (not self.valid or self.latent is None or self.old_log_probability is None
                                    or len(self.latent) != 3 or not math.isfinite(self.old_log_probability)
                                    or not all(math.isfinite(value) for value in self.latent)):
            raise ValueError("actor loss requires a valid attributed residual sample and its old density")


@dataclass(frozen=True)
class RolloutSequence:
    transitions: tuple[Transition, ...]
    bootstrap_value: float = 0.0

    def __post_init__(self):
        if not self.transitions or not math.isfinite(self.bootstrap_value):
            raise ValueError("nonempty ordered rollout with a finite bootstrap required")
        episode = self.transitions[0].episode_id
        for index, item in enumerate(self.transitions):
            if item.episode_id != episode or item.event_index != index:
                raise ValueError("recurrence cannot cross or reorder an episode")
            if index + 1 < len(self.transitions) and (item.task_terminal or item.administrative_cutoff):
                raise ValueError("a sequence must stop at its first terminal/cutoff")
        last = self.transitions[-1]
        if (last.task_terminal or not last.final_observation_valid or not last.valid) and self.bootstrap_value != 0:
            raise ValueError("terminal/invalid final observations cannot bootstrap")


def attributed_actor_sample(event: CaptureEvent) -> tuple[tuple[float, ...], float]:
    """Reject intervention/replacement/mixed intervals before reading a score."""
    sample = event.require_on_policy()
    if sample.space != "latent_residual" or sample.distribution != "scaled-tanh-diagonal-normal/v1":
        raise ValueError("PPO sample does not match its declared residual distribution")
    if sample.values.get("sampling_mode") != "stochastic_draw":
        raise ValueError("PPO requires an explicit original stochastic draw, not a deterministic mean or unknown sampling mode")
    if sample.values.get("log_probability_space") != "bounded_residual":
        raise ValueError("PPO likelihood space is ambiguous")
    latent = tuple(sample.values.get("latent", ()))
    if len(latent) != 3 or not all(math.isfinite(value) for value in latent):
        raise ValueError("original three-channel Gaussian draw is required")
    return latent, float(sample.log_probability)


def generalized_advantages(sequence: RolloutSequence, *, gamma_per_tick: float = .997,
                           lambda_per_tick: float = .99) -> tuple[Tensor, Tensor]:
    """Variable-duration GAE; value bootstrap and recursion are separate masks.

    A valid administrative cutoff bootstraps its own final observation but
    never recurses into a replacement episode. Invalid samples break both
    paths; a task terminal has zero future value.
    """
    if not 0 < gamma_per_tick <= 1 or not 0 < lambda_per_tick <= 1:
        raise ValueError("per-base-tick discounts must lie in (0, 1]")
    rows = sequence.transitions
    advantages = [0.0] * len(rows)
    following = 0.0
    for index in reversed(range(len(rows))):
        item = rows[index]
        if not item.valid:
            following = 0.0
            continue
        at_end = index == len(rows) - 1
        next_valid = at_end or rows[index + 1].valid
        bootstrap = item.final_observation_valid and not item.task_terminal and next_valid
        next_value = sequence.bootstrap_value if at_end else rows[index + 1].value
        discount = gamma_per_tick ** item.actual_ticks
        trace_discount = lambda_per_tick ** item.actual_ticks
        delta = item.reward + (discount * next_value if bootstrap else 0.0) - item.value
        continue_trace = not (at_end or item.task_terminal or item.administrative_cutoff
                              or not item.final_observation_valid or not next_valid)
        following = delta + (discount * trace_discount * following if continue_trace else 0.0)
        advantages[index] = following
    advantage = torch.tensor(advantages, dtype=torch.float32)
    returns = advantage + torch.tensor([row.value for row in rows], dtype=torch.float32)
    return advantage, returns


def ordered_window(policy: RecurrentPolicy, rows: Sequence[Sequence[float]], start: int,
                   length: int = 16) -> tuple[Tensor, Tensor, Tensor]:
    """Rebuild the complete episode prefix under current weights, then unroll.

    Prefix states have no gradient. Every training window starts from zero at
    its true episode boundary, burns in all earlier observations in order,
    and detaches before the loss window. Padding never advances recurrence.
    Window order may shuffle; the samples within a window may not.
    """
    if not rows or not 0 <= start < len(rows) or type(length) is not int or length < 1:
        raise ValueError("ordered window must refer to a nonempty episode suffix")
    hidden = policy.zero_hidden()
    with torch.no_grad():
        for row in rows[:start]:
            _, _, hidden = policy(torch.tensor([row], dtype=torch.float32), hidden)
    hidden = hidden.detach()
    outputs, values, masks = [], [], []
    for index in range(length):
        cursor = start + index
        if cursor < len(rows):
            output, value, hidden = policy(torch.tensor([rows[cursor]], dtype=torch.float32), hidden)
            outputs.append(output[0])
            values.append(value[0])
            masks.append(True)
        else:
            outputs.append(torch.zeros(policy.actor.out_features, dtype=torch.float32))
            values.append(torch.zeros((), dtype=torch.float32))
            masks.append(False)
    return torch.stack(outputs), torch.stack(values), torch.tensor(masks, dtype=torch.bool)


def train_clone(policy: RecurrentPolicy, episodes: Sequence[tuple[Sequence[Sequence[float]],
                                                               Sequence[Sequence[float]]]], *,
                seed: int, epochs: int = 40, unroll: int = 16,
                learning_rate: float = .001, check_budget=None, on_update=None) -> dict[str, Any]:
    if policy.kind != "bc" or not episodes or epochs < 1:
        raise ValueError("cloning needs a BC policy and nonempty training episodes")
    for observations, labels in episodes:
        if not observations or len(observations) != len(labels):
            raise ValueError("demonstration observations and labels must stay ordered and paired")
    labels = torch.tensor([label for _, labels in episodes for label in labels], dtype=torch.float32)
    positives = labels[:, 3:].sum(dim=0)
    positive_weight = ((len(labels) - positives) / positives.clamp_min(1)).clamp(1, 10)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    windows = [(index, start) for index, (rows, _) in enumerate(episodes)
               for start in range(0, len(rows), unroll)]
    rng = random.Random(seed)
    epoch_losses, completed_updates = [], 0
    policy.train()
    for _ in range(epochs):
        order = windows[:]
        rng.shuffle(order)
        weighted_loss, samples = 0.0, 0
        for index, start in order:
            if check_budget is not None:
                check_budget()
            rows, labels_for_episode = episodes[index]
            outputs, _, mask = ordered_window(policy, rows, start, unroll)
            targets = torch.tensor(labels_for_episode[start:start + unroll], dtype=torch.float32)
            outputs = outputs[mask]
            prediction = policy.clone_outputs(outputs)
            continuous_loss = F.mse_loss(prediction[:, :3], targets[:, :3])
            discrete_loss = F.binary_cross_entropy_with_logits(outputs[:, 3:], targets[:, 3:],
                                                              pos_weight=positive_weight)
            loss = continuous_loss + .25 * discrete_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            completed_updates += 1
            if on_update is not None:
                on_update(completed_updates)
            weighted_loss += float(loss.detach()) * len(targets)
            samples += len(targets)
        epoch_losses.append(weighted_loss / samples)
    policy.eval()
    return {"epochs": epochs, "samples_per_epoch": len(labels), "unroll": unroll,
            "burn_in": "complete_episode_prefix_no_grad", "positive_weight": positive_weight.tolist(),
            "epoch_losses": epoch_losses, "optimizer": optimizer.state_dict()}


def ppo_update(policy: RecurrentPolicy, optimizer: torch.optim.Optimizer,
               sequences: Sequence[RolloutSequence], *, seed: int, epochs: int = 4,
               unroll: int = 16, clip_ratio: float = .2,
               value_weight: float = .5, entropy_weight: float = .001) -> dict[str, Any]:
    if policy.kind != "residual" or not sequences or epochs < 1:
        raise ValueError("residual PPO needs nonempty ordered rollout sequences")
    advantages, returns = zip(*(generalized_advantages(sequence) for sequence in sequences))
    actor_values = torch.cat([adv[torch.tensor([row.actor_eligible and row.valid for row in sequence.transitions])]
                              for sequence, adv in zip(sequences, advantages)])
    if actor_values.numel() == 0:
        return {"updates": 0, "actor_samples": 0, "reason": "no_unreplaced_on_policy_samples"}
    mean, std = actor_values.mean(), actor_values.std(unbiased=False).clamp_min(1e-8)
    advantages = tuple((value - mean) / std for value in advantages)
    windows = [(index, start) for index, sequence in enumerate(sequences)
               for start in range(0, len(sequence.transitions), unroll)]
    rng = random.Random(seed)
    losses, ratios, clipped, updates = [], [], [], 0
    policy.train()
    for _ in range(epochs):
        order = windows[:]
        rng.shuffle(order)
        for index, start in order:
            sequence = sequences[index]
            rows = sequence.transitions[start:start + unroll]
            outputs, values, padding = ordered_window(policy, [row.features for row in sequence.transitions], start, unroll)
            n = len(rows)
            actor_mask = torch.tensor([row.actor_eligible and row.valid for row in rows], dtype=torch.bool)
            critic_mask = torch.tensor([row.valid for row in rows], dtype=torch.bool)
            if not bool(critic_mask.any()):
                continue
            loss = value_weight * F.mse_loss(values[:n][critic_mask], returns[index][start:start + n][critic_mask])
            if bool(actor_mask.any()):
                latent = torch.tensor([row.latent or (0., 0., 0.) for row in rows], dtype=torch.float32)
                old = torch.tensor([row.old_log_probability or 0.0 for row in rows], dtype=torch.float32)
                new = residual_log_probability(latent, outputs[:n], policy.log_std)
                ratio = (new[actor_mask] - old[actor_mask]).exp()
                advantage = advantages[index][start:start + n][actor_mask]
                surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)
                # Entropy regularizes the declared latent Gaussian, with its
                # own explicit name. It is not a claimed exact tanh entropy.
                latent_entropy = (policy.log_std.clamp(-5, .5) + .5 * math.log(2 * math.pi * math.e)).sum()
                loss = loss - surrogate.mean() - entropy_weight * latent_entropy
                ratios.extend(ratio.detach().tolist())
                clipped.extend((abs(ratio.detach() - 1) > clip_ratio).to(torch.float32).tolist())
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite PPO objective; preserve the failed run without updating")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            losses.append(float(loss.detach()))
            updates += 1
    policy.eval()
    return {"updates": updates, "actor_samples": actor_values.numel(),
            "excluded_actor_samples": sum(len(seq.transitions) for seq in sequences) - actor_values.numel(),
            "mean_loss": sum(losses) / len(losses) if losses else None,
            "mean_ratio": sum(ratios) / len(ratios) if ratios else None,
            "clip_fraction": sum(clipped) / len(clipped) if clipped else None,
            "epochs": epochs, "unroll": unroll, "burn_in": "complete_episode_prefix_no_grad",
            "entropy_space": "latent_gaussian"}
