"""Ordered masked BC/PPO for the separately gated morphology comparison.

No game collection or gate admission occurs here. These functions accept only
explicit local frames and source-attributed semantic samples. Both candidate
families retain their original parameter counts; sampling variance is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from nca.contracts import ContractError, canonical_json
from nca.controllers.morphology.core import MorphologyFrame, SemanticAction
from nca.controllers.morphology.models import (
    CandidateOutput, MaskedRecurrentPolicy, MorphologyRuntime, SharedGraphPolicy,
)
from nca.data.morphology import FIXED_LATENT_STD, MorphologySample, SAMPLER_SCHEMA
from .recurrent import RolloutSequence, Transition, generalized_advantages
from .morphology_spec import LEARNING_SPEC


def new_policy(family: str, seed: int):
    if family not in ("padded_recurrent", "shared_graph") or type(seed) is not int or seed < 0:
        raise ContractError("morphology_policy", "a declared family and nonnegative independent seed are required")
    torch.manual_seed(seed)
    return MaskedRecurrentPolicy() if family == "padded_recurrent" else SharedGraphPolicy(rounds=2)


def policy_specification(model) -> dict[str, Any]:
    return {"schema": "nca-morphology-policy/v1", "family": model.family,
            "parameters": sum(p.numel() for p in model.parameters()),
            "hidden_shape": list(model.zero_hidden().shape), "message_rounds": getattr(model, "rounds", 0),
            "input_schema": "act004-morphology-local/v1", "outputs": 7}


def policy_sha256(model) -> str:
    digest = hashlib.sha256(canonical_json(policy_specification(model)).encode())
    for name, value in sorted(model.state_dict().items()):
        if value.device.type != "cpu" or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ContractError("morphology_policy", "finite CPU float32 policy weights required")
        digest.update(name.encode())
        digest.update(canonical_json(list(value.shape)).encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _check_logits(logits):
    if (logits.device.type != "cpu" or logits.dtype != torch.float32 or logits.shape[-1] != 7
            or logits.ndim not in (1, 2) or not bool(torch.isfinite(logits).all())):
        raise ContractError("morphology_density", "finite CPU float32 seven-channel logits required")


def latent_log_probability(logits, latent, buttons, mask):
    _check_logits(logits)
    if (latent.shape != (*logits.shape[:-1], 4) or buttons.shape != (*logits.shape[:-1], 3)
            or mask.shape != logits.shape or not bool(torch.isfinite(latent).all())
            or any(value.device.type != "cpu" or value.dtype != torch.float32 for value in (latent, buttons, mask))
            or not bool(((buttons == 0) | (buttons == 1)).all())
            or not bool(((mask == 0) | (mask == 1)).all())):
        raise ContractError("morphology_density", "original finite latents, boolean choices and exact capability masks required")
    # Remove inactive operands before arithmetic: even finite float32 values
    # can overflow when squared, and infinity times a zero mask is still NaN.
    active_logits = torch.where(mask.bool(), logits, torch.zeros_like(logits))
    active_latent = torch.where(mask[..., :4].bool(), latent, torch.zeros_like(latent))
    gaussian = -.5 * ((active_latent - active_logits[..., :4]) / FIXED_LATENT_STD) ** 2
    gaussian = gaussian - math.log(FIXED_LATENT_STD) - .5 * math.log(2 * math.pi)
    discrete = -F.binary_cross_entropy_with_logits(active_logits[..., 4:], buttons, reduction="none")
    result = (gaussian * mask[..., :4]).sum(-1) + (discrete * mask[..., 4:]).sum(-1)
    if not bool(torch.isfinite(result).all()):
        raise ContractError("morphology_density", "nonfinite active-channel likelihood cannot enter optimization")
    return result


def latent_entropy(logits, mask):
    _check_logits(logits)
    if (mask.shape != logits.shape or mask.device.type != "cpu" or mask.dtype != torch.float32
            or not bool(((mask == 0) | (mask == 1)).all())):
        raise ContractError("morphology_density", "exact finite CPU capability mask required")
    logits = torch.where(mask.bool(), logits, torch.zeros_like(logits))
    constant = math.log(FIXED_LATENT_STD) + .5 * math.log(2 * math.pi * math.e)
    probabilities = torch.sigmoid(logits[..., 4:])
    discrete = F.binary_cross_entropy_with_logits(logits[..., 4:], probabilities, reduction="none")
    return constant * mask[..., :4].sum(-1) + (discrete * mask[..., 4:]).sum(-1)


def sample_action(output: CandidateOutput, frame: MorphologyFrame, *, policy_hash: str,
                  stochastic: bool, generator: torch.Generator) -> MorphologySample:
    _check_logits(output.logits)
    if output.logits.ndim != 1 or not output.loss_mask or type(stochastic) is not bool:
        raise ContractError("morphology_sample", "one actual frame output is required")
    mask = torch.tensor(frame.action_mask, dtype=torch.float32)
    with torch.no_grad():
        raw = output.logits[:4]
        if stochastic:
            raw = raw + FIXED_LATENT_STD * torch.randn(4, generator=generator)
            buttons = (torch.rand(3, generator=generator) < torch.sigmoid(output.logits[4:])).float()
        else:
            buttons = (output.logits[4:] >= 0).float()
        latent = raw * mask[:4]
        buttons = buttons * mask[4:]
        continuous = torch.cat((torch.tanh(latent[:3]), torch.sigmoid(latent[3:4]))) * mask[:4]
        semantic = SemanticAction(*continuous.tolist(), *(bool(v) for v in buttons.tolist()))
        density = float(latent_log_probability(output.logits, latent, buttons, mask)) if stochastic else None
    return MorphologySample(policy_hash, frame.policy_sha256, semantic, frame.action_mask,
                            float(output.value.detach()), "stochastic_draw" if stochastic else "deterministic_mean",
                            tuple(latent.tolist()) if stochastic else None, density)


class MorphologyActor(MorphologyRuntime):
    """Specialize sampling only; all issued-source and real-owner fences remain."""

    def __init__(self, model, adapter, *, seed: int, stochastic: bool):
        if type(seed) is not int or seed < 0 or type(stochastic) is not bool:
            raise ContractError("morphology_sampler", "explicit nonnegative seed and boolean sampling mode required")
        super().__init__(model, adapter)
        self.policy_hash = policy_sha256(model)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.stochastic = stochastic
        self.last_frame: MorphologyFrame | None = None
        self.last_sample: MorphologySample | None = None

    def step(self, *args, **kwargs):
        self.last_frame = self.last_sample = None
        return super().step(*args, **kwargs)

    def select_action(self, output: CandidateOutput, frame: MorphologyFrame) -> SemanticAction:
        sample = sample_action(output, frame, policy_hash=self.policy_hash,
                               stochastic=self.stochastic, generator=self.generator)
        self.last_frame, self.last_sample = frame, sample
        return sample.action


def ordered_window(model, frames: Sequence[MorphologyFrame], start: int, length: int = 16):
    if not frames or not 0 <= start < len(frames) or type(length) is not int or length < 1:
        raise ContractError("morphology_unroll", "a nonempty ordered episode suffix is required")
    hidden = model.zero_hidden()
    with torch.no_grad():
        for frame in frames[:start]:
            hidden = model.forward_frame(frame, hidden).hidden
    hidden = hidden.detach()
    outputs, values = [], []
    for frame in frames[start:start + length]:
        result = model.forward_frame(frame, hidden)
        hidden = result.hidden
        outputs.append(result.logits)
        values.append(result.value)
    return torch.stack(outputs), torch.stack(values)


def _validate_episode_frames(frames):
    if not frames:
        raise ContractError("morphology_sequence", "a nonempty original episode is required")
    initial = frames[0]
    if initial.identity.tick != 0 or initial.identity.control_generation != 0:
        raise ContractError("morphology_sequence", "a complete prefix must start at the actual fresh host reset")
    for index, frame in enumerate(frames):
        if (frame.identity.lifetime != initial.identity.lifetime or frame.goal != initial.goal
                or frame.identity.control_generation != initial.identity.control_generation
                or frame.identity.topology_sha256 != initial.identity.topology_sha256
                or any(getattr(frame.identity, name) != getattr(initial.identity, name) for name in
                       ("model", "profile_version", "profile_sha256", "source_generation", "mount_generation"))
                or frame.identity.tick != initial.identity.tick + 6 * index):
            raise ContractError("morphology_sequence", "recurrence cannot cross or reorder an owner, goal or six-tick source")


def train_clone(model, episodes, *, seed: int, epochs: int = 40, check_budget=None, on_update=None):
    if not episodes or type(epochs) is not int or epochs < 1:
        raise ContractError("morphology_clone", "complete eligible ordered teacher episodes required")
    labels, masks = [], []
    for frames, actions in episodes:
        _validate_episode_frames(frames)
        if len(frames) != len(actions):
            raise ContractError("morphology_clone", "all teacher actions must remain paired with their original input")
        labels.extend(action.vector for action in actions)
        masks.extend(frame.action_mask for frame in frames)
    labels, masks = torch.tensor(labels), torch.tensor(masks, dtype=torch.float32)
    positive = (labels[:, 4:] * masks[:, 4:]).sum(0)
    negative = ((1 - labels[:, 4:]) * masks[:, 4:]).sum(0)
    positive_weight = (negative / positive.clamp_min(1)).clamp(1, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_SPEC["bc_learning_rate"])
    windows = [(i, start) for i, (frames, _) in enumerate(episodes) for start in range(0, len(frames), 16)]
    rng, updates, epoch_losses = random.Random(seed), 0, []
    model.train()
    for _ in range(epochs):
        order = windows[:]
        rng.shuffle(order)
        total, count = 0., 0
        for index, start in order:
            if check_budget is not None:
                check_budget()
            frames, actions = episodes[index]
            outputs, _ = ordered_window(model, frames, start)
            target = torch.tensor([action.vector for action in actions[start:start + 16]])
            mask = torch.tensor([frame.action_mask for frame in frames[start:start + 16]], dtype=torch.float32)
            predicted = torch.cat((outputs[:, :3].tanh(), outputs[:, 3:4].sigmoid()), dim=1)
            continuous = (((predicted - target[:, :4]) ** 2) * mask[:, :4]).sum() / mask[:, :4].sum()
            discrete = F.binary_cross_entropy_with_logits(outputs[:, 4:], target[:, 4:], reduction="none",
                                                          pos_weight=positive_weight)
            discrete = (discrete * mask[:, 4:]).sum() / mask[:, 4:].sum().clamp_min(1)
            loss = continuous + LEARNING_SPEC["bc_discrete_weight"] * discrete
            if not bool(torch.isfinite(loss)):
                raise ContractError("morphology_clone", "a nonfinite objective cannot update a candidate")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            updates += 1
            if on_update is not None:
                on_update(updates)
            total += float(loss.detach()) * len(target)
            count += len(target)
        epoch_losses.append(total / count)
    model.eval()
    return {"epochs": epochs, "samples_per_epoch": len(labels), "optimizer_steps": updates,
            "epoch_losses": epoch_losses, "positive_weight": positive_weight.tolist(),
            "unroll": 16, "burn_in": LEARNING_SPEC["burn_in"], "optimizer": optimizer.state_dict()}


@dataclass(frozen=True)
class MorphologyTransition:
    frame: MorphologyFrame
    sample: MorphologySample
    reward: float
    actual_ticks: int
    episode_id: str
    event_index: int
    actor_eligible: bool = True
    valid: bool = True
    task_terminal: bool = False
    administrative_cutoff: bool = False
    final_observation_valid: bool = True

    def __post_init__(self):
        if (not isinstance(self.frame, MorphologyFrame) or not isinstance(self.sample, MorphologySample)
                or any(type(getattr(self, field)) is not bool for field in
                       ("actor_eligible", "valid", "task_terminal", "administrative_cutoff", "final_observation_valid"))
                or type(self.reward) not in (int, float) or not math.isfinite(self.reward)
                or type(self.event_index) is not int or self.event_index < 0
                or not isinstance(self.episode_id, str) or not self.episode_id
                or self.sample.frame_sha256 != self.frame.policy_sha256 or self.sample.action_mask != self.frame.action_mask
                or type(self.actual_ticks) is not int or not 1 <= self.actual_ticks <= 6
                or self.actual_ticks < 6 and not self.task_terminal
                or self.actor_eligible and (not self.valid or self.sample.sampling_mode != "stochastic_draw")):
            raise ContractError("morphology_rollout", "a valid original attributed stochastic interval is required")


@dataclass(frozen=True)
class MorphologyRollout:
    transitions: tuple[MorphologyTransition, ...]
    bootstrap_value: float = 0.
    bootstrap_frame: MorphologyFrame | None = None

    def timing_sequence(self) -> RolloutSequence:
        _validate_episode_frames([item.frame for item in self.transitions])
        for previous, current in zip(self.transitions, self.transitions[1:]):
            if current.frame.identity.tick - previous.frame.identity.tick != previous.actual_ticks:
                raise ContractError("morphology_rollout", "actual receipt duration differs from the next original source tick")
        return RolloutSequence(tuple(Transition(
            features=(), latent=None, old_log_probability=None, value=row.sample.value,
            reward=row.reward, actual_ticks=row.actual_ticks, actor_eligible=False, valid=row.valid,
            task_terminal=row.task_terminal, administrative_cutoff=row.administrative_cutoff,
            final_observation_valid=row.final_observation_valid, episode_id=row.episode_id, event_index=row.event_index,
        ) for row in self.transitions), self.bootstrap_value)

    def __post_init__(self):
        if (not isinstance(self.transitions, tuple) or not self.transitions
                or any(not isinstance(row, MorphologyTransition) for row in self.transitions)
                or type(self.bootstrap_value) not in (int, float) or not math.isfinite(self.bootstrap_value)):
            raise ContractError("morphology_rollout", "immutable ordered transitions and a finite numeric bootstrap required")
        last = self.transitions[-1]
        bootstrap_allowed = last.valid and last.final_observation_valid and last.administrative_cutoff and not last.task_terminal
        if bootstrap_allowed:
            if not isinstance(self.bootstrap_frame, MorphologyFrame):
                raise ContractError("morphology_rollout", "a valid cutoff must retain its actual final local bootstrap input")
            _validate_episode_frames([*(row.frame for row in self.transitions), self.bootstrap_frame])
            if self.bootstrap_frame.identity.tick != last.frame.identity.tick + last.actual_ticks:
                raise ContractError("morphology_rollout", "bootstrap input must follow the final actual receipt duration")
        elif self.bootstrap_frame is not None or self.bootstrap_value != 0:
            raise ContractError("morphology_rollout", "terminal, invalid or undeclared final states cannot bootstrap")
        if last.valid and not (last.task_terminal or last.administrative_cutoff):
            raise ContractError("morphology_rollout", "an ordered collection must end with an explicit task terminal or cutoff")
        self.timing_sequence()


def ppo_update(model, optimizer, sequences: Sequence[MorphologyRollout], *, seed: int,
               check_budget=None, on_update=None):
    if not sequences:
        raise ContractError("morphology_ppo", "ordered complete source-attributed rollout segments required")
    # Reconstruct the original sequence before any optimizer step. A valid
    # same-shaped sample from another policy cannot be relabelled on-policy.
    original_hash = policy_sha256(model)
    with torch.no_grad():
        for sequence in sequences:
            hidden = model.zero_hidden()
            for row in sequence.transitions:
                if check_budget is not None:
                    check_budget()
                if row.sample.policy_sha256 != original_hash:
                    raise ContractError("morphology_ppo_source", "rollout weights differ from the update's frozen initial policy")
                output = model.forward_frame(row.frame, hidden)
                hidden = output.hidden
                if row.valid and abs(float(output.value) - row.sample.value) > 1e-6:
                    raise ContractError("morphology_ppo_source", "recorded original critic value differs from exact recurrent replay")
                if row.actor_eligible:
                    density = latent_log_probability(output.logits, torch.tensor(row.sample.latent),
                        torch.tensor(row.sample.action.vector[4:]), torch.tensor(row.frame.action_mask, dtype=torch.float32))
                    if abs(float(density) - row.sample.log_probability) > 1e-6:
                        raise ContractError("morphology_ppo_source", "recorded original density differs from exact recurrent replay")
            if sequence.bootstrap_frame is not None:
                output = model.forward_frame(sequence.bootstrap_frame, hidden)
                if abs(float(output.value) - sequence.bootstrap_value) > 1e-6:
                    raise ContractError("morphology_ppo_source", "cutoff bootstrap value differs from its actual final recurrent input")
    advantages, returns = zip(*(generalized_advantages(sequence.timing_sequence()) for sequence in sequences))
    eligible = [torch.tensor([row.actor_eligible and row.valid for row in sequence.transitions], dtype=torch.bool)
                for sequence in sequences]
    active = torch.cat([value[mask] for value, mask in zip(advantages, eligible)])
    if not active.numel():
        return {"optimizer_steps": 0, "actor_samples": 0, "reason": "no_original_on_policy_samples"}
    advantages = tuple((value - active.mean()) / active.std(unbiased=False).clamp_min(1e-8) for value in advantages)
    windows = [(i, start) for i, sequence in enumerate(sequences)
               for start in range(0, len(sequence.transitions), 16)]
    rng, updates, losses, ratios = random.Random(seed), 0, [], []
    model.train()
    for _ in range(LEARNING_SPEC["ppo_epochs"]):
        order = windows[:]
        rng.shuffle(order)
        for index, start in order:
            if check_budget is not None:
                check_budget()
            sequence = sequences[index]
            rows = sequence.transitions[start:start + 16]
            outputs, values = ordered_window(model, [r.frame for r in sequence.transitions], start)
            actor_mask = eligible[index][start:start + len(rows)]
            critic_mask = torch.tensor([row.valid for row in rows], dtype=torch.bool)
            if not bool(critic_mask.any()):
                continue
            target = returns[index][start:start + len(rows)]
            loss = .5 * F.mse_loss(values[critic_mask], target[critic_mask])
            if bool(actor_mask.any()):
                latent = torch.tensor([row.sample.latent or (0.,) * 4 for row in rows])
                buttons = torch.tensor([row.sample.action.vector[4:] for row in rows])
                capability = torch.tensor([row.frame.action_mask for row in rows], dtype=torch.float32)
                original = torch.tensor([row.sample.log_probability or 0. for row in rows])
                ratio = (latent_log_probability(outputs, latent, buttons, capability)[actor_mask]
                         - original[actor_mask]).exp()
                advantage = advantages[index][start:start + len(rows)][actor_mask]
                surrogate = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
                loss = loss - surrogate.mean() - .001 * latent_entropy(outputs, capability)[actor_mask].mean()
                ratios.extend(ratio.detach().tolist())
            if not bool(torch.isfinite(loss)):
                raise ContractError("morphology_ppo", "a nonfinite objective cannot update a candidate")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            updates += 1
            if on_update is not None:
                on_update(updates)
            losses.append(float(loss.detach()))
    model.eval()
    return {"optimizer_steps": updates, "actor_samples": active.numel(),
            "excluded_actor_samples": sum(len(s.transitions) for s in sequences) - active.numel(),
            "mean_loss": sum(losses) / len(losses), "mean_ratio": sum(ratios) / len(ratios),
            "entropy_space": LEARNING_SPEC["entropy_space"], "burn_in": LEARNING_SPEC["burn_in"],
            "epochs": LEARNING_SPEC["ppo_epochs"], "unroll": 16}
