"""LEARN-JOLT-002 ordered imitation and recurrent residual learning.

Production defaults are the prospectively declared forty BC epochs and four
PPO epochs, with sixteen-step windows. Masked rows stay inside the original
physical episode. All forward, prefix replay and optimizer work is reserved
before it occurs through the required caller budget callback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from nca.contracts import ContractError, WireAction, sha256_json
from nca.controllers.baseline import SkillGoal
from nca.controllers.jolt_learned import (
    ActionHistory, EPSILON, FEATURE_NAMES, InvocationContext, JoltPolicy,
    combine_residual, features, joint_log_probability, map_action,
    mapped_continuous, policy_state_sha256,
)
from nca.data.jolt_samples import JoltDecisionRecord, JoltDraw, JoltEpisode, JoltEpisodeV3, validate_history


Reserve = Callable[[str, int], Any]
TERMINAL_SCHEMA_V3 = "nca-jolt-learning-terminal/v3"
BC_SPEC = {
    "epochs": 40, "unroll": 16, "learning_rate": .001, "gradient_norm": .5,
    "movement_trim_effort_weight": 1., "launch_aim_weight": .5, "button_weight": .25,
    "aim_selection": "ordinary_requested_primary_only_no_outcome_filter",
    "positive_weight": "training_negatives_div_max_positives_1_clamped_1_10",
    "no_positive_secondary_weight": 1.,
    "window_rule": "fixed_physical_episode_windows_with_internal_masks",
    "burn_in": "complete_episode_prefix_current_weights_no_grad",
}
PPO_SPEC = {
    "updates_per_seed": 8, "decisions_per_update": 256, "epochs": 4, "unroll": 16,
    "learning_rate": .0003, "gradient_norm": .5, "clip_ratio": .2,
    "value_weight": .5, "entropy_weight": .001, "gamma_per_tick": .997,
    "lambda_per_tick": .99, "entropy_space": "raw_normal5_plus_bernoulli2",
    "burn_in": "complete_episode_prefix_current_weights_no_grad",
}


def _reserve(reserve: Reserve, counter: str, amount: int = 1) -> None:
    if not callable(reserve):
        raise ValueError("a pre-work reservation callback is mandatory")
    if amount:
        reserve(counter, amount)


def record_features(record: JoltDecisionRecord) -> tuple[float, ...] | None:
    """Reconstruct inputs from the full observation, original clock and history."""
    if not record.observation_valid:
        return None
    caller = record.event.context
    if caller is None or caller.skill != "directed_hop":
        raise ContractError("jolt_context", "a directed-hop caller context is required")
    goal = SkillGoal(caller.goal_id, caller.skill, caller.target, caller.pace,
                     caller.tolerance, caller.timeout_seconds)
    context = InvocationContext.from_dict(record.invocation)
    history = ActionHistory.from_dict(record.history)
    result = features(record.event.source_observation, goal, context=context, history=history)
    if record.draw is not None and record.draw.feature_sha256 != sha256_json(result):
        raise ContractError("jolt_features", "the original draw's observation/context/features changed")
    return tuple(result)


def action_label(action: WireAction) -> tuple[float, ...]:
    return (action.move[0], action.trim, action.effort, *action.aim,
            float(action.primary), float(action.secondary))


@dataclass(frozen=True)
class LearningRow:
    features: tuple[float, ...] | None
    reset: bool = False
    label: tuple[float, ...] | None = None
    eligible: bool = False
    advance: bool = True
    reset_after: bool = False

    def __post_init__(self) -> None:
        if self.features is not None and (len(self.features) != len(FEATURE_NAMES)
                or not all(math.isfinite(value) for value in self.features)):
            raise ValueError("learning inputs must match the finite declared feature vector")
        if self.label is not None and (len(self.label) != 7
                or not all(math.isfinite(value) for value in self.label)):
            raise ValueError("a complete label has five continuous and two button values")
        if self.eligible and (self.features is None or self.label is None or not self.advance):
            raise ValueError("ordinary imitation loss needs the original observation and proposal")


@dataclass(frozen=True)
class CloneEpisode:
    id: str
    rows: tuple[LearningRow, ...]
    evidence_sha256: str
    status: str

    def __post_init__(self) -> None:
        if not 1 <= len(self.rows) <= 40:
            raise ValueError("a training episode preserves one to forty physical decision attempts")


def clone_sequences(episodes: Sequence[JoltEpisode], *, expected_teacher_sha256: str | None = None,
                    expected_supervisor_sha256: str | None = None) -> tuple[CloneEpisode, ...]:
    """Keep failures, abstention and masked rows; never open or filter holdouts."""
    if not episodes or any(episode.split != "training" for episode in episodes):
        raise ValueError("teacher conversion accepts only the frozen training split")
    if len({episode.id for episode in episodes}) != len(episodes):
        raise ValueError("duplicate demonstration identity")
    converted = []
    for episode in episodes:
        validate_history(episode)
        rows = []
        for index, record in enumerate(episode.decisions):
            if expected_teacher_sha256 is not None and record.event.source.policy_sha256 != expected_teacher_sha256:
                raise ValueError("demonstration teacher differs from the frozen admitted source")
            if expected_supervisor_sha256 is not None and record.supervisor_sha256 != expected_supervisor_sha256:
                raise ValueError("demonstration supervisor differs from the matched declaration")
            if record.event.source.kind != "scripted" or record.draw is not None:
                raise ValueError("this BC dataset requires the admitted authored teacher, including its failures")
            encoded = record_features(record)
            label = None if record.proposal is None else action_label(record.proposal)
            # An override's original proposal remains evidence but supplies no
            # ordinary target. Its transmitted action remains history.
            rows.append(LearningRow(encoded, index == 0 or record.recurrent_reset,
                                    label, record.imitation_eligible,
                                    record.proposal is not None, record.overridden or not record.admitted))
        if rows:
            converted.append(CloneEpisode(episode.id, tuple(rows), episode.sha256, episode.status))
    if not converted or not any(row.eligible for episode in converted for row in episode.rows):
        raise ValueError("no eligible ordinary teacher samples; retain the failed data without fitting")
    return tuple(converted)


def convert_clone_episode_v3(episode: JoltEpisodeV3, *, expected_teacher_sha256: str,
                              expected_supervisor_sha256: str) -> CloneEpisode:
    """Convert one complete capture; keep a masked episode without resampling.

    The worker retains compact features/labels and releases raw native packets
    between demonstrations. The fixed complete dataset must have at least one
    eligible ordinary row before fitting; an all-masked episode is not filtered.
    """
    if (not isinstance(episode, JoltEpisodeV3) or episode.split != "training"
            or episode.outcome["complete"] is not True or not episode.decisions):
        raise ValueError("v3 demonstration conversion needs a complete positive-progress training capture")
    validate_history(episode)
    rows = []
    for index, record in enumerate(episode.decisions):
        if (record.event.source.policy_sha256 != expected_teacher_sha256
                or record.supervisor_sha256 != expected_supervisor_sha256
                or record.event.source.kind != "scripted" or record.draw is not None):
            raise ValueError("v3 demonstration source differs from the frozen authored teacher and supervisor")
        rows.append(LearningRow(record_features(record), index == 0 or record.recurrent_reset,
            None if record.proposal is None else action_label(record.proposal), record.imitation_eligible,
            record.proposal is not None, record.overridden or not record.admitted))
    return CloneEpisode(episode.id, tuple(rows), episode.sha256, episode.status)


@dataclass(frozen=True)
class WindowOutput:
    means: Tensor
    button_logits: Tensor
    values: Tensor
    valid: Tensor
    padding: Tensor
    base_means: Tensor | None = None


def ordered_window(policy: JoltPolicy, rows: Sequence[LearningRow], start: int, *,
                   reserve: Reserve, stage: str, length: int = 16,
                   base_policy: JoltPolicy | None = None) -> WindowOutput:
    """Replay a complete physical prefix; missing rows and padding do not step."""
    if stage not in ("bc", "ppo") or not rows or not 0 <= start < len(rows) or length < 1:
        raise ValueError("ordered window requires a declared stage and episode suffix")
    if (policy.kind == "residual") != (base_policy is not None):
        raise ValueError("a residual window needs its own frozen learned BC base")
    hidden = policy.zero_hidden()
    base_hidden = None if base_policy is None else base_policy.zero_hidden()
    dtype = next(policy.parameters()).dtype

    def advance(row: LearningRow, *, burn_in: bool):
        nonlocal hidden, base_hidden
        if row.reset or row.features is None or not row.advance:
            hidden = policy.zero_hidden()
            base_hidden = None if base_policy is None else base_policy.zero_hidden()
        if row.features is None or not row.advance:
            return None
        tensor = torch.tensor([row.features], dtype=dtype)
        _reserve(reserve, f"{stage}_{'burn_in' if burn_in else 'forward'}")
        means, logits, value, hidden = policy(tensor, hidden)
        base_means = None
        if base_policy is not None:
            _reserve(reserve, f"ppo_base_{'burn_in' if burn_in else 'forward'}")
            with torch.no_grad():
                base_means, base_logits, _, base_hidden = base_policy(tensor, base_hidden)
            _, logits = combine_residual(base_means, base_logits, means, logits)
        result = means[0], logits[0], value[0], None if base_means is None else base_means[0]
        if row.reset_after:
            hidden = policy.zero_hidden()
            base_hidden = None if base_policy is None else base_policy.zero_hidden()
        return result

    with torch.no_grad():
        for row in rows[:start]:
            advance(row, burn_in=True)
    hidden = hidden.detach()
    if base_hidden is not None:
        base_hidden = base_hidden.detach()
    means, logits, values, valid, padding, base = [], [], [], [], [], []
    for cursor in range(start, start + length):
        present = cursor < len(rows)
        output = advance(rows[cursor], burn_in=False) if present else None
        means.append(torch.zeros(5, dtype=dtype) if output is None else output[0])
        logits.append(torch.zeros(2, dtype=dtype) if output is None else output[1])
        values.append(torch.zeros((), dtype=dtype) if output is None else output[2])
        if base_policy is not None:
            base.append(torch.zeros(5, dtype=dtype) if output is None else output[3])
        valid.append(output is not None)
        padding.append(not present)
    return WindowOutput(torch.stack(means), torch.stack(logits), torch.stack(values),
                        torch.tensor(valid, dtype=torch.bool), torch.tensor(padding, dtype=torch.bool),
                        None if base_policy is None else torch.stack(base))


def _normalized_aim(raw: Tensor) -> Tensor:
    vector = mapped_continuous(raw)[:, 3:5]
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    default = torch.zeros_like(vector)
    default[:, 1] = -1.
    return torch.where(norm <= EPSILON, default, vector / norm.clamp_min(EPSILON))


def clone_loss(means: Tensor, button_logits: Tensor, labels: Tensor,
               positive_weight: Tensor) -> tuple[Tensor, Mapping[str, float]]:
    """Aim selection uses requested primary, never a later physical outcome."""
    if means.shape[-1] != 5 or button_logits.shape[-1] != 2 or labels.shape[-1] != 7:
        raise ValueError("full learned actuation loss requires five plus two channels")
    movement = F.mse_loss(mapped_continuous(means)[:, :3], labels[:, :3])
    primary = labels[:, 5].to(torch.bool)
    aim = F.mse_loss(_normalized_aim(means)[primary], labels[primary, 3:5]) if bool(primary.any()) else means.sum() * 0
    buttons = F.binary_cross_entropy_with_logits(button_logits, labels[:, 5:7], pos_weight=positive_weight)
    loss = movement + .5 * aim + .25 * buttons
    return loss, {"movement": float(movement.detach()), "launch_aim": float(aim.detach()),
                  "buttons": float(buttons.detach())}


def train_clone(policy: JoltPolicy, episodes: Sequence[CloneEpisode], *, seed: int, reserve: Reserve,
                epochs: int = 40, unroll: int = 16, learning_rate: float = .001) -> dict[str, Any]:
    if policy.kind != "bc" or not episodes or len(episodes) > 64:
        raise ValueError("cloning needs the BC model and at most 64 frozen physical episodes")
    if (epochs, unroll, learning_rate) != (40, 16, .001):
        raise ValueError("LEARN-JOLT-002 freezes epoch40, unroll16 and Adam rate .001")
    labels = torch.tensor([row.label for episode in episodes for row in episode.rows if row.eligible], dtype=torch.float32)
    if labels.numel() == 0:
        raise ValueError("no eligible teacher labels")
    positives = labels[:, 5:7].sum(dim=0)
    positive_weight = ((len(labels) - positives) / positives.clamp_min(1)).clamp(1, 10)
    if positives[1] == 0:
        positive_weight[1] = 1.
    windows = [(index, start) for index, episode in enumerate(episodes)
               for start in range(0, len(episode.rows), unroll)]
    declared_shape = {"episodes": len(episodes), "physical_rows": sum(len(item.rows) for item in episodes),
                      "eligible_rows": len(labels), "windows_per_epoch": len(windows),
                      "maximum_optimizer_calls": epochs * len(windows)}
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    rng = random.Random(seed)
    losses, updates = [], 0
    policy.train()
    for _ in range(epochs):
        order = windows[:]
        rng.shuffle(order)
        total_loss, samples = 0., 0
        for index, start in order:
            rows = episodes[index].rows
            selected = rows[start:start + unroll]
            mask = torch.tensor([row.eligible for row in selected], dtype=torch.bool)
            if not bool(mask.any()):
                # Later windows still replay these physical rows in burn-in.
                continue
            output = ordered_window(policy, rows, start, reserve=reserve, stage="bc", length=unroll)
            targets = torch.tensor([row.label or (0.,) * 7 for row in selected], dtype=torch.float32)[mask]
            _reserve(reserve, "bc_eligible_samples", len(targets))
            loss, _ = clone_loss(output.means[:len(selected)][mask], output.button_logits[:len(selected)][mask],
                                 targets, positive_weight)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite cloning objective; preserve data without an update")
            _reserve(reserve, "bc_optimizer")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            updates += 1
            total_loss += float(loss.detach()) * len(targets)
            samples += len(targets)
        losses.append(total_loss / samples if samples else None)
    policy.eval()
    return {"spec": dict(BC_SPEC), "resolved_shape": declared_shape, "optimizer_calls": updates,
            "positive_weight": positive_weight.tolist(), "positive_button_labels": positives.tolist(),
            "zero_positive_secondary_demonstrations": bool(positives[1] == 0),
            "data_selection": "all eligible ordinary proposals, including unsuccessful episodes and abstention",
            "epoch_losses": losses, "optimizer": optimizer.state_dict()}


@dataclass(frozen=True)
class JoltTransition:
    record: JoltDecisionRecord
    features: tuple[float, ...] | None
    value: float
    reward: float
    actual_ticks: int
    valid: bool = True
    task_terminal: bool = False
    administrative_cutoff: bool = False
    final_observation_valid: bool = True
    reset: bool = False

    def __post_init__(self) -> None:
        if type(self.actual_ticks) is not int or not 0 <= self.actual_ticks <= 6:
            raise ValueError("a transition retains zero to six actually consumed ticks")
        if self.actual_ticks != self.record.actual_ticks:
            raise ValueError("transition duration differs from native evidence")
        event = self.record.event
        expected_valid = bool(self.record.observation_valid and self.record.proposal is not None
                              and event.valid and event.trace_complete and self.actual_ticks > 0)
        if self.valid != expected_valid or (self.valid and self.features is None):
            raise ValueError("transition validity must match complete positive-duration original evidence")
        if self.task_terminal and self.administrative_cutoff:
            raise ValueError("task terminal and administrative cutoff have different bootstrap semantics")
        if not math.isfinite(self.value) or not math.isfinite(self.reward):
            raise ValueError("nonfinite value/reward")
        if (self.reward != float(event.reward if event.reward is not None else 0.)
                or self.task_terminal != event.terminated or self.administrative_cutoff != event.truncated):
            raise ValueError("reward and terminal/cutoff semantics must match the recorded task evidence")
        if self.final_observation_valid and event.next_observation is None:
            raise ValueError("an absent physical final observation cannot bootstrap")
        if self.features is not None and (len(self.features) != len(FEATURE_NAMES)
                or not all(math.isfinite(value) for value in self.features)):
            raise ValueError("invalid feature vector")

    @property
    def actor_eligible(self) -> bool:
        return self.valid and self.record.actor_eligible


@dataclass(frozen=True)
class JoltRolloutSequence:
    episode_id: str
    transitions: tuple[JoltTransition, ...]
    bootstrap_value: float = 0.

    def __post_init__(self) -> None:
        if not 1 <= len(self.transitions) <= 40 or not math.isfinite(self.bootstrap_value):
            raise ValueError("rollout must retain one physical episode with finite bootstrap")
        for index, row in enumerate(self.transitions):
            if index + 1 < len(self.transitions) and (row.task_terminal or row.administrative_cutoff):
                raise ValueError("a rollout cannot continue beyond a terminal/cutoff")
            if row.reset != (index == 0 or row.record.recurrent_reset):
                raise ValueError("recurrent reset semantics differ from the original episode boundary")
        last = self.transitions[-1]
        if (last.task_terminal or not last.valid or not last.final_observation_valid) and self.bootstrap_value != 0:
            raise ValueError("terminal/invalid final state cannot bootstrap")
        evidence = JoltEpisode(self.episode_id, "training", tuple(row.record for row in self.transitions),
                               "capped" if last.administrative_cutoff else "completed")
        validate_history(evidence)


@dataclass(frozen=True)
class JoltRolloutSequenceV3(JoltRolloutSequence):
    """Execution-v3 terminal boundary, without altering native task flags.

    In particular, a supervisor retirement discovered at a cutoff peek wins
    over administrative bootstrapping, but the already-retained cutoff flag
    and every per-tick reward remain original evidence. Masked override rows
    still receive no actor/critic loss and still cut the original GAE trace.
    """

    terminal_failure: bool = False
    episode_outcome_sha256: str = ""
    terminal_schema: str = TERMINAL_SCHEMA_V3

    def __post_init__(self) -> None:
        super().__post_init__()
        from nca.contracts import digest
        digest(self.episode_outcome_sha256, "episode_outcome_sha256")
        if type(self.terminal_failure) is not bool or self.terminal_schema != TERMINAL_SCHEMA_V3:
            raise ValueError("v3 terminal semantics must be explicit")
        if self.terminal_failure and (self.bootstrap_value != 0.
                or self.transitions[-1].final_observation_valid
                or any(row.task_terminal for row in self.transitions)):
            raise ValueError("safe failure has zero bootstrap and no critic observation or invented task terminal")


def rollout_sequence(episode: JoltEpisode, values: Sequence[float], *, bootstrap_value: float = 0.,
                     final_observation_valid: bool = True) -> JoltRolloutSequence:
    if episode.split != "training" or len(values) != len(episode.decisions):
        raise ValueError("PPO must pair every frozen training decision with its original value")
    rows = []
    for index, (record, value) in enumerate(zip(episode.decisions, values)):
        event = record.event
        valid = bool(record.observation_valid and record.proposal is not None
                     and event.valid and event.trace_complete and record.actual_ticks > 0)
        final_valid = event.next_observation is not None and (final_observation_valid or index + 1 < len(values))
        rows.append(JoltTransition(record, record_features(record), float(value), float(event.reward or 0.),
            record.actual_ticks, valid, event.terminated, event.truncated, final_valid,
            index == 0 or record.recurrent_reset))
    return JoltRolloutSequence(episode.id, tuple(rows), bootstrap_value)


def rollout_sequence_v3(episode: JoltEpisodeV3, values: Sequence[float | None], *,
                        bootstrap_value: float = 0., final_observation_valid: bool = True) -> JoltRolloutSequenceV3:
    """Translate only a receipt-complete v3 episode to the unchanged PPO rows."""
    if not isinstance(episode, JoltEpisodeV3) or episode.outcome["complete"] is not True:
        raise ValueError("v3 PPO requires a complete explicitly versioned outcome")
    terminal_failure = episode.outcome["kind"] == "safe_retirement"
    if terminal_failure and (bootstrap_value != 0. or final_observation_valid):
        raise ValueError("safe failure cannot acquire a bootstrap value observation")
    if len(values) != len(episode.decisions):
        raise ValueError("v3 PPO values must match the exact retained physical rows")
    if type(bootstrap_value) not in (int, float) or not math.isfinite(bootstrap_value):
        raise ValueError("v3 PPO requires a finite original cutoff value")
    for record, value in zip(episode.decisions, values):
        if (value is None) != (record.proposal is None):
            raise ValueError("v3 preserves absent model values on value-less override intervals")
    # This zero is an internal masked tensor placeholder, never a model value
    # observation in retained evidence. Invalid rows train neither loss head.
    numeric = [0. if value is None else float(value) for value in values]
    learning_bootstrap = bootstrap_value
    if episode.decisions:
        last = episode.decisions[-1]
        event = last.event
        if (episode.outcome["kind"] == "administrative_cutoff"
                and final_observation_valid is True and last.proposal is None
                and last.overridden and last.observation_valid and last.admitted
                and event.valid and event.trace_complete and last.actual_ticks == 6
                and event.truncated and not event.terminated and event.next_observation is not None):
            # Recovery can finish at the cutoff peek after a consumed override.
            # Its real value/observation and forward charges remain in capture
            # and independent replay. The unchanged GAE mask discards this last
            # row and cuts the trace, so only its unused tensor placeholder is
            # zero. Ordinary cutoffs and legacy validation keep their meaning.
            learning_bootstrap = 0.
    original = rollout_sequence(episode, numeric, bootstrap_value=learning_bootstrap,
                                final_observation_valid=final_observation_valid)
    return JoltRolloutSequenceV3(original.episode_id, original.transitions, original.bootstrap_value,
                                terminal_failure, sha256_json(episode.outcome))


def generalized_advantages(sequence: JoltRolloutSequence, *, gamma_per_tick: float = .997,
                           lambda_per_tick: float = .99) -> tuple[Tensor, Tensor]:
    """Discount by confirmed duration; cutoff value and trace masks differ."""
    if not 0 < gamma_per_tick <= 1 or not 0 < lambda_per_tick <= 1:
        raise ValueError("discounts are per-base-tick values in (0,1]")
    rows = sequence.transitions
    advantages = [0.] * len(rows)
    following = 0.
    for index in reversed(range(len(rows))):
        row = rows[index]
        if not row.valid:
            following = 0.
            continue
        at_end = index + 1 == len(rows)
        next_valid = at_end or (rows[index + 1].valid and not rows[index + 1].reset
                                and not row.record.overridden and row.record.admitted)
        terminal_failure = at_end and isinstance(sequence, JoltRolloutSequenceV3) and sequence.terminal_failure
        bootstrap = row.final_observation_valid and not row.task_terminal and not terminal_failure and next_valid
        next_value = sequence.bootstrap_value if at_end else rows[index + 1].value
        discount, trace = gamma_per_tick ** row.actual_ticks, lambda_per_tick ** row.actual_ticks
        delta = row.reward + (discount * next_value if bootstrap else 0.) - row.value
        continuation = not (at_end or row.task_terminal or row.administrative_cutoff
                            or not row.final_observation_valid or not next_valid)
        following = delta + (discount * trace * following if continuation else 0.)
        advantages[index] = following
    advantage = torch.tensor(advantages, dtype=torch.float32)
    return advantage, advantage + torch.tensor([row.value for row in rows], dtype=torch.float32)


@torch.no_grad()
def authenticate_rollouts(policy: JoltPolicy, base_policy: JoltPolicy,
                          sequences: Sequence[JoltRolloutSequence], *, reserve: Reserve,
                          source_sha256: str) -> None:
    """Replay original proposal/density before the first PPO update mutates it."""
    if policy.kind != "residual" or base_policy.kind != "bc":
        raise ValueError("PPO requires a residual and its frozen learned BC base")
    policy_sha, base_sha = policy_state_sha256(policy), policy_state_sha256(base_policy)
    dtype = next(policy.parameters()).dtype
    for sequence in sequences:
        hidden, base_hidden = policy.zero_hidden(), base_policy.zero_hidden()
        for row in sequence.transitions:
            if row.reset or row.features is None or row.record.proposal is None:
                hidden, base_hidden = policy.zero_hidden(), base_policy.zero_hidden()
            encoded = record_features(row.record)
            if encoded != row.features:
                raise ValueError("rollout inputs differ from the original ordered observation/history")
            if encoded is None or row.record.proposal is None:
                continue
            _reserve(reserve, "ppo_replay_forward")
            mean, logits, value, hidden = policy(torch.tensor([encoded], dtype=dtype), hidden)
            _reserve(reserve, "ppo_replay_base_forward")
            base_mean, base_logits, _, base_hidden = base_policy(torch.tensor([encoded], dtype=dtype), base_hidden)
            if row.valid and not math.isclose(float(value[0]), row.value, rel_tol=1e-5, abs_tol=2e-5):
                raise ValueError("rollout value differs from its original model/history")
            draw = row.record.draw
            if draw is None:
                if row.actor_eligible:
                    raise ValueError("eligible PPO row lost its original stochastic draw")
                if row.record.overridden or not row.record.admitted:
                    hidden, base_hidden = policy.zero_hidden(), base_policy.zero_hidden()
                continue
            if (draw.policy_sha256 != policy_sha or draw.base_sha256 != base_sha
                    or draw.source_sha256 != source_sha256):
                raise ValueError("sample model, frozen BC base or executing source identity changed")
            latent = torch.tensor([draw.latent], dtype=dtype)
            buttons = torch.tensor([draw.buttons], dtype=dtype)
            combined, combined_logits = combine_residual(base_mean, base_logits, latent, logits)
            proposal = map_action(combined[0].tolist(), draw.buttons)
            if proposal != row.record.proposal:
                raise ValueError("raw latent/button draw no longer maps to its original policy proposal")
            if draw.sampling_mode == "stochastic_draw":
                score = float(joint_log_probability(latent, mean, policy.log_std, buttons, combined_logits)[0])
                if not math.isclose(score, draw.log_probability, rel_tol=1e-5, abs_tol=2e-5):
                    raise ValueError("original raw Normal/Bernoulli behaviour density failed replay")
            else:
                if not torch.equal(latent, mean) or tuple(bool(x) for x in (combined_logits[0] >= 0).tolist()) != draw.buttons:
                    raise ValueError("deterministic sample differs from the original mean/threshold")
            if row.record.overridden or not row.record.admitted:
                hidden, base_hidden = policy.zero_hidden(), base_policy.zero_hidden()


def ppo_update(policy: JoltPolicy, base_policy: JoltPolicy, optimizer: torch.optim.Optimizer,
               sequences: Sequence[JoltRolloutSequence], *, seed: int, reserve: Reserve,
               source_sha256: str, epochs: int = 4, unroll: int = 16) -> dict[str, Any]:
    if not sequences or not 1 <= sum(len(item.transitions) for item in sequences) <= 256:
        raise ValueError("one PPO update admits at most 256 original decision attempts")
    if (epochs, unroll) != (4, 16):
        raise ValueError("LEARN-JOLT-002 freezes four PPO epochs with unroll16")
    if len({item.episode_id for item in sequences}) != len(sequences):
        raise ValueError("duplicate physical episode in PPO rollout")
    if (not isinstance(optimizer, torch.optim.Adam)
            or any(group["lr"] != .0003 for group in optimizer.param_groups)
            or {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in policy.parameters()}):
        raise ValueError("PPO optimizer must own only residual parameters at the frozen Adam rate")
    authenticate_rollouts(policy, base_policy, sequences, reserve=reserve, source_sha256=source_sha256)
    base_sha = policy_state_sha256(base_policy)
    advantages, returns = zip(*(generalized_advantages(item) for item in sequences))
    actor_values = torch.cat([value[torch.tensor([row.actor_eligible for row in item.transitions])]
                              for item, value in zip(sequences, advantages)])
    if actor_values.numel():
        mean, scale = actor_values.mean(), actor_values.std(unbiased=False).clamp_min(1e-8)
        advantages = tuple((value - mean) / scale for value in advantages)
    windows = [(index, start) for index, item in enumerate(sequences)
               for start in range(0, len(item.transitions), unroll)]
    rng = random.Random(seed)
    losses, ratios, clips, updates = [], [], [], 0
    policy.train()
    base_policy.eval()
    for _ in range(epochs):
        order = windows[:]
        rng.shuffle(order)
        for index, start in order:
            sequence = sequences[index]
            selected = sequence.transitions[start:start + unroll]
            actor = torch.tensor([row.actor_eligible for row in selected], dtype=torch.bool)
            critic = torch.tensor([row.valid for row in selected], dtype=torch.bool)
            if not bool(critic.any()):
                continue
            rows = [LearningRow(row.features, row.reset, advance=row.record.proposal is not None,
                                reset_after=row.record.overridden or not row.record.admitted)
                    for row in sequence.transitions]
            output = ordered_window(policy, rows, start, reserve=reserve, stage="ppo", base_policy=base_policy)
            n = len(selected)
            _reserve(reserve, "ppo_critic_samples", int(critic.sum()))
            loss = .5 * F.mse_loss(output.values[:n][critic], returns[index][start:start + n][critic])
            if bool(actor.any()):
                _reserve(reserve, "ppo_actor_samples", int(actor.sum()))
                latent = torch.tensor([row.record.draw.latent if row.record.draw else (0.,) * 5 for row in selected], dtype=torch.float32)
                buttons = torch.tensor([row.record.draw.buttons if row.record.draw else (False, False) for row in selected], dtype=torch.float32)
                old = torch.tensor([row.record.draw.log_probability if row.record.draw and row.record.draw.log_probability is not None else 0.
                                    for row in selected], dtype=torch.float32)
                new = joint_log_probability(latent, output.means[:n], policy.log_std, buttons, output.button_logits[:n])
                ratio = (new[actor] - old[actor]).exp()
                advantage = advantages[index][start:start + n][actor]
                surrogate = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
                normal_entropy = (policy.log_std.clamp(-5, .5) + .5 * math.log(2 * math.pi * math.e)).sum()
                bernoulli_entropy = torch.distributions.Bernoulli(logits=output.button_logits[:n][actor]).entropy().sum(-1).mean()
                loss = loss - surrogate.mean() - .001 * (normal_entropy + bernoulli_entropy)
                ratios.extend(ratio.detach().tolist())
                clips.extend((abs(ratio.detach() - 1) > .2).to(torch.float32).tolist())
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite PPO objective; preserve the failed rollout without updating")
            _reserve(reserve, "ppo_optimizer")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
    policy.eval()
    if policy_state_sha256(base_policy) != base_sha:
        raise ValueError("PPO changed its frozen BC base")
    return {"spec": dict(PPO_SPEC), "optimizer_calls": updates,
            "actor_samples": int(actor_values.numel()),
            "excluded_actor_samples": sum(len(item.transitions) for item in sequences) - int(actor_values.numel()),
            "mean_loss": sum(losses) / len(losses) if losses else None,
            "mean_ratio": sum(ratios) / len(ratios) if ratios else None,
            "clip_fraction": sum(clips) / len(clips) if clips else None,
            "frozen_base_sha256": base_sha}


def ppo_update_v3(policy: JoltPolicy, base_policy: JoltPolicy, optimizer: torch.optim.Optimizer,
                  sequences: Sequence[JoltRolloutSequenceV3], **kwargs) -> dict[str, Any]:
    """Every fixed v3 update consumes exactly 256 confirmed physical decisions."""
    if (not sequences or len(sequences) > 256
            or any(not isinstance(sequence, JoltRolloutSequenceV3) for sequence in sequences)
            or sum(len(sequence.transitions) for sequence in sequences) != 256
            or any(row.actual_ticks != 6 for sequence in sequences for row in sequence.transitions)):
        raise ValueError("v3 PPO requires exactly 256 confirmed six-tick decisions in at most 256 positive episodes")
    result = ppo_update(policy, base_policy, optimizer, sequences, **kwargs)
    result.update(terminal_schema=TERMINAL_SCHEMA_V3,
                  confirmed_decisions=256, episodes=len(sequences),
                  safe_retirements=sum(sequence.terminal_failure for sequence in sequences),
                  new_terminal_penalty=False)
    return result
