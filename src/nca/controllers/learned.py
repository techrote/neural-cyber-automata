"""Small fixed-weight recurrent Jolt policies over the accepted wire boundary.

The residual distribution is defined before the combined wire request is
clipped. Its stored likelihood always belongs to the sampled residual, never
to an actuator impulse or a replacement/fallback request.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nca.contracts import WireAction
from nca.data.records import policy_view
from .baseline import AuthoredController, FeedbackObservation, SkillGoal
from .shaping import clamp, unit


FEATURE_SCHEMA = "jolt-local-goal-history/v1"
ACTION_SCHEMA = "jolt-six-wire-channels/v1"
RESIDUAL_SCHEMA = "jolt-scaled-tanh-residual/v1"
FEATURE_NAMES = (
    "target_dx_div64", "target_dy_div64", "vx_div80", "vy_div100",
    "sin_rotation", "cos_rotation", "angular_velocity_div8",
    "grounded", "grounded_valid", "energy", "energy_valid",
    "cooldown", "cooldown_valid", "ground_distance_div96", "ground_distance_valid",
    "ceiling_distance_div96", "ceiling_distance_valid", "time_remaining_fraction",
    "goal_pace", "goal_tolerance_div8", "previous_move_x", "previous_trim",
    "previous_effort", "previous_primary", "previous_secondary", "previous_auxiliary",
)
ACTION_NAMES = ("move_x", "trim", "effort", "primary", "secondary", "auxiliary")
RESIDUAL_NAMES = ("move_x", "trim", "effort")
RESIDUAL_SCALES = (.25, .20, .15)
HIDDEN_SIZE = 32


def action_vector(action: WireAction | Mapping[str, Any]) -> tuple[float, ...]:
    value = action if isinstance(action, WireAction) else WireAction.from_dict(dict(action))
    return (value.move[0], value.trim, value.effort, float(value.primary),
            float(value.secondary), float(value.auxiliary))


def features(observation: Mapping[str, Any], goal: SkillGoal, *,
             previous: WireAction | None = None, start_tick: int = 0,
             baseline: WireAction | None = None) -> tuple[float, ...]:
    """An explicit whitelist and caller goal are the complete policy interface.

    Optional missing sensors carry a separate validity bit. Motion/target
    scales are game cells and seconds; none are inferred SI units. The
    residual candidate also receives its own authored reference request.
    """
    view = FeedbackObservation.from_host(policy_view(observation))
    if view.actor != "jev" or goal.skill != "directed_hop" or not view.valid:
        raise ValueError("learned v1 features require a valid Jolt directed-hop view")
    if type(start_tick) is not int or not 0 <= start_tick <= view.tick:
        raise ValueError("goal start must identify an earlier completed tick")

    def optional(value, scale=1.0):
        return (0.0, 0.0) if value is None else (clamp(float(value) / scale, 0, 2), 1.0)

    values = (
        clamp((goal.target[0] - view.position[0]) / 64, -2, 2),
        clamp((goal.target[1] - view.position[1]) / 64, -2, 2),
        clamp(view.velocity[0] / 80, -2, 2), clamp(view.velocity[1] / 100, -2, 2),
        math.sin(view.angle), math.cos(view.angle), clamp(view.angular_velocity / 8, -2, 2),
        float(view.supported is True), float(view.supported is not None),
        *optional(view.energy), *optional(view.cooldown),
        *optional(view.ground_distance, 96), *optional(view.ceiling_distance, 96),
        max(0.0, 1.0 - (view.tick - start_tick) / (60 * goal.timeout_seconds)),
        goal.pace, goal.tolerance / 8,
        *action_vector(previous or WireAction(effort=0.0)),
    )
    if baseline is not None:
        values += action_vector(baseline)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite policy feature")
    return values


def mapped_aim(observation: Mapping[str, Any], goal: SkillGoal) -> tuple[float, float]:
    """The same declared directed-hop aiming map used by the authored teacher."""
    view = FeedbackObservation.from_host(policy_view(observation))
    dx, dy = goal.target[0] - view.position[0], goal.target[1] - view.position[1]
    return unit((clamp(dx / 30), min(-.55, dy / 30 - .55)))


def clone_request(output: Sequence[float], observation: Mapping[str, Any], goal: SkillGoal) -> WireAction:
    if len(output) != 6 or not all(math.isfinite(float(value)) for value in output):
        raise ValueError("cloned request needs six finite mapped outputs")
    return WireAction(move=(clamp(float(output[0])), 0.0), aim=mapped_aim(observation, goal),
                      trim=clamp(float(output[1])), effort=clamp(float(output[2]), 0, 1),
                      primary=output[3] >= .5, secondary=output[4] >= .5, auxiliary=output[5] >= .5)


def combine_residual(baseline: WireAction, residual: Sequence[float]) -> WireAction:
    if len(residual) != 3 or any(not math.isfinite(float(value)) or abs(value) > scale + 1e-6
                                 for value, scale in zip(residual, RESIDUAL_SCALES)):
        raise ValueError("residual exceeds its declared per-channel authority")
    original = baseline.to_dict()
    original.update(move=[clamp(baseline.move[0] + float(residual[0])), 0.0],
                    trim=clamp(baseline.trim + float(residual[1])),
                    effort=clamp(baseline.effort + float(residual[2]), 0, 1))
    return WireAction.from_dict(original)


def residual_log_probability(latent: Tensor, mean: Tensor, log_std: Tensor,
                             scales: Sequence[float] = RESIDUAL_SCALES) -> Tensor:
    """Stable density for scale*tanh(z), evaluated at the original Gaussian z.

    Retaining z avoids numerical inversion of a saturated tanh. The final
    combined wire request can clip; that deterministic many-to-one host input
    map does not change the policy's declared sampled-residual action space.
    """
    if latent.shape[-1] != 3 or mean.shape[-1] != 3 or len(scales) != 3:
        raise ValueError("residual density requires exactly three channels")
    if any(not math.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("residual scales must be finite and positive")
    bounded_std = log_std.clamp(-5, .5)
    gaussian = -.5 * ((latent - mean) / bounded_std.exp()).square() - bounded_std - .5 * math.log(2 * math.pi)
    log_jacobian = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
    log_scale = torch.as_tensor(scales, dtype=latent.dtype, device=latent.device).log()
    return (gaussian - log_jacobian - log_scale).sum(dim=-1)


class RecurrentPolicy(nn.Module):
    """One GRU cell with explicit external hidden state and no implicit history."""

    def __init__(self, kind: str, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        if kind not in ("bc", "residual") or hidden_size != HIDDEN_SIZE:
            raise ValueError("v1 admits bc/residual policies with a 32-wide GRU")
        self.kind, self.hidden_size = kind, hidden_size
        self.input_size = len(FEATURE_NAMES) + (6 if kind == "residual" else 0)
        self.encoder = nn.Linear(self.input_size, hidden_size)
        self.recurrent = nn.GRUCell(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, 6 if kind == "bc" else 3)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((3,), -1.0), requires_grad=kind == "residual")
        nn.init.orthogonal_(self.actor.weight, .01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.zeros_(self.critic.bias)

    def zero_hidden(self, batch: int = 1) -> Tensor:
        return torch.zeros((batch, self.hidden_size), dtype=self.encoder.weight.dtype,
                           device=self.encoder.weight.device)

    def forward(self, inputs: Tensor, hidden: Tensor, reset: Tensor | None = None):
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_size or hidden.shape != (len(inputs), self.hidden_size):
            raise ValueError("recurrent feature/hidden shapes differ from the bundle")
        if reset is not None:
            if reset.shape != (len(inputs),):
                raise ValueError("reset mask must contain one value per sequence")
            hidden = hidden * (~reset.bool()).to(hidden.dtype).unsqueeze(-1)
        hidden = self.recurrent(torch.tanh(self.encoder(inputs)), hidden)
        return self.actor(hidden), self.critic(hidden).squeeze(-1), hidden

    @staticmethod
    def clone_outputs(logits: Tensor) -> Tensor:
        return torch.cat((logits[..., :2].tanh(), logits[..., 2:3].sigmoid(),
                          logits[..., 3:].sigmoid()), dim=-1)

    def specification(self) -> dict[str, Any]:
        return {"version": 1, "kind": self.kind, "hidden_size": self.hidden_size,
                "feature_schema": FEATURE_SCHEMA, "feature_names": list(FEATURE_NAMES),
                "residual_reference_names": list(ACTION_NAMES) if self.kind == "residual" else [],
                "action_schema": ACTION_SCHEMA, "residual_schema": RESIDUAL_SCHEMA,
                "residual_scales": list(RESIDUAL_SCALES), "decision_ticks": 6,
                "base_hz": 60, "actor": "jev", "profile_version": 1,
                "hidden_reset": "zero_on_episode_actor_profile_mount_generation_goal_or_source_change",
                "state_update": "one_gru_update_per_completed_decision_observation"}


@dataclass(frozen=True)
class LearnedDecision:
    action: WireAction
    features: tuple[float, ...]
    hidden_before: tuple[float, ...]
    value: float
    latent: tuple[float, ...] | None = None
    residual: tuple[float, ...] | None = None
    log_probability: float | None = None
    baseline: WireAction | None = None
    replaced: bool = False
    phase: str = "learned"
    sampling_mode: Literal["deterministic_wire", "stochastic_draw", "deterministic_mean", "reference_replacement"] = "deterministic_wire"


class LearnedController:
    """Sequential inference with explicit actor/goal ownership fences."""

    def __init__(self, policy: RecurrentPolicy, *, generator: torch.Generator | None = None):
        self.policy = policy
        self.generator = generator
        self.baseline = AuthoredController("jev") if policy.kind == "residual" else None
        self.reset()

    def reset(self) -> None:
        self.hidden = self.policy.zero_hidden()
        self.previous = WireAction(effort=0.0)
        self.identity = None
        self.last_tick = None
        self.start_tick = 0
        self._canceled_identity = None
        self._lifetime = None
        self._goal = None
        self._boundary_signature = None
        self._retired: dict[str, SkillGoal] = {}
        if self.baseline is not None:
            self.baseline.reset()

    def cancel(self) -> WireAction:
        """Owner cancellation must also revoke the outstanding host lease."""
        self._canceled_identity = self.identity
        if self._goal is not None:
            self._retired[self._goal.id] = self._goal
        self.hidden = self.policy.zero_hidden()
        self.previous = WireAction(effort=0.0)
        if self.baseline is not None:
            self.baseline.cancel()
        return self.previous

    def _prepare(self, observation: Mapping[str, Any], goal: SkillGoal):
        view = FeedbackObservation.from_host(policy_view(observation))
        if view.actor != "jev" or not view.valid or not view.controllable:
            raise ValueError("learned controller cannot act on invalid, suspended or other-rig views")
        if self._lifetime != view.lifetime:
            self.reset()
            self._lifetime = view.lifetime
        if self._goal is not None and self._goal.id == goal.id and self._goal != goal:
            raise ValueError("the same goal identity cannot change its parameters")
        if goal.id in self._retired:
            raise ValueError("retired invocation requires a fresh goal identity")
        if self.last_tick is not None and (view.tick <= self.last_tick or view.tick - self.last_tick > 60):
            self.cancel()
            raise ValueError("learned observations must be ordered without a stale feedback gap")
        parts = observation.get("parts", ())
        if not isinstance(parts, (tuple, list)) or any(not isinstance(part, Mapping) for part in parts):
            raise ValueError("own-part lifecycle metadata is malformed")
        signature = (view.control_generation, observation.get("mount_generation"),
                     tuple((part.get("id"), part.get("index"), part.get("detached", False)) for part in parts))
        changed = self._boundary_signature is not None and signature != self._boundary_signature
        self._boundary_signature = signature
        if changed:
            previous_goal = self._goal
            self.cancel()
            if previous_goal is not None and previous_goal.id == goal.id:
                self.last_tick = view.tick
                raise ValueError("owner, part or mount change retires the old learned invocation")
        if self._goal is None or self._goal.id != goal.id:
            if len(self._retired) >= 1024:
                raise ValueError("goal identity capacity reached; reset the episode")
            if self._goal is not None:
                self._retired[self._goal.id] = self._goal
            self._goal = goal
            self.hidden = self.policy.zero_hidden()
            self.previous = WireAction(effort=0.0)
            self.start_tick = view.tick
            self._canceled_identity = None
        self.identity = (view.lifetime, view.control_generation, goal)
        if view.tick - self.start_tick >= round(goal.timeout_seconds * 60):
            self.cancel()
            raise ValueError("learned skill deadline has elapsed")
        self.last_tick = view.tick
        reference = self.baseline.step(view, goal) if self.baseline is not None else None
        encoded = features(observation, goal, previous=self.previous, start_tick=self.start_tick,
                           baseline=None if reference is None else reference.action)
        return view, reference, encoded

    @torch.no_grad()
    def step(self, observation: Mapping[str, Any], goal: SkillGoal, *, deterministic: bool = True) -> LearnedDecision:
        _, reference, encoded = self._prepare(observation, goal)
        before = tuple(float(value) for value in self.hidden[0])
        inputs = torch.tensor([encoded], dtype=self.policy.encoder.weight.dtype)
        output, value, self.hidden = self.policy(inputs, self.hidden)
        if self.policy.kind == "bc":
            action = clone_request(self.policy.clone_outputs(output)[0].tolist(), observation, goal)
            decision = LearnedDecision(action, encoded, before, float(value[0]))
        elif reference.status != "running":
            # This deterministic reference replacement has no residual sample
            # and is explicitly ineligible for an on-policy actor update.
            decision = LearnedDecision(reference.action, encoded, before, float(value[0]),
                                       baseline=reference.action, replaced=True, phase=reference.phase,
                                       sampling_mode="reference_replacement")
        else:
            latent = output if deterministic else output + self.policy.log_std.clamp(-5, .5).exp() * torch.randn(
                output.shape, generator=self.generator, dtype=output.dtype)
            residual = latent.tanh() * torch.tensor(RESIDUAL_SCALES, dtype=latent.dtype)
            # A chosen mean is a deterministic request, not a draw from the
            # Normal residual law. Its density at that point cannot be reused
            # as the behaviour-policy likelihood of an on-policy sample.
            score = None if deterministic else float(residual_log_probability(latent, output, self.policy.log_std)[0])
            action = combine_residual(reference.action, residual[0].tolist())
            decision = LearnedDecision(action, encoded, before, float(value[0]), tuple(latent[0].tolist()),
                                       tuple(residual[0].tolist()), score, reference.action,
                                       phase=reference.phase,
                                       sampling_mode="deterministic_mean" if deterministic else "stochastic_draw")
        self.previous = decision.action
        return decision


def policy_state_sha256(policy: RecurrentPolicy) -> str:
    """Content identity independent of checkpoint ZIP timestamps and file paths."""
    digest = hashlib.sha256()
    digest.update(str(policy.specification()).encode())
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_policy(path: str | Path, *, expected_config_sha256: str | None = None) -> tuple[RecurrentPolicy, dict[str, Any]]:
    """Load a bounded weights-only CPU bundle; no arbitrary pickle execution."""
    location = Path(path)
    if location.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("first-skill checkpoint exceeds the 32 MiB bundle limit")
    payload = torch.load(location, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "nca-first-skill/v1":
        raise ValueError("unknown learned bundle format")
    specification = payload.get("specification", {})
    policy = RecurrentPolicy(specification.get("kind"), specification.get("hidden_size"))
    if specification != policy.specification():
        raise ValueError("bundle feature/action/recurrence contract differs")
    metadata = payload.get("metadata", {})
    if expected_config_sha256 is not None and metadata.get("config_sha256") != expected_config_sha256:
        raise ValueError("checkpoint belongs to a different frozen experiment")
    if policy.kind == "residual":
        baseline_file = Path(__file__).with_name("baseline.py")
        if metadata.get("baseline_source_sha256") != hashlib.sha256(baseline_file.read_bytes()).hexdigest():
            raise ValueError("residual checkpoint's authored environment has changed")
    policy.load_state_dict(payload["state_dict"], strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in policy.state_dict().values()):
        raise ValueError("checkpoint contains nonfinite weights")
    if payload.get("policy_sha256") != policy_state_sha256(policy):
        raise ValueError("checkpoint weight content differs from its identity")
    policy.eval()
    return policy, metadata
