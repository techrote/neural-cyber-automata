"""LEARN-JOLT-002 full learned actuation and immutable invocation contracts.

This optional Torch module leaves the historical first-skill implementation
unchanged. Ordinary execution has no scripted controller, launch scheduler or
goal-derived aim. Likelihoods belong to five original Normal latent draws and
two Bernoulli draws, before any deterministic action transformation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nca.contracts import ContractError, WireAction, canonical_json, sha256_json
from nca.data.records import policy_view
from .baseline import FeedbackObservation, SkillGoal
from .recovery_supervisor import PhysicalRecoverySupervisor, Supervision
from .shaping import clamp, wrapped_angle


FEATURE_SCHEMA = "jolt-local-goal-dual-history/v2"
ACTION_SCHEMA = "jolt-five-continuous-two-buttons/v2"
SAMPLE_SCHEMA = "jolt-pretransform-normal-bernoulli/v2"
BUNDLE_FORMAT = "nca-jolt-learned/v2"
HIDDEN_SIZE = 32
EPSILON = 1e-8
RESIDUAL_SCALES = (.5, .4, .6, 1., 1.)
BASE_LOGIT_LIMIT = 4.0
BUTTON_CORRECTION_LIMIT = 8.0
ACTION_NAMES = ("move_x", "trim", "effort", "aim_x", "aim_y", "primary", "secondary", "auxiliary")
FEATURE_NAMES = (
    "target_dx_div64", "target_dy_div64", "vx_div80", "vy_div100",
    "sin_rotation", "cos_rotation", "angular_velocity_div8",
    "grounded", "grounded_valid", "energy", "energy_valid",
    "cooldown", "cooldown_valid", "ground_distance_div96", "ground_distance_valid",
    "ceiling_distance_div96", "ceiling_distance_valid", "time_remaining_fraction",
    "goal_pace", "goal_tolerance_div8", "blocked", "blocked_valid",
    "wall_left", "wall_left_valid", "wall_right", "wall_right_valid",
    "stuck_time_div4", "stuck_time_valid",
    *(f"previous_proposal_{name}" for name in ACTION_NAMES), "previous_proposal_available",
    *(f"previous_dispatched_{name}" for name in ACTION_NAMES), "previous_dispatch_available",
    "previous_intervention", "previous_admitted",
)


def _sha(value: Any) -> str:
    return sha256_json(value)


def _boundary(observation: Mapping[str, Any]) -> str:
    parts = observation.get("parts", ())
    if not isinstance(parts, (tuple, list)) or any(not isinstance(part, Mapping) for part in parts):
        raise ValueError("own-part lifecycle metadata is malformed")
    return _sha({"mount_generation": observation.get("mount_generation"),
                 "parts": [(part.get("id"), part.get("index"), part.get("detached", False))
                           for part in parts]})


@dataclass(frozen=True)
class InvocationContext:
    """Original invocation identity and time, independent of recurrent windows."""

    goal: SkillGoal
    start_tick: int
    deadline_tick: int
    lifetime: str
    control_generation: int
    boundary_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.goal, SkillGoal) or self.goal.skill != "directed_hop":
            raise ValueError("Jolt requires an immutable directed-hop caller goal")
        duration = math.ceil(self.goal.timeout_seconds * 60 - 1e-12)
        if (type(self.start_tick) is not int or self.start_tick < 0
                or type(self.deadline_tick) is not int or self.deadline_tick != self.start_tick + duration):
            raise ValueError("invocation deadline must preserve its original start and duration")
        if not isinstance(self.lifetime, str) or not self.lifetime:
            raise ValueError("invocation lifetime must be present")
        if type(self.control_generation) is not int or self.control_generation < 0:
            raise ValueError("invocation authority generation must be nonnegative")
        if (not isinstance(self.boundary_sha256, str) or len(self.boundary_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.boundary_sha256)):
            raise ValueError("invocation topology needs an exact SHA-256")

    @classmethod
    def create(cls, observation: Mapping[str, Any], goal: SkillGoal) -> "InvocationContext":
        view = FeedbackObservation.from_host(policy_view(observation))
        if view.actor != "jev" or not view.valid or not view.controllable:
            raise ValueError("cannot start an invocation without valid owned Jolt state")
        return cls(goal, view.tick, view.tick + math.ceil(goal.timeout_seconds * 60 - 1e-12),
                   view.lifetime, view.control_generation, _boundary(observation))

    def validate_observation(self, observation: Mapping[str, Any], goal: SkillGoal) -> FeedbackObservation:
        view = FeedbackObservation.from_host(policy_view(observation))
        if (goal != self.goal or view.actor != "jev" or view.lifetime != self.lifetime
                or view.control_generation != self.control_generation or _boundary(observation) != self.boundary_sha256):
            raise ValueError("observation/goal crosses original invocation authority or topology")
        if not view.valid or not view.controllable or view.tick < self.start_tick:
            raise ValueError("unavailable or pre-invocation observation")
        return view

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "jolt-invocation/v2", **asdict(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvocationContext":
        payload = dict(value)
        if payload.pop("schema", None) != "jolt-invocation/v2":
            raise ValueError("unknown invocation contract")
        payload["goal"] = SkillGoal(**payload["goal"])
        return cls(**payload)


@dataclass(frozen=True)
class ActionHistory:
    """Previous proposal and actual dispatch are distinct, including absence."""

    proposal: WireAction | None = None
    dispatched: WireAction | None = None
    intervention: bool = False
    admitted: bool = False

    def __post_init__(self) -> None:
        if any(value is not None and not isinstance(value, WireAction)
               for value in (self.proposal, self.dispatched)):
            raise ValueError("history actions require the bounded wire contract")
        if type(self.intervention) is not bool or type(self.admitted) is not bool:
            raise ValueError("history masks must be booleans")
        if self.admitted and self.dispatched is None:
            raise ValueError("an absent dispatch cannot be admitted")
        if self.proposal is not None and (self.proposal.move[1] != 0 or self.proposal.reel != 0
                or self.proposal.auxiliary or self.proposal.tool or self.proposal.release_usable
                or self.proposal.attach_usable):
            raise ValueError("ordinary proposal history cannot acquire supervisor or undeclared channels")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "jolt-dual-action-history/v2",
                "proposal": None if self.proposal is None else self.proposal.to_dict(),
                "dispatched": None if self.dispatched is None else self.dispatched.to_dict(),
                "intervention": self.intervention, "admitted": self.admitted}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionHistory":
        payload = dict(value)
        if payload.pop("schema", None) != "jolt-dual-action-history/v2":
            raise ValueError("unknown previous-action contract")
        for key in ("proposal", "dispatched"):
            if payload.get(key) is not None:
                payload[key] = WireAction.from_dict(payload[key])
        return cls(**payload)


def action_vector(action: WireAction) -> tuple[float, ...]:
    return (action.move[0], action.trim, action.effort, *action.aim,
            float(action.primary), float(action.secondary), float(action.auxiliary))


def features(observation: Mapping[str, Any], goal: SkillGoal, *, context: InvocationContext,
             history: ActionHistory = ActionHistory()) -> tuple[float, ...]:
    """Whitelisted local features in world axes; constant scales, no fitted state."""
    view = context.validate_observation(observation, goal)

    def optional(value: Any, scale: float = 1.0) -> tuple[float, float]:
        if value is None:
            return 0.0, 0.0
        if not isinstance(value, (int, float, bool)) or not math.isfinite(value) or value < 0:
            raise ValueError("local sensor must be finite, nonnegative or unavailable")
        return clamp(float(value) / scale, 0., 2.), 1.0

    def previous(action: WireAction | None) -> tuple[float, ...]:
        return (*((0.,) * len(ACTION_NAMES) if action is None else action_vector(action)),
                float(action is not None))

    values = (
        clamp((goal.target[0] - view.position[0]) / 64, -2, 2),
        clamp((goal.target[1] - view.position[1]) / 64, -2, 2),
        clamp(view.velocity[0] / 80, -2, 2), clamp(view.velocity[1] / 100, -2, 2),
        math.sin(view.angle), math.cos(view.angle), clamp(view.angular_velocity / 8, -2, 2),
        float(view.supported is True), float(view.supported is not None),
        *optional(view.energy), *optional(view.cooldown), *optional(view.ground_distance, 96),
        *optional(view.ceiling_distance, 96),
        clamp((context.deadline_tick - view.tick) / (context.deadline_tick - context.start_tick), 0., 1.),
        goal.pace, goal.tolerance / 8,
        *optional(observation.get("blocked")), *optional(observation.get("wall_left")),
        *optional(observation.get("wall_right")), *optional(observation.get("stuck_time"), 4.),
        *previous(history.proposal), *previous(history.dispatched),
        float(history.intervention), float(history.admitted),
    )
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in values):
        raise ValueError("invalid policy feature vector")
    return values


def mapped_continuous(latent: Tensor) -> Tensor:
    if latent.shape[-1] != 5:
        raise ValueError("full learned control requires five continuous channels")
    return torch.cat((latent[..., :2].tanh(), latent[..., 2:3].sigmoid(), latent[..., 3:5].tanh()), dim=-1)


def normalized_aim(mapped: Tensor) -> Tensor:
    if mapped.shape[-1] != 2:
        raise ValueError("learned aim requires two coordinates")
    norm = torch.linalg.vector_norm(mapped, dim=-1, keepdim=True)
    fallback = torch.tensor((0., -1.), dtype=mapped.dtype, device=mapped.device)
    return torch.where(norm > EPSILON, mapped / norm.clamp_min(EPSILON), fallback)


def map_action(latent: Sequence[float], buttons: Sequence[bool | float]) -> WireAction:
    if len(latent) != 5 or not all(math.isfinite(float(v)) for v in latent):
        raise ValueError("five finite pre-transform values required")
    if len(buttons) != 2 or any(v not in (False, True, 0., 1.) for v in buttons):
        raise ValueError("two exact Bernoulli decisions required")
    x, trim, effort, aim_x, aim_y = map(float, latent)
    aim = (math.tanh(aim_x), math.tanh(aim_y))
    magnitude = math.hypot(*aim)
    aim = (aim[0] / magnitude, aim[1] / magnitude) if magnitude > EPSILON else (0., -1.)
    sigmoid = 1 / (1 + math.exp(-effort)) if effort >= 0 else math.exp(effort) / (1 + math.exp(effort))
    return WireAction(move=(math.tanh(x), 0.), trim=math.tanh(trim), effort=sigmoid,
                      aim=aim, primary=bool(buttons[0]), secondary=bool(buttons[1]), auxiliary=False)


def combine_residual(base_means: Tensor, base_logits: Tensor, residual_latent: Tensor,
                     residual_button_logits: Tensor) -> tuple[Tensor, Tensor]:
    if (base_means.shape != residual_latent.shape or base_means.shape[-1] != 5
            or base_logits.shape != residual_button_logits.shape or base_logits.shape[-1] != 2
            or base_means.shape[:-1] != base_logits.shape[:-1]):
        raise ValueError("base and residual must agree on five continuous and two button channels")
    scales = torch.as_tensor(RESIDUAL_SCALES, dtype=base_means.dtype, device=base_means.device)
    return (base_means + scales * residual_latent.tanh(),
            base_logits.clamp(-BASE_LOGIT_LIMIT, BASE_LOGIT_LIMIT)
            + BUTTON_CORRECTION_LIMIT * residual_button_logits.tanh())


def joint_log_probability(latent: Tensor, mean: Tensor, log_std: Tensor,
                          buttons: Tensor, button_logits: Tensor) -> Tensor:
    """Density of original Normal5 + Bernoulli2 draws, before every wire map."""
    if (latent.shape != mean.shape or latent.shape[-1] != 5 or buttons.shape != button_logits.shape
            or buttons.shape[-1] != 2 or latent.shape[:-1] != buttons.shape[:-1]):
        raise ValueError("sample likelihood requires five latents and two Bernoulli draws")
    if log_std.shape not in ((5,), mean.shape):
        raise ValueError("Normal standard deviations must identify all five channels")
    if not bool(torch.isfinite(latent).all() and torch.isfinite(mean).all()
                and torch.isfinite(log_std).all() and torch.isfinite(button_logits).all()):
        raise ValueError("nonfinite distribution or draw")
    if not bool(((buttons == 0) | (buttons == 1)).all()):
        raise ValueError("button likelihood requires original binary draws")
    bounded_std = log_std.clamp(-5., .5)
    normal = -.5 * ((latent - mean) / bounded_std.exp()).square() - bounded_std - .5 * math.log(2 * math.pi)
    bernoulli = -F.binary_cross_entropy_with_logits(button_logits, buttons.to(button_logits.dtype), reduction="none")
    return normal.sum(-1) + bernoulli.sum(-1)


def joint_entropy(log_std: Tensor, button_logits: Tensor) -> Tensor:
    normal = (log_std.clamp(-5., .5) + .5 * math.log(2 * math.pi * math.e)).sum(-1)
    probabilities = button_logits.sigmoid()
    bernoulli = F.binary_cross_entropy_with_logits(button_logits, probabilities, reduction="none").sum(-1)
    return normal + bernoulli


class JoltPolicy(nn.Module):
    """One 32-wide GRU with explicit masks, no implicit recurrent state."""

    def __init__(self, kind: str, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        if kind not in ("bc", "residual") or hidden_size != HIDDEN_SIZE:
            raise ValueError("v2 admits only bc/residual with one 32-wide GRU")
        self.kind, self.hidden_size, self.input_size = kind, hidden_size, len(FEATURE_NAMES)
        self.encoder = nn.Linear(self.input_size, hidden_size)
        self.recurrent = nn.GRUCell(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, 7)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((5,), -1.0), requires_grad=kind == "residual")
        if kind == "residual":
            nn.init.zeros_(self.actor.weight)
        else:
            nn.init.orthogonal_(self.actor.weight, .01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.)
        nn.init.zeros_(self.critic.bias)

    def zero_hidden(self, batch: int = 1) -> Tensor:
        return torch.zeros((batch, self.hidden_size), dtype=self.encoder.weight.dtype, device=self.encoder.weight.device)

    def forward(self, inputs: Tensor, hidden: Tensor, reset: Tensor | None = None,
                valid: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_size or hidden.shape != (len(inputs), HIDDEN_SIZE):
            raise ValueError("feature/hidden shapes differ from the declared architecture")
        for mask in (reset, valid):
            if mask is not None and (mask.shape != (len(inputs),) or mask.dtype != torch.bool):
                raise ValueError("recurrent masks require one boolean per sequence")
        effective = torch.ones(len(inputs), dtype=torch.bool, device=inputs.device) if valid is None else valid
        original = hidden
        if reset is not None:
            hidden = torch.where((reset & effective).unsqueeze(-1), torch.zeros_like(hidden), hidden)
        # Padding neither executes a state update nor clears prior history.
        selected = torch.nonzero(effective, as_tuple=False).flatten()
        if len(selected):
            advanced = self.recurrent(torch.tanh(self.encoder(inputs[selected])), hidden[selected])
            hidden = hidden.index_copy(0, selected, advanced)
        hidden = torch.where(effective.unsqueeze(-1), hidden, original)
        logits, value = self.actor(hidden), self.critic(hidden).squeeze(-1)
        return logits[..., :5], logits[..., 5:], value, hidden

    def specification(self) -> dict[str, Any]:
        return {"version": 2, "kind": self.kind, "hidden_size": HIDDEN_SIZE,
                "feature_schema": FEATURE_SCHEMA, "feature_names": list(FEATURE_NAMES),
                "action_schema": ACTION_SCHEMA, "sample_schema": SAMPLE_SCHEMA,
                "continuous_names": list(ACTION_NAMES[:5]), "button_names": ["primary", "secondary"],
                "continuous_transform": "tanh move/trim/aim; sigmoid effort; unit aim",
                "zero_aim": {"epsilon": EPSILON, "vector": [0., -1.]},
                "button_threshold": "logit >= 0; native held/released edges unchanged",
                "base_logit_limit": BASE_LOGIT_LIMIT, "residual_scales": list(RESIDUAL_SCALES),
                "button_correction_limit": BUTTON_CORRECTION_LIMIT,
                "likelihood": "original Normal5 and Bernoulli2 draws; no transformed-wire likelihood",
                "log_std_bounds": [-5., .5], "initial_log_std": -1.,
                "decision_ticks": 6, "base_hz": 60, "actor": "jev", "profile_version": 1,
                "frame": "world_axes; no goal reflection or scripted aim",
                "history": "previous ordinary proposal and actual dispatch with availability/intervention/admission masks",
                "invocation_time": "immutable original start/deadline, independent of segment and intervention",
                "padding": "does not advance or reset recurrence",
                "residual_base": "same-seed frozen BC-final with separate recurrent state" if self.kind == "residual" else None}


def emergency_request(observation: FeedbackObservation) -> str | None:
    """The qualified teacher's shared observed emergency predicate, unchanged."""
    if (abs(wrapped_angle(observation.angle)) >= PhysicalRecoverySupervisor.EMERGENCY_ANGLE
            or abs(observation.angular_velocity) >= PhysicalRecoverySupervisor.EMERGENCY_ANGULAR_SPEED):
        return "unstable_attitude"
    return None


@dataclass(frozen=True)
class JoltDecision:
    action: WireAction
    proposal: WireAction | None
    supervision: Supervision
    features: tuple[float, ...]
    context: InvocationContext
    history: ActionHistory
    hidden_before: tuple[float, ...]
    base_hidden_before: tuple[float, ...] | None
    value: float | None
    latent: tuple[float, ...] | None
    buttons: tuple[bool, bool] | None
    log_probability: float | None
    sampling_mode: str
    observation_id: str
    observation_sha256: str
    repeat_ticks: int
    policy_sha256: str
    base_sha256: str | None

    @property
    def replaced(self) -> bool:
        return self.supervision.override is not None


class SupervisorRetirement(ContractError):
    """Stable v2 execution diagnostic; the original authority stop is unchanged."""

    REASONS = frozenset({
        "emergency_recovery_deadline", "skill_deadline", "owner_generation_changed",
        "unavailable_observation", "expired_observation", "observation_gap",
        "canceled", "caller_confirmed_complete", "supported_arrival_estimate",
    })

    def __init__(self, reason: str):
        self.reason = reason if reason in self.REASONS else "other_supervisor_retirement"
        super().__init__("jolt_supervisor_retired", "physical supervision retired this invocation")


class JoltController:
    """Owned sequential inference with shared supervision and explicit history."""

    def __init__(self, policy: JoltPolicy, *, base_policy: JoltPolicy | None = None,
                 generator: torch.Generator | None = None,
                 reserve: Callable[[str, int], None] | None = None,
                 structured_retirement: bool = False):
        if not isinstance(policy, JoltPolicy):
            raise ValueError("v2 runtime requires the v2 model contract")
        if (policy.kind == "residual") != (base_policy is not None):
            raise ValueError("only residual policies require a frozen BC base")
        if base_policy is not None and base_policy.kind != "bc":
            raise ValueError("the residual base must be a learned BC policy")
        self.policy = copy.deepcopy(policy).eval().requires_grad_(False)
        self.base_policy = None if base_policy is None else copy.deepcopy(base_policy).eval().requires_grad_(False)
        self.generator = generator
        self.reserve = reserve
        if type(structured_retirement) is not bool:
            raise ValueError("structured retirement is an explicit execution-version choice")
        self.structured_retirement = structured_retirement
        self.policy_sha256 = policy_state_sha256(policy)
        self.base_sha256 = None if self.base_policy is None else policy_state_sha256(self.base_policy)
        self.context: InvocationContext | None = None
        self._retired: set[tuple[str, str]] = set()
        self._retired_lifetimes: set[str] = set()
        self._pending: JoltDecision | None = None
        self._last_tick: int | None = None
        self._last_current_tick: int | None = None
        self._next_observation_tick: int | None = None
        self.supervisor = PhysicalRecoverySupervisor()
        self.history = ActionHistory()
        self.reset_recurrence()

    def reset_recurrence(self) -> None:
        """Segment cuts do not alter invocation time, history or authority."""
        self.hidden = self.policy.zero_hidden()
        self.base_hidden = None if self.base_policy is None else self.base_policy.zero_hidden()

    def clone_state(self) -> "JoltController":
        """Fork recurrent state for bootstrap without duplicating budget ownership."""
        return copy.deepcopy(self, {} if self.reserve is None else {id(self.reserve): self.reserve})

    def start(self, observation: Mapping[str, Any], goal: SkillGoal) -> InvocationContext:
        next_context = InvocationContext.create(observation, goal)
        key = (next_context.lifetime, goal.id)
        if key in self._retired or next_context.lifetime in self._retired_lifetimes:
            raise ValueError("retired invocation or physical lifetime cannot restart")
        if self.context is not None:
            if self.context == next_context:
                raise ValueError("an active invocation cannot be restarted to renew time")
            self.cancel("replaced_invocation")
            if self.context.lifetime != next_context.lifetime:
                self._retired_lifetimes.add(self.context.lifetime)
            elif self.context.goal.id == goal.id:
                raise ValueError("same-lifetime goal identity cannot be reused")
        self.context = next_context
        self.supervisor = PhysicalRecoverySupervisor()
        self.history, self._pending = ActionHistory(), None
        self._last_tick = self._last_current_tick = None
        self._next_observation_tick = None
        self.reset_recurrence()
        return next_context

    def cancel(self, reason: str = "canceled") -> WireAction:
        if self.context is not None:
            self._retired.add((self.context.lifetime, self.context.goal.id))
        self.supervisor.cancel(reason)
        self._pending = None
        self.reset_recurrence()
        return WireAction(effort=0.)

    def complete(self) -> WireAction:
        """Only the caller may confirm per-tick outcome and revoke native intent."""
        return self.cancel("caller_confirmed_complete")

    @torch.no_grad()
    def step(self, observation: Mapping[str, Any], goal: SkillGoal, *, deterministic: bool = True,
             current_tick: int | None = None) -> JoltDecision:
        if self.context is None:
            self.start(observation, goal)
        assert self.context is not None
        if self._pending is not None:
            raise ValueError("record the actual prior dispatch before advancing history")
        if (self.context.lifetime, self.context.goal.id) in self._retired:
            raise ValueError("retired invocation cannot resume")
        try:
            view = self.context.validate_observation(observation, goal)
            now = view.tick if current_tick is None else current_tick
            if type(now) is not int or now < view.tick:
                raise ValueError("current simulation clock precedes observation")
            if self._last_tick is not None and (view.tick <= self._last_tick
                    or self._next_observation_tick is None or view.tick > self._next_observation_tick):
                raise ValueError("stale, unordered or missing observation history")
            if self._last_tick is None and view.tick != self.context.start_tick:
                raise ValueError("the first decision must bind the invocation's original observation")
            if self._last_current_tick is not None and now < self._last_current_tick:
                raise ValueError("current clock rollback")
            if now >= self.context.deadline_tick or now >= view.tick + 12:
                raise ValueError("original invocation deadline or observation lease expired")
            supervision = self.supervisor.admit(view, goal, current_tick=now, emergency=emergency_request(view))
            if supervision.status != "running":
                if self.structured_retirement:
                    self.cancel(supervision.reason)
                    raise SupervisorRetirement(supervision.reason)
                raise ValueError(f"physical supervisor retired invocation: {supervision.reason}")
            encoded = features(observation, goal, context=self.context, history=self.history)
        except SupervisorRetirement:
            # The same invocation is already retired. Do not replace its
            # declared reason with a generic malformed-context diagnostic.
            raise
        except (ValueError, TypeError):
            self.cancel("invalid_context")
            raise
        self._last_tick, self._last_current_tick = view.tick, now
        before = tuple(float(v) for v in self.hidden[0])
        base_before = None if self.base_hidden is None else tuple(float(v) for v in self.base_hidden[0])
        proposal = None
        latent_values = buttons_values = score = value_number = None
        if supervision.override is not None:
            action = supervision.override
            sampling_mode = "supervisor_without_proposal"
            self.reset_recurrence()
        else:
            if policy_state_sha256(self.policy) != self.policy_sha256 or (
                    self.base_policy is not None and policy_state_sha256(self.base_policy) != self.base_sha256):
                self.cancel("policy_content_changed")
                raise ValueError("executing policy or frozen BC base changed after admission")
            inputs = torch.tensor([encoded], dtype=self.policy.encoder.weight.dtype,
                                  device=self.policy.encoder.weight.device)
            if self.reserve is not None:
                self.reserve("inference_forward", 1)
            mean, logits, value, self.hidden = self.policy(inputs, self.hidden)
            latent = mean
            if not deterministic:
                latent = mean + self.policy.log_std.clamp(-5., .5).exp() * torch.randn(
                    mean.shape, generator=self.generator, dtype=mean.dtype, device=mean.device)
            if self.base_policy is not None:
                if self.reserve is not None:
                    self.reserve("inference_forward", 1)
                base_mean, base_logits, _, self.base_hidden = self.base_policy(inputs, self.base_hidden)
                combined, logits = combine_residual(base_mean, base_logits, latent, logits)
            else:
                combined, logits = latent, logits.clamp(-BASE_LOGIT_LIMIT, BASE_LOGIT_LIMIT)
            buttons = (logits >= 0).to(logits.dtype) if deterministic else torch.bernoulli(logits.sigmoid(), generator=self.generator)
            if not deterministic:
                score = float(joint_log_probability(latent, mean, self.policy.log_std, buttons, logits)[0])
            proposal = action = map_action(combined[0].tolist(), buttons[0].tolist())
            latent_values = tuple(float(v) for v in latent[0])
            buttons_values = tuple(bool(v) for v in buttons[0])
            value_number = float(value[0])
            sampling_mode = "deterministic_mean" if deterministic else "stochastic_draw"
        repeat = min(6, view.tick + supervision.action_lease_ticks - now, self.context.deadline_tick - now)
        if repeat <= 0:
            self.cancel("empty_dispatch_lease")
            raise ValueError("no original lease remains for dispatch")
        # A recurrent state cannot bridge an omitted decision observation.
        # Explicit delayed dispatch retains its original source clock; its
        # expected response is anchored to the supplied dispatch clock.
        self._next_observation_tick = now + repeat
        result = JoltDecision(action, proposal, supervision, encoded, self.context, self.history,
                              before, base_before, value_number, latent_values, buttons_values, score,
                              sampling_mode, observation["observation_id"], _sha(observation), repeat,
                              self.policy_sha256, self.base_sha256)
        self._pending = result
        return result

    def record_dispatch(self, decision: JoltDecision, action: WireAction | None, *, admitted: bool,
                        intervention: bool | None = None) -> ActionHistory:
        if decision is not self._pending:
            raise ValueError("dispatch must bind the exact pending source decision")
        changed = action != decision.action
        overridden = decision.replaced if intervention is None else intervention
        if decision.replaced and not overridden:
            raise ValueError("supervisor replacement cannot be relabelled as ordinary history")
        if changed and not overridden:
            raise ValueError("a substituted or missing command must be marked as intervention")
        self.history = ActionHistory(decision.proposal, action, overridden, admitted)
        self._pending = None
        if overridden or not admitted:
            self.reset_recurrence()
        if not admitted:
            self.cancel("native_admission_failed")
        return self.history


def policy_state_sha256(policy: JoltPolicy) -> str:
    digest = hashlib.sha256(canonical_json(policy.specification()).encode())
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _bundle_model(policy: JoltPolicy) -> dict[str, Any]:
    return {"specification": policy.specification(), "state_dict": policy.state_dict(),
            "policy_sha256": policy_state_sha256(policy)}


def save_policy(path: str | Path, policy: JoltPolicy, *, base_policy: JoltPolicy | None = None,
                metadata: Mapping[str, Any] | None = None) -> dict[str, str | None]:
    if (policy.kind == "residual") != (base_policy is not None) or (base_policy is not None and base_policy.kind != "bc"):
        raise ValueError("residual checkpoint requires exactly one learned BC base")
    for model in (policy, base_policy):
        if model is not None and any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
            raise ValueError("cannot publish nonfinite policy weights")
    payload = {"format": BUNDLE_FORMAT, **_bundle_model(policy),
               "base": None if base_policy is None else _bundle_model(base_policy),
               "metadata": dict(metadata or {}),
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    # No checkpoint selection or accidental replacement of an earlier stage.
    with location.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return {"file_sha256": hashlib.sha256(location.read_bytes()).hexdigest(),
            "policy_sha256": payload["policy_sha256"],
            "base_sha256": None if base_policy is None else policy_state_sha256(base_policy)}


def _load_model(payload: Mapping[str, Any]) -> JoltPolicy:
    spec = payload.get("specification", {})
    policy = JoltPolicy(spec.get("kind"), spec.get("hidden_size"))
    if spec != policy.specification():
        raise ValueError("checkpoint feature/action/sample/recurrence contract differs")
    policy.load_state_dict(payload["state_dict"], strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in policy.state_dict().values()):
        raise ValueError("checkpoint contains nonfinite weights")
    if payload.get("policy_sha256") != policy_state_sha256(policy):
        raise ValueError("checkpoint state does not match its content identity")
    return policy.eval()


def load_policy(path: str | Path, *, expected_config_sha256: str | None = None,
                expected_base_sha256: str | None = None) -> tuple[JoltPolicy, JoltPolicy | None, dict[str, Any]]:
    location = Path(path)
    if location.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("learned bundle exceeds 32 MiB")
    payload = torch.load(location, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != BUNDLE_FORMAT:
        raise ValueError("unknown learned bundle format; historical v1 is separate")
    if payload.get("source_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("checkpoint executing controller source differs")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a mapping")
    if expected_config_sha256 is not None and metadata.get("config_sha256") != expected_config_sha256:
        raise ValueError("checkpoint belongs to another frozen experiment")
    policy = _load_model(payload)
    base = None if payload.get("base") is None else _load_model(payload["base"])
    if (policy.kind == "residual") != (base is not None) or (base is not None and base.kind != "bc"):
        raise ValueError("checkpoint has the wrong learned-base relationship")
    if expected_base_sha256 is not None and (base is None or policy_state_sha256(base) != expected_base_sha256):
        raise ValueError("residual checkpoint replaced its frozen BC base")
    if base is not None:
        base.requires_grad_(False)
    return policy, base, metadata
