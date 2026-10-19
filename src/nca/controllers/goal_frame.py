"""A declared Jolt BC coordinate experiment, without a host-symmetry claim.

The frame is fixed by the explicit caller goal and the initial own-root
observation. No evaluator value or new observation channel enters the policy.
The accepted world-frame learner and its checkpoint reader remain unchanged.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from nca.contracts import ContractError, WireAction
from .baseline import SkillGoal
from .learned import (FEATURE_NAMES, LearnedController, LearnedDecision,
                      RecurrentPolicy, policy_state_sha256)

FAMILY_FRAMES = {"direction_frame_bc": "initial_goal_direction", "world_frame_bc": "world"}
SIGNED_FEATURES = (0, 2, 4, 6, 20, 21)


def _sign(value: int) -> int:
    if type(value) is not int or value not in (-1, 1):
        raise ContractError("goal_frame_direction", "a fixed integer direction of -1 or +1 is required")
    return value


def initial_direction(displacement: float) -> int:
    if type(displacement) not in (int, float) or not math.isfinite(displacement):
        raise ContractError("goal_frame_direction", "finite initial caller displacement required")
    return -1 if displacement < 0 else 1


def transform_features(values: Sequence[float], direction: int) -> tuple[float, ...]:
    """An involution on the six declared signed features; masks stay intact."""
    _sign(direction)
    if (len(values) != len(FEATURE_NAMES)
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
        raise ContractError("goal_frame_features", "exactly 26 finite original features required")
    return tuple(float(value) * direction if index in SIGNED_FEATURES else float(value)
                 for index, value in enumerate(values))


def transform_labels(values: Sequence[float], direction: int) -> tuple[float, ...]:
    """Only move X and trim change sign in the six-channel teacher labels."""
    _sign(direction)
    if (len(values) != 6
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
        raise ContractError("goal_frame_labels", "six finite original wire labels required")
    return tuple(float(value) * direction if index < 2 else float(value)
                 for index, value in enumerate(values))


def restore_request(action: WireAction, direction: int) -> WireAction:
    """Return actual world-frame input; the shared world-goal aim is unchanged."""
    _sign(direction)
    if not isinstance(action, WireAction):
        raise ContractError("goal_frame_action", "typed bounded wire request required")
    return replace(action, move=(direction * action.move[0], action.move[1]), trim=direction * action.trim)


def transform_sequences(episodes, family: str):
    """Preserve the same complete ordered training samples for both candidates."""
    if family not in FAMILY_FRAMES:
        raise ContractError("goal_frame_family", "unknown declared candidate family")
    transformed = []
    for rows, labels in episodes:
        if not rows or len(rows) != len(labels):
            raise ContractError("goal_frame_sequence", "complete paired recurrent training sequences required")
        direction = initial_direction(rows[0][0]) if FAMILY_FRAMES[family] == "initial_goal_direction" else 1
        transformed.append(([transform_features(row, direction) for row in rows],
                            [transform_labels(label, direction) for label in labels]))
    return transformed


class GoalFramePolicy(RecurrentPolicy):
    """The same 32-GRU arithmetic with a separately hashed input interpretation."""

    def __init__(self, family: str):
        if family not in FAMILY_FRAMES:
            raise ContractError("goal_frame_family", "unknown declared candidate family")
        self.family, self.frame = family, FAMILY_FRAMES[family]
        super().__init__("bc")

    def specification(self) -> dict[str, Any]:
        base = super().specification()
        base.update(experiment_schema="nca-jolt-goal-frame-policy/v1", family=self.family,
                    feature_frame=self.frame,
                    frame_origin="explicit caller target X minus initial own-root X; fixed per invocation",
                    coincident_horizontal_target_direction=1,
                    signed_feature_names=[FEATURE_NAMES[index] for index in SIGNED_FEATURES],
                    signed_teacher_channels=["move_x", "trim"],
                    wire_restore="fixed direction multiplies move_x and trim; original world-goal aim retained",
                    physical_symmetry_claim=False)
        if self.frame == "initial_goal_direction":
            base["feature_schema"] = "jolt-initial-goal-direction-history/v1"
            base["action_schema"] = "jolt-initial-goal-direction-six-channels/v1"
        return base


class GoalFrameController(LearnedController):
    """Reuse the original lifetime/goal fences and restore actual wire history."""

    def __init__(self, policy: GoalFramePolicy, *, generator: torch.Generator | None = None):
        if not isinstance(policy, GoalFramePolicy):
            raise ContractError("goal_frame_policy", "a declared frame-aware checkpoint is required")
        super().__init__(policy, generator=generator)

    def reset(self) -> None:
        super().reset()
        self._frame_invocation = None
        self.direction = 1

    def _prepare(self, observation: Mapping[str, Any], goal: SkillGoal):
        view, reference, encoded = super()._prepare(observation, goal)
        invocation = (self.identity, self.start_tick)
        if invocation != self._frame_invocation:
            self.direction = initial_direction(encoded[0]) if self.policy.frame == "initial_goal_direction" else 1
            self._frame_invocation = invocation
        return view, reference, transform_features(encoded, self.direction)

    @torch.no_grad()
    def step(self, observation: Mapping[str, Any], goal: SkillGoal, *, deterministic: bool = True) -> LearnedDecision:
        if deterministic is not True:
            raise ContractError("goal_frame_sampling", "this BC experiment has deterministic wire requests only")
        decision = super().step(observation, goal, deterministic=True)
        decision = replace(decision, action=restore_request(decision.action, self.direction))
        self.previous = decision.action
        return decision


def load_goal_frame_policy(path: str | Path, *, expected_config_sha256: str):
    """Bounded weights-only load that cannot relabel normalized weights as v1."""
    location = Path(path)
    if location.stat().st_size > 32 * 1024 * 1024:
        raise ContractError("goal_frame_checkpoint", "checkpoint exceeds the declared 32 MiB limit")
    payload = torch.load(location, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "nca-first-skill/v1":
        raise ContractError("goal_frame_checkpoint", "unknown checkpoint format")
    specification = payload.get("specification", {})
    policy = GoalFramePolicy(specification.get("family"))
    metadata = payload.get("metadata", {})
    if (specification != policy.specification() or metadata.get("config_sha256") != expected_config_sha256
            or metadata.get("family") != policy.family or metadata.get("feature_frame") != policy.frame):
        raise ContractError("goal_frame_checkpoint", "source experiment or feature/action interpretation differs")
    policy.load_state_dict(payload["state_dict"], strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in policy.state_dict().values()):
        raise ContractError("goal_frame_checkpoint", "finite checkpoint weights required")
    if payload.get("policy_sha256") != policy_state_sha256(policy):
        raise ContractError("goal_frame_checkpoint", "policy bytes differ from the declared identity")
    policy.eval()
    return policy, metadata
