"""Bounded declared skill sequences with explicit recovery branches.

This synchronous helper is for script/human-selected tasks and short fixtures.
Async decision sources use the same SkillRuntime from their continuously ticking
owner loop. A task never searches for new goals or silently retries a trigger.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from nca.contracts import ContractError, Record, identifier
from .usables import (SkillInitiation, SkillProgress, SkillRuntime,
                      interval_observations, validate_parameters)


@dataclass(frozen=True)
class SkillSpec(Record):
    skill: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def validate(self):
        validate_parameters(self.skill, self.parameters)


@dataclass(frozen=True)
class TaskStep(Record):
    primary: SkillSpec
    recovery: tuple[SkillSpec, ...] = ()

    def validate(self):
        if len(self.recovery) > 2:
            raise ContractError("recovery_budget", "at most two explicitly declared recovery skills")


@dataclass(frozen=True)
class TaskPlan(Record):
    id: str
    steps: tuple[TaskStep, ...]
    max_decisions: int = 120
    schema_version: Literal[1] = 1

    def validate(self):
        identifier(self.id, "task.id")
        if len(self.id) > 80 or not 1 <= len(self.steps) <= 12 or not 1 <= self.max_decisions <= 600:
            raise ContractError("task_budget", "short task ID, 1..12 steps and 1..600 decisions required")


@dataclass(frozen=True)
class SkillAttempt(Record):
    stage: int
    role: Literal["primary", "recovery"]
    initiation: SkillInitiation
    result: SkillProgress | None


@dataclass(frozen=True)
class TaskResult(Record):
    task_id: str
    status: Literal["completed", "failed", "budget_exhausted"]
    local_success: bool
    decisions: int
    attempts: tuple[SkillAttempt, ...]
    evaluator_success: None = None
    schema_version: Literal[1] = 1


def run_task(env, runtime: SkillRuntime, plan: TaskPlan, view: Mapping[str, Any], *,
             source: Literal["human", "script", "model"] = "script", source_generation: int = 1):
    """Return (typed result, final owned local observation, copied wire trace).

    A recovery branch is part of the requested task's meaning. Failed primary
    attempts remain in the result; a successful allowed branch does not rewrite
    their failures. Every completion/cancellation calls the real host cancel API
    before a fresh subsequent skill is admitted. No world reset occurs here.
    """
    decisions = 0
    attempts = []
    trace = []
    serial = 0

    def execute(spec: SkillSpec, stage: int, role: str):
        nonlocal view, decisions, serial
        serial += 1
        receipt = runtime.start(spec.skill, spec.parameters, view, source=source,
                                source_generation=source_generation,
                                request_id=f"{plan.id}:{stage}:{serial}")
        if not receipt.accepted:
            attempts.append(SkillAttempt(stage, role, receipt, None))
            return False
        try:
            return execute_accepted(receipt, stage, role)
        except BaseException:
            # A transport, recorder, caller interruption or unforeseen error
            # must not leave an accepted charge/tool skill armed. Cancellation
            # never sends a neutral/release step and never masks the first error.
            try:
                runtime.cancel("task_execution_error")
            except Exception:
                pass
            try:
                env.cancel()
            except Exception:
                pass
            raise

    def execute_accepted(receipt: SkillInitiation, stage: int, role: str):
        nonlocal view, decisions
        feedback = ()
        while True:
            result = runtime.progress(view, observations=feedback)
            if result.terminal:
                if result.cancel_host:
                    view = env.cancel()
                attempts.append(SkillAttempt(stage, role, receipt, result))
                return result.local_success is True
            if decisions >= plan.max_decisions:
                result = runtime.cancel("task_decision_budget")
                view = env.cancel()
                attempts.append(SkillAttempt(stage, role, receipt, result))
                return False
            before = view
            view, _, terminated, truncated, info = env.step(result.action.to_dict())
            decisions += 1
            trace.append({"request_id": receipt.request.id, "tick_before": before["tick"],
                          "tick_after": view["tick"], "source": source,
                          "requested": result.action.to_dict(), "phase": result.phase,
                          "position": list(view["position"]), "velocity": list(view["velocity"]),
                          "rotation": view["rotation"], "angular_velocity": view["angular_velocity"],
                          "actual_ticks": info["actual_ticks"],
                          "admission": [tick["admission"] for tick in info["trace"]],
                          "consumed": [{"tick": tick["tick_after"], "applied": tick["applied"],
                                        "passive": tick["passive"], "actual_motion": tick["actual_motion"],
                                        "detached": [part["detached"] for part in tick["observation"]["parts"]]}
                                       for tick in info["trace"]],
                          "effects": [dict(effect) for tick in info["trace"]
                                      for effect in tick["observation"].get("last_effects", ())]})
            try:
                feedback = interval_observations(info)
            except ContractError:
                result = runtime.cancel("invalid_feedback_interval")
                view = env.cancel()
                attempts.append(SkillAttempt(stage, role, receipt, result))
                return False
            if terminated or truncated:
                result = runtime.cancel("host_episode_finished")
                view = env.cancel()
                attempts.append(SkillAttempt(stage, role, receipt, result))
                return False

    success = True
    for index, stage in enumerate(plan.steps):
        if execute(stage.primary, index, "primary"):
            continue
        if decisions >= plan.max_decisions or not stage.recovery:
            success = False
            break
        for recovery in stage.recovery:
            if not execute(recovery, index, "recovery"):
                success = False
                break
        if not success:
            break
    status = "completed" if success else ("budget_exhausted" if decisions >= plan.max_decisions else "failed")
    return TaskResult(plan.id, status, success, decisions, tuple(attempts)), view, tuple(trace)
