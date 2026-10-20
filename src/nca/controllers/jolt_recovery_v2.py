"""Advisory local settling over the preserved Jolt recovery v1 policy.

Sparse policy observations cannot prove uninterrupted base-tick arrival. This
version may hold quietly; outcome-driven retirement belongs to the caller.
It preserves v1 movement, pulse admission and physical supervision.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from nca.contracts import WireAction
from .baseline import ControllerFrame, FeedbackObservation, SkillGoal
from .jolt_recovery import JoltRecoveryController


CONTROLLER_ID = "authored-jolt-recovery/v2"
PRESERVED_RECOVERY_SOURCE_SHA256 = {
    "jolt_recovery.py": "5bc64ff9e09092620ea74f7a6ab881e02acb0e984028674d4d4a5186ac4c9528",
    "recovery_supervisor.py": "e9a0b19aca661643c4a72acd0293ade72b5b569b8adb9abfca201de70205d09d",
}


class JoltRecoveryControllerV2(JoltRecoveryController):
    """Keep a local arrival estimate advisory until the caller retires it.

    ``step`` never receives the evaluator's state. An advisory neutral request
    retains its original observation lease and goal deadline. A later disturbed
    sample resumes ordinary correction. After validating its outcome and
    current ownership, the caller uses the existing ``cancel`` operation and
    separately cancels the native owner. This controller never scores its own
    task success; existing invalid-state and authority boundaries still fence
    ordinary control.
    """

    def __init__(self, actor: str = "jev") -> None:
        for name, expected in PRESERVED_RECOVERY_SOURCE_SHA256.items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
                raise ValueError("the advisory controller's preserved recovery source changed")
        super().__init__(actor)

    @property
    def controller_id(self) -> str:
        return CONTROLLER_ID

    def specification(self) -> dict:
        result = super().specification()
        result.update({
            "schema": "nca-jolt-recovery-reference/v2",
            "preserved_recovery_source_sha256": dict(PRESERVED_RECOVERY_SOURCE_SHA256),
            "arrival": "raw supported samples within tolerance, speed<=6 and angle<=.32 over12 observed ticks with gaps<=6 produce an advisory only",
            "arrival_advisory": "running neutral hold, unchanged original-source lease and goal deadline; fresh disturbance resumes ordinary correction",
            "completion": "caller validates outcome and current ownership, then cancels local/native intent; sampled arrival never retires the goal",
            "continuous_arrival": "unmeasured by sparse policy samples; the unchanged task evaluates every retained physical tick",
        })
        return result

    def reset(self) -> None:
        self._arrival_advisory = False
        super().reset()

    def cancel(self, reason: str = "canceled") -> ControllerFrame:
        self._arrival_advisory = False
        frame = super().cancel(reason)
        self._diagnostics["arrival_advisory"] = False
        return frame

    def _quiet_arrival(self, observation: FeedbackObservation, goal: SkillGoal,
                       distance: float) -> bool:
        # Preserve the sampled detector, but never enter v1's automatic
        # supervisor.complete() branch. The hold is applied below while the
        # original running admission is still intact; no retired lease is
        # renewed and no evaluator observation enters action selection.
        self._arrival_advisory = super()._quiet_arrival(observation, goal, distance)
        return False

    def step(self, observation: FeedbackObservation, goal: SkillGoal, *,
             current_tick: int | None = None) -> ControllerFrame:
        self._arrival_advisory = False
        frame = super().step(observation, goal, current_tick=current_tick)
        if self._arrival_advisory:
            self._x.cancel()
            self._turn.cancel()
            self._previous_move = 0.0
            self._phase, self._reason = "settle_hold", "supported_arrival_advisory"
            self._diagnostics.update(phase=self._phase, reason=self._reason,
                                     pulse_reason=None)
            frame = self._frame(WireAction(effort=0.0), frame.distance)
        self._diagnostics["arrival_advisory"] = self._arrival_advisory
        return frame
