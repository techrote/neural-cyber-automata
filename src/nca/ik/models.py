"""Compact Runner IK references on an explicitly selected optional Torch device.

Importing this module does not import Torch. This fixed pilot requires every
presence, measurement, activity and task bit. Invalid rows carry finite masked
placeholders and must not be interpreted as references. No solver or fallback
runs here; this model alone does not implement the host target contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nca.contracts import ContractError
from .data import FEATURE_NAMES, Normalization
from .device import DeviceContext
from .kinematics import _float32_bound
from .profiles import runner_right_sole_profile


MODEL_ID = "runner-compact-ik"
MODEL_VERSION = 1
FEATURE_TRANSFORM = "runner-compact-features/v1"
FEATURE_COUNT = 14
HIDDEN_WIDTH = 64
NONFINITE_INPUT = 1
INVALID_MASK = 2
MISSING_MEASUREMENT = 4
INACTIVE_CHANNEL = 8
OUT_OF_BOUNDS = 16
NONFINITE_OUTPUT = 32


@dataclass(frozen=True)
class IKPrediction:
    """Batched absolute references, validity and per-row bit flags."""

    reference: Any
    valid: Any
    invalid_code: Any


def build_model(context: DeviceContext, normalization: Normalization):
    """Construct the fixed 14 -> 64 -> 64 -> 2 FP32 model explicitly.

    The first six numeric channels use the immutable training-only fit. The
    eight mask bits stay explicit, unstandardized bits. The measured harness
    must authenticate normalization against its admitted corpus before use.
    Input bounds are strict: a rounded FP32 value outside the exact descriptor
    bound is refused, including nearest rounding of some decimal endpoints.
    """
    if not isinstance(context, DeviceContext):
        raise ContractError("ik_model_device", "explicit IK DeviceContext required")
    if not isinstance(normalization, Normalization):
        raise ContractError("ik_model_normalization", "typed training normalization required")
    if normalization.feature_names != FEATURE_NAMES:
        raise ContractError("ik_model_normalization", "exact permitted feature order required")
    torch = context.torch
    profile = runner_right_sole_profile()
    joints = profile.effector_chain

    def values(items):
        return torch.tensor(items, dtype=torch.float32, device="cpu")

    def bounds(attribute, *, lower):
        return values([_float32_bound(getattr(j, attribute), lower=lower) for j in joints])

    mean, scale = values(normalization.mean[:6]), values(normalization.scale[:6])
    if (not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all())
            or not bool((scale > 0).all()) or not bool(torch.isfinite(1 / scale).all())
            or not bool(torch.isfinite(mean / scale).all())):
        raise ContractError("ik_model_normalization", "normalization must be representable in FP32")

    class RunnerIKMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model_id, self.model_version = MODEL_ID, MODEL_VERSION
            self.feature_transform = FEATURE_TRANSFORM
            self.profile_sha256 = profile.sha256
            self.normalization_sha256 = normalization.sha256
            self.register_buffer("normalization_mean", mean)
            self.register_buffer("normalization_scale", scale)
            self.register_buffer("command_lower", bounds("target_lower", lower=True))
            self.register_buffer("command_upper", bounds("target_upper", lower=False))
            self.register_buffer("current_lower", bounds("lower", lower=True))
            self.register_buffer("current_upper", bounds("upper", lower=False))
            # CPU initialization makes the starting weights device-independent
            # for a fixed configured seed; arithmetic parity is not promised.
            self.network = torch.nn.Sequential(
                torch.nn.Linear(FEATURE_COUNT, HIDDEN_WIDTH, device="cpu", dtype=torch.float32),
                torch.nn.Tanh(),
                torch.nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, device="cpu", dtype=torch.float32),
                torch.nn.Tanh(),
                torch.nn.Linear(HIDDEN_WIDTH, 2, device="cpu", dtype=torch.float32),
            )

        def forward(self, raw_features):
            if (not isinstance(raw_features, torch.Tensor) or raw_features.ndim != 2
                    or raw_features.shape[0] == 0 or raw_features.shape[1] != FEATURE_COUNT
                    or raw_features.dtype != torch.float32 or raw_features.device != context.device
                    or self.normalization_mean.device != context.device
                    or self.normalization_mean.dtype != torch.float32):
                raise ContractError("ik_model_tensor", "nonempty [B,14] FP32 on the selected device required")
            masks = raw_features[:, 6:]
            finite = torch.isfinite(raw_features).all(dim=1)
            binary = ((masks == 0) | (masks == 1)).all(dim=1)
            present_measured = (masks[:, :4] == 1).all(dim=1)
            active_task = (masks[:, 4:] == 1).all(dim=1)
            current, posture = raw_features[:, :2], raw_features[:, 4:6]
            outside = (((current < self.current_lower) | (current > self.current_upper)).any(dim=1)
                       | ((posture < self.command_lower) | (posture > self.command_upper)).any(dim=1))
            code = ((~finite).to(torch.int64) * NONFINITE_INPUT
                    + (~binary).to(torch.int64) * INVALID_MASK
                    + (~present_measured).to(torch.int64) * MISSING_MEASUREMENT
                    + (~active_task).to(torch.int64) * INACTIVE_CHANNEL
                    + outside.to(torch.int64) * OUT_OF_BOUNDS)
            eligible = code == 0
            # Sanitize before arithmetic and before the network. Masking NaN
            # losses after the fact is insufficient to keep gradients finite.
            numeric = torch.where(eligible[:, None], raw_features[:, :6],
                                  torch.zeros_like(raw_features[:, :6]))
            normalized = (numeric - self.normalization_mean) / self.normalization_scale
            normalization_finite = torch.isfinite(normalized).all(dim=1)
            code = code | ((~normalization_finite).to(torch.int64) * NONFINITE_INPUT)
            eligible = code == 0
            features = torch.cat((
                torch.where(eligible[:, None], normalized, torch.zeros_like(normalized)),
                torch.where(eligible[:, None], masks, torch.zeros_like(masks)),
            ), dim=1)
            logits = self.network(features)
            midpoint = (self.command_lower + self.command_upper) / 2
            radius = (self.command_upper - self.command_lower) / 2
            # The fixed Runner intervals are symmetric: tanh scaling stays
            # inside their inward FP32 limits without a hidden projection.
            reference = midpoint + radius * torch.tanh(logits)
            output_finite = torch.isfinite(logits).all(dim=1) & torch.isfinite(reference).all(dim=1)
            code = code | ((~output_finite).to(torch.int64) * NONFINITE_OUTPUT)
            valid = code == 0
            return IKPrediction(torch.where(valid[:, None], reference, torch.zeros_like(reference)),
                                valid, code)

    return RunnerIKMLP().to(context.device, dtype=torch.float32)
