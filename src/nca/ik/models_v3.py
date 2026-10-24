"""Version 3 development Runner IK references on an explicitly selected optional Torch device.

Importing this module does not import Torch. This fixed pilot requires every
presence, measurement, activity and task bit. Invalid rows carry finite masked
placeholders and must not be interpreted as references. No solver or fallback
runs here; this model alone does not implement the host target contract.
"""

from __future__ import annotations


from nca.contracts import ContractError
from .data import FEATURE_NAMES, Normalization
from .device import DeviceContext
from .models import IKPrediction
from .kinematics import _float32_bound
from .profiles import runner_right_sole_profile


MODEL_ID = "runner-compact-ik"
MODEL_VERSION = 3
FEATURE_TRANSFORM = "runner-compact-features/v1"
FEATURE_COUNT = 14
HIDDEN_WIDTH = 64
NONFINITE_INPUT = 1
INVALID_MASK = 2
MISSING_MEASUREMENT = 4
INACTIVE_CHANNEL = 8
OUT_OF_BOUNDS = 16
NONFINITE_OUTPUT = 32


def build_model(context: DeviceContext, normalization, *, head="interior_linear", hidden_width=64):
    from .polar import TRANSFORM
    feature_transform = TRANSFORM
    if head not in ("tanh", "interior_linear") or type(hidden_width) is not int or hidden_width not in (64, 128):
        raise ContractError("ik_model_v3_config", "registered head and width required")
    """Construct the fixed 14 -> 64 -> 64 -> 2 FP32 model explicitly.

    The first six numeric channels use the immutable training-only fit. The
    eight mask bits stay explicit, unstandardized bits. The measured harness
    must authenticate normalization against its admitted corpus before use.
    Input bounds are strict: a rounded FP32 value outside the exact descriptor
    bound is refused, including nearest rounding of some decimal endpoints.
    """
    if not isinstance(context, DeviceContext):
        raise ContractError("ik_model_device", "explicit IK DeviceContext required")
    from .polar import PolarNormalization, TRANSFORM, FEATURES
    polar = feature_transform == TRANSFORM
    if feature_transform not in (FEATURE_TRANSFORM, TRANSFORM):
        raise ContractError("ik_model_transform", "registered feature transform required")
    if not isinstance(normalization, PolarNormalization if polar else Normalization):
        raise ContractError("ik_model_normalization", "typed training normalization required")
    if normalization.feature_names != (FEATURES if polar else FEATURE_NAMES):
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
            self.feature_transform = feature_transform
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
                torch.nn.Linear(FEATURE_COUNT, hidden_width, device="cpu", dtype=torch.float32),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_width, hidden_width, device="cpu", dtype=torch.float32),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_width, 2, device="cpu", dtype=torch.float32),
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
            if polar:
                u, v = numeric[:, 2] - 3.6, numeric[:, 3] - 2.8
                numeric = torch.cat((numeric[:, :2], (u*u+v*v)[:, None],
                                     torch.atan2(v, u)[:, None], numeric[:, 4:6]), dim=1)
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
            reference = midpoint + radius * (interior_linear(torch, logits, radius)
                                               if head == "interior_linear" else torch.tanh(logits))
            output_finite = torch.isfinite(logits).all(dim=1) & torch.isfinite(reference).all(dim=1)
            code = code | ((~output_finite).to(torch.int64) * NONFINITE_OUTPUT)
            valid = code == 0
            return IKPrediction(torch.where(valid[:, None], reference, torch.zeros_like(reference)),
                                valid, code)

    # Both feature and head markers distinguish this version from v2.
    # The polar input interpretation itself remains identical.
    class RunnerIKPolarV3MLP(RunnerIKMLP):
        def __init__(self):
            super().__init__()
            self.register_buffer("feature_transform_version", torch.tensor(3, dtype=torch.int64))
            self.register_buffer("output_head_version", torch.tensor(1 if head == "interior_linear" else 0, dtype=torch.int64))

        def load_state_dict(self, state_dict, strict=True, assign=False):
            marker = state_dict.get("feature_transform_version")
            if (not isinstance(marker, torch.Tensor) or marker.dtype != torch.int64
                    or marker.shape != torch.Size([]) or marker.item() != 3):
                raise ContractError("ik_model_checkpoint", "v3 checkpoint identity required")
            head_marker = state_dict.get("output_head_version")
            if (not isinstance(head_marker, torch.Tensor) or head_marker.shape != torch.Size([])
                    or head_marker.dtype != torch.int64 or head_marker.item() != (1 if head == "interior_linear" else 0)):
                raise ContractError("ik_model_checkpoint", "v3 output head identity differs")
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

    model_type = RunnerIKPolarV3MLP if polar else RunnerIKMLP
    return model_type().to(context.device, dtype=torch.float32)


def interior_linear(torch, logits, radius):
    """Branch-free C1 identity interior, bounded tails; NaN propagates.

    Clamp the inactive tail argument to zero so its gradient is finite even
    for extreme finite FP32 inputs. sign(z)*max(|z|-k,0) has zero derivative
    at z=0, safely outside the tail. No parameters or hidden projection.
    """
    knee = torch.tensor((.62, .22), dtype=logits.dtype, device=logits.device) / radius
    magnitude = logits.abs()
    excess = (magnitude - knee).clamp_min(0)
    tail = logits.sign() * (knee + (1 - knee) * torch.tanh(excess / (1 - knee)))
    return torch.where(magnitude <= knee, logits, tail)

# Copyright 2026 techrote. All rights reserved.
