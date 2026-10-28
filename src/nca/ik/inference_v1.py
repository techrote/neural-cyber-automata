"""CPU batch-one frozen graph for the existing v3 model; no training or solver.

The graph retains input masks, finite/bounds checks, polar transformation and
output refusal bits. Weights are copied at construction and never updated.
Torch remains optional until construction. This is a development runtime, not
a portable export bundle or permission to access an evaluation suite.
"""
from __future__ import annotations

import copy
import warnings

from nca.contracts import ContractError
from .device import DeviceContext
from .models import IKPrediction


INFERENCE_ID = "runner-polar-cpu-frozen/v1"


def freeze_batch_one(context, model):
    """Freeze an authenticated, caller-loaded v3 model for CPU [1,14] FP32.

    Only structural checks become construction-time checks. All row-dependent
    validation remains in the traced tensor graph. No reusable input/output
    buffers or cached answers are used, so callers can safely interleave calls.
    """
    if not isinstance(context, DeviceContext) or context.device.type != "cpu":
        raise ContractError("ik_inference_device", "explicit CPU context required")
    torch = context.torch
    from .polar import TRANSFORM
    if (getattr(model, "model_version", None) != 3
            or getattr(model, "feature_transform", None) != TRANSFORM
            or model.feature_transform_version.item() != 3
            or model.output_head_version.item() not in (0, 1)):
        raise ContractError("ik_inference_identity", "polar v3 model required")
    for value in model.state_dict().values():
        if value.device.type != "cpu" or (value.is_floating_point() and value.dtype != torch.float32):
            raise ContractError("ik_inference_state", "CPU FP32 model state required")

    class TupleGraph(torch.nn.Module):
        def __init__(self, snapshot):
            super().__init__()
            self.snapshot = snapshot

        def forward(self, features):
            result = self.snapshot(features)
            return result.reference, result.valid, result.invalid_code

    snapshot = copy.deepcopy(model).eval()
    example = torch.zeros((1, 14), dtype=torch.float32, device="cpu")
    example[:, 6:] = 1
    # The original forward's Python checks inspect only shape/dtype/device.
    # The wrapper below enforces those same fixed batch-one requirements.
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        graph = torch.jit.freeze(torch.jit.trace(TupleGraph(snapshot).eval(), example, check_trace=True))

    class FrozenBatchOne:
        inference_id = INFERENCE_ID

        def __call__(self, features):
            if (not isinstance(features, torch.Tensor) or features.shape != (1, 14)
                    or features.dtype != torch.float32 or features.device != context.device):
                raise ContractError("ik_inference_tensor", "CPU [1,14] FP32 required")
            with torch.inference_mode():
                reference, valid, code = graph(features)
            return IKPrediction(reference, valid, code)

    return FrozenBatchOne()


# Copyright 2026 techrote. All rights reserved.
