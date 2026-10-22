"""Compact IK training primitives; no experiment loop or final-suite access.

The owning experiment must supply an authenticated corpus, cumulative allocation
and candidate-freeze protocol before using these primitives in a measured run.
Importing this module does not import Torch or execute an optimizer step.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any

from nca.contracts import ContractError, Record, sha256_json
from nca.ik.data import (DatasetConfig, GeometricSample, Normalization,
                         runtime_features, validate_sample)
from nca.ik.device import DeviceContext
from nca.ik.kinematics import TorchFK, chain_from_profile
from nca.ik.models import build_model
from nca.ik.profiles import runner_right_sole_profile


MODEL_ID = "runner-compact-ik"
TRAINER_ID = "runner-compact-ik-step"
LENGTH_CELLS = 6.550781059358212
LEARNING_RATE = 0.001
MAX_GRADIENT_NORM = 1.0
MAX_TARGET_CHANGE_CELLS = 0.10
MAX_REFERENCE_CHANGE_RADIANS = 0.25


@dataclass(frozen=True)
class LossWeights(Record):
    endpoint: float = 1.0
    joint: float = 0.1
    posture: float = 0.00001
    continuity: float = 0.0001
    step: float = 0.01
    bounds: float = 1.0

    def validate(self):
        values = (self.endpoint, self.joint, self.posture,
                  self.continuity, self.step, self.bounds)
        if any(v < 0 for v in values) or self.endpoint <= 0 or self.joint <= 0:
            raise ContractError("ik_loss_weights", "nonnegative weights and positive endpoint/joint terms required")


@dataclass(frozen=True)
class IKTrainingBatch:
    """Trusted in-process tensors from preparation, not a signed corpus artifact.

    Python callers can mutate tensors. The measured harness must authenticate
    the corpus and normalizer, own these objects, and retain ordered sample IDs.
    Failure rows remain present with explicit masks.
    """

    features: Any
    teacher_reference: Any
    pose_mask: Any
    endpoint_mask: Any
    continuity_pairs: Any
    sample_ids: tuple[str, ...]
    dataset_config_sha256: str
    teacher_status_counts: dict[str, int]


def prepare_training_batch(samples, config: DatasetConfig, context: DeviceContext) -> IKTrainingBatch:
    """Validate retained labels without re-running the numerical teacher.

    Only the training partition is admitted. No normalization is fitted here.
    Independent retained-label checking belongs to preparation, not a hidden
    per-update solver. Keep complete rows and status counts, including refusals.
    """
    if not isinstance(config, DatasetConfig) or not isinstance(context, DeviceContext):
        raise ContractError("ik_training_context", "typed dataset configuration and device context required")
    if not isinstance(samples, (tuple, list)) or not 1 <= len(samples) <= 1024:
        raise ContractError("ik_training_batch", "one to 1024 retained training rows required")
    if any(not isinstance(s, GeometricSample) or s.task.partition != "training" for s in samples):
        raise ContractError("ik_training_leakage", "optimization accepts typed training rows only")
    ids = tuple(s.task.id for s in samples)
    if len(set(ids)) != len(ids):
        raise ContractError("ik_training_duplicate", "a prepared batch cannot duplicate task identities")
    profile_hash = runner_right_sole_profile().sha256
    rows, references, pose, endpoint = [], [], [], []
    for sample in samples:
        validate_sample(sample, config)
        task, label = sample.task, sample.label
        supported = task.invalid_specimen is None and task.inputs.profile_sha256 == profile_hash
        rows.append(runtime_features(task.inputs) if supported else (0.,) * 14)
        admitted = supported and all((*task.inputs.present, *task.inputs.measured,
                                      *task.inputs.active, *task.inputs.task_mask))
        pose.append(bool(admitted and label.pose_loss))
        endpoint.append(bool(admitted and label.endpoint_loss))
        # A failed numerical candidate is never substituted as a supervised pose.
        references.append(label.teacher.configuration if pose[-1] else (0., 0.))
    # Actual adjacent trajectory samples only. Random point/current pairs are not
    # continuity observations. Augmentations and families never cross-pair.
    address = {(s.task.family_id, s.task.trajectory_id, s.task.state.augmentation,
                s.task.state.step): i for i, s in enumerate(samples)
               if s.task.state.kind == "trajectory"}
    pairs = []
    for i, sample in enumerate(samples):
        task = sample.task
        if task.state.kind != "trajectory" or not endpoint[i]:
            continue
        key = (task.family_id, task.trajectory_id, task.state.augmentation, task.state.step + 1)
        j = address.get(key)
        if (j is not None and endpoint[j]
                and math.dist(task.inputs.target_root, samples[j].task.inputs.target_root)
                <= MAX_TARGET_CHANGE_CELLS):
            pairs.append((i, j))
    torch, device = context.torch, context.device
    return IKTrainingBatch(
        features=torch.tensor(rows, dtype=torch.float32, device=device),
        teacher_reference=torch.tensor(references, dtype=torch.float32, device=device),
        pose_mask=torch.tensor(pose, dtype=torch.bool, device=device),
        endpoint_mask=torch.tensor(endpoint, dtype=torch.bool, device=device),
        continuity_pairs=torch.tensor(pairs, dtype=torch.int64, device=device).reshape(-1, 2),
        sample_ids=ids, dataset_config_sha256=config.sha256,
        teacher_status_counts=dict(sorted(Counter(s.label.teacher.status for s in samples).items())))


class IKTrainer:
    """One explicitly requested FP32 Adam update, with device/finite receipts.

    Counters are process-local diagnostics, not a durable experiment allocation.
    Any failed attempted step poisons this instance; a caller cannot silently
    continue after uncertain optimizer work. The existing checkpoint/ledger
    integration and measured-run supervisor are owned by ik_experiment and
    ik_run_state; these process-local counters alone never admit a run.
    """

    def __init__(self, context: DeviceContext, normalization: Normalization,
                 dataset_config: DatasetConfig, *, weights: LossWeights | None = None):
        if not isinstance(context, DeviceContext) or not isinstance(dataset_config, DatasetConfig):
            raise ContractError("ik_training_context", "typed device and dataset configuration required")
        if weights is not None and not isinstance(weights, LossWeights):
            raise ContractError("ik_loss_weights", "typed loss weights required")
        self.context = context
        self.config_sha256 = dataset_config.sha256
        self.normalization_sha256 = normalization.sha256 if isinstance(normalization, Normalization) else None
        self.weights = weights or LossWeights()
        self.model = build_model(context, normalization)
        self.fk = TorchFK(chain_from_profile(runner_right_sole_profile()), context)
        self.optimizer = context.torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, foreach=False, fused=False,
            capturable=context.device.type == "cuda")
        self.attempted_steps = 0
        self.committed_steps = 0
        self.sample_presentations = 0
        self.failed = False

    def _validate_batch(self, batch):
        torch, device = self.context.torch, self.context.device
        if (not isinstance(batch, IKTrainingBatch) or not batch.sample_ids
                or batch.dataset_config_sha256 != self.config_sha256):
            raise ContractError("ik_training_batch", "prepared training batch/config identity mismatch")
        count = len(batch.sample_ids)
        specs = ((batch.features, (count, 14), torch.float32),
                 (batch.teacher_reference, (count, 2), torch.float32),
                 (batch.pose_mask, (count,), torch.bool),
                 (batch.endpoint_mask, (count,), torch.bool))
        for tensor, shape, dtype in specs:
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape
                    or tensor.dtype != dtype or tensor.device != device):
                raise ContractError("ik_training_tensor", "batch shape, dtype or device mismatch")
        pairs = batch.continuity_pairs
        if (not isinstance(pairs, torch.Tensor) or pairs.ndim != 2 or pairs.shape[1] != 2
                or pairs.dtype != torch.int64 or pairs.device != device):
            raise ContractError("ik_training_tensor", "continuity pair tensor mismatch")
        if pairs.numel() and bool(((pairs < 0) | (pairs >= count)).any()):
            raise ContractError("ik_training_pairs", "continuity indices outside this batch")
        self.context.verify_tensors({"batch": (batch.features, batch.teacher_reference,
                                               batch.pose_mask, batch.endpoint_mask, pairs)})

    def loss(self, batch: IKTrainingBatch):
        """Return differentiable terms and raw predictions, with no correction."""
        self._validate_batch(batch)
        torch = self.context.torch
        prediction = self.model(batch.features)
        admitted = batch.pose_mask | batch.endpoint_mask
        if bool((admitted & ~prediction.valid).any()):
            raise ContractError("ik_training_admission", "an eligible retained row was refused by the model")
        geometry = self.fk(prediction.reference)
        if bool((prediction.valid & ~geometry.valid).any()):
            raise ContractError("ik_training_geometry", "raw reference failed independent FK admission")
        q = prediction.reference
        scale = torch.tensor((.72, .27), dtype=torch.float32, device=self.context.device)

        def wrap(delta):
            return torch.atan2(torch.sin(delta), torch.cos(delta))

        def average(values, mask):
            # The zero-count path remains connected to the graph, while the
            # caller separately refuses to optimize a wholly unlabelled batch.
            return torch.where(mask, values, torch.zeros_like(values)).sum() / mask.sum().clamp_min(1)

        endpoint_error = ((geometry.point - batch.features[:, 2:4]) / LENGTH_CELLS).square().sum(-1)
        joint_error = (wrap(q - batch.teacher_reference) / scale).square().mean(-1)
        posture_error = (wrap(q - batch.features[:, 4:6]) / scale).square().mean(-1)
        bounds_error = ((torch.relu(-scale - q) + torch.relu(q - scale)) / scale).square().mean(-1)
        terms = {
            "endpoint": average(endpoint_error, batch.endpoint_mask),
            "joint": average(joint_error, batch.pose_mask),
            "posture": average(posture_error, batch.endpoint_mask),
            "bounds": average(bounds_error, admitted),
        }
        pairs = batch.continuity_pairs
        differences = wrap(q[pairs[:, 1]] - q[pairs[:, 0]])
        pair_mask = batch.endpoint_mask[pairs[:, 0]] & batch.endpoint_mask[pairs[:, 1]]
        terms["continuity"] = average((differences / scale).square().mean(-1), pair_mask)
        terms["step"] = average(
            (torch.relu(differences.abs() - MAX_REFERENCE_CHANGE_RADIANS)
             / MAX_REFERENCE_CHANGE_RADIANS).square().mean(-1), pair_mask)
        total = sum(getattr(self.weights, name) * value for name, value in terms.items())
        return total, terms, prediction, geometry

    def step(self, batch: IKTrainingBatch) -> dict:
        if self.failed:
            raise ContractError("ik_training_failed", "failed trainer instance cannot continue")
        self._validate_batch(batch)
        self.attempted_steps += 1
        self.sample_presentations += len(batch.sample_ids)
        torch, ctx = self.context.torch, self.context
        try:
            if not bool((batch.pose_mask | batch.endpoint_mask).any()):
                raise ContractError("ik_training_empty_loss", "no eligible endpoint or pose supervision")
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            total, terms, prediction, geometry = self.loss(batch)
            ctx.verify_tensors({"loss": total, "terms": terms, "output": prediction.reference})
            total.backward()
            gradients = {name: parameter.grad for name, parameter in self.model.named_parameters()}
            if any(value is None for value in gradients.values()):
                raise ContractError("ik_training_gradient", "every model parameter requires a gradient")
            ctx.verify_tensors({"gradients": gradients})
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRADIENT_NORM,
                                                   error_if_nonfinite=True)
            before = [p.detach().clone() for p in self.model.parameters()]
            self.optimizer.step()
            placement = ctx.verify_tensors({
                "batch": (batch.features, batch.teacher_reference, batch.pose_mask, batch.endpoint_mask),
                "model": self.model.state_dict(), "output": prediction.reference,
                "geometry": geometry.point, "loss": total, "gradients": gradients,
                "optimizer": self.optimizer.state})
            changed = any(not torch.equal(old, new.detach()) for old, new in zip(before, self.model.parameters()))
            self.committed_steps += 1
            return {
                "schema": "nca-ik-training-step/v1", "device": str(ctx.device),
                "model": MODEL_ID + "@1", "trainer": TRAINER_ID + "@1",
                "dataset_config_sha256": self.config_sha256,
                "normalization_sha256": self.normalization_sha256,
                "loss_weights_sha256": self.weights.sha256,
                "ordered_sample_ids_sha256": sha256_json(list(batch.sample_ids)),
                "attempted_steps": self.attempted_steps, "committed_steps": self.committed_steps,
                "sample_presentations": self.sample_presentations, "rows": len(batch.sample_ids),
                "pose_rows": int(batch.pose_mask.sum().detach().cpu()),
                "endpoint_rows": int(batch.endpoint_mask.sum().detach().cpu()),
                "continuity_pairs": int(batch.continuity_pairs.shape[0]),
                "refused_input_rows": int((~prediction.valid).sum().detach().cpu()),
                "teacher_status_counts": dict(batch.teacher_status_counts),
                "loss": float(total.detach().cpu()),
                "terms": {key: float(value.detach().cpu()) for key, value in terms.items()},
                "gradient_norm_before_clip": float(norm.detach().cpu()),
                "weights_changed": changed, "placement": placement,
                "correction_iterations": 0, "fallback_calls": 0,
                "evidence_kind": "software_update_not_learned_qualification",
            }
        except BaseException:
            self.failed = True
            raise
