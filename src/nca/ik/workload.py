"""Small fixed Runner IK device fixture; not the general IK solver or trainer.

Torch is imported only by explicit construction. Every committed optimizer step
is a checkpoint boundary. The sampler consumes whole batches; its permutation,
generator and cursor are saved, including across epoch boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import re
from typing import Any

from nca.contracts import sha256_json


WORKLOAD = "runner-point-inverse-smoke/v1"
NORMALIZATION = {"target_center": [3.6, 2.8], "target_scale": 6.55,
                 "joint_scale": [0.72, 0.27]}
SPLIT = {"training": ["runner-smoke-lattice"], "tuning": [],
         "evaluation": ["conformance-only-no-quality-holdout"]}


def dataset_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    value = {"generator": "runner-16x16-lattice/v1", "sampler": "torch-cpu-randperm/v1",
             "size": config["dataset_size"], "hip_range": [-0.6, 0.6],
             "ankle_range": [-0.2, 0.2], "split": SPLIT,
             "normalization": NORMALIZATION, "seed": config["seed"]}
    return {**value, "sha256": sha256_json(value)}


def runner_points(q, torch):
    """The independently hand-checked compact formula in IK_LAYER.md, FP32.

    This fixture is intentionally limited to the accepted Runner descriptor.
    It neither registers a general batched-FK implementation nor claims #52.
    """
    h, a = q[:, 0], q[:, 0] + q[:, 1]
    return torch.stack((3.6 - 4.95 * torch.sin(h) + torch.cos(a) - 1.25 * torch.sin(a),
                        2.8 + 4.95 * torch.cos(h) + torch.sin(a) + 1.25 * torch.cos(a)), dim=1)


def model_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SmokeWorkload:
    """Fixed 4→32→2 inverse MLP with Adam and a geometric endpoint loss."""

    def __init__(self, context, config: Mapping[str, Any]):
        torch = context.torch
        self.context, self.config, self.torch = context, dict(config), torch
        # Initialize on CPU with the same seed before moving to the chosen device.
        # This gives matched initial weights, not a promise of cross-device parity.
        self.model = torch.nn.Sequential(torch.nn.Linear(4, config["hidden_width"]),
                                         torch.nn.Tanh(), torch.nn.Dropout(0.1),
                                         torch.nn.Linear(config["hidden_width"], 2),
                                         torch.nn.Tanh()).to(context.device, dtype=torch.float32)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"],
                                          foreach=False, fused=False,
                                          capturable=context.device.type == "cuda")
        self.generator = torch.Generator(device="cpu").manual_seed(config["seed"] + 1)
        self.permutation = torch.randperm(config["dataset_size"], generator=self.generator)
        self.position = 0
        self.epoch = 0
        self.committed_steps = 0
        self.cursor = 0
        self.initial_model_sha256 = model_digest(self.model)

    @property
    def generators(self):
        return {"sampler": self.generator}

    def state(self, allocation: Mapping[str, Any] | None = None):
        return {"boundary": "optimizer_step", "committed_steps": self.committed_steps,
                "cursor": self.cursor, "sampler_position": self.position, "epoch": self.epoch,
                "permutation": self.permutation.tolist(), "normalization": NORMALIZATION,
                "dataset_identity": dataset_identity(self.config),
                "initial_model_sha256": self.initial_model_sha256,
                "allocation": dict(allocation or {})}

    def restore_state(self, state):
        torch = self.torch
        if (state.get("boundary") != "optimizer_step" or state.get("normalization") != NORMALIZATION
                or state.get("dataset_identity") != dataset_identity(self.config)):
            raise ValueError("checkpoint workload identities differ")
        steps, cursor, position, epoch = (state.get(key) for key in
                                          ("committed_steps", "cursor", "sampler_position", "epoch"))
        if (any(type(v) is not int or v < 0 for v in (steps, cursor, position, epoch))
                or cursor != steps * self.config["batch_size"] or steps > self.config["steps"]
                or not 0 <= position <= self.config["dataset_size"]
                or epoch * self.config["dataset_size"] + position != cursor):
            raise ValueError("checkpoint sampler/counter boundary differs")
        permutation = state.get("permutation")
        if (not isinstance(permutation, list)
                or any(type(value) is not int for value in permutation)
                or sorted(permutation) != list(range(self.config["dataset_size"]))):
            raise ValueError("checkpoint sampler permutation differs")
        initial = state.get("initial_model_sha256")
        if not isinstance(initial, str) or re.fullmatch(r"[a-f0-9]{64}", initial) is None:
            raise ValueError("checkpoint initial model identity missing")
        self.committed_steps, self.cursor, self.position, self.epoch = steps, cursor, position, epoch
        self.permutation = torch.tensor(permutation, dtype=torch.int64)
        self.initial_model_sha256 = initial

    def batch(self):
        torch, config = self.torch, self.config
        ids = []
        while len(ids) < config["batch_size"]:
            if self.position == config["dataset_size"]:
                self.permutation = torch.randperm(config["dataset_size"], generator=self.generator)
                self.position = 0
                self.epoch += 1
            count = min(config["batch_size"] - len(ids), config["dataset_size"] - self.position)
            ids.extend(self.permutation[self.position:self.position + count].tolist())
            self.position += count
        # CPU sampling is deliberate and separately timed; model/data/FK/loss and
        # all optimizer arithmetic are then checked on the selected device.
        indexes = torch.tensor(ids, dtype=torch.int64)
        goal = torch.stack(((indexes // 16).float() / 15 * 1.2 - 0.6,
                            (indexes % 16).float() / 15 * 0.4 - 0.2), dim=1)
        current = (torch.rand((len(ids), 2), generator=self.generator) * 2 - 1)
        current = current * torch.tensor([0.6, 0.2], dtype=torch.float32)
        return ids, {"goal_q": goal.to(self.context.device), "current_q": current.to(self.context.device)}

    def step(self, batch):
        torch, ctx = self.torch, self.context
        self.optimizer.zero_grad(set_to_none=True)
        scale = torch.tensor(NORMALIZATION["joint_scale"], device=ctx.device, dtype=torch.float32)
        center = torch.tensor(NORMALIZATION["target_center"], device=ctx.device, dtype=torch.float32)
        goal = runner_points(batch["goal_q"], torch)
        inputs = torch.cat((batch["current_q"] / scale,
                            (goal - center) / NORMALIZATION["target_scale"]), dim=1)
        predicted_q = self.model(inputs) * scale
        points = runner_points(predicted_q, torch)
        loss = ((points - goal) / NORMALIZATION["target_scale"]).square().mean()
        loss = loss + 0.05 * ((predicted_q - batch["goal_q"]) / scale).square().mean()
        loss.backward()
        gradients = {name: value.grad for name, value in self.model.named_parameters()}
        if any(value is None for value in gradients.values()):
            raise ValueError("missing model gradient")
        placement = {"data": batch, "inputs": inputs, "model": self.model.state_dict(),
                     "output": predicted_q, "geometry": points, "loss": loss,
                     "gradients": gradients}
        ctx.verify_tensors(placement)
        before = [value.detach().clone() for value in self.model.parameters()]
        self.optimizer.step()
        placement["optimizer"] = self.optimizer.state
        receipt = ctx.verify_tensors(placement)
        changed = any(not torch.equal(old, new.detach())
                      for old, new in zip(before, self.model.parameters()))
        if not changed:
            raise ValueError("optimizer did not change model weights")
        self.committed_steps += 1
        self.cursor += self.config["batch_size"]
        return {"loss": float(loss.detach().cpu()), "finite_gradients": True,
                "weights_changed": changed, "placement": receipt}
