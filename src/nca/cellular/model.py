"""Small shared recurrent modules with a declared root readout and no pooling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from nca.contracts import ContractError, WireAction, canonical_json
from nca.controllers.baseline import SkillGoal
from .core import (ACTION_NAMES, EDGES, FEATURE_NAMES, FEATURE_SCHEMA, JOLT_PROFILE_SHA256,
                   PART_IDS, GraphFrame, GraphIdentity, checked_parts, make_frame, mapped_request)
from .messages import FaultSpec, MessageBus


class GraphPolicy(nn.Module):
    def __init__(self, kind: str = "graph"):
        super().__init__()
        if kind not in ("graph", "no_message"):
            raise ContractError("graph_kind", "only the commissioned graph and no-message control are defined")
        self.kind = kind
        self.encoder = nn.Linear(30, 16)
        self.message = nn.Linear(32, 8)
        self.recurrent = nn.GRUCell(24, 16)
        self.readout = nn.Linear(16, 6)
        nn.init.orthogonal_(self.readout.weight, .1)
        nn.init.zeros_(self.readout.bias)

    def zero_hidden(self, batch: int = 1) -> Tensor:
        return self.encoder.weight.new_zeros((batch, 3, 16))

    def encode(self, inputs: Tensor, hidden: Tensor):
        if inputs.ndim != 3 or inputs.shape[1:] != (3, 30) or hidden.shape != (len(inputs), 3, 16):
            raise ContractError("graph_shape", "three local nodes with explicit external hidden state required")
        if not bool(torch.isfinite(inputs).all()) or not bool(torch.isfinite(hidden).all()):
            raise ContractError("graph_finite", "nonfinite graph input or memory")
        encoded = self.encoder(inputs).tanh()
        messages = self.message(torch.cat((encoded, hidden), dim=-1)).tanh()
        return encoded, messages

    def update(self, encoded, messages, hidden, neighbor_sums, neighbor_counts):
        expected = (len(encoded), 3, 8)
        if (encoded.shape != (len(encoded), 3, 16) or messages.shape != expected
                or hidden.shape != (len(encoded), 3, 16)
                or neighbor_sums.shape != expected or neighbor_counts.shape != (len(encoded), 3, 1)):
            raise ContractError("graph_shape", "local delivered message capacity differs")
        if (any(not bool(torch.isfinite(value).all()) for value in
                (encoded, messages, hidden, neighbor_sums, neighbor_counts))
                or bool((neighbor_counts < 0).any()) or bool((neighbor_counts > 2).any())
                or bool((neighbor_counts != neighbor_counts.round()).any())
                or bool((neighbor_sums.abs() > neighbor_counts + 1e-6).any())):
            raise ContractError("graph_message", "invalid finite bounded local message aggregation")
        aggregate = (messages + neighbor_sums) / (1 + neighbor_counts)
        recurrent_input = torch.cat((encoded, aggregate), dim=-1).reshape(-1, 24)
        next_hidden = self.recurrent(recurrent_input, hidden.reshape(-1, 16)).reshape(-1, 3, 16)
        # This is the complete action readout. No pad/global hidden pooling.
        return self.readout(next_hidden[:, 0]), next_hidden

    def forward(self, inputs, hidden):
        encoded, messages = self.encode(inputs, hidden)
        adjacency = inputs.new_zeros((3, 3))
        if self.kind == "graph":
            for source, destination in EDGES:
                adjacency[destination, source] = 1
        sums = torch.einsum("ij,bjm->bim", adjacency, messages)
        counts = adjacency.sum(dim=1).reshape(1, 3, 1).expand(len(inputs), -1, -1)
        return self.update(encoded, messages, hidden, sums, counts)

    @staticmethod
    def outputs(logits):
        return torch.cat((logits[..., :2].tanh(), logits[..., 2:].sigmoid()), dim=-1)

    def specification(self):
        return {"schema": "jolt-three-part-policy/v1", "kind": self.kind,
                "feature_schema": FEATURE_SCHEMA, "features": list(FEATURE_NAMES), "parts": list(PART_IDS),
                "edges": [list(edge) for edge in EDGES] if self.kind == "graph" else [],
                "private_memory": 16, "message_width": 8, "rounds": 1, "readout_node": 0,
                "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
                "actions": list(ACTION_NAMES), "profile_sha256": JOLT_PROFILE_SHA256,
                "decision_ticks": 6, "shared_primary_gate": "supported_known_cooldown_ready_rising_edge"}


def weight_sha256(policy: GraphPolicy) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        digest.update(canonical_json([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_policy(policy: GraphPolicy, path: str | Path, metadata: Mapping[str, Any]):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "nca-jolt-part-graph/v1", "specification": policy.specification(),
               "state_dict": {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()},
               "policy_sha256": weight_sha256(policy), "metadata": dict(metadata)}
    with target.open("xb") as stream:
        torch.save(payload, stream)


def load_policy(path: str | Path, *, expected_config_sha256: str | None = None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "nca-jolt-part-graph/v1":
        raise ContractError("graph_bundle", "unknown graph policy bundle")
    if (not isinstance(payload.get("specification"), dict)
            or not isinstance(payload.get("metadata"), dict)
            or not isinstance(payload.get("state_dict"), dict)):
        raise ContractError("graph_bundle", "policy specification, metadata and weights are required")
    policy = GraphPolicy(payload["specification"].get("kind"))
    if payload["specification"] != policy.specification():
        raise ContractError("graph_bundle", "topology, sensor, memory or readout contract differs")
    metadata = payload.get("metadata", {})
    if expected_config_sha256 is not None and metadata.get("config_sha256") != expected_config_sha256:
        raise ContractError("graph_config", "bundle belongs to another frozen experiment")
    try:
        policy.load_state_dict(payload["state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ContractError("graph_weights", "weight tensors differ from the declared network") from exc
    if any(not bool(torch.isfinite(value).all()) for value in policy.state_dict().values()):
        raise ContractError("graph_weights", "nonfinite policy weight")
    if payload.get("policy_sha256") != weight_sha256(policy):
        raise ContractError("graph_weights", "bundle weight identity differs")
    policy.eval()
    return policy, metadata


@dataclass(frozen=True)
class GraphDecision:
    action: WireAction
    frame: GraphFrame
    hidden_before: tuple[tuple[float, ...], ...]
    communication: Mapping[str, Any]
    reason: str = "graph_root_readout"


class GraphController:
    """Serialized owner-loop inference with actor/goal/source memory fences."""
    def __init__(self, policy: GraphPolicy, *, profile_sha256: str,
                 fault: FaultSpec = FaultSpec(), communication_seed: int = 0):
        if profile_sha256 != JOLT_PROFILE_SHA256 or policy.encoder.weight.device.type != "cpu":
            raise ContractError("graph_profile", "this candidate requires the pinned CPU Jolt profile")
        self.policy, self.fault = policy, fault
        self.bus = MessageBus(fault, seed=communication_seed, enabled=policy.kind == "graph")
        self.reset()

    def reset(self):
        self.hidden = self.policy.zero_hidden()
        self.previous = WireAction(effort=0)
        self.identity: GraphIdentity | None = None
        self.last_tick: int | None = None
        self.start_tick = 0
        self.goal: SkillGoal | None = None
        self.cached: GraphDecision | None = None
        self.retired: set[tuple] = set()
        self.bus.reset()

    @staticmethod
    def _invocation(identity):
        return identity.run_id, identity.actor_id, identity.epoch, identity.goal_id

    def reset_memory(self, nodes=(0, 1, 2)):
        nodes = tuple(nodes)
        if any(type(node) is not int for node in nodes) or not set(nodes) <= {0, 1, 2}:
            raise ContractError("graph_node", "unknown private-memory owner")
        self.hidden = self.hidden.clone()
        self.hidden[:, nodes, :] = 0
        self.bus.clear_nodes(nodes)

    def cancel(self):
        if self.identity is not None:
            self.retired.add(self._invocation(self.identity))
        self.reset_memory()
        self.previous = WireAction(effort=0)
        self.cached = None
        return self.previous

    def step(self, view, goal: SkillGoal, *, source_generation=0, mount_generation=None):
        try:
            checked_parts(view)
            identity = GraphIdentity.from_view(view, goal, source_generation=source_generation,
                                                mount_generation=mount_generation)
        except (ContractError, KeyError, TypeError):
            self.cancel()
            raise
        if self.identity is not None:
            same_actor = (identity.run_id, identity.actor_id, identity.epoch) == (
                self.identity.run_id, self.identity.actor_id, self.identity.epoch)
            fence_changed = (identity.control_generation, identity.source_generation, identity.mount_generation) != (
                self.identity.control_generation, self.identity.source_generation, self.identity.mount_generation)
            if same_actor and fence_changed:
                self.cancel()
            elif not same_actor:
                self.reset()
        if identity != self.identity:
            if self.identity is not None and identity.goal_id != self.identity.goal_id:
                self.retired.add(self._invocation(self.identity))
            self.hidden = self.policy.zero_hidden()
            self.bus.reset(identity)
            self.previous = WireAction(effort=0)
            self.start_tick = int(view["tick"])
            self.last_tick, self.cached = None, None
            self.identity, self.goal = identity, goal
        elif self.goal != goal:
            self.cancel()
            raise ContractError("graph_goal", "a goal identity cannot change its parameters")
        try:
            frame = make_frame(view, goal, previous=self.previous, start_tick=self.start_tick,
                               source_generation=source_generation, mount_generation=mount_generation)
        except (ContractError, ValueError, KeyError, TypeError):
            self.cancel()
            raise
        if self._invocation(identity) in self.retired:
            return GraphDecision(WireAction(effort=0), frame, tuple(tuple(row) for row in self.hidden[0].tolist()),
                                 {"pending": len(self.bus.pending)}, "retired_goal")
        if self.last_tick is not None:
            if frame.tick == self.last_tick:
                if frame.observation_id != self.cached.frame.observation_id:
                    self.cancel()
                    raise ContractError("graph_observation", "the same tick cannot replace its issued source identity")
                return self.cached
            if frame.tick - self.last_tick != 6:
                self.cancel()
                raise ContractError("graph_cadence", "missed or reversed decision tick invalidates this recurrence")
        if frame.root_view.get("suspended") or frame.root_view.get("fault"):
            self.cancel()
            return GraphDecision(WireAction(effort=0), frame, tuple(tuple(row) for row in self.hidden[0].tolist()),
                                 {"pending": 0}, "uncontrollable_owner")
        if self.fault.id == "memory_reset" and frame.tick - self.start_tick == self.fault.reset_tick:
            self.reset_memory()
        before = tuple(tuple(row) for row in self.hidden[0].tolist())
        try:
            with torch.no_grad():
                inputs = self.policy.encoder.weight.new_tensor([frame.nodes])
                encoded, messages = self.policy.encode(inputs, self.hidden)
                sums, counts, communication = self.bus.exchange(messages[0].tolist(), frame.tick, identity)
                logits, self.hidden = self.policy.update(encoded, messages, self.hidden,
                    inputs.new_tensor([sums]), inputs.new_tensor(counts).reshape(1, 3, 1))
                output = self.policy.outputs(logits)[0].tolist()
            action = mapped_request(output, frame.root_view, goal, self.previous)
        except (ContractError, ValueError, RuntimeError, TypeError):
            self.cancel()
            raise
        self.previous, self.last_tick = action, frame.tick
        self.cached = GraphDecision(action, frame, before, communication)
        return self.cached
