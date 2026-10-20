"""Optional CPU candidates for the gated morphology comparison.

The models are trainable, but no trained morphology policy is supplied. A
numeric forward or locality test is not a transfer experiment. All weights are
shared across actor instances; recurrent state is always caller/instance owned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from nca.contracts import ContractError, WireAction
from .core import (
    ACTION_NAMES, COMMAND_NAMES, EDGE_NAMES, MAX_JOINTS, MAX_PARTS, NODE_NAMES, ROOT_NAMES,
    MorphologyAdapter, MorphologyFrame, MorphologyGoal, MorphologyIdentity,
    SemanticAction, StateFence, map_action,
)

PADDED_FEATURES = (MAX_PARTS * (len(NODE_NAMES) * 2 + 2)
                   + MAX_JOINTS * (len(EDGE_NAMES) * 2 + MAX_PARTS * 2 + 2)
                   + len(ROOT_NAMES) * 2 + len(ACTION_NAMES) + len(COMMAND_NAMES) + MAX_PARTS)
GRAPH_NODE_FEATURES = len(NODE_NAMES) * 2 + len(ROOT_NAMES) * 2 + len(ACTION_NAMES) + len(COMMAND_NAMES) + 1
GRAPH_EDGE_FEATURES = len(EDGE_NAMES) * 2 + 1  # Authored a→b orientation, independent of slot permutation.


@dataclass(frozen=True)
class CandidateOutput:
    logits: torch.Tensor
    value: torch.Tensor
    hidden: torch.Tensor
    loss_mask: bool


def _cpu_finite(tensor: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
        raise ContractError("morphology_tensor", f"{name} must be a CPU float32 tensor of shape {shape}")
    if not bool(torch.isfinite(tensor).all()):
        raise ContractError("morphology_tensor", f"{name} contains nonfinite values")


def _padding(hidden: torch.Tensor) -> CandidateOutput:
    # There is no GRU step or value target on padding, including absent timesteps
    # inside a variable-length ordered batch. This is distinct from absent nodes.
    return CandidateOutput(torch.zeros(len(ACTION_NAMES)), torch.tensor(0.), hidden.clone(), False)


class MaskedRecurrentPolicy(nn.Module):
    """Central access to the same eight copied local slots and seven joints."""
    family = "padded_recurrent"
    hidden_size = 32
    projection_size = 10

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(PADDED_FEATURES, self.projection_size)
        self.memory = nn.GRUCell(self.projection_size, self.hidden_size)
        self.readout = nn.Linear(self.hidden_size, len(ACTION_NAMES) + 1)

    def zero_hidden(self) -> torch.Tensor:
        return torch.zeros(1, self.hidden_size)

    def forward_frame(self, frame: MorphologyFrame | None, hidden: torch.Tensor) -> CandidateOutput:
        _cpu_finite(hidden, (1, self.hidden_size), "padded hidden")
        if frame is None:
            return _padding(hidden)
        values = torch.tensor(frame.padded(), dtype=torch.float32).unsqueeze(0)
        following = self.memory(torch.tanh(self.encoder(values)), hidden)
        output = self.readout(following)[0]
        return CandidateOutput(output[:-1], output[-1], following, True)

    def dense_macs(self, frame: MorphologyFrame) -> int:
        del frame
        return (PADDED_FEATURES * self.projection_size
                + 3 * self.hidden_size * (self.projection_size + self.hidden_size)
                + self.hidden_size * (len(ACTION_NAMES) + 1))


class SharedGraphPolicy(nn.Module):
    """Synchronous neighbor messages and a root readout; no global pooling.

    Each node gets its own motion/geometry and explicit caller goal. Root-only
    resources, contact and rays enter only at the root. Two rounds permit two
    graph hops of current information. Memory may carry older information that
    traversed earlier rounds; it does not grant instant global current state.
    """
    family = "shared_graph"
    hidden_size = 32
    message_size = 16

    def __init__(self, rounds: int = 2):
        super().__init__()
        if type(rounds) is not int or not 1 <= rounds <= 2:
            raise ContractError("message_budget", "this candidate declares one or two synchronous rounds")
        self.rounds = rounds
        self.encoder = nn.Linear(GRAPH_NODE_FEATURES, self.hidden_size)
        self.message = nn.Linear(self.hidden_size + GRAPH_EDGE_FEATURES, self.message_size)
        self.memory = nn.GRUCell(self.hidden_size + self.message_size, self.hidden_size)
        self.readout = nn.Linear(self.hidden_size, len(ACTION_NAMES) + 1)

    def zero_hidden(self) -> torch.Tensor:
        return torch.zeros(MAX_PARTS, self.hidden_size)

    @staticmethod
    def node_inputs(frame: MorphologyFrame) -> torch.Tensor:
        rows = []
        for i in range(MAX_PARTS):
            root = i == frame.root_index
            rows.append((*frame.nodes[i], *[float(v) for v in frame.node_valid[i]],
                         *(frame.root_values if root else (0.,) * len(ROOT_NAMES)),
                         *[float(root and v) for v in frame.root_valid],
                         *[float(root and v) for v in frame.action_mask], *frame.command, float(root)))
        return torch.tensor(rows, dtype=torch.float32)

    def forward_frame(self, frame: MorphologyFrame | None, hidden: torch.Tensor,
                      *, node_inputs: torch.Tensor | None = None) -> CandidateOutput:
        _cpu_finite(hidden, (MAX_PARTS, self.hidden_size), "graph hidden")
        if frame is None:
            return _padding(hidden)
        values = self.node_inputs(frame) if node_inputs is None else node_inputs
        _cpu_finite(values, (MAX_PARTS, GRAPH_NODE_FEATURES), "node inputs")
        mask = torch.tensor(frame.active, dtype=torch.float32).unsqueeze(-1)
        embedded = torch.tanh(self.encoder(values)) * mask
        # Current own sensors enter before round one, so r rounds have exactly
        # r-hop current reach. Inactive nodes erase memory and cannot emit bias.
        state = torch.tanh(embedded + hidden * mask) * mask
        for _ in range(self.rounds):
            incoming: list[list[torch.Tensor]] = [[] for _ in range(MAX_PARTS)]
            for edge in frame.edges:
                if not edge.active:
                    continue
                for source, destination, direction in ((edge.source, edge.destination, 1.),
                                                        (edge.destination, edge.source, -1.)):
                    local = torch.tensor((*edge.values, *[float(v) for v in edge.valid], direction), dtype=torch.float32)
                    incoming[destination].append(torch.tanh(self.message(torch.cat((state[source], local)))))
            aggregate = torch.stack([torch.stack(items).mean(0) if items else torch.zeros(self.message_size)
                                     for items in incoming])
            state = self.memory(torch.cat((embedded, aggregate), dim=-1), state) * mask
        output = self.readout(state[frame.root_index])
        return CandidateOutput(output[:-1], output[-1], state, True)

    def dense_macs(self, frame: MorphologyFrame) -> int:
        # Conservative fixed-padding arithmetic: GRUs/encoders compute eight
        # slots even when masks then zero them. Message work uses live edges.
        edges = sum(edge.active for edge in frame.edges) * 2
        return (MAX_PARTS * GRAPH_NODE_FEATURES * self.hidden_size
                + self.rounds * (edges * (self.hidden_size + GRAPH_EDGE_FEATURES) * self.message_size
                                 + MAX_PARTS * 3 * self.hidden_size * (self.hidden_size * 2 + self.message_size))
                + self.hidden_size * (len(ACTION_NAMES) + 1))


def candidate_cost(model: MaskedRecurrentPolicy | SharedGraphPolicy, frame: MorphologyFrame) -> dict[str, Any]:
    return {"family": model.family, "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "dense_multiply_accumulates_per_decision": model.dense_macs(frame),
            "activation_and_dispatch_cost_included": False, "measured_runtime_seconds": None,
            "hidden_float32_values": model.zero_hidden().numel(),
            "message_rounds": getattr(model, "rounds", 0), "learned_transfer_measured": False}


def deterministic_action(logits: torch.Tensor) -> SemanticAction:
    _cpu_finite(logits, (len(ACTION_NAMES),), "actor logits")
    continuous = torch.tanh(logits[:3]).tolist()
    effort = float(torch.sigmoid(logits[3]))
    return SemanticAction(*continuous, effort, *(bool(value >= 0) for value in logits[4:].tolist()))


@dataclass(frozen=True)
class MorphologyDecision:
    action: WireAction
    status: str
    reason: str
    cancel_owner: bool
    source: MorphologyIdentity | None = None
    source_sha256: str | None = None
    semantic_request: SemanticAction | None = None
    capability_masked: tuple[str, ...] = ()
    reset_reasons: tuple[str, ...] = ()
    frame_sha256: str | None = None
    sampling_mode: str = "deterministic_unqualified_candidate"
    log_probability: None = None
    helper_lease_ticks: int = 12
    goal_deadline_tick: int | None = None


class MorphologyRuntime:
    """An unqualified candidate invocation with independent hidden state.

    No optimizer, train/evaluation evidence or deployment promotion is implied.
    A refused decision requires the real owner to be cancelled by dispatch.
    """
    def __init__(self, model: MaskedRecurrentPolicy | SharedGraphPolicy, adapter: MorphologyAdapter):
        self.model = model
        self.adapter = adapter
        self.fence = StateFence()
        self.hidden = model.zero_hidden()
        self.previous = SemanticAction()
        self.pending: MorphologyDecision | None = None
        self.owner_cleanup_unconfirmed = False
        self._owner_may_be_active = False
        self._owner_identity: MorphologyIdentity | None = None

    def cancel(self, reason: str = "cancelled") -> MorphologyDecision:
        self.fence.cancel()
        self.hidden = self.model.zero_hidden()
        self.previous = SemanticAction()
        self.pending = None
        self.owner_cleanup_unconfirmed |= self._owner_may_be_active
        return MorphologyDecision(WireAction(effort=0), "refused", reason, True)

    def _new_world_reset(self, identity: MorphologyIdentity) -> bool:
        previous = self._owner_identity
        return (previous is not None and identity.tick == 0 and identity.control_generation == 0
                and (identity.run_id, identity.epoch) != (previous.run_id, previous.epoch))

    def _clear_owner_fence(self) -> None:
        self.owner_cleanup_unconfirmed = False
        self._owner_may_be_active = False
        self._owner_identity = None

    def confirm_owner_cancel(self, observation: Mapping[str, Any]) -> None:
        """Release an uncertain-owner latch only with a valid revocation receipt.

        Merely returning from a cleanup callback or changing the caller's goal
        ID is not evidence that the old physical helper stopped consuming input.
        """
        if not isinstance(observation, Mapping):
            raise ContractError("morphology_cancel_receipt", "original cancellation observation required")
        previous = self._owner_identity
        current = self.fence.identity
        counters = current or previous
        identity = self.adapter.identity(observation,
                                         source_generation=1 if counters is None else counters.source_generation,
                                         mount_generation=None if counters is None else counters.mount_generation)
        if previous is not None:
            if (identity.lifetime != previous.lifetime or identity.tick < previous.tick
                    or identity.control_generation <= previous.control_generation):
                raise ContractError("morphology_cancel_receipt", "owner generation was not demonstrably revoked")
        if current is not None and (identity.lifetime != current.lifetime or identity.tick < current.tick
                                    or identity.control_generation < current.control_generation):
            raise ContractError("morphology_cancel_receipt", "cancellation cannot rewind the observed incarnation")
        if identity.control_generation < 1:
            raise ContractError("morphology_cancel_receipt", "a cancellation must revoke an existing owner generation")
        self.cancel("owner_cancel_confirmed")
        # Preserve the receipt's authority high-water mark as well as the latch.
        # A fresh goal cannot immediately resubmit the pre-cancel view.
        self.fence.identity = identity
        self._clear_owner_fence()

    def step(self, view: Mapping[str, Any], goal: MorphologyGoal, *, source_generation: int = 1,
             mount_generation: int | None = None, current_tick: int | None = None) -> MorphologyDecision:
        try:
            identity = self.adapter.identity(view, source_generation=source_generation, mount_generation=mount_generation)
            new_world_reset = self._new_world_reset(identity)
            if self.owner_cleanup_unconfirmed and not new_world_reset:
                raise ContractError("owner_cleanup_unconfirmed", "confirmed cancellation or a new host reset is required")
            if self._owner_may_be_active and not new_world_reset:
                previous = self._owner_identity
                same_source = previous is not None and all(
                    getattr(previous, name) == getattr(identity, name) for name in
                    ("run_id", "epoch", "actor_id", "model", "profile_version", "profile_sha256",
                     "control_generation", "source_generation", "topology_sha256", "mount_generation"))
                if not same_source or goal != self.fence.goal:
                    raise ContractError("owner_boundary_requires_cancel",
                                        "a live helper must be revoked before changing invocation or source")
            reset = self.fence.accept(identity, goal)
            if reset:
                self.hidden = self.model.zero_hidden()
                self.previous = SemanticAction()
            frame = self.adapter.frame(view, goal, previous=self.previous, started_tick=self.fence.start_tick,
                                       source_generation=source_generation, mount_generation=mount_generation,
                                       current_tick=current_tick)
            if new_world_reset:
                # Validate the complete reset frame before retiring uncertainty
                # about a previous physical world. A mere new goal is insufficient.
                self._clear_owner_fence()
            with torch.no_grad():
                output = self.model.forward_frame(frame, self.hidden)
            request = self.select_action(output, frame)
            action, masked = map_action(self.adapter, request, view, goal)
            self.hidden, self.previous = output.hidden.detach().clone(), request
            deadline = self.fence.start_tick + goal.deadline_ticks
            self.pending = MorphologyDecision(action, "accepted_unqualified", "candidate_request", False, identity,
                                              self.adapter.source_sha256(view), request, masked, reset, frame.policy_sha256,
                                              helper_lease_ticks=min(12, deadline - identity.tick), goal_deadline_tick=deadline)
            return self.pending
        except (ContractError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            return self.cancel(exc.code if isinstance(exc, ContractError) else "invalid_candidate_input")

    def select_action(self, output: CandidateOutput, frame: MorphologyFrame) -> SemanticAction:
        """The preparation runtime uses the unchanged deterministic readout.

        A separately declared training sampler can specialize only this action
        choice while retaining the same source, lifetime and owner fences.
        """
        del frame
        return deterministic_action(output.logits)


def dispatch(env, runtime: MorphologyRuntime, decision: MorphologyDecision, view: Mapping[str, Any]):
    """Preserve inference source identity and cancel on any construction/refusal error."""
    if decision.cancel_owner:
        runtime.cancel(decision.reason)
        receipt = env.cancel()
        runtime.confirm_owner_cancel(receipt)
        return receipt
    try:
        source = decision.source
        if (runtime.owner_cleanup_unconfirmed or runtime.pending is not decision
                or source is None or decision.status != "accepted_unqualified"
                or runtime.adapter.identity(view, source_generation=source.source_generation,
                                            mount_generation=source.mount_generation) != source
                or runtime.adapter.source_sha256(view) != decision.source_sha256):
            raise ContractError("morphology_source_mismatch", "a decision cannot be refreshed with a new view")
        # Consume the local decision before the RPC. Ambiguous transport cannot
        # authorize submitting it again under a fresh packet sequence.
        runtime.pending = None
        runtime._owner_may_be_active = True
        runtime._owner_identity = source
        packet = env.command(decision.action.to_dict(), observation=view, lease_ticks=decision.helper_lease_ticks)
        transition = env.step_packet(packet)
        # Native rejection is reported in the completed transition, not raised
        # by CybersandEnv. A returned RPC alone cannot confirm our request.
        if not isinstance(transition, (tuple, list)) or len(transition) != 5:
            raise ContractError("morphology_dispatch_receipt", "complete native transition required")
        after, _, terminated, truncated, info = transition
        if not isinstance(after, Mapping) or not isinstance(info, Mapping):
            raise ContractError("morphology_dispatch_receipt", "native observation and trace required")
        from .conformance import validate_interval
        validate_interval(packet, view, after, terminated, truncated, info, allow_terminal=True)
        if any(after.get(name) != view.get(name) for name in
               ("version", "representation", "observation_scope", "fault", "run_id", "epoch", "actor_id",
                "model", "rig_id", "profile_version")):
            raise ContractError("morphology_dispatch_boundary", "native result changed the issued actor or profile")
        if terminated or truncated:
            following = runtime.adapter.terminal_identity(after, source_generation=source.source_generation,
                                                          mount_generation=source.mount_generation)
            if following.control_generation != source.control_generation + 1:
                raise ContractError("morphology_dispatch_boundary", "terminal result did not revoke owner generation")
            # The terminal action is unavailable after revocation. Erase local
            # recurrence and require a new real episode before another request.
            runtime.cancel("native_terminal_requires_reset")
        else:
            following = runtime.adapter.identity(after, source_generation=source.source_generation,
                                                 mount_generation=source.mount_generation)
            if (following.control_generation != source.control_generation
                    or following.topology_sha256 != source.topology_sha256):
                raise ContractError("morphology_dispatch_boundary", "native result changed owner or topology")
        return transition
    except BaseException:
        runtime.cancel()
        try:
            receipt = env.cancel()
            runtime.confirm_owner_cancel(receipt)
        except Exception:
            pass
        raise
