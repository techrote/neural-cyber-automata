"""Bounded one-hop communication faults; this never mutates physical topology."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from nca.contracts import ContractError, Record, digest
from .core import EDGES, GraphIdentity, finite


@dataclass(frozen=True)
class FaultSpec(Record):
    id: Literal["normal", "delay", "loss", "partition", "memory_reset"] = "normal"
    delay_ticks: int = 6
    loss_probability: float = .25
    reset_tick: int = 60

    def validate(self):
        if self.delay_ticks != 6 or self.loss_probability != .25 or self.reset_tick != 60:
            raise ContractError("graph_fault", "the preregistered fault parameters are fixed")


@dataclass(frozen=True)
class Message(Record):
    owner_sha256: str
    source: int
    destination: int
    observed_tick: int
    delivery_tick: int
    values: tuple[float, ...]

    def validate(self):
        digest(self.owner_sha256, "message.owner")
        if (self.source, self.destination) not in EDGES:
            raise ContractError("message_edge", "only declared one-hop edges can carry messages")
        if self.observed_tick < 0 or self.delivery_tick < self.observed_tick:
            raise ContractError("message_clock", "message delivery cannot precede its source")
        if len(self.values) != 8 or any(abs(value) > 1.000001 for value in self.values):
            raise ContractError("message_capacity", "the local message is eight bounded floats")


class MessageBus:
    """One serialized actor-owned queue with explicit age and no implicit hold."""
    def __init__(self, fault: FaultSpec = FaultSpec(), *, seed: int = 0, enabled: bool = True):
        if type(seed) is not int or not 0 <= seed < 2**64 or type(enabled) is not bool:
            raise ContractError("message_seed", "finite unsigned seed and Boolean communication switch required")
        self.fault, self.enabled = fault, enabled
        self.seed = seed
        self.random = random.Random(seed)
        self.owner: GraphIdentity | None = None
        self.pending: list[Message] = []
        self.last_tick: int | None = None
        self.totals = dict(sent=0, delivered=0, lost=0, partitioned=0, expired=0, resets=0)

    def reset(self, owner: GraphIdentity | None = None):
        self.pending.clear()
        self.last_tick = None
        self.owner = owner
        self.random.seed(self.seed)
        self.totals["resets"] += 1

    def clear_nodes(self, nodes):
        nodes = tuple(nodes)
        if any(type(node) is not int for node in nodes) or not set(nodes) <= {0, 1, 2}:
            raise ContractError("message_node", "unknown memory owner")
        nodes = set(nodes)
        self.pending = [message for message in self.pending
                        if message.source not in nodes and message.destination not in nodes]
        self.totals["resets"] += 1

    def exchange(self, local_messages, tick: int, owner: GraphIdentity):
        if not isinstance(owner, GraphIdentity):
            raise ContractError("message_owner", "a typed owner identity is required")
        if type(tick) is not int or tick < 0:
            raise ContractError("message_clock", "one exchange per increasing completed decision tick")
        if self.owner != owner:
            self.reset(owner)
        if self.last_tick is not None and tick <= self.last_tick:
            raise ContractError("message_clock", "one exchange per increasing completed decision tick")
        self.last_tick = tick
        if len(local_messages) != 3:
            raise ContractError("message_population", "exactly three local senders are required")
        local_messages = tuple(tuple(finite(value, "message") for value in values) for values in local_messages)
        if any(len(values) != 8 or any(abs(value) > 1.000001 for value in values) for values in local_messages):
            raise ContractError("message_capacity", "each local message contains eight bounded floats")
        sent_now = []
        if self.enabled:
            for source, destination in EDGES:
                message = Message(owner.sha256, source, destination, tick,
                                  tick + (self.fault.delay_ticks if self.fault.id == "delay" else 0),
                                  tuple(local_messages[source]))
                self.totals["sent"] += 1
                if self.fault.id == "partition" and {source, destination} == {0, 2}:
                    self.totals["partitioned"] += 1
                elif self.fault.id == "loss" and self.random.random() < self.fault.loss_probability:
                    self.totals["lost"] += 1
                else:
                    if len(self.pending) >= 16:
                        self.pending.clear()
                        raise ContractError("message_capacity", "the fixed communication queue is full")
                    self.pending.append(message)
                    sent_now.append(message)
        sums = [[0.0] * 8 for _ in range(3)]
        counts = [0, 0, 0]
        arrived, future = [], []
        for message in self.pending:
            if message.owner_sha256 != owner.sha256 or tick >= message.observed_tick + 12:
                self.totals["expired"] += 1
            elif message.delivery_tick <= tick:
                for channel, value in enumerate(message.values):
                    sums[message.destination][channel] += value
                counts[message.destination] += 1
                self.totals["delivered"] += 1
                arrived.append(message)
            else:
                future.append(message)
        self.pending = future
        return sums, counts, {"tick": tick, "sent": tuple(sent_now), "arrived": tuple(arrived),
                              "pending": len(future), "totals": dict(self.totals)}
