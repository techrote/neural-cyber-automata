"""Named random streams and explicit, immutable scenario partitions."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .contracts import ContractError, Record, identifier, unique


def derive_seed(seed: int, stream: str, index: int = 0) -> int:
    """Order-independent derivation; uses no process-global random state."""
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ContractError("invalid_seed", "seed must be an unsigned 64-bit integer")
    if type(index) is not int or index < 0:
        raise ContractError("invalid_seed", "stream index must be a nonnegative integer")
    if type(stream) is not str:
        raise ContractError("invalid_seed", "stream must be text")
    identifier(stream, "random.stream")
    payload = f"nca-seed-v1\0{seed}\0{stream}\0{index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def random_stream(seed: int, stream: str, index: int = 0) -> random.Random:
    return random.Random(derive_seed(seed, stream, index))


@dataclass(frozen=True)
class ScenarioSplit(Record):
    """Whole scenario-family partitions, not a claim of independent seed holdout."""

    training: tuple[str, ...]
    tuning: tuple[str, ...]
    evaluation: tuple[str, ...]
    version: int = 1

    def validate(self) -> None:
        if self.version != 1:
            raise ContractError("unsupported_version", "ScenarioSplit version")
        all_values = self.training + self.tuning + self.evaluation
        for family in all_values:
            identifier(family, "scenario.family")
        unique(all_values, "scenario split families must be disjoint")
        if not self.evaluation:
            raise ContractError("empty_split", "an evaluation partition must be named")

    def partition(self, family: str) -> str:
        for name in ("training", "tuning", "evaluation"):
            if family in getattr(self, name):
                return name
        raise ContractError("unknown_scenario", family)
