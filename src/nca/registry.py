"""Small versioned plugin registry. Discovery never imports optional trainers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from .contracts import ContractError, Record, identifier, unique

PluginKind = Literal["adapter", "profile", "policy", "evaluator", "trainer", "skill", "cellular", "kinematics", "ik_solver"]
KINDS = ("adapter", "profile", "policy", "evaluator", "trainer", "skill", "cellular", "kinematics", "ik_solver")


@dataclass(frozen=True)
class PluginDescriptor(Record):
    kind: PluginKind
    id: str
    version: int
    capabilities: tuple[str, ...] = ()
    interface_version: Literal[1] = 1

    def validate(self) -> None:
        identifier(self.id, "plugin.id")
        if self.version < 1:
            raise ContractError("invalid_version", "plugin version must be positive")
        unique(self.capabilities, "plugin.capabilities")
        for capability in self.capabilities:
            identifier(capability, "plugin.capability")


class Registry:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, int], tuple[PluginDescriptor, Callable[..., Any]]] = {}

    def register(self, kind: PluginKind, id: str, version: int,
                 factory: Callable[..., Any], capabilities: Iterable[str] = ()) -> PluginDescriptor:
        description = PluginDescriptor(kind, id, version, tuple(capabilities))
        if not callable(factory):
            raise ContractError("invalid_factory", f"{kind}:{id} factory must be callable")
        key = (kind, id, version)
        if key in self._entries:
            raise ContractError("duplicate_registry_id", f"{kind}:{id}@{version}")
        self._entries[key] = description, factory
        return description

    def describe(self, kind: PluginKind, id: str, version: int = 1) -> PluginDescriptor:
        return self._entry(kind, id, version)[0]

    def _entry(self, kind: PluginKind, id: str, version: int):
        # Validate before dictionary lookup: bool and 1 are equal Python keys.
        PluginDescriptor(kind, id, version)
        try:
            return self._entries[(kind, id, version)]
        except KeyError as exc:
            versions = sorted(key[2] for key in self._entries if key[:2] == (kind, id))
            code = "incompatible_version" if versions else "unknown_plugin"
            raise ContractError(code, f"{kind}:{id}@{version}; available versions {versions}") from exc

    def create(self, kind: PluginKind, id: str, version: int = 1,
               *, require: Iterable[str] = (), **kwargs: Any) -> Any:
        description, factory = self._entry(kind, id, version)
        missing = set(require) - set(description.capabilities)
        if missing:
            raise ContractError("missing_capability", f"{kind}:{id}: {sorted(missing)}")
        return factory(**kwargs)

    def list(self, kind: PluginKind | None = None) -> tuple[PluginDescriptor, ...]:
        if kind is not None and kind not in KINDS:
            raise ContractError("unknown_plugin_kind", str(kind))
        return tuple(self._entries[key][0] for key in sorted(self._entries)
                     if kind is None or key[0] == kind)


registry = Registry()
