"""Immutable per-run provenance, content verification and atomic publication."""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .contracts import (ContractError, Record, digest, identifier, nonnegative,
                        sha256_json, strict_json_loads)
from .randomness import ScenarioSplit


def immutable_bytes(path: str | Path, payload: bytes) -> str:
    """Publish complete bytes without ever replacing an existing artifact.

    Shared by the original learner and optional IK lane. The hard link is the
    commit operation; a filesystem without this primitive fails visibly.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".bundle-", delete=False) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise ValueError(f"immutable artifact already exists: {target.name}") from exc
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def immutable_json(path: str | Path, value: Mapping[str, Any]) -> str:
    from .contracts import canonical_json
    return immutable_bytes(path, (canonical_json(value) + "\n").encode())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _run_identifier(run_id: str) -> None:
    identifier(run_id, "run_id")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    if ":" in run_id or run_id.split(".", 1)[0].upper() in reserved:
        raise ContractError("invalid_run_id", "run_id must be portable as a filename")


@dataclass(frozen=True)
class ArtifactRef(Record):
    uri: str
    content_sha256: str
    size_bytes: int
    kind: str

    def validate(self) -> None:
        digest(self.content_sha256, "artifact.content_sha256")
        nonnegative(self.size_bytes, "artifact.size_bytes")
        identifier(self.kind, "artifact.kind")
        if not self.uri or self.uri.startswith(("/", "\\", "file:")) or re.match(r"^[A-Za-z]:", self.uri):
            raise ContractError("private_path", "artifact URI must be relative or an artifact-store URL")


@dataclass(frozen=True)
class RunBudget(Record):
    max_decisions: int = 2000
    max_seconds: float = 120.0

    def validate(self) -> None:
        if self.max_decisions < 1 or self.max_seconds <= 0:
            raise ContractError("invalid_budget", "both decision and wall-time ceilings must be positive")

    def exhausted(self, decisions: int, seconds: float) -> bool:
        if type(decisions) is not int or decisions < 0 or type(seconds) not in (float, int):
            raise ContractError("invalid_measurement", "budget counters must be nonnegative numbers")
        if not math.isfinite(seconds) or seconds < 0:
            raise ContractError("invalid_measurement", "elapsed time must be finite and nonnegative")
        return decisions >= self.max_decisions or seconds >= self.max_seconds


@dataclass(frozen=True)
class RunManifest(Record):
    run_id: str
    experiment: str
    learner_revision: str
    host_revision: str
    profile_sha256: str
    resolved_config: Mapping[str, Any]
    environment: Mapping[str, Any]
    seeds: tuple[int, ...]
    scenario_split: ScenarioSplit
    status: Literal["prepared", "running", "completed", "failed", "capped", "canceled", "inconclusive"]
    device: str = "cpu"
    gameplay_lane: Literal["recorded_fixture", "sampled_ACT003", "physical_ACT004"] = "recorded_fixture"
    execution_lane: Literal["stepped", "realtime", "offline"] = "offline"
    observation_schema: str = "nca-observation-v1"
    action_schema: str = "nca-command-v1"
    suite: str = "cpu-conformance-v1"
    budget: RunBudget = RunBudget()
    decisions: int = 0
    runtime_seconds: float | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    failures: tuple[str, ...] = ()
    parent_run_id: str | None = None
    continuation: Literal["fresh_episode", "none"] = "none"
    created_utc: str = field(default_factory=utc_now)
    config_sha256: str = ""
    schema_version: Literal[1] = 1

    def validate(self) -> None:
        for name in ("run_id", "experiment", "suite", "observation_schema", "action_schema"):
            identifier(getattr(self, name), name)
        _run_identifier(self.run_id)
        for name in ("learner_revision", "host_revision"):
            if not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", getattr(self, name)):
                raise ContractError("invalid_revision", f"{name} must be a full source revision")
        digest(self.profile_sha256, "profile_sha256")
        actual_config_hash = sha256_json(self.resolved_config)
        if self.config_sha256 and self.config_sha256 != actual_config_hash:
            raise ContractError("config_mismatch", "resolved configuration digest differs")
        object.__setattr__(self, "config_sha256", actual_config_hash)
        if not self.seeds or any(not 0 <= seed < 2**64 for seed in self.seeds):
            raise ContractError("invalid_seed", "a nonempty unsigned 64-bit seed set is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ContractError("duplicate_seed", "run seeds must be distinct")
        nonnegative(self.decisions, "decisions")
        if self.runtime_seconds is not None:
            nonnegative(self.runtime_seconds, "runtime_seconds")
        if self.status not in ("prepared", "running") and self.runtime_seconds is None:
            raise ContractError("missing_measurement", "finished run must record elapsed time")
        if self.status in ("failed", "capped", "canceled", "inconclusive") and not self.failures:
            raise ContractError("missing_reason", "non-successful run must retain reasons")
        if self.parent_run_id is not None:
            identifier(self.parent_run_id, "parent_run_id")
            if self.parent_run_id == self.run_id or self.continuation != "fresh_episode":
                raise ContractError("invalid_lineage", "resume requires a new ID and fresh-episode declaration")
        elif self.continuation != "none":
            raise ContractError("invalid_lineage", "continuation requires parent run identity")
        try:
            created = datetime.fromisoformat(self.created_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ContractError("invalid_clock", "created_utc must be an ISO timestamp") from exc
        if created.utcoffset() is None or created.utcoffset().total_seconds() != 0:
            raise ContractError("invalid_clock", "created_utc requires explicit UTC")


class RunStore:
    """Unique files, not concurrent appends to a shared Markdown ledger.

    An atomic hard-link publish refuses replacement even under concurrent writers.
    If the filesystem lacks hard links, publication fails visibly rather than
    falling back to a replacement-prone write. Temporary files are always removed.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write(self, manifest: RunManifest) -> Path:
        if not isinstance(manifest, RunManifest):
            raise ContractError("invalid_manifest", "RunStore requires a validated RunManifest")
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / f"{manifest.run_id}.json"
        from .contracts import canonical_json
        envelope = {"manifest": manifest.to_dict(), "record_sha256": manifest.sha256}
        payload = (canonical_json(envelope) + "\n").encode("utf-8")
        if len(payload) > 1_048_576:
            raise ContractError("size_limit", "run record exceeds 1 MiB")
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, prefix=".publish-", delete=False) as stream:
                temporary = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ContractError("immutable_run", f"run {manifest.run_id} already exists") from exc
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
        return destination

    def read(self, run_id: str) -> RunManifest:
        _run_identifier(run_id)
        path = self.root / f"{run_id}.json"
        # Refuse oversized records before reading the entire file into memory.
        if path.stat().st_size > 1_048_576:
            raise ContractError("size_limit", "run record exceeds 1 MiB")
        envelope = strict_json_loads(path.read_bytes())
        if not isinstance(envelope, dict) or set(envelope) != {"manifest", "record_sha256"}:
            raise ContractError("invalid_record", "run envelope fields differ")
        manifest = RunManifest.from_dict(envelope["manifest"])
        if envelope["record_sha256"] != manifest.sha256:
            raise ContractError("hash_mismatch", f"run {run_id} content was changed")
        if manifest.run_id != run_id:
            raise ContractError("identity_mismatch", "filename differs from manifest run ID")
        return manifest

    def list(self) -> tuple[RunManifest, ...]:
        if not self.root.exists():
            return ()
        return tuple(self.read(path.stem) for path in sorted(self.root.glob("*.json")))

    def verify(self) -> tuple[str, ...]:
        return tuple(manifest.run_id for manifest in self.list())
