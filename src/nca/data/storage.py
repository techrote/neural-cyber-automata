"""Bounded compressed episode artifacts with no replacement or shared appends."""

from __future__ import annotations

import gzip
import hashlib
import os
import tempfile
from pathlib import Path

from nca.contracts import ContractError, canonical_json, strict_json_loads
from .records import EpisodeRecord

MAX_EPISODE_BYTES = 32 * 1024 * 1024


def write_episode(path: str | Path, episode: EpisodeRecord) -> str:
    target = Path(path)
    payload = (canonical_json({"episode": episode.to_dict(), "sha256": episode.sha256}) + "\n").encode()
    if len(payload) > MAX_EPISODE_BYTES:
        raise ContractError("dataset_size", "episode exceeds 32 MiB; use smaller episode segments")
    data = gzip.compress(payload, compresslevel=9, mtime=0) if target.suffix == ".gz" else payload
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".episode-", delete=False) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise ContractError("immutable_dataset", f"episode file {target.name} already exists") from exc
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def read_episode(path: str | Path) -> EpisodeRecord:
    target = Path(path)
    if target.stat().st_size > MAX_EPISODE_BYTES:
        raise ContractError("dataset_size", "episode file exceeds bounded size")
    opener = gzip.open if target.suffix == ".gz" else open
    try:
        with opener(target, "rb") as stream:
            payload = stream.read(MAX_EPISODE_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise ContractError("dataset_encoding", "episode compression is invalid") from exc
    value = strict_json_loads(payload, max_bytes=MAX_EPISODE_BYTES)
    if not isinstance(value, dict) or set(value) != {"episode", "sha256"}:
        raise ContractError("dataset_schema", "episode envelope fields differ")
    episode = EpisodeRecord.from_dict(value["episode"])
    if value["sha256"] != episode.sha256:
        raise ContractError("dataset_hash", "episode content differs from receipt")
    return episode
