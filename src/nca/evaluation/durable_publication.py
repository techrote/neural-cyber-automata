"""Opt-in v2 diagnostic publication without temporary hard-linked siblings.

An exclusive final-path write may leave incomplete bytes if interrupted. Those
bytes are preserved and cannot be resumed or overwritten. Only a complete seal
after native replay, followed by the independent exact-inventory check, admits
evidence. Historical v1 writers and verifiers remain unchanged.
"""

from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Callable, Mapping

from nca.contracts import ContractError, canonical_json, sha256_json, strict_json_loads
from nca.evaluation.durable_journal import (
    file_identity, sync_directory, validate_record_chain,
)


PUBLICATION_METHOD = "exclusive-final-write/v1"
SEAL_SCHEMA = "nca-owner-journal-seal/v2"


def publish_json_v2(path: Path, value: Mapping[str, Any]) -> str:
    """Create once, flush, close and read back; never rename, link or repair.

    File existence is not a completion signal. Interrupted files remain invalid
    input to the enclosing receipt/strict JSON verification and cannot be retried.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode()
    with path.open("xb") as target:
        if target.write(payload) != len(payload):
            raise OSError("short immutable JSON write")
        target.flush()
        os.fsync(target.fileno())
    sync_directory(path.parent)
    identity = file_identity(path)
    if identity != {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}:
        raise ContractError("journal_publication", "exclusive final JSON bytes changed during publication")
    return identity["sha256"]


def _safe_relative(value: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value or "\0" in value
            or PurePosixPath(value).is_absolute()
            or any(part in (".", "..") for part in value.split("/"))
            or PurePosixPath(value).as_posix() != value):
        raise ContractError("journal_inventory", "a canonical relative artifact path is required")
    return value


def diagnostic_artifact_paths(rows, *, compressed=True, sealed=True, summary=False):
    """Derive the allowed paths from episode identities, never from disk files."""
    paths = {"manifest.json", "owner-journal.jsonl"}
    if compressed:
        paths.add("owner-journal.jsonl.gz")
    if sealed:
        paths.add("owner-journal-seal.json")
    if summary:
        paths.add("summary.json")
    identities = set()
    for row in rows:
        family, case = row.get("family"), row.get("case_id")
        if (not isinstance(family, str) or not isinstance(case, str)
                or "/" in family or "/" in case or "\\" in family or "\\" in case
                or not family or not case):
            raise ContractError("journal_inventory", "explicit episode family and case identity are required")
        key = (family, case)
        artifact = _safe_relative(row.get("artifact"))
        metric = _safe_relative(f"metrics/{family}-{case}.json")
        if artifact != f"episodes/{family}-{case}.json.gz":
            raise ContractError("journal_inventory", "episode artifact must match its declared family and case")
        if key in identities or artifact in paths or metric in paths:
            raise ContractError("journal_inventory", "episode and metric artifact paths must be unique")
        identities.add(key)
        paths.update((artifact, metric))
    return paths


def _listed_artifacts(root: Path):
    if root.is_symlink() or not root.is_dir():
        raise ContractError("journal_inventory", "the evidence root must be an owned directory")
    files, directories = set(), set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            directories.add(relative)
        elif stat.S_ISREG(mode):
            files.add(relative)
        else:
            raise ContractError("journal_inventory", "links and special files cannot enter the final inventory")
    return files, directories


def strict_artifact_inventory(root: Path, expected_paths):
    """Hash exactly declared regular files and reject all surplus paths.

    A staging file cannot become legitimate merely by appearing before a scan.
    Empty surplus directories, symlinks and changes during scanning also fail.
    No failed or unexpected bytes are removed by this function.
    """
    expected = {_safe_relative(value) for value in expected_paths}
    expected_directories = {parent.as_posix() for value in expected
        for parent in PurePosixPath(value).parents if parent.as_posix() != "."}
    listing = _listed_artifacts(root)
    if listing != (expected, expected_directories):
        raise ContractError("journal_inventory", "missing or surplus final artifact paths")
    hashes = {}
    for relative in sorted(expected):
        path = root / relative
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ContractError("journal_inventory", "an artifact was replaced by a link or special file")
        identity = file_identity(path)
        after = path.lstat()
        fingerprint = lambda value: (value.st_dev, value.st_ino, value.st_mode,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if (not stat.S_ISREG(after.st_mode) or fingerprint(before) != fingerprint(after)
                or identity["bytes"] != after.st_size):
            raise ContractError("journal_inventory", "artifact changed while its final identity was read")
        hashes[relative] = identity["sha256"]
    if _listed_artifacts(root) != listing:
        raise ContractError("journal_inventory", "final artifact paths changed during verification")
    return hashes


def diagnostic_artifact_inventory(root: Path, rows, *, compressed=True, sealed=True, summary=False):
    return strict_artifact_inventory(root, diagnostic_artifact_paths(rows,
        compressed=compressed, sealed=sealed, summary=summary))


def seal_owner_journal_v2(raw_path: Path, *, expected_counts, expected_rows,
                          durability, validate_owner: Callable, families, artifact_root: Path):
    """Seal one closed invocation with no staging-name lifecycle or cleanup."""
    if (raw_path != artifact_root / "owner-journal.jsonl"
            or durability.get("closed") is not True or durability.get("poisoned") is not False
            or durability.get("record_fsyncs") != durability.get("records")):
        raise ContractError("journal_durability", "uncertain live persistence cannot be sealed")
    before = diagnostic_artifact_inventory(artifact_root, expected_rows, compressed=False, sealed=False)
    raw = validate_record_chain(raw_path)
    if (raw["records"] != durability.get("records")
            or raw["chain_sha256"] != durability.get("chain_sha256")):
        raise ContractError("journal_durability", "closed bytes differ from the writer's durable receipt")
    compressed_path = artifact_root / "owner-journal.jsonl.gz"
    with compressed_path.open("xb") as target:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=target, mtime=0) as compressed:
            with raw_path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    compressed.write(block)
        target.flush()
        os.fsync(target.fileno())
    directory_fsync = sync_directory(artifact_root)
    if validate_record_chain(compressed_path, compressed=True) != raw:
        raise ContractError("journal_transport", "compressed journal differs from the durable original")
    validation = validate_owner(compressed_path, expected_counts=expected_counts,
        expected_rows=expected_rows, families=families, artifact_root=artifact_root)
    compressed_identity = file_identity(compressed_path)
    after = diagnostic_artifact_inventory(artifact_root, expected_rows, sealed=False)
    if ({name: digest for name, digest in after.items() if name != compressed_path.name} != before
            or after[compressed_path.name] != compressed_identity["sha256"]):
        raise ContractError("journal_inventory", "an artifact changed while the invocation was sealed")
    seal = {"schema": SEAL_SCHEMA, "raw": {"path": raw_path.name, **raw},
        "compressed": {"path": compressed_path.name, **compressed_identity},
        "durability": dict(durability), "owner_validation": validation,
        "expected_counts_sha256": sha256_json(expected_counts),
        "expected_rows_sha256": sha256_json(expected_rows), "families": list(families),
        "publication": {"method": PUBLICATION_METHOD, "directory_fsync": directory_fsync,
                        "preseal_inventory_sha256": sha256_json(after)}}
    seal_path = artifact_root / "owner-journal-seal.json"
    seal_sha256 = publish_json_v2(seal_path, seal)
    final = diagnostic_artifact_inventory(artifact_root, expected_rows)
    if {name: digest for name, digest in final.items() if name != seal_path.name} != after:
        raise ContractError("journal_inventory", "an artifact changed during final seal publication")
    return {"verified": True, "seal_sha256": seal_sha256,
        "raw_sha256": raw["sha256"], "compressed_sha256": compressed_identity["sha256"],
        "owner_validation": validation}


def verify_sealed_journal_v2(root: Path, *, expected_counts, expected_rows,
                            validate_owner: Callable, families, summary=True):
    """Strict, read-only native replay and inventory checks after worker exit."""
    before = diagnostic_artifact_inventory(root, expected_rows, summary=summary)
    seal_path = root / "owner-journal-seal.json"
    seal = strict_json_loads(seal_path.read_bytes())
    publication = seal.get("publication", {})
    preseal = {name: digest for name, digest in before.items()
               if name not in ("owner-journal-seal.json", "summary.json")}
    if (seal.get("schema") != SEAL_SCHEMA
            or seal.get("expected_counts_sha256") != sha256_json(expected_counts)
            or seal.get("expected_rows_sha256") != sha256_json(expected_rows)
            or seal.get("families") != list(families)
            or set(publication) != {"method", "directory_fsync", "preseal_inventory_sha256"}
            or publication.get("method") != PUBLICATION_METHOD
            or type(publication.get("directory_fsync")) is not bool
            or publication.get("preseal_inventory_sha256") != sha256_json(preseal)):
        raise ContractError("journal_seal", "seal does not bind its publication, result and declaration")
    raw, compressed = root / "owner-journal.jsonl", root / "owner-journal.jsonl.gz"
    raw_chain = validate_record_chain(raw)
    compressed_identity = file_identity(compressed)
    if (seal.get("raw") != {"path": raw.name, **raw_chain}
            or seal.get("compressed") != {"path": compressed.name, **compressed_identity}
            or validate_record_chain(compressed, compressed=True) != raw_chain):
        raise ContractError("journal_seal", "retained raw or compressed bytes differ from their seal")
    durability = seal.get("durability", {})
    if (durability.get("closed") is not True or durability.get("poisoned") is not False
            or durability.get("records") != raw_chain["records"]
            or durability.get("record_fsyncs") != raw_chain["records"]
            or durability.get("chain_sha256") != raw_chain["chain_sha256"]):
        raise ContractError("journal_seal", "seal lacks its complete persistence receipt")
    validation = validate_owner(compressed, expected_counts=expected_counts,
        expected_rows=expected_rows, families=families, artifact_root=root)
    if validation != seal.get("owner_validation"):
        raise ContractError("journal_seal", "independent owner/episode replay differs from the seal")
    if diagnostic_artifact_inventory(root, expected_rows, summary=summary) != before:
        raise ContractError("journal_inventory", "artifact changed during independent native replay")
    return {"verified": True, "seal_sha256": before[seal_path.name],
        "raw_sha256": raw_chain["sha256"], "compressed_sha256": compressed_identity["sha256"],
        "owner_validation": validation}
