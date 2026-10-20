"""Immutable finite morphology checkpoints with explicit source and phase lineage."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import torch

from nca.contracts import ContractError, canonical_json, digest, sha256_json
from .artifacts import immutable_bytes
from .morphology_learning import new_policy, policy_sha256, policy_specification

CHECKPOINT_SCHEMA = "nca-morphology-checkpoint/v1"
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024


def _metadata(value, model):
    required = {"family", "training_seed", "phase", "config_sha256", "manifest_sha256",
                "learner_revision", "learner_sources_sha256", "dataset_sha256", "training_privilege"}
    if (not isinstance(value, dict) or not required <= set(value) or value["family"] != model.family
            or value["training_seed"] not in (6113, 6229, 6337) or type(value["training_seed"]) is not int
            or value["phase"] not in ("initial", "bc", "shared_update", "shared_final", "adapted_final")
            or value["training_privilege"] != "none"):
        raise ContractError("morphology_checkpoint", "complete exact candidate/seed/phase/source lineage required")
    for name in ("config_sha256", "manifest_sha256", "learner_sources_sha256", "dataset_sha256"):
        digest(value[name], "checkpoint." + name)
    if (not isinstance(value["learner_revision"], str) or len(value["learner_revision"]) != 40
            or any(c not in "0123456789abcdef" for c in value["learner_revision"])):
        raise ContractError("morphology_checkpoint", "the original public source revision is required")
    canonical_json(value)


def checkpoint_file(root: Path, receipt):
    name = receipt["file"]
    root = root.resolve()
    path = (root / name).resolve()
    if (not isinstance(name, str) or not path.is_relative_to(root) or path == root
            or path.relative_to(root).as_posix() != name):
        raise ContractError("morphology_checkpoint", "canonical contained checkpoint paths required")
    if (type(receipt["bytes"]) is not int or not 0 < receipt["bytes"] <= MAX_CHECKPOINT_BYTES
            or path.stat().st_size != receipt["bytes"]):
        raise ContractError("morphology_checkpoint", "checkpoint file size differs from its bounded original receipt")
    data = path.read_bytes()
    if len(data) != receipt["bytes"] or hashlib.sha256(data).hexdigest() != receipt["sha256"]:
        raise ContractError("morphology_checkpoint", "checkpoint bytes differ from the original immutable write receipt")
    if len(data) > MAX_CHECKPOINT_BYTES:
        raise ContractError("morphology_checkpoint", "checkpoint exceeds the fixed CPU bundle limit")
    return path


def write_checkpoint(root: Path, relative: str, model, metadata, *, optimizer=None, training_state=None):
    _metadata(metadata, model)
    state = {} if training_state is None else dict(training_state)
    state["torch_global_rng_state"] = torch.get_rng_state()
    payload = {"schema": CHECKPOINT_SCHEMA, "specification": policy_specification(model),
               "policy_sha256": policy_sha256(model), "state_dict": model.state_dict(),
               "metadata": dict(metadata), "optimizer": None if optimizer is None else optimizer.state_dict(),
               "training_state": state}
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()) or target.relative_to(root.resolve()).as_posix() != relative:
        raise ContractError("morphology_checkpoint", "canonical contained output checkpoint path required")
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    if len(data) > MAX_CHECKPOINT_BYTES:
        raise ContractError("morphology_checkpoint", "checkpoint exceeds the fixed CPU bundle limit")
    digest_value = immutable_bytes(target, data)
    return {"file": relative, "sha256": digest_value, "bytes": len(data),
            "policy_sha256": payload["policy_sha256"], "specification": payload["specification"],
            "metadata": dict(metadata), "training_state_keys": sorted(state)}


def load_checkpoint(root: Path, receipt, *, expected_metadata):
    path = checkpoint_file(root, receipt)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or set(payload) != {"schema", "specification", "policy_sha256", "state_dict",
                                                       "metadata", "optimizer", "training_state"}
            or payload["schema"] != CHECKPOINT_SCHEMA
            or sha256_json(payload["metadata"]) != sha256_json(expected_metadata)
            or sha256_json(payload["metadata"]) != sha256_json(receipt["metadata"])):
        raise ContractError("morphology_checkpoint", "checkpoint cannot be relabelled as another seed, phase, source or dataset")
    with torch.random.fork_rng(devices=[]):
        model = new_policy(payload["metadata"]["family"], payload["metadata"]["training_seed"])
    _metadata(payload["metadata"], model)
    if payload["specification"] != policy_specification(model) or payload["specification"] != receipt["specification"]:
        raise ContractError("morphology_checkpoint", "checkpoint architecture differs from its fixed candidate")
    expected_state = model.state_dict()
    if (not isinstance(payload["state_dict"], dict) or set(payload["state_dict"]) != set(expected_state)
            or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != "cpu"
                   or value.shape != expected_state[name].shape or not bool(torch.isfinite(value).all())
                   for name, value in payload["state_dict"].items())):
        raise ContractError("morphology_checkpoint", "exact finite CPU float32 tensors required without implicit conversion")
    model.load_state_dict(payload["state_dict"], strict=True)
    if policy_sha256(model) != payload["policy_sha256"] or payload["policy_sha256"] != receipt["policy_sha256"]:
        raise ContractError("morphology_checkpoint", "checkpoint finite numeric weights differ from their original identity")
    if sorted(payload["training_state"]) != receipt["training_state_keys"]:
        raise ContractError("morphology_checkpoint", "recorded optimizer/RNG boundary fields differ")
    # Recheck bytes after parsing as well as before; the original receipt never
    # adopts the hash of whichever valid checkpoint happens to occupy a path.
    checkpoint_file(root, receipt)
    model.eval()
    return model, payload
