"""Hash-bound IK checkpoints at completed optimizer-step boundaries.

Importing this module needs no optional packages. A tensor file becomes a
published checkpoint only when its immutable JSON commit marker also exists.
The caller owns allocation accounting and must validate it before restoration.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import math
from pathlib import Path
import random
import re
from typing import Any

from nca.contracts import canonical_json, sha256_json, strict_json_loads
from nca.runs import immutable_bytes, immutable_json


FORMAT = "nca-ik-checkpoint/v1"
MAX_BYTES = 32 * 1024 * 1024
_IDENTITIES = ("config_sha256", "source_sha256", "profile_sha256", "dataset_sha256")


def _runtime():
    try:
        import numpy
        import torch
    except ImportError as exc:
        raise ValueError("IK checkpoint operations require the optional Torch/NumPy runtime") from exc
    return torch, numpy


def _json_copy(value: Any) -> Any:
    return strict_json_loads(canonical_json(value))


def _identity(value: Any, name: str) -> None:
    if type(value) is not str or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError(f"checkpoint {name} must be a SHA-256 identity")


def _metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint metadata must be a mapping")
    result = _json_copy(dict(value))
    for name in _IDENTITIES:
        _identity(result.get(name), name)
    return result


def _training_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint training state must be a mapping")
    result = _json_copy(dict(value))
    if result.get("boundary") != "optimizer_step":
        raise ValueError("IK checkpoints require a completed optimizer-step boundary")
    return result


def _name(value: Any) -> str | None:
    return None if value is None else f"{type(value).__module__}.{type(value).__qualname__}"


def _cpu_copy(value: Any) -> Any:
    torch, _ = _runtime()
    if isinstance(value, torch.Tensor):
        copied = value.detach().to("cpu").clone()
        if copied.is_floating_point() and not bool(torch.isfinite(copied).all()):
            raise ValueError("checkpoint contains a nonfinite tensor")
        return copied
    if isinstance(value, Mapping):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        raise ValueError("checkpoint contains a nonfinite number")
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("checkpoint state contains an unsupported object")


def capture_rng(generators: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Capture global RNGs and every explicitly used Torch sampler generator.

    CUDA is captured only after the selected runtime initialized it. CPU-only
    checkpointing does not initialize a GPU simply to gather unused RNGs.
    """
    torch, numpy = _runtime()
    np_state = numpy.random.get_state()
    named = {}
    for name, generator in (generators or {}).items():
        if type(name) is not str or not name or not isinstance(generator, torch.Generator):
            raise ValueError("named RNGs must be nonempty names of Torch generators")
        named[name] = {"device": str(generator.device), "state": generator.get_state().cpu().clone()}
    return {
        "schema": "nca-ik-rng/v1",
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])],
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized() else [],
        "generators": named,
    }


def _validate_rng(state: Any, generators: Mapping[str, Any] | None = None) -> None:
    torch, numpy = _runtime()
    if not isinstance(state, dict) or set(state) != {
            "schema", "python", "numpy", "torch_cpu", "torch_cuda", "generators"}:
        raise ValueError("checkpoint RNG fields differ")
    if state["schema"] != "nca-ik-rng/v1":
        raise ValueError("unknown checkpoint RNG schema")
    try:
        random.Random().setstate(state["python"])
        name, words, position, has_gauss, cached_gaussian = state["numpy"]
        if name != "MT19937" or len(words) != 624 or any(
                type(word) is not int or not 0 <= word < 2**32 for word in words):
            raise ValueError("invalid NumPy RNG words")
        if (type(position) is not int or not 0 <= position <= 624
                or type(has_gauss) is not int or has_gauss not in (0, 1)
                or type(cached_gaussian) is not float or not math.isfinite(cached_gaussian)):
            raise ValueError("invalid NumPy RNG cursor/cache")
        numpy.random.RandomState().set_state(
            (name, numpy.asarray(words, dtype=numpy.uint32), position, has_gauss, cached_gaussian))
        cpu = state["torch_cpu"]
        if not isinstance(cpu, torch.Tensor) or cpu.dtype != torch.uint8 or cpu.device.type != "cpu":
            raise ValueError("invalid Torch CPU RNG tensor")
        torch.Generator(device="cpu").set_state(cpu)
        if not isinstance(state["torch_cuda"], list):
            raise ValueError("invalid Torch CUDA RNG list")
        for item in state["torch_cuda"]:
            if not isinstance(item, torch.Tensor) or item.dtype != torch.uint8 or item.device.type != "cpu" or item.ndim != 1:
                raise ValueError("invalid Torch CUDA RNG tensor")
        if not isinstance(state["generators"], dict):
            raise ValueError("invalid named RNGs")
        for name, item in state["generators"].items():
            if type(name) is not str or not name or not isinstance(item, dict) or set(item) != {"device", "state"}:
                raise ValueError("invalid named RNG entry")
            if not isinstance(item["state"], torch.Tensor) or item["state"].dtype != torch.uint8 or item["state"].device.type != "cpu":
                raise ValueError("invalid named RNG tensor")
            if item["device"] == "cpu":
                torch.Generator(device="cpu").set_state(item["state"])
            elif type(item["device"]) is not str or not re.fullmatch(r"cuda:\d+", item["device"]):
                raise ValueError("unsupported named RNG device")
        if generators is not None and set(generators) != set(state["generators"]):
            raise ValueError("named RNG set differs from checkpoint")
    except (TypeError, RuntimeError, KeyError, IndexError, OverflowError) as exc:
        raise ValueError("invalid checkpoint RNG state") from exc


def restore_rng(state: Mapping[str, Any], generators: Mapping[str, Any] | None = None, *,
                allow_device_change: bool = False) -> list[str]:
    """Restore saved streams; explicit device continuations disclose omissions."""
    torch, numpy = _runtime()
    generators = {} if generators is None else dict(generators)
    _validate_rng(state, generators)
    cuda_count = torch.cuda.device_count() if torch.cuda.is_initialized() else 0
    saved_count = len(state["torch_cuda"])
    limits = []
    if cuda_count != saved_count:
        if not allow_device_change:
            raise ValueError("CUDA RNG device count differs; explicit continuation required")
        limits.append("CUDA RNG device count changed; unmatched global CUDA streams were not restored")
    for name, generator in generators.items():
        if str(generator.device) != state["generators"][name]["device"]:
            # Torch's CPU and CUDA generators use different algorithms/state
            # formats. A flag cannot make them resumable at the same cursor.
            raise ValueError("sampler generator device changed; exact stream restoration is unsupported")
    random.setstate(state["python"])
    name, words, position, has_gauss, cached_gaussian = state["numpy"]
    numpy.random.set_state((name, numpy.asarray(words, dtype=numpy.uint32), position, has_gauss, cached_gaussian))
    torch.set_rng_state(state["torch_cpu"])
    if cuda_count == saved_count and cuda_count:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for name, generator in generators.items():
        generator.set_state(state["generators"][name]["state"])
    return limits


def write_checkpoint(path: str | Path, model: Any, optimizer: Any, *, metadata: Mapping[str, Any],
                     training_state: Mapping[str, Any], generators: Mapping[str, Any] | None = None,
                     scheduler: Any = None, scaler: Any = None, max_bytes: int = MAX_BYTES) -> dict[str, Any]:
    """Publish a complete tensor file followed by its immutable commit marker.

    A failed marker publication leaves an uncommitted artifact; it never becomes
    a resumable checkpoint. Neither file is overwritten on retry.
    """
    torch, _ = _runtime()
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES:
        raise ValueError("checkpoint byte cap must be positive and at most 32 MiB")
    target = Path(path)
    metadata_value = _metadata(metadata)
    state = _training_state(training_state)
    payload = {
        "format": FORMAT, "metadata": metadata_value, "training_state": state,
        "components": {"model": _name(model), "optimizer": _name(optimizer),
                       "scheduler": _name(scheduler), "scaler": _name(scaler)},
        "model": _cpu_copy(model.state_dict()), "optimizer": _cpu_copy(optimizer.state_dict()),
        "scheduler": None if scheduler is None else _cpu_copy(scheduler.state_dict()),
        "scaler": None if scaler is None else _cpu_copy(scaler.state_dict()),
        "rng": capture_rng(generators),
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    data = stream.getvalue()
    if len(data) > max_bytes:
        raise ValueError("checkpoint exceeds its declared byte cap")
    digest = immutable_bytes(target, data)
    reference = {"file": target.name, "sha256": digest, "bytes": len(data), "format": FORMAT,
                 "metadata": metadata_value, "training_state_sha256": sha256_json(state)}
    marker = {"checkpoint": reference, "record_sha256": sha256_json(reference)}
    immutable_json(target.with_name(target.name + ".json"), marker)
    return reference


def load_checkpoint(path: str | Path, *, expected_sha256: str, expected_metadata: Mapping[str, Any],
                    max_bytes: int = MAX_BYTES) -> dict[str, Any]:
    """Verify bytes and commit marker before safe tensor-only deserialization."""
    _identity(expected_sha256, "expected_sha256")
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES:
        raise ValueError("checkpoint byte cap must be positive and at most 32 MiB")
    expected = _metadata(expected_metadata)
    target = Path(path)
    if target.stat().st_size > max_bytes:
        raise ValueError("checkpoint exceeds its declared byte cap")
    marker_path = target.with_name(target.name + ".json")
    if marker_path.stat().st_size > 1_048_576:
        raise ValueError("checkpoint marker exceeds 1 MiB")
    with marker_path.open("rb") as stream:
        marker = strict_json_loads(stream.read(1_048_577))
    if not isinstance(marker, dict) or set(marker) != {"checkpoint", "record_sha256"}:
        raise ValueError("checkpoint marker fields differ")
    reference = marker["checkpoint"]
    if not isinstance(reference, dict) or set(reference) != {
            "file", "sha256", "bytes", "format", "metadata", "training_state_sha256"}:
        raise ValueError("checkpoint reference fields differ")
    if marker["record_sha256"] != sha256_json(reference):
        raise ValueError("checkpoint marker hash mismatch")
    if reference["file"] != target.name or reference["format"] != FORMAT or reference["sha256"] != expected_sha256:
        raise ValueError("checkpoint marker identity mismatch")
    marker_metadata = _metadata(reference["metadata"])
    if any(marker_metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint configuration/source/profile/dataset identity differs")
    with target.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes or reference["bytes"] != len(data) or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("checkpoint bytes or SHA-256 mismatch")
    torch, _ = _runtime()
    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("checkpoint tensor payload could not be decoded") from exc
    if not isinstance(payload, dict) or set(payload) != {
            "format", "metadata", "training_state", "components", "model", "optimizer", "scheduler", "scaler", "rng"}:
        raise ValueError("checkpoint payload fields differ")
    if payload["format"] != FORMAT:
        raise ValueError("unknown checkpoint format")
    metadata_value = _metadata(payload["metadata"])
    if metadata_value != reference["metadata"] or any(metadata_value.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint configuration/source/profile/dataset identity differs")
    state = _training_state(payload["training_state"])
    if reference["training_state_sha256"] != sha256_json(state):
        raise ValueError("checkpoint training state hash differs")
    _validate_rng(payload["rng"])
    _cpu_copy(payload)  # Reject nonfinite tensors or objects outside the schema.
    return payload


def restore_checkpoint(payload: Mapping[str, Any], model: Any, optimizer: Any, *,
                       generators: Mapping[str, Any] | None = None, scheduler: Any = None,
                       scaler: Any = None, allow_device_change: bool = False) -> tuple[dict[str, Any], list[str]]:
    """Restore into fresh components after the caller validates allocation state.

    Call only with a payload returned by ``load_checkpoint``. The final RNG
    restoration occurs after model/optimizer construction and state loading.
    """
    torch, _ = _runtime()
    expected_components = {"model": _name(model), "optimizer": _name(optimizer),
                           "scheduler": _name(scheduler), "scaler": _name(scaler)}
    if payload.get("format") != FORMAT or payload.get("components") != expected_components:
        raise ValueError("checkpoint model/optimizer/scheduler/scaler types differ")
    current = model.state_dict()
    saved = payload["model"]
    if set(current) != set(saved) or any(not isinstance(saved[key], torch.Tensor)
            or current[key].shape != saved[key].shape or current[key].dtype != saved[key].dtype for key in current):
        raise ValueError("checkpoint model shape or dtype differs")
    parameters = list(model.parameters())
    if not parameters or len({parameter.device for parameter in parameters}) != 1:
        raise ValueError("IK restoration requires a single-device model")
    current_optimizer = optimizer.state_dict()
    old_groups, new_groups = payload["optimizer"]["param_groups"], current_optimizer["param_groups"]
    if len(old_groups) != len(new_groups) or any(
            len(old["params"]) != len(new["params"]) for old, new in zip(old_groups, new_groups)):
        raise ValueError("checkpoint optimizer parameter groups differ")
    _validate_rng(payload["rng"], {} if generators is None else generators)
    saved_states = payload["optimizer"]["state"]
    for group, live_group in zip(old_groups, optimizer.param_groups):
        for saved_id, parameter in zip(group["params"], live_group["params"]):
            for name, value in saved_states.get(saved_id, {}).items():
                if isinstance(value, torch.Tensor) and name != "step" and value.shape != parameter.shape:
                    raise ValueError("checkpoint optimizer tensor shape differs")
    for old, new in zip(old_groups, new_groups):
        if old.get("capturable", False) != new.get("capturable", False) and not allow_device_change:
            raise ValueError("optimizer capturable mode changed; explicit device continuation required")
    model.load_state_dict(saved, strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    device = parameters[0].device
    limits = []
    for old, new, restored in zip(old_groups, new_groups, optimizer.param_groups):
        if old.get("capturable", False) != new.get("capturable", False):
            if not allow_device_change:
                raise ValueError("optimizer capturable mode changed; explicit device continuation required")
            restored["capturable"] = new.get("capturable", False)
            limits.append("optimizer capturable mode changed for the explicitly selected continuation device")
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value
    for parameter in optimizer.state:
        optimizer.state[parameter] = move(optimizer.state[parameter])
    for component, name in ((scheduler, "scheduler"), (scaler, "scaler")):
        if component is not None:
            component.load_state_dict(payload[name])
    limits.extend(restore_rng(payload["rng"], generators, allow_device_change=allow_device_change))
    return _training_state(payload["training_state"]), limits
