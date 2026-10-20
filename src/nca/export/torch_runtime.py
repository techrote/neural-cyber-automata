"""Explicit CPU Torch operator backend for an unchanged evaluated numeric policy.

The NumPy bundle remains a separate, unqualified lane. This wrapper binds the
additional runtime dependency and operator path; it does not relabel that bundle
or import a learner/checkpoint implementation. All owner and recurrence fences
remain in ExportController. No benchmark or host starts on import.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from nca.contracts import ContractError, canonical_json, digest, sha256_json, strict_json_loads
from .runtime import ExportController, NumpyGRU, RuntimeBundle
from .spec import RuntimeContext, source_files


BACKEND = {
    "id": "torch-operators-cpu/v1", "torch": "2.8.0+cpu", "threads": 1,
    "device": "cpu", "dtype": "float32", "batch": 1, "cpu_autocast": False,
    "float32_matmul_precision": "highest", "mkldnn_enabled": True,
    "operators": "Linear26x32,tanh,GRUCell32,Linear32x6,Linear32x1,tanh/sigmoid",
    "output_partition": [2, 1, 3], "state": "explicit feature26 and hidden32",
    "learner_import": False, "optimizer": False, "checkpoint_unpickling": False,
}


def backend_sources() -> dict[str, str]:
    result = source_files()
    result["src/nca/export/torch_runtime.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def _require_runtime():
    if (str(torch.__version__) != BACKEND["torch"] or torch.get_num_threads() != BACKEND["threads"]
            or torch.is_autocast_enabled("cpu")
            or torch.get_float32_matmul_precision() != BACKEND["float32_matmul_precision"]
            or torch.backends.mkldnn.enabled != BACKEND["mkldnn_enabled"]):
        raise ContractError("torch_runtime_context", "declared one-thread CPU FP32 Torch context required")


class TorchGRU:
    """Same batch-one operators as the reference, with owned frozen numeric data."""

    def __init__(self, weights: Mapping[str, Any]):
        _require_runtime()
        admitted = NumpyGRU(weights)
        self.policy_sha256 = admitted.policy_sha256
        self.nbytes = admitted.nbytes
        # Construction must not consume the caller's action or training RNG.
        with torch.random.fork_rng(devices=[]):
            self._encoder = torch.nn.Linear(26, 32, device="cpu", dtype=torch.float32)
            self._cell = torch.nn.GRUCell(32, 32, device="cpu", dtype=torch.float32)
            self._actor = torch.nn.Linear(32, 6, device="cpu", dtype=torch.float32)
            self._critic = torch.nn.Linear(32, 1, device="cpu", dtype=torch.float32)
        for prefix, module in (("encoder", self._encoder), ("recurrent", self._cell),
                               ("actor", self._actor), ("critic", self._critic)):
            module.load_state_dict({key.removeprefix(prefix + "."): torch.tensor(value.copy(), device="cpu")
                                    for key, value in admitted.weights.items() if key.startswith(prefix + ".")})
            module.eval().requires_grad_(False)

    def forward(self, inputs, hidden, *, reset=False):
        _require_runtime()
        x, h = np.asarray(inputs, dtype=np.float32), np.asarray(hidden, dtype=np.float32)
        if x.shape != (26,) or h.shape != (32,) or not np.isfinite(x).all() or not np.isfinite(h).all():
            raise ContractError("network_input", "finite feature26 and hidden32 vectors required")
        if type(reset) is not bool:
            raise ContractError("network_reset", "explicit boolean reset required")
        if reset:
            h = np.zeros(32, dtype=np.float32)
        with torch.no_grad():
            # Keep the original batch shape and split sigmoid calls: operator
            # equivalence is deliberate, not an assertion about all FP32 math.
            tx = torch.tensor(x[None, :], device="cpu", dtype=torch.float32)
            th = torch.tensor(h[None, :], device="cpu", dtype=torch.float32)
            following = self._cell(torch.tanh(self._encoder(tx)), th)
            logits = self._actor(following)
            value = self._critic(following).squeeze(-1)
            outputs = torch.cat((logits[..., :2].tanh(), logits[..., 2:3].sigmoid(),
                                 logits[..., 3:].sigmoid()), dim=-1)
        if any(not torch.isfinite(item).all().item() for item in (following, logits, value, outputs)):
            raise ContractError("network_nonfinite", "nonfinite inference result")
        return outputs[0].numpy().copy(), float(value[0]), following[0].numpy().copy(), logits[0].numpy().copy()


def make_torch_bundle(numeric_payload: Mapping[str, Any], context: RuntimeContext, *,
                      numeric_file_sha256: str) -> dict[str, Any]:
    """Wrap an already admitted evaluated artifact; conversion never qualifies it."""
    _require_runtime()
    digest(numeric_file_sha256, "numeric_file_sha256")
    numeric = RuntimeBundle(numeric_payload, context)
    result = {"format": "nca-torch-cpu-policy/v1", "numeric_bundle": numeric.value,
              "source_numeric_file_sha256": numeric_file_sha256,
              "source_numeric_content_sha256": numeric.sha256,
              "backend": strict_json_loads(canonical_json(BACKEND)), "backend_sources": backend_sources(),
              "qualification": {"runtime_qualified": False, "receipt_required": True,
                                "scope": "Linux external Python Torch operators FP32 CPU loop"}}
    result["sha256"] = sha256_json(result)
    return result


class TorchRuntimeBundle:
    """Independent backend identity around the unchanged numeric policy identity."""

    def __init__(self, value: Mapping[str, Any], context: RuntimeContext):
        if not isinstance(value, Mapping):
            raise ContractError("torch_bundle_schema", "structured Torch CPU bundle required")
        payload = strict_json_loads(canonical_json(value))
        fields = {"format", "numeric_bundle", "source_numeric_file_sha256", "source_numeric_content_sha256",
                  "backend", "backend_sources", "qualification", "sha256"}
        if set(payload) != fields or payload["format"] != "nca-torch-cpu-policy/v1":
            raise ContractError("torch_bundle_version", "exact explicit Torch CPU bundle v1 required")
        declared = payload.pop("sha256")
        if sha256_json(payload) != declared:
            raise ContractError("torch_bundle_hash", "Torch CPU bundle content changed")
        payload["sha256"] = declared
        if payload["backend"] != BACKEND:
            raise ContractError("torch_backend", "declared original operator graph and CPU runtime required")
        _require_runtime()
        if payload["backend_sources"] != backend_sources():
            raise ContractError("torch_backend_sources", "backend or owner boundary source changed")
        if payload["qualification"] != {"runtime_qualified": False, "receipt_required": True,
                "scope": "Linux external Python Torch operators FP32 CPU loop"}:
            raise ContractError("torch_qualification", "conversion cannot manufacture a qualification receipt")
        digest(payload["source_numeric_file_sha256"], "source_numeric_file_sha256")
        self.numeric = RuntimeBundle(payload["numeric_bundle"], context)
        if self.numeric.sha256 != payload["source_numeric_content_sha256"]:
            raise ContractError("torch_numeric_identity", "original evaluated numeric content differs")
        self.network = TorchGRU(self.numeric.value["weights"])
        self._value, self.sha256, self.context = payload, declared, context
        self.part_ids = self.numeric.part_ids

    @property
    def value(self):
        return strict_json_loads(canonical_json(self._value))

    @classmethod
    def load(cls, path: str | Path, context: RuntimeContext, *, expected_sha256: str):
        location = Path(path)
        if location.stat().st_size > 2 * 1024 * 1024:
            raise ContractError("torch_bundle_size", "numeric Torch CPU bundle limit is 2 MiB")
        data = location.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ContractError("torch_artifact_hash", "external Torch bundle file receipt differs")
        return cls(strict_json_loads(data), context)


def torch_export_controller(bundle: TorchRuntimeBundle) -> ExportController:
    if not isinstance(bundle, TorchRuntimeBundle):
        raise ContractError("torch_bundle_required", "explicit Torch backend artifact required")
    return ExportController(bundle)
