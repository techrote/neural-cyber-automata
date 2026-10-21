"""Optional FP32 device support for the bounded IK qualification workload.

Importing this module does not import NumPy or Torch.  A successful preflight is
only a report of capabilities; only the separately recorded workload can qualify
forward/backward/optimizer execution.  No function silently changes devices.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import importlib
from importlib import metadata
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from typing import Any

from nca.contracts import sha256_json


TORCH_VERSION = "2.8.0"
NUMPY_VERSION = "2.3.5"
CUDA_VERSION = "12.6"


class DeviceError(ValueError):
    """A public-safe failure with retained preflight facts, never raw paths."""

    def __init__(self, code: str, message: str, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.report = dict(report or {})


def parse_device(spec: str) -> tuple[str, int | None]:
    """Require an explicit, canonical CPU or CUDA ordinal."""
    if spec == "cpu":
        return "cpu", None
    if isinstance(spec, str) and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", spec):
        return "cuda", int(spec.split(":", 1)[1])
    raise DeviceError("invalid_device", "device must be cpu or an explicit cuda:<index>")


def _digest_argument(value: str | None, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value)):
        raise DeviceError("invalid_identity", f"{name} must be a lowercase SHA-256 digest")


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _prepare_cublas() -> None:
    value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if value is None:
        existing = sys.modules.get("torch")
        if existing is not None and existing.cuda.is_initialized():
            raise DeviceError("late_determinism_configuration",
                              "set CUBLAS_WORKSPACE_CONFIG before initializing CUDA")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif value not in (":4096:8", ":16:8"):
        raise DeviceError("invalid_determinism_configuration",
                          "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8")


def _driver_version() -> str | None:
    # Do not emit the command, stderr, executable path, device UUID or hostname.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=2,
        )
        values = set(result.stdout.strip().splitlines())
        if len(values) == 1:
            value = values.pop().strip()
            if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", value):
                return value
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _architecture_route(capability: tuple[int, int], architectures: list[str]) -> str | None:
    """Identify a compiled binary/PTX route; this still is not operation proof."""
    major, minor = capability
    exact = f"sm_{major}{minor}"
    if exact in architectures:
        return exact
    routes = []
    for arch in architectures:
        match = re.fullmatch(r"(sm|compute)_([0-9]+)", arch)
        if match is None:
            continue
        number = int(match.group(2))
        arch_major, arch_minor = divmod(number, 10)
        if ((match.group(1) == "sm" and arch_major == major and arch_minor <= minor)
                or (match.group(1) == "compute" and (arch_major, arch_minor) <= capability)):
            routes.append(arch)
    return sorted(routes)[-1] if routes else None


def _precision_report(torch: Any) -> dict[str, Any]:
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8", ":16:8"):
        workspace = "unrecognized"
    return {
        "dtype": "float32",
        "autocast": False,
        "compilation": False,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cublas_workspace_config": workspace,
        "torch_threads": torch.get_num_threads(),
    }


def environment_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable preflight/configuration facts for continuation comparison."""
    # Free/allocated memory is transient; a changed free-memory reading is not a
    # changed software/hardware environment.  Configuration/source remain their
    # separate identities and are not folded into this environment identity.
    return {name: value for name, value in report.items() if name not in {
        "source_sha256", "config_sha256", "memory", "status", "reason",
        "environment_sha256", "qualification",
    }}


def _finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    report["environment_sha256"] = sha256_json(environment_identity(report))
    return report


def device_report(spec: str, *, source_sha256: str | None = None,
                  config_sha256: str | None = None) -> dict[str, Any]:
    """Inspect the exact optional runtime without performing a training step.

    CUDA context creation and memory inspection may occur.  The caller must
    charge this startup/preflight time to its declared allocation.  The returned
    ``ready`` state does not claim that any kernel, backward or optimizer works.
    """
    kind, index = parse_device(spec)
    _digest_argument(source_sha256, "source_sha256")
    _digest_argument(config_sha256, "config_sha256")
    report: dict[str, Any] = {
        "schema": "nca-ik-device/v1", "requested_device": spec,
        "status": "unavailable", "reason": None, "qualification": "unmeasured",
        "source_sha256": source_sha256, "config_sha256": config_sha256,
        "python": platform.python_version(), "python_implementation": platform.python_implementation(),
        "os": platform.system(), "os_release": platform.release(),
        "architecture": platform.machine(), "cpu_processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": _package_version("numpy"), "torch": _package_version("torch"),
        "torch_git_version": None, "torch_build_config_sha256": None,
        "cuda_build": None, "cudnn_version": None,
        "gpu": None, "compiled_architectures": [], "architecture_route": None,
        "driver": None, "precision": None,
        "memory": {"free_bytes": None, "total_bytes": None},
    }
    if kind == "cuda":
        try:
            _prepare_cublas()
        except DeviceError as exc:
            report.update(status="unsupported", reason=exc.code)
            return _finalize_report(report)
    try:
        torch = importlib.import_module("torch")
        numpy = importlib.import_module("numpy")
    except ImportError:
        report["reason"] = "optional_runtime_missing"
        return _finalize_report(report)
    except (OSError, RuntimeError):
        report["reason"] = "optional_runtime_import_failed"
        return _finalize_report(report)
    report.update(
        torch=str(torch.__version__), numpy=str(numpy.__version__),
        torch_git_version=torch.version.git_version,
        torch_build_config_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        cuda_build=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        precision=_precision_report(torch),
    )
    # The named CPU and CUDA 12.6 wheel routes are separate identified lanes.
    if (report["torch"].split("+", 1)[0] != TORCH_VERSION
            or report["numpy"] != NUMPY_VERSION
            or report["cuda_build"] not in (None, CUDA_VERSION)):
        report.update(status="unsupported", reason="runtime_version_mismatch")
        return _finalize_report(report)
    if kind == "cpu":
        report.update(status="ready", reason=None)
        return _finalize_report(report)
    report["driver"] = _driver_version()
    try:
        report["compiled_architectures"] = list(torch.cuda.get_arch_list())
        if not torch.cuda.is_available():
            report["reason"] = "cuda_unavailable"
            return _finalize_report(report)
        count = torch.cuda.device_count()
        if index is None or index >= count:
            report.update(status="unsupported", reason="cuda_index_unavailable")
            return _finalize_report(report)
        props = torch.cuda.get_device_properties(index)
        capability = (props.major, props.minor)
        report["gpu"] = {"name": props.name, "capability": list(capability),
                         "multiprocessors": props.multi_processor_count,
                         "total_memory_bytes": int(props.total_memory)}
        report["architecture_route"] = _architecture_route(capability, report["compiled_architectures"])
        free, total = torch.cuda.mem_get_info(index)
        report["memory"] = {"free_bytes": int(free), "total_bytes": int(total)}
        if report["architecture_route"] is None:
            report.update(status="unsupported", reason="cuda_architecture_not_compiled")
        else:
            report.update(status="ready", reason=None)
    except (RuntimeError, OSError, AssertionError):
        report.update(status="unavailable", reason="cuda_preflight_failed")
    return _finalize_report(report)


@dataclass
class DeviceContext:
    torch: Any
    numpy: Any
    device: Any
    report: dict[str, Any]

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def reset_peak_memory(self) -> None:
        if self.device.type == "cuda":
            self.synchronize()
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def memory_stats(self) -> dict[str, Any]:
        if self.device.type != "cuda":
            return {"device": "cpu", "allocated_bytes": None, "reserved_bytes": None,
                    "peak_allocated_bytes": None, "peak_reserved_bytes": None,
                    "free_bytes": None, "total_bytes": None,
                    "measurement": "CUDA allocator counters do not measure CPU memory"}
        self.synchronize()
        free, total = self.torch.cuda.mem_get_info(self.device)
        return {"device": str(self.device),
                "allocated_bytes": int(self.torch.cuda.memory_allocated(self.device)),
                "reserved_bytes": int(self.torch.cuda.memory_reserved(self.device)),
                "peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
                "peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved(self.device)),
                "free_bytes": int(free), "total_bytes": int(total),
                "measurement": "PyTorch allocator; external CUDA allocations are excluded"}

    def timed(self, operation: Callable[[], Any]) -> tuple[Any, dict[str, float | None]]:
        """Report synchronized host duration and CUDA event duration separately."""
        self.synchronize()
        start_event = end_event = None
        if self.device.type == "cuda":
            with self.torch.cuda.device(self.device):
                start_event = self.torch.cuda.Event(enable_timing=True)
                end_event = self.torch.cuda.Event(enable_timing=True)
                start_event.record(self.torch.cuda.current_stream(self.device))
        start = time.perf_counter()
        value = operation()
        if end_event is not None:
            end_event.record(self.torch.cuda.current_stream(self.device))
        self.synchronize()
        host_seconds = time.perf_counter() - start
        device_seconds = None if start_event is None else start_event.elapsed_time(end_event) / 1000
        return value, {"synchronized_host_seconds": host_seconds, "cuda_event_seconds": device_seconds}

    def verify_tensors(self, named: Mapping[str, Any], *, require_finite: bool = True) -> dict[str, Any]:
        """Inspect tensors recursively; do not infer placement from the model.

        Each named category must contain at least one tensor.  Floating tensors
        must be FP32 and finite, integer/bool tensors retain their own dtype.
        Passing RNG checkpoint tensors here is inappropriate: host RNG state is
        intentionally CPU metadata and is verified by the checkpoint layer.
        """
        if not isinstance(named, Mapping) or not named:
            raise DeviceError("missing_tensor_evidence", "tensor categories must be nonempty")
        counts: dict[str, int] = {}
        dtypes: dict[str, int] = {}

        def visit(value: Any, category: str) -> None:
            if isinstance(value, self.torch.Tensor):
                if value.device != self.device:
                    raise DeviceError("device_placement_mismatch", f"{category} contains a tensor on another device")
                if value.is_floating_point() and value.dtype != self.torch.float32:
                    raise DeviceError("dtype_mismatch", f"{category} contains a non-FP32 floating tensor")
                if value.is_complex():
                    raise DeviceError("dtype_mismatch", f"{category} contains an unsupported complex tensor")
                if require_finite and not bool(self.torch.isfinite(value).all()):
                    raise DeviceError("nonfinite_tensor", f"{category} contains a nonfinite tensor")
                counts[category] += 1
                key = str(value.dtype)
                dtypes[key] = dtypes.get(key, 0) + 1
            elif isinstance(value, Mapping):
                for child in value.values():
                    visit(child, category)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child, category)
            elif value is not None and not isinstance(value, (str, int, float, bool)):
                raise DeviceError("unsupported_tensor_container", "tensor evidence must use mappings or sequences")

        for category, value in named.items():
            if not isinstance(category, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", category):
                raise DeviceError("invalid_tensor_category", "tensor categories require public identifiers")
            counts[category] = 0
            visit(value, category)
            if counts[category] == 0:
                raise DeviceError("missing_tensor_evidence", f"{category} contains no tensors")
        return {"schema": "nca-ik-placement/v1", "device": str(self.device),
                "tensor_counts": counts, "dtypes": dtypes,
                "finite_checked": require_finite, "all_on_selected_device": True}


def configure_device(spec: str, seed: int, *, memory_fraction: float = 0.5,
                     source_sha256: str | None = None,
                     config_sha256: str | None = None) -> DeviceContext:
    """Configure the isolated deterministic FP32 lane or fail explicitly.

    The caller owns the process and its budget.  CUDA allocator fraction limits
    allocator-managed memory, not driver/context or arbitrary external memory.
    """
    kind, _ = parse_device(spec)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise DeviceError("invalid_seed", "seed must be an unsigned 32-bit integer")
    if (type(memory_fraction) not in (int, float) or not math.isfinite(memory_fraction)
            or not 0 < memory_fraction <= 1):
        raise DeviceError("invalid_memory_limit", "memory fraction must be finite and in (0, 1]")
    report = device_report(spec, source_sha256=source_sha256, config_sha256=config_sha256)
    if report["status"] != "ready":
        raise DeviceError(report["reason"], "requested IK device runtime is unavailable or unsupported", report)
    torch = importlib.import_module("torch")
    numpy = importlib.import_module("numpy")
    device = torch.device(spec)
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_num_threads(1)
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if kind == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.set_per_process_memory_fraction(float(memory_fraction), device)
            torch.cuda.manual_seed(seed)
    except (RuntimeError, OSError, AssertionError) as exc:
        report.update(status="unsupported", reason="device_configuration_failed")
        raise DeviceError("device_configuration_failed", "requested device settings could not be applied",
                          _finalize_report(report)) from exc
    report["precision"] = _precision_report(torch)
    report["allocator_memory_fraction"] = float(memory_fraction) if kind == "cuda" else None
    return DeviceContext(torch, numpy, device, _finalize_report(report))
