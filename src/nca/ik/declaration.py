"""Dependency-free prospective admission for one bounded IK allocation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import hashlib
from importlib import metadata
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import uuid
from typing import Any

import nca
from nca.contracts import sha256_json, strict_json_loads
from nca.data.capture import source_identity
from nca.ik.device import parse_device
from nca.ik.profiles import runner_right_sole_profile
from nca.ik.workload import WORKLOAD, dataset_identity
from nca.runs import immutable_json, utc_now


SCHEMA = "nca-ik-allocation/v1"
_CONFIG_FIELDS = {
    "schema", "id", "profile", "workload", "seed", "batch_size", "steps", "dataset_size", "hidden_width",
    "learning_rate", "memory_fraction", "max_optimizer_steps", "max_samples", "max_wall_seconds",
    "max_checkpoint_bytes",
}
_DECLARATION_FIELDS = {
    "schema", "run_id", "device", "config", "config_sha256", "source", "profile_sha256", "host_revision",
    "dataset", "runtime", "acceptance", "created_utc", "sha256",
}
_EXTRA_SOURCE_FILES = ("tools/setup_ik.sh", "tools/setup_ik.ps1", "configs/ik/runtime-smoke-v1.json")
_PACKAGES = (
    "torch", "numpy", "setuptools", "wheel", "pip", "filelock", "typing_extensions", "sympy", "mpmath",
    "networkx", "jinja2", "MarkupSafe", "fsspec", "triton", "nvidia-cublas-cu12", "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12", "nvidia-cufft-cu12",
    "nvidia-cufile-cu12", "nvidia-curand-cu12", "nvidia-cusolver-cu12", "nvidia-cusparse-cu12",
    "nvidia-cusparselt-cu12", "nvidia-nccl-cu12", "nvidia-nvjitlink-cu12", "nvidia-nvtx-cu12",
)


def _digest(value: Any, name: str) -> None:
    if type(value) is not str or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA-256 identity")


def _revision(value: Any, name: str) -> None:
    if type(value) is not str or re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", value) is None:
        raise ValueError(f"{name} must be a full Git revision")


def _integer(value: Any, name: str, lower: int, upper: int) -> None:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer within its declared limit")


def _number(value: Any, name: str, lower: float, upper: float) -> None:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or not lower < value <= upper:
        raise ValueError(f"{name} is outside its finite limit")


def validate_config(value: Any) -> dict[str, Any]:
    """Validate the fixed smoke workload; all configurable costs remain finite."""
    if not isinstance(value, dict) or set(value) != _CONFIG_FIELDS:
        raise ValueError("IK smoke configuration fields differ")
    constants = {"schema": "nca-ik-runtime-smoke/v1", "id": "runner-ik-device-smoke-v1",
                 "profile": "runner-right-sole@1", "workload": WORKLOAD}
    if any(value[key] != expected for key, expected in constants.items()):
        raise ValueError("unknown IK smoke configuration/profile/workload")
    _integer(value["seed"], "seed", 0, 2**32 - 1)
    _integer(value["batch_size"], "batch_size", 1, 256)
    _integer(value["max_optimizer_steps"], "max_optimizer_steps", 1, 256)
    _integer(value["steps"], "steps", 1, value["max_optimizer_steps"])
    _integer(value["dataset_size"], "dataset_size", 256, 256)
    _integer(value["hidden_width"], "hidden_width", 32, 32)
    _integer(value["max_samples"], "max_samples", value["steps"] * value["batch_size"],
             value["max_optimizer_steps"] * value["batch_size"])
    _integer(value["max_checkpoint_bytes"], "max_checkpoint_bytes", 1, 32 * 1024 * 1024)
    _number(value["learning_rate"], "learning_rate", 0., 1.)
    _number(value["memory_fraction"], "memory_fraction", 0., 1.)
    _number(value["max_wall_seconds"], "max_wall_seconds", 0., 120.)
    return dict(value)


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        return validate_config(strict_json_loads(stream.read(1_048_577)))


def _project(root: str | Path | None) -> Path:
    package = Path(nca.__file__).resolve().parent
    source_tree = package.parents[1]
    selected = root if root is not None else os.environ.get("NCA_SOURCE_ROOT")
    if selected is None:
        selected = source_tree if (source_tree / "src/nca").resolve() == package else Path.cwd()
    try:
        return Path(subprocess.check_output(["git", "-C", str(selected), "rev-parse", "--show-toplevel"],
                                            text=True, stderr=subprocess.DEVNULL).strip()).resolve()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("a matching NCA source checkout is required") from exc


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), *arguments],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("could not verify NCA source checkout") from exc


def _clean(root: Path) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"):
        raise ValueError("formal IK admission requires a clean source checkout including untracked files")


def _verify_committed_sources(root: Path, hashes: Mapping[str, str]) -> None:
    """Check HEAD blobs even when index flags hide worktree changes.

    Git's normal filters are deliberate: Windows CRLF checkouts must agree with
    their normalized committed blobs. The separate SHA-256 closure still binds
    the exact raw bytes that the interpreter imports. Batch hashing avoids a
    process launch for every source file during bounded startup.
    """
    entries = {}
    for entry in _git(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD").split("\0"):
        if not entry:
            continue
        information, relative = entry.split("\t", 1)
        mode, kind, digest = information.split(" ")
        entries[relative] = (mode, kind, digest)
    paths = sorted(hashes)
    for relative in paths:
        if "\n" in relative or "\r" in relative or "\0" in relative:
            raise ValueError("source closure contains an unsupported path")
        if relative not in entries or entries[relative][1] != "blob":
            raise ValueError("source closure contains a missing or untracked committed file")
        if entries[relative][0] not in ("100644", "100755"):
            raise ValueError("source closure must contain ordinary committed files")
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[relative]:
            raise ValueError("raw source bytes changed during snapshot")
    try:
        result = subprocess.run(["git", "-C", str(root), "hash-object", "--stdin-paths"],
                                input="".join(relative + "\n" for relative in paths),
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("could not verify committed source blobs") from exc
    normalized = result.stdout.splitlines()
    if len(normalized) != len(paths) or any(
            actual != entries[relative][2] for relative, actual in zip(paths, normalized)):
        raise ValueError("committed source bytes differ from HEAD despite Git status")


def source_snapshot(root: str | Path | None = None, require_clean: bool = True) -> dict[str, Any]:
    project = _project(root)
    if require_clean:
        _clean(project)
    revision, hashes, _ = source_identity(project)
    _revision(revision, "source revision")
    package = Path(nca.__file__).resolve().parent
    def json_hashes(folder):
        return {"src/nca/" + path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(folder.rglob("*.json")) if path.is_file()}
    installed_json = json_hashes(package)
    if installed_json != json_hashes(project / "src/nca"):
        raise ValueError("installed NCA JSON profile/contract sources differ from the selected checkout")
    hashes.update(installed_json)
    for relative in _EXTRA_SOURCE_FILES:
        path = project / relative
        if not path.is_file():
            raise ValueError("required IK setup/configuration source file is missing")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if require_clean:
        _verify_committed_sources(project, hashes)
    if _git(project, "rev-parse", "HEAD") != revision:
        raise ValueError("source revision changed during snapshot")
    if require_clean:
        _clean(project)
    return {"revision": revision, "source_sha256": sha256_json(hashes), "files": hashes}


def runtime_expectation() -> dict[str, Any]:
    """Read public build identities without importing Torch or initializing CUDA."""
    packages = {}
    for name in _PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "python_implementation": platform.python_implementation(),
            "os": platform.system(), "os_release": platform.release(), "architecture": platform.machine(),
            "byte_order": sys.byteorder, "packages": packages}


def _acceptance(config: Mapping[str, Any]) -> dict[str, Any]:
    return {"finite_loss": True, "finite_gradients": True, "weights_changed": True,
            "selected_device_placement": True, "completed_steps": config["steps"], "quality_qualification": False}


def prepare(config_path: str | Path, device: str, output: str | Path, *,
            source_root: str | Path | None = None) -> dict[str, Any]:
    """Freeze one allocation before optional imports or measured execution."""
    parse_device(device)
    config = read_config(config_path)
    source = source_snapshot(source_root, require_clean=True)
    profile = runner_right_sole_profile()
    declaration = {
        "schema": SCHEMA, "run_id": "ik-" + uuid.uuid4().hex, "device": device,
        "config": config, "config_sha256": sha256_json(config), "source": source,
        "profile_sha256": profile.sha256, "host_revision": profile.source.commit,
        "dataset": dataset_identity(config), "runtime": runtime_expectation(),
        "acceptance": _acceptance(config), "created_utc": utc_now(),
    }
    declaration["sha256"] = sha256_json(declaration)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    immutable_json(destination / "declaration.json", declaration)
    return declaration


def _validate_declaration(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _DECLARATION_FIELDS or value["schema"] != SCHEMA:
        raise ValueError("IK allocation declaration fields/schema differ")
    if type(value["run_id"]) is not str or re.fullmatch(r"ik-[a-f0-9]{32}", value["run_id"]) is None:
        raise ValueError("IK allocation run ID is not portable")
    parse_device(value["device"])
    config = validate_config(value["config"])
    _digest(value["sha256"], "declaration sha256")
    if value["sha256"] != sha256_json({key: item for key, item in value.items() if key != "sha256"}):
        raise ValueError("IK allocation semantic hash differs")
    if value["config_sha256"] != sha256_json(config):
        raise ValueError("IK allocation configuration hash differs")
    source = value["source"]
    if not isinstance(source, dict) or set(source) != {"revision", "source_sha256", "files"}:
        raise ValueError("IK allocation source fields differ")
    _revision(source["revision"], "source revision")
    files = source["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("IK allocation source closure is empty")
    for relative, digest in files.items():
        if (type(relative) is not str or not relative or "\\" in relative or ":" in relative
                or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts):
            raise ValueError("IK allocation source path must be repository-relative")
        _digest(digest, "source content")
    if source["source_sha256"] != sha256_json(files):
        raise ValueError("IK allocation source closure hash differs")
    _digest(value["profile_sha256"], "profile_sha256")
    _revision(value["host_revision"], "host_revision")
    profile = runner_right_sole_profile()
    if value["profile_sha256"] != profile.sha256 or value["host_revision"] != profile.source.commit:
        raise ValueError("IK allocation profile/host identities differ from the registered Runner workload")
    if (sha256_json(value["dataset"]) != sha256_json(dataset_identity(config))
            or sha256_json(value["acceptance"]) != sha256_json(_acceptance(config))):
        raise ValueError("IK allocation dataset or acceptance differs from the fixed workload")
    runtime = value["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {
            "python", "python_implementation", "os", "os_release", "architecture", "byte_order", "packages"}:
        raise ValueError("IK allocation runtime fields differ")
    if any(type(runtime[key]) is not str or not runtime[key] for key in runtime if key != "packages"):
        raise ValueError("IK allocation runtime values are invalid")
    if runtime["byte_order"] not in ("little", "big") or not isinstance(runtime["packages"], dict):
        raise ValueError("IK allocation runtime package values are invalid")
    if set(runtime["packages"]) != set(_PACKAGES) or any(
            item is not None and (type(item) is not str or not item) for item in runtime["packages"].values()):
        raise ValueError("IK allocation runtime package fields differ")
    try:
        created = datetime.fromisoformat(value["created_utc"].replace("Z", "+00:00"))
        if created.utcoffset() is None or created.utcoffset().total_seconds() != 0:
            raise ValueError("not UTC")
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("IK allocation timestamp must have explicit UTC") from exc
    return value


def load_declaration(run: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(run) / "declaration.json"
    with path.open("rb") as stream:
        data = stream.read(1_048_577)
    if expected_sha256 is not None:
        _digest(expected_sha256, "expected declaration hash")
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError("IK allocation raw declaration SHA-256 differs")
    return _validate_declaration(strict_json_loads(data))


def verify_source(declaration: Mapping[str, Any], source_root: str | Path | None = None) -> None:
    source = source_snapshot(source_root, require_clean=True)
    if (source["revision"] != declaration["source"]["revision"]
            or source["source_sha256"] != declaration["source"]["source_sha256"]):
        raise ValueError("executed source differs from the prospectively frozen source")


def metadata_for(declaration: Mapping[str, Any], environment_sha256: str) -> dict[str, str]:
    _digest(environment_sha256, "environment_sha256")
    return {"config_sha256": declaration["config_sha256"],
            "source_sha256": declaration["source"]["source_sha256"],
            "profile_sha256": declaration["profile_sha256"], "dataset_sha256": declaration["dataset"]["sha256"],
            "environment_sha256": environment_sha256}
