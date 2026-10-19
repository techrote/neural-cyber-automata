"""Portable environment facts, without machine identity or private paths."""

from importlib import metadata
import hashlib
import json
from pathlib import Path
import platform
import sys

from . import __version__


def environment_report() -> dict:
    packages = {}
    for name in ("setuptools", "numpy", "torch", "gymnasium", "onnxruntime"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).parent
    sources = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(root.rglob("*"))
               if path.is_file() and path.suffix in (".py", ".gd", ".json")}
    source_digest = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": 1,
        "package_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "byte_order": sys.byteorder,
        "packages": packages,
        "package_source_sha256": source_digest,
        "package_source_files": len(sources),
        "core_runtime_dependencies": [],
        "accelerator_qualification": "unmeasured",
        "host_qualification": "requires a separately verified host run",
    }
