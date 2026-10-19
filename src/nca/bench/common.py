"""Small immutable measurement helpers; no benchmark starts on import."""

from __future__ import annotations

import gzip
import hashlib
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any, Mapping, Sequence

from nca.contracts import ContractError, canonical_json, sha256_json
from nca.export.build import immutable_json


def durations(values: Sequence[int]) -> dict[str, Any]:
    """Nanoseconds in; nearest-rank tails in milliseconds, without interpolation."""
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "median_ms": None, "p95_ms": None, "p99_ms": None, "maximum_ms": None}
    return {"count": len(values), "total_seconds": sum(values) / 1e9,
            "median_ms": statistics.median(ordered) / 1e6,
            "p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1] / 1e6,
            "p99_ms": ordered[math.ceil(.99 * len(ordered)) - 1] / 1e6,
            "maximum_ms": ordered[-1] / 1e6, "tail_method": "nearest_rank"}


def process_memory(pid: int) -> dict[str, int | None]:
    values = {"rss_bytes": None, "peak_rss_bytes": None}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("VmRSS", "VmHWM"):
                values["rss_bytes" if key == "VmRSS" else "peak_rss_bytes"] = int(rest.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def numpy_build() -> dict[str, Any]:
    """Record relevant wheel/build details while excluding build-machine paths."""
    import numpy
    config = numpy.__config__.CONFIG
    dependencies = config.get("Build Dependencies", {})
    return {"version": numpy.__version__, "byte_order": sys.byteorder,
            "libraries": {name: {key: entry.get(key) for key in ("name", "version", "openblas configuration")}
                          for name, entry in dependencies.items() if name in ("blas", "lapack")},
            "simd_extensions": config.get("SIMD Extensions", {}),
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "unset_thread_environment": "library default; not an asserted one-thread NumPy pool"}


def write_artifact(root: Path, name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    """A gzip extension stores deterministic compressed JSON; existing files refuse."""
    path = root / name
    if path.suffix == ".gz":
        data = gzip.compress((canonical_json(value) + "\n").encode(), mtime=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".measure-", delete=False) as stream:
                temporary = stream.name
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ContractError("immutable_exists", name) from exc
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)
        file_hash = hashlib.sha256(data).hexdigest()
    else:
        file_hash = immutable_json(path, value)
    return {"file": name, "file_sha256": file_hash, "content_sha256": sha256_json(value),
            "bytes": path.stat().st_size}


def wire_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    flags = ("primary", "secondary", "auxiliary", "tool", "attach_usable", "release_usable")
    continuous = [abs(float(left[key]) - float(right[key])) for key in ("trim", "effort", "reel")]
    continuous.extend(abs(float(a) - float(b)) for key in ("move", "aim")
                      for a, b in zip(left[key], right[key]))
    return {"maximum_continuous_difference": max(continuous),
            "boolean_actions_exact": all(left[key] == right[key] for key in flags)}


def pose(view: Mapping[str, Any]) -> dict[str, Any]:
    return {"tick": view["tick"], "position": view["position"], "velocity": view["velocity"],
            "rotation": view["rotation"], "angular_velocity": view["angular_velocity"],
            "parts": [{key: part[key] for key in ("id", "index", "position", "rotation", "detached")}
                      for part in view["parts"]]}


def pose_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    if (left["tick"] != right["tick"] or len(left["parts"]) != len(right["parts"])
            or any((a["id"], a["index"]) != (b["id"], b["index"])
                   for a, b in zip(left["parts"], right["parts"]))):
        raise ContractError("trajectory_identity", "matched tick and own-part identities required")
    position = angle = 0.0
    detached = True
    for a, b in zip(left["parts"], right["parts"]):
        position = max(position, math.dist(a["position"], b["position"]))
        angle = max(angle, abs((a["rotation"] - b["rotation"] + math.pi) % (2 * math.pi) - math.pi))
        detached &= a["detached"] == b["detached"]
    return {"maximum_part_position_difference_cells": position,
            "maximum_part_angle_difference_radians": angle, "detached_exact": detached}
