"""Durable child output, including unsuccessful and interrupted postprocessing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def run_captured(command, *, output, label, timeout, cwd, env):
    """Run once; preserve both streams and a receipt before propagating failure."""
    output = Path(output)
    names = {stream: f"{label}.{stream}.log" for stream in ("stdout", "stderr")}
    receipt = {"format": "nca-captured-child/v1", "label": label, "completed": False,
               "returncode": None, "timeout": False}
    started = time.monotonic()
    try:
        with (output / names["stdout"]).open("xb") as stdout, (output / names["stderr"]).open("xb") as stderr:
            try:
                result = subprocess.run(command, check=True, timeout=timeout, cwd=cwd,
                                        env=env, stdout=stdout, stderr=stderr)
                receipt.update(completed=True, returncode=result.returncode)
                return result
            finally:
                for stream in (stdout, stderr):
                    stream.flush()
                    os.fsync(stream.fileno())
    except BaseException as exc:
        receipt.update(error_type=type(exc).__name__, returncode=getattr(exc, "returncode", None),
                       timeout=isinstance(exc, subprocess.TimeoutExpired))
        raise
    finally:
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["streams"] = {stream: {"file": name, "file_sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
                                      "bytes": (output / name).stat().st_size}
                              for stream, name in names.items() if (output / name).is_file()}
        with (output / f"{label}.json").open("x") as target:
            json.dump(receipt, target, sort_keys=True, separators=(",", ":"), allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
