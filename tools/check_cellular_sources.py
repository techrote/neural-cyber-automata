"""Verify the preserved T110 source manifests against their immutable Git trees.

This read-only check does not recreate, train or qualify a policy. The retained
manifests were reconstructed from Git and match the file hashes recorded by the
original screen; experiment trajectories and checkpoints remain separate data.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    "t110-frozen-source-manifest.json":
        "dd853bc3a9e2ff207b2918b4e079f0ace8397d99012ff75388956ab10487156e",
    "t110-failed-source-manifest.json":
        "3ab9e6064918c58396820b63ba528ed62683c053b2a15bedf72527f095313c1e",
}
ADDITIONAL = {
    "pyproject.toml", "tools/capture_replay.py", "tools/cellular_screen.py",
    "configs/cellular/jolt-three-part-hop-v1.json",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args],
                                   stderr=subprocess.PIPE)


def verify(name, expected_hash):
    raw = (ROOT / "docs/evidence" / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f"{name}: retained bytes differ from the historical hash")
    manifest = json.loads(raw)
    revision, files = manifest["revision"], manifest["files"]
    recorded_paths = git("ls-tree", "-r", "--name-only", revision).decode().splitlines()
    selected_paths = {
        path for path in recorded_paths
        if (path.startswith("src/nca/") and path.endswith((".py", ".gd")))
        or path in ADDITIONAL
    }
    if selected_paths != set(files):
        raise ValueError(f"{name}: executed source inventory differs from Git")
    if hashlib.sha256(canonical(files)).hexdigest() != manifest["sha256"]:
        raise ValueError(f"{name}: aggregate source digest differs")
    for path, expected in files.items():
        if hashlib.sha256(git("show", f"{revision}:{path}")).hexdigest() != expected:
            raise ValueError(f"{name}: {path} differs from its recorded source")
    print(f"{name}: {len(files)} source files and historical manifest hash match {revision}")


def main():
    try:
        for name, expected_hash in MANIFESTS.items():
            verify(name, expected_hash)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"T110 source verification failed: {type(exc).__name__}: {exc}")
        return 1
    print("Source provenance verified; this check does not verify experiment artifact availability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
