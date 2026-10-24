#!/usr/bin/env python3
"""Prepare the separately pinned Cybersand joint-target host for IK-CUDA-006."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nca.ik.host import PINNED_JOINT_HOST_REVISION, inspect_joint_host  # noqa: E402

GODOT_LINUX_RELEASE = (
    "https://github.com/godotengine/godot-builds/releases/download/4.7-stable/"
    "Godot_v4.7-stable_linux.x86_64.zip"
)


def run(arguments: list[str], *, env: dict[str, str] | None = None,
        timeout: float | None = 300) -> None:
    subprocess.run(arguments, check=True, env=env, timeout=timeout)


def prepare(destination: Path, executable: Path | None = None) -> tuple[Path, Path, dict]:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    host = destination / "cybersand"
    if not (host / ".git").is_dir():
        if host.exists() and any(host.iterdir()):
            raise RuntimeError("destination contains an unrelated nonempty host directory")
        host.mkdir(exist_ok=True)
        run(["git", "-C", str(host), "init"])
        run(["git", "-C", str(host), "remote", "add", "origin",
             "https://github.com/techrote/cybersand.git"])
        environment = os.environ.copy()
        environment["GIT_LFS_SKIP_SMUDGE"] = "1"
        run(["git", "-C", str(host), "fetch", "--depth=1", "origin",
             PINNED_JOINT_HOST_REVISION], env=environment)
        run(["git", "-C", str(host), "checkout", "--detach", "FETCH_HEAD"], env=environment)
    actual = subprocess.check_output(
        ["git", "-C", str(host), "rev-parse", "HEAD"], text=True).strip()
    if actual != PINNED_JOINT_HOST_REVISION:
        raise RuntimeError("existing checkout is a different joint-host revision")
    if not (sys.platform.startswith("linux") and platform.machine().lower() in {"x86_64", "amd64"}):
        raise RuntimeError("joint-target actual-host setup is qualified only on Linux x86_64")
    libraries = (
        "godot/addons/cybersand_native/bin/libcybersand_native.linux.x86_64.so,"
        "godot/addons/godot-rapier2d/bin/libgodot_rapier.linux.x86_64-unknown-linux-gnu.so"
    )
    run(["git", "-C", str(host), "lfs", "install", "--local"])
    run(["git", "-C", str(host), "lfs", "pull", f"--include={libraries}", "--exclude="])
    lock = json.loads((host / "third_party/godot-runtime.lock.json").read_text())
    if executable is None:
        archive = destination / lock["editor"]["filename"]
        if not archive.exists():
            temporary = archive.with_suffix(".download")
            try:
                with urllib.request.urlopen(GODOT_LINUX_RELEASE, timeout=120) as source, temporary.open("wb") as output:
                    remaining = int(lock["editor"]["size"])
                    while remaining > 0:
                        block = source.read(min(1024 * 1024, remaining))
                        if not block:
                            break
                        output.write(block)
                        remaining -= len(block)
                    if source.read(1):
                        raise RuntimeError("runtime download exceeds the locked size")
                temporary.replace(archive)
            finally:
                temporary.unlink(missing_ok=True)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if archive.stat().st_size != lock["editor"]["size"] or digest != lock["editor"]["sha256"]:
            raise RuntimeError("Godot archive does not match the joint-host lock")
        runtime_dir = destination / "godot"
        runtime_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as package:
            names = package.namelist()
            if names != ["Godot_v4.7-stable_linux.x86_64"] or package.testzip() is not None:
                raise RuntimeError("unexpected pinned Godot archive layout or CRC failure")
            executable = Path(package.extract(names[0], runtime_dir))
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    executable = executable.resolve()
    identity = inspect_joint_host(host, executable)
    environment = os.environ.copy()
    for variable, folder in (("XDG_DATA_HOME", "data"), ("XDG_CONFIG_HOME", "config"),
                             ("XDG_CACHE_HOME", "cache")):
        path = destination / folder
        path.mkdir(exist_ok=True)
        environment[variable] = str(path)
    imported = subprocess.run(
        [str(executable), "--headless", "--path", str(host / "godot"), "--editor", "--import"],
        env=environment, capture_output=True, text=True, timeout=120)
    if imported.returncode or "ERROR:" in imported.stdout or "ERROR:" in imported.stderr:
        raise RuntimeError("Godot import failed: " + imported.stdout[-4000:] + imported.stderr[-4000:])
    inspect_joint_host(host, executable)
    return host, executable, identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--godot", type=Path)
    parser.add_argument("--github-env", type=Path)
    args = parser.parse_args()
    host, godot, identity = prepare(args.destination, args.godot)
    if args.github_env:
        with args.github_env.open("a", encoding="utf-8") as output:
            output.write(f"NCA_IK_HOST_CHECKOUT={host}\nNCA_IK_GODOT={godot}\n")
    local = {"NCA_IK_HOST_CHECKOUT": str(host), "NCA_IK_GODOT": str(godot),
             "identity": identity}
    (args.destination / "ik-host-environment.json").write_text(
        json.dumps(local, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "host_revision": identity["host_revision"],
        "host_tree": identity["host_tree"],
        "runtime_sha256": identity["runtime_sha256"],
        "native_artifact_sha256": identity["native_artifact_sha256"],
        "rapier_artifact_sha256": identity["rapier_artifact_sha256"],
        "godot_binary_sha256": identity["godot_binary_sha256"],
        "setup": "verified",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
