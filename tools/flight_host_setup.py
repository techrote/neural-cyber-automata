#!/usr/bin/env python3
"""Prepare the exact accepted Cybersand #249 Linux flight runtime."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nca.adapters.flight_transport import (  # noqa: E402
    ACCEPTED_HOST,
    ACCEPTED_RUNTIME_BUILD,
    inspect_registered_host,
)

GODOT_RELEASE = (
    "https://github.com/godotengine/godot-builds/releases/download/4.7-stable/"
    "Godot_v4.7-stable_linux.x86_64.zip"
)


def command(arguments: list[str], *, env=None, timeout: float = 300.0) -> None:
    subprocess.run(arguments, check=True, env=env, timeout=timeout)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(destination: Path) -> tuple[Path, Path, dict]:
    if sys.platform != "linux" or os.uname().machine.lower() not in {
        "x86_64", "amd64",
    }:
        raise RuntimeError(
            "4LY bridge qualification is registered only for Linux x86_64")
    if ACCEPTED_RUNTIME_BUILD is None:
        raise RuntimeError("accepted flight runtime record is missing")

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    host = destination / "cybersand"
    if not (host / ".git").is_dir():
        if host.exists() and any(host.iterdir()):
            raise RuntimeError("destination contains unrelated host data")
        host.mkdir(exist_ok=True)
        command(["git", "-C", str(host), "init"])
        command([
            "git", "-C", str(host), "remote", "add", "origin",
            "https://github.com/techrote/cybersand.git",
        ])
        environment = os.environ.copy()
        environment["GIT_LFS_SKIP_SMUDGE"] = "1"
        command([
            "git", "-C", str(host), "fetch", "--depth=1", "origin",
            ACCEPTED_HOST["revision"],
        ], env=environment)
        command([
            "git", "-C", str(host), "checkout", "--detach", "FETCH_HEAD",
        ], env=environment)

    revision = subprocess.check_output(
        ["git", "-C", str(host), "rev-parse", "HEAD"], text=True).strip()
    if revision != ACCEPTED_HOST["revision"]:
        raise RuntimeError("checkout is not the accepted flight host")

    include = ",".join([
        ACCEPTED_RUNTIME_BUILD["native_artifact"],
        ACCEPTED_RUNTIME_BUILD["rapier_artifact"],
    ])
    command(["git", "-C", str(host), "lfs", "install", "--local"])
    command([
        "git", "-C", str(host), "lfs", "pull",
        f"--include={include}", "--exclude=",
    ])

    lock = json.loads(
        (host / "third_party/godot-runtime.lock.json").read_text())
    if (
        lock["godot_version"] != ACCEPTED_RUNTIME_BUILD["godot_version"]
        or lock["editor"]["size"] != 75664355
        or lock["editor"]["sha256"]
            != "0b1a6c54c2c619c12e169fe9241edda4b81080b519451cec2984bf0d2c6cb73c"
    ):
        raise RuntimeError("accepted Godot lock identity changed")

    archive = destination / lock["editor"]["filename"]
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        try:
            with (
                urllib.request.urlopen(GODOT_RELEASE, timeout=120) as source,
                temporary.open("wb") as output,
            ):
                while block := source.read(1024 * 1024):
                    output.write(block)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)

    if (
        archive.stat().st_size != lock["editor"]["size"]
        or sha256_file(archive) != lock["editor"]["sha256"]
    ):
        raise RuntimeError("Godot archive differs from accepted lock")

    runtime_dir = destination / "godot"
    runtime_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        if (
            package.namelist() != ["Godot_v4.7-stable_linux.x86_64"]
            or package.testzip() is not None
        ):
            raise RuntimeError("unexpected Godot archive layout")
        godot = Path(package.extract(package.namelist()[0], runtime_dir))
    godot.chmod(godot.stat().st_mode | stat.S_IXUSR)
    godot = godot.resolve()
    if sha256_file(godot) != ACCEPTED_RUNTIME_BUILD["godot_sha256"]:
        raise RuntimeError("Godot binary identity mismatch")

    environment = os.environ.copy()
    for variable, folder in (
        ("XDG_DATA_HOME", "data"),
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_CACHE_HOME", "cache"),
    ):
        isolated = destination / folder
        isolated.mkdir(exist_ok=True)
        environment[variable] = str(isolated)
    imported = subprocess.run([
        str(godot), "--headless", "--path", str(host / "godot"),
        "--editor", "--import",
    ], env=environment, capture_output=True, text=True, timeout=300)
    if (
        imported.returncode
        or "ERROR:" in imported.stdout
        or "ERROR:" in imported.stderr
    ):
        raise RuntimeError(
            "Godot import failed:\n"
            + imported.stdout[-3000:] + imported.stderr[-3000:])

    return host, godot, inspect_registered_host(host, godot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--github-env", type=Path)
    args = parser.parse_args()
    host, godot, identity = prepare(args.destination)
    if args.github_env is not None:
        with args.github_env.open("a", encoding="utf-8") as output:
            output.write(f"NCA_FLIGHT_HOST={host}\n")
            output.write(f"NCA_FLIGHT_GODOT={godot}\n")
    print(json.dumps(identity, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
