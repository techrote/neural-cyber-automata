"""Persistent transport preparation for Cybersand 4LY flight episodes.

The external GDScript launcher wraps CyberPhysicalFlightEpisode directly. This
module deliberately refuses to launch an unaccepted PR #229 build: source review
can prepare the bridge, but actual-host conformance begins only after an accepted
host/runtime identity is registered here.
"""

from __future__ import annotations

import collections
import hashlib
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
from typing import Any, Mapping

from .flight import (
    ACTION_PROFILE,
    ACTION_VERSION,
    ACTOR_ID,
    CONTROLLER_SHA256,
    COURSE_ID,
    COURSE_VERSION,
    DECISION_TICKS,
    EPISODE_VERSION,
    FLAT_CHANNELS,
    FLAT_OBSERVATION_VERSION,
    HOST_SCHEDULE,
    LEASE_TICKS,
    OBSERVATION_VERSION,
    PHYSICS_HZ,
    TRAINER_PROFILE_SHA256,
    FlightProtocolError,
    UnknownConsumptionError,
)
from .protocol import JsonChannel

FLIGHT_PROTOCOL = "nca-4ly-flight/1"

CANDIDATE_HOST = {
    "repository": "techrote/cybersand",
    "pull_request": 229,
    "source_review_head": "2e20731de0559ed59280d7a11f84f596d1cba985",
    "status": "SOURCE_CANDIDATE_NOT_ADMITTED",
}

# Populate only after #229's exact revision/build is accepted and independently
# qualified. Required keys are documented in docs/FLIGHT_LEARNING.md.
ACCEPTED_HOST: dict[str, Any] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_registered_host(
    checkout: str | Path, godot: str | Path
) -> dict[str, Any]:
    """Verify the future accepted #229 source/runtime without running Godot.

    The current preparation has no accepted host identity, so it fails before
    reading or executing the supplied checkout.
    """
    if ACCEPTED_HOST is None:
        raise FlightProtocolError(
            "accepted #229 flight host/build is not registered; "
            "actual-host conformance pending"
        )

    root = Path(checkout).resolve()
    expected = ACCEPTED_HOST
    try:
        def git(*args: str) -> str:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True
            ).strip()

        revision = git("rev-parse", "HEAD")
        tree = git("rev-parse", "HEAD^{tree}")
        if revision != expected["revision"] or tree != expected["tree"]:
            raise FlightProtocolError(
                "flight host source differs from the accepted revision/tree"
            )
        if git("diff", "--name-only", "HEAD"):
            raise FlightProtocolError("tracked flight host source is dirty")

        executable = Path(shutil.which(str(godot)) or str(godot)).resolve()
        version = subprocess.check_output(
            [str(executable), "--version"], text=True, timeout=15
        ).strip()
        if (
            version != expected["godot_version"]
            or sha256_file(executable) != expected["godot_sha256"]
        ):
            raise FlightProtocolError(
                "Godot runtime differs from the accepted flight build"
            )

        native_path = root / expected["native_artifact"]
        rapier_path = root / expected["rapier_artifact"]
        if sha256_file(native_path) != expected["native_sha256"]:
            raise FlightProtocolError(
                "native artifact differs from the accepted flight build"
            )
        if sha256_file(rapier_path) != expected["rapier_sha256"]:
            raise FlightProtocolError(
                "Rapier artifact differs from the accepted flight build"
            )

        source_hashes = {}
        for relative, digest in expected["host_source_sha256"].items():
            actual = hashlib.sha256(
                (root / relative).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            if actual != digest:
                raise FlightProtocolError(
                    f"accepted flight source mismatch: {relative}"
                )
            source_hashes[relative] = actual

        return {
            "repository": "techrote/cybersand",
            "host_revision": revision,
            "host_tree": tree,
            "godot_version": version,
            "godot_binary_sha256": expected["godot_sha256"],
            "native_artifact_sha256": expected["native_sha256"],
            "rapier_artifact_sha256": expected["rapier_sha256"],
            "host_source_sha256": source_hashes,
            "bridge_sha256": sha256_file(
                Path(__file__).with_name("flight_host_bridge.gd")
            ),
            "conformance_status":
                "REGISTERED_ACCEPTED_BUILD_NOT_YET_EXERCISED",
        }
    except FlightProtocolError:
        raise
    except (OSError, KeyError, ValueError, subprocess.SubprocessError) as exc:
        raise FlightProtocolError(
            f"cannot establish accepted flight host identity: {exc}"
        ) from exc


class FlightRpcTransport:
    """Strict one-request-in-flight client for nca-4ly-flight/1.

    Once send succeeds, any missing, malformed or correlation-lost reply has
    unknown consumption. The channel is closed and the caller receives
    UnknownConsumptionError; no retry or replay path exists.
    """

    def __init__(
        self, channel: JsonChannel, *, hello: Mapping[str, Any] | None = None
    ):
        self._channel = channel
        self._closed = False
        self._last_sequence = 0
        if hello is not None:
            self._validate_hello(hello)

    @staticmethod
    def _validate_hello(hello: Mapping[str, Any]) -> None:
        expected = {
            "protocol": FLIGHT_PROTOCOL,
            "id": 0,
            "episode_version": EPISODE_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "flat_observation_version": FLAT_OBSERVATION_VERSION,
            "flat_observation_length": FLAT_CHANNELS,
            "action_version": ACTION_VERSION,
            "action_profile": ACTION_PROFILE,
            "course_id": COURSE_ID,
            "course_version": COURSE_VERSION,
            "host_schedule": HOST_SCHEDULE,
            "physics_hz": PHYSICS_HZ,
            "decision_ticks": DECISION_TICKS,
            "lease_ticks": LEASE_TICKS,
            "actor_id": ACTOR_ID,
            "producer": "policy",
            "trainer_profile_sha256": TRAINER_PROFILE_SHA256,
            "controller_sha256": CONTROLLER_SHA256,
            "native_coupling": True,
        }
        for key, value in expected.items():
            if hello.get(key) != value:
                raise FlightProtocolError(
                    f"flight transport handshake mismatch: {key}"
                )

    def request(self, command: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise FlightProtocolError("flight transport is closed")
        if (
            type(command.get("sequence")) is not int
            or command["sequence"] != self._last_sequence + 1
        ):
            raise FlightProtocolError(
                "nonsequential flight client request"
            )
        sequence = int(command["sequence"])
        operation = str(command.get("command", ""))
        fields = {
            key: value
            for key, value in command.items()
            if key not in ("sequence", "command")
        }
        envelope = {
            "protocol": FLIGHT_PROTOCOL,
            "id": sequence,
            "op": operation,
            **fields,
        }
        send_started = False
        try:
            # Once socket transmission is attempted, a write failure cannot
            # prove that the peer consumed no complete request.
            send_started = True
            self._channel.send(envelope)
            result = self._channel.receive()
        except BaseException as exc:
            self._close_channel()
            if send_started:
                raise UnknownConsumptionError(
                    operation, sequence,
                    "transport_failed_after_send_started",
                ) from exc
            raise

        try:
            if (
                type(result) is not dict
                or result.get("protocol") != FLIGHT_PROTOCOL
                or result.get("id") != sequence
            ):
                raise UnknownConsumptionError(
                    operation, sequence, "reply_correlation_lost"
                )
            if type(result.get("ok")) is not bool:
                raise UnknownConsumptionError(
                    operation, sequence, "malformed_reply_status"
                )
            consumption = result.get("consumption")
            if consumption not in ("none", "confirmed", "unknown"):
                raise UnknownConsumptionError(
                    operation, sequence, "malformed_consumption_status"
                )
            if consumption == "unknown":
                raise UnknownConsumptionError(
                    operation,
                    sequence,
                    str(result.get("error", "host_unknown_consumption")),
                )
            if not result["ok"]:
                if consumption != "none":
                    raise UnknownConsumptionError(
                        operation,
                        sequence,
                        str(
                            result.get(
                                "error",
                                "host_refusal_after_possible_work",
                            )
                        ),
                    )
                raise FlightProtocolError(
                    str(result.get("error", "flight host refused request"))
                )
            if consumption != "confirmed":
                raise FlightProtocolError(
                    "successful flight reply must confirm consumption"
                )
            reply = dict(result)
            reply.pop("protocol", None)
            reply.pop("id", None)
            reply.pop("ok", None)
            reply["sequence"] = sequence
            self._last_sequence = sequence
            return reply
        except UnknownConsumptionError:
            self._close_channel()
            raise
        except BaseException:
            self._close_channel()
            raise

    def _close_channel(self) -> None:
        self._closed = True
        try:
            self._channel.connection.close()
        except BaseException:
            pass

    def close(self) -> None:
        self._close_channel()


class FlightGodotTransport(FlightRpcTransport):
    """Future actual-host launcher, hard-gated on an accepted #229 build."""

    def __init__(
        self,
        checkout: str | Path,
        godot: str | Path,
        *,
        timeout: float = 30.0,
    ):
        if not 0 < timeout <= 120:
            raise ValueError(
                "host deadline must be in (0,120] seconds"
            )
        self.identity = inspect_registered_host(checkout, godot)
        self._process: subprocess.Popen[bytes] | None = None
        self._temporary = tempfile.TemporaryDirectory(
            prefix="nca-4ly-flight-"
        )
        self._diagnostics: collections.deque[str] = collections.deque(
            maxlen=100
        )
        self._reader: threading.Thread | None = None
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        channel: JsonChannel | None = None
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(timeout)
            session_token = secrets.token_hex(32)
            environment = os.environ.copy()
            if os.name != "nt":
                for key, folder in (
                    ("XDG_DATA_HOME", "data"),
                    ("XDG_CONFIG_HOME", "config"),
                    ("XDG_CACHE_HOME", "cache"),
                ):
                    path = Path(self._temporary.name) / folder
                    path.mkdir()
                    environment[key] = str(path)
            command = [
                str(godot),
                "--headless",
                "--path",
                str(Path(checkout).resolve() / "godot"),
                "--script",
                str(Path(__file__).with_name(
                    "flight_host_bridge.gd"
                ).resolve()),
                "--",
                f"--port={listener.getsockname()[1]}",
                f"--token={session_token}",
            ]
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=environment,
            )
            self._reader = threading.Thread(
                target=self._drain_diagnostics, daemon=True
            )
            self._reader.start()
            connection, address = listener.accept()
            if address[0] != "127.0.0.1":
                connection.close()
                raise FlightProtocolError(
                    "nonlocal flight host connection"
                )
            connection.settimeout(timeout)
            connection.setsockopt(
                socket.IPPROTO_TCP, socket.TCP_NODELAY, 1
            )
            channel = JsonChannel(connection)
            hello = channel.receive()
            if hello.get("token") != session_token:
                raise FlightProtocolError(
                    "flight host authentication mismatch"
                )
            hello = dict(hello)
            hello.pop("token", None)
            super().__init__(channel, hello=hello)
        except BaseException:
            if channel is not None:
                try:
                    channel.connection.close()
                except BaseException:
                    pass
            self._shutdown_process()
            raise
        finally:
            listener.close()

    def _drain_diagnostics(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        for data in iter(self._process.stdout.readline, b""):
            self._diagnostics.append(
                data.decode(
                    "utf-8", errors="replace"
                ).rstrip()[:2000]
            )

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return tuple(self._diagnostics)

    def _shutdown_process(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
            if self._reader is not None:
                self._reader.join(timeout=2)
            if self._process.stdout is not None:
                self._process.stdout.close()
        if hasattr(self, "_temporary"):
            self._temporary.cleanup()

    def close(self) -> None:
        super().close()
        self._shutdown_process()
