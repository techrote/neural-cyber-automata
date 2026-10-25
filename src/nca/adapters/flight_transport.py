"""Persistent transport for source-bound Cybersand 4LY flight episodes.

The external GDScript launcher wraps CyberPhysicalFlightEpisode directly. The
runtime is pinned to the accepted #249 lease-expiry repair merge while the
qualified frozen-teacher lineage remains bound to #229. Experiment admission
and final-suite custody remain separate fail-closed gates.
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
    TEACHER_SHA256,
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
    SPLIT_GEOMETRY_VERSION,
    TRAINER_PROFILE_SHA256,
    FlightProtocolError,
    UnknownConsumptionError,
)
from .protocol import JsonChannel

FLIGHT_PROTOCOL = "nca-4ly-flight/1"

ACCEPTED_HOST = {
    "repository": "techrote/cybersand",
    "pull_request": 249,
    "revision": "9935d5696721d2c72a8273ea70454ab0c5b11361",
    "status": "ACCEPTED_HOST_LEASE_REPAIR",
    "teacher_source_merge": "b5a8570e5123e1f633766e9390761a1c8134cfa2",
    "trainer_profile_sha256": TRAINER_PROFILE_SHA256,
    "controller_sha256": CONTROLLER_SHA256,
    "teacher_sha256": TEACHER_SHA256,
    "teacher_qualification_checkout": "2bc2b6cfa9ad62512266df5804c5662b5ba8bd04",
    "teacher_final_suite_sha256": "4bd18d728581f64fa770c4dc29f765eff956f61999b96d03d8e378d54efc3401",
    "teacher_final_may_be_reused_as_learner_final": False,
}

# Exact accepted Linux build carried by Cybersand #249. Registration allows the
# finite bridge conformance run; it is not an experiment/training admission.
ACCEPTED_RUNTIME_BUILD: dict[str, Any] | None = {
    "tree": "607158a7b938471022edee62d150988db769bb78",
    "godot_version": "4.7.stable.official.5b4e0cb0f",
    "godot_sha256": "f85bbc6b15e22416c7d797cd60b63286dd67b9cb13498847056c18520ae55a75",
    "native_artifact": "godot/addons/cybersand_native/bin/libcybersand_native.linux.x86_64.so",
    "native_sha256": "ff3fb29e2260a587ab0f65b902abdd3f3e132885a31ed3a3b3a85cb80ab102cc",
    "rapier_artifact": "godot/addons/godot-rapier2d/bin/libgodot_rapier.linux.x86_64-unknown-linux-gnu.so",
    "rapier_sha256": "566e68ae53bfb2eb7018c8e7d0dd3727121bf696462039b780a33d6b8369c66a",
    "host_source_sha256": {
        "godot/scripts/physical_actor_arena.gd": "39ed6d5afcde4a6096d9a1ff9df28a78fc8007b79723304e3894f2117fef18c2",
        "godot/scripts/physical_actor_session.gd": "993fdca10bc9ad6be42b8ffd7b1f352de9d460b9c2bdafcc6b668e721057bf83",
        "godot/scripts/physical_flight_controller.gd": "47ce24b4253d8e4f7aee2d79bc329d2589fa416d258125c540a999e0151e7b4d",
        "godot/scripts/physical_flight_course.gd": "29557a84577fce245fc84c9d5922939bf118404c223dcd0dad42b537e49c2915",
        "godot/scripts/physical_flight_episode.gd": "d546b9142b92c7f0b1f1510392aad9ceb9d6aedfa866eadf9e49059c0611c0d2",
        "godot/scripts/physical_flight_observation_adapter.gd": "afa43dad5ed8737961567070bbf04754107547c065ab61cb58c56b2df447d71d",
        "godot/scripts/physical_flight_profile.gd": "82e9e7f156c29d9baf0fd4c6ac36ff6a130467e4209b7b28d22030554f57a440",
    },
}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_registered_host(
    checkout: str | Path, godot: str | Path
) -> dict[str, Any]:
    """Verify the accepted #249 source and explicitly registered runtime bytes."""
    if ACCEPTED_RUNTIME_BUILD is None:
        raise FlightProtocolError(
            "accepted #249 source is registered; actual-host runtime build/"
            "conformance pending"
        )

    root = Path(checkout).resolve()
    expected = ACCEPTED_RUNTIME_BUILD
    try:
        def git(*args: str) -> str:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True
            ).strip()

        revision = git("rev-parse", "HEAD")
        tree = git("rev-parse", "HEAD^{tree}")
        if revision != ACCEPTED_HOST["revision"] or tree != expected["tree"]:
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
        self,
        channel: JsonChannel,
        *,
        hello: Mapping[str, Any] | None = None,
        accounting=None,
        capture_labels_per_step: int = 0,
    ):
        if type(capture_labels_per_step) is not int or capture_labels_per_step not in (0, 1):
            raise ValueError("capture_labels_per_step must be zero or one")
        self._channel = channel
        self._closed = False
        self._last_sequence = 0
        self._accounting = accounting
        self._capture_labels_per_step = capture_labels_per_step
        self._drop_next_reply_for_conformance = False
        if hello is not None:
            self._validate_hello(hello)

    @staticmethod
    def _validate_hello(hello: Mapping[str, Any]) -> None:
        expected = {
            "protocol": FLIGHT_PROTOCOL,
            "id": 0,
            "ok": True,
            "episode_version": EPISODE_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "flat_observation_version": FLAT_OBSERVATION_VERSION,
            "flat_observation_length": FLAT_CHANNELS,
            "action_version": ACTION_VERSION,
            "action_profile": ACTION_PROFILE,
            "course_id": COURSE_ID,
            "course_version": COURSE_VERSION,
            "split_geometry_version": SPLIT_GEOMETRY_VERSION,
            "host_schedule": HOST_SCHEDULE,
            "physics_hz": PHYSICS_HZ,
            "decision_ticks": DECISION_TICKS,
            "lease_ticks": LEASE_TICKS,
            "actor_id": ACTOR_ID,
            "producer": "policy",
            "teacher_capture": "owner_reference_action_development_only",
            "trainer_profile_sha256": TRAINER_PROFILE_SHA256,
            "controller_sha256": CONTROLLER_SHA256,
            "native_coupling": True,
        }
        for key, value in expected.items():
            if hello.get(key) != value:
                raise FlightProtocolError(
                    f"flight transport handshake mismatch: {key}"
                )

    def set_capture_label_accounting(self, enabled: bool) -> None:
        """Switch local accounting phase without changing the wire protocol."""
        if type(enabled) is not bool:
            raise ValueError("capture accounting phase must be boolean")
        self._capture_labels_per_step = 1 if enabled else 0

    def arm_reply_loss_for_conformance(self) -> None:
        """Deliberately discard one post-dispatch reply, then fail-stop."""
        if self._closed:
            raise FlightProtocolError("flight transport is closed")
        if self._drop_next_reply_for_conformance:
            raise FlightProtocolError("conformance reply loss already armed")
        self._drop_next_reply_for_conformance = True

    def abort_after_reply(self, reason: str) -> None:
        """Record a semantic/integrity failure after a settled host receipt."""
        if type(reason) is not str or not reason:
            raise ValueError("post-receipt abort reason is required")
        if self._accounting is not None:
            self._accounting.abort(reason)

    def _reserve_accounting(self, command: Mapping[str, Any]):
        if self._accounting is None:
            return None
        operation = str(command.get("command", ""))
        if operation == "reset":
            split = command.get("split")
            producer = command.get("producer")
            if split not in ("development", "tuning"):
                raise FlightProtocolError(
                    "accounted reset requires development or tuning split"
                )
            if producer not in ("policy", "teacher"):
                raise FlightProtocolError(
                    "accounted reset requires policy or teacher owner source"
                )
            if producer == "teacher" and split != "development":
                raise FlightProtocolError(
                    "teacher capture reset must use development split"
                )
            stage = (
                "capture"
                if split == "development" and (
                    producer == "teacher" or self._capture_labels_per_step
                )
                else "tuning" if split == "tuning" else None
            )
            return self._accounting.reserve_reset(
                tuning=split == "tuning", stage=stage)
        if operation == "teacher_step":
            if self._capture_labels_per_step:
                raise FlightProtocolError(
                    "teacher capture labels must be charged only after durable "
                    "capture eligibility; transport precharge is forbidden"
                )
            repeat = command.get("repeat_ticks")
            if type(repeat) is not int:
                raise FlightProtocolError(
                    "accounted teacher step requires integer repeat_ticks"
                )
            return self._accounting.reserve_step(
                maximum_ticks=repeat, capture_labels=0, stage="capture")
        if operation == "step":
            repeat = command.get("repeat_ticks")
            if type(repeat) is not int:
                raise FlightProtocolError(
                    "accounted step requires an integer repeat_ticks"
                )
            decision = command.get("action") is not None
            return self._accounting.reserve_step(
                maximum_ticks=repeat,
                capture_labels=(
                    self._capture_labels_per_step if decision else 0),
                **({} if decision else {"decision": False}),
            )
        if operation == "close":
            return self._accounting.reserve_cleanup(host_call=True)
        if operation in ("cancel", "takeover"):
            return self._accounting.reserve_host_call(
                scope=f"host_{operation}")
        raise FlightProtocolError(
            f"unsupported accounted host operation: {operation}"
        )

    def _mark_accounting_unknown(self, reservation, reason: str) -> None:
        if reservation is None:
            return
        try:
            self._accounting.mark_unknown(reservation, reason)
        except BaseException:
            # A failed journal append keeps the already-durable reservation
            # pending; replay therefore remains fail-closed.
            pass

    def _mark_accounting_failure(self, reservation, reason: str,
                                 confirmed_ticks: int = 0) -> None:
        if reservation is None:
            return
        try:
            self._accounting.fail(
                reservation, reason,
                confirmed_physical_ticks=confirmed_ticks)
        except BaseException:
            pass

    def _complete_accounting(self, reservation, operation: str,
                             command: Mapping[str, Any],
                             reply: Mapping[str, Any]) -> None:
        if reservation is None:
            return
        confirmed_ticks = 0
        failed = False
        reason = "host_operation_failed"
        if operation in ("reset", "step", "teacher_step"):
            transition = reply.get("transition")
            if not isinstance(transition, Mapping):
                raise UnknownConsumptionError(
                    operation, int(command["sequence"]),
                    "accounting_transition_receipt_missing")
            ticks = transition.get("ticks_consumed")
            if type(ticks) is not int:
                raise UnknownConsumptionError(
                    operation, int(command["sequence"]),
                    "accounting_tick_receipt_missing")
            maximum = 0 if operation == "reset" else int(command["repeat_ticks"])
            if ticks < 0 or ticks > maximum:
                raise UnknownConsumptionError(
                    operation, int(command["sequence"]),
                    "accounting_tick_receipt_out_of_range")
            confirmed_ticks = ticks
            if type(transition.get("failed")) is not bool:
                raise UnknownConsumptionError(
                    operation, int(command["sequence"]),
                    "accounting_failure_receipt_missing")
            failed = bool(transition["failed"])
            reason = str(transition.get("reason", "host_transition_failed"))
        elif operation in ("cancel", "takeover"):
            if not isinstance(reply.get("authority"), Mapping):
                raise UnknownConsumptionError(
                    operation, int(command["sequence"]),
                    "accounting_authority_receipt_missing")
        elif operation == "close":
            if reply.get("closed") is not True:
                raise UnknownConsumptionError(
                    operation, int(command["sequence"]),
                    "accounting_close_receipt_missing")

        if failed:
            self._accounting.fail(
                reservation, reason,
                confirmed_physical_ticks=confirmed_ticks)
        else:
            self._accounting.complete(
                reservation,
                confirmed_physical_ticks=confirmed_ticks)

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
        reservation = self._reserve_accounting(command)
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
        try:
            # Once socket transmission is attempted, a write failure cannot
            # prove that the peer consumed no complete request.
            self._channel.send(envelope)
        except BaseException as exc:
            self._close_channel()
            self._mark_accounting_unknown(
                reservation, "transport_failed_after_send_started")
            raise UnknownConsumptionError(
                operation, sequence,
                "transport_failed_after_send_started",
            ) from exc

        if self._drop_next_reply_for_conformance:
            self._drop_next_reply_for_conformance = False
            self._close_channel()
            self._mark_accounting_unknown(
                reservation, "conformance_deliberate_reply_loss")
            raise UnknownConsumptionError(
                operation, sequence, "conformance_deliberate_reply_loss")

        try:
            result = self._channel.receive()
        except BaseException as exc:
            self._close_channel()
            self._mark_accounting_unknown(
                reservation, "transport_failed_after_send_started")
            raise UnknownConsumptionError(
                operation, sequence,
                "transport_failed_after_send_started",
            ) from exc

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
            self._complete_accounting(
                reservation, operation, command, reply)
            self._last_sequence = sequence
            return reply
        except UnknownConsumptionError as exc:
            self._mark_accounting_unknown(
                reservation, exc.reason)
            self._close_channel()
            raise
        except BaseException as exc:
            self._mark_accounting_failure(
                reservation,
                f"host_reply_failure:{type(exc).__name__}")
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
    """Actual-host launcher, hard-gated on registered runtime build/conformance."""

    def __init__(
        self,
        checkout: str | Path,
        godot: str | Path,
        *,
        timeout: float = 30.0,
        accounting=None,
        capture_labels_per_step: int = 0,
    ):
        if not 0 < timeout <= 120:
            raise ValueError(
                "host deadline must be in (0,120] seconds"
            )
        # Exact source/runtime bytes are registered; bridge qualification and
        # experiment admission remain separate fail-closed gates.
        if ACCEPTED_RUNTIME_BUILD is None:
            self.identity = inspect_registered_host(checkout, godot)
            raise AssertionError("unreachable")
        if accounting is None:
            raise FlightProtocolError(
                "durable accounting is required before actual-host launch"
            )
        # Once a host is accepted, reserve startup before even invoking its
        # runtime for identity inspection.  A failed identity/startup attempt is
        # therefore retained rather than retried for free.
        startup_reservation = accounting.reserve_host_start()
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
            self.identity = inspect_registered_host(checkout, godot)
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
            super().__init__(
                channel,
                hello=hello,
                accounting=accounting,
                capture_labels_per_step=capture_labels_per_step,
            )
            accounting.complete(startup_reservation)
            startup_reservation = None
        except BaseException as exc:
            if startup_reservation is not None:
                try:
                    accounting.fail(
                        startup_reservation,
                        f"host_startup_failure:{type(exc).__name__}")
                except BaseException:
                    pass
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

    @property
    def process_id(self) -> int:
        if self._process is None:
            raise FlightProtocolError("flight host process is unavailable")
        return int(self._process.pid)

    def wait_for_process_exit(self, timeout: float = 5.0) -> int | None:
        if self._process is None:
            raise FlightProtocolError("flight host process is unavailable")
        try:
            return self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

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
