"""V3 host diagnostic buffering; the pinned physical host remains unchanged.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import collections
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import threading
import time

from nca.adapters.godot import GodotHost, PINNED_HOST_REVISION, PHYSICAL_SCHEDULE
from nca.adapters.protocol import JsonChannel, PROTOCOL, ProtocolError, owned_json
from nca.evaluation.jolt_arrival import _AbsoluteDeadlineSocket
from nca.contracts import ContractError
from nca.train.jolt_streams_v3 import MAX_STREAM_BYTES, STREAM_CHUNK_BYTES


class BoundedGodotHostV3(GodotHost):
    """Keep fixed diagnostic chunks and revoke an overflowing native process."""
    def __init__(self, checkout, godot, *, original_clock, verified_identity,
                 lane="act004", mode="stepped", timeout=15):
        # Source/runtime inspection already occurs under the outer original
        # clock before v3 admission. Reusing that exact bound identity avoids
        # silently repeating a separate --version subprocess/startup allowance.
        if (lane != "act004" or mode != "stepped" or not 0 < timeout <= 15
                or not isinstance(verified_identity, dict)
                or verified_identity.get("host_revision") != PINNED_HOST_REVISION):
            raise ContractError("jolt_v3_native_identity", "one already verified pinned stepped physical host is required")
        original_clock.check("active", operation_seconds=timeout)
        deadline = min(original_clock.started + 6000, time.monotonic() + timeout)
        def remaining():
            original_clock.check("active")
            value = deadline - time.monotonic()
            if value <= 0:
                raise ContractError("jolt_v3_native_startup", "the original complete native startup interval expired")
            return value
        self.identity = dict(verified_identity)
        self.lane, self.mode = lane, mode
        self._request_id, self._closed = 0, False
        self._channel = self._process = None
        self._lock = threading.Lock()
        self._diagnostics = collections.deque(maxlen=100)
        self._diagnostic_failure, self._diagnostic_bytes = False, 0
        self._temporary = tempfile.TemporaryDirectory(prefix="nca-host-v3-")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            session_token = secrets.token_hex(32)
            environment = os.environ.copy()
            for key, folder in (("XDG_DATA_HOME", "data"), ("XDG_CONFIG_HOME", "config"), ("XDG_CACHE_HOME", "cache")):
                path = Path(self._temporary.name) / folder
                path.mkdir()
                environment[key] = str(path)
            from nca.adapters import godot as host_module
            command = [str(godot), "--headless", "--path", str(Path(checkout).resolve() / "godot"),
                "--script", str(Path(host_module.__file__).with_name("host_bridge.gd")), "--",
                f"--port={listener.getsockname()[1]}", f"--token={session_token}", f"--lane={lane}", f"--mode={mode}"]
            remaining()
            self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, env=environment)
            self._reader = threading.Thread(target=self._drain_diagnostics, daemon=True)
            self._reader.start()
            listener.settimeout(remaining())
            connection, address = listener.accept()
            if address[0] != "127.0.0.1":
                connection.close()
                raise ProtocolError("nonlocal host connection")
            connection.settimeout(remaining())
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._channel = JsonChannel(_AbsoluteDeadlineSocket(connection, deadline, time.monotonic))
            hello = self._channel.receive()
            remaining()
            if (hello.get("protocol") != PROTOCOL or hello.get("id") != 0 or hello.get("token") != session_token
                    or hello.get("lane") != lane or hello.get("mode") != mode
                    or hello.get("godot_version", {}).get("major") != 4
                    or hello.get("godot_version", {}).get("minor") != 7
                    or not str(hello.get("godot_version", {}).get("hash", "")).startswith("5b4e0cb0f")
                    or not hello.get("native_available") or not hello.get("rapier_available")
                    or hello.get("schedule") != PHYSICAL_SCHEDULE or hello.get("physics_engine") != "Rapier2D"):
                raise ProtocolError("host capability handshake mismatch")
            # Every later owner request installs its own absolute same-ledger
            # exchange deadline; remove the completed handshake wrapper only.
            self._channel.connection = connection
            connection.settimeout(timeout)
            hello.pop("token")
            self.capabilities = owned_json(hello)
            self.identity.update(lane=lane, mode=mode, schedule=hello["schedule"])
            self._check_diagnostics()
        except BaseException:
            self.close(graceful=False)
            raise
        finally:
            listener.close()

    def _check_diagnostics(self):
        if self._diagnostic_failure:
            raise ContractError("jolt_v3_native_output", "native diagnostics exceeded their fixed v3 byte envelope")

    def _drain_diagnostics(self):
        assert self._process is not None and self._process.stdout is not None
        try:
            while True:
                data = os.read(self._process.stdout.fileno(), STREAM_CHUNK_BYTES)
                if not data:
                    return
                self._diagnostic_bytes += len(data)
                if self._diagnostic_bytes > MAX_STREAM_BYTES:
                    self._diagnostic_failure = True
                    # The external supervisor still independently reaps the
                    # original owned tree. This only revokes this known child.
                    self._process.kill()
                    return
                self._diagnostics.append(data.decode("utf-8", errors="replace").rstrip()[:2000])
        except OSError:
            self._diagnostic_failure = True

    def request(self, operation, **fields):
        self._check_diagnostics()
        value = super().request(operation, **fields)
        self._check_diagnostics()
        return value

    def close(self, *, graceful=True):
        super().close(graceful=graceful and not self._diagnostic_failure)
        self._check_diagnostics()
