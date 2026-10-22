"""Disposable Python children only; no physical host or model execution."""
from collections import deque
import io
import subprocess
import sys
import tempfile
import time
from unittest import TestCase, main, mock, skipUnless

from nca.contracts import ContractError
from nca.train.jolt_streams_v3 import wait_bounded_child, MAX_STREAM_BYTES
from nca.train.jolt_native_v3 import BoundedGodotHostV3


@skipUnless(sys.platform.startswith("linux"), "v3 selector pipes belong to the declared Linux executor")
class BoundedStreamsV3Tests(TestCase):
    def spawn(self, source):
        process = subprocess.Popen([sys.executable, "-c", source], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(self.reap, process)
        return process

    @staticmethod
    def reap(process):
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)

    def test_original_stdout_stderr_and_observed_exit(self):
        child = self.spawn("import os; os.write(1,b'original stdout'); os.write(2,b'original stderr'); raise SystemExit(7)")
        stdout, stderr = io.BytesIO(), io.BytesIO()
        self.assertEqual(wait_bounded_child(child, stdout, stderr, deadline=time.monotonic()+5), 7)
        self.assertEqual(stdout.getvalue(), b"original stdout")
        self.assertEqual(stderr.getvalue(), b"original stderr")
        self.assertTrue(child.stdout.closed and child.stderr.closed)

    def test_large_newline_free_output_cannot_exceed_envelope(self):
        for fd in (1, 2):
            with self.subTest(fd=fd):
                child = self.spawn("import os; os.write(%d,b'x'*(2*1024*1024))" % fd)
                stdout, stderr = io.BytesIO(), io.BytesIO()
                with self.assertRaisesRegex(ContractError, "stream envelope"):
                    wait_bounded_child(child, stdout, stderr, deadline=time.monotonic()+5)
                self.assertEqual(len((stdout, stderr)[fd-1].getvalue()), MAX_STREAM_BYTES)
                self.assertEqual(len((stdout, stderr)[2-fd].getvalue()), 0)
                self.assertTrue(child.stdout.closed and child.stderr.closed)

    def test_exact_limit_is_accepted_and_deadline_does_not_wait_for_eof(self):
        child = self.spawn("import os; os.write(1,b'x'*1024)")
        stdout = io.BytesIO()
        self.assertEqual(wait_bounded_child(child, stdout, io.BytesIO(), deadline=time.monotonic()+5,
                                           maximum_bytes=1024), 0)
        self.assertEqual(len(stdout.getvalue()), 1024)
        child = self.spawn("import time; time.sleep(10)")
        with self.assertRaises(subprocess.TimeoutExpired):
            wait_bounded_child(child, io.BytesIO(), io.BytesIO(), deadline=time.monotonic()+.02)
        self.assertTrue(child.stdout.closed and child.stderr.closed)

    def test_native_newline_free_diagnostics_are_chunked_and_fail_closed(self):
        host = BoundedGodotHostV3.__new__(BoundedGodotHostV3)
        host._diagnostics = deque(maxlen=100)
        host._diagnostic_bytes = 0
        host._diagnostic_failure = False
        with tempfile.TemporaryFile() as stream:
            stream.write(b"x" * (MAX_STREAM_BYTES + 1))
            stream.seek(0)
            host._process = mock.Mock(stdout=stream)
            host._drain_diagnostics()
        self.assertTrue(host._diagnostic_failure)
        host._process.kill.assert_called_once_with()
        self.assertLessEqual(sum(len(part) for part in host.diagnostics), 100*2000)
        with self.assertRaises(ContractError):
            host._check_diagnostics()


class NativeStartupV3Tests(TestCase):
    def fixture(self, *, delay_per_fragment=0., start=0.):
        from nca.train import jolt_native_v3 as native
        from nca.train.jolt_execution_v3 import OriginalClockV3
        from types import SimpleNamespace
        from contextlib import ExitStack
        import json
        now = [start]
        hello = {"protocol": "nca-host/1", "id": 0, "token": "f" * 64,
            "lane": "act004", "mode": "stepped", "godot_version": {"major": 4, "minor": 7, "hash": "5b4e0cb0f"},
            "native_available": True, "rapier_available": True,
            "schedule": native.PHYSICAL_SCHEDULE, "physics_engine": "Rapier2D"}
        raw = (json.dumps(hello)+"\n").encode()
        connection = mock.Mock()
        connection.gettimeout.return_value = 15
        chunks = [raw[:20], raw[20:]]
        def recv(size):
            now[0] += delay_per_fragment
            return chunks.pop(0)
        connection.recv.side_effect = recv
        listener = mock.Mock()
        listener.accept.return_value = (connection, ("127.0.0.1", 1234))
        listener.getsockname.return_value = ("127.0.0.1", 4567)
        process = mock.Mock()
        process.poll.return_value = 0
        process.wait.return_value = 0
        stack = ExitStack()
        stack.enter_context(mock.patch.object(native.socket, "socket", return_value=listener))
        spawn = stack.enter_context(mock.patch.object(native.subprocess, "Popen", return_value=process))
        stack.enter_context(mock.patch.object(native.secrets, "token_hex", return_value="f" * 64))
        stack.enter_context(mock.patch.object(native.threading, "Thread"))
        stack.enter_context(mock.patch.object(native, "time", SimpleNamespace(monotonic=lambda:now[0])))
        self.addCleanup(stack.close)
        return native, OriginalClockV3(0, monotonic=lambda:now[0]), now, listener, connection, spawn

    def test_fragmented_handshake_has_one_absolute_deadline_and_no_extra_identity_probe(self):
        native, clock, now, listener, connection, spawn = self.fixture(delay_per_fragment=8.)
        with self.assertRaisesRegex(native.ProtocolError, "deadline"):
            native.BoundedGodotHostV3("unused", "unused", original_clock=clock,
                                     verified_identity={"host_revision": native.PINNED_HOST_REVISION})
        self.assertEqual(spawn.call_count, 1)
        self.assertEqual(now[0], 16.)
        connection.close.assert_called()
        listener.close.assert_called_once()

    def test_original_active_cutoff_rejects_spawn_before_new_startup(self):
        native, clock, _, _, _, spawn = self.fixture(start=5990.)
        with self.assertRaises(ContractError):
            native.BoundedGodotHostV3("unused", "unused", original_clock=clock,
                                     verified_identity={"host_revision": native.PINNED_HOST_REVISION})
        self.assertFalse(spawn.called)

    def test_success_restores_raw_connection_for_original_metered_owner(self):
        native, clock, _, _, connection, spawn = self.fixture(delay_per_fragment=1.)
        host = native.BoundedGodotHostV3("unused", "unused", original_clock=clock,
                                       verified_identity={"host_revision": native.PINNED_HOST_REVISION})
        self.assertIs(host._channel.connection, connection)
        self.assertNotIn("token", host.capabilities)
        self.assertEqual(spawn.call_count, 1)
        host.close(graceful=False)


if __name__ == "__main__":
    main()
