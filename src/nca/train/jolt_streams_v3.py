"""Bounded v3 child output; no unbounded line or whole-stream allocation.

The caller remains responsible for independent process-tree revocation on every
exit path. A truncated stream is retained as a failed original prefix, never as
complete evidence. Legacy v1/v2 process capture is unchanged.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import os
import selectors
import subprocess
import time

from nca.contracts import ContractError

MAX_STREAM_BYTES = 1024 * 1024
STREAM_CHUNK_BYTES = 64 * 1024


def wait_bounded_child(process, stdout, stderr, *, deadline,
                       monotonic=time.monotonic, maximum_bytes=MAX_STREAM_BYTES):
    """Drain original PIPEs into capped files until observed exit and both EOFs.

    Every read is at most 64 KiB. No byte beyond a stream's declared maximum is
    written. Overflow revokes the pipes and raises; the supervisor must revoke
    and reap its complete owned process tree in its independently reserved tail.
    The deadline is absolute on the caller's original clock, including drains.
    """
    if type(maximum_bytes) is not int or not 0 < maximum_bytes <= MAX_STREAM_BYTES:
        raise ValueError("a positive stream limit at most 1 MiB is required")
    pipes = (process.stdout, process.stderr)
    if any(pipe is None for pipe in pipes):
        raise ValueError("child stdout and stderr must be PIPEs")
    counts = [0, 0]
    try:
        with selectors.DefaultSelector() as selector:
            for index, pipe in enumerate(pipes):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, index)
            while selector.get_map() or process.poll() is None:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(process.args, timeout=0)
                ready = selector.select(timeout=min(.05, remaining))
                for key, _ in ready:
                    index = key.data
                    block = os.read(key.fd, STREAM_CHUNK_BYTES)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    available = maximum_bytes - counts[index]
                    retained = block[:available]
                    if retained:
                        if (stdout, stderr)[index].write(retained) != len(retained):
                            raise OSError("short original child stream write")
                        counts[index] += len(retained)
                    if len(block) > available:
                        raise ContractError("jolt_v3_stream_bytes", "original child output exceeds its 1 MiB stream envelope")
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout=0)
            return process.wait(timeout=remaining)
    finally:
        for pipe in pipes:
            pipe.close()
