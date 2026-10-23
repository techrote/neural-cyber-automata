"""Durable single-supervisor ownership; uncertain phases remain blocking."""
from contextlib import contextmanager
import os
from pathlib import Path
import uuid
from nca.contracts import canonical_json

@contextmanager
def serial_phase(root, *, phase, seed, declaration_sha256):
    root=Path(root)
    token=uuid.uuid4().hex
    active=root/"serial-phase-active.json"
    claim={"schema":"nca-ik-serial-phase/v1","token":token,"phase":phase,
           "seed":seed,"declaration_file_sha256":declaration_sha256}
    raw=(canonical_json(claim)+"\n").encode("utf-8")
    # Exclusive creation is the interprocess admission point. Never steal a claim.
    with active.open("xb") as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:
        yield
    except BaseException:
        # Retain ownership after uncertainty; no automatic retry or lock stealing.
        raise
    else:
        if active.read_bytes()!=raw:
            raise ValueError("serial phase ownership changed; preserve evidence and stop")
        completed=root/("serial-phase-completed-"+token+".json")
        if completed.exists():
            raise ValueError("serial phase completion already exists")
        active.rename(completed)  # Preserve the exact claim rather than deleting evidence.

# Copyright 2026 techrote. All rights reserved.
