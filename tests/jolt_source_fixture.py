"""Artificial bounded provenance for synthetic legacy Jolt lifecycle captures.

No real source attribution or runtime qualification is claimed. Production
source_identity, recorder validation and the frozen 16 KiB writer stay real
outside this explicitly scoped fixture.
"""
from contextlib import contextmanager
from unittest.mock import patch
from nca.contracts import sha256_json


@contextmanager
def synthetic_capture_provenance():
    files={'src/nca/synthetic_fixture_only.py':'a'*64}
    identity=('f'*40,files,sha256_json(files))
    with patch('nca.data.capture.source_identity',return_value=identity):
        yield


# Copyright 2026 techrote. All rights reserved.
