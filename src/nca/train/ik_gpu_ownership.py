"""One fixed local TITAN ownership file across probe and training packets."""
from pathlib import Path
import tempfile
from nca.runs import immutable_json
from nca.train.ik_experiment import _read
from nca.train.ik_matched_prepare import TITAN_FINGERPRINT

def path():
    return Path(tempfile.gettempdir())/('nca-ik-titan-'+TITAN_FINGERPRINT+'.json')

def acquire(record):
    immutable_json(path(),record)

def release(record):
    if _read(path())!=record: raise ValueError('TITAN ownership changed')
    path().unlink()

def verify(record):
    if _read(path())!=record: raise ValueError('exclusive TITAN ownership required')

# Copyright 2026 techrote. All rights reserved.
