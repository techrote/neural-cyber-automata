"""Versioned target representation; no solver, labels or experiment launcher."""
from dataclasses import dataclass
import math
from typing import Literal

from nca.contracts import Record, ContractError, digest, sha256_json
from nca.ik.data import FEATURE_NAMES, runtime_features

TRANSFORM = "runner-compact-features/polar-v2"
FEATURES = FEATURE_NAMES[:2] + ("target_hip_radius_squared", "target_hip_bearing") + FEATURE_NAMES[4:]


def transform_numeric(values):
    """Transform permitted inputs only; original Cartesian tensors stay intact."""
    u, v = values[2] - 3.6, values[3] - 2.8
    return (*values[:2], u*u + v*v, math.atan2(v, u), *values[4:])


@dataclass(frozen=True)
class PolarNormalization(Record):
    count: int
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    training_inputs_sha256: str
    original_normalization_sha256: str
    feature_names: tuple[str, ...] = FEATURES
    transform: Literal["runner-compact-features/polar-v2"] = TRANSFORM
    partition: Literal["training"] = "training"
    schema: Literal["nca-ik-polar-normalization/v2"] = "nca-ik-polar-normalization/v2"

    def validate(self):
        digest(self.training_inputs_sha256, "training inputs")
        digest(self.original_normalization_sha256, "original normalization")
        if (self.count <= 0 or self.feature_names != FEATURES or len(self.mean) != 14
                or len(self.scale) != 14 or any(not math.isfinite(x) for x in (*self.mean, *self.scale))
                or any(x <= 0 for x in self.scale)):
            raise ContractError("ik_polar_normalization", "exact finite training feature fit required")


def fit_training_normalization(corpus):
    """Derive from the authenticated original 5,140 inputs, never from outcomes."""
    from nca.train.ik_corpus import TrainingCorpus
    if not isinstance(corpus, TrainingCorpus) or not corpus.accepted:
        raise ContractError("ik_polar_corpus", "authenticated original training corpus required")
    tasks = sorted((s.task for s in corpus.training if s.task.state.kind != "stress"
                    or s.task.state.stress_class == "inner_limit_boundary"), key=lambda t: t.id)
    identity = sha256_json([[t.id, t.input_sha256] for t in tasks])
    if (len(tasks) != 5140 or any(t.partition != "training" for t in tasks)
            or identity != corpus.normalization.training_inputs_sha256):
        raise ContractError("ik_polar_provenance", "original training-only fit population required")
    rows = [transform_numeric(runtime_features(t.inputs)) for t in tasks]
    mean = tuple(math.fsum(r[i] for r in rows)/len(rows) for i in range(14))
    variance = tuple(math.fsum((r[i]-mean[i])**2 for r in rows)/len(rows) for i in range(14))
    scale = tuple(math.sqrt(x) if x > 1e-24 else 1. for x in variance)
    return PolarNormalization(len(rows), mean, scale, identity, corpus.normalization.sha256)


def build_model(context, normalization):
    from nca.ik.models import build_model as build
    return build(context, normalization, feature_transform=TRANSFORM)


def build_trainer(context, normalization, dataset_config):
    from nca.train.ik import IKTrainer
    return IKTrainer(context, normalization, dataset_config, feature_transform=TRANSFORM)

# Copyright 2026 techrote. All rights reserved.
