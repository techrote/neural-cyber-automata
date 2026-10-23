"""Authenticated training views and deterministic, pair-preserving IK sampling.

The accepted reader opens only training/tuning shards and the original registry.
It never reads pending.jsonl (which contains final rows), final shards or teacher
labels regenerated from inputs. Final-suite opening remains tools/ik_dataset.py's
separate, candidate-bound operation. Fixture helpers cannot authenticate a run.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import re
from typing import Any

from nca.contracts import ContractError, sha256_json, strict_json_loads
from nca.ik.data import (DatasetConfig, GeometricSample, Normalization,
                         build_split_plan, fit_normalization, iter_tasks,
                         runtime_features, validate_dataset, validate_sample,
                         _task_matches_replay)
from nca.ik.profiles import runner_right_sole_profile


# Trust anchors from docs/evidence/ik-dataset-20261022/index.json and its
# authenticated archive inventory. A self-rehashed replacement is not accepted.
ACCEPTED_MANIFEST_SHA256 = "86bc852ba3e7dd43f0a64b5374829769d803a1461f0aa53165e14fc76b17fc6c"
ACCEPTED_MANIFEST_FILE_SHA256 = "73540d9b337b2385f605d4bb4bfb7b6775def40e28449bfb096e9ead23678ded"
ACCEPTED_CONFIG_SHA256 = "fb8614412a62f7495519f463e3052ab060b0701701a3dfdb7265f0ad64488b51"
ACCEPTED_NORMALIZATION_FILE_SHA256 = "9fafdc038b0232b5d734ba33a27442a2c56d447b1605928b3972b5085adf38c2"
ACCEPTED_TRAINING_INPUTS_SHA256 = "00bf161af280724ac0629547c1cf2b797a47cdcd71dd52f15a0fee9bcb67dfeb"
ACCEPTED_FINAL_SUITE_SHA256 = "aa32e10577e40fb3ab29ba2e1f8ebb8c8324356572da5484609e82f4deba40a1"
ACCEPTED_DECLARATION_SHA256 = "126889a65bc889c11bfe7021de50371cfbb1e288669434e74dd5f66a42d110dc"
ACCEPTED_REGISTRY_ID = "29aca11b1c074416a753b27a7f64ddcc"
_REGISTRY_FILES = {
    "registry.json": (80, "59798096376a6c552ffc2417535cdb91ca8fd41cd120cc112351a76ee3aebdbe"),
    ACCEPTED_DECLARATION_SHA256 + ".envelope-started.json":
        (286, "cd4151f877bba0cb47ec4d8f16a2593769a8c4b7420aa0bb1b136de90f57adbd"),
    ACCEPTED_DECLARATION_SHA256 + ".generation-started.json":
        (218, "ed1057f7d1f120e5a07894c6cc0f3aa9ef2800db90d27cf9f706022d4cb6cc47"),
    ACCEPTED_DECLARATION_SHA256 + ".worker-started.json":
        (182, "1444aaf181cc34b9cc63d04e91b0f5a71e7a0c5ec996099f2ae7370055b27741"),
}
_ACCEPTED_TOKEN = object()
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_ROW_BYTES = 256 * 1024
_MANIFEST_FIELDS = {
    "schema", "run_id", "status", "declaration_sha256", "registry_id",
    "config_sha256", "split_plan_sha256", "schedule", "shards", "artifacts",
    "final_suite_identity_sha256", "final_state", "all_samples_sha256",
    "counts", "generation_cost", "failures", "completed_utc", "sha256",
}
_PARTITIONS = ("training", "tuning", "final")


def _fail(detail):
    raise ContractError("ik_corpus", detail)


def _sha(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _fail("exact SHA-256 identity required")
    return value


def _read_bytes(path, *, expected_sha256=None, size=None, maximum=_MAX_JSON_BYTES):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        _fail("missing or symlinked authenticated input")
    observed_size = path.stat().st_size
    if (observed_size > maximum or observed_size <= 0
            or size is not None and observed_size != size):
        _fail("authenticated input size mismatch")
    with path.open("rb") as stream:
        value = stream.read(maximum + 1)
    if len(value) != observed_size:
        _fail("authenticated input changed during read")
    if expected_sha256 is not None and hashlib.sha256(value).hexdigest() != _sha(expected_sha256):
        _fail("authenticated input digest mismatch")
    return value


def _json(value, *, sealed=False):
    result = strict_json_loads(value, max_bytes=_MAX_JSON_BYTES)
    if not isinstance(result, dict):
        _fail("JSON object required")
    if sealed and result.get("sha256") != sha256_json(
            {key: entry for key, entry in result.items() if key != "sha256"}):
        _fail("sealed record digest mismatch")
    return result


@dataclass(frozen=True)
class TrainingCorpus:
    """Immutable in-process read view; a Boolean cannot confer authentication.

    The internal token is set only after the pinned accepted reader completes.
    These Python objects are an in-process trust boundary, not a sandbox against
    callers deliberately modifying interpreter internals.
    """

    config: DatasetConfig
    normalization: Normalization
    training: tuple[GeometricSample, ...]
    tuning: tuple[GeometricSample, ...]
    manifest_sha256: str
    normalization_file_sha256: str
    final_suite_identity_sha256: str
    registry_id: str
    _authentication: Any = field(default=None, repr=False, compare=False)

    @property
    def training_inputs_sha256(self):
        return self.normalization.training_inputs_sha256

    @property
    def accepted(self):
        auth = self._authentication
        return (isinstance(auth, tuple) and len(auth) == 5 and auth[0] is _ACCEPTED_TOKEN
                and self.training is auth[1] and self.tuning is auth[2]
                and self.config is auth[3] and self.normalization is auth[4]
                and self.config.sha256 == ACCEPTED_CONFIG_SHA256
                and self.manifest_sha256 == ACCEPTED_MANIFEST_SHA256
                and self.normalization_file_sha256 == ACCEPTED_NORMALIZATION_FILE_SHA256
                and self.training_inputs_sha256 == ACCEPTED_TRAINING_INPUTS_SHA256
                and self.final_suite_identity_sha256 == ACCEPTED_FINAL_SUITE_SHA256
                and self.registry_id == ACCEPTED_REGISTRY_ID)

    @property
    def identity(self):
        return {
            "schema": "nca-ik-training-corpus-view/v1",
            "evidence_kind": "authenticated_accepted_corpus" if self.accepted else "development_fixture",
            "dataset_config_sha256": self.config.sha256,
            "manifest_sha256": self.manifest_sha256,
            "normalization_file_sha256": self.normalization_file_sha256,
            "normalization_sha256": self.normalization.sha256,
            "training_inputs_sha256": self.training_inputs_sha256,
            "final_suite_identity_sha256": self.final_suite_identity_sha256,
            "registry_id": self.registry_id,
            "training_rows": len(self.training), "tuning_rows": len(self.tuning),
            "raw_final_rows_returned": False,
        }


def _verify_normalization(samples, normalization):
    # Verification against retained training inputs, not a replacement fit.
    # Do not use tuning values or teacher success to select the fit population.
    selected = sorted((s.task for s in samples
                       if s.task.state.kind != "stress"
                       or s.task.state.stress_class == "inner_limit_boundary"),
                      key=lambda task: task.id)
    if not selected:
        _fail("empty normalization provenance")
    identity = sha256_json([[task.id, task.input_sha256] for task in selected])
    if normalization.count != len(selected) or normalization.training_inputs_sha256 != identity:
        _fail("normalization training provenance mismatch")
    rows = [runtime_features(task.inputs) for task in selected]
    mean = tuple(math.fsum(row[i] for row in rows) / len(rows) for i in range(14))
    variance = tuple(math.fsum((row[i] - mean[i]) ** 2 for row in rows) / len(rows)
                     for i in range(14))
    scale = tuple(math.sqrt(value) if value > 1e-24 else 1. for value in variance)
    if mean != normalization.mean or scale != normalization.scale:
        _fail("normalization differs from the retained training inputs")


def _load_corpus(corpus_dir, registry_dir, *, manifest_file_sha256, accepted):
    root, registry = Path(corpus_dir), Path(registry_dir)
    if root.is_symlink() or registry.is_symlink() or not root.is_dir() or not registry.is_dir():
        _fail("real corpus and retained registry directories required")
    manifest = _json(_read_bytes(root / "manifest.json",
                                 expected_sha256=manifest_file_sha256), sealed=True)
    if (set(manifest) != _MANIFEST_FIELDS or manifest["schema"] != "nca-ik-dataset-manifest/v1"
            or manifest["status"] != "complete" or manifest["failures"] != []):
        _fail("complete retained manifest required")
    if accepted and (manifest["sha256"] != ACCEPTED_MANIFEST_SHA256
                     or manifest["config_sha256"] != ACCEPTED_CONFIG_SHA256
                     or manifest["final_suite_identity_sha256"] != ACCEPTED_FINAL_SUITE_SHA256
                     or manifest["registry_id"] != ACCEPTED_REGISTRY_ID
                     or manifest["declaration_sha256"] != ACCEPTED_DECLARATION_SHA256
                     or manifest["final_state"] != "sealed"):
        _fail("accepted corpus identity mismatch")
    artifacts = {}
    if not isinstance(manifest["artifacts"], list):
        _fail("artifact inventory required")
    for item in manifest["artifacts"]:
        if (not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}
                or type(item["path"]) is not str
                or re.fullmatch(r"[a-z-]+\.(json|jsonl)", item["path"]) is None
                or item["path"] in artifacts or type(item["bytes"]) is not int
                or item["bytes"] <= 0):
            _fail("invalid artifact inventory")
        _sha(item["sha256"])
        artifacts[item["path"]] = item
    if set(artifacts) != {"declaration.json", "normalization.json", "coverage.json",
                         "attempts.jsonl", "pending.jsonl"}:
        _fail("unexpected or incomplete artifact inventory")

    def metadata(name):
        item = artifacts[name]
        return _read_bytes(root / name, expected_sha256=item["sha256"], size=item["bytes"])

    declaration = _json(metadata("declaration.json"), sealed=True)
    if declaration.get("schema") != "nca-ik-dataset-declaration/v1":
        _fail("dataset declaration schema mismatch")
    config = DatasetConfig.from_dict(declaration.get("config"))
    if accepted != (config.mode == "prospective"):
        _fail("fixture and accepted corpus modes cannot be exchanged")
    plan_hash = sha256_json(build_split_plan(config))
    if (declaration["sha256"] != manifest["declaration_sha256"]
            or declaration.get("config_sha256") != config.sha256
            or manifest["config_sha256"] != config.sha256
            or declaration.get("split_plan_sha256") != plan_hash
            or sha256_json(declaration.get("split_plan")) != plan_hash
            or manifest["split_plan_sha256"] != plan_hash
            or declaration.get("registry_id") != manifest["registry_id"]
            or declaration.get("run_id") != manifest["run_id"]
            or declaration.get("schedule") != manifest["schedule"]
            or declaration.get("profile_sha256") != runner_right_sole_profile().sha256):
        _fail("declaration/configuration/split binding mismatch")
    schedule = manifest["schedule"]
    expected_counts = {
        p: bins * config.points_per_bin[i] + config.trajectories_per_partition * config.trajectory_steps
        + 7 * config.stress_per_class
        for i, (p, bins) in enumerate(zip(_PARTITIONS, (72, 72, 80)))
    }
    if (not isinstance(schedule, dict) or schedule.get("partitions") != expected_counts
            or schedule.get("tasks") != config.expected_tasks
            or schedule.get("final_suite_identity_sha256") != manifest["final_suite_identity_sha256"]):
        _fail("declared partition populations mismatch")
    _sha(manifest["final_suite_identity_sha256"])
    marker = _json(_read_bytes(registry / "registry.json"))
    if (set(marker) != {"schema", "id"} or marker["schema"] != "nca-ik-dataset-registry/v1"
            or marker["id"] != manifest["registry_id"]
            or re.fullmatch(r"[0-9a-f]{32}", str(marker["id"])) is None):
        _fail("retained registry identity mismatch")
    if accepted:
        for name, (size, expected_sha) in _REGISTRY_FILES.items():
            _read_bytes(registry / name, expected_sha256=expected_sha, size=size)
    opened = registry / (manifest["final_suite_identity_sha256"] + ".final-opened.json")
    if opened.exists() or opened.is_symlink():
        _fail("training cannot use a corpus whose final suite has already opened")

    shards = defaultdict(list)
    names = set()
    if not isinstance(manifest["shards"], list):
        _fail("shard inventory required")
    for item in manifest["shards"]:
        if (not isinstance(item, dict)
                or set(item) != {"path", "bytes", "sha256", "partition", "rows", "sample_ids_sha256"}
                or item["partition"] not in _PARTITIONS
                or type(item["path"]) is not str or item["path"] in names
                or item["path"] != f'{item["partition"]}-{len(shards[item["partition"]]):05d}.jsonl'
                or type(item["rows"]) is not int or type(item["bytes"]) is not int
                or item["bytes"] <= 0 or not 1 <= item["rows"] <= config.batch_size):
            _fail("invalid, reordered or duplicated shard inventory")
        _sha(item["sha256"])
        _sha(item["sample_ids_sha256"])
        names.add(item["path"])
        shards[item["partition"]].append(item)
    for partition in _PARTITIONS:
        sizes = [min(config.batch_size, expected_counts[partition] - offset)
                 for offset in range(0, expected_counts[partition], config.batch_size)]
        if [item["rows"] for item in shards[partition]] != sizes:
            _fail("incomplete shard population")

    partitions = {}
    for partition in ("training", "tuning"):
        rows, scheduled = [], iter(iter_tasks(config, partition))
        for item in shards[partition]:
            raw = _read_bytes(root / item["path"], expected_sha256=item["sha256"],
                              size=item["bytes"], maximum=config.batch_size * _MAX_ROW_BYTES)
            lines = raw.splitlines(keepends=True)
            if len(lines) != item["rows"]:
                _fail("shard row count mismatch")
            ids = []
            for line in lines:
                if not line.endswith(b"\n") or len(line) > _MAX_ROW_BYTES:
                    _fail("unterminated or oversized sample row")
                sample = GeometricSample.from_dict(strict_json_loads(line, max_bytes=_MAX_ROW_BYTES))
                if (not _task_matches_replay(sample.task, next(scheduled, None), config,
                                             _accepted_archive=accepted)
                        or sample.task.partition != partition):
                    _fail("sample differs from the declared ordered partition schedule")
                # The accepted original shards are pinned by manifest and per-shard
                # byte hashes before this platform-specific replay check. Fixtures
                # retain byte-for-byte independent label equality.
                validate_sample(sample, config, _accepted_archive=accepted)
                rows.append(sample)
                ids.append(sample.task.id)
            if sha256_json(ids) != item["sample_ids_sha256"]:
                _fail("ordered shard sample identity mismatch")
        if len(rows) != expected_counts[partition] or next(scheduled, None) is not None:
            _fail("partition population incomplete")
        partitions[partition] = tuple(rows)
    # Includes family, whole-path, permitted-input and spatial split checks.
    validate_dataset(partitions["training"] + partitions["tuning"], config,
                     _accepted_archive=accepted)

    raw_normalization = metadata("normalization.json")
    normalization_hash = hashlib.sha256(raw_normalization).hexdigest()
    normalization = Normalization.from_dict(_json(raw_normalization))
    _verify_normalization(partitions["training"], normalization)
    if accepted and (normalization_hash != ACCEPTED_NORMALIZATION_FILE_SHA256
                     or normalization.training_inputs_sha256 != ACCEPTED_TRAINING_INPUTS_SHA256):
        _fail("accepted train-only normalization identity mismatch")
    return TrainingCorpus(config, normalization, partitions["training"], partitions["tuning"],
                          manifest["sha256"], normalization_hash, manifest["final_suite_identity_sha256"],
                          marker["id"],
                          (_ACCEPTED_TOKEN, partitions["training"], partitions["tuning"],
                           config, normalization) if accepted else None)


def read_training_corpus(corpus_dir, registry_dir):
    """Read the one pinned accepted corpus without exposing sealed final rows."""
    return _load_corpus(corpus_dir, registry_dir,
                        manifest_file_sha256=ACCEPTED_MANIFEST_FILE_SHA256, accepted=True)


def read_fixture_corpus(corpus_dir, registry_dir, *, manifest_file_sha256):
    """Exercise byte/split authentication on an explicitly opened fixture only."""
    return _load_corpus(corpus_dir, registry_dir,
                        manifest_file_sha256=_sha(manifest_file_sha256), accepted=False)


def make_fixture_corpus(config, training, tuning=()):
    """Build tiny in-process CPU fixtures; never grants measured-run admission."""
    if not isinstance(config, DatasetConfig) or config.mode != "fixture":
        _fail("explicit fixture configuration required")
    training, tuning = tuple(training), tuple(tuning)
    if (not training
            or any(not isinstance(s, GeometricSample) or s.task.partition != "training" for s in training)
            or any(not isinstance(s, GeometricSample) or s.task.partition != "tuning" for s in tuning)):
        _fail("fixture rows must retain their training/tuning partitions")
    validate_dataset(training + tuning, config)
    normalization = fit_normalization(training, config)
    identity = sha256_json({"kind": "opened_fixture", "config": config.sha256,
                            "training": [s.sha256 for s in training],
                            "tuning": [s.sha256 for s in tuning]})
    return TrainingCorpus(config, normalization, training, tuning, identity,
                          hashlib.sha256(normalization.to_json().encode()).hexdigest(),
                          sha256_json(["opened_fixture", identity]), "0" * 32)


class PairPreservingSampler:
    """Finite curriculum schedule with complete chronological trajectory groups.

    Each epoch presents every retained row exactly once. Group order is a pure
    SHA-256 function of seed, epoch and stage, so restoration needs no hidden
    Python/NumPy RNG state. Invalid/unlabelled rows are retained; the launcher
    accounts all presentations and explicitly records batches without a loss.
    """

    def __init__(self, samples, seed, *, batch_size=128, epochs=40):
        samples = tuple(samples)
        if (type(seed) is not int or not 0 <= seed < 2**53
                or type(batch_size) is not int or not 1 <= batch_size <= 1024
                or type(epochs) is not int or not 1 <= epochs <= 1000
                or not samples or len(samples) * epochs > 10_000_000
                or any(not isinstance(s, GeometricSample) or s.task.partition != "training"
                       for s in samples)):
            _fail("bounded training-only sampler configuration required")
        if len({s.task.id for s in samples}) != len(samples) or len({s.task.config_sha256 for s in samples}) != 1:
            _fail("sampler requires unique rows from one dataset configuration")
        # Input sequence order is authenticated too: reordering is not a resume.
        self.samples = samples
        self.seed, self.batch_size, self.epochs = seed, batch_size, epochs
        groups = defaultdict(list)
        for i, sample in enumerate(samples):
            task = sample.task
            key = (task.stage, "trajectory", task.family_id, task.trajectory_id,
                   task.state.augmentation) if task.state.kind == "trajectory" else (task.stage, "single", task.id)
            groups[key].append(i)
        ordered_groups = {}
        for key, indexes in groups.items():
            indexes.sort(key=lambda i: samples[i].task.state.step)
            if key[1] == "trajectory" and [samples[i].task.state.step for i in indexes] != list(range(len(indexes))):
                _fail("trajectory groups must retain chronological contiguous steps from zero")
            if len(indexes) > batch_size:
                _fail("batch cannot split a complete trajectory group")
            ordered_groups[key] = tuple(indexes)
        batches, counts = [], []
        for epoch in range(epochs):
            start = len(batches)
            for stage in range(4):
                keys = sorted((key for key in ordered_groups if key[0] == stage),
                              key=lambda key: (sha256_json(["ik-pair-sampler/v1", seed, epoch, key]), key))
                batch = []
                for key in keys:
                    group = ordered_groups[key]
                    if batch and len(batch) + len(group) > batch_size:
                        batches.append(tuple(batch))
                        batch = []
                    batch.extend(group)
                if batch:
                    batches.append(tuple(batch))
            counts.append(len(batches) - start)
        self._batches = tuple(batches)
        self.batches_per_epoch = tuple(counts)
        self.planned_batches = len(batches)
        self.total_presentations = len(samples) * epochs
        self.schedule_sha256 = sha256_json({
            "schema": "nca-ik-pair-sampler-plan/v1", "seed": seed,
            "batch_size": batch_size, "epochs": epochs,
            "sample_hashes": [s.sha256 for s in samples],
            "batches": batches, "curriculum": [0, 1, 2, 3],
        })
        self._prefix = [0]
        for batch in batches:
            self._prefix.append(self._prefix[-1] + len(batch))
        if self._prefix[-1] != self.total_presentations:
            _fail("sampler schedule lost or duplicated presentations")
        self._cursor = 0

    @property
    def cursor(self):
        return self._cursor

    @property
    def sample_presentations(self):
        return self._prefix[self._cursor]

    @property
    def exhausted(self):
        return self._cursor == self.planned_batches

    def next_batch(self):
        if self.exhausted:
            raise StopIteration
        indexes = self._batches[self._cursor]
        self._cursor += 1
        return tuple(self.samples[i] for i in indexes)

    def state_dict(self):
        return {"schema": "nca-ik-pair-sampler-state/v1",
                "schedule_sha256": self.schedule_sha256, "seed": self.seed,
                "batch_size": self.batch_size, "epochs": self.epochs,
                "cursor": self.cursor, "sample_presentations": self.sample_presentations}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != set(self.state_dict()):
            _fail("sampler state fields mismatch")
        current = self.state_dict()
        for key in ("schema", "schedule_sha256", "seed", "batch_size", "epochs"):
            if type(state[key]) is not type(current[key]) or state[key] != current[key]:
                _fail("sampler state identity mismatch")
        cursor, presentations = state["cursor"], state["sample_presentations"]
        if (type(cursor) is not int or not 0 <= cursor <= self.planned_batches
                or type(presentations) is not int or presentations != self._prefix[cursor]):
            _fail("sampler cursor/presentation accounting mismatch")
        self._cursor = cursor
