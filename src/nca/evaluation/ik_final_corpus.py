"""Candidate-bound, original-byte final opening for compact IK, contract v1.

This extension never reads the mixed pending journal. All three complete
training histories and their tuning-selected checkpoints precede the durable
opening. Only then can authenticated original final shards be parsed. Labels
remain in their original typed records but are not supervision or scoring truth.
The independent geometric checks use the existing dataset tolerances; this is
not a change to model accuracy, split, normalization or promotion requirements.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
import hashlib
import io
import math
from pathlib import Path
import tarfile

from nca.contracts import ContractError, sha256_json, strict_json_loads, digest
from nca.ik.data import (GeometricSample, RuntimeInputs, iter_tasks,
                         independent_two_link_point, _segment_distance)
from nca.ik.kinematics import chain_from_profile
from nca.ik.profiles import runner_right_sole_profile
from nca.runs import immutable_json, utc_now
from nca.train.ik_corpus import (TrainingCorpus, ACCEPTED_MANIFEST_FILE_SHA256,
    ACCEPTED_MANIFEST_SHA256, ACCEPTED_REGISTRY_ID, _REGISTRY_FILES, _read_bytes)
from nca.train.ik_run_state import IKRunLedger
from nca.evaluation.ik_training import (SEEDS, select_tuning_checkpoint,
                                        freeze_candidates, validate_final_opening)

_TOKEN = object()
CONTRACT = "nca-ik-authenticated-final-reader/v1"
MAX_TUNING_REPORT_BYTES = 16 * 1024 * 1024
ACCEPTED_ARCHIVE_SHA256 = "97e3c547c7056e6ef727dc07437f469fde54b545bb091503a791aed85a4f99ce"
ACCEPTED_ARCHIVE_BYTES = 7340716


def _fail(detail):
    raise ContractError("ik_final_corpus", detail)


def _json(path, *, sha256=None, size=None, maximum=1_048_576):
    return strict_json_loads(_read_bytes(path, expected_sha256=sha256, size=size,
                                       maximum=maximum), max_bytes=maximum)


def _seal(value):
    return dict(value, sha256=sha256_json(value))


def _verified_training(run, declaration, corpus, frozen):
    """Check all epoch reports/checkpoint lineage before final payload access."""
    from nca.train.ik_experiment import identity_for
    selected = {}
    tuning_ids = [sample.task.id for sample in corpus.tuning]
    tuning_schedule_sha256 = sha256_json(tuning_ids)
    for seed in SEEDS:
        root = Path(run) / f"seed-{seed}"
        # A supervisor may discover late publication after writing an otherwise
        # verified terminal receipt. That durable overrun must block the final
        # opening itself, before a later begin_attempt would reject it.
        if any(root.glob("supervisor-overrun-*.json")):
            _fail("whole-process training overrun blocks final opening")
        ledger = IKRunLedger.open(root, expected_identity=identity_for(declaration, seed),
                                  expected_allocation=declaration["allocations"][str(seed)])
        state = ledger.inspect()
        starts = sorted(root.glob("supervisor-start-*.json"))
        if not starts:
            _fail("verified whole-process supervisor receipts required")
        for start in starts:
            terminal = start.with_name(start.name.replace("start", "result", 1))
            if not terminal.is_file():
                _fail("missing whole-process terminal receipt")
            receipt = _json(terminal)
            if (receipt.get("returncode") != 0
                    or receipt.get("verified_within_charged_wall") is not True
                    or receipt.get("unknown_work_blocks_resume") is not False
                    or receipt.get("whole_process_seconds", math.inf)
                    > receipt.get("charged_wall_seconds", -1)):
                _fail("uncertain whole-process training outcome")
        latest = state["latest_checkpoint"]
        if (state["active"] or state["unresolved_work"] or latest is None
                or state["last_status"] != "paused"):
            _fail("every seed requires a verified complete paused training boundary")
        checkpoint_state = latest["state"]
        extra = checkpoint_state["extra_state"]
        history = extra.get("tuning_history", ())
        sampler = checkpoint_state["sampler"]
        if (extra.get("completed_epochs") != 40 or len(history) != 40
                or sampler.get("epochs") != 40
                or sampler.get("sample_presentations") != 210400
                or state["charged"]["sample_presentations"] != 210400
                or sampler.get("cursor") != declaration["allocations"][str(seed)]["max_optimizer_steps"]):
            _fail("all forty declared epochs and presentations must finish before opening")
        # Production inspection verifies every checkpoint's bytes and marker.
        checkpoints = {}
        for event_path in sorted(root.glob("attempts/attempt-*/event-*.json")):
            event = _json(event_path)["record"]
            receipt = event.get("payload", {}).get("receipt", {})
            if event["kind"] == "complete" and "reference" in receipt and "state" in receipt:
                saved = receipt["state"]
                epoch = saved["extra_state"].get("completed_epochs", 0)
                if epoch and saved["sampler"]["cursor"] > 0:
                    checkpoints.setdefault(epoch, []).append(receipt)
        records = []
        for epoch, item in enumerate(history, 1):
            name = f"tuning-epoch-{epoch:04d}.json"
            if item.get("epoch") != epoch or item.get("file") != name:
                _fail("complete exact epoch tuning cadence required")
            report = _json(root / name, sha256=item["file_sha256"],
                           maximum=MAX_TUNING_REPORT_BYTES)
            if sha256_json(report) != item["report_sha256"]:
                _fail("tuning semantic receipt mismatch")
            if (report.get("config_sha256") != corpus.config.sha256
                    or report.get("schedule_sha256") != tuning_schedule_sha256
                    or report.get("scheduled_calls") != len(tuning_ids)
                    or report.get("received_samples") != len(tuning_ids)
                    or report.get("received_predictions") != len(tuning_ids)):
                _fail("tuning report must retain the exact original denominator and configuration")
            matches = [c for c in checkpoints.get(epoch, ())
                       if c["state"]["completed_optimizer_steps"] == item["step"]
                       and c["state"]["extra_state"].get("tuning_history", [])[-1:] == [item]]
            if not matches:
                _fail("tuning report lacks matching complete epoch checkpoint")
            # If a pause later duplicates an epoch, the earliest authentic
            # checkpoint represents that same tuning-scored parameter state.
            ref = matches[0]["reference"]
            records.append({"step": item["step"], "checkpoint_sha256": ref["sha256"],
                            "tuning": report})
        selected[seed] = select_tuning_checkpoint(records)
    expected = freeze_candidates(selected, experiment=frozen["experiment"],
        dataset_manifest_sha256=corpus.manifest_sha256,
        final_suite_identity_sha256=corpus.final_suite_identity_sha256,
        registry_id=corpus.registry_id)
    if expected != frozen:
        _fail("candidate freeze differs from all three actual tuning selections")


def _task_geometry(task, expected):
    """Exact discrete replay plus independent retained-coordinate geometry.

    Original accepted shard hashes are the authority for numeric bytes. Only
    target and original-frame point coordinates may differ from OS replay;
    existing 1e-9 frame / 1e-8 generated-FK guards remain unchanged. A 1e-9
    replay-coordinate guard also prevents a materially different target.
    """
    if (replace(task, inputs=replace(task.inputs, target_root=expected.inputs.target_root),
                original_point=expected.original_point) != expected
            or math.dist(task.inputs.target_root, expected.inputs.target_root) > 1e-9
            or math.dist(task.original_point, expected.original_point) > 1e-9):
        _fail("final task identity, schedule, semantics or replay coordinates differ")
    if task.invalid_specimen is not None:
        try:
            RuntimeInputs.from_dict(strict_json_loads(task.invalid_specimen))
        except ContractError:
            pass
        else:
            _fail("original malformed request must remain refused")
    x, y, angle = task.root_pose
    dx, dy = task.original_point[0] - x, task.original_point[1] - y
    restored = (math.cos(angle)*dx + math.sin(angle)*dy,
                -math.sin(angle)*dx + math.cos(angle)*dy)
    if math.dist(restored, task.inputs.target_root) > 1e-9:
        _fail("retained original/root-local frame geometry differs")
    if task.target_configuration is not None:
        chain = chain_from_profile(runner_right_sole_profile())
        if math.dist(independent_two_link_point(chain, task.target_configuration),
                     task.inputs.target_root) > 1e-8:
            _fail("retained legal target fails independent FK")


def _split_guards(samples, config):
    """Original complete-object family, trajectory and spatial split guards."""
    ids, groups, inputs, trajectories = set(), {}, {}, defaultdict(dict)
    segments = []
    for sample in samples:
        task = sample.task
        if task.id in ids:
            _fail("duplicate task identity")
        ids.add(task.id)
        previous = inputs.get(task.input_sha256)
        if previous is not None and previous != task.partition:
            _fail("identical permitted inputs cross partitions")
        inputs[task.input_sha256] = task.partition
        for key in (task.family_id, task.trajectory_id):
            if key in groups and groups[key] != task.partition:
                _fail("family or trajectory crosses partitions")
            groups[key] = task.partition
        point = task.inputs.target_root
        segments.append((point, point, task.partition))
        if task.state.kind == "trajectory":
            steps = trajectories[task.trajectory_id]
            if task.state.step in steps:
                _fail("duplicate trajectory step")
            steps[task.state.step] = task
    for steps in trajectories.values():
        if set(steps) != set(range(config.trajectory_steps)):
            _fail("incomplete declared trajectory")
        for index in range(config.trajectory_steps - 1):
            a, b = steps[index], steps[index + 1]
            segments.append((a.inputs.target_root, b.inputs.target_root, a.partition))
    grid, stored, guard, cell = defaultdict(list), [], config.point_guard_cells, .05
    for a, b, partition in segments:
        cells = [(x, y)
                 for x in range(math.floor((min(a[0], b[0])-guard)/cell), math.floor((max(a[0], b[0])+guard)/cell)+1)
                 for y in range(math.floor((min(a[1], b[1])-guard)/cell), math.floor((max(a[1], b[1])+guard)/cell)+1)]
        visited = set()
        for key in cells:
            for index in grid[key]:
                if index not in visited:
                    visited.add(index)
                    c, d, other = stored[index]
                    if other != partition and _segment_distance(a, b, c, d) <= guard:
                        _fail("cross-partition spatial point/trajectory guard")
        index = len(stored)
        stored.append((a, b, partition))
        for key in cells:
            grid[key].append(index)
    if ids != {task.id for task in iter_tasks(config)}:
        _fail("complete original three-partition schedule required")


@dataclass(frozen=True)
class FinalCorpus:
    config: object
    samples: tuple
    opening: dict
    frozen: dict
    amendment_sha256: str
    source_sha256: str
    _authentication: object = field(default=None, repr=False, compare=False)

    @property
    def accepted(self):
        auth = self._authentication
        return (isinstance(auth, tuple) and len(auth) == 4 and auth[0] is _TOKEN
                and auth[1] is self.samples and auth[2] is self.config
                and auth[3] == sha256_json([self.opening, self.frozen,
                                          self.amendment_sha256, self.source_sha256]))

    @property
    def canonical_tasks(self):
        return tuple(sample.task for sample in self.samples)

    @property
    def tasks(self):
        return self.canonical_tasks

    def authenticates(self, samples, config, opening=None, frozen=None):
        return (self.accepted and config is self.config
                and len(samples) == len(self.samples)
                and all(a is b for a, b in zip(samples, self.samples))
                and (opening is None or opening == self.opening)
                and (frozen is None or frozen == self.frozen))


def _read_rows(root, manifest, config, archive_bytes=None):
    """Only callable after a verified durable opening in the production path."""
    if archive_bytes is not None:
        # Decode the authenticated compressed archive only after the opening.
        # No extractall, mixed pending journal, teacher journal or filesystem
        # extraction is involved; only exact original final members are read.
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            members = archive.getmembers()
            if len({member.name for member in members}) != len(members):
                _fail("duplicate archive member names")
            def read_final(item):
                member = archive.getmember("corpus/" + item["path"])
                if not member.isfile() or member.size != item["bytes"]:
                    _fail("original final archive member metadata differs")
                stream = archive.extractfile(member)
                if stream is None:
                    _fail("original final archive member unavailable")
                with stream:
                    raw = stream.read(item["bytes"] + 1)
                if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                    _fail("original final archive member bytes differ")
                return raw
            return _parse_rows(manifest, config, read_final)
    return _parse_rows(manifest, config, lambda item: _read_bytes(
        Path(root) / item["path"], expected_sha256=item["sha256"],
        size=item["bytes"], maximum=64*256*1024))


def _parse_rows(manifest, config, read_final):
    rows, schedule, index = [], iter(iter_tasks(config, "final")), 0
    for item in manifest["shards"]:
        if item["partition"] != "final":
            continue
        if item["path"] != f"final-{index:05d}.jsonl":
            _fail("final shard order differs")
        index += 1
        raw = read_final(item)
        lines = raw.splitlines(keepends=True)
        if len(lines) != item["rows"]:
            _fail("final shard population differs")
        chunk = []
        for line in lines:
            if not line.endswith(b"\n"):
                _fail("unterminated final sample")
            sample = GeometricSample.from_dict(strict_json_loads(line, max_bytes=256*1024))
            expected = next(schedule, None)
            if expected is None:
                _fail("extra final task")
            _task_geometry(sample.task, expected)
            rows.append(sample)
            chunk.append(sample.task.id)
        if sha256_json(chunk) != item["sample_ids_sha256"]:
            _fail("final ordered sample IDs differ")
    if next(schedule, None) is not None or len(rows) != manifest["schedule"]["partitions"]["final"]:
        _fail("incomplete final schedule")
    return tuple(rows)


def open_final_corpus(corpus_root, registry, run, declaration_file_sha256, frozen_candidates,
                      *, training_corpus, amendment_sha256, source_sha256, source_root=None,
                      archive=None):
    """Open original final once, after complete freeze; read no pending journal.

    The calling continuation supervisor validates its immutable amendment and
    admission before invoking this API. This function independently binds that
    amendment hash, current clean executable source and original declaration,
    and verifies every training/candidate proof before the opening transition.
    """
    from nca.train.ik_experiment import load_declaration, _source, corpus_identity
    digest(amendment_sha256, "amendment_sha256")
    digest(source_sha256, "source_sha256")
    if (not isinstance(training_corpus, TrainingCorpus) or not training_corpus.accepted
            or _source(source_root)["source_sha256"] != source_sha256):
        _fail("original authenticated training corpus and clean amended source required")
    declaration = load_declaration(run, declaration_file_sha256)
    if corpus_identity(training_corpus) != declaration["dataset"]:
        _fail("training corpus differs from original declaration")
    root, registry = Path(corpus_root), Path(registry)
    if root.is_symlink() or registry.is_symlink() or not root.is_dir() or not registry.is_dir():
        _fail("real original corpus and registry directories required")
    manifest = _json(root / "manifest.json", sha256=ACCEPTED_MANIFEST_FILE_SHA256)
    if manifest.get("sha256") != ACCEPTED_MANIFEST_SHA256:
        _fail("original accepted manifest required")
    for name, (size, expected) in _REGISTRY_FILES.items():
        _read_bytes(registry / name, expected_sha256=expected, size=size)
    if training_corpus.registry_id != ACCEPTED_REGISTRY_ID:
        _fail("original registry required")
    # Hash opaque compressed bytes without decoding any final payload before
    # the gates. Retain the authenticated bytes to prevent path replacement
    # between this check and the post-opening selected-member read.
    archive_bytes = None if archive is None else _read_bytes(
        archive, expected_sha256=ACCEPTED_ARCHIVE_SHA256, size=ACCEPTED_ARCHIVE_BYTES)
    _verified_training(run, declaration, training_corpus, frozen_candidates)
    candidate_hashes = [entry["checkpoint_sha256"] for entry in frozen_candidates["candidates"]]
    opening_path = registry / (training_corpus.final_suite_identity_sha256 + ".final-opened.json")
    binding_path = registry / (training_corpus.final_suite_identity_sha256 + ".compact-reader-v1.json")
    binding = _seal({"schema": CONTRACT, "declaration_sha256": declaration["sha256"],
                     "frozen_candidates_sha256": frozen_candidates["sha256"],
                     "amendment_sha256": amendment_sha256, "source_sha256": source_sha256})
    if opening_path.exists():
        if not binding_path.exists() or _json(binding_path) != binding:
            _fail("existing final opening lacks identical continuation binding")
        opening = _json(opening_path)
    else:
        # The companion binding is durable first. A crash before the opening
        # can only resume this same freeze/source; it never returns final data.
        if binding_path.exists():
            if _json(binding_path) != binding:
                _fail("prior pending opening binding differs")
        else:
            immutable_json(binding_path, binding)
        opening = _seal({"schema": "nca-ik-final-opening/v1", "experiment": frozen_candidates["experiment"],
            "frozen_candidates_sha256": candidate_hashes, "opened_utc": utc_now(),
            "suite_identity_sha256": training_corpus.final_suite_identity_sha256,
            "dataset_manifest_sha256": training_corpus.manifest_sha256,
            "registry_id": training_corpus.registry_id, "future_experiments_status": "development_only"})
        immutable_json(opening_path, opening)
    for seed in SEEDS:
        validate_final_opening(opening, frozen_candidates, seed=seed)
    # Verify both durable receipts by readback before any final payload read.
    if _json(opening_path) != opening or _json(binding_path) != binding:
        _fail("durable opening readback mismatch")
    rows = _read_rows(root, manifest, training_corpus.config, archive_bytes)
    _split_guards(training_corpus.training + training_corpus.tuning + rows, training_corpus.config)
    authentication = (_TOKEN, rows, training_corpus.config,
        sha256_json([opening, frozen_candidates, amendment_sha256, source_sha256]))
    return FinalCorpus(training_corpus.config, rows, opening, frozen_candidates,
                       amendment_sha256, source_sha256, authentication)


def register(registry):
    return registry.register("evaluator", "ik-authenticated-final-corpus", 1,
        open_final_corpus, ("original-byte-authentication", "candidate-bound-opening",
                            "retained-input-geometry", "no-pending-journal"))


# Copyright 2026 techrote. All rights reserved.
