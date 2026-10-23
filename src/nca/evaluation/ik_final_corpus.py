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
from pathlib import Path, PurePosixPath
import tarfile

from nca.contracts import ContractError, canonical_json, sha256_json, strict_json_loads, digest
from nca.ik.data import (GeometricSample, RuntimeInputs, iter_tasks,
                         independent_two_link_point, _segment_distance)
from nca.ik.kinematics import chain_from_profile
from nca.ik.profiles import runner_right_sole_profile
from nca.runs import immutable_json, utc_now
from nca.train.ik_corpus import (TrainingCorpus, ACCEPTED_MANIFEST_FILE_SHA256,
    ACCEPTED_MANIFEST_SHA256, ACCEPTED_REGISTRY_ID, _REGISTRY_FILES, _read_bytes,
    read_recovery_training_corpus)
from nca.train.ik_run_state import IKRunLedger
from nca.evaluation.ik_training import (SEEDS, select_tuning_checkpoint,
                                        freeze_candidates, validate_final_opening)

_TOKEN = object()
CONTRACT = "nca-ik-authenticated-final-reader/v1"
MAX_TUNING_REPORT_BYTES = 16 * 1024 * 1024
ACCEPTED_ARCHIVE_SHA256 = "97e3c547c7056e6ef727dc07437f469fde54b545bb091503a791aed85a4f99ce"
ACCEPTED_ARCHIVE_BYTES = 7340716
ACCEPTED_ARCHIVE_ROOT = "runner-ik-corpus-v1-20261022"
RECOVERY_OPENING_FILE_SHA256 = "89194b0b8466c56fbfa1d0f64020833a7168663a5da69caa7f210856a173c66c"
RECOVERY_BINDING_FILE_SHA256 = "0cf151369b4188a941b33a357991bdec31666320c2fa005c7b5fedae913acce1"
RECOVERY_FROZEN_SHA256 = "e9e1ea0e1bb4831b4323a78fef14a9a1012c97c12a61eae8833f5933c0161b85"
RECOVERY_DECLARATION_SHA256 = "37183126562dae4d3d917b46bca2197a8e58ad41bf0365eb8ecc7558ef600ca3"
RECOVERY_HISTORICAL_RECEIPTS = {
    "seed-41011/supervisor-result-0003.json": "bcc001699229c6bd139f64cd9b6b9498ed90923c0550f4e073a345db67c4e0b3",
    "seed-42013/supervisor-result-0002.json": "88fea26e1fbe99523946526ae0778aefdb37a504a002a0695f560ccb802cc4a2",
    "seed-43019/supervisor-result-0002.json": "569ee6b75b8969a3aa39e8775874c55fcdb67df659ccd335c1c476a693d7e0ee",
    **{f"seed-{seed}/supervisor-overrun-{3 if seed == 41011 else 2:04d}.json":
       "1192c573a6be4ca4b13f7d356a159def1f750fc7cfff69c6068335858d850e47" for seed in SEEDS},
}
_RECOVERY_TOKEN = object()


def _fail(detail):
    raise ContractError("ik_final_corpus", detail)


def _json(path, *, sha256=None, size=None, maximum=1_048_576):
    return strict_json_loads(_read_bytes(path, expected_sha256=sha256, size=size,
                                       maximum=maximum), max_bytes=maximum)


def _seal(value):
    return dict(value, sha256=sha256_json(value))


@dataclass(frozen=True)
class FinalRecoveryProof:
    """In-process capability for this one already-opened, frozen experiment."""
    document: dict
    _authentication: object = field(default=None, repr=False, compare=False)

    @property
    def accepted(self):
        return self._authentication == (_RECOVERY_TOKEN, sha256_json(self.document))

    @property
    def ledger_recoveries(self):
        return self.document["ledger_recoveries"]

    def authenticates_registry(self, registry, frozen=None):
        if not self.accepted or Path(registry).resolve() != Path(self.document["registry"]).resolve():
            return False
        if frozen is not None and frozen != self.document["frozen_candidates"]:
            return False
        suite = self.document["frozen_candidates"]["final_suite_identity_sha256"]
        for suffix, field in ((".final-opened.json", "opening_file_sha256"),
                              (".compact-reader-v1.json", "prior_binding_file_sha256")):
            _read_bytes(Path(registry) / (suite + suffix), expected_sha256=self.document[field])
        return True

    def allows_historical_receipt(self, path):
        if not self.accepted:
            return False
        try:
            relative = Path(path).resolve().relative_to(Path(self.document["run"]).resolve()).as_posix()
        except ValueError:
            return False
        expected = self.document["historical_receipts"].get(relative)
        if expected is None:
            return False
        _read_bytes(path, expected_sha256=expected)
        return True


def validate_recovery_proof(document):
    """Authenticate fixed historical evidence; caller separately gates admission.

    This never reads final payloads or grants a generic after-opening reader.
    New source/amendment identities are reviewed by the registered recovery
    supervisor; the original opening, binding, freeze and six failure files
    have explicit immutable pins here.
    """
    fields = {"schema", "run", "registry", "opening_file_sha256", "prior_binding_file_sha256",
              "frozen_candidates", "recovery_amendment_sha256", "source_sha256",
              "historical_receipts", "ledger_recoveries"}
    if isinstance(document, dict) and "historical_source_sha256" in document:
        fields.add("historical_source_sha256")
        if document["historical_source_sha256"] != "81d2588a0aef03b4d4a73479ed6c4656c8e57605f03b4dfeca4c41b136345f70":
            _fail("exact original paid recovery source required")
    if (not isinstance(document, dict) or set(document) != fields
            or document["schema"] != "nca-ik-final-recovery-proof/v1"
            or document["opening_file_sha256"] != RECOVERY_OPENING_FILE_SHA256
            or document["prior_binding_file_sha256"] != RECOVERY_BINDING_FILE_SHA256
            or document["historical_receipts"] != RECOVERY_HISTORICAL_RECEIPTS
            or not isinstance(document["ledger_recoveries"], dict)
            or set(document["ledger_recoveries"]) != {str(seed) for seed in SEEDS}
            or not isinstance(document["frozen_candidates"], dict)
            or any(type(document[key]) is not str or not Path(document[key]).is_absolute()
                   for key in ("run", "registry"))
            or any(not isinstance(value, dict) for value in document["ledger_recoveries"].values())):
        _fail("exact historical final recovery proof required")
    for key in ("recovery_amendment_sha256", "source_sha256"):
        digest(document[key], key)
    frozen = document["frozen_candidates"]
    if (frozen.get("sha256") != RECOVERY_FROZEN_SHA256
            or frozen.get("sha256") != sha256_json({k: v for k, v in frozen.items() if k != "sha256"})):
        _fail("original frozen candidates cannot change during recovery")
    registry = Path(document["registry"])
    suite = frozen["final_suite_identity_sha256"]
    opening = _json(registry / (suite + ".final-opened.json"), sha256=document["opening_file_sha256"])
    binding = _json(registry / (suite + ".compact-reader-v1.json"), sha256=document["prior_binding_file_sha256"])
    if (binding.get("sha256") != sha256_json({k: v for k, v in binding.items() if k != "sha256"})
            or binding.get("frozen_candidates_sha256") != frozen["sha256"]
            or binding.get("declaration_sha256") != RECOVERY_DECLARATION_SHA256):
        _fail("original final binding differs")
    for seed in SEEDS:
        validate_final_opening(opening, frozen, seed=seed)
        ledger_proof = document["ledger_recoveries"][str(seed)]
        if (ledger_proof.get("seed") != seed
                or ledger_proof.get("source_sha256") != document.get("historical_source_sha256", document["source_sha256"])
                or ledger_proof.get("frozen_candidates_sha256") != frozen["sha256"]
                or ledger_proof.get("opening_sha256") != opening["sha256"]):
            _fail("ledger recovery must bind this same source, opening and frozen seed")
    for relative, expected in document["historical_receipts"].items():
        _read_bytes(Path(document["run"]) / relative, expected_sha256=expected)
    copied = strict_json_loads(canonical_json(document))
    return FinalRecoveryProof(copied, (_RECOVERY_TOKEN, sha256_json(copied)))


def _verified_training(run, declaration, corpus, frozen, recovery_proof=None, *, _preflight_only=False):
    """Check all epoch reports/checkpoint lineage before final payload access."""
    from nca.train.ik_experiment import identity_for
    selected = {}
    tuning_ids = [sample.task.id for sample in corpus.tuning]
    tuning_schedule_sha256 = sha256_json(tuning_ids)
    recovering = recovery_proof is not None
    if _preflight_only and not recovering:
        _fail("unpaid preflight requires the exact typed recovery proof")
    if recovering and (type(recovery_proof) is not FinalRecoveryProof or not recovery_proof.accepted
                       or Path(run).resolve() != Path(recovery_proof.document["run"]).resolve()
                       or frozen != recovery_proof.document["frozen_candidates"]):
        _fail("exact same-run frozen recovery proof required")
    for seed in SEEDS:
        root = Path(run) / f"seed-{seed}"
        # A supervisor may discover late publication after writing an otherwise
        # verified terminal receipt. That durable overrun must block the final
        # opening itself, before a later begin_attempt would reject it.
        if not recovering and any(root.glob("supervisor-overrun-*.json")):
            _fail("whole-process training overrun blocks final opening")
        ledger_arguments = {} if not recovering else {
            "recovery_proof": recovery_proof.ledger_recoveries[str(seed)]}
        ledger = IKRunLedger.open(root, expected_identity=identity_for(declaration, seed),
                                  expected_allocation=declaration["allocations"][str(seed)],
                                  **ledger_arguments)
        state = ledger.inspect()
        historical = [root / Path(relative).name for relative in
                      (recovery_proof.document["historical_receipts"] if recovering else ())
                      if relative.startswith(f"seed-{seed}/")]
        if _preflight_only:
            baseline = recovery_proof.ledger_recoveries[str(seed)]
            if (state["attempt_count"] != baseline["prior_attempt_count"]
                    or state["parent_sha256"] != baseline["prior_parent_sha256"]
                    or state["latest_checkpoint"] is None
                    or state["latest_checkpoint"]["reference"]["sha256"] != baseline["checkpoint_sha256"]
                    or state["last_status"] != "paused"):
                _fail("unpaid recovery preflight requires the exact untouched training boundary")
        recovered = recovering and all(recovery_proof.allows_historical_receipt(path)
            and (_preflight_only or ledger.allows_recovered_failure(path)) for path in historical)
        for path in root.glob("supervisor-overrun-*.json"):
            if not recovered or not recovery_proof.allows_historical_receipt(path):
                _fail("whole-process training overrun blocks final opening")
        starts = sorted(root.glob("supervisor-start-*.json"))
        if not starts:
            _fail("verified whole-process supervisor receipts required")
        terminals = set(root.glob("supervisor-result-*.json"))
        for start in starts:
            terminal = start.with_name(start.name.replace("start", "result", 1))
            if not terminal.is_file():
                _fail("missing whole-process terminal receipt")
            terminals.add(terminal)
        for terminal in sorted(terminals):
            receipt = _json(terminal)
            if (receipt.get("returncode") != 0
                    or receipt.get("verified_within_charged_wall") is not True
                    or receipt.get("unknown_work_blocks_resume") is not False
                    or receipt.get("whole_process_seconds", math.inf)
                    > receipt.get("charged_wall_seconds", -1)):
                if not recovered or not recovery_proof.allows_historical_receipt(terminal):
                    _fail("uncertain whole-process training outcome")
        latest = state["latest_checkpoint"]
        if (state["active"] or state["unresolved_work"] or latest is None
                or state["last_status"] not in ({"paused", "failed"} if recovered else {"paused"})):
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


def verify_recovery_training_preflight(run, declaration, corpus, frozen, recovery_proof):
    """Read-only validation before any historical accounting mutation.

    This proves the exact original paused boundaries and all 120 reports and
    selections. It returns no final data and cannot grant a payload read; the
    final reader always performs the separate paid accounting checks again.
    """
    if (type(recovery_proof) is not FinalRecoveryProof or not recovery_proof.accepted
            or not corpus.recovery_only or corpus._recovery_proof is not recovery_proof):
        _fail("same authenticated recovery-only corpus and proof required")
    _verified_training(run, declaration, corpus, frozen, recovery_proof,
                       _preflight_only=True)
    return {"complete_training_histories": 3, "tuning_reports_verified": 120,
            "frozen_candidates_sha256": frozen["sha256"],
            "historical_accounting_mutated": False, "final_payloads_read": False}


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


def _archive_headers(archive, manifest):
    """Validate every header and exact selected paths without payload reads."""
    members = archive.getmembers()
    names = [member.name for member in members]
    if len(set(names)) != len(names):
        _fail("duplicate archive member names")
    for member in members:
        path = PurePosixPath(member.name)
        if (member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                or "\\" in member.name or path.is_absolute() or ".." in path.parts
                or path.as_posix() != member.name
                or not member.name.startswith(ACCEPTED_ARCHIVE_ROOT + "/")):
            _fail("original archive regular rooted member required")
    indexed = {member.name: member for member in members}
    for item in manifest["shards"] + manifest.get("artifacts", []):
        name = ACCEPTED_ARCHIVE_ROOT + "/corpus/" + item["path"]
        member = indexed.get(name)
        if member is None or member.size != item["bytes"]:
            _fail("original archive manifest member metadata differs")
    return indexed


def inspect_final_archive_headers(archive, corpus_root):
    """Metadata-only preflight; never select or parse a final/pending payload."""
    manifest = _json(Path(corpus_root) / "manifest.json", sha256=ACCEPTED_MANIFEST_FILE_SHA256)
    raw = _read_bytes(archive, expected_sha256=ACCEPTED_ARCHIVE_SHA256, size=ACCEPTED_ARCHIVE_BYTES)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as stream:
        indexed = _archive_headers(stream, manifest)
    return {"archive_sha256": ACCEPTED_ARCHIVE_SHA256, "archive_bytes": len(raw),
            "archive_root": ACCEPTED_ARCHIVE_ROOT, "headers_verified": len(indexed),
            "final_members": sum(item["partition"] == "final" for item in manifest["shards"]),
            "payloads_selected": 0}


def _read_rows(root, manifest, config, archive_bytes=None):
    """Only callable after a verified durable opening in the production path."""
    if archive_bytes is not None:
        # Decode the authenticated compressed archive only after the opening.
        # No extractall, mixed pending journal, teacher journal or filesystem
        # extraction is involved; only exact original final members are read.
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            members = _archive_headers(archive, manifest)
            def read_final(item):
                member = members[ACCEPTED_ARCHIVE_ROOT + "/corpus/" + item["path"]]
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
                      archive=None, recovery_proof=None):
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
    if recovery_proof is not None:
        if (type(recovery_proof) is not FinalRecoveryProof
                or not recovery_proof.authenticates_registry(registry, frozen_candidates)
                or training_corpus._recovery_proof is not recovery_proof
                or recovery_proof.document["recovery_amendment_sha256"] != amendment_sha256
                or recovery_proof.document["source_sha256"] != source_sha256):
            _fail("same authenticated opened experiment and exact recovery source required")
    elif training_corpus.recovery_only:
        _fail("recovery-only view requires its typed recovery proof")
    # Hash opaque compressed bytes without decoding any final payload before
    # the gates. Retain the authenticated bytes to prevent path replacement
    # between this check and the post-opening selected-member read.
    archive_bytes = None if archive is None else _read_bytes(
        archive, expected_sha256=ACCEPTED_ARCHIVE_SHA256, size=ACCEPTED_ARCHIVE_BYTES)
    if archive_bytes is not None:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as stream:
            _archive_headers(stream, manifest)
    if recovery_proof is None:
        _verified_training(run, declaration, training_corpus, frozen_candidates)
    else:
        _verified_training(run, declaration, training_corpus, frozen_candidates,
                           recovery_proof=recovery_proof)
    candidate_hashes = [entry["checkpoint_sha256"] for entry in frozen_candidates["candidates"]]
    opening_path = registry / (training_corpus.final_suite_identity_sha256 + ".final-opened.json")
    binding_path = registry / (training_corpus.final_suite_identity_sha256 + ".compact-reader-v1.json")
    binding = _seal({"schema": CONTRACT, "declaration_sha256": declaration["sha256"],
                     "frozen_candidates_sha256": frozen_candidates["sha256"],
                     "amendment_sha256": amendment_sha256, "source_sha256": source_sha256})
    recovery_binding_path = None
    recovery_binding = None
    if recovery_proof is not None:
        # The original opening and binding are historical evidence, never
        # overwritten or relabelled as a successful fresh opening.
        opening = _json(opening_path, sha256=recovery_proof.document["opening_file_sha256"])
        binding = _json(binding_path, sha256=recovery_proof.document["prior_binding_file_sha256"])
        recovery_binding_path = registry / (training_corpus.final_suite_identity_sha256
                                            + ".compact-recovery-reader-v1.json")
        recovery_binding = _seal({"schema": "nca-ik-authenticated-final-recovery-reader/v1",
            "declaration_sha256": declaration["sha256"],
            "frozen_candidates_sha256": frozen_candidates["sha256"],
            "original_opening_file_sha256": recovery_proof.document["opening_file_sha256"],
            "original_binding_file_sha256": recovery_proof.document["prior_binding_file_sha256"],
            "amendment_sha256": amendment_sha256, "source_sha256": source_sha256})
        if recovery_binding_path.exists():
            if _json(recovery_binding_path) != recovery_binding:
                _fail("existing recovery binding differs")
        else:
            immutable_json(recovery_binding_path, recovery_binding)
    elif opening_path.exists():
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
    if recovery_binding_path is not None and _json(recovery_binding_path) != recovery_binding:
        _fail("durable recovery binding readback mismatch")
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
