"""Fail-closed continuation admission and fixed morphology scenario allocation.

The original blocked declaration remains unchanged. This module verifies a
distinct complete direction-frame prerequisite before any morphology model is
imported or host is opened; a result flag without its original files is invalid.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from itertools import product
import math
from pathlib import Path
import re
import subprocess
import time

from nca.contracts import ContractError, sha256_json, strict_json_loads
from nca.controllers.morphology.admission import COMPARISON_SHA256, comparison_plan
from .goal_frame_config import FOLLOWUP_SHA256, read_followup_config, summarize_prerequisite, verify_original_boundary

CONTINUATION_SCHEMA = "nca-morphology-continuation-declaration/v1"
PREREQUISITE_SELECTION_SHA256 = "0c111fbd8dea20a2b69aead84a10a1405035b801303c7b5fd3dec517cc0dd721"
PHASE_CAPS = {"qualification": 160, "teacher": 640, "shared_training": 4608, "tuning": 480,
              "zero_shot_final": 1920, "zero_shot_initial": 1920, "authored_and_idle": 640,
              "adaptation": 1536, "adapted_evaluation": 960}


def read_json(path):
    value = strict_json_loads(Path(path).read_bytes(), max_bytes=32 * 1024 * 1024)
    if not isinstance(value, dict):
        raise ContractError("morphology_evidence", "an immutable JSON object is required")
    return value


def verify_self(value):
    if value.get("sha256") != sha256_json({k: v for k, v in value.items() if k != "sha256"}):
        raise ContractError("morphology_evidence", "immutable evidence content differs from its self hash")


def read_prerequisite_selection(root: Path):
    selection = read_json(root / "configs/morphology/prerequisite-selection-v1.json")
    if sha256_json(selection) != PREREQUISITE_SELECTION_SHA256:
        raise ContractError("morphology_prerequisite_selection", "the immutable primary-execution designation differs")
    return selection


def verify_primary_direction_identity(root: Path, summary, summary_file):
    """Keep the designated result fixed before examining any utility outcome.

    A concurrent repeat can have valid bytes and source lineage. It is still not
    a substitute prerequisite, and this separate designation never rewrites the
    original experiment declaration or its preserved result.
    """
    selection = read_prerequisite_selection(root)
    primary = selection["primary"]
    if (summary.get("config_sha256") != selection["followup_config_sha256"]
            or any(summary.get(key) != primary[key] for key in
                   ("learner_revision", "learner_sources_sha256", "manifest_sha256"))
            or summary.get("sha256") != primary["summary_content_sha256"]
            or summary_file != {"file": "summary.json", "sha256": primary["summary_file_sha256"],
                                "bytes": primary["summary_bytes"]}):
        raise ContractError("morphology_prerequisite_selection", "only the designated original primary result may enter this gate")
    return selection


def file_receipt(path, root):
    with Path(path).open("rb") as stream:
        content_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"file": Path(path).relative_to(root).as_posix(), "sha256": content_hash, "bytes": Path(path).stat().st_size}


def verify_files(root: Path, files):
    root = root.resolve()
    names = set()
    for row in files:
        if (not isinstance(row, dict) or set(row) != {"file", "sha256", "bytes"}
                or not isinstance(row["file"], str)
                or not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                or type(row["bytes"]) is not int or row["bytes"] < 0):
            raise ContractError("morphology_evidence", "exact portable file name, byte count and SHA256 are required")
        name = row["file"]
        path = (root / name).resolve()
        if (not isinstance(name, str) or not path.is_relative_to(root) or path == root
                or path.relative_to(root).as_posix() != name or name in names
                or file_receipt(path, root) != row):
            raise ContractError("morphology_evidence", "missing, changed, duplicate or nonportable evidence file")
        names.add(name)
    return names


def verify_direction_sources(root: Path, manifest, inputs):
    """Bind the complete original source closure to its own public Git tree.

    The continuation checkout can contain new modules. Its current file set is
    therefore not a valid substitute for the earlier experiment's source scope.
    """
    revision = manifest["learner_revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ContractError("morphology_prerequisite", "an exact original Git revision is required")
    try:
        tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "-z", revision],
                                          cwd=root, stderr=subprocess.DEVNULL).decode().split("\0")
        expected_sources = {name for name in tracked if name.startswith("src/nca/")
                            and Path(name).suffix in (".py", ".gd")}
        expected_sources.update(name for name in ("pyproject.toml", "tools/capture_replay.py") if name in tracked)
        expected_inputs = expected_sources | {
            "tools/direction_frame.py", "configs/first-skill/jolt-direction-frame-v1.json",
            "configs/first-skill/jolt-screen-v1.json", "configs/morphology/entry-gate-v1.json",
            "docs/evidence/t106-prerequisite-result.json"}
        source_files = manifest["source_files"]
        if (not isinstance(source_files, dict) or set(source_files) != expected_sources
                or manifest["learner_sources_sha256"] != sha256_json(source_files)
                or inputs["revision"] != revision or set(inputs["files"]) != expected_inputs
                or inputs["sha256"] != sha256_json(inputs["files"])
                or any(inputs["files"].get(name) != value for name, value in source_files.items())):
            raise ContractError("morphology_prerequisite", "the complete original published source closure differs")
        for name, expected_hash in inputs["files"].items():
            blob = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root, stderr=subprocess.DEVNULL)
            if hashlib.sha256(blob).hexdigest() != expected_hash:
                raise ContractError("morphology_prerequisite", "original executed bytes differ from their public commit")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("morphology_prerequisite", "the original public source freeze must be locally available") from exc


def admit_direction_prerequisite(directory: Path, root: Path):
    config = read_followup_config(root / "configs/first-skill/jolt-direction-frame-v1.json")
    verify_original_boundary(config, root)
    summary_file = file_receipt(directory / "summary.json", directory)
    summary = read_json(directory / "summary.json")
    verify_self(summary)
    selection = verify_primary_direction_identity(root, summary, summary_file)
    if (summary.get("schema") != "nca-jolt-direction-frame-followup-result/v1"
            or summary.get("status") != "completed" or summary.get("phase") != "complete"
            or summary.get("error") is not None or summary.get("config_sha256") != FOLLOWUP_SHA256
            or summary.get("unconfirmed_game_requests") != 0 or summary.get("unknown_tick_counts") != 0
            or summary.get("host_start_attempts") != 1
            or type(summary.get("wall_seconds")) not in (int, float)
            or not math.isfinite(summary["wall_seconds"]) or not 0 < summary["wall_seconds"] <= 1800
            or summary.get("original_results_rewritten") is not False
            or summary.get("automatic_budget_extension") is not False
            or sha256_json(summary.get("budget")) != sha256_json(config["budget"])):
        raise ContractError("morphology_prerequisite_incomplete", "one complete unchanged bounded prerequisite is required")
    actual = summarize_prerequisite(summary["results"], config, run_completed=True)
    if sha256_json(actual) != sha256_json(summary["evaluation"]):
        raise ContractError("morphology_prerequisite", "the stored gate differs from the complete original outcomes")
    if (len(summary["results"]) != 136 or len(summary["capture_attempts"]) != 136
            or any(row["status"] != "completed" for row in summary["capture_attempts"])
            or len(summary["fitting_attempts"]) != 6
            or any(row["status"] != "completed" for row in summary["fitting_attempts"])
            or len(summary["learning"]) != 6 or any(row["epochs"] != 40 for row in summary["learning"])
            or summary["attempted_game_requests"] != summary["recorded_game_requests"]
            or summary["recorded_game_requests"] != sum(row["decisions"] for row in summary["results"])
            or summary["confirmed_physical_ticks"] != sum(row["completed_ticks"] for row in summary["results"])
            or not 0 < summary["recorded_game_requests"] <= 6000
            or summary["completed_optimizer_updates"] != sum(row["completed_optimizer_updates"] for row in summary["fitting_attempts"])):
        raise ContractError("morphology_prerequisite", "the complete teacher, six-candidate and actual work receipts are required")
    indexed = verify_files(directory, summary["artifacts"])
    indexed_hashes = {row["file"]: row["sha256"] for row in summary["artifacts"]}
    if indexed | {"summary.json"} != {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}:
        raise ContractError("morphology_prerequisite", "the complete original artifact inventory must remain unchanged")
    manifest = read_json(directory / "manifest.json")
    frozen = read_json(directory / "frozen-candidates.json")
    opening = read_json(directory / "evaluation-opened.json")
    for value in (manifest, frozen):
        verify_self(value)
    if (manifest["sha256"] != summary["manifest_sha256"] or sha256_json(manifest["config"]) != FOLLOWUP_SHA256
            or manifest["learner_revision"] != summary["learner_revision"]
            or manifest["learner_sources_sha256"] != summary["learner_sources_sha256"]
            or opening["manifest_sha256"] != manifest["sha256"]
            or opening["frozen_candidates_sha256"] != frozen["sha256"]
            or opening["all_six_final_checkpoint_hashes_frozen"] is not True
            or frozen["checkpoints"] != summary["checkpoints"]):
        raise ContractError("morphology_prerequisite", "the original source and all-candidate pre-evaluation freeze must agree")
    # A rehashed summary cannot substitute its own metrics for the actual
    # immutable captured episodes and the original per-episode metric files.
    from nca.data import read_episode
    environment = read_json(directory / "environment.json")
    for row in summary["results"]:
        episode = read_episode(directory / row["artifact"])
        raw = read_json(directory / "metrics" / f"{row['episode_id']}.json")
        audit = read_json(directory / "boundary-audit" / f"{row['episode_id']}.json")
        verify_self(audit)
        additions = {"source_boundary_regressions": audit["source_boundary_regressions"],
                     "boundary_audit_sha256": audit["sha256"], "completed_ticks": episode.consumed_ticks}
        if (sha256_json({**raw, **additions}) != sha256_json(row) or episode.sha256 != row["record_sha256"]
                or indexed_hashes.get(row["artifact"]) != row["artifact_sha256"]
                or episode.id != row["episode_id"] or episode.status != "completed"
                or episode.decisions != row["decisions"] or episode.consumed_ticks != 6 * episode.decisions
                or sha256_json(episode.evaluation.get("metrics")) != sha256_json(row["metrics"])
                or episode.evaluation.get("error") is not None or row["error"] is not None
                or episode.learner_revision != manifest["learner_revision"]
                or episode.learner_sources_sha256 != manifest["learner_sources_sha256"]
                or dict(episode.source_files) != manifest["source_files"]
                or sha256_json(episode.host_identity) != sha256_json(environment["host_identity"])
                or episode.profile_sha256 != config["profile_sha256"]
                or audit["record_sha256"] != episode.sha256 or audit["passed"] is not True
                or audit["source_boundary_regressions"] != 0 or audit["errors"]):
            raise ContractError("morphology_prerequisite", "original physical episodes, metrics, boundaries and source files must agree")
    expected = set(product(config["candidates_in_priority_order"], config["training_seeds"], ("initial", "final")))
    observed = [(row["metadata"]["family"], row["metadata"]["training_seed"], row["metadata"]["phase"])
                for row in frozen["checkpoints"]]
    if len(observed) != 12 or set(observed) != expected:
        raise ContractError("morphology_prerequisite", "every declared initial and fixed-final candidate is required")
    for row in frozen["checkpoints"]:
        metadata = row["metadata"]
        if (indexed_hashes.get("checkpoints/" + row["file"]) != row["sha256"]
                or metadata["config_sha256"] != FOLLOWUP_SHA256
                or metadata["manifest_sha256"] != manifest["sha256"]
                or metadata["learner_sources_sha256"] != manifest["learner_sources_sha256"]
                or metadata["training_privilege"] != "none" or metadata["profile_sha256"] != config["profile_sha256"]):
            raise ContractError("morphology_prerequisite", "frozen candidate bytes and source metadata must remain linked")
    inputs = read_json(directory / "published-inputs.json")
    revision = manifest["learner_revision"]
    verify_direction_sources(root, manifest, inputs)
    original_tree = subprocess.check_output(["git", "rev-parse", revision + "^{tree}"],
                                            cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    if original_tree != selection["primary"]["learner_tree"]:
        raise ContractError("morphology_prerequisite_selection", "the designated original public tree differs")
    verify_files(directory, summary["artifacts"])
    if file_receipt(directory / "summary.json", directory) != summary_file or read_json(directory / "summary.json") != summary:
        raise ContractError("morphology_prerequisite", "the original summary changed during admission")
    if not actual["complete"] or not actual["mechanical_utility_prerequisite_passed"]:
        raise ContractError("morphology_prerequisite_failed", "the unchanged mean, minimum-seed, gain and character gates failed")
    files = [*summary["artifacts"], summary_file]
    return {"prerequisite_selection_sha256": PREREQUISITE_SELECTION_SHA256,
            "summary_content_sha256": summary["sha256"], "summary_file_sha256": files[-1]["sha256"],
            "artifact_index_sha256": sha256_json(sorted(files, key=lambda r: r["file"])),
            "source_revision": revision, "source_sha256": summary["learner_sources_sha256"],
            "selected_family": actual["selected_family"], "config_sha256": FOLLOWUP_SHA256,
            "thresholds_changed": False, "original_blocked_admission_changed": False}


def scenario(profiles, profile: str, family: str, seed: int):
    match = re.fullmatch(r"(floor|drop(\d+))-(left|right)-(\d+)", family)
    if match is None or type(seed) is not int or seed < 0:
        raise ContractError("morphology_scenario", "an exact declared family and seed are required")
    row = profiles["profiles"][profile]
    drop, side, distance = int(match.group(2) or 0), match.group(3), int(match.group(4))
    x = 264. if side == "left" else 176.
    # ACT004's accepted arena places the actor origin relative to FLOOR_Y=352.
    # The root part can have an authored offset (Monowheel has +1.5 in Y).
    spawn_y = 352. - row["definition"]["spawn_floor_clearance"] - drop
    root_floor_y = row["observation"]["parts"][0]["position"][1]
    root_offset_x = row["definition"]["parts"][0]["offset"][0]
    return {"id": f"{profile}-{family}", "profile": profile, "family": family,
            "spawn": [x, spawn_y], "target": [x + root_offset_x + (-distance if side == "left" else distance), root_floor_y],
            "host_seed": seed}


def continuation_spec(root: Path, prerequisite):
    from .morphology_spec import LEARNING_SPEC
    from .morphology_task import TASK_SPEC, REFERENCE_SPEC, qualification_cases
    from .morphology_metrics import COMPARISON_METRICS_SPEC
    comparison = read_json(root / "configs/morphology/comparison-v1.json")
    profiles = read_json(root / "configs/morphology/profiles-v1.json")
    comparison_plan(comparison, profiles)
    tuning = [scenario(profiles, row["profile"], row["family"], 81020001 + i)
              for i, row in enumerate(comparison["splits"]["tuning_assignments"])]
    zero_shot = [scenario(profiles, profile, family, 81100001 + i)
                 for i, (profile, family) in enumerate(product(comparison["splits"]["zero_shot_profiles"],
                                                              comparison["splits"]["zero_shot_families"]))]
    adapted = [scenario(profiles, profile, family, 81300001 + i)
               for i, (profile, family) in enumerate(product(comparison["splits"]["zero_shot_profiles"],
                                                            comparison["splits"]["adaptation_evaluation_families"]))]
    return {"schema": CONTINUATION_SCHEMA, "id": "t106-conditional-morphology-v1", "prerequisite": prerequisite,
            "comparison_sha256": COMPARISON_SHA256, "learning": LEARNING_SPEC,
            "task": TASK_SPEC, "reference": REFERENCE_SPEC, "qualification": qualification_cases(profiles),
            "comparison_metrics": COMPARISON_METRICS_SPEC,
            "runtime": {"python": "3.12.14", "torch": "2.8.0+cpu", "numpy": "2.3.5",
                        "device": "cpu", "torch_threads": 1},
            "tuning": tuning, "zero_shot": zero_shot, "adapted_evaluation": adapted,
            "budget": {"maximum_game_requests": 13000, "maximum_wall_seconds": 1800,
                       "maximum_wall_seconds_per_candidate_seed": 900, "maximum_episode_wall_seconds": 120,
                       "phase_caps": PHASE_CAPS, "planned_phase_sum": 12864, "attempts": 1,
                       "device": "existing CPU only", "automatic_retry": False, "automatic_extension": False},
            "collection": {"shared_training": "exact16 decisions per training profile/family in each of six128-decision updates; fresh complete prefixes and valid final bootstrap at allocation cutoffs",
                           "shared_host_seed": "81000000 + policy_seed_index*10000 + update_index*1000 + case_index*100 + repeat_index",
                           "adaptation": "exact64 decisions per declared adaptation collection family,128 per independent heldout-profile fork",
                           "adaptation_host_seed": "81200000 + policy_seed_index*10000 + profile_index*1000 + case_index*100 + repeat_index",
                           "sampling_seed": "host_seed + declared_policy_seed; fresh generator at actual episode boundary",
                           "optimizer_order_seed": "BC=declared_policy_seed; shared PPO=declared_policy_seed+zero_based_update_index; adaptation=declared_policy_seed+100+zero_based_heldout_profile_index",
                           "case_pairing": "same host and sampling seed schedule for both candidate families; all evaluation policies share each case's fixed host seed"},
            "freeze": {"zero_shot": "all six initial and all six shared final checkpoints before any heldout profile opens",
                       "adaptation": "fresh weights fork from the same shared final; fresh optimizer, data and hidden state for each heldout profile; all twelve adapted checkpoints before adaptation evaluation",
                       "selection": "fixed final update; no tuning-result, holdout-result or best-seed selection"},
            "owner_feel_accepted": False, "unsupported_geometry_damage_or_assist_sweeps": "unmeasured"}


def validate_continuation(root: Path, declaration, prerequisite_directory: Path):
    prerequisite = admit_direction_prerequisite(prerequisite_directory, root)
    expected = continuation_spec(root, prerequisite)
    if sha256_json(declaration) != sha256_json(expected):
        raise ContractError("morphology_continuation", "the complete new declaration differs from its unchanged scope and actual prerequisite")
    return expected


class MorphologyBudgetExceeded(RuntimeError):
    pass


class MorphologyBudget:
    def __init__(self):
        self.started = time.monotonic()
        self.requests = 0
        self.phases = {name: 0 for name in PHASE_CAPS}
        self.candidate_seconds = {}
        self.active_candidate = None
        self.active_started = None

    def remaining_seconds(self):
        available = 1800. - (time.monotonic() - self.started)
        if self.active_candidate is not None:
            available = min(available, 900. - self.candidate_seconds.get(self.active_candidate, 0.)
                            - (time.monotonic() - self.active_started))
        return available

    def check(self):
        if self.remaining_seconds() <= 0:
            raise MorphologyBudgetExceeded("the declared aggregate or candidate-seed wall budget elapsed")

    def consume(self, phase):
        self.check()
        if phase not in PHASE_CAPS or self.phases[phase] >= PHASE_CAPS[phase] or self.requests >= 13000:
            raise MorphologyBudgetExceeded("the declared phase or aggregate request allocation is exhausted")
        self.requests += 1
        self.phases[phase] += 1

    @contextmanager
    def candidate(self, family, seed):
        if self.active_candidate is not None:
            raise ContractError("morphology_budget", "candidate wall accounting cannot be nested or shared")
        name = f"{family}-seed{seed}"
        self.active_candidate, self.active_started = name, time.monotonic()
        try:
            self.check()
            yield
        finally:
            self.candidate_seconds[name] = self.candidate_seconds.get(name, 0.) + time.monotonic() - self.active_started
            self.active_candidate = self.active_started = None

    def receipt(self):
        return {"attempted_game_requests": self.requests, "requests_by_phase": dict(self.phases),
                "wall_seconds": time.monotonic() - self.started, "candidate_wall_seconds": dict(self.candidate_seconds),
                "cooperative_wall_checks": True, "an_inflight_operation_may_cross_the_ceiling": True}
