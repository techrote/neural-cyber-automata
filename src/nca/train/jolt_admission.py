"""Prospective LEARN-JOLT-002 admission, static splits and conserved costs.

Importing this module constructs neither a host nor a neural model.  Historical
teacher qualification is an input to a new experiment, never a learned result.
Every reservation is charged before work and is retained after uncertain work.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

from nca.contracts import canonical_json, sha256_json, strict_json_loads


SCHEMA = "nca-jolt-learn-002/v1"
TEACHER_REVISION = "46a22f98dce78be3c4ba534d955e58d5691c7d96"
TEACHER_SPEC_SHA256 = "2a04924885e3147881fef5ed3421c7c0e43457faa83e5c13b8e3c351c0384038"
TEACHER_DECLARATION_SHA256 = "bc3cf6843d670a9305fa2f70bfa7531c865fa05363f3d55c88f8c554e520d30c"
HOST_REVISION = "f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497"
HOST_LAYOUT_SHA256 = "8d1e80547101ba752cf4d54812f7073957eed3a279a2c0c56e50e660e9067a29"
POLICY_SEEDS = (190103, 190207, 190301)
STAGE_DECISIONS = {"teacher": 2560, "ppo": 6144, "open": 1280, "sealed": 8960}
STAGE_EPISODES = {"teacher": 64, "ppo": 6144, "open": 32, "sealed": 224}
REQUIRED_COUNTERS = frozenset({
    "attempts", "host_starts", "host_start_attempts", "host_terminations",
    "owner_requests", "owner_reset", "owner_profile", "owner_configure",
    "owner_cancel", "owner_close", "owner_step", "owner_observe", "decisions",
    "requested_ticks", "confirmed_ticks", "refused_ticks", "uncertain_ticks",
    "episodes_started", "episodes_completed", "episodes_aborted",
    "checkpoint_writes", "checkpoint_loads", "checkpoint_executions",
    "inference_forward", "bootstrap_forward", "audit_forward",
    "bc_forward", "bc_burn_in", "bc_optimizer", "bc_eligible_samples",
    "ppo_forward", "ppo_burn_in", "ppo_base_forward", "ppo_base_burn_in",
    "ppo_optimizer", "ppo_actor_samples", "ppo_critic_samples",
    "ppo_replay_forward", "ppo_replay_base_forward",
})

# These are factual rectangle coordinates/material IDs from the source-pinned
# public arena layout.  No physics, native state or random generator is ported.
STATIC_RECTANGLES = (
    (128, 352, 512, 8, 1), (128, 140, 4, 220, 1),
    (636, 140, 4, 220, 1), (128, 136, 512, 4, 1),
    (242, 294, 56, 4, 1), (436, 265, 64, 4, 1),
    (350, 212, 60, 4, 1), (324, 232, 8, 16, 56),
    (318, 336, 9, 16, 56), (152, 348, 52, 4, 2),
    (536, 346, 68, 6, 2),
)


class AdmissionError(ValueError):
    """A prospective identity, split, or finite cost contract was not met."""

    def __init__(self, message: str, *, code: str = "jolt_admission"):
        self.code = code
        super().__init__(message)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AdmissionError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise AdmissionError(f"{name} must be a SHA-256 digest")
    return value


def _relative_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise AdmissionError("source paths must be repository-relative")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise AdmissionError(f"missing or escaping input: {relative}")
    return candidate


class BudgetLedger:
    """Single-invocation monotonic ledger; caps cannot be refilled by a reset.

    ``reserve_many`` is atomic across related counters.  Use it for a native
    step so ticks, decisions, owner requests and the stage cannot disagree.
    ``started`` is the outer invocation's original monotonic start, including
    startup and validation.  Cleanup gets the same original absolute deadline.
    """

    def __init__(self, budget: Mapping[str, Any], *, started: float | None = None,
                 monotonic: Callable[[], float] = time.monotonic):
        self._budget = deepcopy(dict(budget))
        self._clock = monotonic
        self.started = float(monotonic() if started is None else started)
        if not math.isfinite(self.started) or self.started > monotonic():
            raise AdmissionError("original invocation clock is invalid or in the future")
        self.seconds = float(budget["maximum_wall_seconds"])
        self.reserve_seconds = float(budget["cleanup_reserve_seconds"])
        if not (math.isfinite(self.seconds) and math.isfinite(self.reserve_seconds)
                and 0 < self.reserve_seconds < self.seconds):
            raise AdmissionError("finite wall and cleanup ceilings are required")
        self.deadline = self.started + self.seconds
        self.active_deadline = self.deadline - self.reserve_seconds
        self.limits = {key: _integer(value, key) for key, value in budget["counters"].items()}
        self.stage_limits = {
            stage: {key: _integer(value, f"{stage}.{key}") for key, value in limits.items()}
            for stage, limits in budget["stages"].items()
        }
        self.counts = {key: 0 for key in self.limits}
        self.stage_counts = {stage: {key: 0 for key in limits} for stage, limits in self.stage_limits.items()}
        self._reservations: list[dict[str, Any]] = []
        self._reservation_digest = hashlib.sha256(b"[")
        self._lock = threading.Lock()
        self._closed = False

    def check(self, *, cleanup: bool = False, allowance_seconds: float = 0.0) -> None:
        if self._closed:
            raise AdmissionError("invocation ledger is closed")
        if not math.isfinite(allowance_seconds) or allowance_seconds < 0:
            raise AdmissionError("operation allowance must be finite and nonnegative")
        limit = self.deadline if cleanup else self.active_deadline
        if self._clock() + allowance_seconds >= limit:
            raise AdmissionError("original whole-process deadline or cleanup reserve exhausted",
                                 code="budget_whole_deadline" if cleanup else "budget_active_deadline")

    def reserve(self, counter: str, amount: int = 1, *, stage: str | None = None,
                cleanup: bool = False, allowance_seconds: float = 0.0) -> None:
        self.reserve_many({counter: amount}, stage=stage, cleanup=cleanup,
                          allowance_seconds=allowance_seconds)

    def reserve_many(self, increments: Mapping[str, int], *, stage: str | None = None,
                     cleanup: bool = False, allowance_seconds: float = 0.0) -> None:
        with self._lock:
            self.check(cleanup=cleanup, allowance_seconds=allowance_seconds)
            if not increments:
                raise AdmissionError("empty reservation")
            if stage is not None and stage not in self.stage_limits:
                raise AdmissionError(f"unknown stage: {stage}")
            for counter, amount in increments.items():
                _integer(amount, counter, 1)
                if counter not in self.limits:
                    raise AdmissionError(f"undeclared counter: {counter}")
                if self.counts[counter] + amount > self.limits[counter]:
                    raise AdmissionError(f"counter exhausted: {counter}", code=f"budget_counter_{counter}")
                # These counters always require a declared physical stage;
                # omitting one cannot spend the aggregate margin on extra work.
                if counter in {"decisions", "requested_ticks", "episodes_started"} and stage is None:
                    raise AdmissionError(f"{counter} requires its original stage")
                if stage is not None and counter in self.stage_limits[stage]:
                    if self.stage_counts[stage][counter] + amount > self.stage_limits[stage][counter]:
                        raise AdmissionError(f"stage counter exhausted: {stage}.{counter}",
                                             code=f"budget_stage_{stage}_{counter}")
            elapsed = self._clock() - self.started
            for counter, amount in increments.items():
                self.counts[counter] += amount
                if stage is not None and counter in self.stage_limits[stage]:
                    self.stage_counts[stage][counter] += amount
            record = {"index": len(self._reservations), "elapsed_seconds": elapsed,
                      "stage": stage, "cleanup": cleanup, "increments": dict(increments)}
            if self._reservations:
                self._reservation_digest.update(b",")
            self._reservation_digest.update(canonical_json(record).encode())
            self._reservations.append(record)

    @property
    def reservations(self) -> list[dict[str, Any]]:
        """An owned publication copy cannot mutate the charged hash prefix."""
        return deepcopy(self._reservations)

    def snapshot(self) -> dict[str, Any]:
        digest = self._reservation_digest.copy()
        digest.update(b"]")
        return {"schema": "nca-jolt-learn-budget/v1", "counts": dict(self.counts),
                "stage_counts": deepcopy(self.stage_counts),
                "wall_seconds": self._clock() - self.started,
                "maximum_wall_seconds": self.seconds, "cleanup_reserve_seconds": self.reserve_seconds,
                "reservation_count": len(self._reservations),
                "reservations_sha256": digest.hexdigest(),
                "closed": self._closed, "refused_or_uncertain_work_refunded": False}

    def remaining_seconds(self, *, cleanup: bool = False) -> float:
        return max(0.0, (self.deadline if cleanup else self.active_deadline) - self._clock())

    def close(self) -> dict[str, Any]:
        self.check(cleanup=True)
        self._closed = True
        return self.snapshot()


def layout_rectangles(heights: list[int] | tuple[int, ...]) -> list[list[int]]:
    if len(heights) != 5 or any(type(height) is not int or not 3 <= height <= 9 for height in heights):
        raise AdmissionError("layout requires five realized integer pile heights in [3,9]")
    return [list(rectangle) for rectangle in STATIC_RECTANGLES] + [
        [205 + index * 84, 352 - height, 7, height, 2] for index, height in enumerate(heights)]


def terrain_fingerprint(heights: list[int], anchor_x: float) -> str:
    """Translation-normalized initial composed material around an actual encounter.

    A fixed 96 by 48-cell crop ends at floor+8 and is centered on the contacted
    pile.  Rasterization follows the arena's ordered rectangle fills; hidden
    sand under an existing sand bed cannot masquerade as novel local geometry.
    Distant pile heights and the numeric seed do not enter this fingerprint.
    """
    if anchor_x not in (208.5, 292.5, 376.5, 460.5, 544.5):
        raise AdmissionError("contact anchor must identify a real host pile")
    origin_x, origin_y, width, height = math.floor(anchor_x) - 48, 312, 96, 48
    pixels = bytearray(width * height)
    for x, y, w, h, material in layout_rectangles(heights):
        for row in range(max(y, origin_y), min(y + h, origin_y + height)):
            left, right = max(x, origin_x), min(x + w, origin_x + width)
            if left < right:
                offset = (row - origin_y) * width
                pixels[offset + left - origin_x:offset + right - origin_x] = bytes([material]) * (right - left)
    return hashlib.sha256(bytes(pixels)).hexdigest()


def case_fingerprints(case: Mapping[str, Any]) -> dict[str, str]:
    heights = case["pile_heights"]
    spawn, target = case["spawn"], case["target"]
    for name, pair in (("spawn", spawn), ("target", target)):
        if not isinstance(pair, list) or len(pair) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in pair):
            raise AdmissionError(f"invalid {name}")
        if not (128 <= pair[0] < 640 and 128 <= pair[1] < 384):
            raise AdmissionError(f"{name} outside admitted arena")
    anchor = case["contact_anchor_x"]
    if not min(spawn[0], target[0]) - 8 <= anchor <= max(spawn[0], target[0]) + 8:
        raise AdmissionError("declared contact geometry is distant from the approach")
    _integer(case["host_seed"], "host_seed")
    if case["host_seed"] > 2_147_483_647:
        raise AdmissionError("host seed exceeds accepted native range")
    return {"layout_sha256": sha256_json(layout_rectangles(heights)),
            "contact_sha256": terrain_fingerprint(heights, anchor),
            "start_approach_sha256": sha256_json({"spawn": spawn, "target": target,
                                                     "initial_rotation": 0.0, "initial_velocity": [0.0, 0.0]})}


def validate_splits(split: Mapping[str, Any], opened: Mapping[str, Any]) -> dict[str, Any]:
    if split.get("schema") != "nca-jolt-static-split/v1" or opened.get("schema") != "nca-jolt-opened-inventory/v1":
        raise AdmissionError("unknown static split or opened inventory")
    if split.get("host_layout_source_sha256") != HOST_LAYOUT_SHA256:
        raise AdmissionError("split describes a different layout implementation")
    expected_counts = {"teacher": 64, "open": 4, "sealed": 16}
    ids: set[str] = set()
    footprints: dict[str, set[str]] = {stage: set() for stage in expected_counts}
    layouts: dict[str, set[str]] = {stage: set() for stage in expected_counts}
    starts: dict[str, list[list[float]]] = {stage: [] for stage in expected_counts}
    approaches: dict[str, set[str]] = {stage: set() for stage in expected_counts}
    historical_heights = set(opened["opened_layout_height_keys"])
    realizations = split["rng_realizations"]
    for stage, expected in expected_counts.items():
        cases = split.get(stage)
        if not isinstance(cases, list) or len(cases) != expected:
            raise AdmissionError(f"{stage} must contain the prospectively fixed {expected} cases")
        for case in cases:
            if not isinstance(case.get("id"), str) or case["id"] in ids:
                raise AdmissionError("duplicate or absent case identity")
            ids.add(case["id"])
            if realizations.get(str(case["host_seed"])) != case["pile_heights"]:
                raise AdmissionError("case differs from the static Godot RNG realization")
            fingerprints = case_fingerprints(case)
            if any(case.get(key) != value for key, value in fingerprints.items()):
                raise AdmissionError("static case fingerprint mismatch")
            if "".join(map(str, case["pile_heights"])) in historical_heights:
                raise AdmissionError("previously opened whole layout")
            footprints[stage].add(fingerprints["contact_sha256"])
            layouts[stage].add(fingerprints["layout_sha256"])
            starts[stage].append(case["spawn"])
            approaches[stage].add(fingerprints["start_approach_sha256"])
            for rectangle in opened["conservative_start_regions"]:
                x, y, w, h = rectangle
                if x - 8 <= case["spawn"][0] <= x + w + 8 and y - 8 <= case["spawn"][1] <= y + h + 8:
                    raise AdmissionError("start overlaps a previously opened or unlocalized historical region")
            if fingerprints["start_approach_sha256"] in opened["opened_start_approach_sha256"]:
                raise AdmissionError("previously opened start and target approach")
            if fingerprints["contact_sha256"] in opened["opened_contact_sha256"]:
                raise AdmissionError("previously opened contact geometry")
    for a, b in (("teacher", "open"), ("teacher", "sealed"), ("open", "sealed")):
        if layouts[a] & layouts[b] or footprints[a] & footprints[b] or approaches[a] & approaches[b]:
            raise AdmissionError(f"split geometry/approach overlap: {a}/{b}")
        if any(abs(x[0] - y[0]) <= 16 and abs(x[1] - y[1]) <= 16 for x in starts[a] for y in starts[b]):
            raise AdmissionError(f"split starting-region overlap: {a}/{b}")
    for stage in expected_counts:
        coverage = {case["coverage"] for case in split[stage]}
        if not {"floor_contact", "drop_landing", "edge_contact"} <= coverage:
            raise AdmissionError(f"{stage} lacks the predeclared contact/drop/edge coverage")
    return {"schema": "nca-jolt-split-admission/v1", "accepted": True,
            "case_counts": expected_counts,
            "contact_family_counts": {stage: len(values) for stage, values in footprints.items()},
            "split_sha256": sha256_json(split), "opened_inventory_sha256": sha256_json(opened),
            "physical_outcomes_inspected": False,
            "claim": "Unseen initial composed material encounters and starts in the pinned authored lab only; no arbitrary terrain or support mutation."}


def verify_prerequisite(root: str | Path, record: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(root)
    if record.get("schema") != "nca-jolt-learn-prerequisite/v1":
        raise AdmissionError("unknown teacher prerequisite")
    expected = {"physical_source_revision": TEACHER_REVISION,
                "teacher_specification_sha256": TEACHER_SPEC_SHA256,
                "teacher_declaration_sha256": TEACHER_DECLARATION_SHA256,
                "host_revision": HOST_REVISION}
    if any(record.get(key) != value for key, value in expected.items()):
        raise AdmissionError("accepted physical teacher identity was substituted")
    for relative, digest in record["source_files"].items():
        if file_sha256(_relative_file(root, relative)) != _digest(digest, relative):
            raise AdmissionError(f"preserved teacher or evaluator source changed: {relative}")
    for relative, digest in record["evidence_files"].items():
        if file_sha256(_relative_file(root, relative)) != _digest(digest, relative):
            raise AdmissionError(f"teacher evidence changed: {relative}")
    review = strict_json_loads(_relative_file(root, record["teacher_review"]).read_text())
    reconciliation = strict_json_loads(_relative_file(root, record["acceptance_reconciliation"]).read_text())
    checks = review["teacher_behavior"]["checks"]
    if not (review.get("teacher_admitted") is True and len(checks) == 13 and all(value is True for value in checks.values())
            and reconciliation.get("verified") is True
            and reconciliation["qualification"].get("all13_unchanged_gates_pass") is True
            and reconciliation["qualification"].get("complete_original_evidence") is True):
        raise AdmissionError("teacher/evidence acceptance is incomplete")
    if review.get("source_revision") != TEACHER_REVISION or review.get("declaration_sha256") != TEACHER_DECLARATION_SHA256:
        raise AdmissionError("teacher review does not bind the frozen physical source")
    # Constructor/spec inspection is static, with no observations or actuation.
    from nca.controllers.jolt_recovery_v4 import JoltRecoveryControllerV4
    from nca.controllers.recovery_supervisor import PhysicalRecoverySupervisor
    if sha256_json(JoltRecoveryControllerV4().specification()) != TEACHER_SPEC_SHA256:
        raise AdmissionError("executing teacher specification differs")
    if sha256_json(PhysicalRecoverySupervisor().specification()) != record["supervisor_specification_sha256"]:
        raise AdmissionError("executing supervisor specification differs")
    return {"accepted": True, "prerequisite_sha256": sha256_json(record),
            "teacher_only": True, "learned_skill_qualified": False, "morphology_admitted": False}


def load_config(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    path = Path(path)
    root = Path(root) if root is not None else path.resolve().parents[2]
    config = strict_json_loads(path.read_text())
    validate_config(root, config)
    return config


def validate_config(root_or_config: str | Path | Mapping[str, Any],
                    config: Mapping[str, Any] | None = None) -> None:
    root = None
    if config is None:
        config = root_or_config
    else:
        root = Path(root_or_config)
    if not isinstance(config, Mapping):
        raise AdmissionError("configuration must be a mapping")
    if config.get("schema") != SCHEMA or config.get("training_seeds") != list(POLICY_SEEDS):
        raise AdmissionError("unknown experiment version or seed selection")
    if config.get("host_revision") != HOST_REVISION or config.get("decision_ticks") != 6:
        raise AdmissionError("host or decision cadence changed")
    if config.get("architecture") != {"family": "gru", "hidden_size": 32, "continuous": 5, "buttons": 2}:
        raise AdmissionError("the fixed architecture/actuation family changed")
    budget = config["budget"]
    if (budget.get("maximum_wall_seconds") != 1800 or budget.get("maximum_attempts") != 1
            or budget.get("device") != "cpu" or budget.get("torch_threads") != 1
            or budget.get("automatic_retry") is not False or budget.get("automatic_extension") is not False):
        raise AdmissionError("single-invocation CPU allocation changed")
    counters = budget["counters"]
    if not REQUIRED_COUNTERS <= counters.keys():
        raise AdmissionError("an operation ceiling is unset")
    for name, maximum in (("decisions", 20000), ("requested_ticks", 120000), ("host_starts", 1),
                          ("bc_optimizer", 23040), ("ppo_optimizer", 24576)):
        if _integer(counters.get(name), name, 1) != maximum:
            raise AdmissionError(f"fixed aggregate ceiling differs: {name}")
    for stage, maximum in STAGE_DECISIONS.items():
        if budget["stages"][stage] != {"decisions": maximum, "requested_ticks": maximum * 6,
                                       "episodes_started": STAGE_EPISODES[stage]}:
            raise AdmissionError(f"stage allocation changed: {stage}")
    if config["bc"].get("epochs") != 40 or config["bc"].get("unroll") != 16:
        raise AdmissionError("fixed BC schedule changed")
    if any(config["ppo"].get(key) != value for key, value in
           {"updates_per_seed": 8, "decisions_per_update": 256, "epochs": 4, "unroll": 16}.items()):
        raise AdmissionError("fixed PPO schedule changed")
    required_gates = {
        "minimum_mean_success": .75, "minimum_every_seed_success": .5,
        "minimum_initial_gain": .125, "bootstrap_resamples": 27,
        "strictly_positive_lower95": True, "integrity_failures_allowed": 0,
        "failed_admissions_allowed": 0, "maximum_trim_increase_per_second": .5,
        "maximum_angle_increase_radians": .1, "minimum_peak_speed_ratio": .7,
        "maximum_peak_speed_ratio": 1.5, "maximum_airborne_fraction_loss": .125,
        "maximum_case_inference_p95_ms": 5.0,
    }
    if config.get("learned_gate") != required_gates:
        raise AdmissionError("unchanged learned acceptance thresholds were altered")
    # Validate all counters now: None, bool, negative or omitted fields may not
    # become an unlimited fallback at the first operation.
    for key, value in counters.items():
        _integer(value, key)
    BudgetLedger(budget, monotonic=lambda: 0.0, started=0.0)
    if root is not None:
        inputs = {}
        for name in ("prerequisite", "split", "opened_inventory"):
            reference = config[name]
            source = _relative_file(root, reference["path"])
            if file_sha256(source) != _digest(reference["file_sha256"], name):
                raise AdmissionError(f"{name} file identity mismatch")
            inputs[name] = strict_json_loads(source.read_text())
        verify_prerequisite(root, inputs["prerequisite"])
        for relative, digest in inputs["opened_inventory"]["source_files"].items():
            if file_sha256(_relative_file(root, relative)) != _digest(digest, relative):
                raise AdmissionError(f"historical opened inventory input changed: {relative}")
        validate_splits(inputs["split"], inputs["opened_inventory"])
        if config.get("cases") != {key: inputs["split"][key] for key in ("teacher", "open", "sealed")}:
            raise AdmissionError("execution cases differ from the frozen static split")


def frozen_finals(checkpoints: list[Mapping[str, Any]]) -> dict[str, str]:
    expected = {(family, seed) for family in ("bc", "residual") for seed in POLICY_SEEDS}
    found: dict[tuple[str, int], str] = {}
    for item in checkpoints:
        if item.get("phase") != "final":
            continue
        key = (item.get("family"), item.get("seed"))
        if key not in expected or key in found:
            raise AdmissionError("duplicate or unexpected final checkpoint")
        found[key] = _digest(item.get("sha256"), "final checkpoint")
    if set(found) != expected or len(set(found.values())) != 6:
        raise AdmissionError("all six independent learned finals must be frozen before any sealed outcome")
    return {f"{family}:{seed}": found[(family, seed)] for family, seed in sorted(found)}
