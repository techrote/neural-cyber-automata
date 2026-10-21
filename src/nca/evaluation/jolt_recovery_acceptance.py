"""Prospective authored-recovery acceptance; no new physical or learned run.

Complete evidence and useful behavior are independent requirements. The fixed
development cases cannot establish learned reliability or morphology transfer.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
from pathlib import Path

from nca.contracts import ContractError, sha256_json, strict_json_loads


REFERENCE = "jolt-arrival-v1"
CANDIDATE = "jolt-recovery-v1"
CANDIDATE_V2 = "jolt-recovery-v2"
CANDIDATE_V3 = "jolt-recovery-v3"
CANDIDATE_V4 = "jolt-recovery-v4"
_CANDIDATE_VERSIONS = {CANDIDATE: "v1", CANDIDATE_V2: "v2", CANDIDATE_V3: "v3", CANDIDATE_V4: "v4"}
HISTORICAL_FAILURES = (
    "demo-floor-left-60-1", "demo-floor-left-60-2",
    "reference-floor-left-63", "reference-drop28-left-57",
    "reference-drop12-right-45",
)
UNSUPPORTED_FAILURE = "reference-drop12-right-45"
HISTORICAL_MISSING_AIRBORNE = tuple(
    f"demo-floor-left-{distance}-{index}"
    for distance in (24, 36) for index in range(3)
) + ("reference-floor-left-27",)
RECOVERY_REASONS = {
    "supported_blocked_progress", "supported_no_progress",
    "unsupported_edge_stall", "unexpected_support_loss_descent",
}


def acceptance_specification(candidate_family=CANDIDATE):
    _require(candidate_family in _CANDIDATE_VERSIONS, "unknown recovery candidate version")
    return {
        "schema": "nca-jolt-recovery-acceptance/" + _CANDIDATE_VERSIONS[candidate_family],
        "reference": REFERENCE, "candidate": candidate_family,
        "paired_cases": 32,
        "minimum_extra_arrivals": 1,
        "minimum_historical_rescues": 3,
        "historical_failures": list(HISTORICAL_FAILURES),
        "required_unsupported_rescue": UNSUPPORTED_FAILURE,
        "require_supported_failure_rescue": True,
        "maximum_regressions": 1,
        "historical_missing_airborne": list(HISTORICAL_MISSING_AIRBORNE),
        "minimum_restored_airborne_cases": 1,
        "minimum_overall_purposeful_launch_change": 0,
        "maximum_trim_reversal_increase_per_second": .5,
        "maximum_absolute_angle_increase_radians": .1,
        "minimum_toward_speed_ratio": .7,
        "maximum_toward_speed_ratio": 1.5,
        "every_case_inference_p95_maximum_ms": 5.,
        "maximum_requested_primary_edges": 2,
        "second_primary_requires_fresh_physical_reason": True,
        "aggregation": "equal weight over32 already-open paired development cases",
        "learned_qualification": False,
    }


def _require(condition, detail):
    if not condition:
        raise ContractError("jolt_recovery_acceptance", detail)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value),
             "finite numeric character measurements are required")
    return value


def assess_teacher(rows, *, pulse_authority_verified, candidate_family=CANDIDATE):
    """Compute the published rule from complete receipt-verified result rows."""
    _require(candidate_family in _CANDIDATE_VERSIONS, "unknown recovery candidate version")
    paired = {}
    for family in (REFERENCE, candidate_family):
        selected = [row for row in rows if row["family"] == family]
        mapped = {row["case_id"]: row for row in selected}
        _require(len(selected) == len(mapped) == 32, "all32 distinct paired cases are required")
        for row in selected:
            metrics, character = row["metrics"], row["character_metrics"]
            _require(row.get("error") is None and row.get("owner_closure_confirmed") is True
                     and metrics.get("administratively_capped") is False
                     and metrics.get("integrity_failure") is False
                     and metrics.get("failed_admissions") == 0,
                     "incomplete or invalid episodes cannot qualify a teacher")
            _require(type(metrics.get("success")) is bool
                     and type(metrics.get("purposeful_launch")) is bool,
                     "task and purposeful-launch outcomes must be measured Booleans")
            _require(type(character.get("requested_primary_edges")) is int
                     and character["requested_primary_edges"] >= 0,
                     "requested primary edges must be measured integers")
            for field in ("ordinary_trim_reversals_per_second", "mean_absolute_angle_radians",
                          "peak_toward_speed_cells_per_second", "inference_p95_ms"):
                _number(metrics[field])
        paired[family] = mapped
    reference, candidate = paired[REFERENCE], paired[candidate_family]
    _require(set(reference) == set(candidate)
             and set(HISTORICAL_FAILURES + HISTORICAL_MISSING_AIRBORNE) <= set(reference),
             "the paired cases must retain all historical failures and character cases")
    successes = lambda selected: {case for case, row in selected.items() if row["metrics"]["success"]}
    reference_success, candidate_success = successes(reference), successes(candidate)
    rescued = sorted((candidate_success - reference_success) & set(HISTORICAL_FAILURES))
    regressions = sorted(reference_success - candidate_success)
    purposeful = lambda selected: {case for case, row in selected.items() if row["metrics"]["purposeful_launch"]}
    reference_air, candidate_air = purposeful(reference), purposeful(candidate)
    restored = sorted((candidate_air - reference_air) & set(HISTORICAL_MISSING_AIRBORNE))
    def mean(selected, key):
        return sum(row["metrics"][key] for row in selected.values()) / 32
    measurements = {}
    for family, selected in paired.items():
        measurements[family] = {
            "arrivals": len(successes(selected)), "cases": 32,
            "purposeful_airborne_cases": len(purposeful(selected)),
            "mean_ordinary_trim_reversals_per_second": mean(selected, "ordinary_trim_reversals_per_second"),
            "mean_absolute_angle_radians": mean(selected, "mean_absolute_angle_radians"),
            "mean_peak_toward_speed_cells_per_second": mean(selected, "peak_toward_speed_cells_per_second"),
            "maximum_case_inference_p95_ms": max(row["metrics"]["inference_p95_ms"] for row in selected.values()),
            "maximum_requested_primary_edges": max(row["character_metrics"]["requested_primary_edges"] for row in selected.values()),
        }
    before, after = measurements[REFERENCE], measurements[candidate_family]
    speed = before["mean_peak_toward_speed_cells_per_second"]
    ratio = None if speed <= 0 else after["mean_peak_toward_speed_cells_per_second"] / speed
    checks = {
        "strict_arrival_improvement": len(candidate_success) >= len(reference_success) + 1,
        "at_least_three_historical_rescues": len(rescued) >= 3,
        "unsupported_drop_rescued": UNSUPPORTED_FAILURE in rescued,
        "supported_failure_rescued": any(case != UNSUPPORTED_FAILURE for case in rescued),
        "at_most_one_regression": len(regressions) <= 1,
        "restore_airborne_character": len(restored) >= 1,
        "retain_overall_airborne_character": len(candidate_air) >= len(reference_air),
        "ordinary_trim_limit": after["mean_ordinary_trim_reversals_per_second"] <= before["mean_ordinary_trim_reversals_per_second"] + .5,
        "angle_limit": after["mean_absolute_angle_radians"] <= before["mean_absolute_angle_radians"] + .1,
        "vigor_limit": ratio is not None and .7 <= ratio <= 1.5,
        "every_case_latency_limit": after["maximum_case_inference_p95_ms"] <= 5.,
        "at_most_two_requested_primary_edges": after["maximum_requested_primary_edges"] <= 2,
        "fresh_physical_pulse_authority": pulse_authority_verified is True,
    }
    return {
        "schema": "nca-jolt-recovery-behavior-review/" + _CANDIDATE_VERSIONS[candidate_family],
        "teacher_accepted": all(checks.values()), "checks": checks,
        "measurements": measurements, "toward_speed_ratio": ratio,
        "historical_rescues": rescued, "regressions": regressions,
        "restored_airborne_cases": restored,
        "improved_cases": sorted(candidate_success - reference_success),
        "remaining_failures": sorted(set(candidate) - candidate_success),
        "learned_skill_qualified": False, "morphology_admitted": False,
        "owner_feel_accepted": False,
    }


def _candidate_controller(candidate_family):
    if candidate_family == CANDIDATE:
        from nca.controllers.jolt_recovery import JoltRecoveryController
        return JoltRecoveryController()
    if candidate_family == CANDIDATE_V2:
        from nca.controllers.jolt_recovery_v2 import JoltRecoveryControllerV2
        return JoltRecoveryControllerV2()
    if candidate_family == CANDIDATE_V3:
        from nca.controllers.jolt_recovery_v3 import JoltRecoveryControllerV3
        return JoltRecoveryControllerV3()
    if candidate_family == CANDIDATE_V4:
        from nca.controllers.jolt_recovery_v4 import JoltRecoveryControllerV4
        return JoltRecoveryControllerV4()
    raise ContractError("jolt_recovery_acceptance", "unknown recovery candidate version")


def verify_controller_replay(output, rows, *, candidate_family=CANDIDATE):
    """Replay the frozen authored policy on retained local observations only.

    This issues no owner request and consumes no new physics. It independently
    checks the frame/telemetry against the observed source and verifies pulse
    count, readiness and declared reasons without trusting a phase label.
    """
    from nca.controllers.baseline import FeedbackObservation, SkillGoal
    from nca.data import read_episode

    decisions = pulses = second_pulses = 0
    for row in rows:
        if row["family"] != candidate_family:
            continue
        episode = read_episode(output / row["artifact"])
        events = [event for event in episode.events if event.kind == "step"]
        frames = episode.evaluation["controller_frames"]
        _require(len(events) == len(frames), "every control decision needs a retained frame")
        context = events[0].context
        goal = SkillGoal(context.goal_id, context.skill, context.target,
                         context.pace, context.tolerance, context.timeout_seconds)
        controller = _candidate_controller(candidate_family)
        previous, count = False, 0
        for event, frame in zip(events, frames):
            observation = FeedbackObservation.from_host(event.source_observation)
            replay = controller.step(observation, goal)
            _require(sha256_json(replay.action.to_dict()) == sha256_json(frame["action"])
                     and sha256_json(controller.diagnostic_state()) == sha256_json(frame["diagnostic_state"])
                     and (replay.phase, replay.status, replay.reason) ==
                         (frame["phase"], frame["status"], frame["reason"]),
                     "recorded request or recovery reason differs from frozen observed-source replay")
            decisions += 1
            active = replay.action.primary
            if active and not previous:
                count += 1
                pulses += 1
                _require(observation.cooldown is not None and observation.cooldown <= 0
                         and observation.energy is not None and observation.energy >= .15
                         and frame["diagnostic_state"].get("tick") == observation.tick,
                         "a pulse needs fresh observed energy and cooldown readiness")
                if count > 1:
                    second_pulses += 1
                    _require(frame["diagnostic_state"].get("pulse_reason") in RECOVERY_REASONS,
                             "a second pulse needs a declared physical recovery reason")
            previous = active
        _require(count <= 2, "the invocation exceeded its two-pulse authority")
        controller.cancel()
    return {"verified": True, "authored_decisions_replayed": decisions,
            "requested_pulses": pulses, "second_pulses": second_pulses,
            "new_owner_requests": 0, "new_simulation_ticks": 0, "optimizer_updates": 0}


def verify_replay_source(manifest, *, expected_revision: str, source_root: Path | None = None):
    """Require the replay's imported package and Git blobs to match the run."""
    from nca.data.capture import source_identity

    package = Path(__file__).resolve().parents[1]
    source_tree = package.parents[1]
    selected = source_root if source_root is not None else os.environ.get("NCA_SOURCE_ROOT")
    if selected is None:
        selected = source_tree if (source_tree / "src/nca").resolve() == package else Path.cwd()
    root = Path(selected).resolve()
    revision, imported, _ = source_identity(root)
    frozen = manifest.get("source_files")
    _require(revision == expected_revision and isinstance(frozen, dict)
             and sha256_json(frozen) == manifest.get("source_files_sha256"),
             "replay requires the published revision and intact source identity")
    package = {path: digest for path, digest in frozen.items()
               if path.startswith("src/nca/") or path in ("pyproject.toml", "tools/capture_replay.py")}
    _require(imported == package, "the imported replay package differs from the frozen physical run")
    for relative, digest in frozen.items():
        _require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                 "source identity must contain only repository-relative paths")
        committed = subprocess.check_output(["git", "show", f"{revision}:{relative}"],
                                            cwd=root, stderr=subprocess.DEVNULL)
        _require(hashlib.sha256(committed).hexdigest() == digest
                 and hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest,
                 "replay source or declared input differs from its published Git blob")
    candidate_family = manifest["declaration"]["families"][-1]
    _require(manifest["declaration"]["candidate_specifications"].get(candidate_family)
             == _candidate_controller(candidate_family).specification(),
             "replay controller specification differs from the prospectively frozen candidate")
    return {"verified": True, "source_revision": revision,
            "source_files_sha256": manifest["source_files_sha256"]}


def audit_recovery(output: Path, *, expected_revision: str, expected_declaration_sha256: str,
                   source_root: Path | None = None):
    from nca.evaluation.jolt_arrival import verify_diagnostic_output
    verified = verify_diagnostic_output(output, expected_revision=expected_revision,
                                       expected_declaration_sha256=expected_declaration_sha256)
    manifest = strict_json_loads((output / "manifest.json").read_bytes())
    replay_source = verify_replay_source(manifest, expected_revision=expected_revision, source_root=source_root)
    summary = strict_json_loads((output / "summary.json").read_bytes(), max_bytes=8 * 1024 * 1024)
    candidate_family = manifest["declaration"]["families"][-1]
    _require(manifest["declaration"].get("teacher_acceptance") == acceptance_specification(candidate_family),
             "the prospective teacher acceptance rule cannot be changed after the run")
    supervisor_path = output.with_name(output.name + "-supervisor.json")
    supervisor = strict_json_loads(supervisor_path.read_bytes())
    _require(supervisor.get("completed") is True and supervisor.get("source_revision") == expected_revision
             and supervisor.get("declaration_sha256") == expected_declaration_sha256
             and supervisor.get("attempts") == 1 and supervisor.get("timeout") is False
             and supervisor.get("worker_terminated") is True
             and supervisor.get("owned_process_group_terminated") is True
             and supervisor.get("final_output_verification") == verified
             and _number(supervisor.get("wall_seconds")) <= manifest["declaration"]["budget"]["maximum_wall_seconds"],
             "the full invocation and independent final verification must close within the declared budget")
    finalization_sha256 = executor_sha256 = None
    if candidate_family == CANDIDATE_V4:
        _require(supervisor.get("whole_process_started_monotonic") == manifest.get("whole_process_started_monotonic")
                 == summary["counts"].get("whole_process_started_monotonic")
                 and supervisor.get("cleanup_reserve_seconds") == 120
                 and supervisor.get("independent_verification_reserve_seconds") == 60
                 and supervisor.get("supervisor_finalization_reserve_seconds") == 6,
                 "v4 source/evidence must share the original outer clock and prospectively divided reserve")
        finalization_path = output.with_name(output.name + "-finalization.json")
        finalization = strict_json_loads(finalization_path.read_bytes())
        _require(finalization.get("schema") == "nca-jolt-arrival-finalization/v1"
                 and finalization.get("source_revision") == expected_revision
                 and finalization.get("declaration_sha256") == expected_declaration_sha256
                 and finalization.get("supervisor_sha256") == hashlib.sha256(supervisor_path.read_bytes()).hexdigest()
                 and finalization.get("whole_process_started_monotonic") == manifest["whole_process_started_monotonic"]
                 and finalization.get("completed") is True
                 and supervisor["wall_seconds"] <= _number(finalization.get("wall_seconds")) <= 900,
                 "v4 requires the unchanged original post-receipt finalization within its whole-process ceiling")
        finalization_sha256 = hashlib.sha256(finalization_path.read_bytes()).hexdigest()
        executor_path = output.with_name(output.name + "-executor.json")
        executor = strict_json_loads(executor_path.read_bytes())
        terminal = executor.get("terminal_report")
        _require(executor.get("schema") == "nca-jolt-arrival-executor/v1"
                 and executor.get("source_revision") == expected_revision
                 and executor.get("declaration_sha256") == expected_declaration_sha256
                 and executor.get("executor_source_sha256") == manifest["source_files"].get("tools/execute_jolt_diagnostic.py")
                 and type(executor.get("executor_source_sha256")) is str
                 and len(executor["executor_source_sha256"]) == 64
                 and type(executor.get("maximum_wall_seconds")) is int and executor["maximum_wall_seconds"] == 900
                 and type(executor.get("attempts")) is int and executor["attempts"] == 1
                 and type(executor.get("returncode")) is int and executor["returncode"] == 0
                 and executor.get("timeout") is False and executor.get("completed") is True
                 and executor.get("supervisor_terminated") is True
                 and executor.get("owned_process_tree_terminated") is True
                 and executor.get("error_type") is None
                 and executor.get("whole_process_started_monotonic") == manifest["whole_process_started_monotonic"]
                 and executor.get("supervisor_sha256") == hashlib.sha256(supervisor_path.read_bytes()).hexdigest()
                 and executor.get("finalization_sha256") == finalization_sha256
                 and _number(executor.get("process_exited_monotonic")) - executor["whole_process_started_monotonic"]
                     == _number(executor.get("wall_seconds"))
                 and isinstance(terminal, dict) and terminal.get("completed") is True
                 and type(terminal.get("returncode")) is int and terminal["returncode"] == 0
                 and terminal.get("timeout") is False and terminal.get("finalization_sha256") == finalization_sha256
                 and finalization["wall_seconds"] <= _number(terminal.get("wall_seconds"))
                     <= executor["wall_seconds"] <= 900,
                 "v4 requires the unchanged original external executor through child exit within its whole-process ceiling")
        stream_paths = [output.with_name(output.name + suffix)
                        for suffix in ("-executor-stdout.log", "-executor-stderr.log")]
        streams = executor.get("streams")
        _require(isinstance(streams, dict) and set(streams) == {path.name for path in stream_paths},
                 "v4 external executor must retain its exact original stdout/stderr inventory")
        for path in stream_paths:
            metadata = streams[path.name]
            _require(path.is_file() and not path.is_symlink() and isinstance(metadata, dict),
                     "v4 external executor streams must be retained regular files")
            payload = path.read_bytes()
            _require(type(metadata.get("bytes")) is int and metadata["bytes"] == len(payload)
                     and metadata.get("sha256") == hashlib.sha256(payload).hexdigest(),
                     "v4 external executor stream bytes differ from the unchanged original receipt")
            if path == stream_paths[0]:
                lines = payload.splitlines()
                _require(bool(lines) and strict_json_loads(lines[-1]) == terminal,
                         "v4 external executor terminal report differs from original CLI stdout")
        executor_sha256 = hashlib.sha256(executor_path.read_bytes()).hexdigest()
    pulse_audit = verify_controller_replay(output, summary["rows"], candidate_family=candidate_family)
    result = assess_teacher(summary["rows"], pulse_authority_verified=pulse_audit["verified"],
                            candidate_family=candidate_family)
    result.update(source_revision=expected_revision, declaration_sha256=expected_declaration_sha256,
                  evidence_verification=verified, pulse_authority=pulse_audit,
                  replay_source=replay_source,
                  supervisor_sha256=hashlib.sha256(supervisor_path.read_bytes()).hexdigest())
    if finalization_sha256 is not None:
        result["finalization_sha256"] = finalization_sha256
        result["executor_sha256"] = executor_sha256
    return result
