"""Bounded public-host profile conformance, with no learned candidate execution.

The action schedule is an explicit diagnostic fixture. It does not optimize a
policy, score task success or open the failed morphology-transfer prerequisite.
Every attempted game request is charged before its potentially ambiguous RPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

from nca.contracts import ContractError, canonical_json, sha256_json
from .admission import _sources, comparison_plan, entry_decision, read_json
from .core import (MorphologyAdapter, MorphologyGoal, SemanticAction, StateFence,
                   information_reach, map_action)

RECOVERY_SHA256 = "4d47e4ac6f179da58bc0ceb909cc9f4f9060042fc7f5c207e1e8f99c6cc171d9"


def _require(value: bool, code: str, detail: str) -> None:
    if not value:
        raise ContractError(code, detail)


def _write(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    data = (canonical_json(value) + "\n").encode()
    if path.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return {"artifact": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


@dataclass
class ConformanceBudget:
    """A failed or uncertain request still consumes its finite allocation."""
    maximum_decisions: int = 32
    maximum_wall_seconds: float = 60.
    started: float = 0.
    attempted_decisions: int = 0
    completed_decisions: int = 0
    completed_ticks: int = 0

    def __post_init__(self):
        if (type(self.maximum_decisions) is not int or not 1 <= self.maximum_decisions <= 32
                or type(self.maximum_wall_seconds) not in (int, float)
                or not math.isfinite(self.maximum_wall_seconds) or not 0 < self.maximum_wall_seconds <= 60):
            raise ContractError("morphology_conformance_budget", "at most 32 decisions and 60 seconds")
        self.started = time.monotonic()

    def check(self) -> None:
        _require(time.monotonic() - self.started < self.maximum_wall_seconds,
                 "morphology_conformance_wall", "fixed wall ceiling reached; no further game work")

    def reserve(self) -> None:
        self.check()
        _require(self.attempted_decisions < self.maximum_decisions,
                 "morphology_conformance_decisions", "fixed decision ceiling reached")
        self.attempted_decisions += 1

    def receipt(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started
        return {"attempted_decisions": self.attempted_decisions,
                "completed_decisions": self.completed_decisions, "completed_ticks": self.completed_ticks,
                "maximum_decisions": self.maximum_decisions, "maximum_wall_seconds": self.maximum_wall_seconds,
                "wall_seconds": elapsed, "wall_ceiling_respected": elapsed <= self.maximum_wall_seconds,
                "unconfirmed_decisions": self.attempted_decisions - self.completed_decisions,
                "learned_policy_calls": 0, "optimizer_updates": 0, "transfer_evaluations": 0}


def _same_action(requested: Mapping[str, Any], consumed: Any) -> bool:
    if not isinstance(consumed, Mapping) or set(requested) != set(consumed):
        return False
    for name, expected in requested.items():
        actual = consumed[name]
        if type(expected) is bool:
            if actual is not expected:
                return False
        elif isinstance(expected, (list, tuple)):
            if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
                return False
            if any(type(value) not in (int, float) or not math.isfinite(value)
                   or abs(value - target) > 1e-6 for target, value in zip(expected, actual)):
                return False
        elif type(actual) not in (int, float) or not math.isfinite(actual) or abs(actual - expected) > 1e-6:
            return False
    return True


def validate_interval(packet: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any],
                      terminated: bool, truncated: bool, info: Mapping[str, Any], *,
                      allow_terminal: bool = False) -> None:
    """Require fresh admission and every available original-input receipt.

    The scripted diagnostic requires six continuing ticks. The unqualified
    runtime may also consume a native terminal result; its last action readback
    is explicitly unavailable after owner revocation, never inferred as applied.
    """
    _require(type(terminated) is bool and type(truncated) is bool,
             "morphology_conformance_terminal", "explicit terminal flags required")
    terminal = terminated or truncated
    _require((allow_terminal and terminal and after.get("suspended") is True)
             or (not terminal and not after.get("suspended")),
             "morphology_conformance_terminal", "diagnostic cap must leave a valid continuing observation")
    actual = info.get("actual_ticks")
    _require(type(actual) is int and (1 <= actual <= 6 if terminal else actual == 6)
             and after.get("tick") == before["tick"] + actual
             and info.get("elapsed_ticks") == actual and info.get("delay_ticks_before_repeat") == 0,
             "morphology_conformance_cadence", "six synchronous completed ticks required")
    trace = info.get("trace")
    _require(isinstance(trace, list) and len(trace) == actual
             and info.get("consumed_trace_complete") is True and not info.get("idle_trace")
             and not info.get("unconsumed_terminal_commands"),
             "morphology_conformance_trace", "complete, unmixed per-tick receipts required")
    for offset, row in enumerate(trace):
        admissions = row.get("admission", [])
        requests = row.get("requested", [])
        _require(row.get("tick_before") == before["tick"] + offset
                 and row.get("tick_after") == before["tick"] + offset + 1
                 and row.get("failed") is False,
                 "morphology_conformance_trace", "causal valid tick receipt required")
        if offset == 0:
            _require(len(admissions) == 1 and admissions[0].get("accepted") is True
                     and len(requests) == 1
                     and all(requests[0].get(key) == packet[key] for key in
                             ("run_id", "epoch", "actor_id", "sequence", "observation_id", "observed_tick", "lease_ticks"))
                     and _same_action(packet["action"], requests[0].get("action")),
                     "morphology_conformance_admission", "fresh admission must name the original issued packet")
        else:
            _require(not admissions and not requests,
                     "morphology_conformance_mixed", "no replacement packet inside a held repeat")
        applied = row.get("applied", {})
        if terminal and offset == actual - 1:
            _require(applied.get("action_available") is False and applied.get("terminal_revocation") is True
                     and applied.get("action") is None,
                     "morphology_conformance_consumption", "native terminal readback must remain explicitly unavailable")
            continue
        _require(applied.get("action_available") is True and applied.get("terminal_revocation") is False
                 and _same_action(packet["action"], applied.get("action")),
                 "morphology_conformance_consumption", "actual held inputs differ from the diagnostic request")


def cancellation_receipt(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Cancellation revokes authority at the same physical state, without stepping."""
    # ACT004 cancellation clears requested motor targets while preserving every
    # measured joint value. A revoked target is not a changed physical state.
    original, cancelled = before.get("joints"), after.get("joints")
    _require(isinstance(original, (list, tuple)) and isinstance(cancelled, (list, tuple))
             and len(original) == len(cancelled)
             and all(isinstance(joint, Mapping) for joint in (*original, *cancelled)),
             "morphology_conformance_cancel", "complete original and cancelled joint records required")
    targets_before = [joint.get("target") for joint in original]
    targets_after = [joint.get("target") for joint in cancelled]
    _require(all(type(value) in (int, float) and math.isfinite(value)
                 for value in (*targets_before, *targets_after)) and all(value == 0 for value in targets_after),
             "morphology_conformance_cancel", "owner cancellation must clear each requested motor target")
    joint_physics = all({key: value for key, value in left.items() if key != "target"}
                       == {key: value for key, value in right.items() if key != "target"}
                       for left, right in zip(original, cancelled))
    keys = ("position", "velocity", "rotation", "angular_velocity", "parts")
    unchanged = joint_physics and all(before.get(key) == after.get(key) for key in keys)
    identity_keys = ("run_id", "epoch", "actor_id", "model", "profile_version", "tick")
    same_incarnation = all(before.get(key) == after.get(key) for key in identity_keys)
    revoked = (type(after.get("control_generation")) is int
               and after["control_generation"] == before["control_generation"] + 1)
    stamped = after.get("observation_id") == ":".join(str(after.get(key)) for key in
                        ("run_id", "epoch", "actor_id", "control_generation", "tick"))
    _require(unchanged and same_incarnation and revoked and stamped,
             "morphology_conformance_cancel", "owner cancellation must preserve tick and physical state")
    return {"same_tick_pose_velocity_parts_and_joint_physics": unchanged,
            "joint_physical_state_comparison_excludes": ["target"],
            "joint_targets_before": targets_before, "joint_targets_after": targets_after,
            "joint_targets_revoked": True,
            "control_generation_before": before["control_generation"],
            "control_generation_after": after["control_generation"], "neutral_step_used": False,
            "observation_before": before, "observation_after": after}


def _runtime_sources(root: Path) -> dict[str, Any]:
    """Include the loaded package closure and reject an editable-install mismatch."""
    record = _sources(root)
    files = record["files"]
    for name, module in tuple(sys.modules.items()):
        if name == "nca" or name.startswith("nca."):
            imported = getattr(module, "__file__", None)
            if not imported or not imported.endswith(".py"):
                continue
            path = "src/" + name.replace(".", "/") + ("/__init__.py" if Path(imported).name == "__init__.py" else ".py")
            data = (root / path).read_bytes()
            _require(data == Path(imported).read_bytes(), "morphology_source_identity",
                     "the provided source checkout differs from an imported dependency")
            files[path] = hashlib.sha256(data).hexdigest()
    for name in ("src/nca/adapters/host_bridge.gd", "evals/morphology/conformance.py"):
        files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    record["files"] = dict(sorted(files.items()))
    record["sources_sha256"] = sha256_json(record["files"])
    return record


def _error(exc: BaseException) -> dict[str, str]:
    # Local diagnostic paths are not copied into portable public evidence.
    result = {"type": type(exc).__name__, "code": exc.code if isinstance(exc, ContractError) else "conformance_error",
              "detail": str(exc) if isinstance(exc, ContractError) else "fixed diagnostic did not complete"}
    if exc.__cause__ is not None:
        result["cause_type"] = type(exc.__cause__).__name__
    if type(exc).__name__ == "HostError":
        message = str(exc)
        if "persistent host did not start" in message:
            result["detail"] = "local host startup or capability handshake failed"
        elif "cannot establish the pinned host/runtime identity" in message:
            result["detail"] = "pinned host/runtime identity verification failed"
        if "SCRIPT ERROR" in message or "Failed to load script" in message:
            result["diagnostic_kind"] = "host_script_load_error"
        elif "host capability handshake mismatch" in message:
            result["diagnostic_kind"] = "capability_handshake_mismatch"
    return result


def recovery_plan(declaration: Mapping[str, Any], previous: Mapping[str, Any], *,
                  previous_file_sha256: str) -> dict[str, Any]:
    """One predeclared startup continuation, charged to the original allocation."""
    _require(sha256_json(declaration) == RECOVERY_SHA256,
             "morphology_recovery_changed", "the linked startup recovery declaration differs")
    content = dict(previous)
    digest = content.pop("sha256", None)
    _require(digest == declaration["previous_summary_sha256"] and sha256_json(content) == digest
             and previous_file_sha256 == declaration["previous_summary_file_sha256"],
             "morphology_recovery_evidence", "the complete immutable failed attempt is required")
    budget = previous["budget"]
    _require(previous["status"] == "failed" and not previous["episodes"]
             and previous["sources"]["revision"] == declaration["previous_source_revision"]
             and previous["host_start_attempts"] == 1
             and previous["config_sha256"] == declaration["comparison_sha256"]
             and budget["attempted_decisions"] == budget["completed_decisions"] == budget["completed_ticks"] == 0
             and budget["learned_policy_calls"] == budget["optimizer_updates"] == budget["transfer_evaluations"] == 0
             and budget["wall_seconds"] == declaration["previous_wall_seconds"],
             "morphology_recovery_scope", "only the declared zero-request startup failure may continue")
    _require(budget["wall_seconds"] + declaration["maximum_remaining_wall_seconds"]
             <= declaration["maximum_aggregate_wall_seconds"],
             "morphology_recovery_budget", "earlier time remains charged to the original ceiling")
    return {"declaration_sha256": RECOVERY_SHA256,
            "previous_summary_sha256": digest, "previous_summary_file_sha256": previous_file_sha256,
            "previous_wall_seconds": budget["wall_seconds"], "previous_host_start_attempts": 1,
            "remaining_wall_seconds": declaration["maximum_remaining_wall_seconds"],
            "startup_transport_timeout_seconds": declaration["startup_transport_timeout_seconds"],
            "maximum_aggregate_wall_seconds": declaration["maximum_aggregate_wall_seconds"],
            "maximum_aggregate_game_decisions": declaration["maximum_aggregate_game_decisions"],
            "maximum_aggregate_completed_ticks": declaration["maximum_aggregate_completed_ticks"],
            "automatic_retry": False}


def run_conformance(*, root: Path, checkout: Path, godot: Path, config_path: Path,
                    profiles_path: Path, declaration_path: Path, result_path: Path,
                    output: Path, recovery_path: Path | None = None,
                    previous_summary_path: Path | None = None) -> dict[str, Any]:
    """Run exactly the declared scripted interface cases in one accepted host."""
    output.mkdir(parents=True, exist_ok=False)
    budget = ConformanceBudget()
    report: dict[str, Any] = {
        "schema": "nca-morphology-conformance/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed", "evidence_class": "actual-game-interface-conformance",
        "learned_transfer_measured": False, "owner_feel_accepted": False, "task_success_scored": False,
        "python": platform.python_version(), "platform": platform.system(), "architecture": platform.machine(),
        "episodes": [], "errors": [], "host_start_attempts": 0,
        "missing_qualification": ["learned zero-shot transfer", "learned fixed-budget adaptation",
                                  "independent geometry sweep", "arbitrary damage/mass/assist perturbations",
                                  "Windows runtime", "owner feel judgment"],
    }
    source_receipt = None
    continuation = None
    try:
        root = root.resolve()
        startup_timeout = 10.
        if recovery_path is not None or previous_summary_path is not None:
            _require(recovery_path is not None and previous_summary_path is not None,
                     "morphology_recovery_evidence", "both the frozen declaration and previous summary are required")
            continuation = recovery_plan(read_json(recovery_path), read_json(previous_summary_path),
                previous_file_sha256=hashlib.sha256(previous_summary_path.read_bytes()).hexdigest())
            report["continuation"] = continuation
            budget.maximum_wall_seconds = continuation["remaining_wall_seconds"]
            startup_timeout = continuation["startup_transport_timeout_seconds"]
            budget.check()
        config, profiles = read_json(config_path), read_json(profiles_path)
        comparison_plan(config, profiles)
        report["prerequisite"] = entry_decision(read_json(declaration_path), read_json(result_path))
        declared = config["conformance_only"]
        _require(declared["learned_policy_calls"] == 0 and declared["optimizer_updates"] == 0
                 and declared["task_success_scoring"] is False,
                 "morphology_conformance_scope", "conformance cannot become candidate training or task evaluation")
        # These imports permit physical interface calls but never load Torch or a candidate.
        from nca.adapters.godot import CybersandEnv, GodotHost, physical_config
        _require("nca.controllers.morphology.models" not in sys.modules and "torch" not in sys.modules,
                 "morphology_conformance_scope", "run the interface diagnostic in a fresh process without candidate imports")
        source_receipt = _runtime_sources(root)
        report["sources"] = source_receipt
        report["config_sha256"] = sha256_json(config)
        report["profile_fixture_sha256"] = sha256_json(profiles)
        report["declaration"] = declared
        manifest = {"schema": "nca-morphology-conformance-prepared/v1", "created_utc": report["created_utc"],
                    "sources": source_receipt, "config_sha256": report["config_sha256"],
                    "profile_fixture_sha256": report["profile_fixture_sha256"], "declaration": declared,
                    "prerequisite": report["prerequisite"], "no_model_or_trainer_imports": True,
                    "continuation": continuation,
                    "maximum_attempt_wall_seconds": budget.maximum_wall_seconds,
                    "startup_transport_timeout_seconds": startup_timeout}
        manifest["sha256"] = sha256_json(manifest)
        report["prepared_manifest"] = _write(output / "prepared.json", manifest)
        budget.check()
        report["host_start_attempts"] = 1
        with GodotHost(checkout, godot, lane="act004", mode="stepped", timeout=startup_timeout) as host:
            report["host_identity"] = host.identity
            _require(host.identity["host_revision"] == config["host"]["revision"]
                     and host.identity["bridge_sha256"] == config["host"]["bridge_sha256"],
                     "morphology_conformance_host", "accepted host and frozen bridge required")
            for model in declared["profiles"]:
                fence, previous_identity = StateFence(), None
                env = CybersandEnv(host, physical_config(model, seed=declared["seed"],
                                                       ticks=declared["host_episode_lifetime_ticks"]))
                for index in range(declared["episodes_per_profile"]):
                    budget.check()
                    episode: dict[str, Any] = {"id": f"{model}-{index}", "model": model, "episode": index,
                                               "status": "failed", "decisions": [], "errors": [],
                                               "scripted_diagnostic": True, "learned_policy_calls": 0}
                    view = None
                    try:
                        view, initial_info = env.reset(seed=declared["seed"])
                        profile = host.request("profile")["profile"]
                        expected = profiles["profiles"][model]["profile_sha256"]
                        adapter = MorphologyAdapter(profile, expected_sha256=expected)
                        episode.update(profile=profile, profile_sha256=expected, resolved_host_config=env.config,
                                       initial_observation=view, initial_info=initial_info)
                        offset = declared["goal_direction_by_episode"][index]
                        position = view["parts"][0]["position"]
                        goal = MorphologyGoal(f"conformance-{model}-{index}", (position[0] + offset, position[1]))
                        episode["explicit_diagnostic_context"] = goal.to_dict()
                        previous, reset_checked = SemanticAction(), False
                        for step_index, action in enumerate(declared["fixed_semantic_requests"]):
                            identity = adapter.identity(view)
                            resets = fence.accept(identity, goal)
                            if previous_identity is not None and not reset_checked:
                                _require("actor_lifetime" in resets and identity.tick == 0,
                                         "morphology_conformance_reset", "host reset must reset the recurrence clock")
                                try:
                                    fence.accept(previous_identity, goal)
                                except ContractError as exc:
                                    _require(exc.code == "retired_morphology", "morphology_conformance_reset",
                                             "old host incarnation must be refused explicitly")
                                    episode["old_incarnation_refused_by_adapter"] = exc.code
                                else:
                                    raise ContractError("morphology_conformance_reset", "old host incarnation resumed")
                                reset_checked = True
                            frame = adapter.frame(view, goal, previous=previous, started_tick=fence.start_tick)
                            request = SemanticAction.from_dict(action)
                            wire, masked = map_action(adapter, request, view, goal)
                            row = {"index": step_index, "source": identity.to_dict(),
                                   "source_sha256": adapter.source_sha256(view), "frame": frame.to_dict(),
                                   "padded_values": len(frame.padded()), "frame_sha256": frame.policy_sha256,
                                   "current_input_reach": {str(rounds): information_reach(frame, rounds) for rounds in (0, 1, 2)},
                                   "reset_reasons": resets, "semantic_request": request.to_dict(),
                                   "capability_masked": masked, "requested_wire": wire.to_dict()}
                            episode["decisions"].append(row)
                            packet = env.command(wire.to_dict(), observation=view)
                            row["requested_packet"] = packet
                            budget.reserve()
                            following, _, terminated, truncated, info = env.step_packet(packet)
                            budget.completed_decisions += 1
                            budget.completed_ticks += info["actual_ticks"]
                            row.update(final_observation=following, terminated=terminated, truncated=truncated, info=info)
                            validate_interval(packet, view, following, terminated, truncated, info)
                            view, previous = following, request
                            budget.check()
                        _require(view["tick"] == declared["per_episode_owned_cutoff_ticks"],
                                 "morphology_conformance_cutoff", "diagnostic must stop at its declared physical tick")
                        final = adapter.frame(view, goal, previous=previous, started_tick=fence.start_tick)
                        episode["final_frame_sha256"] = final.policy_sha256
                        previous_identity = final.identity
                        before = view
                        view = env.cancel()
                        episode["cancellation"] = cancellation_receipt(before, view)
                        fence.cancel()
                        episode["status"] = "passed"
                    except BaseException as exc:
                        episode["errors"].append(_error(exc))
                        fence.cancel()
                        if view is not None:
                            try:
                                episode["cleanup_observation"] = env.cancel()
                            except Exception as cleanup:
                                episode["errors"].append(_error(cleanup))
                        raise
                    finally:
                        episode["sha256"] = sha256_json(episode)
                        artifact = _write(output / f"{model}-{index}.json.gz", episode)
                        report["episodes"].append({"id": episode["id"], "status": episode["status"],
                                                   "decisions_prepared": len(episode["decisions"]),
                                                   "profile_sha256": episode.get("profile_sha256"),
                                                   "errors": episode["errors"], **artifact})
        _require(len(report["episodes"]) == 8 and budget.attempted_decisions == budget.completed_decisions == 32
                 and budget.completed_ticks == 192, "morphology_conformance_coverage", "all fixed cases must complete")
        budget.check()
        report["status"] = "passed"
    except BaseException as exc:
        report["errors"].append(_error(exc))
    if source_receipt is not None:
        try:
            _require(all(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
                         for name, expected in source_receipt["files"].items()),
                     "morphology_source_drift", "source bytes changed during conformance")
        except BaseException as exc:
            report["errors"].append(_error(exc))
    report["budget"] = budget.receipt()
    if continuation is not None:
        report["aggregate_budget"] = {
            "host_start_attempts": continuation["previous_host_start_attempts"] + report["host_start_attempts"],
            "attempted_decisions": budget.attempted_decisions, "completed_decisions": budget.completed_decisions,
            "completed_ticks": budget.completed_ticks,
            "wall_seconds": continuation["previous_wall_seconds"] + report["budget"]["wall_seconds"],
            "maximum_wall_seconds": continuation["maximum_aggregate_wall_seconds"],
            "maximum_game_decisions": continuation["maximum_aggregate_game_decisions"],
            "maximum_completed_ticks": continuation["maximum_aggregate_completed_ticks"],
            "failed_first_attempt_retained": True}
    if report["errors"] or not report["budget"]["wall_ceiling_respected"]:
        report["status"] = "failed"
    report["sha256"] = sha256_json(report)
    _write(output / "summary.json", report)
    return report
