"""Fail-closed T106 comparison admission and a fixed, inspectable job plan.

This module intentionally imports no model or game runtime. The frozen useful
skill prerequisite failed, so this revision supplies no route to start a trainer
by changing a result flag. A future commissioned experiment needs fresh reviewed
dependency evidence and a qualified waypoint reference under this action map.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import hashlib
import os
import subprocess
from typing import Any, Mapping

from nca.contracts import ContractError, canonical_json, sha256_json, strict_json_loads

COMPARISON_SHA256 = "e4113cc9d5860cce76dc73a18e4b279ad4f39cfc3338652069e4f7eb1e02bb9a"
ENTRY_SHA256 = "6d4b567e99308d57e0dfb076e2137ea693112ab8bf4ea30af144111bb9516654"
RESULT_SHA256 = "810ef64d9c95a682050e6e5420a847d1229f908bc490df499ad078e200935678"
SUMMARY_SHA256 = "5799a3fe61924d55244bc278e36d4ae0c264c6da1ad54d18a258ba4ed3aed784"
SOURCE_NAMES = (
    "src/nca/controllers/morphology/__init__.py", "src/nca/controllers/morphology/core.py",
    "src/nca/controllers/morphology/models.py", "src/nca/controllers/morphology/admission.py",
    "src/nca/controllers/morphology/conformance.py", "evals/morphology/conformance.py",
    "src/nca/contracts.py", "src/nca/data/records.py", "evals/morphology/run.py",
    "configs/morphology/comparison-v1.json", "configs/morphology/entry-gate-v1.json",
    "configs/morphology/profiles-v1.json", "configs/morphology/conformance-recovery-v1.json",
)


def read_json(path: Path) -> dict[str, Any]:
    value = strict_json_loads(path.read_bytes(), max_bytes=4 * 1024 * 1024)
    if not isinstance(value, dict):
        raise ContractError("morphology_evidence", "a JSON object is required")
    return value


def _self_hash(record: Mapping[str, Any], expected: str) -> None:
    value = dict(record)
    recorded = value.pop("sha256", None)
    if recorded != expected or sha256_json(value) != expected:
        raise ContractError("morphology_evidence_hash", "prerequisite evidence differs from its frozen receipt")


def entry_decision(declaration: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact evidence and recompute the already fixed numerical gate."""
    if sha256_json(declaration) != ENTRY_SHA256:
        raise ContractError("morphology_gate_changed", "the pre-holdout utility criterion cannot be relaxed")
    _self_hash(result, RESULT_SHA256)
    if (result.get("declaration_sha256") != ENTRY_SHA256 or result.get("t105_summary_sha256") != SUMMARY_SHA256
            or result.get("candidate_config_sha256") != declaration["candidate_config_sha256"]
            or result.get("declaration_predates_evaluation") is not True
            or result.get("thresholds_changed_after_holdout") is not False):
        raise ContractError("morphology_dependency", "wrong dependency, chronology or threshold provenance")
    expected_families = declaration["candidate_families_in_priority_order"]
    if set(result.get("candidates", {})) != set(expected_families):
        raise ContractError("morphology_coverage", "all declared candidate families must remain present")
    rows, selected = {}, None
    for family in expected_families:
        candidate = result["candidates"][family]
        gains = candidate["gain_over_initial"]
        checks = {
            "minimum_mean_success": candidate["mean_success"] >= declaration["minimum_mean_success"],
            "minimum_every_seed_success": candidate["minimum_seed_success"] >= declaration["minimum_every_seed_success"],
            "minimum_gain_over_initial": gains["mean"] >= declaration["minimum_mean_success_gain_over_initial"],
            "positive_success_gain_lower95": gains["lower95"] > 0,
            "all_original_character_runtime_integrity_gates": candidate["gates"]["all_original_character_runtime_integrity_gates"] is True,
        }
        passed = all(checks.values())
        if checks != candidate["gates"] or passed != candidate["passed"]:
            raise ContractError("morphology_gate_inconsistent", "recorded boolean cannot replace the numeric gate")
        rows[family] = {"passed": passed, "checks": checks,
                        "mean_success": candidate["mean_success"],
                        "minimum_seed_success": candidate["minimum_seed_success"],
                        "mean_paired_gain_over_initial": gains["mean"], "paired_gain_lower95": gains["lower95"],
                        "failed": [name for name, value in checks.items() if not value]}
        if passed and selected is None:
            selected = family
    if (result.get("selected_family") != selected
            or result.get("mechanical_utility_prerequisite_passed") is not (selected is not None)):
        raise ContractError("morphology_gate_inconsistent", "family selection differs from the fixed priority")
    return {"admitted": selected is not None, "selected_family": selected, "candidates": rows,
            "reason": "useful_per_rig_prerequisite_failed" if selected is None else "useful_per_rig_prerequisite_passed",
            "entry_declaration_sha256": ENTRY_SHA256, "prerequisite_result_sha256": RESULT_SHA256,
            "t105_summary_sha256": SUMMARY_SHA256}


def comparison_plan(config: Mapping[str, Any], profiles: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve finite job identities and allocations without executing a policy."""
    if sha256_json(config) != COMPARISON_SHA256:
        raise ContractError("morphology_comparison_changed", "comparison preparation differs from its declaration")
    if sha256_json(profiles) != config["host"]["profile_fixture_sha256"]:
        raise ContractError("morphology_profile_fixture", "accepted profile fixtures differ")
    split = config["splits"]
    training, heldout = set(split["training_profiles"]), set(split["zero_shot_profiles"])
    if training & heldout or not set(split["tuning_profiles"]) <= training:
        raise ContractError("morphology_split", "whole heldout profiles cannot enter training or tuning")
    geometry, arrangements = {}, {}
    for name in (*split["training_profiles"], *split["zero_shot_profiles"]):
        row = profiles["profiles"][name]
        definition = row["definition"]
        if sha256_json(definition) != row["profile_sha256"]:
            raise ContractError("morphology_profile_hash", "full profile receipt differs")
        geometry[name] = sha256_json({"parts": [{key: part[key] for key in ("size", "mass", "offset", "angle")}
                                                for part in definition["parts"]],
                                     "joints": [{key: joint.get(key) for key in ("anchor", "axis", "limits", "stiffness", "damping", "rest_length")}
                                                for joint in definition["joints"]]})
        arrangements[name] = sha256_json({"roles": [part["role"] for part in definition["parts"]],
                                         "joints": [(joint["a"], joint["b"], joint["type"], joint["role"])
                                                    for joint in definition["joints"]],
                                         "capabilities": config["capability_matching"][name]})
    if ({geometry[name] for name in training} & {geometry[name] for name in heldout}
            or {arrangements[name] for name in training} & {arrangements[name] for name in heldout}):
        raise ContractError("morphology_split", "whole heldout geometry/arrangement duplicates training")
    families = ("padded_recurrent", "shared_graph")
    seeds = config["future_learning"]["policy_seeds"]
    jobs = []
    for family, seed in product(families, seeds):
        jobs.append({"id": f"{family}-seed{seed}", "family": family, "seed": seed,
                     "training_profiles": split["training_profiles"],
                     "shared_training_decision_ceiling": config["future_learning"]["ppo_decisions_per_candidate_seed"],
                     "zero_shot": [{"profile": profile, "scenario_family": scenario}
                                   for profile, scenario in product(split["zero_shot_profiles"], split["zero_shot_families"])],
                     "adaptation_forks": [{"profile": profile, "decisions": config["future_learning"]["adaptation_decisions_per_candidate_seed_and_heldout_profile"],
                                           "starting_checkpoint": "this job's fixed shared final checkpoint", "fresh_hidden_optimizer_and_data_boundary": True}
                                          for profile in split["zero_shot_profiles"]]})
    budget = config["future_budget"]
    if sum(budget["decision_ceiling_by_phase"].values()) != budget["sum_phase_ceilings"] or budget["sum_phase_ceilings"] > budget["maximum_game_decisions"]:
        raise ContractError("morphology_budget", "phase allocation exceeds the fixed pilot ceiling")
    return {"jobs": jobs, "geometry_fingerprints": geometry, "arrangement_fingerprints": arrangements,
            "maximum_game_decisions": budget["maximum_game_decisions"],
            "allocated_phase_ceiling_sum": budget["sum_phase_ceilings"],
            "current_training_authorized": False, "current_optimization_or_transfer_jobs_started": 0,
            "independent_geometry_sweep_available": False,
            "waypoint_reference_qualified_under_this_map": False,
            "future_trainer_backend_implemented_or_admitted": False}


def _sources(root: Path) -> dict[str, Any]:
    root = root.resolve()
    for name in ("src/nca/contracts.py", "src/nca/controllers/morphology/admission.py"):
        source = root / name
        imported = Path(__file__) if name.endswith("admission.py") else Path(__import__("nca.contracts", fromlist=["__file__"]).__file__)
        if source.read_bytes() != imported.read_bytes():
            raise ContractError("morphology_source_identity", "the provided source checkout differs from imported code")
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_NAMES}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    return {"revision": revision, "files": files, "sources_sha256": sha256_json(files)}


def prepare_comparison(*, root: Path, config_path: Path, declaration_path: Path,
                       result_path: Path, profiles_path: Path, output: Path) -> dict[str, Any]:
    """Publish a reviewable blocked result, preserving failures and zero work counts."""
    output.mkdir(parents=True, exist_ok=False)
    record: dict[str, Any] = {
        "schema": "nca-morphology-admission/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "evidence_invalid", "admitted": False, "errors": [],
        "work": {"host_starts": 0, "game_decisions": 0, "model_loads": 0, "policy_calls": 0,
                 "optimizer_updates": 0, "holdout_evaluations": 0},
        "learned_transfer_measured": False, "owner_feel_accepted": False,
    }
    try:
        config, declaration, result, profiles = (read_json(path) for path in
                                                 (config_path, declaration_path, result_path, profiles_path))
        record["sources"] = _sources(root)
        record["config_sha256"] = sha256_json(config)
        # The utility gate precedes model, checkpoint, host or trainer imports.
        record["prerequisite"] = entry_decision(declaration, result)
        record["plan"] = comparison_plan(config, profiles)
        if not record["prerequisite"]["admitted"]:
            record["status"] = "training_blocked"
            record["reason"] = record["prerequisite"]["reason"]
        else:
            # This branch cannot be reached by the pinned failed T105 evidence.
            # It deliberately does not pretend a prepared plan is a trainer.
            record["status"] = "implementation_not_admitted"
            record["reason"] = "qualified_task_reference_and_reviewed_future_trainer_required"
    except (ContractError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        record["errors"].append({"code": exc.code if isinstance(exc, ContractError) else "invalid_preparation_input",
                                 "detail": str(exc) if isinstance(exc, ContractError) else type(exc).__name__})
    record["sha256"] = sha256_json(record)
    with (output / "admission.json").open("x", encoding="utf-8", newline="\n") as file:
        file.write(canonical_json(record) + "\n")
        file.flush()
        os.fsync(file.fileno())
    return record
