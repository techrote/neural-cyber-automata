"""Prospective, read-only admission of the commissioned LEARN-JOLT-003.

This module opens neither a host nor a model. Historical inputs are immutable;
the distinct experiment changes only explicitly declared terminal/execution
semantics. Review flags and syntactic identifiers alone never prove admission.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess

from nca.contracts import ContractError, canonical_json, sha256_json, strict_json_loads
from nca.train.jolt_admission import (
    AdmissionError, HOST_LAYOUT_SHA256, POLICY_SEEDS, _relative_file,
    case_fingerprints, file_sha256, validate_splits, verify_prerequisite,
)
from nca.train.jolt_execution_v3 import DECLARATION_SCHEMA, validate_budget_v3
from nca.train.jolt_execution_contract import validate_v3_allocation_identity
from nca.train.jolt_runtime_v3 import capability_sha256, capability_contract_v3, POLICY as CPU_PORTABILITY_POLICY

CONFIG_SCHEMA = "nca-jolt-learn-003/v1"
UNLIMITED_CONFIG_SCHEMA = "nca-jolt-learn-003/v2"
PORTABLE_RESOURCE_SCHEMA = "nca-jolt-003-resource-admission/v2"
PORTABLE_REVIEW_SCHEMA = "nca-learn-jolt-003-source-admission-review/v2"
SPLIT_SCHEMA = "nca-jolt-static-split/v2"
OPENED_SCHEMA = "nca-jolt-opened-inventory/v2"
STREAM_SCHEMA = "nca-jolt-ppo-case-stream/v1"
ALLOCATION_ID = "learn-jolt-003-20261022-a1"
AUTHORIZATION_REF = "https://github.com/techrote/neural-cyber-automata/issues/62#issuecomment-6347420584"
CLOCK_AUTHORIZATION_REF = "docs/DECISIONS.md#d-020--remove-operational-wall-clock-limits-and-rerun-ci"
BASE_CONFIG = "configs/learning/jolt-learn-002-v1.json"
BASE_CONFIG_SHA256 = "4114806d1317cfb189c8648c530e2667bc0e5f236fc54d44f2f23c7a5a309902"
BASE_SPLIT = "configs/learning/jolt-static-split-v1.json"
BASE_OPENED = "configs/learning/jolt-opened-inventory-v1.json"
FIXED_KEYS = (
    "actor", "architecture", "authored_assessment", "bc", "decision_ticks",
    "emergency_predicate", "excluded_claims", "goal", "helper_lease_ticks",
    "host_revision", "issue", "learned_gate", "learning_contract_notes", "mode",
    "ppo", "profile_sha256", "seed_rules", "selection", "training_seeds",
)
FIXED_SEMANTICS_SHA256 = "c25d66e01762213f4f588b19463f2e703d79cdd033909793d8f239bb1a409a64"
RETIREMENT_CONTRACT = {
    "schema": "nca-jolt-safe-retirement-contract/v3",
    "eligible_reason": "emergency_recovery_deadline",
    "emergency_consumed_ticks": 48,
    "success": False,
    "capped_completion_seconds": 4.0,
    "terminal_bootstrap": 0.0,
    "new_terminal_penalty": False,
    "ordinary_attempt_charged_before_supervision": True,
    "cutoff_peek_retirement_has_no_forward": True,
    "complete_native_cancellation_required": True,
    "raw_per_tick_task_flags_unchanged": True,
    "unknown_or_unverified_retirement_aborts": True,
    "maximum_ppo_starts_per_update": 256,
    "minimum_confirmed_decisions_per_ppo_episode": 1,
}


def _need(condition, reason):
    if not condition:
        raise AdmissionError(reason, code="jolt_v3_admission")


def _read(root, relative, *, expected=None):
    path = _relative_file(Path(root), relative)
    _need(path.stat().st_size <= 64 * 1024 * 1024, "admission input exceeds its bounded reader")
    if expected is not None:
        _need(file_sha256(path) == expected, "immutable admission input changed: " + relative)
    return strict_json_loads(path.read_bytes(), max_bytes=64 * 1024 * 1024)


def make_ppo_case_stream(cases):
    """Freeze every possible draw once, independently of episode outcomes."""
    ids = [case["id"] for case in cases]
    _need(len(ids) == 64 and len(set(ids)) == 64, "sixty-four fixed teacher cases are required")
    streams = {}
    for seed in POLICY_SEEDS:
        rng = random.Random(seed + 300000)
        streams[str(seed)] = [rng.choice(ids) for _ in range(8 * 256)]
    return {"schema": STREAM_SCHEMA, "rng": "python-random-choice-seed-plus-300000",
            "maximum_starts_per_seed": 2048, "seeds": streams}


def comparison_order(cases):
    """Exact case-major reference/BC/residual order, frozen before outcomes."""
    models = [f"{family}-seed{seed}-{phase}" for family in ("bc", "residual")
              for seed in POLICY_SEEDS for phase in ("initial", "final")]
    return {stage: [{"case_id": case["id"], "controller_id": name}
                    for case in cases[stage]
                    for name in ["authored", "idle"] + [model for model in models
                            if stage == "sealed" or model.endswith("-final")]]
            for stage in ("open", "sealed")}


def expanded_opened_inventory(root):
    """Conservatively include every original training candidate and open case.

    PPO could use only the original sixty-four teacher cases. All are opened,
    including unsuccessful and partial captures; outcome filtering is absent.
    The four diagnostic candidates are also excluded conservatively even though
    the original worker never reached evaluation. Sealed reuse is a separate
    causal proof and never follows from an old case label.
    """
    root = Path(root)
    old = _read(root, BASE_OPENED)
    split = _read(root, BASE_SPLIT)
    result = deepcopy(old)
    result["schema"] = OPENED_SCHEMA
    result["source_revision"] = "ab1706d47ccb12939673258d02901b54c84c828f"
    result["scope"] = ("Original complete historical lineage plus all LEARN-JOLT-002 teacher/PPO "
                       "candidates and conservatively opened diagnostic candidates; failures and partials retained.")
    sources = (BASE_CONFIG, BASE_SPLIT, BASE_OPENED,
        "docs/evidence/jolt-learn-002-result-20261021.json",
        "docs/evidence/jolt-learn-002-original-inventory-20261021.json",
        "docs/evidence/jolt-learn-002-training-postmortem-review-20261021.json",
        "docs/evidence/jolt-learn-002-costs-20261021.json",
        "docs/evidence/jolt-learn-002-original-readback-20261021.json",
        "docs/evidence/jolt-learn-002-archive-members-20261021.json",
        "docs/evidence/jolt-learn-002-downloaded-readback-20261021.json")
    result["source_files"].update({name: file_sha256(root / name) for name in sources})
    added = split["teacher"] + split["open"]
    for field, values in {
        "opened_layout_height_keys": ["".join(map(str, case["pile_heights"])) for case in added],
        "opened_contact_sha256": [case["contact_sha256"] for case in added],
        "opened_start_approach_sha256": [case["start_approach_sha256"] for case in added],
    }.items():
        result[field] = sorted(set(result[field]) | set(values))
    result["learn_jolt_002_opened_cases"] = deepcopy(added)
    result["learn_jolt_002_sealed_reuse_requires_separate_proof"] = True
    result["consumed_allocations"] = [{
        "experiment": "LEARN-JOLT-002", "execution_schema": "nca-learn-jolt-002-execution/v1",
        "source_revision": "a68e687ced52bef778375a2753b1362dea3704fa",
        "declaration_sha256": "29b8db695828f7c512a86e120c596645f23d1ff520ce6dab104fc8fbf6b66d5f",
        "status": "consumed_failed_not_resumable", "allocation_id": None,
    }]
    return result


def validate_splits_v3(root, split, opened):
    _need(split.get("schema") == SPLIT_SCHEMA and opened.get("schema") == OPENED_SCHEMA,
          "cross-version split or opened inventory")
    original_split, original_opened = _read(root, BASE_SPLIT), _read(root, BASE_OPENED)
    validate_splits(original_split, original_opened)
    _need(sha256_json(opened) == sha256_json(expanded_opened_inventory(root)),
          "complete opened inventory changed or omits original learning cases")
    _need(split.get("host_layout_source_sha256") == HOST_LAYOUT_SHA256,
          "host layout source changed")
    for stage in ("teacher", "open", "sealed"):
        _need(sha256_json(split.get(stage)) == sha256_json(original_split[stage]),
              "original case distributions, host seeds or case order changed")
        for case in split[stage]:
            _need(all(case.get(key) == value for key, value in case_fingerprints(case).items()),
                  "static contact/start fingerprint changed")
    _need(split.get("rng_realizations") == original_split["rng_realizations"],
          "static RNG realizations changed")
    for case in split["sealed"]:
        _need("".join(map(str, case["pile_heights"])) not in opened["opened_layout_height_keys"],
              "sealed whole layout was opened")
        _need(case["contact_sha256"] not in opened["opened_contact_sha256"],
              "sealed contact geometry was opened")
        _need(case["start_approach_sha256"] not in opened["opened_start_approach_sha256"],
              "sealed start/approach was opened")
        for x, y, width, height in opened["conservative_start_regions"]:
            sx, sy = case["spawn"]
            _need(not (x - 8 <= sx <= x + width + 8 and y - 8 <= sy <= y + height + 8),
                  "sealed start overlaps an unlocalized historical region")
    proof_ref = split.get("unopened_sealed_proof", {})
    proof = _read(root, proof_ref.get("path", ""), expected=proof_ref.get("file_sha256"))
    validate_unopened_proof(root, proof, original_split)
    return {"schema": "nca-jolt-split-admission/v3", "accepted": True,
            "case_counts": {"teacher": 64, "open": 4, "sealed": 16},
            "opened_inventory_sha256": sha256_json(opened), "split_sha256": sha256_json(split),
            "unopened_proof_sha256": sha256_json(proof), "new_outcomes_inspected": False}


def validate_unopened_proof(root, proof, split):
    _need(proof.get("schema") == "nca-jolt-003-unopened-sealed-proof/v1"
          and proof.get("source_revision") == "a68e687ced52bef778375a2753b1362dea3704fa"
          and proof.get("sealed_cases_sha256") == sha256_json(split["sealed"]),
          "unopened-case proof identifies a different source or candidate set")
    expected_inputs = {"docs/evidence/" + name for name in (
        "jolt-execution-learn-002-20261021.json", "jolt-learn-002-archive-members-20261021.json",
        "jolt-learn-002-costs-20261021.json", "jolt-learn-002-downloaded-readback-20261021.json",
        "jolt-learn-002-original-inventory-20261021.json", "jolt-learn-002-result-20261021.json",
        "jolt-learn-002-training-postmortem-review-20261021.json")}
    _need(set(proof.get("input_files", {})) == expected_inputs,
          "unopened-case proof lost an original input identity")
    for name, digest in proof["input_files"].items():
        _need(file_sha256(_relative_file(Path(root), name)) == digest,
              "unopened-case proof input changed")
    original = _read(root, "docs/evidence/jolt-execution-learn-002-20261021.json")
    _need(original["source_revision"] == proof["source_revision"]
          and proof.get("source_file") == "src/nca/train/jolt_screen.py"
          and proof.get("source_file_sha256") == original["source_files"][proof["source_file"]],
          "unopened-case causal proof lost the exact original executable source")
    readback = _read(root, "docs/evidence/jolt-learn-002-downloaded-readback-20261021.json")
    _need(readback["byte_readback"].get("verified") is True
          and readback["copied_original_readback"].get("verified") is True,
          "unopened-case original inventory lacks independent byte readback")
    required = {"six_final_barrier_before_evaluation", "durable_freeze_before_sealed_reset",
                "durable_sealed_marker_before_sealed_reset", "no_marker_deletion_path",
                "original_inventory_byte_verified", "no_evaluation_stage_reservations",
                "five_finals_only", "zero_open_and_sealed_rows", "no_barrier_artifacts"}
    _need(set(proof.get("checks", {})) == required and all(value is True for value in proof["checks"].values()),
          "unopened-case causal proof is incomplete")
    _need(proof.get("original_owner_journal_complete") is False
          and proof.get("original_result_qualified") is False,
          "unopened proof cannot repair the original failed evidence")
    result = _read(root, "docs/evidence/jolt-learn-002-result-20261021.json")
    _need(result["status"] == "failed", "original failure changed")
    for stage in ("open", "sealed"):
        _need(all(value == 0 for value in result["costs"]["stage_counts"][stage].values()),
              "original evaluation work exists")
    inventory = _read(root, "docs/evidence/jolt-learn-002-original-inventory-20261021.json")["files"]
    _need(not any(name in inventory for name in ("frozen-finals.json", "sealed-opened.json")),
          "original sealed barrier was published")
    finals = [name for name in inventory if name.startswith("checkpoints/") and "-final." in name]
    _need(len(finals) == 5, "original checkpoint inventory does not prove five finals")
    _need(not any(name.startswith(("episodes/open-", "episodes/sealed-", "samples/open-", "samples/sealed-"))
                  for name in inventory), "original evaluation artifacts exist")


def validate_config_v3(root_or_config, config=None):
    root = None if config is None else Path(root_or_config)
    if config is None:
        config = root_or_config
    _need(isinstance(config, dict) and (config.get("schema"), config.get("id")) in (
              (CONFIG_SCHEMA, "learn-jolt-003-v1"),
              (UNLIMITED_CONFIG_SCHEMA, "learn-jolt-003-v2")),
          "unknown separately versioned experiment")
    _need(sha256_json({key: config.get(key) for key in FIXED_KEYS}) == FIXED_SEMANTICS_SHA256,
          "fixed learning, policy, teacher, seeds, reward or qualification semantics changed")
    _need(config.get("allocation_id") == ALLOCATION_ID and config.get("authorization_ref") == AUTHORIZATION_REF,
          "configuration does not identify the actual new owner commission")
    if config["schema"] == UNLIMITED_CONFIG_SCHEMA:
        _need(config.get("clock_authorization_ref") == CLOCK_AUTHORIZATION_REF,
              "configuration does not bind the owner-authorized wall-clock revision")
    _need(config.get("execution_version") == 3 and type(config["execution_version"]) is int,
          "configuration has another execution version")
    validate_budget_v3(config["budget"])
    _need((config["budget"]["maximum_wall_seconds"] is None)
          == (config["schema"] == UNLIMITED_CONFIG_SCHEMA),
          "configuration version differs from its declared wall-clock mode")
    _need(sha256_json(config.get("retirement_contract")) == sha256_json(RETIREMENT_CONTRACT),
          "safe-retirement semantics changed")
    _need(config.get("ppo_case_stream") == make_ppo_case_stream(config["cases"]["teacher"]),
          "complete seeded PPO case stream changed")
    _need(config.get("comparison_order") == comparison_order(config["cases"]),
          "frozen comparator/case order changed")
    if root is not None:
        original = _read(root, BASE_CONFIG, expected=BASE_CONFIG_SHA256)
        _need(config.get("prerequisite") == original["prerequisite"],
              "original qualified teacher prerequisite was replaced")
        inputs = {name: _read(root, config[name]["path"], expected=config[name]["file_sha256"])
                  for name in ("prerequisite", "opened_inventory", "split")}
        verify_prerequisite(root, inputs["prerequisite"])
        for name, digest in inputs["opened_inventory"]["source_files"].items():
            _need(file_sha256(_relative_file(root, name)) == digest, "historical input changed: " + name)
        validate_splits_v3(root, inputs["split"], inputs["opened_inventory"])
        _need(config["cases"] == {stage: inputs["split"][stage] for stage in ("teacher", "open", "sealed")},
              "executing cases differ from their frozen split")


def prospective_binding(declaration):
    """Non-self-referential content reviewed before publication of final reviews.

    Reviews can be published after the executable commit in a documentation-only
    commit. Their full original bytes are embedded in the final declaration,
    while the executable checkout remains at the exact reviewed source commit.
    """
    result = {key: value for key, value in declaration.items()
              if key not in {"independent_reviews", "created_utc"}}
    # The v2 resource/review policy binds *capabilities*, not the volatile
    # GitHub VM model, kernel or number of available logical CPUs. Original
    # actual runtime metadata remains in the published declaration and receipts.
    resource = declaration.get("resource_admission")
    if isinstance(resource, dict) and resource.get("schema") == PORTABLE_RESOURCE_SCHEMA:
        result["runtime"] = {"cpu_capability_sha256": capability_sha256(declaration["runtime"]),
                             "policy": CPU_PORTABILITY_POLICY}
    return result


def validate_declaration_reviews(declaration):
    reviews = declaration.get("independent_reviews")
    _need(isinstance(reviews, list) and 2 <= len(reviews) <= 4,
          "two independent final executable-source reviews are required")
    reviewers, roles = set(), set()
    binding = sha256_json(prospective_binding(declaration))
    resource = declaration.get("resource_admission", {})
    portable = (isinstance(resource, dict) and resource.get("schema") == PORTABLE_RESOURCE_SCHEMA)
    expected_schema = PORTABLE_REVIEW_SCHEMA if portable else "nca-learn-jolt-003-source-admission-review/v1"
    runtime_digest = (capability_sha256(declaration["runtime"]) if portable
                      else sha256_json(declaration["runtime"]))
    for review in reviews:
        _need(isinstance(review, dict) and set(review) == {"path", "file_sha256", "content"},
              "versioned original review bytes are required")
        relative, content = review["path"], review["content"]
        _need(isinstance(relative, str) and relative.startswith("docs/evidence/")
              and ".." not in Path(relative).parts and relative.endswith(".json"),
              "review publication path is not repository evidence")
        _need(isinstance(content, str) and len(content.encode()) <= 256 * 1024
              and hashlib.sha256(content.encode()).hexdigest() == review["file_sha256"],
              "original independent review bytes changed")
        record = strict_json_loads(content, max_bytes=256 * 1024)
        _need(record.get("schema") == expected_schema
              and record.get("accepted") is True and record.get("offline_scope_only") is False
              and record.get("prospective_experiment_admitted") is True,
              "offline or planning review cannot admit this allocation")
        _need(record.get("source_revision") == declaration["source_revision"]
              and record.get("source_sha256") == declaration["source_sha256"]
              and record.get("config_sha256") == declaration["config_sha256"]
              and record.get("prospective_binding_sha256") == binding
              and record.get("host_identity_sha256") == sha256_json(declaration["host_identity"])
              and record.get("runtime_contract_sha256" if portable else "runtime_sha256") == runtime_digest
              and (not portable or record.get("cpu_comparison_policy") == CPU_PORTABILITY_POLICY),
              "source/config/runtime/prospective declaration differs from final review")
        reviewer = record.get("reviewer")
        _need(isinstance(reviewer, str) and 1 <= len(reviewer) <= 128 and reviewer not in reviewers,
              "independent reviewers must have distinct bounded identities")
        reviewers.add(reviewer)
        roles.update(record.get("review_roles", []))
        _need(record.get("unresolved_blocking_findings") == [], "source review retains blocking findings")
    _need({"terminal_learning_qualification", "operation_serialization_deadline_feasibility"} <= roles,
          "independent review does not cover both required admission roles")


def validate_execution_v3(root, declaration, *, config, host_identity):
    """Bind a clean executable checkout and all static prospective inputs."""
    from nca.train.jolt_screen import source_inventory, runtime_identity
    root = Path(root)
    validate_config_v3(root, config)
    _need(declaration.get("schema") == DECLARATION_SCHEMA
          and type(declaration.get("attempts")) is int and declaration["attempts"] == 1,
          "exactly one new v3 declaration is required")
    opened = _read(root, config["opened_inventory"]["path"])
    consumed = [row["allocation_id"] for row in opened["consumed_allocations"] if row["allocation_id"]]
    validate_v3_allocation_identity(declaration, consumed_allocation_ids=consumed)
    _need(declaration.get("allocation_id") == config["allocation_id"]
          and declaration.get("authorization_ref") == config["authorization_ref"],
          "declaration changed the commission")
    inventory = source_inventory(root)
    _need(declaration.get("source_files") == inventory
          and declaration.get("source_sha256") == sha256_json(inventory), "executing source closure changed")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    _need(declaration.get("source_revision") == revision, "execution checkout is not the exact reviewed commit")
    _need(not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip(),
          "execution checkout is dirty")
    _need(declaration.get("config_sha256") == sha256_json(config)
          and declaration.get("budget") == config["budget"]
          and declaration.get("automatic_retry") is False and declaration.get("automatic_extension") is False,
          "declaration changed its fixed configuration or budget")
    _need(declaration.get("host_identity") == host_identity,
          "host/build differs from the reviewed declaration")
    portable = (isinstance(declaration.get("resource_admission"), dict)
                and declaration["resource_admission"].get("schema") == PORTABLE_RESOURCE_SCHEMA)
    current_runtime = runtime_identity()
    runtime = declaration["runtime"]
    if portable:
        _need(capability_sha256(runtime) == capability_sha256(current_runtime),
              "pinned CPU software/architecture differs; hardware names are advisory")
    else:
        _need(runtime == current_runtime, "historical exact runtime identity differs")
    _need(runtime.get("torch") == "2.8.0+cpu" and runtime.get("numpy") == "2.3.5"
          and runtime.get("device") == "cpu", "the pinned CPU runtime is required")
    _need(declaration.get("teacher_prerequisite") == config["prerequisite"]
          and declaration.get("opened_inventory") == config["opened_inventory"]
          and declaration.get("split") == config["split"]
          and declaration.get("ppo_case_stream_sha256") == sha256_json(config["ppo_case_stream"])
          and declaration.get("comparison_order_sha256") == sha256_json(config["comparison_order"]),
          "prospective input or order identity changed")
    _need(declaration.get("output_identity") == {"basename": config["allocation_id"],
                "lifecycle_basename": config["allocation_id"] + "-lifecycle", "reuse_forbidden": True},
          "output identity changed")
    validate_resource_admission(declaration)
    validate_declaration_reviews(declaration)


def validate_resource_admission(declaration, *, output_parent=None):
    from nca.train.jolt_storage_v3 import storage_bounds_v3
    record = declaration.get("resource_admission", {})
    if not isinstance(record, dict):
        record = {}
    portable = record.get("schema") == PORTABLE_RESOURCE_SCHEMA
    if portable:
        from nca.train.jolt_memory_watchdog_v3 import validate_watchdog_policy_v3
        validate_watchdog_policy_v3(record.get("memory_watchdog"))
    _need(record.get("schema") in {"nca-jolt-003-resource-admission/v1", PORTABLE_RESOURCE_SCHEMA}
          and record.get("accepted") is True and record.get("unresolved_blocking_findings") == [],
          "full pipeline resource feasibility is unresolved")
    runtime_digest = (capability_sha256(declaration["runtime"]) if portable
                      else sha256_json(declaration["runtime"]))
    _need(record.get("source_sha256") == declaration.get("source_sha256")
          and record.get("config_sha256") == declaration.get("config_sha256")
          and record.get("host_identity_sha256") == sha256_json(declaration["host_identity"])
          and record.get("runtime_contract_sha256" if portable else "runtime_sha256") == runtime_digest
          and (not portable or record.get("cpu_comparison_policy") == CPU_PORTABILITY_POLICY),
          "resource admission describes a different execution")
    required = {"canonical_reservation_maximum", "reader_limits", "journal_aggregate_bytes",
                "total_artifact_bytes", "total_file_count", "peak_resident_memory",
                "available_disk", "original_clock_phase_enforcement", "complete_pipeline_time_feasibility"}
    budget = declaration.get("budget")
    if budget is not None:
        validate_budget_v3(budget)
    unlimited = budget is not None and budget["maximum_wall_seconds"] is None
    _need(not portable or unlimited,
          "portable CPU admission requires the separately authorized unlimited timing mode")
    expected_gates = {name: True for name in required}
    if unlimited:
        # Null records that no deadline feasibility or expiry enforcement is
        # being claimed. Source-bound reviews still cover the original clock,
        # ordered phases, complete work, finite counters and all resource gates.
        expected_gates.update(original_clock_phase_enforcement=None,
                              complete_pipeline_time_feasibility=None)
    gates = record.get("gates", {})
    _need(set(gates) == required and all(gates[name] is expected for name, expected in expected_gates.items()),
          "a required resource gate is unresolved or differs from the declared wall-clock mode")
    bounds = storage_bounds_v3()
    _need(record.get("serialization") == bounds,
          "resource admission changed the source-derived serialization envelope")
    required_bytes = record.get("required_free_disk_bytes")
    _need(type(required_bytes) is int
          and required_bytes >= bounds["required_disk_bytes_including_1_gib_reserve"],
          "resource admission omitted the complete artifact and reserve requirement")
    if output_parent is not None:
        _need(shutil.disk_usage(output_parent).free >= required_bytes,
              "available disk no longer satisfies the admitted bound")

