"""Strict staged evidence for one fixed candidate from an interrupted archival run.

This is a separate admission contract. The complete-T105 converter continues
to require a completed experiment and never admits this interrupted prefix.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path

from nca.contracts import ContractError, digest, sha256_json
from nca.data import read_episode
from nca.envs import physical_config
from .build import immutable_json, make_bundle, read_json, verify_artifact_files, verify_self_hash
from .spec import PINNED_PROFILE_SHA256, SCREEN_SHA256, context_for

DECLARATION_SCHEMA = "nca-fixed-candidate-evaluation-declaration/v1"
MANIFEST_SCHEMA = "nca-fixed-candidate-evaluation-manifest/v1"
SUMMARY_SCHEMA = "nca-fixed-candidate-evaluation-result/v1"
EXPORT_DECLARATION_SCHEMA = "nca-fixed-candidate-export-declaration/v1"
EVIDENCE_KIND = "staged_fixed_candidate/v1"
CANDIDATE = "bc-seed1103-final"
FIXED_POLICY_SHA256 = "8196e77d5ecc6ea0ef3dff597b89e5f80e1b32804cc17eb7f305900c2c4d9a1b"
LEARNER_REVISION = "745578947a6f1ac2159830c4ae4de5de5593f32f"
LEARNER_SOURCES = "bbb731f68316ef799710d3a0fc73f1a541b6c65292fea2b28a60d4bdf805ed8b"
LEARNER_CONFIG = "8e40c1f65a2fe32597ecc97d10bd6f4414f96762bdc198f5063a56acad68616c"
BUDGET = {"maximum_decisions": 320, "maximum_ticks": 1920, "maximum_wall_seconds": 300,
          "attempts": 1, "optimizer_updates": 0, "tuning": False, "automatic_retry": False}
ORIGIN_STATUS = "interrupted_no_summary"
ORIGIN_SCHEMA = "nca-interrupted-training-artifact-index/v1"
ORIGIN_RECEIPT = "interruption-index.json"
ORIGIN_NAMES = {"manifest.json", ORIGIN_RECEIPT, "interruption-fence.json", "environment.json", "dataset.json",
                "learning/bc-seed1103.json", "checkpoints/bc-seed1103-final.pt",
                "exports/bc-seed1103-final.weights.json", "checkpoints/residual-seed1103-u07.pt"}


def validate_declaration(value):
    expected = {"schema", "id", "candidate", "policy_sha256", "learner_revision", "learner_sources_sha256",
                "learner_config_sha256", "export_criteria_sha256", "training_status", "training_files", "budget",
                "scenario_selection", "scope", "original_termination_confirmed", "training_receipt_file",
                "training_receipt_schema"}
    if (set(value) != expected or value["schema"] != DECLARATION_SCHEMA or not value["id"]
            or value["candidate"] != CANDIDATE or value["policy_sha256"] != FIXED_POLICY_SHA256
            or value["learner_revision"] != LEARNER_REVISION
            or value["learner_sources_sha256"] != LEARNER_SOURCES
            or value["learner_config_sha256"] != LEARNER_CONFIG
            or value["export_criteria_sha256"] != SCREEN_SHA256 or value["training_status"] != ORIGIN_STATUS
            or value["training_receipt_file"] != ORIGIN_RECEIPT or value["training_receipt_schema"] != ORIGIN_SCHEMA
            or value["original_termination_confirmed"] is not False
            or sha256_json(value["budget"]) != sha256_json(BUDGET)
            or value["scenario_selection"] != "all original eight evaluation families, seeds and goals"
            or value["scope"] != "single fixed candidate evaluation; no efficacy comparison or deployment qualification"):
        raise ContractError("candidate_declaration", "exact bounded fixed-candidate declaration required")
    digest(value["policy_sha256"], "candidate_policy_sha256")
    if not isinstance(value["training_files"], dict) or not ORIGIN_NAMES <= set(value["training_files"]):
        raise ContractError("candidate_training_files", "all original training input identities required")
    for name, value in value["training_files"].items():
        if not isinstance(name, str):
            raise ContractError("candidate_training_files", "canonical file names required")
        digest(value, "training_file_sha256")


def _record(root, name):
    value, file_hash = read_json(root / name)
    verify_self_hash(value)
    return value, file_hash


def verify_metrics(metrics, *, decisions, ticks):
    """Exact original task metrics, including failed outcomes and timing cost."""
    booleans = {"success", "administratively_capped", "integrity_failure", "purposeful_launch", "owner_feel_accepted"}
    integers = {"failed_admissions", "elapsed_ticks", "primary_edges", "ordinary_trim_reversals"}
    nullable = {"success_seconds", "response_seconds"}
    numbers = {"return", "final_distance_cells", "best_distance_cells", "capped_completion_seconds",
               "peak_toward_speed_cells_per_second", "overshoot_cells", "airborne_fraction",
               "ordinary_trim_reversals_per_second", "trim_variation_per_second", "mean_absolute_angle_radians",
               "peak_angular_speed", "inference_p50_ms", "inference_p95_ms", "inference_max_ms"}
    if (set(metrics) != booleans | integers | nullable | numbers | {"reason"}
            or any(type(metrics[k]) is not bool for k in booleans)
            or any(type(metrics[k]) is not int or metrics[k] < 0 for k in integers)
            or any(type(metrics[k]) not in (int, float) or not math.isfinite(metrics[k]) for k in numbers)
            or any(metrics[k] is not None and (type(metrics[k]) not in (int, float)
                   or not math.isfinite(metrics[k]) or not 0 < metrics[k] <= ticks / 60) for k in nullable)
            or any(metrics[k] < 0 for k in numbers - {"return"})
            or metrics["elapsed_ticks"] != ticks or not 0 < ticks <= 240
            or metrics["failed_admissions"] != 0 or metrics["integrity_failure"]
            or metrics["administratively_capped"] or metrics["owner_feel_accepted"]
            or metrics["primary_edges"] > ticks or metrics["ordinary_trim_reversals"] > decisions
            or not 0 <= metrics["airborne_fraction"] <= 1
            or not 0 <= metrics["mean_absolute_angle_radians"] <= math.pi
            or metrics["best_distance_cells"] > metrics["final_distance_cells"]
            or not metrics["inference_p50_ms"] <= metrics["inference_p95_ms"] <= metrics["inference_max_ms"]
            or metrics["reason"] != ("arrival" if metrics["success"] else "task_deadline")
            or (metrics["success_seconds"] is not None) != metrics["success"]
            or metrics["capped_completion_seconds"] != (metrics["success_seconds"] if metrics["success"] else 4.)
            or not metrics["success"] and ticks != 240):
        raise ContractError("candidate_metrics", "complete finite original task metrics and actual receipt counts required")


def verify_episode(root, row, *, training, scenario, seed, split, policy_id, policy_hash):
    verify_artifact_files(root, {row["artifact"]: row["artifact_sha256"]})
    episode = read_episode(root / row["artifact"])
    expected_config = physical_config("jev", ticks=240, seed=seed)
    expected_config["actors"][0]["position"] = list(scenario["spawn"])
    if (row["episode_id"] != episode.id or row["record_sha256"] != episode.sha256
            or row["split"] != split or episode.split != split or row["policy_id"] != policy_id
            or row["policy_sha256"] != policy_hash or row["host_seed"] != seed
            or row["scenario_family"] != scenario["id"] or episode.scenario_family != scenario["id"]
            or episode.learner_revision != LEARNER_REVISION or episode.learner_sources_sha256 != LEARNER_SOURCES
            or episode.source_files != training["manifest"]["source_files"]
            or episode.profile_sha256 != PINNED_PROFILE_SHA256
            or row["profile_sha256"] != PINNED_PROFILE_SHA256
            or sha256_json(episode.host_identity) != sha256_json(training["environment"]["host_identity"])
            or sha256_json(episode.config) != sha256_json(expected_config)
            or row["decisions"] != episode.decisions or not 1 <= episode.decisions <= 40
            or episode.consumed_ticks != 6 * episode.decisions or episode.status != "completed"
            or episode.reward_kind != "no_task_reward" or not episode.final_observation_valid
            or episode.evaluation.get("sample_policy_sha256") != policy_hash
            or episode.evaluation.get("policy_id") != policy_id
            or sha256_json(episode.evaluation.get("metrics")) != sha256_json(row["metrics"])
            or row["error"] is not None or episode.evaluation.get("error") is not None):
        raise ContractError("candidate_episode", "complete original-family, source and policy receipts required")
    metrics = row["metrics"]
    verify_metrics(metrics, decisions=episode.decisions, ticks=episode.consumed_ticks)
    for event in episode.events:
        if event.kind != "step":
            continue
        if (not event.imitation_eligible or event.intervention != "none" or event.reward != 0
                or event.context is None or tuple(event.context.target) != tuple(scenario["target"])
                or (event.context.skill, event.context.pace, event.context.tolerance, event.context.timeout_seconds)
                    != ("directed_hop", .8, 4., 4.)
                or event.source.policy_id != policy_id or event.source.policy_sha256 != policy_hash
                or event.source.id != policy_id
                or event.source.kind != ("learned" if policy_id == CANDIDATE else "scripted")
                or event.source.training_privilege != "none"):
            raise ContractError("candidate_episode_admission", "all requests must retain their declared admitted source")
        if policy_id == CANDIDATE and (event.sample is None or event.sample.space != "wire_request"
                or event.sample.policy_sha256 != policy_hash or event.sample.log_probability is not None
                or event.sample.distribution is not None
                or sha256_json(event.sample.values) != sha256_json(event.requested_packet["action"])):
            raise ContractError("candidate_sample", "deterministic BC wire samples must retain exact requested values")
    return episode


def _verify_interruption_index(root, index, declaration, manifest):
    """Verify a separately named inventory; never synthesize a learner summary."""
    if ((root / "summary.json").exists() or (root / "evaluation-opened.json").exists()
            or index["schema"] != ORIGIN_SCHEMA or index["status"] != ORIGIN_STATUS
            or index["original_summary_present"] is not False
            or index["original_termination_confirmed"] is not False
            or index["original_evaluation_opened_receipt_present"] is not False
            or index["unconfirmed_pending_outcome"] is not True
            or index["retained_record_counts_are_original_run_totals"] is not False
            or any(index[name] is not None for name in ("original_exit_code", "original_measured_wall_seconds",
                   "original_total_attempted_decisions", "original_total_consumed_ticks"))
            or index["inventory_before_and_after_copy_equal"] is not True
            or index["snapshot_bytes_equal_original"] is not True
            or index["evaluation"] != {"complete": False, "efficacy_comparison": False}
            or index["runtime_qualified"] is not False
            or index["declared_budget"] != manifest["config"]["budget"]
            or index["learner_revision"] != LEARNER_REVISION
            or index["receipt_origin"] != "post-interruption read-only inventory of an unchanged retained prefix"):
        raise ContractError("candidate_interruption", "explicit incomplete prefix with unknown termination required")
    inventory = index["original_files"]
    if (not isinstance(inventory, dict) or sha256_json(inventory) != index["original_files_sha256"]
            or any(name in inventory for name in ("summary.json", "evaluation-opened.json", ORIGIN_RECEIPT,
                                                  "interruption-fence.json"))
            or any(set(row) != {"sha256", "bytes", "mtime_ns"} or type(row["bytes"]) is not int
                   or row["bytes"] <= 0 or type(row["mtime_ns"]) is not int for row in inventory.values())):
        raise ContractError("candidate_interruption_inventory", "exact retained original file inventory required")
    expected_files = {name: row["sha256"] for name, row in inventory.items()}
    expected_files[ORIGIN_RECEIPT] = declaration["training_files"][ORIGIN_RECEIPT]
    expected_files["interruption-fence.json"] = index["fencing_receipt_file_sha256"]
    if declaration["training_files"] != expected_files:
        raise ContractError("candidate_interruption_inventory", "declaration must bind the complete unchanged snapshot")
    verify_artifact_files(root, expected_files)
    if any((root / name).stat().st_size != row["bytes"] for name, row in inventory.items()):
        raise ContractError("candidate_interruption_inventory", "retained file sizes differ")
    fence, _ = _record(root, "interruption-fence.json")
    if (index["fencing_receipt_file"] != "interruption-fence.json"
            or fence["schema"] != "nca-interrupted-tool-session-fencing-receipt/v1"
            or fence["original_termination_confirmed"] is not False or fence["original_exit_code"] is not None
            or fence["operation"] != "write_stdin with Ctrl-C to the original known session handle"
            or fence["exact_result"] != "write_stdin failed: Unknown process id 32967"):
        raise ContractError("candidate_interruption_fencing", "actual unsuccessful original-session cancellation receipt required")
    actual_rows = [read_json(root / name)[0] for name in sorted(inventory) if name.startswith("metrics/")]
    if (sha256_json(actual_rows) != sha256_json(index["results"])
            or len({r["episode_id"] for r in actual_rows}) != len(actual_rows)
            or {r["artifact"] for r in actual_rows} != {name for name in inventory if name.startswith("episodes/")}
            or any(inventory[r["artifact"]]["sha256"] != r["artifact_sha256"] for r in actual_rows)):
        raise ContractError("candidate_interruption_records", "inventory must retain every actual metric and episode file")
    checkpoints = index["checkpoints"]
    if (len({r["file"] for r in checkpoints}) != len(checkpoints)
            or {"checkpoints/" + r["file"] for r in checkpoints}
                != {name for name in inventory if name.startswith("checkpoints/")}
            or any(inventory["checkpoints/" + r["file"]]["sha256"] != r["sha256"]
                   or inventory["checkpoints/" + r["file"]]["bytes"] != r["bytes"] for r in checkpoints)):
        raise ContractError("candidate_interruption_checkpoints", "index must retain all actual checkpoint file identities")
    updates = [{"file": name, **read_json(root / name)[0]} for name in sorted(inventory)
               if name.startswith("learning/residual-")]
    if sha256_json(updates) != sha256_json(index["ppo_updates"]):
        raise ContractError("candidate_interruption_updates", "completed update ledgers must retain their original bytes")
    counts = {"episodes": len(actual_rows), "decisions": sum(r["decisions"] for r in actual_rows),
              "consumed_ticks": sum(r["metrics"]["elapsed_ticks"] for r in actual_rows),
              "checkpoint_files": len(checkpoints), "completed_ppo_update_ledgers": len(updates)}
    if index["retained_record_counts"] != counts:
        raise ContractError("candidate_interruption_counts", "retained receipt counts must not be replaced with estimated totals")


def verify_origin(training_root: Path, declaration):
    """Admit completed BC training while keeping original termination unknown."""
    validate_declaration(declaration)
    verify_artifact_files(training_root, declaration["training_files"])
    manifest, manifest_hash = _record(training_root, "manifest.json")
    summary, summary_hash = _record(training_root, ORIGIN_RECEIPT)
    _verify_interruption_index(training_root, summary, declaration, manifest)
    dataset, dataset_hash = _record(training_root, "dataset.json")
    environment, environment_hash = read_json(training_root / "environment.json")
    numeric, numeric_hash = read_json(training_root / "exports/bc-seed1103-final.weights.json")
    learning, _ = read_json(training_root / "learning/bc-seed1103.json")
    meta = numeric["metadata"]
    if (summary["status"] != ORIGIN_STATUS or summary["evaluation"]["complete"] is not False
            or manifest["learner_revision"] != LEARNER_REVISION
            or sha256_json(manifest["config"]) != LEARNER_CONFIG or manifest["config_sha256"] != LEARNER_CONFIG
            or sha256_json(manifest["source_files"]) != LEARNER_SOURCES
            or manifest["learner_sources_sha256"] != LEARNER_SOURCES
            or summary["manifest_sha256"] != manifest["sha256"]
            or summary["config_sha256"] != LEARNER_CONFIG or summary["learner_sources_sha256"] != LEARNER_SOURCES
            or environment["manifest_sha256"] != manifest["sha256"]
            or meta.get("manifest_sha256") != manifest["sha256"] or meta.get("dataset_sha256") != dataset["sha256"]
            or meta.get("config_sha256") != LEARNER_CONFIG or meta.get("learner_sources_sha256") != LEARNER_SOURCES
            or (meta.get("phase"), meta.get("kind"), meta.get("training_seed"), meta.get("epochs"),
                meta.get("profile_sha256"), meta.get("training_privilege"))
                != ("final", "bc", 1103, 40, PINNED_PROFILE_SHA256, "none")
            or numeric["policy_sha256"] != declaration["policy_sha256"]
            or (learning["seed"], learning["epochs"], learning["unroll"]) != (1103, 40, 16)
            or len(learning["epoch_losses"]) != 40 or learning["samples_per_epoch"] != dataset["imitation_samples"]):
        raise ContractError("candidate_training_lineage", "completed BC40 training and explicitly interrupted original lineage required")
    for name in ("bc-seed1103-final.pt", "residual-seed1103-u07.pt"):
        matches = [row for row in summary["checkpoints"] if row["file"] == name]
        if len(matches) != 1 or matches[0]["sha256"] != declaration["training_files"]["checkpoints/" + name]:
            raise ContractError("candidate_checkpoint", "original checkpoint receipt must match actual declared bytes")
    expected_demos = {f"demo-{family['id']}-{index}": (family, seed)
                      for family in manifest["config"]["training_families"]
                      for index, seed in enumerate(family["demo_seeds"])}
    members = {row["id"]: row for row in dataset["episodes"]}
    rows = {row["episode_id"]: row for row in summary["results"] if row["episode_id"] in expected_demos}
    if (len(dataset["episodes"]) != 24 or set(members) != set(expected_demos) or set(rows) != set(expected_demos)
            or len([r for r in summary["results"] if r["episode_id"] in expected_demos]) != 24):
        raise ContractError("candidate_teacher_coverage", "all 24 original teacher episodes required")
    result = {"manifest": manifest, "summary": summary, "dataset": dataset, "environment": environment,
              "numeric": numeric, "metadata": meta, "manifest_file_sha256": manifest_hash,
              "summary_file_sha256": summary_hash, "dataset_file_sha256": dataset_hash,
              "environment_file_sha256": environment_hash, "numeric_file_sha256": numeric_hash}
    samples = 0
    for name, (family, seed) in expected_demos.items():
        row = rows[name]
        if row["kind"] != "baseline" or row["record_sha256"] != members[name]["record_sha256"]:
            raise ContractError("candidate_teacher_membership", "dataset must retain its original teacher records")
        episode = verify_episode(training_root, row, training=result, scenario=family, seed=seed,
                                 split="training", policy_id="authored-balanced",
                                 policy_hash=manifest["config"]["baseline_source_sha256"])
        samples += sum(1 for _ in episode.imitation_samples())
    if samples != dataset["imitation_samples"]:
        raise ContractError("candidate_teacher_samples", "all completed BC samples must be accounted for")
    from nca.controllers.learned import load_policy, policy_state_sha256
    policy, loaded_meta = load_policy(training_root / "checkpoints/bc-seed1103-final.pt",
                                     expected_config_sha256=LEARNER_CONFIG)
    if loaded_meta != meta or policy_state_sha256(policy) != declaration["policy_sha256"]:
        raise ContractError("candidate_numeric_checkpoint", "selected checkpoint and numeric lineage differ")
    import numpy as np
    if any(not np.array_equal(value.detach().cpu().numpy(), np.asarray(numeric["weights"][name], dtype=np.float32))
           for name, value in policy.state_dict().items()):
        raise ContractError("candidate_numeric_weights", "selected numeric arrays differ from checkpoint")
    verify_artifact_files(training_root, declaration["training_files"])
    return result


def _verify_evaluation_source_record(root, manifest, declaration):
    from nca.bench.deployment import _sources
    import subprocess
    if _sources(root) != manifest["evaluation_source_files"]:
        raise ContractError("candidate_evaluation_sources", "conversion must use the unchanged evaluated implementation")
    declaration_file = manifest["declaration_file"]
    verify_artifact_files(root, {declaration_file: manifest["declaration_file_sha256"]})
    actual_declaration, _ = read_json(root / declaration_file)
    if sha256_json(actual_declaration) != sha256_json(declaration):
        raise ContractError("candidate_evaluation_declaration", "original committed declaration must remain available and unchanged")
    revision = manifest["evaluation_revision"]
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ContractError("candidate_evaluation_revision", "original evaluation source revision required")
    try:
        committed = subprocess.check_output(["git", "show", f"{revision}:{declaration_file}"], cwd=root)
        tracked = set(subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision], cwd=root,
                                             text=True).splitlines())
        unchanged = subprocess.run(["git", "diff", "--quiet", revision, "--", *sorted(manifest["evaluation_source_files"])],
                                   cwd=root).returncode == 0
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("candidate_evaluation_revision", "recorded evaluation revision must remain inspectable") from exc
    if (committed != (root / declaration_file).read_bytes() or not unchanged
            or not set(manifest["evaluation_source_files"]) <= tracked):
        raise ContractError("candidate_evaluation_revision", "recorded source and declaration must match their original commit")


def verify_evaluation(training_root: Path, evaluation_root: Path):
    manifest, manifest_hash = _record(evaluation_root, "manifest.json")
    summary, summary_hash = _record(evaluation_root, "summary.json")
    worker, worker_hash = _record(evaluation_root, "worker-result.json")
    declaration = manifest["declaration"]
    training = verify_origin(training_root, declaration)
    if (manifest["schema"] != MANIFEST_SCHEMA or summary["schema"] != SUMMARY_SCHEMA
            or manifest["declaration_content_sha256"] != sha256_json(declaration)
            or summary["manifest_sha256"] != manifest["sha256"] or summary["status"] != "completed"
            or summary["complete_eight_family_coverage"] is not True or summary["errors"]
            or summary["policy_sha256"] != declaration["policy_sha256"]
            or summary["new_optimizer_updates"] != 0 or summary["runtime_qualified"] is not False
            or summary["efficacy_comparison"] is not False or summary["training_status"] != ORIGIN_STATUS
            or manifest["training_status"] != ORIGIN_STATUS
            or manifest["original_termination_confirmed"] is not False
            or summary["original_termination_confirmed"] is not False
            or not 0 < summary["wall_seconds"] <= BUDGET["maximum_wall_seconds"]
            or manifest["phase"] != "frozen_before_any_candidate_evaluation"
            or sha256_json(manifest["evaluation_source_files"]) != manifest["evaluation_sources_sha256"]
            or manifest["policy_execution_revision"] != LEARNER_REVISION
            or manifest["policy_execution_sources_sha256"] != LEARNER_SOURCES
            or summary["source_files_unchanged"] is not True or summary["training_files_unchanged"] is not True):
        raise ContractError("candidate_evaluation_lineage", "complete bounded staged evaluation required")
    _verify_evaluation_source_record(Path(__file__).resolve().parents[3], manifest, declaration)
    if (worker_hash != summary["worker_result_file_sha256"]
            or worker["schema"] != "nca-fixed-candidate-worker-result/v1"
            or worker["manifest_sha256"] != manifest["sha256"] or worker["status"] != "completed"
            or worker["errors"] or worker["host_start_attempts"] != 1
            or worker["new_optimizer_updates"] != 0 or worker["source_files_unchanged"] is not True
            or summary["worker_invocations"] != 1 or summary["host_start_attempts"] != 1
            or summary["worker_exit_code"] != 0 or summary["worker_termination_confirmed"] is not True
            or summary["unconfirmed_pending_outcome"] is not False
            or summary["unconfirmed_decision_reservations"] != 0
            or worker["unconfirmed_pending_outcome"] is not False
            or worker["unconfirmed_decision_reservations"] != 0
            or any(worker[key] != summary[key] for key in
                   ("policy_sha256", "results", "attempted_decisions", "actual_decisions", "actual_ticks"))
            or summary["attempted_decisions"] != summary["actual_decisions"]):
        raise ContractError("candidate_worker_evidence", "one complete actual worker result required")
    families = {item["id"]: item for item in training["manifest"]["config"]["evaluation_families"]}
    rows = summary["results"]
    if len(rows) != 8 or {row["scenario_family"] for row in rows} != set(families):
        raise ContractError("candidate_evaluation_coverage", "each of the original eight families must appear once")
    files = {"manifest.json": manifest_hash, "summary.json": summary_hash, "worker-result.json": worker_hash}
    decisions = ticks = 0
    for row in rows:
        if row["kind"] != "bc" or row["artifact"] in files:
            raise ContractError("candidate_evaluation_identity", "distinct fixed-BC episode artifacts required")
        family = families[row["scenario_family"]]
        episode = verify_episode(evaluation_root, row, training=training, scenario=family, seed=family["host_seed"],
                                 split="evaluation", policy_id=CANDIDATE, policy_hash=declaration["policy_sha256"])
        files[row["artifact"]] = row["artifact_sha256"]
        decisions += episode.decisions
        ticks += episode.consumed_ticks
    if (decisions != summary["actual_decisions"] or ticks != summary["actual_ticks"]
            or decisions > BUDGET["maximum_decisions"] or ticks > BUDGET["maximum_ticks"]):
        raise ContractError("candidate_evaluation_budget", "exact bounded episode accounting required")
    verify_artifact_files(evaluation_root, files)
    return training, manifest, summary, files


def _candidate_evidence(training, manifest, summary, evaluation_files, relative, accepted_t105_commit):
    all_files = dict(manifest["declaration"]["training_files"])
    all_files.update({f"{relative}/{name}": value for name, value in evaluation_files.items()})
    meta = training["metadata"]
    evidence = {**{key: meta[key] for key in ("config_sha256", "manifest_sha256", "learner_sources_sha256",
                    "dataset_sha256", "profile_sha256", "training_privilege", "kind", "training_seed")},
        "status": "trained_and_evaluated", "evidence_kind": EVIDENCE_KIND, "training_status": ORIGIN_STATUS,
        "training_receipt_file": ORIGIN_RECEIPT, "training_receipt_schema": ORIGIN_SCHEMA,
        "original_termination_confirmed": False,
        "checkpoint_sha256": all_files["checkpoints/bc-seed1103-final.pt"],
        "numeric_source_sha256": training["numeric_file_sha256"],
        "summary_sha256": evaluation_files["summary.json"],
        "training_receipt_sha256": training["summary_file_sha256"],
        "manifest_file_sha256": training["manifest_file_sha256"], "environment_file_sha256": training["environment_file_sha256"],
        "dataset_file_sha256": training["dataset_file_sha256"], "original_learner_revision": LEARNER_REVISION,
        "accepted_t105_commit": accepted_t105_commit, "selected_evaluation_rows": 8,
        "performance_conclusion": "single fixed candidate evaluated; no method or seed-family efficacy comparison",
        "selected_successes": sum(row["metrics"]["success"] for row in summary["results"]),
        "candidate_evaluation_manifest_sha256": manifest["sha256"], "candidate_evaluation_summary_sha256": summary["sha256"],
        "candidate_evaluation_root": relative, "staged_artifact_files": all_files,
        "evaluation_artifacts": [{"name": f"{relative}/{row['artifact']}", "sha256": row["artifact_sha256"]}
                                 for row in summary["results"]]}
    return evidence


def verify_staged_bundle(training_root, bundle):
    """Re-admit all actual staged inputs before a deployment screen."""
    evidence = bundle.value["training_evaluation"]
    relative = evidence["candidate_evaluation_root"]
    evaluation_root = (training_root / relative).resolve()
    if (not evaluation_root.is_relative_to(training_root.resolve()) or evaluation_root == training_root.resolve()
            or evaluation_root.relative_to(training_root.resolve()).as_posix() != relative):
        raise ContractError("candidate_collection", "canonical contained evaluation collection required")
    training, manifest, summary, files = verify_evaluation(training_root, evaluation_root)
    expected = _candidate_evidence(training, manifest, summary, files, relative, evidence["accepted_t105_commit"])
    if sha256_json(evidence) != sha256_json(expected) or bundle.network.policy_sha256 != FIXED_POLICY_SHA256:
        raise ContractError("candidate_bundle_lineage", "bundle differs from strict complete staged evaluation")
    return expected["staged_artifact_files"]


def convert_candidate(training_root: Path, evaluation_root: Path, profile, *, accepted_t105_commit: str,
                      output: Path, screen_config: Path, declaration_path: Path):
    """Convert separately evaluated BC bytes without completing their old run."""
    training_root, evaluation_root = training_root.resolve(), evaluation_root.resolve()
    if not evaluation_root.is_relative_to(training_root) or evaluation_root == training_root:
        raise ContractError("candidate_collection", "distinct contained evaluation collection required")
    if len(accepted_t105_commit) != 40 or any(c not in "0123456789abcdef" for c in accepted_t105_commit):
        raise ContractError("dependency_gate", "reviewed T105 dependency commit required")
    config, _ = read_json(screen_config)
    if sha256_json(config) != SCREEN_SHA256:
        raise ContractError("screen_config", "original deployment criteria must remain unchanged")
    training, manifest, summary, evaluation_files = verify_evaluation(training_root, evaluation_root)
    relative = evaluation_root.relative_to(training_root).as_posix()
    evidence = _candidate_evidence(training, manifest, summary, evaluation_files, relative, accepted_t105_commit)
    numeric = training["numeric"]
    context = context_for(profile, training["environment"]["host_identity"])
    if context.profile_sha256 != training["metadata"]["profile_sha256"]:
        raise ContractError("trained_profile", "deployment profile differs from completed BC training")
    all_files = evidence["staged_artifact_files"]
    export_declaration, declaration_hash = read_json(declaration_path)
    expected = {"schema": EXPORT_DECLARATION_SCHEMA, "candidate": CANDIDATE,
                "policy_sha256": FIXED_POLICY_SHA256, "training_status": ORIGIN_STATUS,
                "original_termination_confirmed": False,
                "training_receipt_file_sha256": training["summary_file_sha256"],
                "evaluation_manifest_sha256": manifest["sha256"],
                "evaluation_summary_sha256": summary["sha256"],
                "evaluation_artifact_files": evaluation_files,
                "export_criteria_sha256": SCREEN_SHA256, "attempts": 1,
                "new_optimizer_updates": 0, "automatic_retry": False}
    if sha256_json(export_declaration) != sha256_json(expected):
        raise ContractError("candidate_export_declaration", "prospective second declaration must bind the complete evaluated candidate")
    from nca.bench.candidate_evaluation import committed_sources
    conversion_root = Path(__file__).resolve().parents[3]
    conversion_revision, conversion_sources, declaration_relative = committed_sources(conversion_root, declaration_path)
    bundle = make_bundle(numeric, profile, context, evidence)
    verify_artifact_files(training_root, all_files)
    verify_artifact_files(conversion_root, {declaration_relative: declaration_hash, **conversion_sources})
    artifact_hash = immutable_json(output, bundle)
    return {"format": "nca-export-conversion/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "artifact": output.name, "artifact_sha256": artifact_hash, "bundle_sha256": bundle["sha256"],
            "policy_sha256": numeric["policy_sha256"], "training_evaluation": evidence,
            "runtime_context": context.to_dict(), "runtime_qualified": False,
            "export_declaration": {"file": declaration_relative, "file_sha256": declaration_hash,
                "content_sha256": sha256_json(export_declaration), "conversion_revision": conversion_revision,
                "conversion_sources_sha256": sha256_json(conversion_sources)}}
