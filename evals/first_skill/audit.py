"""Read-only provenance and partition audit of a completed first-skill screen.

This postprocessor performs no host step, inference, optimizer update or policy
selection. The frozen experiment source remains distinct from this auditor.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from nca.contracts import canonical_json, sha256_json, strict_json_loads
from nca.data import read_episode


def _read(path: Path):
    return strict_json_loads(path.read_bytes())


def _check_self_hash(value):
    if sha256_json({key: item for key, item in value.items() if key != "sha256"}) != value["sha256"]:
        raise ValueError("artifact content differs from its recorded canonical hash")


def _file_hash(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory: Path, source: Path):
    manifest, summary, dataset = (_read(directory / name) for name in ("manifest.json", "summary.json", "dataset.json"))
    for record in (manifest, summary, dataset):
        _check_self_hash(record)
    if summary["status"] != "completed" or not summary["evaluation"]["complete"]:
        raise ValueError("an incomplete screen cannot pass the completed-screen audit")
    if summary["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("summary belongs to another pre-registration")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != manifest["learner_revision"]:
        raise ValueError("audit source must be the original experiment revision")
    for relative, digest in manifest["source_files"].items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source.resolve()) or _file_hash(path) != digest:
            raise ValueError(f"original experiment source differs: {relative}")
    frozen = _read(directory / "frozen-candidates.json")
    opened = _read(directory / "evaluation-opened.json")
    freeze_time = datetime.fromisoformat(frozen["frozen_utc"])
    opening_time = datetime.fromisoformat(opened["opened_utc"])
    if not freeze_time < opening_time or not opened["all_final_checkpoint_hashes_frozen"]:
        raise ValueError("holdout opened before final checkpoint selection froze")
    frozen_final = [item for item in frozen["checkpoints"] if item["file"].endswith("-final.pt")]
    if len(frozen_final) != 6:
        raise ValueError("all three seeds of both final candidate families are required")
    for item in frozen["checkpoints"]:
        if _file_hash(directory / "checkpoints" / item["file"]) != item["sha256"]:
            raise ValueError("checkpoint bytes changed after the candidate freeze")

    config = manifest["config"]
    families = {split: {item["id"]: item for item in config[f"{split}_families"]}
                for split in ("training", "tuning", "evaluation")}
    if any(set(families[left]) & set(families[right]) for left, right in
           (("training", "tuning"), ("training", "evaluation"), ("tuning", "evaluation"))):
        raise ValueError("scenario family partitions overlap")
    geometry = [(tuple(item["spawn"]), tuple(item["target"]))
                for split in families.values() for item in split.values()]
    if len(geometry) != len(set(geometry)):
        raise ValueError("identical geometry was relabeled across declared families")

    dataset_ids = {item["id"] for item in dataset["episodes"]}
    if len(dataset_ids) != 24:
        raise ValueError("the declared all-outcome teacher dataset is incomplete")
    counts, partitions, defects = Counter(), defaultdict(Counter), Counter()
    profile_ids, seen_ids, paired_configs = set(), set(), {}
    training_last, diagnostics_first = 0, None
    eligible_ppo, imitation = 0, 0
    for row in summary["results"]:
        path = (directory / row["artifact"]).resolve()
        if not path.is_relative_to(directory.resolve()) or _file_hash(path) != row["artifact_sha256"]:
            raise ValueError("episode bytes differ from the immutable result receipt")
        episode = read_episode(path)
        if episode.id in seen_ids or episode.id != row["episode_id"] or episode.sha256 != row["record_sha256"]:
            raise ValueError("episode identity is duplicated or changed")
        seen_ids.add(episode.id)
        if (episode.learner_revision != revision or episode.learner_sources_sha256 != manifest["learner_sources_sha256"]
                or episode.host_identity["host_revision"] != config["host_sha"]
                or episode.host_identity["bridge_sha256"] != config["bridge_sha256"]):
            raise ValueError("episode runtime/source provenance differs from the frozen screen")
        if episode.split != row["split"] or episode.scenario_family not in families[episode.split]:
            raise ValueError("record escaped its declared family partition")
        scenario = families[episode.split][episode.scenario_family]
        if tuple(episode.config["actors"][0]["position"]) != tuple(scenario["spawn"]):
            raise ValueError("actual configured spawn differs from its declared family")
        if sha256_json(episode.evaluation["metrics"]) != sha256_json(row["metrics"]):
            raise ValueError("separate evaluator result differs from its captured episode")
        if episode.reward_kind != "no_task_reward":
            raise ValueError("task evaluator reward was relabeled as a host reward")
        if episode.split != "training":
            key = (episode.split, episode.scenario_family)
            if key in paired_configs and paired_configs[key] != (episode.config_sha256, episode.profile_sha256):
                raise ValueError("paired candidates used different physical configurations or profiles")
            paired_configs[key] = episode.config_sha256, episode.profile_sha256
        if episode.id in dataset_ids:
            if episode.split != "training" or row["kind"] != "baseline":
                raise ValueError("nontraining data entered the teacher dataset")
            imitation += sum(1 for _ in episode.imitation_samples())
        profile_ids.add(episode.profile_sha256)
        counts["episodes"] += 1
        counts["decisions"] += episode.decisions
        counts["actual_base_ticks"] += episode.consumed_ticks
        partitions[episode.split]["episodes"] += 1
        partitions[episode.split]["decisions"] += episode.decisions
        for event in episode.events:
            if event.kind != "step":
                counts[event.kind + "_events"] += 1
                continue
            if event.context is None or tuple(event.context.target) != tuple(scenario["target"]):
                raise ValueError("policy goal differs from its explicitly declared caller context")
            if event.reward != 0:
                raise ValueError("evaluator task reward leaked into the stored host reward")
            counts["replaced_decisions"] += int(event.replaced)
            counts["interventions"] += int(event.intervention != "none")
            counts["invalid_decisions"] += int(not event.valid or not event.trace_complete)
            counts["refused_admissions"] += sum(admission["accepted"] is not True
                                                 for tick in event.ticks for admission in tick.admission)
            if episode.split == "training":
                # CaptureEvent wall times are episode-relative. The accepted
                # bridge's base receipts use one persistent host's monotonic
                # clock, which can compare capture order across episode resets.
                training_last = max(training_last, *(tick.completed_monotonic_usec for tick in event.ticks))
                if row["kind"] == "residual":
                    eligible_ppo += int(event.on_policy_eligible)
            else:
                first_usec = min(tick.received_monotonic_usec for tick in event.ticks)
                diagnostics_first = first_usec if diagnostics_first is None else min(diagnostics_first, first_usec)
                # Original source 7455789 evaluated deterministic residual means
                # with a Normal point-density score. Keep the preserved defect
                # visible; these episodes never entered training sequences.
                if row["kind"] == "residual" and event.on_policy_eligible:
                    defects[episode.split + "_deterministic_mean_marked_eligible"] += 1
    if not dataset_ids <= seen_ids or imitation != dataset["imitation_samples"]:
        raise ValueError("teacher dataset membership or imitation count changed")
    if counts["decisions"] != summary["actual_decisions"] or len(profile_ids) != 1:
        raise ValueError("screen decision accounting or physical profile changed")
    if eligible_ppo != sum(item["actor_samples"] for item in summary["ppo_updates"]):
        raise ValueError("optimizer actor sample count differs from admitted training events")
    if diagnostics_first is None or training_last >= diagnostics_first:
        raise ValueError("training and diagnostic/holdout capture timelines overlap")
    updates = list((directory / "learning").glob("residual-*.json"))
    if len(updates) != 24 or any(path.stat().st_mtime_ns >= int(freeze_time.timestamp() * 1e9) for path in updates):
        raise ValueError("PPO update ledger did not finish before the candidate freeze")
    if any(counts[name] for name in ("interventions", "invalid_decisions", "refused_admissions")):
        raise ValueError("screen contains an intervention, invalid interval or refused admission")
    result = {
        "schema": "nca-first-skill-artifact-audit/v1", "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_sha256": _file_hash(Path(__file__)), "manifest_sha256": manifest["sha256"],
        "summary_sha256": summary["sha256"], "learner_revision": revision,
        "learner_sources_sha256": manifest["learner_sources_sha256"], "source_files_verified": len(manifest["source_files"]),
        "counts": dict(counts), "partitions": {name: dict(items) for name, items in partitions.items()},
        "profile_sha256": next(iter(profile_ids)), "imitation_samples": imitation,
        "ppo_actor_samples": eligible_ppo, "ppo_update_count": len(updates),
        "all_checkpoint_bytes_unchanged": True, "checkpoint_freeze_utc": frozen["frozen_utc"],
        "evaluation_opened_utc": opened["opened_utc"], "training_capture_ended_before_all_diagnostics": True,
        "persistent_host_monotonic_usec": {"last_training_tick_completed": training_last,
                                           "first_diagnostic_tick_started": diagnostics_first},
        "training_dataset_uses_only_declared_teacher_episodes": True, "matched_physical_configurations": True,
        "evaluation_or_tuning_episodes_entered_optimizer": 0,
        "sequence_separation_evidence": "The verified frozen capture_task appends Transitions only under training=True; only the PPO call enables it. All optimizer updates precede the immutable candidate freeze, and every admitted training sample count matches its update ledger.",
        "preserved_original_metadata_defect": dict(defects),
        "metadata_defect_limit": "Deterministic evaluation residual means had point-density scores in original source. They were excluded by phase from all training. Original records remain immutable; corrected publication source removes those scores and requires explicit stochastic-draw identity.",
        "passed": True,
    }
    result["sha256"] = sha256_json(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source", type=Path, required=True, help="Unchanged original experiment checkout")
    args = parser.parse_args()
    result = audit(args.directory, args.source)
    with (args.directory / "audit.json").open("x") as stream:
        stream.write(canonical_json(result) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
