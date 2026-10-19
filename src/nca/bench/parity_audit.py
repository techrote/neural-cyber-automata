"""Read-only diagnosis of an immutable export screen; no host or training runs."""

from __future__ import annotations

import gzip
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from nca.contracts import ContractError, WireAction, sha256_json, strict_json_loads
from nca.controllers.baseline import SkillGoal
from nca.export import RuntimeBundle, context_for
from nca.export.build import read_json, verify_artifact_files, verify_self_hash
from nca.export.runtime import encode, map_action
from nca.export.spec import SCREEN_SHA256
from .common import pose_difference, wire_difference, write_artifact
from .deployment import _compare_episodes, _sources


def divergence_points(left: Mapping[str, Any], right: Mapping[str, Any], config):
    """Locate divergence without changing any original parity threshold."""
    comparison = _compare_episodes(left, right, config)
    if "reason" in comparison:
        return {"comparison": comparison, "first_action_difference": None,
                "first_pose_difference": None, "first_pose_tolerance_failure": None}
    first_action = first_pose = first_pose_failure = None
    for a, b in zip(left["steps"], right["steps"]):
        delta = wire_difference(a["action"], b["action"])
        if first_action is None and (delta["maximum_continuous_difference"] > 0 or not delta["boolean_actions_exact"]):
            first_action = {"issued_tick": a["issued_tick"], **delta}
    for a, b in zip(left["frames"], right["frames"]):
        delta = pose_difference(a, b)
        nonzero = (delta["maximum_part_position_difference_cells"] > 0
                   or delta["maximum_part_angle_difference_radians"] > 0 or not delta["detached_exact"])
        if nonzero and first_pose is None:
            first_pose = {"tick": a["tick"], **delta}
        failed = (delta["maximum_part_position_difference_cells"] > config["trajectory"]["maximum_part_position_difference_cells"]
                  or delta["maximum_part_angle_difference_radians"] > config["trajectory"]["maximum_part_angle_difference_radians"]
                  or not delta["detached_exact"])
        if failed and first_pose_failure is None:
            first_pose_failure = {"tick": a["tick"], **delta}
    return {"comparison": comparison, "initial_pose_exact": left["frames"][0] == right["frames"][0],
            "first_action_difference": first_action, "first_pose_difference": first_pose,
            "first_pose_tolerance_failure": first_pose_failure}


def _common_inputs(episode, bundle, policy, config):
    """Use each stored feature/hidden pair for both independent implementations.

    This isolates arithmetic from the different state observations caused by
    closed-loop trajectories. It never feeds an evaluator value to a policy.
    """
    import torch
    from nca.controllers.learned import clone_request, features
    if len(episode["observations"]) != len(episode["steps"]) or not episode["steps"]:
        raise ContractError("audit_sequence", "complete original decision observations required")
    goal = SkillGoal(**episode["goal"])
    previous = WireAction(effort=0)
    maxima = {name: 0.0 for name in ("features", "hidden", "logits", "value", "outputs", "wire",
                                    "recorded_features", "recorded_hidden", "recorded_wire", "recorded_value")}
    flags = recorded_flags = True
    with torch.no_grad():
        for view, step in zip(episode["observations"], episode["steps"]):
            if (view["observation_id"] != step["issued_observation_id"]
                    or view["tick"] != step["issued_tick"] or len(step["hidden_after"]) != 32):
                raise ContractError("audit_sequence", "stored input, output and source identity must agree")
            x = np.asarray(step["features"], dtype=np.float32)
            h = np.asarray(step["hidden_before"], dtype=np.float32)
            no, nv, nh, nl = bundle.network.forward(x, h)
            tl, tv, th = policy(torch.tensor(x).unsqueeze(0), torch.tensor(h).unsqueeze(0))
            to = policy.clone_outputs(tl)[0].numpy()
            na, ta = map_action(no, view, goal), clone_request(to.tolist(), view, goal)
            nf = encode(view, goal, previous, episode["initial"]["tick"])
            tf = features(view, goal, previous=previous, start_tick=episode["initial"]["tick"])
            wire = wire_difference(na.to_dict(), ta.to_dict())
            recorded_action = na if episode["controller"] == "numpy" else ta
            recorded_hidden = nh if episode["controller"] == "numpy" else th[0].numpy()
            recorded_value = nv if episode["controller"] == "numpy" else float(tv[0])
            recorded_wire = wire_difference(recorded_action.to_dict(), step["action"])
            values = {"features": max(abs(a-b) for a, b in zip(nf, tf)),
                      "hidden": float(np.max(np.abs(nh-th[0].numpy()))),
                      "logits": float(np.max(np.abs(nl-tl[0].numpy()))),
                      "value": abs(nv-float(tv[0])), "outputs": float(np.max(np.abs(no-to))),
                      "wire": wire["maximum_continuous_difference"],
                      "recorded_features": max(abs(a-b) for a, b in zip(nf, step["features"])),
                      "recorded_hidden": float(np.max(np.abs(recorded_hidden-np.asarray(step["hidden_after"], dtype=np.float32)))),
                      "recorded_wire": recorded_wire["maximum_continuous_difference"],
                      "recorded_value": abs(recorded_value-step["value"])}
            for name, value in values.items():
                if not np.isfinite(value):
                    raise ContractError("audit_nonfinite", "finite arithmetic comparisons required")
                maxima[name] = max(maxima[name], value)
            flags &= wire["boolean_actions_exact"]
            recorded_flags &= recorded_wire["boolean_actions_exact"]
            previous = WireAction.from_dict(step["action"])
    limits = config["arithmetic"]
    arithmetic_passed = (maxima["features"] <= limits["feature_absolute_tolerance"]
                         and all(maxima[key] <= limits["hidden_logit_value_absolute_tolerance"]
                                 for key in ("hidden", "logits", "value", "outputs"))
                         and maxima["wire"] <= limits["continuous_action_absolute_tolerance"] and flags)
    reconstruction_passed = (maxima["recorded_features"] <= limits["feature_absolute_tolerance"]
                              and max(maxima["recorded_hidden"], maxima["recorded_value"])
                                  <= limits["hidden_logit_value_absolute_tolerance"]
                              and maxima["recorded_wire"] <= limits["continuous_action_absolute_tolerance"]
                              and recorded_flags)
    return {"samples": len(episode["steps"]), "arithmetic_passed": arithmetic_passed,
            "recorded_reconstruction_passed": reconstruction_passed, "maximum_absolute_differences": maxima,
            "boolean_actions_exact": flags, "recorded_boolean_actions_exact": recorded_flags,
            "scope": "identical stored features and hidden input; original local observation and requested history"}


def audit_screen(*, root: Path, screen_root: Path, training_root: Path, bundle_path: Path,
                 bundle_hash: str, output: Path):
    """Produce a separate immutable diagnosis; preserve the original no-go."""
    import torch
    from nca.controllers.learned import load_policy, policy_state_sha256
    torch.set_num_threads(1)
    sources = _sources(root)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest, manifest_hash = read_json(screen_root / "manifest.json")
    summary, summary_hash = read_json(screen_root / "summary.json")
    verify_self_hash(manifest)
    verify_self_hash(summary)
    config = manifest["config"]
    if (sha256_json(config) != SCREEN_SHA256
            or summary["manifest_content_sha256"] != manifest["sha256"]
            or manifest["bundle_file_sha256"] != bundle_hash
            or summary["bundle_file_sha256"] != bundle_hash):
        raise ContractError("audit_lineage", "original screen, declaration and bundle identities must agree")
    if str(torch.__version__) != config["runtime"]["torch_reference"]:
        raise ContractError("reference_runtime", "original CPU Torch reference required")
    payload, _ = read_json(bundle_path)
    bundle = RuntimeBundle.load(bundle_path, context_for(payload["profile"], payload["context"]),
                                expected_sha256=bundle_hash)
    training_files = {
        "checkpoints/bc-seed1103-final.pt": payload["training_evaluation"]["checkpoint_sha256"]}
    verify_artifact_files(training_root, training_files)
    policy, _ = load_policy(training_root / "checkpoints/bc-seed1103-final.pt",
                            expected_config_sha256=config["selection"]["learner_config_sha256"])
    if policy_state_sha256(policy) != bundle.network.policy_sha256:
        raise ContractError("audit_policy", "checkpoint and runtime bundle must contain the same policy")
    files = {row["file"]: row["file_sha256"] for row in summary["artifacts"]}
    needed = {f"trajectories/{case['id']}-{kind}.json.gz" for case in config["trajectory"]["cases"]
              for kind in ("torch", "numpy")}
    if not needed.issubset(files):
        raise ContractError("audit_sequence", "original screen has no complete matched trajectory pair")
    verify_artifact_files(screen_root, {name: files[name] for name in needed})
    pairs = []
    for case in config["trajectory"]["cases"]:
        episodes, arithmetic = [], {}
        for kind in ("torch", "numpy"):
            filename = f"trajectories/{case['id']}-{kind}.json.gz"
            with gzip.open(screen_root / filename, "rb") as stream:
                data = stream.read(128 * 1024 * 1024 + 1)
            episode = strict_json_loads(data, max_bytes=128 * 1024 * 1024)
            if episode["controller"] != kind or episode["case"] != case:
                raise ContractError("audit_identity", "recorded trajectory identity differs")
            arithmetic[kind] = _common_inputs(episode, bundle, policy, config)
            episodes.append(episode)
        pairs.append({"case": case["id"], "common_input_arithmetic": arithmetic,
                      **divergence_points(*episodes, config)})
    verify_artifact_files(screen_root, {"manifest.json": manifest_hash, "summary.json": summary_hash,
                                       **{name: files[name] for name in needed}})
    verify_artifact_files(training_root, training_files)
    verify_artifact_files(bundle_path.parent, {bundle_path.name: bundle_hash})
    if _sources(root) != sources:
        raise ContractError("audit_source_drift", "diagnostic source changed during the audit")
    result = {"format": "nca-export-parity-audit/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "audit_base_revision": revision, "original_summary_file_sha256": summary_hash,
              "original_manifest_file_sha256": manifest_hash, "original_status": summary["status"],
              "original_runtime_qualified": summary["runtime_qualified"], "bundle_file_sha256": bundle_hash,
              "policy_sha256": bundle.network.policy_sha256, "screen_content_sha256": SCREEN_SHA256,
              "original_screen_source_sha256": summary["source_sha256"],
              "runtime_context": bundle.context.to_dict(),
              "reference_runtime": {"torch": str(torch.__version__), "threads": torch.get_num_threads()},
              "audit_source_files": sources, "audit_source_sha256": sha256_json(sources),
              "input_trajectory_files": {name: files[name] for name in sorted(needed)},
              "cases": pairs, "new_host_ticks": 0, "new_optimizer_updates": 0,
              "qualification_effect": "diagnostic only; original qualification result remains unchanged",
              "causal_limit": "common-input arithmetic cannot establish contact sensitivity or host repeatability"}
    result["sha256"] = sha256_json(result)
    write_artifact(output, "parity-audit.json", result)
    return result
