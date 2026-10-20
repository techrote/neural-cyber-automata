"""Explicit conversion of the fixed evaluated BC checkpoint into a CPU bundle."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from nca.contracts import ContractError, canonical_json, digest, sha256_json, strict_json_loads
from .runtime import NumpyGRU, RuntimeBundle
from .spec import (ACTIONS, EXPECTED_SPECIFICATION, NORMALIZATION, RECURRENCE,
                   SENSORS, SCREEN_SHA256, RuntimeContext, context_for, source_files)


def immutable_json(path: str | Path, value: Mapping[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical_json(value)+"\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".export-", delete=False) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise ContractError("immutable_exists", target.name) from exc
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def read_json(path: str | Path, *, maximum_bytes: int = 16*1024*1024):
    path = Path(path)
    if path.stat().st_size > maximum_bytes:
        raise ContractError("artifact_size", path.name)
    data = path.read_bytes()
    return strict_json_loads(data, max_bytes=maximum_bytes), hashlib.sha256(data).hexdigest()


def verify_self_hash(value: Mapping[str, Any]):
    unsigned = dict(value)
    declared = unsigned.pop("sha256", None)
    if declared != sha256_json(unsigned):
        raise ContractError("evidence_hash", "immutable evidence self-hash differs")


def verify_artifact_files(root: str | Path, files: Mapping[str, str]) -> None:
    """Verify contained, canonical artifact paths against external file hashes.

    A summary's self-hash identifies its claims, not the episode bytes those
    claims describe. Conversion and screening must verify both identities.
    """
    root = Path(root).resolve()
    for name, expected in files.items():
        if (not isinstance(name, str) or not name or "\\" in name
                or any(part in ("", ".", "..") for part in name.split("/"))):
            raise ContractError("artifact_path", "canonical relative artifact path required")
        location = (root / name).resolve()
        if not location.is_relative_to(root) or location == root:
            raise ContractError("artifact_path", "artifact must remain inside its training collection")
        digest(expected, "artifact_sha256")
        actual = hashlib.sha256()
        try:
            with location.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    actual.update(chunk)
        except OSError as exc:
            raise ContractError("artifact_unavailable", "a declared training artifact is unavailable") from exc
        if actual.hexdigest() != expected:
            raise ContractError("artifact_hash", "a declared training artifact differs from its file hash")


def make_bundle(numeric: Mapping[str, Any], profile: Mapping[str, Any], context: RuntimeContext,
                training_evaluation: Mapping[str, Any]) -> dict[str, Any]:
    if numeric.get("format") != "nca-gru-numeric/v1" or numeric.get("specification") != EXPECTED_SPECIFICATION:
        raise ContractError("numeric_specification", "fixed BC numeric export v1 required")
    if (numeric.get("matrix_layout") != "row_major_out_features_by_in_features"
            or numeric.get("gru_gate_order") != ["reset", "update", "new"]
            or numeric.get("gru_equations") != {
                "r": "sigmoid(W_ir*x+b_ir+W_hr*h+b_hr)",
                "z": "sigmoid(W_iz*x+b_iz+W_hz*h+b_hz)",
                "n": "tanh(W_in*x+b_in+r*(W_hn*h+b_hn))", "h_next": "(1-z)*n+z*h"}):
        raise ContractError("numeric_equations", "matrix/gate/candidate convention differs")
    NumpyGRU(numeric["weights"])
    value = {"format": "nca-cpu-policy/v1", "specification": numeric["specification"],
             "context": context.to_dict(), "normalization": NORMALIZATION, "sensors": SENSORS,
             "actions": ACTIONS, "recurrence": RECURRENCE, "profile": dict(profile),
             "weights": numeric["weights"], "weights_sha256": sha256_json(numeric["weights"]),
             "policy_sha256": numeric["policy_sha256"], "training_evaluation": dict(training_evaluation),
             "runtime_sources": source_files(),
             "qualification": {"state": "requires_external_screen_receipt", "screen_sha256": SCREEN_SHA256,
                               "target": "Linux external Python NumPy FP32 CPU controller",
                               "owner_feel_accepted": False, "whole_game_determinism": False}}
    value["sha256"] = sha256_json(value)
    RuntimeBundle(value, context)
    return value


def convert(training_root: str | Path, profile: Mapping[str, Any], *, accepted_t105_commit: str,
            output: str | Path, screen_config: str | Path) -> dict[str, Any]:
    """Verify final weights against complete evaluated lineage before conversion.

    accepted_t105_commit is supplied by the enclosing reviewed issue workflow;
    the runtime does not equate an arbitrary metadata flag with remote acceptance.
    The dependency commit and exact original training sources are distinct pins.
    """
    if len(accepted_t105_commit) != 40 or any(c not in "0123456789abcdef" for c in accepted_t105_commit):
        raise ContractError("dependency_gate", "accepted T105 merge commit required")
    root = Path(training_root)
    config, _ = read_json(screen_config)
    if sha256_json(config) != SCREEN_SHA256:
        raise ContractError("screen_config", "predeclared deployment criteria changed")
    numeric, numeric_hash = read_json(root / "exports/bc-seed1103-final.weights.json")
    manifest, manifest_file_hash = read_json(root / "manifest.json")
    summary, summary_hash = read_json(root / "summary.json")
    environment, environment_hash = read_json(root / "environment.json")
    dataset, dataset_file_hash = read_json(root / "dataset.json")
    for record in (manifest, summary, dataset):
        verify_self_hash(record)
    meta = numeric["metadata"]
    if (summary.get("status") != "completed" or summary.get("evaluation", {}).get("complete") is not True
            or summary.get("manifest_sha256") != meta.get("manifest_sha256")
            or manifest["sha256"] != meta.get("manifest_sha256")
            or meta.get("config_sha256") != config["selection"]["learner_config_sha256"]
            or sha256_json(manifest.get("config")) != meta.get("config_sha256")
            or manifest.get("learner_revision") != config["selection"]["learner_revision"]
            or sha256_json(manifest.get("source_files")) != meta.get("learner_sources_sha256")
            or manifest.get("learner_sources_sha256") != meta.get("learner_sources_sha256")
            or dataset["sha256"] != meta.get("dataset_sha256")
            or summary.get("config_sha256") != meta.get("config_sha256")
            or summary.get("learner_sources_sha256") != meta.get("learner_sources_sha256")
            or meta.get("phase") != "final" or meta.get("kind") != "bc" or meta.get("training_seed") != 1103
            or meta.get("epochs") != 40 or meta.get("training_privilege") != "none"):
        raise ContractError("evaluation_gate", "complete fixed-final BC1103 training/evaluation lineage required")
    selected = [row for row in summary["results"]
                if row["split"] == "evaluation" and row["policy_id"] == "bc-seed1103-final"]
    families = {row["id"] for row in manifest["config"]["evaluation_families"]}
    if (len(selected) != len(families) or {r["scenario_family"] for r in selected} != families
            or any(row["policy_sha256"] != numeric["policy_sha256"] for row in selected)):
        raise ContractError("evaluation_coverage", "selected checkpoint was not evaluated on its complete fixed suite")
    selected_artifacts = {row["artifact"]: row["artifact_sha256"] for row in selected}
    if len(selected_artifacts) != len(selected):
        raise ContractError("evaluation_coverage", "distinct evaluation episodes required for each fixed family")
    verify_artifact_files(root, selected_artifacts)
    checkpoint = root / "checkpoints/bc-seed1103-final.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    entries = [row for row in summary["checkpoints"] if row["file"] == checkpoint.name]
    if len(entries) != 1 or entries[0]["sha256"] != checkpoint_hash:
        raise ContractError("checkpoint_hash", "trained checkpoint differs from evaluation summary")
    # Conversion is a Torch-enabled preparation command; deployed runtime.py
    # has an independently implemented GRU and does not import this learner.
    from nca.controllers.learned import load_policy, policy_state_sha256
    policy, loaded_meta = load_policy(checkpoint, expected_config_sha256=meta["config_sha256"])
    if loaded_meta != meta or policy_state_sha256(policy) != numeric["policy_sha256"]:
        raise ContractError("numeric_checkpoint", "numeric export and checkpoint lineage differ")
    import numpy as np
    for name, array in policy.state_dict().items():
        if not np.array_equal(array.detach().cpu().numpy(), np.asarray(numeric["weights"][name], dtype=np.float32)):
            raise ContractError("numeric_weights", name)
    context = context_for(profile, environment["host_identity"])
    if context.profile_sha256 != meta["profile_sha256"]:
        raise ContractError("trained_profile", "deployment mechanics differ from training")
    evidence = {**{key: meta[key] for key in ("config_sha256", "manifest_sha256", "learner_sources_sha256",
                                          "dataset_sha256", "profile_sha256", "training_privilege", "kind", "training_seed")},
                "status": "trained_and_evaluated", "checkpoint_sha256": checkpoint_hash,
                "numeric_source_sha256": numeric_hash, "summary_sha256": summary_hash,
                "manifest_file_sha256": manifest_file_hash, "environment_file_sha256": environment_hash,
                "dataset_file_sha256": dataset_file_hash,
                "original_learner_revision": manifest["learner_revision"],
                "accepted_t105_commit": accepted_t105_commit, "selected_evaluation_rows": len(selected),
                "performance_conclusion": summary["evaluation"]["candidates"]["bc"]["conclusion"],
                "selected_successes": sum(bool(row["metrics"]["success"]) for row in selected),
                "evaluation_artifacts": [{"name": row["artifact"], "sha256": row["artifact_sha256"]} for row in selected]}
    bundle = make_bundle(numeric, profile, context, evidence)
    artifact_hash = immutable_json(output, bundle)
    return {"format": "nca-export-conversion/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "artifact": Path(output).name, "artifact_sha256": artifact_hash, "bundle_sha256": bundle["sha256"],
            "policy_sha256": numeric["policy_sha256"], "training_evaluation": evidence,
            "runtime_context": context.to_dict(), "runtime_qualified": False}
