"""Immutable small learner bundles, complete hashes and explicit resume state."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from nca.contracts import canonical_json, sha256_json
from nca.controllers.learned import RecurrentPolicy, load_policy, policy_state_sha256
from nca.data.capture import source_identity


def immutable_bytes(path: str | Path, payload: bytes) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".bundle-", delete=False) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise ValueError(f"immutable artifact already exists: {target.name}") from exc
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def immutable_json(path: str | Path, value: Mapping[str, Any]) -> str:
    return immutable_bytes(path, (canonical_json(value) + "\n").encode())


def read_config(path: str | Path) -> tuple[dict[str, Any], str]:
    from nca.contracts import strict_json_loads
    from nca.randomness import ScenarioSplit
    config = strict_json_loads(Path(path).read_bytes())
    if config.get("schema") != "nca-first-skill-screen/v1" or config.get("actor") != "jev":
        raise ValueError("unknown frozen first-skill configuration")
    ScenarioSplit(tuple(item["id"] for item in config["training_families"]),
                  tuple(item["id"] for item in config["tuning_families"]),
                  tuple(item["id"] for item in config["evaluation_families"]))
    if len(config["training_seeds"]) != 3 or len(set(config["training_seeds"])) != 3:
        raise ValueError("screen requires three independent declared seeds")
    expected = {"skill": "directed_hop", "lane": "act004", "mode": "stepped", "base_hz": 60,
                "decision_ticks": 6, "helper_lease_ticks": 12, "baseline_tuning": "balanced"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("screen cadence, capability or authored mapping differs from implemented v1")
    expected_goal = {"pace": .8, "tolerance_cells": 4., "deadline_ticks": 240, "quiet_ticks": 12,
                     "maximum_arrival_speed": 6., "maximum_arrival_angle": .32, "require_support": True}
    if config.get("goal") != expected_goal:
        raise ValueError("task goal/deadline predicates differ from implemented v1")
    fixed_reward = {"schema": "jolt-arrival-best-progress/v1", "success_bonus_once": 2.,
                    "task_deadline_penalty_once": -.5, "time_per_second": -.05,
                    "effort_proxy_per_second": -.005, "wobble_per_second": -.02,
                    "ordinary_trim_variation": -.01}
    if any(config["reward"].get(key) != value for key, value in fixed_reward.items()):
        raise ValueError("reward manifest differs from implemented v1 terms")
    if (config["models"]["hidden_size"] != 32 or config["models"]["residual_scales"] != [.25, .2, .15]
            or config["features"]["count_bc"] != 26 or config["features"]["count_residual"] != 32
            or config["ppo"]["gamma_per_base_tick"] != .997 or config["ppo"]["lambda_per_base_tick"] != .99
            or config["ppo"]["decisions_per_seed"] != config["ppo"]["rollout_decisions"] * config["ppo"]["updates"]):
        raise ValueError("model, discount or rollout arithmetic differs from implemented v1")
    scenarios = [item for split in ("training", "tuning", "evaluation") for item in config[f"{split}_families"]]
    geometries = [(tuple(item["spawn"]), tuple(item["target"])) for item in scenarios]
    if len(set(geometries)) != len(geometries):
        raise ValueError("relabeling identical scenario geometry would contaminate family partitions")
    for spawn, target in geometries:
        if any(len(point) != 2 or not 128 <= point[0] < 640 or not 136 <= point[1] < 384 for point in (spawn, target)):
            raise ValueError("scenario goal/spawn lies outside its declared arena")
    budget = config["budget"]
    if not 1 <= budget["maximum_total_decisions"] <= 13000 or not 0 < budget["maximum_wall_seconds"] <= 1800:
        raise ValueError("this screen cannot silently expand its commissioned cost")
    return config, sha256_json(config)


def freeze_manifest(output: str | Path, config: Mapping[str, Any], dependency_commits: Mapping[str, str],
                    *, root: str | Path | None = None) -> dict[str, Any]:
    """Called before any game rollout; refuses missing dependency/source pins."""
    if set(dependency_commits) != {"t102", "t103", "t104"} or any(
            len(value) != 40 or any(char not in "0123456789abcdef" for char in value)
            for value in dependency_commits.values()):
        raise ValueError("accepted T102/T103/T104 merge commits must be supplied before a formal run")
    project = Path(root) if root else Path(__file__).resolve().parents[3]
    revision, sources, source_hash = source_identity(project)
    for relative, expected in (("src/nca/controllers/baseline.py", config["baseline_source_sha256"]),
                               ("src/nca/adapters/host_bridge.gd", config["bridge_sha256"])):
        if sources.get(relative) != expected:
            raise ValueError(f"frozen dependency source changed: {relative}")
    if torch.__version__ != config["runtime"]["torch"]:
        raise ValueError("PyTorch runtime differs from the declared CPU screen")
    import numpy
    if numpy.__version__ != config["runtime"]["numpy"]:
        raise ValueError("NumPy runtime differs from the declared screen")
    manifest = {"id": config["id"], "created_utc": datetime.now(timezone.utc).isoformat(),
                "phase": "pre_registered_before_any_rollout", "config": dict(config),
                "config_sha256": sha256_json(config), "learner_revision": revision,
                "source_files": sources, "learner_sources_sha256": source_hash,
                "accepted_dependency_commits": dict(dependency_commits),
                "runtime": {"torch": str(torch.__version__), "numpy": str(numpy.__version__),
                            "device": "cpu", "torch_threads": 1},
                "training_privilege": "none", "evaluation_opened": False}
    manifest["sha256"] = sha256_json(manifest)
    immutable_json(Path(output) / "manifest.json", manifest)
    return manifest


def write_checkpoint(path: str | Path, policy: RecurrentPolicy, metadata: Mapping[str, Any], *,
                     optimizer: torch.optim.Optimizer | None = None,
                     training_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {"format": "nca-first-skill/v1", "specification": policy.specification(),
               "policy_sha256": policy_state_sha256(policy),
               "state_dict": {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()},
               "metadata": dict(metadata), "optimizer": None if optimizer is None else optimizer.state_dict(),
               "training_state": None if training_state is None else dict(training_state)}
    stream = io.BytesIO()
    torch.save(payload, stream)
    data = stream.getvalue()
    if len(data) > 32 * 1024 * 1024:
        raise ValueError("screen checkpoint exceeds the explicit artifact limit")
    digest = immutable_bytes(path, data)
    return {"file": Path(path).name, "sha256": digest, "bytes": len(data),
            "policy_sha256": payload["policy_sha256"], "metadata": dict(metadata)}


def load_training_checkpoint(path: str | Path, *, expected_config_sha256: str,
                             expected_sources_sha256: str):
    """Restore optimizer/RNG only for a fresh-episode continuation lineage."""
    policy, metadata = load_policy(path, expected_config_sha256=expected_config_sha256)
    if metadata.get("learner_sources_sha256") != expected_sources_sha256:
        raise ValueError("resume source bytes differ; declare a new experiment instead")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if policy.kind != "residual" or payload.get("optimizer") is None or payload.get("training_state") is None:
        raise ValueError("checkpoint is not a resumable residual training boundary")
    state = payload["training_state"]
    if state.get("next_episode_boundary") is not True:
        raise ValueError("mid-episode restoration requires an unavailable complete host snapshot")
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
    optimizer.load_state_dict(payload["optimizer"])
    return policy, optimizer, state, metadata


def export_weights(path: str | Path, policy: RecurrentPolicy, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Portable numeric export for an independently validated runtime adapter.

    Export availability is distinct from ONNX/Godot runtime qualification.
    Gate order and PyTorch's reset-after affine candidate convention are
    explicit so a consumer can test sequential equivalence, including resets.
    """
    value = {"format": "nca-gru-numeric/v1", "specification": policy.specification(),
             "metadata": dict(metadata), "policy_sha256": policy_state_sha256(policy),
             "matrix_layout": "row_major_out_features_by_in_features",
             "gru_gate_order": ["reset", "update", "new"],
             "gru_equations": {"r": "sigmoid(W_ir*x+b_ir+W_hr*h+b_hr)",
                               "z": "sigmoid(W_iz*x+b_iz+W_hz*h+b_hz)",
                               "n": "tanh(W_in*x+b_in+r*(W_hn*h+b_hn))",
                               "h_next": "(1-z)*n+z*h"},
             "weights": {name: item.detach().cpu().tolist() for name, item in policy.state_dict().items()},
             "runtime_qualified": False}
    digest = immutable_json(path, value)
    return {"file": Path(path).name, "sha256": digest,
            "policy_sha256": value["policy_sha256"], "runtime_qualified": False}
