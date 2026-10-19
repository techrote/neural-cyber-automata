"""Exact v1 preprocessing, mechanics and runtime identity for a Jolt BC bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import platform
from typing import Any, Mapping

from nca.contracts import ContractError, Record, digest, sha256_json

FEATURE_NAMES = (
    "target_dx_div64", "target_dy_div64", "vx_div80", "vy_div100",
    "sin_rotation", "cos_rotation", "angular_velocity_div8", "grounded", "grounded_valid",
    "energy", "energy_valid", "cooldown", "cooldown_valid", "ground_distance_div96",
    "ground_distance_valid", "ceiling_distance_div96", "ceiling_distance_valid",
    "time_remaining_fraction", "goal_pace", "goal_tolerance_div8", "previous_move_x",
    "previous_trim", "previous_effort", "previous_primary", "previous_secondary", "previous_auxiliary")
NORMALIZATION = {
    "schema": "jolt-fixed-normalization/v1", "learned_statistics": False,
    "feature_names": FEATURE_NAMES,
    "target_displacement": {"divisor": 64, "clip": [-2, 2]},
    "velocity_x": {"divisor": 80, "clip": [-2, 2]},
    "velocity_y": {"divisor": 100, "clip": [-2, 2]},
    "rotation": "sin(angle),cos(angle); clockwise radians",
    "angular_velocity": {"divisor": 8, "clip": [-2, 2]},
    "optional": {"grounded": "value then validity", "energy_divisor": 1,
                 "cooldown_divisor": 1, "ground_distance_divisor": 96,
                 "ceiling_distance_divisor": 96, "clip": [0, 2], "missing": [0, 0]},
    "time_remaining": "max(0,1-(tick-start_tick)/(60*goal.timeout_seconds))",
    "goal": "pace,tolerance/8; explicit caller context",
    "history": "previous requested move_x,trim,effort,primary,secondary,auxiliary; zero initially",
    "units": "host cells, seconds and clockwise radians; no SI inference"}
SENSORS = {"schema": "ACT004-local/v2", "feature_schema": "jolt-local-goal-history/v1",
           "scope": "self, rig rays and bounded composed native cells",
           "actor": "jev", "feature_names": FEATURE_NAMES, "privilege": "none"}
ACTIONS = {"schema": "physical_wire_action/v2", "head_schema": "jolt-six-wire-channels/v1",
           "channels": ["move_x", "trim", "effort", "primary", "secondary", "auxiliary"],
           "head_map": ["tanh", "tanh", "sigmoid", "sigmoid>=0.5", "sigmoid>=0.5", "sigmoid>=0.5"],
           "aim": "unit(clamp(dx/30,-1,1),min(-0.55,dy/30-0.55))",
           "fixed": {"move_y": 0, "reel": 0, "tool": False, "attach_usable": False, "release_usable": False}}
RECURRENCE = {"hidden_size": 32, "dtype": "float32", "gate_order": ["reset", "update", "new"],
              "candidate": "tanh(W_in*x+b_in+r*(W_hn*h+b_hn))",
              "update": "(1-z)*n+z*h", "encoder": "tanh(linear(features))",
              "state_update": "one update per completed six-tick decision observation",
              "reset": "zero on episode, actor, profile, mount, goal or source/control change",
              "retirement": "same cancelled goal ID cannot resume in its actor lifetime",
              "mid_episode_host_restore": False}
EXPECTED_SPECIFICATION = {
    "version": 1, "kind": "bc", "hidden_size": 32,
    "feature_schema": "jolt-local-goal-history/v1", "feature_names": list(FEATURE_NAMES),
    "residual_reference_names": [], "action_schema": "jolt-six-wire-channels/v1",
    "residual_schema": "jolt-scaled-tanh-residual/v1", "residual_scales": [.25, .2, .15],
    "decision_ticks": 6, "base_hz": 60, "actor": "jev", "profile_version": 1,
    "hidden_reset": "zero_on_episode_actor_profile_mount_generation_goal_or_source_change",
    "state_update": "one_gru_update_per_completed_decision_observation"}
WEIGHT_SHAPES = {"encoder.weight": (32, 26), "encoder.bias": (32,),
                 "recurrent.weight_ih": (96, 32), "recurrent.weight_hh": (96, 32),
                 "recurrent.bias_ih": (96,), "recurrent.bias_hh": (96,),
                 "actor.weight": (6, 32), "actor.bias": (6,),
                 "critic.weight": (1, 32), "critic.bias": (1,), "log_std": (3,)}
PINNED_PROFILE_SHA256 = "97559dcca7b1523ee622fd886615f05f5d7f85bacc0655ba9d642e909dc112bc"
SCREEN_SHA256 = "f499cb94ec537f5c1cc1b601cfcfa37fdaf9f5e81abe365aecb23d72ad0f1df0"
PINNED_HOST = {"host_revision": "f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497",
               "bridge_sha256": "f58eb49369fff2b98f7d16da16c0531522aff589bf3008f6fff01972c9629938",
               "godot_binary_sha256": "f85bbc6b15e22416c7d797cd60b63286dd67b9cb13498847056c18520ae55a75",
               "native_artifact_sha256": "5bdc55e7f355004fc8a6468234b0d018d3919339ebfe47faca9997de3af17847",
               "rapier_artifact_sha256": "147ab5673b9687ccb7a3b12fec07ea55cc674ad8a75f529d1085bf7d61fbd831"}
# Include the static package import closure. Importing data.records also runs
# data.__init__, whose capture/replay exports import the owner adapter. These
# bytes are pinned even though inference itself never launches that owner.
SOURCE_FILES = ("src/nca/__init__.py", "src/nca/contracts.py", "src/nca/randomness.py", "src/nca/runs.py",
                "src/nca/export/__init__.py", "src/nca/export/runtime.py", "src/nca/export/spec.py",
                "src/nca/controllers/__init__.py", "src/nca/controllers/baseline.py", "src/nca/controllers/shaping.py",
                "src/nca/data/__init__.py", "src/nca/data/records.py", "src/nca/data/capture.py",
                "src/nca/data/replay.py", "src/nca/data/storage.py", "src/nca/adapters/__init__.py",
                "src/nca/adapters/godot.py", "src/nca/adapters/protocol.py")


def source_files() -> dict[str, str]:
    # Installed packages and source checkouts use the same file bytes. Paths in
    # the manifest are logical repository names, never machine-local paths.
    package = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((package / name.removeprefix("src/nca/")).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


@dataclass(frozen=True)
class RuntimeContext(Record):
    profile_sha256: str
    assist_sha256: str
    sensor_sha256: str
    action_sha256: str
    normalization_sha256: str
    host_revision: str
    bridge_sha256: str
    godot_binary_sha256: str
    native_artifact_sha256: str
    rapier_artifact_sha256: str
    python: str
    numpy: str
    platform: str
    architecture: str
    base_hz: int = 60
    decision_ticks: int = 6
    helper_lease_ticks: int = 12
    profile_version: int = 1
    actor: str = "jev"
    lane: str = "act004"

    def validate(self):
        for key in self.__dataclass_fields__:
            if key.endswith("sha256"):
                digest(getattr(self, key), key)
        if len(self.host_revision) != 40 or any(c not in "0123456789abcdef" for c in self.host_revision):
            raise ContractError("host_revision", "exact accepted host commit required")


def context_for(profile: Mapping[str, Any], host_identity: Mapping[str, Any], *,
                base_hz: int = 60, decision_ticks: int = 6, helper_lease_ticks: int = 12) -> RuntimeContext:
    import numpy
    if (profile.get("id") != "jev" or profile.get("version") != 1
            or profile.get("representation") != "articulated-rapier"):
        raise ContractError("profile_mismatch", "export v1 requires the declared physical Jolt profile")
    # Full profile digest includes part shape, channels, mounts and authored
    # assistance. The separate assist digest includes every physical tune and
    # connection, rather than selecting only flattering gyro coefficients.
    assists = {"profile": profile["profile"], "joints": profile["joints"]}
    return RuntimeContext(sha256_json(profile), sha256_json(assists), sha256_json(SENSORS),
                          sha256_json(ACTIONS), sha256_json(NORMALIZATION),
                          *(host_identity[k] for k in ("host_revision", "bridge_sha256", "godot_binary_sha256",
                                                       "native_artifact_sha256", "rapier_artifact_sha256")),
                          platform.python_version(), numpy.__version__, platform.system(), platform.machine(),
                          base_hz, decision_ticks, helper_lease_ticks)
