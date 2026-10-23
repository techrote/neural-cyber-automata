"""Trainer-only immutable capture, bounded cloning and exact CPU export preparation.

No function in this module launches a host, opens a final split, grants execution
admission, or initializes CUDA.  Physical collection is owned by the separately
qualified 4LY host/bridge.  This module accepts already-recorded JSON values,
validates them fail-closed, and can exercise tiny synthetic CPU learning fixtures.

© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import io
import math
import random
import re
from typing import Any, Mapping, Sequence

from nca.adapters.flight import finite_vector, velocity_action
from nca.contracts import ContractError, canonical_json, sha256_json, strict_json_loads
from nca.train.flight import build_policy
from nca.train.flight_accounting import POLICY_SEEDS


CAPTURE_SCHEMA = "nca-4ly-capture-episode/v1"
CHECKPOINT_SCHEMA = "nca-4ly-bc-checkpoint/v1"
CHECKPOINT_RECEIPT_SCHEMA = "nca-4ly-bc-checkpoint-receipt/v1"
EXPORT_SCHEMA = "nca-4ly-direct-export/v1"
EXPORT_WRAPPER_SCHEMA = "nca-4ly-direct-export-wrapper/v1"

MAX_CAPTURE_EPISODES = 120
MAX_CAPTURE_LABELS = 19_200
BC_EPOCHS = 20
BC_BATCH_SIZE = 128
BC_LEARNING_RATE = 0.001
MAX_OPTIMIZER_UPDATES_PER_SEED = 3_000
MAX_SAMPLE_PRESENTATIONS_PER_SEED = 384_000
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024

PREPROCESSING_SPEC = {
    "schema": "nca-4ly-preprocessing/v1",
    "source_observation_schema": "4ly-observation/v1",
    "flat_observation_schema": "4ly-flat-observation/v1",
    "field_count": 54,
    "normalization": "host_defined_fixed",
    "validity_masks": "host_embedded_in_54_channels",
    "additional_fitted_statistics": None,
    "fitted_from": None,
    "evaluator_fields": "excluded",
}
ACTION_SPEC = {
    "schema": "nca-4ly-action-export/v1",
    "host_action_schema": "4ly-action/v1",
    "network_outputs": 2,
    "network_transform": "tanh",
    "dispatch_transform": "unit_disk_projection",
    "range": [-1.0, 1.0],
    "semantics": "normalized_desired_velocity",
}
BC_SCHEDULE = {
    "schema": "nca-4ly-bc-schedule/v1",
    "policy_seeds": list(POLICY_SEEDS),
    "epochs": BC_EPOCHS,
    "batch_size": BC_BATCH_SIZE,
    "learning_rate": BC_LEARNING_RATE,
    "loss": "mean_squared_normalized_desired_velocity",
    "maximum_optimizer_updates_per_seed": MAX_OPTIMIZER_UPDATES_PER_SEED,
    "maximum_sample_presentations_per_seed": MAX_SAMPLE_PRESENTATIONS_PER_SEED,
    "checkpoint_epochs": list(range(BC_EPOCHS + 1)),
    "device": "cpu",
    "cuda_allowed": False,
}
PREPROCESSING_SHA256 = sha256_json(PREPROCESSING_SPEC)
ACTION_SHA256 = sha256_json(ACTION_SPEC)
BC_SCHEDULE_SHA256 = sha256_json(BC_SCHEDULE)

_SOURCE_KEYS = {
    "host_revision", "host_build_sha256", "profile_sha256",
    "controller_sha256", "teacher_sha256",
}
_OUTCOME_KEYS = {"status", "reason", "terminated", "truncated", "failed", "success"}


def _fail(code: str, detail: str) -> None:
    raise ContractError(code, detail)


def _digest(value: Any, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        _fail("flight_learning_digest", f"{name} must be a SHA-256 digest")
    return value


def _revision(value: Any, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[a-f0-9]{40}", value) is None:
        _fail("flight_learning_revision", f"{name} must be a full Git revision")
    return value


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value) is None:
        _fail("flight_learning_id", f"{name} is invalid")
    return value


def _exact_mapping(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        _fail("flight_learning_shape", f"{name} must contain exactly {sorted(keys)}")
    return value


def _json_copy(value: Any) -> Any:
    return strict_json_loads(canonical_json(value), max_bytes=32 * 1024 * 1024)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ContractError("flight_learning_runtime", "optional CPU Torch runtime is required") from exc
    return torch


def _validate_source(value: Any) -> dict[str, str]:
    source = _exact_mapping(value, _SOURCE_KEYS, "capture.source")
    result = dict(source)
    _revision(result["host_revision"], "capture.source.host_revision")
    for key in _SOURCE_KEYS - {"host_revision"}:
        _digest(result[key], f"capture.source.{key}")
    return result


def _validate_observation(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("flight_capture_observation", f"{name} must be an authoritative owner observation")
    observation = _json_copy(dict(value))
    if observation.get("version") != "4ly-observation/v1" or observation.get("action_version") != "4ly-action/v1":
        _fail("flight_capture_schema", f"{name} has an incompatible observation/action schema")
    for key in ("run_id", "actor_id", "observation_id"):
        if type(observation.get(key)) is not str or not observation[key]:
            _fail("flight_capture_identity", f"{name}.{key} is missing")
    if observation["actor_id"] != "Trainer":
        _fail("flight_capture_profile", "capture is Trainer-only")
    for key in ("epoch", "observed_tick", "control_generation"):
        if type(observation.get(key)) is not int or observation[key] < 0:
            _fail("flight_capture_identity", f"{name}.{key} is invalid")
    observation["flat"] = list(finite_vector(observation.get("flat"), 54))
    return observation


def policy_input(observation: Mapping[str, Any]) -> tuple[float, ...]:
    """Return only the host-authored 54-value learner view.

    Extra dictionaries (including evaluator-only values or callables) are never
    traversed, serialized or exposed to the model.
    """
    if not isinstance(observation, Mapping):
        _fail("flight_policy_observation", "policy input must be an observation mapping")
    if observation.get("version") != "4ly-observation/v1" or observation.get("action_version") != "4ly-action/v1":
        _fail("flight_policy_schema", "scored policy observation schema differs")
    if observation.get("actor_id") != "Trainer":
        _fail("flight_policy_profile", "scored policy is Trainer-only")
    for key in ("run_id", "observation_id"):
        if type(observation.get(key)) is not str or not observation[key]:
            _fail("flight_policy_identity", f"missing {key}")
    for key in ("epoch", "observed_tick", "control_generation"):
        if type(observation.get(key)) is not int or observation[key] < 0:
            _fail("flight_policy_identity", f"invalid {key}")
    return finite_vector(observation.get("flat"), 54)


@dataclass(frozen=True)
class FlightSample:
    observation: tuple[float, ...]
    action: tuple[float, float]
    episode_id: str
    transition_index: int
    consumed_ticks: int
    terminal: bool

    @property
    def sha256(self) -> str:
        return sha256_json({
            "observation": list(self.observation),
            "action": list(self.action),
            "episode_id": self.episode_id,
            "transition_index": self.transition_index,
            "consumed_ticks": self.consumed_ticks,
            "terminal": self.terminal,
        })


@dataclass(frozen=True)
class ValidatedFlightCapture:
    canonical: str
    record_sha256: str
    declaration_sha256: str
    capture_id: str
    episode_id: str
    source_sha256: str
    status: str
    samples: tuple[FlightSample, ...]
    exclusions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return strict_json_loads(self.canonical, max_bytes=32 * 1024 * 1024)


@dataclass(frozen=True)
class FlightCaptureDataset:
    declaration_sha256: str
    capture_id: str
    source_sha256: str
    episodes: tuple[ValidatedFlightCapture, ...]
    samples: tuple[FlightSample, ...]
    sha256: str


def _validate_outcome(value: Any, name: str) -> dict[str, Any]:
    outcome = dict(_exact_mapping(value, _OUTCOME_KEYS, name))
    if outcome["status"] not in ("completed", "failed", "refused", "canceled", "inconclusive"):
        _fail("flight_capture_outcome", f"{name}.status is invalid")
    if type(outcome["reason"]) is not str or not outcome["reason"]:
        _fail("flight_capture_outcome", f"{name}.reason is required")
    for key in ("terminated", "truncated", "failed", "success"):
        if type(outcome[key]) is not bool:
            _fail("flight_capture_outcome", f"{name}.{key} must be boolean")
    if outcome["success"] and (not outcome["terminated"] or outcome["truncated"] or outcome["failed"]):
        _fail("flight_capture_outcome", f"{name} has an impossible success classification")
    return outcome


def _validate_teacher_action(value: Any, teacher: Mapping[str, str], before: Mapping[str, Any], name: str):
    action = dict(_exact_mapping(value, {
        "source_id", "source_sha256", "observation_id", "status",
        "requested_action", "admitted_action", "reason",
    }, name))
    if action["source_id"] != teacher["id"] or action["source_sha256"] != teacher["sha256"]:
        _fail("flight_capture_teacher", f"{name} teacher identity differs")
    if action["observation_id"] != before["observation_id"]:
        _fail("flight_capture_teacher", f"{name} action belongs to another observation")
    if action["status"] not in ("accepted", "refused", "unknown"):
        _fail("flight_capture_teacher", f"{name}.status is invalid")
    if type(action["reason"]) is not str or not action["reason"]:
        _fail("flight_capture_teacher", f"{name}.reason is required")
    if action["requested_action"] is not None:
        action["requested_action"] = list(velocity_action(action["requested_action"])["move"])
    if action["admitted_action"] is not None:
        action["admitted_action"] = list(velocity_action(action["admitted_action"])["move"])
    if action["status"] == "accepted":
        if action["requested_action"] is None or action["admitted_action"] is None:
            _fail("flight_capture_teacher", f"{name} accepted action lacks requested/admitted values")
    elif action["admitted_action"] is not None:
        _fail("flight_capture_teacher", f"{name} refused/unknown action cannot invent an admitted label")
    return action


def _validate_physical(value: Any, name: str) -> dict[str, Any]:
    physical = dict(_exact_mapping(value, {"status", "consumed_ticks", "reason"}, name))
    if physical["status"] not in ("confirmed", "refused", "unknown"):
        _fail("flight_capture_physical", f"{name}.status is invalid")
    if type(physical["reason"]) is not str or not physical["reason"]:
        _fail("flight_capture_physical", f"{name}.reason is required")
    ticks = physical["consumed_ticks"]
    if physical["status"] == "confirmed":
        if type(ticks) is not int or not 1 <= ticks <= 6:
            _fail("flight_capture_physical", f"{name} confirmed tick count is invalid")
    elif ticks is not None:
        _fail("flight_capture_physical", f"{name} unconfirmed work cannot claim consumed ticks")
    return physical


def _same_lifetime(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return all(before[key] == after[key] for key in ("run_id", "actor_id", "epoch", "control_generation"))


def validate_capture_episode(value: Mapping[str, Any], *, declaration_sha256: str,
                             expected_source: Mapping[str, Any]) -> ValidatedFlightCapture:
    """Validate one immutable already-recorded episode without performing host work."""
    _digest(declaration_sha256, "declaration_sha256")
    expected = _validate_source(expected_source)
    record = _json_copy(dict(value))
    _exact_mapping(record, {
        "schema", "declaration_sha256", "capture_id", "episode_id", "profile",
        "split", "scenario_id", "host_seed", "source", "teacher",
        "initial_observation", "transitions", "outcome",
    }, "capture")
    if record["schema"] != CAPTURE_SCHEMA:
        _fail("flight_capture_schema", "unsupported capture schema")
    if record["declaration_sha256"] != declaration_sha256:
        _fail("flight_capture_declaration", "capture belongs to another accounting declaration")
    capture_id = _identifier(record["capture_id"], "capture_id")
    episode_id = _identifier(record["episode_id"], "episode_id")
    if record["profile"] != "Trainer" or record["split"] != "training":
        _fail("flight_capture_scope", "only Trainer training capture can enter behavior cloning")
    _identifier(record["scenario_id"], "scenario_id")
    if type(record["host_seed"]) is not int or record["host_seed"] < 0:
        _fail("flight_capture_seed", "host seed must be an explicit nonnegative integer")
    source = _validate_source(record["source"])
    if source != expected:
        _fail("flight_capture_source", "capture source differs from the declared host/profile/controller/teacher")
    source_sha = sha256_json(source)

    teacher = dict(_exact_mapping(record["teacher"], {"id", "sha256"}, "capture.teacher"))
    _identifier(teacher["id"], "capture.teacher.id")
    _digest(teacher["sha256"], "capture.teacher.sha256")
    if teacher["sha256"] != source["teacher_sha256"]:
        _fail("flight_capture_teacher", "teacher source digest differs from capture source binding")

    outcome = _validate_outcome(record["outcome"], "capture.outcome")
    transitions = record["transitions"]
    if not isinstance(transitions, list):
        _fail("flight_capture_shape", "transitions must be an ordered array")

    if record["initial_observation"] is None:
        if transitions or outcome["status"] not in ("failed", "refused", "inconclusive", "canceled"):
            _fail("flight_capture_initial", "missing initial observation is allowed only for retained non-completions")
        exclusions = ("episode_without_authoritative_initial_observation",)
        canonical = canonical_json(record)
        return ValidatedFlightCapture(
            canonical, hashlib.sha256(canonical.encode()).hexdigest(), declaration_sha256,
            capture_id, episode_id, source_sha, outcome["status"], (), exclusions)

    current = _validate_observation(record["initial_observation"], "capture.initial_observation")
    record["initial_observation"] = current
    if current["observed_tick"] != 0:
        _fail("flight_capture_initial", "capture must begin at the actual reset tick zero")

    samples: list[FlightSample] = []
    exclusions: list[str] = []
    uncertain = False
    saw_terminal = False
    for index, raw in enumerate(transitions):
        transition = dict(_exact_mapping(raw, {
            "index", "observation", "teacher_action", "physical",
            "next_observation", "outcome",
        }, f"capture.transitions[{index}]"))
        if transition["index"] != index:
            _fail("flight_capture_order", "transition indices must preserve original order")
        before = _validate_observation(transition["observation"], f"capture.transitions[{index}].observation")
        if sha256_json(before) != sha256_json(current):
            _fail("flight_capture_order", "transition source observation differs from preceding authoritative state")
        teacher_action = _validate_teacher_action(
            transition["teacher_action"], teacher, before, f"capture.transitions[{index}].teacher_action")
        physical = _validate_physical(transition["physical"], f"capture.transitions[{index}].physical")
        step_outcome = _validate_outcome(transition["outcome"], f"capture.transitions[{index}].outcome")
        terminal = step_outcome["terminated"] or step_outcome["truncated"] or step_outcome["failed"]
        if saw_terminal:
            _fail("flight_capture_terminal", "capture contains work after first terminal outcome")

        next_observation = transition["next_observation"]
        if physical["status"] == "confirmed":
            if next_observation is None:
                _fail("flight_capture_physical", "confirmed work requires its authoritative final observation")
            after = _validate_observation(next_observation, f"capture.transitions[{index}].next_observation")
            if not _same_lifetime(before, after):
                _fail("flight_capture_authority", "confirmed transition changed owner lifetime/control identity")
            if after["observation_id"] == before["observation_id"]:
                _fail("flight_capture_stale", "confirmed transition reused the source observation")
            if after["observed_tick"] != before["observed_tick"] + physical["consumed_ticks"]:
                _fail("flight_capture_ticks", "confirmed transition tick count differs from owner observations")
            if physical["consumed_ticks"] != 6 and not terminal:
                _fail("flight_capture_ticks", "short confirmed work is allowed only on a terminal interval")
            current = after
            transition["next_observation"] = after
        else:
            if next_observation is not None:
                _validate_observation(next_observation, f"capture.transitions[{index}].next_observation")
            uncertain = True
            exclusions.append(f"{index}:{physical['status']}_physical_consumption")

        if teacher_action["status"] != "accepted":
            exclusions.append(f"{index}:teacher_{teacher_action['status']}")
            if teacher_action["status"] == "unknown":
                uncertain = True
        if step_outcome["failed"]:
            exclusions.append(f"{index}:failed_transition")
        eligible = (
            not uncertain
            and teacher_action["status"] == "accepted"
            and physical["status"] == "confirmed"
            and not step_outcome["failed"]
        )
        if eligible:
            target = finite_vector(teacher_action["admitted_action"], 2)
            samples.append(FlightSample(
                policy_input(before), target, episode_id, index,
                physical["consumed_ticks"], terminal))
        elif not exclusions or not exclusions[-1].startswith(f"{index}:"):
            exclusions.append(f"{index}:not_supervised")

        transition["observation"] = before
        transition["teacher_action"] = teacher_action
        transition["physical"] = physical
        transition["outcome"] = step_outcome
        record["transitions"][index] = transition
        saw_terminal = terminal

    if transitions:
        last = record["transitions"][-1]["outcome"]
        for key in ("terminated", "truncated", "failed", "success"):
            if last[key] != outcome[key]:
                _fail("flight_capture_outcome", "episode outcome differs from its final transition")
    elif outcome["status"] == "completed":
        _fail("flight_capture_outcome", "completed episode must retain at least one transition")

    canonical = canonical_json(record)
    return ValidatedFlightCapture(
        canonical=canonical,
        record_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        declaration_sha256=declaration_sha256,
        capture_id=capture_id,
        episode_id=episode_id,
        source_sha256=source_sha,
        status=outcome["status"],
        samples=tuple(samples),
        exclusions=tuple(exclusions),
    )


def build_capture_dataset(episodes: Sequence[ValidatedFlightCapture]) -> FlightCaptureDataset:
    if not isinstance(episodes, Sequence) or not episodes:
        _fail("flight_capture_dataset", "at least one validated training episode is required")
    values = tuple(episodes)
    if len(values) > MAX_CAPTURE_EPISODES:
        _fail("flight_capture_budget", "capture exceeds the 120-episode design maximum")
    if any(not isinstance(item, ValidatedFlightCapture) for item in values):
        _fail("flight_capture_dataset", "dataset accepts validated capture records only")
    declaration = values[0].declaration_sha256
    capture_id = values[0].capture_id
    source_sha = values[0].source_sha256
    if any((item.declaration_sha256, item.capture_id, item.source_sha256) !=
           (declaration, capture_id, source_sha) for item in values):
        _fail("flight_capture_dataset", "capture/declaration/source identities cannot be mixed")
    ids = [item.episode_id for item in values]
    if len(ids) != len(set(ids)):
        _fail("flight_capture_dataset", "episode identity was recorded twice")
    samples = tuple(sample for episode in values for sample in episode.samples)
    if not samples:
        _fail("flight_capture_dataset", "no admitted supervised labels remain after validation")
    if len(samples) > MAX_CAPTURE_LABELS:
        _fail("flight_capture_budget", "capture exceeds the 19,200-label design maximum")
    identity = {
        "schema": "nca-4ly-capture-dataset/v1",
        "declaration_sha256": declaration,
        "capture_id": capture_id,
        "source_sha256": source_sha,
        "episode_sha256": [item.record_sha256 for item in values],
        "sample_sha256": [sample.sha256 for sample in samples],
        "episode_count": len(values),
        "label_count": len(samples),
    }
    return FlightCaptureDataset(
        declaration, capture_id, source_sha, values, samples, sha256_json(identity))


def _fresh_policy(seed: int):
    torch = _torch()
    if type(seed) is not int or seed < 0:
        _fail("flight_policy_seed", "policy seed must be a nonnegative integer")
    saved = torch.get_rng_state().clone()
    seeded = torch.Generator(device="cpu").manual_seed(seed).get_state()
    try:
        torch.set_rng_state(seeded)
        model = build_policy()
    finally:
        torch.set_rng_state(saved)
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        _fail("flight_policy_device", "compact policy must remain on CPU")
    return model


def policy_specification(model) -> dict[str, Any]:
    torch = _torch()
    parameters = list(model.parameters())
    if not parameters or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in parameters):
        _fail("flight_policy_device", "finite CPU float32 compact policy required")
    return {
        "schema": "nca-4ly-direct-policy/v1",
        "architecture": "54-64-64-2-tanh",
        "parameters": sum(p.numel() for p in parameters),
        "input_count": 54,
        "output_count": 2,
        "preprocessing_sha256": PREPROCESSING_SHA256,
        "action_sha256": ACTION_SHA256,
        "device": "cpu",
        "dtype": "float32",
    }


def policy_sha256(model) -> str:
    torch = _torch()
    digest = hashlib.sha256(canonical_json(policy_specification(model)).encode())
    for name, value in sorted(model.state_dict().items()):
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            _fail("flight_policy_state", "finite CPU float32 model state required")
        digest.update(name.encode())
        digest.update(canonical_json(list(value.shape)).encode())
        digest.update(bytes(value.detach().contiguous().reshape(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()


@dataclass
class FlightTrainingResult:
    model: Any
    optimizer: Any
    shuffle_rng: random.Random
    seed: int
    dataset: FlightCaptureDataset
    completed_epochs: int
    optimizer_updates: int
    sample_presentations: int
    epoch_losses: tuple[float, ...]


def train_behavior_clone(dataset: FlightCaptureDataset, *, seed: int) -> FlightTrainingResult:
    """Run only the fixed finite CPU BC schedule over already-validated samples."""
    if not isinstance(dataset, FlightCaptureDataset):
        _fail("flight_clone_dataset", "behavior cloning requires a validated capture dataset")
    if type(seed) is not int or seed not in POLICY_SEEDS:
        _fail("flight_clone_seed", "behavior cloning seed is not one of the three frozen policy seeds")
    samples = dataset.samples
    presentations = len(samples) * BC_EPOCHS
    updates = math.ceil(len(samples) / BC_BATCH_SIZE) * BC_EPOCHS
    if presentations > MAX_SAMPLE_PRESENTATIONS_PER_SEED or updates > MAX_OPTIMIZER_UPDATES_PER_SEED:
        _fail("flight_clone_budget", "fixed cloning schedule exceeds its prospective work ceiling")
    torch = _torch()
    model = _fresh_policy(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=BC_LEARNING_RATE)
    shuffle_rng = random.Random(seed ^ 0x4C59004)
    losses: list[float] = []
    completed_updates = 0
    completed_presentations = 0
    model.train()
    for _epoch in range(BC_EPOCHS):
        order = list(range(len(samples)))
        shuffle_rng.shuffle(order)
        total_loss = 0.0
        count = 0
        for start in range(0, len(order), BC_BATCH_SIZE):
            chosen = order[start:start + BC_BATCH_SIZE]
            features = torch.tensor(
                [samples[index].observation for index in chosen], dtype=torch.float32, device="cpu")
            labels = torch.tensor(
                [samples[index].action for index in chosen], dtype=torch.float32, device="cpu")
            prediction = model(features)
            loss = ((prediction - labels) ** 2).mean()
            if not bool(torch.isfinite(loss)):
                _fail("flight_clone_nonfinite", "nonfinite cloning objective cannot update the policy")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            completed_updates += 1
            completed_presentations += len(chosen)
            if (completed_updates > MAX_OPTIMIZER_UPDATES_PER_SEED
                    or completed_presentations > MAX_SAMPLE_PRESENTATIONS_PER_SEED):
                _fail("flight_clone_budget", "cloning work exceeded its fixed ceiling")
            total_loss += float(loss.detach()) * len(chosen)
            count += len(chosen)
        losses.append(total_loss / count)
    model.eval()
    return FlightTrainingResult(
        model, optimizer, shuffle_rng, seed, dataset, BC_EPOCHS,
        completed_updates, completed_presentations, tuple(losses))


def _cpu_copy(value: Any) -> Any:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        copied = value.detach().to("cpu").clone()
        if copied.is_floating_point() and not bool(torch.isfinite(copied).all()):
            _fail("flight_checkpoint_nonfinite", "checkpoint contains a nonfinite tensor")
        return copied
    if isinstance(value, Mapping):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    _fail("flight_checkpoint_state", "checkpoint contains an unsupported state value")


def _random_state_json(rng: random.Random) -> dict[str, Any]:
    version, words, gaussian = rng.getstate()
    return {"version": version, "words": list(words), "gaussian": gaussian}


def _restore_random_state(value: Any) -> random.Random:
    state = dict(_exact_mapping(value, {"version", "words", "gaussian"}, "checkpoint.shuffle_rng"))
    if type(state["version"]) is not int or not isinstance(state["words"], list):
        _fail("flight_checkpoint_rng", "shuffle RNG state is invalid")
    if any(type(item) is not int for item in state["words"]):
        _fail("flight_checkpoint_rng", "shuffle RNG words are invalid")
    if state["gaussian"] is not None and (
            type(state["gaussian"]) not in (int, float) or not math.isfinite(float(state["gaussian"]))):
        _fail("flight_checkpoint_rng", "shuffle RNG Gaussian cache is invalid")
    rng = random.Random()
    try:
        rng.setstate((state["version"], tuple(state["words"]), state["gaussian"]))
    except (TypeError, ValueError) as exc:
        raise ContractError("flight_checkpoint_rng", "shuffle RNG state cannot be restored") from exc
    return rng


def checkpoint_lineage(result: FlightTrainingResult, *, checkpoint_id: str,
                       learner_revision: str, learner_source_sha256: str,
                       parent_checkpoint_sha256: str | None = None) -> dict[str, Any]:
    _identifier(checkpoint_id, "checkpoint_id")
    _revision(learner_revision, "learner_revision")
    _digest(learner_source_sha256, "learner_source_sha256")
    if parent_checkpoint_sha256 is not None:
        _digest(parent_checkpoint_sha256, "parent_checkpoint_sha256")
    return {
        "schema": "nca-4ly-bc-lineage/v1",
        "checkpoint_id": checkpoint_id,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "declaration_sha256": result.dataset.declaration_sha256,
        "capture_id": result.dataset.capture_id,
        "capture_dataset_sha256": result.dataset.sha256,
        "capture_source_sha256": result.dataset.source_sha256,
        "policy_seed": result.seed,
        "bc_schedule_sha256": BC_SCHEDULE_SHA256,
        "preprocessing_sha256": PREPROCESSING_SHA256,
        "action_sha256": ACTION_SHA256,
        "completed_epochs": result.completed_epochs,
        "optimizer_updates": result.optimizer_updates,
        "sample_presentations": result.sample_presentations,
        "learner_revision": learner_revision,
        "learner_source_sha256": learner_source_sha256,
    }


def serialize_checkpoint(result: FlightTrainingResult, *, checkpoint_id: str,
                         learner_revision: str, learner_source_sha256: str,
                         parent_checkpoint_sha256: str | None = None) -> tuple[bytes, dict[str, Any]]:
    torch = _torch()
    lineage = checkpoint_lineage(
        result, checkpoint_id=checkpoint_id, learner_revision=learner_revision,
        learner_source_sha256=learner_source_sha256,
        parent_checkpoint_sha256=parent_checkpoint_sha256)
    model_hash = policy_sha256(result.model)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "lineage": lineage,
        "lineage_sha256": sha256_json(lineage),
        "policy_specification": policy_specification(result.model),
        "policy_sha256": model_hash,
        "model": _cpu_copy(result.model.state_dict()),
        "optimizer": _cpu_copy(result.optimizer.state_dict()),
        "rng": {
            "shuffle": _random_state_json(result.shuffle_rng),
            "torch_cpu": torch.get_rng_state().clone(),
            "cuda": "not_initialized_or_used_by_4ly_bc",
        },
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    data = stream.getvalue()
    if len(data) > MAX_CHECKPOINT_BYTES:
        _fail("flight_checkpoint_size", "checkpoint exceeds the 32 MiB preparation cap")
    digest = hashlib.sha256(data).hexdigest()
    receipt = {
        "schema": CHECKPOINT_RECEIPT_SCHEMA,
        "checkpoint_id": checkpoint_id,
        "sha256": digest,
        "bytes": len(data),
        "policy_sha256": model_hash,
        "lineage_sha256": payload["lineage_sha256"],
        "declaration_sha256": result.dataset.declaration_sha256,
        "capture_dataset_sha256": result.dataset.sha256,
    }
    return data, receipt


@dataclass
class RestoredFlightCheckpoint:
    model: Any
    optimizer: Any
    shuffle_rng: random.Random
    lineage: dict[str, Any]
    receipt: dict[str, Any]


def load_checkpoint_bytes(data: bytes, receipt: Mapping[str, Any]) -> RestoredFlightCheckpoint:
    if not isinstance(data, bytes) or not data or len(data) > MAX_CHECKPOINT_BYTES:
        _fail("flight_checkpoint_size", "checkpoint bytes are absent or exceed the preparation cap")
    expected = dict(_exact_mapping(receipt, {
        "schema", "checkpoint_id", "sha256", "bytes", "policy_sha256",
        "lineage_sha256", "declaration_sha256", "capture_dataset_sha256",
    }, "checkpoint.receipt"))
    if expected["schema"] != CHECKPOINT_RECEIPT_SCHEMA:
        _fail("flight_checkpoint_schema", "unknown checkpoint receipt schema")
    for key in ("sha256", "policy_sha256", "lineage_sha256", "declaration_sha256", "capture_dataset_sha256"):
        _digest(expected[key], f"checkpoint.receipt.{key}")
    _identifier(expected["checkpoint_id"], "checkpoint.receipt.checkpoint_id")
    if expected["bytes"] != len(data) or hashlib.sha256(data).hexdigest() != expected["sha256"]:
        _fail("flight_checkpoint_hash", "checkpoint bytes differ from their immutable receipt")
    torch = _torch()
    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ContractError("flight_checkpoint_decode", "checkpoint tensor payload could not be decoded") from exc
    _exact_mapping(payload, {
        "schema", "lineage", "lineage_sha256", "policy_specification", "policy_sha256",
        "model", "optimizer", "rng",
    }, "checkpoint.payload")
    if payload["schema"] != CHECKPOINT_SCHEMA:
        _fail("flight_checkpoint_schema", "unknown checkpoint schema")
    lineage = dict(payload["lineage"])
    if (payload["lineage_sha256"] != sha256_json(lineage)
            or payload["lineage_sha256"] != expected["lineage_sha256"]):
        _fail("flight_checkpoint_lineage", "checkpoint lineage changed")
    if (lineage.get("checkpoint_id") != expected["checkpoint_id"]
            or lineage.get("declaration_sha256") != expected["declaration_sha256"]
            or lineage.get("capture_dataset_sha256") != expected["capture_dataset_sha256"]):
        _fail("flight_checkpoint_lineage", "checkpoint receipt and lineage identities differ")
    if (lineage.get("bc_schedule_sha256") != BC_SCHEDULE_SHA256
            or lineage.get("preprocessing_sha256") != PREPROCESSING_SHA256
            or lineage.get("action_sha256") != ACTION_SHA256):
        _fail("flight_checkpoint_lineage", "checkpoint training/preprocessing/action contract differs")
    if lineage.get("policy_seed") not in POLICY_SEEDS:
        _fail("flight_checkpoint_lineage", "checkpoint policy seed is not declared")
    _cpu_copy(payload)
    model = _fresh_policy(0)
    if payload["policy_specification"] != policy_specification(model):
        _fail("flight_checkpoint_policy", "checkpoint architecture differs from the compact direct policy")
    current = model.state_dict()
    saved = payload["model"]
    if set(saved) != set(current):
        _fail("flight_checkpoint_policy", "checkpoint model fields differ")
    for name in current:
        value = saved[name]
        if (not isinstance(value, torch.Tensor) or value.shape != current[name].shape
                or value.dtype != current[name].dtype or value.device.type != "cpu"):
            _fail("flight_checkpoint_policy", f"checkpoint tensor {name} differs")
    model.load_state_dict(saved, strict=True)
    model.eval()
    if policy_sha256(model) != payload["policy_sha256"] or payload["policy_sha256"] != expected["policy_sha256"]:
        _fail("flight_checkpoint_policy", "checkpoint weights differ from their identity")
    optimizer = torch.optim.Adam(model.parameters(), lr=BC_LEARNING_RATE)
    try:
        optimizer.load_state_dict(payload["optimizer"])
    except (ValueError, RuntimeError, KeyError) as exc:
        raise ContractError("flight_checkpoint_optimizer", "optimizer state cannot be restored") from exc
    rng_payload = dict(_exact_mapping(payload["rng"], {"shuffle", "torch_cpu", "cuda"}, "checkpoint.rng"))
    if rng_payload["cuda"] != "not_initialized_or_used_by_4ly_bc":
        _fail("flight_checkpoint_rng", "checkpoint claims an unsupported CUDA lineage")
    cpu_rng = rng_payload["torch_cpu"]
    if not isinstance(cpu_rng, torch.Tensor) or cpu_rng.device.type != "cpu" or cpu_rng.dtype != torch.uint8:
        _fail("flight_checkpoint_rng", "Torch CPU RNG state is invalid")
    torch.Generator(device="cpu").set_state(cpu_rng)
    shuffle = _restore_random_state(rng_payload["shuffle"])
    return RestoredFlightCheckpoint(model, optimizer, shuffle, lineage, expected)


def _weight_export(model) -> dict[str, Any]:
    torch = _torch()
    output = {}
    for name, value in sorted(model.state_dict().items()):
        if (value.device.type != "cpu" or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())):
            _fail("flight_export_weights", "finite CPU float32 weights required")
        output[name] = {
            "dtype": "float32",
            "shape": list(value.shape),
            "values": value.detach().contiguous().view(-1).tolist(),
        }
    return output


def build_export_bundle(checkpoint: RestoredFlightCheckpoint) -> dict[str, Any]:
    """Produce an exact numeric bundle; this does not qualify any runtime."""
    if not isinstance(checkpoint, RestoredFlightCheckpoint):
        _fail("flight_export_checkpoint", "export requires a verified restored checkpoint")
    model_hash = policy_sha256(checkpoint.model)
    bundle = {
        "schema": EXPORT_SCHEMA,
        "policy_specification": policy_specification(checkpoint.model),
        "policy_sha256": model_hash,
        "checkpoint": {
            "id": checkpoint.receipt["checkpoint_id"],
            "sha256": checkpoint.receipt["sha256"],
            "lineage_sha256": checkpoint.receipt["lineage_sha256"],
        },
        "preprocessing": deepcopy(PREPROCESSING_SPEC),
        "preprocessing_sha256": PREPROCESSING_SHA256,
        "action": deepcopy(ACTION_SPEC),
        "action_sha256": ACTION_SHA256,
        "weights": _weight_export(checkpoint.model),
        "inference_contract": {
            "navigation_teacher_calls_allowed": False,
            "evaluator_state_in_policy_input": False,
            "host_normalization_reimplemented": False,
            "device": "cpu",
        },
        "qualification": {
            "trained_quality": False,
            "actual_host_parity": False,
            "realtime_qualified": False,
            "final_access": False,
        },
    }
    return {
        "schema": EXPORT_WRAPPER_SCHEMA,
        "bundle": bundle,
        "bundle_sha256": sha256_json(bundle),
    }


class ScoredFlightPolicy:
    """Direct learned inference surface.  It owns no teacher/oracle reference."""
    __slots__ = ("model", "policy_sha256")

    def __init__(self, model):
        self.model = model
        self.policy_sha256 = policy_sha256(model)

    def act(self, observation: Mapping[str, Any]) -> dict[str, list[float]]:
        torch = _torch()
        features = policy_input(observation)
        with torch.no_grad():
            outputs = self.model(torch.tensor([features], dtype=torch.float32, device="cpu"))[0]
        if not bool(torch.isfinite(outputs).all()):
            _fail("flight_inference_nonfinite", "scored policy produced a nonfinite action")
        return velocity_action(outputs.tolist())


def load_export_bundle(wrapper: Mapping[str, Any]) -> ScoredFlightPolicy:
    envelope = dict(_exact_mapping(wrapper, {"schema", "bundle", "bundle_sha256"}, "export.wrapper"))
    if envelope["schema"] != EXPORT_WRAPPER_SCHEMA:
        _fail("flight_export_schema", "unknown export wrapper schema")
    _digest(envelope["bundle_sha256"], "export.bundle_sha256")
    bundle = dict(_exact_mapping(envelope["bundle"], {
        "schema", "policy_specification", "policy_sha256", "checkpoint",
        "preprocessing", "preprocessing_sha256", "action", "action_sha256",
        "weights", "inference_contract", "qualification",
    }, "export.bundle"))
    if sha256_json(bundle) != envelope["bundle_sha256"]:
        _fail("flight_export_hash", "export bundle changed after publication")
    if bundle["schema"] != EXPORT_SCHEMA:
        _fail("flight_export_schema", "unknown export schema")
    _digest(bundle["policy_sha256"], "export.policy_sha256")
    if (bundle["preprocessing"] != PREPROCESSING_SPEC
            or bundle["preprocessing_sha256"] != PREPROCESSING_SHA256
            or bundle["action"] != ACTION_SPEC
            or bundle["action_sha256"] != ACTION_SHA256):
        _fail("flight_export_contract", "export preprocessing/action contract differs")
    expected_inference = {
        "navigation_teacher_calls_allowed": False,
        "evaluator_state_in_policy_input": False,
        "host_normalization_reimplemented": False,
        "device": "cpu",
    }
    expected_qualification = {
        "trained_quality": False,
        "actual_host_parity": False,
        "realtime_qualified": False,
        "final_access": False,
    }
    if bundle["inference_contract"] != expected_inference or bundle["qualification"] != expected_qualification:
        _fail("flight_export_contract", "export attempted to broaden inference authority or qualification")
    checkpoint = dict(_exact_mapping(bundle["checkpoint"], {"id", "sha256", "lineage_sha256"}, "export.checkpoint"))
    _identifier(checkpoint["id"], "export.checkpoint.id")
    _digest(checkpoint["sha256"], "export.checkpoint.sha256")
    _digest(checkpoint["lineage_sha256"], "export.checkpoint.lineage_sha256")
    torch = _torch()
    model = _fresh_policy(0)
    if bundle["policy_specification"] != policy_specification(model):
        _fail("flight_export_policy", "export architecture differs")
    state = model.state_dict()
    weights = bundle["weights"]
    if not isinstance(weights, Mapping) or set(weights) != set(state):
        _fail("flight_export_weights", "export weight fields differ")
    restored = {}
    for name, reference in state.items():
        item = dict(_exact_mapping(weights[name], {"dtype", "shape", "values"}, f"export.weights.{name}"))
        if item["dtype"] != "float32" or item["shape"] != list(reference.shape):
            _fail("flight_export_weights", f"export weight shape/dtype differs for {name}")
        values = item["values"]
        if not isinstance(values, list) or len(values) != reference.numel() or any(
                type(value) not in (int, float) or type(value) is bool or not math.isfinite(float(value))
                for value in values):
            _fail("flight_export_weights", f"export weight values are invalid for {name}")
        restored[name] = torch.tensor(values, dtype=torch.float32).reshape(reference.shape)
    model.load_state_dict(restored, strict=True)
    model.eval()
    if policy_sha256(model) != bundle["policy_sha256"]:
        _fail("flight_export_policy", "export numeric weights differ from policy identity")
    return ScoredFlightPolicy(model)
