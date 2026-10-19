"""One separately declared morphology comparison behind actual prerequisite evidence.

Importing or preparing this module constructs no model, opens no host and loads
no checkpoint. The original blocked admission remains immutable. A complete new
declaration and a passing original direction-frame artifact set are mandatory.
"""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
from itertools import product
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

from nca.contracts import ContractError, canonical_json, sha256_json
from .morphology_config import (MorphologyBudget, MorphologyBudgetExceeded, admit_direction_prerequisite,
    continuation_spec, file_receipt, read_json, read_prerequisite_selection, scenario, validate_continuation, verify_files)

FAMILIES = ("padded_recurrent", "shared_graph")
SEEDS = (6113, 6229, 6337)


def _error(exc):
    return {"type": type(exc).__name__, "code": exc.code if isinstance(exc, ContractError) else "execution_failed",
            "message": str(exc) if isinstance(exc, ContractError) else "declared operation did not complete"}


def _record(path: Path, value):
    """Atomic immutable JSON without importing the optional learner runtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical_json(value) + "\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".morphology-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def _sealed(value):
    return {**value, "sha256": sha256_json(value)}


def prepare_continuation(*, root: Path, prerequisite_directory: Path, output: Path):
    """Produce reviewable inputs only after actual unmodified evidence passes."""
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": "nca-morphology-continuation-preparation/v1", "admitted": False,
              "status": "evidence_invalid", "host_starts": 0, "model_constructions": 0,
              "checkpoint_loads": 0, "policy_calls": 0, "optimizer_updates": 0, "error": None,
              "original_blocked_admission_changed": False}
    try:
        selection = read_prerequisite_selection(root)
        result["prerequisite_selection"] = {"sha256": sha256_json(selection), "primary": selection["primary"]}
        result["sources"] = _source_inputs(root, None)
        prerequisite = admit_direction_prerequisite(prerequisite_directory, root)
        declaration = continuation_spec(root, prerequisite)
        _record(output / "proposed-declaration.json", declaration)
        result.update(status="ready_for_public_declaration", prerequisite=prerequisite,
                      declaration_sha256=sha256_json(declaration))
    except (ContractError, OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        result["status"] = "training_blocked" if getattr(exc, "code", "") == "morphology_prerequisite_failed" else "evidence_invalid"
        result["error"] = _error(exc)
    result = _sealed(result)
    _record(output / "admission.json", result)
    return result


def _source_inputs(root: Path, declaration_path: Path | None):
    from nca.data.capture import source_identity
    revision, package_sources, digest = source_identity(root)
    if subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=root, text=True).strip():
        raise ContractError("morphology_source", "the declared public source must be clean")
    files = dict(package_sources)
    additional = ("tools/morphology_compare.py", "configs/morphology/comparison-v1.json",
                  "configs/morphology/profiles-v1.json", "configs/morphology/entry-gate-v1.json",
                  "configs/morphology/prerequisite-selection-v1.json",
                  "configs/first-skill/jolt-direction-frame-v1.json", "configs/first-skill/jolt-screen-v1.json",
                  "docs/evidence/t106-prerequisite-result.json")
    paths = [root / name for name in additional]
    if declaration_path is not None:
        paths.append(declaration_path)
    for path in paths:
        name = path.resolve().relative_to(root.resolve()).as_posix()
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        subprocess.run(["git", "ls-files", "--error-unmatch", "--", *files], cwd=root, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for name, expected in files.items():
            blob = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root, stderr=subprocess.DEVNULL)
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ContractError("morphology_source", "executed bytes must equal the original public Git blobs")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("morphology_source", "all executed source and declaration files must be tracked at the freeze") from exc
    return {"revision": revision, "source_files": package_sources, "learner_sources_sha256": digest,
            "files": files, "sha256": sha256_json(files)}


def run_comparison(*, root: Path, prerequisite_directory: Path, declaration_path: Path,
                   checkout: Path, godot: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    budget = MorphologyBudget()
    state = {"status": "incomplete", "phase": "prerequisite_admission", "error": None,
             "prerequisite_selection": None,
             "manifest": None, "source": None, "declaration": None, "qualification": None,
             "host_start_attempts": 0, "model_construction_attempts": 0, "checkpoint_load_attempts": 0,
             "host_closed": False, "results": [], "capture_attempts": [], "fitting_attempts": [],
             "checkpoints": [], "learning": [], "completed_optimizer_updates": 0, "evaluation": None}
    try:
        selection = read_prerequisite_selection(root)
        state["prerequisite_selection"] = {"sha256": sha256_json(selection), "primary": selection["primary"]}
        declaration = read_json(declaration_path)
        validate_continuation(root, declaration, prerequisite_directory)
        budget.check()
        source = _source_inputs(root, declaration_path)
        state.update(declaration=declaration, source=source)
        _record(output / "declaration.json", declaration)
        _record(output / "published-inputs.json", source)
        _record(output / "admission.json", _sealed({
            "schema": "nca-morphology-continuation-admission/v1", "admitted": True,
            "prerequisite_selection": state["prerequisite_selection"],
            "prerequisite": declaration["prerequisite"], "declaration_sha256": sha256_json(declaration),
            "learner_revision": source["revision"], "published_inputs_sha256": source["sha256"],
            "host_starts": 0, "model_constructions": 0, "checkpoint_loads": 0,
            "policy_calls": 0, "optimizer_updates": 0, "original_blocked_admission_changed": False}))
        # The actual evidence gate and public execution closure precede these
        # optional imports, model construction and the sole host launch.
        _execute(root, checkout, godot, output, declaration, source, budget, state)
        budget.check()
        if _source_inputs(root, declaration_path) != source:
            raise ContractError("morphology_source", "executed source changed during the comparison")
        if admit_direction_prerequisite(prerequisite_directory, root) != declaration["prerequisite"]:
            raise ContractError("morphology_prerequisite", "original prerequisite bytes changed during the comparison")
        state.update(status="completed", phase="complete")
    except BaseException as exc:
        state["error"] = _error(exc)
        state["status"] = "capped" if isinstance(exc, MorphologyBudgetExceeded) else "failed"
        if state["declaration"] is None:
            state["status"] = "training_blocked" if getattr(exc, "code", "") == "morphology_prerequisite_failed" else "evidence_invalid"
            _record(output / "admission.json", _sealed({
                "schema": "nca-morphology-continuation-admission/v1", "admitted": False,
                "prerequisite_selection": state["prerequisite_selection"],
                "status": state["status"], "error": state["error"], "host_starts": 0,
                "model_constructions": 0, "checkpoint_loads": 0, "policy_calls": 0,
                "optimizer_updates": 0, "original_blocked_admission_changed": False}))
    recorded = sum(row["decisions"] for row in state["results"])
    if state["declaration"] is not None:
        try:
            from .morphology_task import summarize_comparison
            state["evaluation"] = summarize_comparison(state["results"], state["declaration"],
                                                         run_completed=state["status"] == "completed")
        except BaseException as exc:
            state.update(status="failed", error=_error(exc))
            state["evaluation"] = {"complete": False, "passed": False, "error": _error(exc)}
    artifacts = [file_receipt(path, output) for path in sorted(output.rglob("*"))
                 if path.is_file() and not path.name.startswith(".morphology-")]
    receipt = budget.receipt()
    if (receipt["wall_seconds"] > 1800 or any(seconds > 900 for seconds in receipt["candidate_wall_seconds"].values())
            or budget.requests != recorded or not state["host_closed"] and state["host_start_attempts"]):
        if state["status"] == "completed":
            state.update(status="capped" if budget.remaining_seconds() <= 0 else "failed",
                         error={"type": "IncompleteWork", "code": "morphology_work_receipts",
                                "message": "the declared ceiling, actual request accounting or host closure did not complete"})
            from .morphology_task import summarize_comparison
            state["evaluation"] = summarize_comparison(state["results"], state["declaration"], run_completed=False)
            receipt = budget.receipt()
    summary = _sealed({"schema": "nca-morphology-comparison-result/v1", "status": state["status"],
        "prerequisite_selection": state["prerequisite_selection"],
        "phase": state["phase"], "error": state["error"], "declaration_sha256": None if state["declaration"] is None else sha256_json(state["declaration"]),
        "manifest_sha256": None if state["manifest"] is None else state["manifest"]["sha256"],
        "learner_revision": None if state["source"] is None else state["source"]["revision"],
        "learner_sources_sha256": None if state["source"] is None else state["source"]["learner_sources_sha256"],
        **receipt, "recorded_game_requests": recorded, "unconfirmed_game_requests": max(0, budget.requests - recorded),
        "confirmed_physical_ticks": sum(row["completed_ticks"] for row in state["results"]),
        "unknown_tick_counts": int(budget.requests != recorded), "host_start_attempts": state["host_start_attempts"],
        "host_closed": state["host_closed"], "host_termination": state.get("host_termination"),
        "model_construction_attempts": state["model_construction_attempts"],
        "checkpoint_load_attempts": state["checkpoint_load_attempts"],
        "completed_optimizer_updates": state["completed_optimizer_updates"],
        "qualification": state["qualification"], "results": state["results"], "evaluation": state["evaluation"],
        "capture_attempts": state["capture_attempts"], "fitting_attempts": state["fitting_attempts"],
        "learning": state["learning"], "checkpoints": state["checkpoints"], "artifacts": artifacts,
        "abrupt_loss_limit": "without this completed summary, retained prefixes have an unknown request/update tail",
        "original_blocked_admission_changed": False, "automatic_extension": False,
        "owner_feel_accepted": False, "unsupported_geometry_damage_or_assist_sweeps": "unmeasured"})
    _record(output / "summary.json", summary)
    return summary


def _execute(root, checkout, godot, output, declaration, source, budget, state):
    import torch
    import numpy
    from nca.adapters.godot import GodotHost
    from .morphology_artifacts import checkpoint_file, load_checkpoint, write_checkpoint
    from .morphology_capture import capture_waypoint
    from .morphology_learning import new_policy, policy_sha256, policy_specification, ppo_update, train_clone
    from .morphology_task import evaluate_qualification

    budget.check()
    if (str(torch.__version__) != declaration["runtime"]["torch"]
            or numpy.__version__ != declaration["runtime"]["numpy"]
            or platform.python_version() != declaration["runtime"]["python"]):
        raise ContractError("morphology_runtime", "the declared original CPU Torch/NumPy runtime is required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    comparison = read_json(root / "configs/morphology/comparison-v1.json")
    profiles = read_json(root / "configs/morphology/profiles-v1.json")
    config_hash = sha256_json(declaration)
    manifest = _sealed({"schema": "nca-morphology-comparison-manifest/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "declaration": declaration,
        "config_sha256": config_hash, "learner_revision": source["revision"],
        "source_files": source["source_files"], "learner_sources_sha256": source["learner_sources_sha256"],
        "published_inputs_sha256": source["sha256"], "profile_fixture_sha256": sha256_json(profiles),
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": numpy.__version__,
                    "platform": platform.system(), "machine": platform.machine(), "device": "cpu", "torch_threads": 1},
        "training_privilege": "none", "heldout_profiles_opened": False})
    state["manifest"] = manifest
    _record(output / "manifest.json", manifest)
    common_metadata = {"config_sha256": config_hash, "manifest_sha256": manifest["sha256"],
        "learner_revision": source["revision"], "learner_sources_sha256": source["learner_sources_sha256"],
        "training_privilege": "none"}
    initial, finals, adapted = {}, {}, {}
    captured_files, created_files = [], []

    def record(path, value):
        _record(path, value)
        created_files.append(file_receipt(path, output))

    @contextmanager
    def owned_host():
        budget.check()
        host = GodotHost(checkout, godot, lane="act004", mode="stepped", timeout=30)
        try:
            yield host
        finally:
            try:
                host.close()
            finally:
                process = host._process
                state["host_closed"] = process is not None and process.poll() is not None
                state["host_termination"] = {"termination_confirmed": state["host_closed"],
                                             "exit_code": None if process is None else process.poll()}
                record(output / "host-termination.json", state["host_termination"])

    def save(model, family, seed, phase, dataset_hash, *, suffix, optimizer=None, extra=None, training_state=None):
        budget.check()
        metadata = {**common_metadata, "family": family, "training_seed": seed,
                    "phase": phase, "dataset_sha256": dataset_hash, **(extra or {})}
        receipt = write_checkpoint(output, f"checkpoints/{family}-seed{seed}-{suffix}.pt", model, metadata,
                                   optimizer=optimizer, training_state=training_state)
        state["checkpoints"].append(receipt)
        record(output / "checkpoint-receipts" / f"{family}-seed{seed}-{suffix}.json", receipt)
        return receipt

    def load(receipt):
        budget.check()
        state["checkpoint_load_attempts"] += 1
        state["model_construction_attempts"] += 1
        return load_checkpoint(output, receipt, expected_metadata=receipt["metadata"])[0]

    def fit(model, family, seed, stage, operation):
        attempt = {"id": f"{family}-seed{seed}-{stage}", "family": family, "training_seed": seed,
                   "stage": stage, "status": "started", "completed_optimizer_updates": 0,
                   "initial_policy_sha256": policy_sha256(model)}
        state["fitting_attempts"].append(attempt)
        record(output / "fit-starts" / f"{attempt['id']}.json", attempt)
        started = time.monotonic()
        def completed(count):
            increment = count - attempt["completed_optimizer_updates"]
            if increment != 1:
                raise ContractError("morphology_optimizer_receipt", "completed-step accounting must advance exactly once")
            attempt["completed_optimizer_updates"] = count
            state["completed_optimizer_updates"] += 1
            if count % 64 == 0:
                record(output / "fit-progress" / f"{attempt['id']}-{count:06}.json", {
                    "id": attempt["id"], "completed_optimizer_updates": count,
                    "policy_sha256": policy_sha256(model), "elapsed_seconds": time.monotonic() - started,
                    "final": False, "subsequent_update_tail": "unknown until a complete outcome receipt"})
        try:
            result = operation(completed)
            if (result["optimizer_steps"] != attempt["completed_optimizer_updates"]
                    or result["optimizer_steps"] <= 0
                    or stage == "bc" and result["epochs"] != 40
                    or stage != "bc" and (result["actor_samples"] != 128 or result["excluded_actor_samples"] != 0
                                           or result["epochs"] != 4)):
                raise ContractError("morphology_optimizer_receipt", "the complete fixed source-attributed update allocation is required")
            attempt.update(status="completed", final_policy_sha256=policy_sha256(model))
            recorded = {name: value for name, value in result.items() if name != "optimizer"}
            state["learning"].append({"id": attempt["id"], **recorded})
            return result
        except BaseException as exc:
            attempt.update(status="capped" if isinstance(exc, MorphologyBudgetExceeded) else "failed", error=_error(exc))
            raise
        finally:
            attempt["wall_seconds"] = time.monotonic() - started
            record(output / "fit-outcomes" / f"{attempt['id']}.json", attempt)

    budget.check()
    state.update(phase="host_startup", host_start_attempts=1)
    with owned_host() as host:
        if (host.identity["host_revision"] != comparison["host"]["revision"]
                or host.identity["bridge_sha256"] != comparison["host"]["bridge_sha256"]):
            raise ContractError("morphology_host", "the unchanged accepted host and bridge are required")
        record(output / "environment.json", {"host_identity": host.identity, "manifest_sha256": manifest["sha256"]})

        def collect(case, *, phase, kind, model=None, training_seed=None, suffix="", maximum=40, training=False):
            budget.check()
            identity = kind if training_seed is None else f"{kind}-seed{training_seed}"
            episode_id = f"{phase}-{identity}-{case['id']}{suffix}"
            policy_id = ("waypoint-reference-v1" if kind == "reference" else "waypoint-idle-v1" if kind == "idle" else
                         f"{identity}-{'initial' if phase == 'zero_shot_initial' else 'adapted-' + case['profile'] if phase == 'adapted_evaluation' else 'shared_final' if phase in ('zero_shot_final', 'tuning') else phase}")
            attempt = {"episode_id": episode_id, "phase": phase, "kind": kind,
                       "training_seed": training_seed, "profile": case["profile"], "host_seed": case["host_seed"],
                       "status": "started", "request_reservations_before": budget.requests}
            state["capture_attempts"].append(attempt)
            record(output / "capture-starts" / f"{episode_id}.json", attempt)
            try:
                episode, row, rollout = capture_waypoint(host, case, episode_id=episode_id,
                    split="training" if phase in ("teacher", "shared_training", "adaptation") else "tuning" if phase == "tuning" else "evaluation",
                    phase=phase, policy_id=policy_id, kind=kind, model=model, seed=case["host_seed"],
                    sample_seed=case["host_seed"] + (training_seed or 0), training_seed=training_seed,
                    budget=budget, output=output, profiles=profiles, root=root, maximum_decisions=maximum, training=training)
                state["results"].append(row)
                for path in (output / row["artifact"], output / "metrics" / f"{episode_id}.json",
                             output / "boundary-audit" / f"{episode_id}.json"):
                    captured_files.append(file_receipt(path, output))
                if (episode.learner_revision != source["revision"]
                        or episode.learner_sources_sha256 != source["learner_sources_sha256"]
                        or dict(episode.source_files) != source["source_files"]
                        or row["error"] is not None or row["source_boundary_regressions"] != 0
                        or row["actual_owner_closed"] is not True):
                    raise ContractError("morphology_capture_source", "actual owner, policy input and frozen source must remain linked")
                attempt.update(status="completed", decisions=row["decisions"], completed_ticks=row["completed_ticks"],
                               record_sha256=row["record_sha256"])
                return episode, row, rollout
            except BaseException as exc:
                result = getattr(exc, "result", None)
                if result is not None and not any(row["episode_id"] == result["episode_id"] for row in state["results"]):
                    state["results"].append(result)
                attempt.update(status="capped" if isinstance(exc, MorphologyBudgetExceeded) else "failed", error=_error(exc))
                raise
            finally:
                attempt["request_reservations_after"] = budget.requests
                record(output / "capture-outcomes" / f"{episode_id}.json", attempt)

        state["phase"] = "waypoint_reference_qualification"
        qualification_rows = [collect(case, phase="qualification", kind="reference")[1] for case in declaration["qualification"]]
        state["qualification"] = evaluate_qualification(qualification_rows)
        record(output / "qualification.json", _sealed(state["qualification"]))
        if state["qualification"]["passed"] is not True:
            raise ContractError("morphology_teacher_unqualified", "every separately declared waypoint reference case must pass before collection")

        state["phase"] = "teacher_collection"
        demos = []
        for profile, family, host_seed in product(comparison["splits"]["training_profiles"],
                                                comparison["splits"]["training_families"],
                                                comparison["future_learning"]["teacher_host_seeds"]):
            case = scenario(profiles, profile, family, host_seed)
            episode, _, _ = collect(case, phase="teacher", kind="reference", suffix=f"-seed{host_seed}")
            if episode.status != "completed" or len(list(episode.imitation_samples())) != episode.decisions:
                raise ContractError("morphology_teacher", "all complete source-attributed teacher trajectories are required")
            demos.append(episode)
        dataset = _sealed({"id": "t106-morphology-waypoint-teacher-v1", "episodes": [
            {"id": episode.id, "record_sha256": episode.sha256, "profile_sha256": episode.profile_sha256,
             "decisions": episode.decisions} for episode in demos],
            "selection": "all sixteen complete eligible teacher trajectories including unsuccessful outcomes; qualification excluded",
            "training_profiles": comparison["splits"]["training_profiles"],
            "training_families": comparison["splits"]["training_families"], "teacher_host_seeds": comparison["future_learning"]["teacher_host_seeds"]})
        record(output / "dataset.json", dataset)
        sequences = []
        for episode in demos:
            samples = list(episode.imitation_samples())
            sequences.append(([frame for frame, _ in samples], [action for _, action in samples]))

        def quota(model, family, seed, profile, geometry, *, phase, count, base_seed, prefix):
            rows, remaining, repeat = [], count, 0
            original = policy_sha256(model)
            while remaining:
                case = scenario(profiles, profile, geometry, base_seed + repeat)
                _, row, rollout = collect(case, phase=phase, kind=family, model=model, training_seed=seed,
                    suffix=f"-{prefix}-repeat{repeat}", maximum=min(40, remaining), training=True)
                if (rollout is None or not 0 < row["decisions"] <= remaining
                        or len(rollout.transitions) != row["decisions"] or policy_sha256(model) != original):
                    raise ContractError("morphology_collection", "actual fixed-policy transitions must exactly consume the allocated quota")
                rows.append(rollout)
                remaining -= row["decisions"]
                repeat += 1
            return rows

        training_cases = tuple(product(comparison["splits"]["training_profiles"], comparison["splits"]["training_families"]))
        state["phase"] = "shared_fitting"
        for family, seed in product(FAMILIES, SEEDS):
            with budget.candidate(family, seed):
                budget.check()
                state["model_construction_attempts"] += 1
                model = new_policy(family, seed)
                if policy_specification(model)["parameters"] != comparison["candidates"][family]["parameters"]:
                    raise ContractError("morphology_architecture", "the unchanged parameter allocation is required")
                initial[family, seed] = save(model, family, seed, "initial", dataset["sha256"], suffix="initial",
                    training_state={"next_episode_boundary": True, "completed_optimizer_updates": 0})
                cloned = fit(model, family, seed, "bc", lambda completed: train_clone(model, sequences, seed=seed,
                    epochs=40, check_budget=budget.check, on_update=completed))
                bc_optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                bc_optimizer.load_state_dict(cloned["optimizer"])
                save(model, family, seed, "bc", dataset["sha256"], suffix="bc", optimizer=bc_optimizer,
                     training_state={"next_episode_boundary": True, "epochs_completed": 40})
                optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
                for update in range(6):
                    rollouts = []
                    for case_index, (profile, geometry) in enumerate(training_cases):
                        base_seed = 81000000 + SEEDS.index(seed) * 10000 + update * 1000 + case_index * 100
                        rollouts.extend(quota(model, family, seed, profile, geometry, phase="shared_training", count=16,
                                              base_seed=base_seed, prefix=f"update{update}-case{case_index}"))
                    fit(model, family, seed, f"shared-update{update}", lambda completed: ppo_update(
                        model, optimizer, rollouts, seed=seed + update, check_budget=budget.check, on_update=completed))
                    save(model, family, seed, "shared_update", dataset["sha256"], suffix=f"shared-update{update}",
                         optimizer=optimizer, extra={"completed_shared_updates": update + 1},
                         training_state={"next_episode_boundary": True, "completed_shared_decisions": (update + 1) * 128})
                finals[family, seed] = save(model, family, seed, "shared_final", dataset["sha256"], suffix="shared-final",
                    optimizer=optimizer, extra={"completed_shared_updates": 6},
                    training_state={"next_episode_boundary": True, "completed_shared_decisions": 768})

        for receipt in (*initial.values(), *finals.values()):
            checkpoint_file(output, receipt)
        frozen = _sealed({"schema": "nca-morphology-shared-freeze/v1", "manifest_sha256": manifest["sha256"],
                          "initial": list(initial.values()), "shared_final": list(finals.values()),
                          "heldout_profiles_opened": False, "selection": "fixed final update for every declared seed"})
        record(output / "frozen-shared-candidates.json", frozen)
        state["phase"] = "tuning_diagnostic"
        for family, seed in product(FAMILIES, SEEDS):
            with budget.candidate(family, seed):
                model = load(finals[family, seed])
                for case in declaration["tuning"]:
                    collect(case, phase="tuning", kind=family, model=model, training_seed=seed)
                checkpoint_file(output, finals[family, seed])
        record(output / "zero-shot-opened.json", {"manifest_sha256": manifest["sha256"],
            "frozen_shared_candidates_sha256": frozen["sha256"], "all_six_initial_and_final_checkpoints_frozen": True})
        state["phase"] = "zero_shot_evaluation"
        for kind in ("reference", "idle"):
            for case in declaration["zero_shot"]:
                collect(case, phase="authored_and_idle", kind=kind)
        for phase, receipts in (("zero_shot_initial", initial), ("zero_shot_final", finals)):
            for family, seed in product(FAMILIES, SEEDS):
                with budget.candidate(family, seed):
                    model = load(receipts[family, seed])
                    for case in declaration["zero_shot"]:
                        collect(case, phase=phase, kind=family, model=model, training_seed=seed)
                    checkpoint_file(output, receipts[family, seed])

        state["phase"] = "independent_adaptation"
        for family, seed, profile in product(FAMILIES, SEEDS, comparison["splits"]["zero_shot_profiles"]):
            with budget.candidate(family, seed):
                model = load(finals[family, seed])
                optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
                rollouts = []
                before = len(state["results"])
                started = time.monotonic()
                for case_index, geometry in enumerate(comparison["splits"]["adaptation_collection_families"]):
                    base_seed = 81200000 + SEEDS.index(seed) * 10000 + comparison["splits"]["zero_shot_profiles"].index(profile) * 1000 + case_index * 100
                    rollouts.extend(quota(model, family, seed, profile, geometry, phase="adaptation", count=64,
                                          base_seed=base_seed, prefix=f"adapt-{profile}-case{case_index}"))
                fit(model, family, seed, f"adapt-{profile}", lambda completed: ppo_update(model, optimizer, rollouts,
                    seed=seed + 100 + comparison["splits"]["zero_shot_profiles"].index(profile),
                    check_budget=budget.check, on_update=completed))
                adaptation_dataset = _sealed({"profile": profile, "starting_checkpoint": finals[family, seed],
                    "episodes": [{"id": row["episode_id"], "record_sha256": row["record_sha256"]} for row in state["results"][before:]],
                    "actual_decisions": sum(row["decisions"] for row in state["results"][before:]),
                    "wall_seconds": time.monotonic() - started, "fresh_optimizer_hidden_and_data": True})
                record(output / "adaptation-datasets" / f"{family}-seed{seed}-{profile}.json", adaptation_dataset)
                adapted[family, seed, profile] = save(model, family, seed, "adapted_final", adaptation_dataset["sha256"],
                    suffix=f"adapted-{profile}", optimizer=optimizer,
                    extra={"adaptation_profile": profile, "shared_starting_policy_sha256": finals[family, seed]["policy_sha256"]},
                    training_state={"next_episode_boundary": True, "completed_adaptation_decisions": 128})
                checkpoint_file(output, finals[family, seed])
        for receipt in adapted.values():
            checkpoint_file(output, receipt)
        adapted_freeze = _sealed({"schema": "nca-morphology-adapted-freeze/v1", "manifest_sha256": manifest["sha256"],
                                  "adapted": list(adapted.values()), "adapted_evaluation_opened": False})
        record(output / "frozen-adapted-candidates.json", adapted_freeze)
        record(output / "adapted-evaluation-opened.json", {"manifest_sha256": manifest["sha256"],
            "frozen_adapted_candidates_sha256": adapted_freeze["sha256"], "all_twelve_adapted_checkpoint_hashes_frozen": True})
        state["phase"] = "adapted_evaluation"
        for family, seed, profile in product(FAMILIES, SEEDS, comparison["splits"]["zero_shot_profiles"]):
            with budget.candidate(family, seed):
                model = load(adapted[family, seed, profile])
                for case in declaration["adapted_evaluation"]:
                    if case["profile"] == profile:
                        collect(case, phase="adapted_evaluation", kind=family, model=model, training_seed=seed)
                checkpoint_file(output, adapted[family, seed, profile])
        for receipt in state["checkpoints"]:
            checkpoint_file(output, receipt)
        verify_files(output, captured_files)
        verify_files(output, created_files)
        for row in state["results"]:
            if file_receipt(output / row["artifact"], output)["sha256"] != row["artifact_sha256"]:
                raise ContractError("morphology_artifact", "an original captured episode changed")
        state["phase"] = "host_closure"
    if not state["host_closed"]:
        raise ContractError("morphology_host_closure", "the owned host termination must be confirmed")
    verify_files(output, created_files)
