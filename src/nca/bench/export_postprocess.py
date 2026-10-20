"""One fixed, zero-host completion of already captured T109 pose evidence."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from nca.contracts import ContractError, canonical_json, sha256_json, strict_json_loads
from .subprocess_capture import run_captured


DECLARATION_SHA256 = "035151338deb32ec69b7ec77be5b92942d2c2f64dd27e49928aaedfef65b6771"
FORBIDDEN_MODULES = ("torch", "nca.controllers.learned", "nca.train", "nca.learning", "nca.training",
                     "nca.cellular.training", "nca.export.torch_runtime")


def require(condition, code, detail):
    if not condition:
        raise ContractError(code, detail)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    data = Path(path).read_bytes()
    if Path(path).suffix == ".gz":
        data = gzip.decompress(data)
    return strict_json_loads(data, max_bytes=32 * 1024 * 1024)


def write_json(path, value):
    path = Path(path)
    data = (canonical_json(value) + "\n").encode()
    if path.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def load_declaration(path):
    result = read_json(path)
    require(sha256_json(result) == DECLARATION_SHA256, "postprocess_declaration", "fixed zero-host scope required")
    return result


def forbidden_imports():
    return sorted(name for name in sys.modules
                  if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_MODULES))


def _identity(observation):
    return tuple(observation[key] for key in ("run_id", "epoch", "actor_id"))


def _pose(observation):
    return {key: observation[key] for key in ("tick", "position", "velocity", "rotation", "angular_velocity")} | {
        "parts": [{key: part[key] for key in ("id", "index", "position", "rotation", "detached")}
                  for part in observation["parts"]]}


def read_exchange_prefix(path, maximum_serial):
    """The entire compressed file is hash-bound; only the required prefix is parsed."""
    rows, pending, next_serial = {}, None, 1
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = strict_json_loads(line, max_bytes=4 * 1024 * 1024)
            serial = row["serial"]
            if row["event"] == "request_reserved":
                require(pending is None and serial == next_serial, "postprocess_exchange", "ordered complete request prefix required")
                pending = row
            else:
                require(row["event"] == "response_received" and pending is not None and serial == pending["serial"],
                        "postprocess_exchange", "response must confirm the pending request")
                require(row["response"].get("ok") is True, "postprocess_exchange", "successful recorded owner response required")
                rows[serial] = {"request": pending, "response": row["response"]}
                pending, next_serial = None, next_serial + 1
                if serial == maximum_serial:
                    return rows
    raise ContractError("postprocess_exchange", "required profile prefix is incomplete")


def recover_variant(episode, binding, exchanges):
    """Supplement only recorded profile metadata; preserve every original value."""
    require("profile" not in episode and "profile_sha256" not in episode and "profile_provenance" not in episode,
            "postprocess_variant", "only the known missing-envelope shape is eligible")
    reset = exchanges[binding["reset_serial"]]
    profile_row = exchanges[binding["profile_serial"]]
    cancelled = exchanges[binding["cancel_serial"]]
    require(binding["profile_serial"] == binding["reset_serial"] + 1
            and binding["cancel_serial"] == binding["profile_serial"] + 1
            and binding["step_serials"] == list(range(binding["cancel_serial"] + 1, binding["step_serials"][-1] + 1)),
            "postprocess_variant", "uninterrupted reset, profile, cancel and step sequence required")
    require(reset["request"]["operation"] == "reset" and profile_row["request"]["operation"] == "profile"
            and profile_row["request"]["fields"] == {} and cancelled["request"]["operation"] == "cancel"
            and all(exchanges[serial]["request"]["mode"] == "stepped"
                    for serial in range(binding["reset_serial"], binding["step_serials"][-1] + 1)),
            "postprocess_variant", "captured stepped owner sequence differs")
    initial = reset["response"]["observation"]
    configuration = reset["request"]["fields"]["config"]
    profile = profile_row["response"]["profile"]
    require(configuration["actors"] == [{"model": binding["model"], "position": episode["case"]["spawn"], "producer": "policy"}]
            and configuration["max_ticks"] == 24 and configuration["seed"] == 73
            and reset["request"]["fields"]["actor_id"] == initial["actor_id"] == 0
            and initial["model"] == profile["id"] == binding["model"] and initial["tick"] == 0
            and sha256_json(profile) == binding["profile_sha256"],
            "postprocess_profile", "exact captured actor, profile and reset context required")
    require(cancelled["request"]["fields"] == {"run_id": initial["run_id"], "epoch": initial["epoch"]}
            and _identity(cancelled["response"]["observation"]) == _identity(initial)
            and _pose(cancelled["response"]["observation"]) == _pose(initial),
            "postprocess_variant", "recorded cancellation must belong to the same reset")
    traces = []
    for serial in binding["step_serials"]:
        row = exchanges[serial]
        require(row["request"]["operation"] == "step"
                and row["request"]["fields"] == {"commands": [], "ticks": binding["decision_ticks"]}
                and row["response"]["actual_ticks"] == binding["decision_ticks"]
                and row["response"]["idle_trace"] == [] and row["response"]["idle_trace_dropped"] == 0
                and row["response"]["unconsumed_terminal_commands"] == [],
                "postprocess_variant", "only the originally captured neutral refusal interval is eligible")
        traces.extend(row["response"]["traces"])
    require(episode["passed"] is True and episode["requested_actions"] == 0 and episode["actual_ticks"] == 24
            and len(traces) == 24 and traces == episode["trace"]
            and episode["frames"] == [_pose(initial)] + [_pose(row["observation"]) for row in traces],
            "postprocess_frames", "every original frame must be linked to its captured response")
    for index, row in enumerate(traces, 1):
        observation = row["observation"]
        require(row["tick_before"] == index - 1 and row["tick_after"] == index
                and _identity(observation) == _identity(initial) and observation["tick"] == index
                and observation["model"] == binding["model"] and row["requested"] == [] and row["admission"] == [],
                "postprocess_frames", "continuous same-actor recorded frames required")
    result = deepcopy(episode)
    result.update(profile=deepcopy(profile), profile_sha256=sha256_json(profile),
        profile_provenance={"source": "exact original owner exchange; no new host query",
            "reset_serial": binding["reset_serial"], "profile_serial": binding["profile_serial"],
            "cancel_serial": binding["cancel_serial"], "step_serials": binding["step_serials"],
            "reset_observation_id": initial["observation_id"], "run_id": initial["run_id"],
            "epoch": initial["epoch"], "actor_id": initial["actor_id"],
            "profile_response_sha256": sha256_json(profile_row["response"])})
    validate_frames(result, 24)
    return result


def validate_frames(episode, ticks):
    profile, frames = episode["profile"], episode["frames"]
    parts = profile["parts"]
    ids = [part["id"] for part in parts]
    require(bool(parts) and len(set(ids)) == len(ids) and profile["part_count"] == len(parts)
            and [frame["tick"] for frame in frames] == list(range(ticks + 1)),
            "postprocess_frames", "complete finite frame/profile correspondence required")
    for part in parts:
        require(len(part["size"]) == 2 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in part["size"])
                and len(part["color"]) >= 3 and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in part["color"]),
                "postprocess_profile", "finite authored size and color required")
    for frame in frames:
        require([part["id"] for part in frame["parts"]] == ids
                and [part["index"] for part in frame["parts"]] == list(range(len(parts))),
                "postprocess_frames", "no omitted, reordered or substituted rendered part")
        for part in frame["parts"]:
            require(len(part["position"]) == 2 and type(part["detached"]) is bool
                    and all(type(v) in (int, float) and math.isfinite(v) for v in (*part["position"], part["rotation"])),
                    "postprocess_frames", "finite recorded part pose required")


def verify_inputs(roots, declaration):
    for group, files in declaration["inputs"].items():
        root = roots[group]
        for name, expected in files.items():
            path = (root / name).resolve()
            require(path.is_relative_to(root) and file_hash(path) == expected,
                    "postprocess_input", f"immutable {group} input differs: {name}")


def validate_original(roots, declaration):
    original = declaration["original_runtime_source"]
    source = original["revision"]
    runtime = roots["runtime"]
    summary, manifest = read_json(runtime / "summary.json"), read_json(runtime / "manifest.json")
    unsigned = dict(summary)
    require(unsigned.pop("sha256") == sha256_json(unsigned), "postprocess_summary", "original summary self-hash differs")
    require(summary["status"] == "no_go" and summary["runtime_qualified"] is False
            and summary["numpy_runtime_qualified"] is False and summary["source_revision"] == source
            and manifest["source_revision"] == source and summary["training_updates"] == 0
            and summary["errors"] == [{"code": "execution_failure", "phase": "clips", "type": "CalledProcessError"}],
            "postprocess_gate", "only the exact retained clips-stage no-go is eligible")
    pinned = declaration["inputs"]["runtime"]
    require(summary["artifact_files"] == {name: value for name, value in pinned.items() if name != "summary.json"},
            "postprocess_summary", "complete original artifact index required")
    for key, filename in (("arithmetic", "arithmetic.json"), ("negative_contracts", "negative-contracts.json"),
                          ("realtime", "realtime.json.gz"), ("throughput", "throughput.json"), ("inference_cost", "inference-cost.json")):
        require(summary[key] == read_json(runtime / filename), "postprocess_gate", "summary and original gate receipt disagree")
    require(summary["arithmetic"]["passed"] is True and summary["arithmetic"]["boolean_actions_exact"] is True
            and all(value == 0 for value in summary["arithmetic"]["maximum_absolute_differences"].values()),
            "postprocess_gate", "original exact arithmetic gate required")
    trajectories = summary["trajectories"]
    require(trajectories["passed"] is True
            and [row["case"] for row in trajectories["cases"]] == declaration["renderer"]["pairs"][:2]
            and all(row["passed"] is True for row in trajectories["cases"]),
            "postprocess_gate", "both original paired trajectory gates required")
    negative = summary["negative_contracts"]
    require(negative["passed"] is True and negative["required_coverage_exact"] is True
            and len(negative["cases"]) == len({row["case"] for row in negative["cases"]}) == 16
            and all(row["passed"] is True for row in negative["cases"])
            and [row["case"] for row in summary["variants"]] == declaration["renderer"]["pairs"][2:]
            and all(row["passed"] is True for row in summary["variants"])
            and summary["realtime"]["passed"] is True and all(summary["realtime"]["checks"].values()),
            "postprocess_gate", "original negative, variant and realtime gates required")
    throughput = summary["throughput"]
    configurations = throughput["configurations"]
    setting = manifest["declaration"]["throughput"]
    require(throughput["passed"] is True and len(configurations) == 2
            and all(row["completed"] is True and row["configuration"] == config and row["active_window_seconds"] >= 300
                    and type(row["decisions"]) is int and row["decisions"] > 0
                    for row, config in zip(configurations, setting["actor_configurations"]))
            and sum(row["active_window_seconds"] for row in configurations) == throughput["active_window_seconds"] >= 600
            and throughput["phase_seconds"] <= 660 and sum(row["decisions"] for row in configurations) <= 20000
            and throughput["idle_padding_seconds"] == 0
            and throughput["reset_rpc"]["count"] == 10 and throughput["observe_rpc"]["count"] == 100,
            "postprocess_gate", "the complete original active-duration gate is required")
    inference = summary["inference_cost"]
    bundle_hash = pinned["bc-seed1103-torch-cpu.json"]
    require(inference["inference"]["count"] == 1000 and inference["learner_imported"] is False
            and inference["host_starts"] == inference["training_updates"] == 0
            and inference["bundle_file_sha256"] == summary["bundle_file_sha256"] == bundle_hash
            and inference["inputs_file_sha256"] == pinned["inference-inputs.json"],
            "postprocess_gate", "exact original standalone inference receipt required")
    work, limits = summary["work"], manifest["declaration"]["budget"]
    require(work["pending_requests"] == [] and work["unconfirmed_tail"] is False
            and work["host_terminations"] == work["host_start_attempts"] == 2
            and work["host_requests"] <= limits["maximum_host_requests"]
            and work["step_requests"] <= limits["maximum_step_requests"]
            and work["requested_step_ticks"] == work["confirmed_step_ticks"] <= limits["maximum_requested_step_ticks"]
            and work["wall_seconds"] <= limits["maximum_wall_seconds"],
            "postprocess_gate", "complete original work and cleanup bounds required")
    launch, completed = (read_json(roots["launcher"] / name) for name in ("launch.json", "completion.json"))
    require(all(row["source_revision"] == source and row["source_tree"] == original["tree"] for row in (launch, completed))
            and launch["source_clean_before"] is True and completed["source_unchanged"] is True
            and completed["host_source_unchanged"] is True and completed["child_termination_confirmed"] is True
            and completed["child_exit_code"] == 1 and completed["signals"] == []
            and completed["owned_members_after_cleanup"] == [] and completed["automatic_retries"] == 0
            and completed["wall_seconds"] == original["external_wall_seconds"] < 900,
            "postprocess_gate", "original external cleanup and aggregate wall evidence required")
    retained = {name: read_json(roots["retained"] / f"screen-v1/{name}.json") for name in ("summary", "resume", "learner-cost")}
    require(retained["summary"]["status"] == "no_go" and retained["summary"]["runtime_qualified"] is False
            and retained["resume"]["passed"] is True
            and declaration["inputs"]["retained"]["screen-v1/resume.json"] == manifest["declaration"]["original"]["resume_file_sha256"]
            and declaration["inputs"]["retained"]["screen-v1/learner-cost.json"] == manifest["declaration"]["original"]["learner_cost_file_sha256"],
            "postprocess_gate", "pinned passing fresh-episode resume and retained costs required")
    return {"original_physical_gates": True, "standalone_inference_count": 1000,
            "retained_fresh_episode_resume_passed": True, "retained_learner_cost_verified": True,
            "qualified_bundle_file_sha256": bundle_hash, "original_work": work}


def render_only(output):
    """The only subprocess work: unchanged pose renderer and decoded-file audit."""
    output = Path(output)
    request = read_json(output / "render-request.json")
    import PIL
    from PIL import Image
    require(platform.python_version() == request["python"] and PIL.__version__ == request["pillow"],
            "postprocess_renderer", "exact original renderer environment required")
    from nca.bench.pose_clip import render
    require(not forbidden_imports(), "postprocess_import", "Torch and learner imports are forbidden")
    result = render(output)
    spec = read_json(output / "clip-spec.json")
    require(result["complete"] is True and len(result["clips"]) == len(spec["pairs"]) == 4
            and result["camera"] == spec["camera"] and result["clip_spec_file_sha256"] == file_hash(output / "clip-spec.json"),
            "postprocess_clips", "all four unchanged camera pairs required")
    decoded = []
    width, height = spec["camera"]["panel_pixels"]
    for record, pair in zip(result["clips"], spec["pairs"]):
        expected_frames = len(range(0, pair["ticks"] + 1, spec["frame_stride"]))
        require(record["file"] == f"clips/{pair['id']}.gif" and record["labels"] == pair["labels"]
                and record["frames"] == expected_frames and record["file_sha256"] == file_hash(output / record["file"]),
                "postprocess_clips", "rendered pair receipt differs")
        with Image.open(output / record["file"]) as gif:
            require(gif.n_frames == expected_frames and gif.size == (2 * width + 32, height + 110),
                    "postprocess_clips", "decoded frame count or canvas differs")
            for index in range(gif.n_frames):
                gif.seek(index)
                gif.load()
                require(gif.info["duration"] == spec["frame_duration_ms"], "postprocess_clips", "decoded frame duration differs")
        decoded.append({"file": record["file"], "frames": expected_frames, "all_frames_decoded": True})
    require(not forbidden_imports(), "postprocess_import", "Torch and learner imports are forbidden")
    write_json(output / "render-audit.json", {"complete": True, "decoded": decoded,
        "python": platform.python_version(), "pillow": PIL.__version__, "forbidden_imports": [],
        "work": "unchanged pose rendering and file decoding only; no model construction, inference or host"})


def run_completion(*, root, declaration_path, runtime_root, launcher_root, retained_root,
                   output, source_revision, render_python, started=None):
    started = time.monotonic() if started is None else started
    root, output = Path(root).resolve(), Path(output).resolve()
    declaration = load_declaration(declaration_path)
    roots = {name: Path(path).resolve() for name, path in
             (("runtime", runtime_root), ("launcher", launcher_root), ("retained", retained_root))}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    require(revision == source_revision and not dirty, "postprocess_source", "clean explicitly published postprocessor source required")
    require(not any(output == base or output.is_relative_to(base) or base.is_relative_to(output) for base in roots.values())
            and not output.is_relative_to(root), "postprocess_output", "separate derived directory required")
    require(not forbidden_imports(), "postprocess_import", "no Torch or learner runtime allowed")
    output.mkdir(parents=True, exist_ok=False)
    source_names = list(declaration["unchanged_sources"]) + ["src/nca/bench/export_postprocess.py",
        "src/nca/bench/subprocess_capture.py", "tools/complete_export_clips.py", "configs/export/torch-cpu-clips-completion-v1.json"]
    sources = {name: file_hash(root / name) for name in source_names}
    record = {"format": "nca-export-clips-completion-result/v1", "status": "no_go", "composite_runtime_qualified": False,
        "original_status": "no_go", "original_runtime_qualified": False, "numpy_runtime_qualified": False,
        "original_runtime_source": declaration["original_runtime_source"],
        "postprocessor_source": {"revision": revision, "tree": tree}, "source_files": sources,
        "declaration_sha256": DECLARATION_SHA256, "input_file_sha256": declaration["inputs"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "errors": [],
        "work": {"host_starts": 0, "host_requests": 0, "new_simulation_ticks": 0, "policy_evaluations": 0,
                 "training_updates": 0, "automatic_retries": 0, "offline_invocations": 1}}
    write_json(output / "manifest.json", {key: value for key, value in record.items() if key not in ("status", "errors")})
    maximum = declaration["budget"]["maximum_offline_wall_seconds"]
    try:
        require(all(sources[name] == expected for name, expected in declaration["unchanged_sources"].items()),
                "postprocess_source", "original runtime and renderer source closure must remain unchanged")
        verify_inputs(roots, declaration)
        record["original_gate_audit"] = validate_original(roots, declaration)
        bindings = {binding["file"]: binding for binding in declaration["profile_recovery"]}
        exchanges = read_exchange_prefix(roots["runtime"] / "owner-exchanges.jsonl.gz",
                                         max(row["step_serials"][-1] for row in bindings.values()))
        spec = read_json(roots["runtime"] / "clip-spec.json")
        require([pair["id"] for pair in spec["pairs"]] == declaration["renderer"]["pairs"],
                "postprocess_clips", "all original pairs and ordering required")
        shutil.copyfile(roots["runtime"] / "clip-spec.json", output / "clip-spec.json")
        recovery = []
        for name in sorted({pair[side] for pair in spec["pairs"] for side in ("left", "right")}):
            episode = read_json(roots["runtime"] / name)
            if name in bindings:
                derived = recover_variant(episode, bindings[name], exchanges)
                write_json(output / name, derived)
                recovery.append({"file": name, "original_file_sha256": declaration["inputs"]["runtime"][name],
                                 "derived_file_sha256": file_hash(output / name), "profile_sha256": derived["profile_sha256"],
                                 "provenance": derived["profile_provenance"], "original_fields_unchanged": True})
            else:
                validate_frames(episode, 240)
                (output / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(roots["runtime"] / name, output / name)
        record["profile_recovery"] = recovery
        write_json(output / "profile-recovery.json", {"variants": recovery})
        write_json(output / "render-request.json", declaration["renderer"])
        remaining = maximum - (time.monotonic() - started) - 5
        require(remaining > 0, "postprocess_budget", "no render budget remains")
        environment = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
        run_captured([render_python, "-m", "nca.bench.export_postprocess", "--render-only", str(output)],
                     output=output, label="renderer", timeout=min(45, remaining), cwd=root, env=environment)
        record["clips"] = read_json(output / "clips.json")
        record["render_audit"] = read_json(output / "render-audit.json")
        nominal = [f"clips/{name}.gif" for name in declaration["renderer"]["pairs"][:2]]
        record["nominal_clips_byte_identical"] = all(file_hash(output / name) == declaration["inputs"]["runtime"][name] for name in nominal)
        require(record["nominal_clips_byte_identical"] and record["render_audit"]["complete"] is True
                and record["clips"]["complete"] is True, "postprocess_clips", "complete clips and exact nominal bytes required")
        record["composite_runtime_qualified"] = True
    except BaseException as exc:
        record["errors"].append({"type": type(exc).__name__, "code": exc.code if isinstance(exc, ContractError) else "postprocess_failure"})
        import traceback
        traceback.print_exc(file=sys.stderr)
    try:
        verify_inputs(roots, declaration)
        require(all(file_hash(root / name) == expected for name, expected in sources.items()),
                "postprocess_source", "source changed during postprocessing")
        require(not forbidden_imports(), "postprocess_import", "forbidden runtime imported")
    except (ContractError, OSError) as exc:
        record["errors"].append({"type": type(exc).__name__, "code": "postprocess_final_identity"})
    record["artifact_files"] = {str(path.relative_to(output)): file_hash(path) for path in output.rglob("*") if path.is_file()}
    elapsed = time.monotonic() - started
    original_wall = declaration["original_runtime_source"]["external_wall_seconds"]
    record["work"].update(offline_wall_seconds=elapsed, original_external_wall_seconds=original_wall,
        aggregate_observed_wall_seconds=original_wall + elapsed, aggregate_reserved_wall_seconds=original_wall + maximum,
        maximum_offline_wall_seconds=maximum, maximum_aggregate_wall_seconds=declaration["budget"]["maximum_aggregate_wall_seconds"],
        scope="postprocessor entry through final input/source/output checks; external watchdog receipt remains separate")
    if elapsed > maximum or original_wall + maximum >= declaration["budget"]["maximum_aggregate_wall_seconds"]:
        record["errors"].append({"code": "postprocess_budget"})
    record["composite_runtime_qualified"] = record["composite_runtime_qualified"] and not record["errors"]
    record["status"] = "complete_derived_evidence" if record["composite_runtime_qualified"] else "no_go"
    record["sha256"] = sha256_json(record)
    write_json(output / "summary.json", record)
    return record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", type=Path, required=True)
    render_only(parser.parse_args().render_only)
