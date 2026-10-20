"""One declared authored-teacher diagnostic; no learner or checkpoint imports.

All cases are already-open development cases. This comparison cannot admit a
learned skill, select a checkpoint, or change the morphology prerequisite.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping
import zlib

from nca.adapters.godot import CybersandEnv, GodotHost, inspect_host, physical_config
from nca.contracts import ContractError, canonical_json, sha256_json, strict_json_loads
from nca.controllers.baseline import AuthoredController, FeedbackObservation, SkillGoal
from nca.controllers.morphology import MorphologyAdapter
from nca.controllers.morphology.conformance import cancellation_receipt, validate_interval
from nca.data import ActionSource, EpisodeRecorder, PolicyContext, write_episode
from nca.data.capture import source_identity
from nca.runs import RunBudget
from nca.train.task import HopTask


DECLARATION_SHA256 = "269cef7a852fe8d690bb95ab9187576a3babe0f1fa5d1e8ee28fafc2730c474a"
FAMILIES = ("authored-balanced-v1", "jolt-arrival-v1")


def write_json(path: Path, value: Mapping[str, Any]) -> str:
    """Exclusive, durable publication; an existing result is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode()
    with path.open("xb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    return hashlib.sha256(payload).hexdigest()


def read_declaration(path: Path):
    value = strict_json_loads(path.read_bytes())
    if sha256_json(value) != DECLARATION_SHA256:
        raise ContractError("arrival_declaration", "the frozen prospective diagnostic declaration is required")
    return value


def diagnostic_cases(original):
    """Retain each of the original24 training and8 evaluation case/seed pairs."""
    cases = []
    for source in original["training_families"]:
        for index, seed in enumerate(source["demo_seeds"]):
            cases.append({"id": f"demo-{source['id']}-{index}", "family": source["id"],
                          "origin": "previously_open_training", "spawn": source["spawn"],
                          "target": source["target"], "host_seed": seed})
    for source in original["evaluation_families"]:
        cases.append({"id": f"reference-{source['id']}", "family": source["id"],
                      "origin": "previously_open_evaluation", "spawn": source["spawn"],
                      "target": source["target"], "host_seed": source["host_seed"]})
    if len(cases) != 32 or len({row["id"] for row in cases}) != 32:
        raise ContractError("arrival_cases", "exactly32 distinct previously open cases are required")
    return cases


class DiagnosticBudget:
    """Reserve all owner calls before transmission; uncertainty consumes budget."""
    def __init__(self, limits, journal, *, clock=time.monotonic):
        self.limits, self.journal, self.clock = limits, journal, clock
        self.started = clock()
        self.counts = {"host_requests": 0, "step_requests": 0, "requested_step_ticks": 0,
                       "confirmed_step_ticks": 0, "episodes_started": 0,
                       "host_start_attempts": 0, "host_terminations": 0}
        self.pending = {}

    def note(self, event, **values):
        self.journal.write((canonical_json({"event": event,
            "elapsed_seconds": self.clock() - self.started, **values}) + "\n").encode())
        self.journal.flush()

    def check(self):
        if self.clock() - self.started >= self.limits["maximum_wall_seconds"]:
            raise ContractError("arrival_wall_budget", "declared complete invocation ceiling reached")

    def start_episode(self, case, family):
        self.check()
        if self.counts["episodes_started"] >= self.limits["maximum_episodes"]:
            raise ContractError("arrival_episode_budget", "declared episode allocation exhausted")
        self.counts["episodes_started"] += 1
        self.note("episode_started", case=case, family=family)

    def reserve(self, operation, fields):
        if operation not in ("cancel", "close"):
            self.check()
        if self.counts["host_requests"] >= self.limits["maximum_host_requests"]:
            raise ContractError("arrival_owner_budget", "declared owner-request allocation exhausted")
        ticks = fields.get("ticks", 0) if operation == "step" else 0
        if operation == "step":
            if (type(ticks) is not int or ticks != 6
                    or self.counts["step_requests"] >= self.limits["maximum_decisions"]
                    or self.counts["requested_step_ticks"] + ticks > self.limits["maximum_requested_ticks"]):
                raise ContractError("arrival_step_budget", "declared six-tick decision allocation exhausted")
            self.counts["step_requests"] += 1
            self.counts["requested_step_ticks"] += ticks
        self.counts["host_requests"] += 1
        serial = self.counts["host_requests"]
        self.pending[serial] = {"serial": serial, "operation": operation, "requested_ticks": ticks}
        self.note("request_reserved", **self.pending[serial], fields=fields)
        return serial

    def confirm(self, serial, response):
        pending = self.pending[serial]
        self.note("response_received", serial=serial, response=response)
        if pending["operation"] == "step":
            actual = response.get("actual_ticks")
            if type(actual) is not int or not 0 <= actual <= pending["requested_ticks"]:
                raise ContractError("arrival_tick_receipt", "native actual tick receipt is malformed")
            self.counts["confirmed_step_ticks"] += actual
        del self.pending[serial]

    def snapshot(self):
        return {**self.counts, "pending_requests": list(self.pending.values()),
                "unconfirmed_tail": bool(self.pending), "wall_seconds": self.clock() - self.started,
                "optimizer_updates": 0, "checkpoint_executions": 0, "learned_policy_calls": 0}


class MeteredHost:
    def __init__(self, host, budget):
        self.host, self.budget = host, budget

    def __getattr__(self, key):
        return getattr(self.host, key)

    def request(self, operation, **fields):
        serial = self.budget.reserve(operation, fields)
        response = self.host.request(operation, **fields)
        self.budget.confirm(serial, response)
        return response


def validate_closed_owner_journal(path: Path, *, expected_counts, expected_rows):
    """Replay the closed gzip completely before trusting its in-memory summary.

    This is an offline accounting check, not a repair or a host invocation. A
    valid gzip footer alone is insufficient: every declared episode, request,
    native response and final close must reconcile with the retained results.
    """
    def require(condition, code, detail):
        if not condition:
            raise ContractError(f"arrival_journal_{code}", detail)

    counters = ("host_requests", "step_requests", "requested_step_ticks",
                "confirmed_step_ticks", "episodes_started", "host_start_attempts",
                "host_terminations")
    require(isinstance(expected_counts, Mapping) and all(
        type(expected_counts.get(key)) is int and expected_counts[key] >= 0 for key in counters),
        "counts", "complete nonnegative expected counters are required")
    require(expected_counts.get("pending_requests") == []
        and expected_counts.get("unconfirmed_tail") is False
        and all(type(expected_counts.get(key)) is int and expected_counts[key] == 0
                for key in ("optimizer_updates", "checkpoint_executions", "learned_policy_calls")),
        "counts", "pending or learned activity cannot support diagnostic completion")
    wall = expected_counts.get("wall_seconds")
    require(type(wall) in (int, float) and math.isfinite(wall) and wall >= 0,
            "counts", "a finite expected accounting interval is required")
    require(expected_rows and all(isinstance(row, Mapping) for row in expected_rows),
            "episodes", "the complete expected episode rows are required")
    keys = [(row.get("case_id"), row.get("family")) for row in expected_rows]
    require(all(type(case) is str and case and family in FAMILIES for case, family in keys)
        and len(keys) == len(set(keys)) and len(keys) == expected_counts["episodes_started"],
        "episodes", "unique expected case/family rows must match episode accounting")
    counts = dict.fromkeys(counters, 0)
    active, pending, native_close, host_closed = None, None, False, False
    completed, records, last_elapsed = 0, 0, 0.
    try:
        with gzip.open(path, "rb") as source:
            while True:
                # Read through EOF, including the gzip CRC/length trailer. A
                # streaming prefix must never be mistaken for a closed file.
                line = source.readline(1_048_577)
                if not line:
                    break
                records += 1
                require(len(line) <= 1_048_576 and line.endswith(b"\n"),
                        "record", "bounded complete JSONL records are required")
                try:
                    record = strict_json_loads(line)
                except ContractError as exc:
                    raise ContractError("arrival_journal_record", "invalid journal JSON") from exc
                require(isinstance(record, dict), "record", "journal records must be objects")
                elapsed, event = record.get("elapsed_seconds"), record.get("event")
                require(type(elapsed) in (int, float) and math.isfinite(elapsed)
                    and last_elapsed <= elapsed <= wall and not host_closed,
                    "order", "records must be chronological and precede final host closure")
                last_elapsed = elapsed
                if event == "host_start_reserved":
                    require(records == 1, "order", "exactly one initial host start is required")
                    counts["host_start_attempts"] += 1
                elif event == "episode_started":
                    key = (record.get("case"), record.get("family"))
                    require(counts["host_start_attempts"] == 1 and active is None
                        and pending is None and not native_close and completed < len(keys)
                        and key == keys[completed],
                        "episodes", "episodes must start once in the declared result order")
                    counts["episodes_started"] += 1
                    active = {"key": key, "stage": "reset", "decisions": 0, "ticks": 0}
                elif event == "request_reserved":
                    serial, operation = record.get("serial"), record.get("operation")
                    fields, ticks = record.get("fields"), record.get("requested_ticks")
                    require(counts["host_start_attempts"] == 1 and pending is None and not native_close
                        and type(serial) is int and serial == counts["host_requests"] + 1
                        and type(operation) is str and isinstance(fields, dict) and type(ticks) is int,
                        "order", "each synchronous request needs its next unique reservation")
                    if operation == "close":
                        require(active is None and completed == len(keys) and fields == {} and ticks == 0,
                                "close", "native close must follow every completed episode")
                    else:
                        stages = {"reset": ("reset",), "profile": ("profile",), "step": ("step",),
                                  "cancel": ("step", "terminal")}
                        require(operation in stages and active is not None
                            and active["stage"] in stages[operation],
                            "order", "episode owner operations are missing or out of order")
                        require(ticks == (6 if operation == "step" else 0)
                            and (operation != "step" or type(fields.get("ticks")) is int
                                 and fields["ticks"] == 6),
                            "counts", "each step must reserve the declared six physical ticks")
                        if operation == "step":
                            counts["step_requests"] += 1
                            counts["requested_step_ticks"] += ticks
                        if operation == "cancel":
                            require(active["decisions"] > 0, "close", "episode closure needs a physical interval")
                    counts["host_requests"] += 1
                    pending = {"serial": serial, "operation": operation, "requested_ticks": ticks}
                elif event == "response_received":
                    response, serial = record.get("response"), record.get("serial")
                    require(pending is not None and type(serial) is int and serial == pending["serial"],
                            "order", "a native response must match the sole pending reservation")
                    require(isinstance(response, dict) and response.get("protocol") == "nca-host/1"
                        and type(response.get("id")) is int and response["id"] == serial
                        and response.get("ok") is True,
                        "response", "a successful correlated native response is required")
                    operation = pending["operation"]
                    if operation == "close":
                        require(response.get("closed") is True, "close", "native host close was not confirmed")
                        native_close = True
                    elif operation == "profile":
                        require(isinstance(response.get("profile"), dict), "response", "native profile response is missing")
                        active["stage"] = "step"
                    else:
                        view = response.get("observation")
                        require(isinstance(view, dict) and type(view.get("tick")) is int,
                                "response", "native observation and physical tick are required")
                        if operation == "reset":
                            require(view["tick"] == 0 and view.get("suspended") is False,
                                    "response", "an active native reset observation is required")
                            active["stage"] = "profile"
                            active["view"] = view
                        elif operation == "cancel":
                            require(response.get("cancelled") is True
                                and view["tick"] == active["ticks"],
                                "close", "native episode cancellation was not confirmed")
                            try:
                                cancellation_receipt(active["view"], view)
                            except (ContractError, KeyError, TypeError) as exc:
                                raise ContractError("arrival_journal_close", "native cancellation must revoke ownership without stepping") from exc
                            active["stage"] = "closed"
                        else:
                            actual, start, end = (response.get(key) for key in ("actual_ticks", "start_tick", "tick"))
                            traces = response.get("traces")
                            require(type(actual) is int and actual == pending["requested_ticks"] == 6
                                and type(start) is int and start == active["ticks"]
                                and type(end) is int and end == start + actual and view["tick"] == end
                                and isinstance(traces, list) and len(traces) == actual
                                and all(isinstance(trace, dict)
                                    and type(trace.get("tick_before")) is int and trace["tick_before"] == start + index
                                    and type(trace.get("tick_after")) is int and trace["tick_after"] == start + index + 1
                                    for index, trace in enumerate(traces)),
                                "response", "complete contiguous native six-tick receipts are required")
                            require(all(type(response.get(key)) is bool for key in ("terminated", "truncated", "finished")),
                                    "response", "native terminal status is missing")
                            terminal = response["terminated"] or response["truncated"]
                            require(response["finished"] is terminal and view.get("suspended") is terminal,
                                    "close", "native terminal status and ownership closure must agree")
                            counts["confirmed_step_ticks"] += actual
                            active["decisions"] += 1
                            active["ticks"] = end
                            active["view"] = view
                            if terminal:
                                active["stage"] = "terminal"
                    pending = None
                elif event == "episode_completed":
                    row = record.get("row")
                    require(active is not None and pending is None and active["stage"] in ("terminal", "closed")
                        and (record.get("case"), record.get("family")) == active["key"]
                        and isinstance(row, dict) and sha256_json(row) == sha256_json(expected_rows[completed])
                        and row.get("owner_closure_confirmed") is True and row.get("error") is None
                        and type(row.get("decisions")) is int and row["decisions"] == active["decisions"]
                        and type(row.get("consumed_ticks")) is int and row["consumed_ticks"] == active["ticks"],
                        "episodes", "each completed row must match its native intervals and closure")
                    completed += 1
                    active = None
                elif event == "host_closed":
                    require(native_close and active is None and pending is None and completed == len(keys)
                        and record.get("termination_confirmed") is True,
                        "close", "a final confirmed host termination must follow its native close response")
                    counts["host_terminations"] += 1
                    host_closed = True
                else:
                    require(False, "record", "unknown or missing journal event")
    except (EOFError, OSError, zlib.error) as exc:
        raise ContractError("arrival_journal_stream", "the complete gzip stream and trailer are required") from exc
    require(host_closed and native_close and pending is None and active is None and completed == len(keys),
            "incomplete", "the journal has a missing episode, response or final native closure")
    require(counts == {key: expected_counts[key] for key in counters}
        and counts["host_start_attempts"] == counts["host_terminations"] == 1
        and counts["requested_step_ticks"] == counts["confirmed_step_ticks"],
        "counts", "replayed owner accounting differs from the expected complete counters")
    return {"verified": True, "records": records, "counts": counts,
            "episodes_completed": completed, "native_close_confirmed": True, "gzip_eof_verified": True}


def _error(exc):
    # Exception text and engine diagnostics can contain local paths. Preserve
    # class/code publicly and retain detailed execution logs only locally.
    return {"type": type(exc).__name__, "code": getattr(exc, "code", None)}


def validate_owner_interval(adapter, source, following, *, terminal):
    """Terminal suspension must carry exactly one native ownership revocation."""
    before = adapter.identity(source)
    after = adapter.terminal_identity(following) if terminal else adapter.identity(following)
    if (after.lifetime != before.lifetime
            or after.control_generation != before.control_generation + int(terminal)
            or after.topology_sha256 != before.topology_sha256
            or source.get("mount_generation") != following.get("mount_generation")
            or following.get("tick") != source.get("tick") + 6):
        raise ContractError("arrival_owner_interval", "exact source/physical identity and native terminal revocation are required")


def _capture(host, budget, case, family, output, original, declaration, root):
    from nca.controllers.jolt_arrival import JoltArrivalController
    budget.start_episode(case["id"], family)
    identifier = f"{family}-{case['id']}"
    settings = physical_config("jev", ticks=240, seed=case["host_seed"])
    settings["actors"][0]["position"] = list(case["spawn"])
    env = CybersandEnv(host, settings, decision_ticks=6, lease_ticks=12)
    controller = AuthoredController("jev") if family == FAMILIES[0] else JoltArrivalController()
    policy_hash = (original["baseline_source_sha256"] if family == FAMILIES[0]
                   else controller.specification_sha256)
    goal = SkillGoal(identifier, "directed_hop", tuple(case["target"]), .8, 4., 4.)
    source = ActionSource(family, "scripted", controller.controller_id, policy_hash)
    context = PolicyContext(goal.id, goal.skill, goal.target, goal.pace, goal.tolerance, goal.timeout_seconds)
    recorder = EpisodeRecorder(env, identifier, source, context=context, seed=case["host_seed"],
        scenario_family=case["family"], split="evaluation", budget=RunBudget(40, 600), source_root=root)
    if sha256_json(recorder.profile) != original["profile_sha256"]:
        raise ContractError("arrival_profile", "the declared accepted Jolt profile changed")
    adapter = MorphologyAdapter(recorder.profile, expected_sha256=original["profile_sha256"])
    task = HopTask(goal, recorder.initial, deadline_ticks=240, quiet_ticks_required=12)
    phases, terms, error, closure = [], [], None, False
    try:
        for _ in range(40):
            budget.check()
            view = recorder.view
            started = time.perf_counter_ns()
            frame = controller.step(FeedbackObservation.from_host(view), goal)
            inference_ns = time.perf_counter_ns() - started
            phases.append({"tick": view["tick"], "phase": frame.phase, "status": frame.status,
                           "reason": frame.reason, "action": frame.action.to_dict()})
            after, _, terminated, truncated, info = recorder.step(frame.action.to_dict())
            event = recorder.events[-1]
            validate_interval(event.requested_packet, event.source_observation, after,
                              terminated, truncated, info, allow_terminal=True)
            validate_owner_interval(adapter, event.source_observation, after,
                                    terminal=terminated or truncated)
            if not event.imitation_eligible or not event.trace_complete:
                raise ContractError("arrival_consumption", "complete unchanged admitted action consumption is required")
            terms.append(task.consume(frame.action, info["trace"], inference_ns=inference_ns))
            if task.integrity_failure:
                raise ContractError("arrival_integrity", "a native refusal or physical integrity failure occurred")
            if task.done:
                break
            if terminated or truncated:
                raise ContractError("arrival_early_terminal", "host stopped before the unchanged task outcome")
        if not task.done:
            raise ContractError("arrival_episode_capped", "fixed episode ended without an intrinsic outcome")
        if task.deadline_missed:
            closure = bool(recorder.view.get("suspended"))
        else:
            before = recorder.view
            canceled = recorder.cancel("diagnostic_task_arrival")
            cancellation_receipt(before, canceled)
            closure = True
        if not closure:
            raise ContractError("arrival_owner_closure", "native owner cancellation or terminal closure was not confirmed")
    except Exception as exc:
        error = _error(exc)
        try:
            recorder.cancel("diagnostic_error_cleanup")
        except Exception:
            pass
    finally:
        controller.cancel()
    metrics = task.metrics(administratively_capped=not task.done)
    evaluation = {"diagnostic_only": True, "previous_case_origin": case["origin"],
        "task_reward_schema": "jolt-arrival-best-progress/v1", "metrics": metrics,
        "reward_terms": terms, "controller_frames": phases, "error": error,
        "owner_closure_confirmed": closure, "declaration_sha256": sha256_json(declaration)}
    episode = recorder.finish("failed" if error else "completed", metrics["reason"], evaluation=evaluation)
    path = output / "episodes" / f"{identifier}.json.gz"
    artifact_sha = write_episode(path, episode)
    row = {"case_id": case["id"], "family": family, "origin": case["origin"],
        "host_seed": case["host_seed"], "profile_sha256": episode.profile_sha256,
        "record_sha256": episode.sha256, "artifact": path.relative_to(output).as_posix(),
        "artifact_sha256": artifact_sha, "decisions": episode.decisions,
        "consumed_ticks": episode.consumed_ticks, "metrics": metrics,
        "owner_closure_confirmed": closure, "error": error}
    write_json(output / "metrics" / f"{identifier}.json", row)
    budget.note("episode_completed", case=case["id"], family=family, row=row)
    return row


def summarize(rows, cases):
    expected = {(case["id"], family) for case in cases for family in FAMILIES}
    keys = [(row["case_id"], row["family"]) for row in rows]
    if len(keys) != len(set(keys)) or not set(keys).issubset(expected):
        raise ContractError("arrival_pairing", "duplicate or foreign diagnostic outcomes")
    valid = (set(keys) == expected and all(row["error"] is None and row["owner_closure_confirmed"]
        and not row["metrics"]["administratively_capped"] and not row["metrics"]["integrity_failure"]
        and row["metrics"]["failed_admissions"] == 0 for row in rows))
    metrics = ("overshoot_cells", "final_distance_cells", "capped_completion_seconds",
        "peak_toward_speed_cells_per_second", "mean_absolute_angle_radians",
        "ordinary_trim_reversals_per_second", "purposeful_launch", "primary_edges")
    families = {}
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        n = len(selected)
        if not n:
            continue
        if any(type(row["metrics"].get(name)) not in (float, int, bool)
               or not math.isfinite(row["metrics"][name]) for row in selected for name in metrics):
            raise ContractError("arrival_metrics", "finite complete recorded metrics are required")
        families[family] = {"episodes": n, "successes": sum(row["metrics"]["success"] for row in selected),
            "mean": {name: sum(row["metrics"][name] for row in selected) / n for name in metrics},
            "failed_cases": [row["case_id"] for row in selected if not row["metrics"]["success"]]}
        for origin in ("previously_open_training", "previously_open_evaluation"):
            partition = [row for row in selected if row["origin"] == origin]
            families[family][origin] = {"episodes": len(partition),
                "successes": sum(row["metrics"]["success"] for row in partition)}
    return {"complete_valid_pairing": valid, "families": families,
            "interpretation": "descriptive paired development diagnostic; no sealed learned-policy inference",
            "morphology_admitted": False, "learned_skill_qualified": False, "owner_feel_accepted": False}


def run_diagnostic(*, root: Path, checkout: Path, godot: Path, declaration_path: Path,
                   output: Path, expected_revision: str):
    """Single source-frozen invocation; stop on uncertain consumption or cost."""
    declaration = read_declaration(declaration_path)
    from nca.controllers.jolt_arrival import JoltArrivalController
    candidate = JoltArrivalController()
    if candidate.specification() != declaration["candidate_specification"]:
        raise ContractError("arrival_candidate", "candidate specification differs from its declaration")
    original_path = root / declaration["case_source"]["path"]
    original = strict_json_loads(original_path.read_bytes())
    if sha256_json(original) != declaration["case_source"]["sha256"]:
        raise ContractError("arrival_case_source", "original open case/configuration bytes changed")
    cases = diagnostic_cases(original)
    output.mkdir(parents=True, exist_ok=False)
    rows, failure, identity, manifest, host = [], None, None, None, None
    with gzip.open(output / "owner-journal.jsonl.gz", "xb") as journal:
        budget = DiagnosticBudget(declaration["budget"], journal)
        try:
            revision, files, _ = source_identity(root)
            if revision != expected_revision or len(revision) != 40:
                raise ContractError("arrival_revision", "the explicitly published source revision is required")
            if subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=root).strip():
                raise ContractError("arrival_dirty_source", "the complete published checkout must be clean")
            for path in (declaration_path, original_path, root / "tools/jolt_arrival.py"):
                files[path.resolve().relative_to(root.resolve()).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            for relative, expected in files.items():
                committed = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=root,
                                                   stderr=subprocess.DEVNULL)
                if hashlib.sha256(committed).hexdigest() != expected:
                    raise ContractError("arrival_unpublished_source", "every input must match its published Git blob")
            for relative, expected in declaration["preserved_sources"].items():
                if files.get(relative) != expected:
                    raise ContractError("arrival_changed_dependency", "a preserved original dependency changed")
            identity = inspect_host(checkout, godot)
            if identity["host_revision"] != original["host_sha"] or identity["bridge_sha256"] != original["bridge_sha256"]:
                raise ContractError("arrival_host_identity", "the accepted physical host and unchanged bridge are required")
            manifest = {"schema": "nca-jolt-arrival-diagnostic/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_revision": revision, "source_files": files, "source_files_sha256": sha256_json(files),
                "declaration": declaration, "declaration_sha256": sha256_json(declaration),
                "host_identity": identity, "cases": cases, "python": platform.python_version(),
                "platform": platform.system(), "architecture": platform.machine(), "device": "cpu",
                "training": False, "evaluation_cases_previously_open": True}
            write_json(output / "manifest.json", manifest)
            budget.check()
            budget.counts["host_start_attempts"] += 1
            budget.note("host_start_reserved")
            host = GodotHost(checkout, godot, lane="act004", mode="stepped", timeout=15.)
            metered = MeteredHost(host, budget)
            for case in cases:
                for family in FAMILIES:
                    row = _capture(metered, budget, case, family, output, original, declaration, root)
                    rows.append(row)
                    print(canonical_json({"case": case["id"], "family": family,
                        "success": row["metrics"]["success"], "error": row["error"]}), flush=True)
                    if row["error"] is not None:
                        raise ContractError("arrival_episode_failed", "diagnostic retained an unsuccessful execution boundary")
            budget.check()
        except BaseException as exc:
            failure = _error(exc)
        finally:
            if host is not None:
                try:
                    if host._process is not None and host._process.poll() is None:
                        host._channel.connection.settimeout(2.)
                        MeteredHost(host, budget).request("close")
                except Exception as exc:
                    failure = failure or _error(exc)
                finally:
                    host.close(graceful=False)
                    closed = host._process is None or host._process.poll() is not None
                    budget.counts["host_terminations"] += int(closed)
                    budget.note("host_closed", termination_confirmed=closed)
            if manifest is not None:
                drift = any(hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected
                            for relative, expected in manifest["source_files"].items())
                if drift:
                    failure = failure or {"type": "SourceDrift", "code": "arrival_source_changed"}
                try:
                    if inspect_host(checkout, godot) != identity:
                        raise ContractError("arrival_host_drift", "verified host/runtime bytes changed during the diagnostic")
                except Exception as exc:
                    failure = failure or _error(exc)
            measured = budget.snapshot()
    try:
        journal_validation = validate_closed_owner_journal(output / "owner-journal.jsonl.gz",
            expected_counts=measured, expected_rows=rows)
    except Exception as exc:
        journal_validation = {"verified": False, "error": _error(exc)}
        failure = failure or _error(exc)
    comparison = summarize(rows, cases)
    completed = (failure is None and journal_validation["verified"]
        and comparison["complete_valid_pairing"] and not measured["unconfirmed_tail"]
        and measured["host_start_attempts"] == measured["host_terminations"] == 1
        and measured["requested_step_ticks"] == measured["confirmed_step_ticks"]
        and measured["wall_seconds"] <= declaration["budget"]["maximum_wall_seconds"])
    artifacts = {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(output.rglob("*")) if path.is_file()}
    result = {"schema": "nca-jolt-arrival-diagnostic-result/v1", "status": "completed" if completed else "incomplete",
        "source_revision": expected_revision, "declaration_sha256": sha256_json(declaration),
        "counts": measured, "comparison": comparison, "rows": rows, "error": failure,
        "artifact_sha256": artifacts, "host_identity": identity, "owner_journal_validation": journal_validation,
        "evidence_limit": "already-open authored development cases; unchanged learned prerequisite remains blocked"}
    write_json(output / "summary.json", result)
    return result
