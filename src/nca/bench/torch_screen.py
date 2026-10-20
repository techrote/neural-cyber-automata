"""Single prospectively declared Torch CPU follow-up with active ten-minute timing."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import asdict, replace
import gzip
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from nca.contracts import ContractError, canonical_json, sha256_json, strict_json_loads
from nca.export import ExportController, ExportDecision, RuntimeBundle, context_for, dispatch
from nca.export.build import immutable_json, read_json
from nca.export.torch_runtime import TorchRuntimeBundle, BACKEND, make_torch_bundle
from .common import durations, pose, process_memory, wire_difference, write_artifact, numpy_build
from .deployment import (_arithmetic, _compare_episodes, _environment, _goal,
                         _sources, _expired_twelve_tick_refusals, _processor_model)
from .subprocess_capture import run_captured


DECLARATION_SHA256 = "eaceadbfd76b5f09156e6ba2173df6205bea9554159b324e99200820ac8b1435"


def load_declaration(path: Path):
    value, file_hash = read_json(path)
    if sha256_json(value) != DECLARATION_SHA256:
        raise ContractError("followup_predeclaration", "fixed prospective follow-up scope required")
    return value, file_hash


class WorkBudget:
    """Reserve before a host operation and retain every unconfirmed tail."""

    def __init__(self, limits, journal, *, clock=time.monotonic, exchanges=None):
        self.limits, self.journal, self.clock = limits, journal, clock
        self.exchanges = exchanges
        self.started = clock()
        self.counts = {"host_requests": 0, "step_requests": 0, "requested_step_ticks": 0,
                       "confirmed_step_ticks": 0, "host_start_attempts": 0, "host_terminations": 0}
        self.pending = {}
        self.observed = {}
        self.requested_by_mode = {"stepped": 0, "realtime": 0}
        self.realtime_reserved_ticks = 0
        self.phase = "admission"

    def note(self, event, **values):
        self.journal.write(canonical_json({"event": event, "phase": self.phase,
                                          "elapsed_seconds": self.clock() - self.started, **values}) + "\n")
        self.journal.flush()

    def check(self):
        if self.clock() - self.started >= self.limits["maximum_wall_seconds"]:
            raise ContractError("followup_wall_budget", "declared invocation wall ceiling reached")

    def reserve(self, operation, fields, *, mode="stepped"):
        # Cancellation remains available after a budget refusal; it revokes
        # existing input and does not request another physical step.
        if operation not in ("cancel", "close"):
            self.check()
        if self.counts["host_requests"] >= self.limits["maximum_host_requests"]:
            raise ContractError("followup_request_budget", "declared owner request ceiling reached")
        ticks = fields.get("ticks", 0) if operation == "step" else 0
        if operation == "step":
            if (type(ticks) is not int or not 1 <= ticks <= 18
                    or self.counts["step_requests"] >= self.limits["maximum_step_requests"]
                    or self.counts["requested_step_ticks"] + ticks > self.limits["maximum_requested_step_ticks"]):
                raise ContractError("followup_step_budget", "declared step/tick ceiling reached")
            self.counts["step_requests"] += 1
            self.counts["requested_step_ticks"] += ticks
            self.requested_by_mode[mode] += ticks
        self.counts["host_requests"] += 1
        serial = self.counts["host_requests"]
        item = {"serial": serial, "operation": operation, "requested_ticks": ticks, "mode": mode}
        self.pending[serial] = item
        self.note("request_reserved", **item)
        if self.exchanges is not None:
            self.exchanges.write((canonical_json({"event": "request_reserved", **item, "fields": fields}) + "\n").encode())
            self.exchanges.flush()
        return serial

    def confirm(self, serial, response):
        if self.exchanges is not None:
            self.exchanges.write((canonical_json({"event": "response_received", "serial": serial,
                                                 "response": response}) + "\n").encode())
            self.exchanges.flush()
        pending = self.pending.pop(serial)
        actual = response.get("actual_ticks", 0)
        if type(actual) is int and actual >= 0:
            self.counts["confirmed_step_ticks"] += actual
        view = response.get("observation")
        if isinstance(view, dict) and type(view.get("tick")) is int:
            key = (pending["mode"], view.get("run_id"), view.get("epoch"))
            self.observed[key] = max(self.observed.get(key, 0), view["tick"])
        self.note("request_confirmed", serial=serial, counts=self.counts)

    def snapshot(self):
        return {**self.counts, "confirmed_observed_episode_ticks": sum(self.observed.values()),
                "observed_tick_lower_bounds_by_mode": {mode: sum(value for key, value in self.observed.items() if key[0] == mode)
                                                       for mode in ("stepped", "realtime")},
                "requested_step_ticks_by_mode": self.requested_by_mode,
                "realtime_horizon_reserved_upper_bound_ticks": self.realtime_reserved_ticks,
                "native_tick_upper_bound": self.requested_by_mode["stepped"] + self.realtime_reserved_ticks,
                "realtime_accounting": "last observed tick is a lower bound; autonomous close tail is bounded by the fixed episode horizon",
                "pending_requests": list(self.pending.values()), "unconfirmed_tail": bool(self.pending),
                "wall_seconds": self.clock() - self.started}


class BudgetHost:
    def __init__(self, host, budget):
        self.host, self.budget = host, budget

    def __getattr__(self, name):
        return getattr(self.host, name)

    def request(self, operation, **fields):
        serial = self.budget.reserve(operation, fields, mode=getattr(self.host, "mode", "stepped"))
        response = self.host.request(operation, **fields)
        self.budget.confirm(serial, response)
        return response


@contextmanager
def managed_host(host_path, godot_path, budget, *, mode="stepped"):
    from nca.envs import GodotHost
    budget.check()
    if budget.counts["host_start_attempts"] >= budget.limits["maximum_host_starts"]:
        raise ContractError("followup_host_budget", "declared host start ceiling reached")
    budget.counts["host_start_attempts"] += 1
    if mode == "realtime":
        budget.realtime_reserved_ticks += budget.limits["realtime_episode_maximum_ticks"]
    budget.note("host_start_reserved", mode=mode)
    host = GodotHost(host_path, godot_path, mode=mode)
    wrapped = BudgetHost(host, budget)
    try:
        yield wrapped
    finally:
        # Meter graceful closure too. At the request ceiling, terminate only
        # this owned process without asking the plant for another operation.
        if host._process is not None and host._process.poll() is None and budget.counts["host_requests"] < budget.limits["maximum_host_requests"]:
            try:
                host._channel.connection.settimeout(2.0)
                wrapped.request("close")
            except Exception:
                pass
        host.close(graceful=False)
        terminated = host._process is None or host._process.poll() is not None
        budget.counts["host_terminations"] += int(terminated)
        budget.note("host_closed", mode=mode, termination_confirmed=terminated,
                    returncode=host._process.returncode if host._process else None)


def throughput_complete(rows, setting, *, phase_seconds):
    """A sample count cannot substitute for the declared active duration."""
    if (len(rows) != len(setting["actor_configurations"])
            or type(phase_seconds) not in (int, float) or not math.isfinite(phase_seconds)
            or any(type(row.get("active_window_seconds")) not in (int, float)
                   or not math.isfinite(row["active_window_seconds"])
                   or type(row.get("decisions")) is not int for row in rows)):
        return False
    return (all(row["configuration"] == config and row["decisions"] > 0
                and row["active_window_seconds"] >= setting["seconds_per_configuration"]
                and row["completed"] is True for row, config in zip(rows, setting["actor_configurations"]))
            and sum(row["active_window_seconds"] for row in rows) >= setting["minimum_active_window_seconds"]
            and phase_seconds <= setting["maximum_phase_seconds"]
            and sum(row["decisions"] for row in rows) <= setting["maximum_decisions"])


def require_consumed_decision(source, action, final, terminated, truncated, info, *, action_tolerance=2e-5,
                              helper_lease_ticks=12):
    """Count only a complete, admitted six-tick request on the stepped owner."""
    from nca.adapters.protocol import make_physical_command
    wire_difference(action, action)
    traces = info.get("trace")
    if (info.get("actual_ticks") != 6 or not isinstance(traces, list) or len(traces) != 6
            or final.get("tick") != source["tick"] + 6 or info.get("idle_trace")
            or info.get("consumed_trace_complete") is not True or info.get("idle_trace_dropped") != 0
            or info.get("unconsumed_terminal_commands") != []
            or type(terminated) is not bool or type(truncated) is not bool
            or terminated or truncated != (final.get("tick") == 240)):
        raise ContractError("throughput_receipt", "complete six-tick stepped interval required")
    for offset, row in enumerate(traces):
        tick = source["tick"] + offset
        observation = row.get("observation", {})
        terminal_revocation = offset == 5 and tick + 1 == 240 and truncated
        if (any(row.get(key) != value for key, value in (("tick_before", tick), ("tick_after", tick + 1),
                ("world_tick", tick + 1), ("cellular_sample_tick", tick)))
                or row.get("failed") is not False or observation.get("fault") != ""
                or observation.get("suspended") is not bool(terminal_revocation) or observation.get("tick") != tick + 1
                or observation.get("control_generation") != source["control_generation"] + int(terminal_revocation)
                or any(observation.get(key) != source.get(key) for key in ("run_id", "epoch", "actor_id"))):
            raise ContractError("throughput_receipt", "invalid native, owner or clock receipt")
        requested, admitted = row.get("requested"), row.get("admission")
        if offset == 0:
            if (not isinstance(requested, list) or len(requested) != 1 or not isinstance(admitted, list)
                    or len(admitted) != 1 or admitted[0].get("accepted") is not True or admitted[0].get("reason") != ""):
                raise ContractError("throughput_admission", "exactly one admitted intended packet required")
            packet = requested[0]
            sequence = packet.get("sequence")
            if (type(sequence) not in (int, float) or not math.isfinite(sequence)
                    or sequence != int(sequence) or sequence <= source.get("last_sequence", -1)):
                raise ContractError("throughput_admission", "fresh owner sequence required")
            expected = make_physical_command(source, int(sequence), action, helper_lease_ticks)
            if set(packet) != set(expected) or any(packet[key] != expected[key] for key in expected if key != "action"):
                raise ContractError("throughput_admission", "receipt must refer to the original issued source")
            difference = wire_difference(packet["action"], action)
            if not difference["boolean_actions_exact"] or difference["maximum_continuous_difference"] > 1e-12:
                raise ContractError("throughput_admission", "receipt does not match the intended request")
        elif requested != [] or admitted != []:
            raise ContractError("throughput_admission", "replacement or mixed request inside a measured decision")
        applied = row.get("applied", {})
        if terminal_revocation:
            if (applied.get("terminal_revocation") is not True or applied.get("action_available") is not False
                    or applied.get("action") is not None):
                raise ContractError("throughput_consumption", "exact declared horizon revocation required")
        else:
            if applied.get("action_available") is not True or applied.get("terminal_revocation") is not False:
                raise ContractError("throughput_consumption", "unavailable consumed request")
            consumed = applied.get("action")
            flags = ("primary", "secondary", "auxiliary", "tool", "attach_usable", "release_usable")
            if (not isinstance(consumed, dict) or set(consumed) != set(action)
                    or any(type(consumed[key]) is not bool or consumed[key] != action[key] for key in flags)
                    or any(not isinstance(consumed[key], (list, tuple)) or len(consumed[key]) != 2
                           for key in ("move", "aim"))):
                raise ContractError("throughput_consumption", "incomplete or different consumed wire action")
            # These are post-owner FP32 values. Normalizing a unit aim can
            # round its magnitude just above one; it is not a new wire packet.
            # Compare its finite components with the admitted request using the
            # unchanged action tolerance instead of reparsing it as a request.
            pairs = [(consumed[key], action[key]) for key in ("trim", "effort", "reel")]
            pairs.extend((a, b) for key in ("move", "aim") for a, b in zip(consumed[key], action[key]))
            if any(type(a) not in (int, float) or not math.isfinite(a) or abs(a - b) > action_tolerance for a, b in pairs):
                raise ContractError("throughput_consumption", "owner did not consume the intended bounded request")


def _paired_episode(host, case, config, controller, *, reference):
    """Capture the actual issued features and every owner tick before comparing."""
    kind = "original-torch-reference" if reference else BACKEND["id"]
    env, view, _, reset_ns = _environment(host, case, config)
    profile = host.request("profile")["profile"]
    goal = _goal(case, config)
    initial, observations, steps, frames, times = view, [], [], [pose(view)], []
    try:
        while view["tick"] < config["trajectory"]["ticks"]:
            issued = view
            observations.append(issued)
            before = time.perf_counter_ns()
            result = controller.step(issued, goal, deterministic=True) if reference else controller.step(issued, goal)
            times.append(time.perf_counter_ns() - before)
            hidden_after = controller.hidden[0].tolist() if reference else result.hidden_after
            host.budget.note("policy_decision", controller=kind, case=case["id"], issued_tick=issued["tick"],
                action=result.action.to_dict(), features=result.features, hidden_before=result.hidden_before,
                hidden_after=hidden_after, value=result.value)
            if reference:
                lease = min(12, initial["tick"] + round(goal.timeout_seconds * 60) - issued["tick"])
                view, _, terminated, truncated, info = env.step_packet(
                    env.command(result.action.to_dict(), observation=issued, lease_ticks=lease))
            else:
                if result.status != "accepted":
                    raise ContractError("deployment_fallback", result.reason)
                view, _, terminated, truncated, info = dispatch(env, controller, result, issued)
                lease = result.helper_lease_ticks
            require_consumed_decision(issued, result.action.to_dict(), view, terminated, truncated, info,
                                      helper_lease_ticks=lease)
            steps.append({"issued_observation_id": issued["observation_id"], "issued_tick": issued["tick"],
                "action": result.action.to_dict(), "features": result.features, "hidden_before": result.hidden_before,
                "hidden_after": hidden_after, "value": result.value, "actual_ticks": info["actual_ticks"],
                "terminated": terminated, "truncated": truncated, "trace": info["trace"],
                "consumed_trace_complete": info["consumed_trace_complete"], "idle_trace_dropped": info["idle_trace_dropped"],
                "unconsumed_terminal_commands": info["unconsumed_terminal_commands"]})
            frames.extend(pose(row["observation"]) for row in info["trace"])
            if terminated or truncated:
                break
        controller.cancel()
        cancelled = env.cancel()
        cancellation_exact = (pose(cancelled) == pose(view) and cancelled["tick"] == view["tick"]
                              and cancelled["control_generation"] == view["control_generation"] + 1)
    except BaseException:
        try:
            controller.cancel()
            env.cancel()
        except Exception:
            pass
        raise
    return {"format": "nca-export-trajectory/v1", "case": dict(case), "controller": kind,
        "controller_interface": "LearnedController reference" if reference else "ExportController with original-source dispatch",
        "config": env.config, "profile": profile, "profile_sha256": sha256_json(profile), "goal": asdict(goal),
        "initial": initial, "steps": steps, "frames": frames, "observations": observations,
        "final": view, "cancelled": cancelled, "cancellation_exact": cancellation_exact,
        "reset_ns": reset_ns, "inference": durations(times), "actual_ticks": sum(step["actual_ticks"] for step in steps),
        "decisions": len(steps), "stop_reason": "declared_host_horizon"}


def _throughput(host, bundle, original, setting, budget, output):
    started, total_decisions, rows = time.monotonic(), 0, []
    case = original["trajectory"]["cases"][0]
    for index, configuration in enumerate(setting["actor_configurations"]):
        wall_started = time.monotonic()
        reset_times, inference_times, transport_times = [], [], []
        ticks = decisions = episodes = 0
        controller = env = None
        trace_path = output / f"throughput-{index}-receipts.jsonl.gz"
        try:
            with trace_path.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
                while time.monotonic() - wall_started < setting["seconds_per_configuration"]:
                    budget.check()
                    if (time.monotonic() - started >= setting["maximum_phase_seconds"]
                            or total_decisions >= setting["maximum_decisions"]):
                        raise ContractError("throughput_budget", "active duration not reached before declared ceiling")
                    if controller is None:
                        companions = () if configuration["total_actors"] == 1 else (configuration["companion_model"],)
                        env, view, _, reset_ns = _environment(host, case, original, companions=companions)
                        reset_times.append(reset_ns)
                        controller = ExportController(bundle)
                        goal = _goal(case, original, f"-throughput-{index}-{episodes}")
                        episodes += 1
                    before = time.perf_counter_ns()
                    decision = controller.step(view, goal)
                    inference_ns = time.perf_counter_ns() - before
                    if decision.status != "accepted":
                        raise ContractError("throughput_fallback", decision.reason)
                    before = time.perf_counter_ns()
                    issued = view
                    view, _, terminal, truncated, info = dispatch(env, controller, decision, issued)
                    transport_ns = time.perf_counter_ns() - before
                    require_consumed_decision(issued, decision.action.to_dict(), view, terminal, truncated, info,
                                              helper_lease_ticks=decision.helper_lease_ticks)
                    ticks += info["actual_ticks"]
                    decisions += 1
                    total_decisions += 1
                    inference_times.append(inference_ns)
                    transport_times.append(transport_ns)
                    stream.write((canonical_json({"decision": decisions, "episode": episodes,
                        "action": decision.action.to_dict(), "source_tick": decision.source.tick,
                        "inference_ns": inference_ns, "step_ns": transport_ns,
                        "actual_ticks": info["actual_ticks"], "trace": info["trace"]}) + "\n").encode())
                    stream.flush()
                    if decisions % 100 == 0:
                        budget.note("throughput_progress", configuration=index, decisions=decisions,
                                    actual_ticks=ticks, active_seconds=time.monotonic() - wall_started)
                    if terminal or truncated:
                        controller.cancel()
                        env.cancel()
                        controller = None
        finally:
            try:
                if controller is not None and env is not None:
                    controller.cancel()
                    env.cancel()
            finally:
                budget.note("throughput_window_closed", configuration=index, decisions=decisions,
                            actual_ticks=ticks, episodes=episodes, active_seconds=time.monotonic() - wall_started)
        elapsed = time.monotonic() - wall_started
        row = {"configuration": configuration, "active_window_seconds": elapsed,
               "decisions": decisions, "actual_ticks": ticks, "episodes": episodes,
               "decisions_per_second": decisions / elapsed, "ticks_per_second": ticks / elapsed,
               "inference": durations(inference_times), "step_transport_and_host": durations(transport_times),
               "reset": durations(reset_times), "host_process_memory": process_memory(host.pid),
               "combined_reference_deployment_process_memory": process_memory(os.getpid()),
               "receipt_file": trace_path.name, "receipt_file_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
               "receipt_bytes": trace_path.stat().st_size, "completed": elapsed >= setting["seconds_per_configuration"]}
        rows.append(row)
        write_artifact(output, f"throughput-configuration-{index}.json", row)
        budget.note("throughput_configuration_complete", configuration=index, decisions=decisions, active_seconds=elapsed)
    resets, observations = [], []
    env, _, _, _ = _environment(host, case, original)
    for _ in range(setting["reset_samples"]):
        before = time.perf_counter_ns()
        env.reset()
        resets.append(time.perf_counter_ns() - before)
    for _ in range(setting["observe_samples"]):
        before = time.perf_counter_ns()
        env.observe()
        observations.append(time.perf_counter_ns() - before)
    env.cancel()
    elapsed = time.monotonic() - started
    return {"passed": throughput_complete(rows, setting, phase_seconds=elapsed), "configurations": rows,
            "active_window_seconds": sum(row["active_window_seconds"] for row in rows), "phase_seconds": elapsed,
            "reset_rpc": durations(resets), "observe_rpc": durations(observations),
            "multiple_learned_actors": "unmeasured", "idle_padding_seconds": 0,
            "scope": "two continuously executing 300-second host/controller/capture windows"}


def _negative_contracts(host, bundle, config):
    rows = []
    for name, alter in (("bundle_version", lambda p: p.update(format="future")),
                        ("backend_version", lambda p: p["backend"].update(id="other")),
                        ("backend_sources", lambda p: p.update(backend_sources={})),
                        ("manufactured_qualification", lambda p: p["qualification"].update(runtime_qualified=True))):
        payload = bundle.value
        alter(payload)
        payload.pop("sha256")
        payload["sha256"] = sha256_json(payload)
        reason = None
        try:
            TorchRuntimeBundle(payload, bundle.context)
        except ContractError as exc:
            reason = exc.code
        rows.append({"case": name, "passed": reason is not None, "reason": reason, "host_work": 0})
    nonfinite = bundle.value
    nonfinite["numeric_bundle"]["weights"]["encoder.bias"][0] = float("nan")
    reason = None
    try:
        TorchRuntimeBundle(nonfinite, bundle.context)
    except ContractError as exc:
        reason = exc.code
    rows.append({"case": "nonfinite_weights", "passed": reason is not None, "reason": reason, "host_work": 0})
    # Existing numeric profile/action/assist/weights admission remains an
    # authenticated nested contract, and owner lifecycle tests use this backend.
    from dataclasses import replace
    for name, changes in (("profile_hash", {"profile_sha256": "a" * 64}),
                          ("sensor_hash", {"sensor_sha256": "a" * 64}),
                          ("action_hash", {"action_sha256": "a" * 64}),
                          ("assist_hash", {"assist_sha256": "a" * 64}),
                          ("base_hz_30", {"base_hz": 30}),
                          ("decision_ticks_12", {"decision_ticks": 12}),
                          ("helper_lease_ticks_18", {"helper_lease_ticks": 18})):
        reason = None
        try:
            TorchRuntimeBundle(bundle.value, replace(bundle.context, **changes))
        except ContractError as exc:
            reason = exc.code
        rows.append({"case": name, "passed": reason is not None, "reason": reason, "host_work": 0})
    case = config["trajectory"]["cases"][0]
    for name in ("old_actor_epoch", "old_control_generation", "stale_observation", "reordered_observation"):
        env, first, _, _ = _environment(host, case, config)
        controller, goal = ExportController(bundle), _goal(case, config, "-negative")
        initial = controller.step(first, goal)
        if initial.status != "accepted":
            raise ContractError("negative_initial", initial.reason)
        if name == "old_actor_epoch":
            current, _ = env.reset()
            controller.step(current, goal)
            result = controller.step(first, goal)
        elif name == "old_control_generation":
            current = env.cancel()
            result = controller.step(current, goal)
        else:
            current, _, _, _, _ = env.step_packet(None)
            if name == "stale_observation":
                current, _, _, _, _ = env.step_packet(None)
                result = controller.step(first, goal, current_tick=current["tick"])
            else:
                controller.step(current, goal)
                result = controller.step(first, goal)
        cancelled = dispatch(env, controller, result, current)
        rows.append({"case": name, "passed": result.cancel_owner and pose(cancelled) == pose(current),
                     "reason": result.reason, "cancel_preserves_pose": pose(cancelled) == pose(current)})
    names = {row["case"] for row in rows}
    required = set(config["negative_cases"]) | {"backend_version", "backend_sources", "manufactured_qualification"}
    return {"passed": names == required and all(row["passed"] for row in rows), "cases": rows,
            "required_coverage_exact": names == required}


def _variant(host, bundle, config, *, model="jev", decision_ticks=6):
    from nca.contracts import WireAction
    case = config["trajectory"]["cases"][0]
    env, view, _, _ = _environment(host, case, config, ticks=24, model=model, decision_ticks=decision_ticks)
    initial = view
    profile_response = host.request("profile")
    profile = profile_response["profile"]
    reason = None
    try:
        TorchRuntimeBundle(bundle.value, context_for(profile, host.identity, decision_ticks=decision_ticks))
    except ContractError as exc:
        reason = exc.code
    env.cancel()
    frames, traces = [pose(view)], []
    while view["tick"] < 24:
        view, _, terminal, truncated, info = env.step_packet(None)
        frames.extend(pose(row["observation"]) for row in info["trace"])
        traces.extend(info["trace"])
        if terminal or truncated:
            break
    return {"format": "nca-export-refusal-trajectory/v1", "passed": reason is not None,
            "case": case, "controller": "refused-and-owner-cancelled", "config": env.config,
            "initial": initial, "final": view, "profile": profile, "profile_sha256": sha256_json(profile),
            "profile_provenance": {"source": "live profile response after reset", "response_id": profile_response["id"],
                "reset_observation_id": initial["observation_id"], "run_id": initial["run_id"],
                "epoch": initial["epoch"], "actor_id": initial["actor_id"]},
            "frames": frames, "trace": traces, "reason": reason, "requested_actions": 0, "actual_ticks": view["tick"]}


def _realtime(host_path, godot_path, bundle, config, budget):
    started = time.perf_counter()
    limit = config["realtime"]
    case = config["trajectory"]["cases"][0]
    calls = 0
    with managed_host(host_path, godot_path, budget, mode="realtime") as host:
        env, first, _, _ = _environment(host, case, config, ticks=600)
        preparation = {"move": [1., 0.], "aim": [.8, -.6], "effort": .8, "primary": True}
        first, _, _, _, preparation_info = env.step(preparation)
        controller = ExportController(bundle)
        goal = _goal(case, config)
        decision = controller.step(first, goal)
        calls += 1
        if decision.status != "accepted":
            raise ContractError("realtime_initial", decision.reason)
        # The delayed output retains this exact first view and twelve-tick lease.
        packet = env.command(decision.action.to_dict(), observation=first)
        time.sleep(limit["delay_seconds"])
        view, _, _, _, info = env.step_packet(packet)
        receipts = info["idle_trace"] + info["trace"]
        refusals = _expired_twelve_tick_refusals(first, packet, receipts)
        result = controller.step(first, goal, current_tick=view["tick"])
        calls += 1
        before_response = host.request("observe")
        before = before_response["observation"]
        cancelled = dispatch(env, controller, result, before)
        # A realtime request can straddle a tick. The immutable cancellation
        # result is compared to the last available receipt at its own tick.
        after_response = host.request("observe")
        available = before_response.get("idle_trace", []) + after_response.get("idle_trace", [])
        matching = [row["observation"] for row in available if row["tick_after"] == cancelled["tick"]]
        prior_same_tick = before if before["tick"] == cancelled["tick"] else (matching[-1] if matching else None)
        preserved = (prior_same_tick is not None and pose(prior_same_tick) == pose(cancelled)
                     and math.hypot(*cancelled["velocity"]) > 1e-3)
        # A distinct invocation is alive when the source generation changes.
        # This check therefore cannot pass solely because expiry retired the
        # earlier invocation. Cancellation clears its old cadence cursor only.
        next_goal = replace(goal, id=goal.id + "-fresh-owner")
        fresh = cancelled
        launched = controller.step(fresh, next_goal, source_generation=1)
        calls += 1
        if launched.status != "accepted":
            raise ContractError("realtime_fresh_invocation", launched.reason)
        fresh, _, _, _, fresh_info = dispatch(env, controller, launched, fresh)
        takeover = controller.step(fresh, next_goal, source_generation=2)
        calls += 1
        transfer_before = host.request("observe")
        transferred = dispatch(env, controller, takeover, fresh)
        transfer_after = host.request("observe")
        transfer_candidates = [transfer_before["observation"]]
        transfer_candidates.extend(row["observation"] for response in (transfer_before, transfer_after)
                                   for row in response.get("idle_trace", []))
        transfer_same_tick = next((row for row in reversed(transfer_candidates) if row["tick"] == transferred["tick"]), None)
        transfer_preserved = transfer_same_tick is not None and pose(transfer_same_tick) == pose(transferred)
        fresh_admitted = any(item.get("accepted") is True for row in fresh_info["trace"] for item in row["admission"])
        elapsed = time.perf_counter() - started
        checks = {"host_advances_during_delay": view["tick"] - first["tick"] >= 12,
                  "old_twelve_tick_motor_lease_refused": bool(refusals),
                  "controller_source_age_refused": result.reason == "source_expired",
                  "owner_cancel_preserves_existing_motion": preserved,
                  "source_takeover_retires_old_invocation": fresh_admitted and takeover.cancel_owner
                    and takeover.reason == "owner_or_mount_changed" and transfer_preserved,
                  "bounded_calls_and_time": calls <= limit["maximum_policy_calls"] and elapsed <= limit["maximum_seconds"]}
        return {"passed": all(checks.values()), "checks": checks, "host_identity": host.identity,
                "config": env.config, "policy_calls": calls, "wall_seconds": elapsed,
                "authored_motion_preparation": {"action": preparation, "trace": preparation_info["trace"],
                                                "purpose": "nonzero actual motion before delayed learned output and cancellation"},
                "source": first, "packet": packet, "after_delay": view, "refusals": refusals,
                "source_fallback": result.reason, "cancellation": cancelled,
                "same_tick_before_cancel": prior_same_tick, "after_takeover": fresh,
                "fresh_invocation": {"goal": asdict(next_goal), "status": launched.status,
                                     "admitted": fresh_admitted, "trace": fresh_info["trace"]},
                "takeover_reason": takeover.reason, "takeover_cancelled": transferred,
                "same_tick_before_takeover_cancel": transfer_same_tick, "takeover_sent_packets": 0,
                "trace": receipts,
                "plan_lifetime_ticks": 120, "plan_scope": "not exercised; source planning is the separately qualified T108 layer",
                "learned_performance_promotion": False}


def _file_identity(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ContractError("followup_input_identity", path.name)


def runtime_resources():
    """Public-safe device, build and effective container limits; no host names."""
    import torch
    limits = {}
    for name in ("cpu.max", "memory.max", "cpuset.cpus.effective"):
        try:
            limits[name] = Path("/sys/fs/cgroup", name).read_text().strip()
        except OSError:
            limits[name] = None
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {"processor_model": _processor_model(), "logical_cpu_count": os.cpu_count(),
            "process_affinity_cpu_count": affinity, "cgroup_limits": limits, "numpy_build": numpy_build(),
            "torch_build": {"version": str(torch.__version__), "git_version": torch.version.git_version,
                            "debug": torch.version.debug, "cuda": torch.version.cuda,
                            "mkl_available": torch.backends.mkl.is_available(),
                            "mkldnn_available": torch.backends.mkldnn.is_available(),
                            "cpu_capability": torch.backends.cpu.get_cpu_capability()}}


def run_followup(*, root, declaration_path, original_root, host_path, godot_path,
                 output, source_revision, render_python):
    import torch
    from nca.controllers.learned import LearnedController, RecurrentPolicy
    root, original_root, output = (Path(path).resolve() for path in (root, original_root, output))
    declaration, declaration_file_hash = load_declaration(Path(declaration_path))
    original = declaration["original"]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if revision != source_revision or dirty:
        raise ContractError("followup_source", "clean explicitly published source revision required")
    files = {"bc-seed1103-cpu.json": original["numeric_file_sha256"],
             "screen-v1/manifest.json": original["manifest_file_sha256"],
             "screen-v1/summary.json": original["summary_file_sha256"],
             "screen-v1/resume.json": original["resume_file_sha256"],
             "screen-v1/learner-cost.json": original["learner_cost_file_sha256"]}
    for name, expected in files.items():
        _file_identity(original_root / name, expected)
    _file_identity(root / "src/nca/controllers/learned.py", original["reference_source_sha256"])
    config, _ = read_json(root / "configs/export/cpu-bc-v1.json")
    if sha256_json(config) != original["criteria_content_sha256"]:
        raise ContractError("followup_criteria", "original arithmetic/trajectory criteria changed")
    old, _ = read_json(original_root / "screen-v1/summary.json")
    if old["status"] != "no_go" or old["runtime_qualified"] is not False:
        raise ContractError("followup_history", "the original failed screen must remain unchanged")
    torch.set_num_threads(1)
    numeric_payload, _ = read_json(original_root / "bc-seed1103-cpu.json")
    context = context_for(numeric_payload["profile"], numeric_payload["context"])
    numeric = RuntimeBundle.load(original_root / "bc-seed1103-cpu.json", context,
                                expected_sha256=original["numeric_file_sha256"])
    if numeric.sha256 != original["numeric_content_sha256"] or numeric.network.policy_sha256 != original["policy_sha256"]:
        raise ContractError("followup_policy", "fixed evaluated policy identity differs")
    sources = _sources(root)
    sources.update({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                    ("tools/export_torch_policy.py", "configs/export/torch-cpu-followup-v1.json")})
    output.mkdir(parents=True, exist_ok=False)
    with (output / "work-journal.jsonl").open("x") as journal, (output / "owner-exchanges.jsonl.gz").open("xb") as exchange_raw, gzip.GzipFile(fileobj=exchange_raw, mode="wb", mtime=0) as exchanges:
        budget = WorkBudget(declaration["budget"], journal, exchanges=exchanges)
        budget.note("invocation_reserved", source_revision=revision, maximum_invocations=1)
        manifest = {"format": "nca-torch-cpu-followup-manifest/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_revision": revision, "source_files": sources, "declaration": declaration,
                    "declaration_file_sha256": declaration_file_hash, "original_file_sha256": files,
                    "backend": BACKEND, "runtime_context": context.to_dict(), "resources": runtime_resources(),
                    "training_updates": 0}
        write_artifact(output, "manifest.json", manifest)
        report = {"format": "nca-torch-cpu-followup-result/v1", "runtime_qualified": False,
                  "qualification_scope": "Linux external Python Torch operators FP32 CPU loop",
                  "numpy_runtime_qualified": False, "original_summary_file_sha256": original["summary_file_sha256"],
                  "source_revision": revision, "errors": [], "training_updates": 0, "owner_feel_accepted": False,
                  "whole_game_determinism": False, "backend": BACKEND}
        try:
            payload = make_torch_bundle(numeric.value, context, numeric_file_sha256=original["numeric_file_sha256"])
            bundle_path = output / "bc-seed1103-torch-cpu.json"
            bundle_hash = immutable_json(bundle_path, payload)
            bundle = TorchRuntimeBundle.load(bundle_path, context, expected_sha256=bundle_hash)
            report["bundle_file_sha256"] = bundle_hash
            policy = RecurrentPolicy("bc")
            policy.load_state_dict({name: torch.tensor(value) for name, value in numeric.value["weights"].items()})
            policy.eval().requires_grad_(False)
            budget.phase = "arithmetic"
            report["arithmetic"] = _arithmetic(bundle, policy, config)
            write_artifact(output, "arithmetic.json", report["arithmetic"])
            if not report["arithmetic"]["passed"]:
                raise ContractError("followup_arithmetic", "stop before host work after arithmetic failure")
            pairs, sequences = [], []
            with managed_host(host_path, godot_path, budget) as host:
                if context_for(numeric.value["profile"], host.identity) != context:
                    raise ContractError("live_host_context", "actual host differs from the declared lane")
                report["host_identity"] = host.identity
                budget.phase = "paired_trajectories"
                comparisons = []
                for case in config["trajectory"]["cases"]:
                    left = _paired_episode(host, case, config, LearnedController(policy), reference=True)
                    right = _paired_episode(host, case, config, ExportController(bundle), reference=False)
                    names = [f"trajectories/{case['id']}-{kind}.json.gz" for kind in ("reference", "torch-export")]
                    for name, episode in zip(names, (left, right)):
                        write_artifact(output, name, episode)
                    comparison = _compare_episodes(left, right, config)
                    comparisons.append({"case": case["id"], **comparison})
                    write_artifact(output, f"comparison-{case['id']}.json", comparisons[-1])
                    if not comparison["passed"]:
                        raise ContractError("followup_trajectory", "stop before throughput after unchanged trajectory gate failure")
                    pairs.append({"id": case["id"], "left": names[0], "right": names[1],
                                  "labels": ["Original Torch reference", "Torch CPU export"], "ticks": 240})
                    sequences.append({"goal": right["goal"], "observations": right["observations"]})
                report["trajectories"] = {"passed": True, "cases": comparisons}
                budget.phase = "negative_contracts"
                report["negative_contracts"] = _negative_contracts(host, bundle, config)
                write_artifact(output, "negative-contracts.json", report["negative_contracts"])
                if not report["negative_contracts"]["passed"]:
                    raise ContractError("followup_negative", "negative contract gate failed")
                budget.phase = "profile_and_cadence"
                variants = []
                for name, kwargs in (("decision-12-refused", {"decision_ticks": 12}),
                                     ("astra-profile-refused", {"model": "astra"})):
                    variant = _variant(host, bundle, config, **kwargs)
                    filename = f"trajectories/{name}.json.gz"
                    write_artifact(output, filename, variant)
                    variants.append({"case": name, "passed": variant["passed"], "reason": variant["reason"]})
                    pairs.append({"id": name, "left": pairs[0]["right"], "right": filename,
                                  "labels": ["Nominal Torch CPU export", name], "ticks": 24})
                report["variants"] = variants
                if not all(row["passed"] for row in variants):
                    raise ContractError("followup_variant", "profile/cadence refusal failed")
                budget.phase = "realtime"
                report["realtime"] = _realtime(host_path, godot_path, bundle, config, budget)
                write_artifact(output, "realtime.json.gz", report["realtime"])
                if not report["realtime"]["passed"]:
                    raise ContractError("followup_realtime", "realtime lifetime gate failed")
                budget.phase = "throughput"
                budget.note("phase_started", target_active_seconds=600)
                report["throughput"] = _throughput(host, bundle, config, declaration["throughput"], budget, output)
                write_artifact(output, "throughput.json", report["throughput"])
                if not report["throughput"]["passed"]:
                    raise ContractError("followup_duration", "full active ten-minute gate failed")
            budget.phase = "standalone_inference"
            budget.check()
            write_artifact(output, "inference-inputs.json", {"sequences": sequences})
            child_environment = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
            subprocess.run([sys.executable, "-m", "nca.bench.torch_inference_worker", "--bundle", str(bundle_path),
                            "--sha256", bundle_hash, "--inputs", str(output / "inference-inputs.json"),
                            "--output", str(output / "inference-cost.json")], check=True,
                           timeout=min(45, declaration["budget"]["maximum_wall_seconds"] - budget.snapshot()["wall_seconds"]),
                           cwd=root, env=child_environment, capture_output=True)
            report["inference_cost"], _ = read_json(output / "inference-cost.json")
            budget.phase = "clips"
            budget.check()
            write_artifact(output, "clip-spec.json", {"format": "nca-export-pose-pairs/v1", "pairs": pairs,
                "camera": {"world_bounds": [128, 200, 320, 368], "panel_pixels": [576, 504]},
                "source": "actual own-part poses; reconstruction, not engine-rendered frames",
                "base_hz": 60, "frame_stride": 6, "frame_duration_ms": 100})
            run_captured([render_python, "-m", "nca.bench.pose_clip", "--root", str(output)],
                         output=output, label="renderer",
                         timeout=min(45, declaration["budget"]["maximum_wall_seconds"] - budget.snapshot()["wall_seconds"]),
                         cwd=root, env=child_environment)
            report["clips"], _ = read_json(output / "clips.json")
            report["retained_fresh_episode_resume"], _ = read_json(original_root / "screen-v1/resume.json")
            report["retained_learner_cost"], _ = read_json(original_root / "screen-v1/learner-cost.json")
            report["runtime_qualified"] = (report["inference_cost"]["inference"]["count"] == 1000
                and report["inference_cost"]["learner_imported"] is False
                and report["clips"].get("complete") is True and report["retained_fresh_episode_resume"]["passed"] is True)
        except BaseException as exc:
            report["errors"].append({"phase": budget.phase, "type": type(exc).__name__,
                                     "code": exc.code if isinstance(exc, ContractError) else "execution_failure"})
            import traceback
            traceback.print_exc(file=sys.stderr)
        for name, expected in files.items():
            try:
                _file_identity(original_root / name, expected)
            except (ContractError, OSError):
                report["errors"].append({"phase": "final_input_check", "code": "input_changed", "file": name})
        for name, expected in sources.items():
            try:
                unchanged = hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
            except OSError:
                unchanged = False
            if not unchanged:
                report["errors"].append({"phase": "final_source_check", "code": "source_changed", "file": name})
        report["work"] = budget.snapshot()
        if (report["errors"] or report["work"]["unconfirmed_tail"]
                or report["work"]["host_terminations"] != report["work"]["host_start_attempts"]
                or report["work"]["host_requests"] > declaration["budget"]["maximum_host_requests"]
                or report["work"]["step_requests"] > declaration["budget"]["maximum_step_requests"]
                or report["work"]["requested_step_ticks"] > declaration["budget"]["maximum_requested_step_ticks"]
                or report["work"]["wall_seconds"] > declaration["budget"]["maximum_wall_seconds"]):
            report["runtime_qualified"] = False
        report["status"] = "qualified" if report["runtime_qualified"] else "no_go"
        budget.note("invocation_complete", status=report["status"], counts=report["work"])
    report["artifact_files"] = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in output.rglob("*") if path.is_file()}
    report["sha256"] = sha256_json(report)
    write_artifact(output, "summary.json", report)
    return report
