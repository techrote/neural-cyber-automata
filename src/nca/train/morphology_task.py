"""Frozen semantic waypoint reference, native-receipt evaluator and closed gate.

This standard-library module neither opens a host nor admits an experiment.
Its fixtures can test accounting; only separately retained physical episodes
can qualify the reference under a published continuation declaration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from nca.contracts import ContractError, WireAction, sha256_json
from nca.controllers.morphology import MorphologyAdapter, MorphologyGoal, SemanticAction, StateFence
from nca.controllers.shaping import wrapped_angle


TASK_SPEC = {
    "schema": "nca-morphology-waypoint-task/v1",
    "base_hz": 60, "decision_ticks": 6, "lease_ticks": 12, "deadline_ticks": 240,
    "position_source": "observed parts[0], validated against the accepted profile ordering",
    "arrival": {"distance_maximum_cells": 4., "speed_maximum_cells_per_second": 6.,
                "absolute_angle_maximum_radians": .35, "consecutive_native_ticks": 12,
                "support_required": False, "deadline_inclusive": True},
    "reward": {"schema": "morphology-distance-potential/v1", "distance_scale_cells": 48.,
               "time_cost_per_second": .01, "squared_request_cost_per_second": .002,
               "request_components": ["move_x", "move_y", "trim", "effort"],
               "arrival_bonus": 1., "missed_deadline_penalty": -.5,
               "potential": "sum (previous_distance-current_distance)/48 without clipping",
               "priority": "award arrival at tick240 before testing missed deadline",
               "administrative_cutoff": "no invented terminal penalty",
               "invalid_interval": "no eligible reward; original receipts remain with the recorder",
               "completed_repeat": "account for every actual tick even after arrival within the repeat"},
    "receipt_contract": {
        "action_float_tolerance": 1e-6,
        "native_schedule": "tick_after=world_tick=observation.tick=tick_before+1; cellular_sample_tick=tick_before",
        "admission": "one accepted original source request at the first tick; no replacements in its six-tick hold",
        "sequence": "nonnegative first sequence followed by strictly consecutive issued sequences",
        "request_echo_numbers": "finite exact integral int/float values match the original typed identity; booleans and fractional numbers refuse",
        "terminal": "only at the intrinsic deadline; final readback explicitly unavailable after owner revocation",
        "source": "same run, actor, epoch, profile and topology; terminal control revocation advances generation once",
    },
    "metrics": {
        "trim_deadband": .05,
        "ordinary_trim_reversals": "last nonzero trim sign across deadband; all completed task seconds, no excluded request buttons",
        "angle": "mean and peak absolute wrapped observed root angle over completed native ticks",
        "useful_speed": "max(0, root_velocity dot unit(current target-root displacement)); zero at coincidence",
        "response_onset_minimum_cells_per_second": 3.,
        "inference": "one measured nonnegative nanosecond sample per complete request; linear interpolated p50/p95",
        "settled_after_landing": "null until observed support true-to-false-to-true followed by12 supported speed<=6 and abs(angle)<=0.35 ticks; latest reacquisition before first such settlement",
        "capped_completion": "first arrival tick/60 when valid; otherwise intrinsic deadline240/60",
        "owner_feel_accepted": False,
    },
    "qualification": {
        "case_ids": ["qualification-jev-floor-left-26", "qualification-jev-floor-right-26",
                     "qualification-astra-floor-left-26", "qualification-astra-floor-right-26"],
        "profiles": ["jev", "jev", "astra", "astra"],
        "host_seeds": [610651, 610652, 610653, 610654],
        "spawn_x": [264., 176., 264., 176.], "target_dx": [-26., 26., -26., 26.],
        "floor_y": 352., "actor_y": "floor_y minus authored spawn_floor_clearance",
        "target_y": "accepted observed root floor position, distinct from actor origin",
        "pace": .8, "maximum_requests_per_case": 40, "maximum_total_requests": 160,
        "maximum_inference_p95_ms": 5., "all_four_must_pass": True, "retries": 0,
        "gate": "exact case/seed/profile, reference/task hashes, quiet arrival, complete counters, no error/cap/refusal/integrity/source regression and verified final owner closure",
        "evidence_limit": "row predicate only; caller must verify original artifact bytes, source audit, runtime and aggregate budget before admission",
    },
}

REFERENCE_SPEC = {
    "schema": "nca-morphology-waypoint-reference/v1",
    "input": "explicit caller goal, actual observed root pose/motion and permitted numeric mechanics",
    "speed_denominator": {"profile_field": "speed", "effort": "goal.pace", "minimum": 5., "maximum": 240.},
    "position_gain": 1.8, "desired_speed_maximum": 30., "velocity_feedback_gain": .35,
    "inside_tolerance": "desired velocity zero on every available axis",
    "drive": "clamp((desired_velocity-0.35*root_velocity)/clamp(speed*pace,5,240),-1,1)",
    "trim_angle_gain": 1.6, "trim_angular_velocity_gain": .25,
    "trim": "clamp(-1.6*wrapped_root_angle-0.25*root_angular_velocity,-1,1)",
    "brake_speed_minimum": 2.,
    "brake": "only when capable, speed>2 and (inside tolerance or velocity dot displacement<0)",
    "impulse": False, "rebound": False, "effort": "goal.pace",
    "vertical_drive": "zero unless the accepted semantic capability mask permits it",
    "wire_mapping": "unchanged shared map_action; unit-normalized movement and caller target aim",
    "fences": "original source hash, six-tick cadence, finite root, goal, lifetime, control, source and topology; cancellation retires invocation",
    "duplicate": "exact same source and goal returns the same cached semantic request without advancing state",
    "identity_labels": "evidence only; no rig or part embeddings and no evaluator input",
}


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise ContractError(code, detail)


def _number(value: Any, field: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value),
             "waypoint_number", f"{field} must be finite and not boolean")
    return float(value)


def _pair(value: Any, field: str) -> tuple[float, float]:
    _require(isinstance(value, (tuple, list)) and len(value) == 2,
             "waypoint_shape", f"{field} must have two finite components")
    return tuple(_number(item, field) for item in value)


def _root(view: Mapping[str, Any], *, terminal: bool = False):
    root = view["parts"][0]
    _require(terminal or root["detached"] is False, "waypoint_root", "an active observed root is required")
    support = view.get("grounded")
    _require(support is None or type(support) is bool,
             "waypoint_support", "support must be observed boolean or explicitly unavailable")
    angle = _number(root.get("rotation"), "root.rotation")
    # Preserve an already wrapped value at the exact inclusive .35 boundary;
    # adding/subtracting pi needlessly rounds .35 upward on binary floats.
    angle = angle if -math.pi <= angle < math.pi else wrapped_angle(angle)
    velocity = _pair(root.get("velocity"), "root.velocity")
    _number(math.hypot(*velocity), "root.speed")
    return (_pair(root.get("position"), "root.position"), velocity, angle,
            _number(root.get("angular_velocity"), "root.angular_velocity"), support)


def _fixed_goal(goal: MorphologyGoal) -> None:
    _require(isinstance(goal, MorphologyGoal) and goal.tolerance == 4. and goal.deadline_ticks == 240,
             "waypoint_goal", "the frozen task requires four-cell tolerance and240 native ticks")


def _clip(value: float, limit: float = 1.) -> float:
    return max(-limit, min(limit, value))


def _same_action(action: WireAction, actual: Any) -> bool:
    expected = action.to_dict()
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        return False
    for key, value in expected.items():
        other = actual[key]
        if type(value) is bool:
            if other is not value:
                return False
        else:
            targets = value if isinstance(value, list) else [value]
            values = other if isinstance(value, list) and isinstance(other, (list, tuple)) else [other]
            if len(targets) != len(values) or any(type(item) not in (int, float) or not math.isfinite(item)
                    or abs(item - target) > 1e-6 for target, item in zip(targets, values)):
                return False
    return True


class WaypointReference:
    """A fixed bounded servo; caller alone owns dispatch and physical cancellation."""
    def __init__(self, adapter: MorphologyAdapter):
        _require(isinstance(adapter, MorphologyAdapter), "waypoint_adapter", "accepted morphology adapter required")
        self.adapter, self.fence = adapter, StateFence()
        self.previous = SemanticAction()
        self._cache = None

    def cancel(self) -> None:
        self.fence.cancel()
        self.previous, self._cache = SemanticAction(), None

    def reset(self) -> None:
        # Memory clears without erasing retired source/lifetime evidence.
        self.cancel()

    def step(self, observation: Mapping[str, Any], goal: MorphologyGoal, *,
             source_generation: int = 1, mount_generation: int | None = None,
             current_tick: int | None = None) -> SemanticAction:
        try:
            _fixed_goal(goal)
            identity = self.adapter.identity(observation, source_generation=source_generation,
                                             mount_generation=mount_generation)
            source_hash = self.adapter.source_sha256(observation)
            if self._cache is not None and self._cache[0] == identity and self._cache[1] == goal:
                _require(source_hash == self._cache[2], "waypoint_source_changed",
                         "the same source identity cannot carry changed local input")
                self.adapter.frame(observation, goal, previous=self.previous, started_tick=self.fence.start_tick,
                                   source_generation=source_generation, mount_generation=mount_generation,
                                   current_tick=current_tick)
                return self._cache[3]
            reasons = self.fence.accept(identity, goal)
            if reasons:
                self.previous = SemanticAction()
            self.adapter.frame(observation, goal, previous=self.previous, started_tick=self.fence.start_tick,
                               source_generation=source_generation, mount_generation=mount_generation,
                               current_tick=current_tick)
            position, velocity, angle, omega, _ = _root(observation)
            delta = tuple(target - current for target, current in zip(goal.target, position))
            inside = math.hypot(*delta) <= goal.tolerance
            speed = _number(self.adapter.profile["profile"]["speed"], "profile.speed")
            scale = max(5., min(240., speed * goal.pace))
            limit = min(30., scale)
            desired = (0., 0.) if inside else tuple(_clip(1.8 * item, limit) for item in delta)
            drive = tuple(_clip((wanted - .35 * actual) / scale) for wanted, actual in zip(desired, velocity))
            brake = (self.adapter.capability["brake"] is not None and math.hypot(*velocity) > 2.
                     and (inside or sum(v * d for v, d in zip(velocity, delta)) < 0.))
            action = SemanticAction(drive_x=drive[0], drive_y=drive[1] if self.adapter.action_mask[1] else 0.,
                                    trim=_clip(-1.6 * angle - .25 * omega), effort=goal.pace, brake=brake)
            self.previous = action
            self._cache = (identity, goal, source_hash, action)
            return action
        except (ContractError, KeyError, TypeError, ValueError):
            self.cancel()
            raise


class WaypointTask:
    """Score only complete actual intervals; never supply model inputs or game work."""
    def __init__(self, adapter: MorphologyAdapter, goal: MorphologyGoal, initial: Mapping[str, Any]):
        _fixed_goal(goal)
        self.adapter, self.goal = adapter, goal
        self._identity = adapter.identity(initial)
        position, _, _, _, support = _root(initial)
        self._initial_tick = self._identity.tick
        self._last_distance = _number(math.dist(position, goal.target), "initial_distance")
        self._sequence = None
        self.elapsed_ticks = self.quiet_ticks = self.failed_admissions = 0
        self.succeeded = self.deadline_missed = self.integrity_failure = False
        self.success_tick = self.first_response_tick = None
        self.trim_reversals = self._trim_sign = 0
        self.absolute_angle_sum = self.peak_absolute_angle = self.peak_useful_speed = 0.
        self.inference_ns: list[int] = []
        self.reward_totals = {key: 0. for key in ("progress", "time", "request_cost", "arrival", "deadline")}
        self._support = support
        self._observed_support_loss = False
        self._landing_tick = None
        self._landing_quiet = 0
        self._settled_after_landing = None

    @property
    def done(self) -> bool:
        return self.succeeded or self.deadline_missed or self.integrity_failure

    def _validate_trace(self, action: WireAction, trace: Sequence[Mapping[str, Any]]):
        _require(isinstance(action, WireAction), "waypoint_action", "typed bounded wire action required")
        _require(not any((action.auxiliary, action.tool, action.reel, action.release_usable, action.attach_usable))
                 and (self.adapter.action_mask[1] or action.move[1] == 0.),
                 "waypoint_capability", "only the frozen semantic capability map is available")
        _require(isinstance(trace, (list, tuple)) and len(trace) == 6,
                 "waypoint_cadence", "one complete six-tick request interval required")
        source = self._identity
        _require(self.elapsed_ticks + len(trace) <= self.goal.deadline_ticks,
                 "waypoint_deadline", "native receipts cannot cross the observable task deadline")
        validated, sequence = [], None
        for offset, row in enumerate(trace):
            _require(isinstance(row, Mapping), "waypoint_receipt", "actual native receipt required")
            tick = source.tick + offset
            for field, expected in (("tick_before", tick), ("tick_after", tick + 1), ("world_tick", tick + 1),
                                    ("cellular_sample_tick", tick)):
                _require(type(row.get(field)) is int and row[field] == expected,
                         "waypoint_cadence", "contiguous native physics/material schedule required")
            _require(row.get("failed") is False, "waypoint_integrity", "a failed host tick is not task progress")
            received, completed = row.get("received_monotonic_usec"), row.get("completed_monotonic_usec")
            _require(type(received) is int and type(completed) is int and 0 <= received <= completed,
                     "waypoint_clock", "ordered actual receipt timestamps required")
            admissions, requests = row.get("admission"), row.get("requested")
            _require(isinstance(admissions, (list, tuple)) and isinstance(requests, (list, tuple)),
                     "waypoint_admission", "explicit admission and original request lists required")
            refused = sum(not isinstance(item, Mapping) or item.get("accepted") is not True for item in admissions)
            self.failed_admissions += refused
            _require(refused == 0, "waypoint_admission", "refused requests cannot become learning reward")
            if offset == 0:
                _require(len(admissions) == len(requests) == 1 and admissions[0].get("reason") == "",
                         "waypoint_admission", "one fresh unrefused original request required")
                packet = requests[0]
                _require(isinstance(packet, Mapping), "waypoint_request", "original issued packet required")
                for field, expected in (("version", 2), ("kind", "physical_wire_action"), ("run_id", source.run_id),
                                        ("epoch", source.epoch), ("actor_id", source.actor_id),
                                        ("observation_id", source.observation_id), ("observed_tick", source.tick),
                                        ("lease_ticks", min(12, self.goal.deadline_ticks - self.elapsed_ticks))):
                    actual = packet.get(field)
                    # Godot preserves the copied JSON request with integral
                    # floats. The separately recorded original issued packet
                    # and every actual native observation/counter stay strict.
                    numeric_echo = type(expected) is int and type(actual) in (int, float) and math.isfinite(actual)
                    _require((numeric_echo or type(actual) is type(expected)) and actual == expected,
                             "waypoint_source", "request must name the unchanged original source and lease")
                sequence = packet.get("sequence")
                _require(type(sequence) in (int, float) and math.isfinite(sequence)
                         and 0 <= sequence <= 9_007_199_254_740_991 and int(sequence) == sequence
                         and (self._sequence is None or sequence == self._sequence + 1)
                         and _same_action(action, packet.get("action")),
                         "waypoint_request", "consecutive issued action and actual requested wire must agree")
                sequence = int(sequence)
            else:
                _require(not admissions and not requests, "waypoint_replacement", "no replacement inside the held interval")
            view, applied = row.get("observation"), row.get("applied")
            _require(isinstance(view, Mapping) and isinstance(applied, Mapping),
                     "waypoint_receipt", "original observation and explicit actuation availability required")
            terminal = view.get("suspended") is True
            identity = self.adapter.terminal_identity(view) if terminal else self.adapter.identity(view)
            _require(identity.tick == tick + 1 and identity.lifetime == source.lifetime
                     and identity.control_generation == source.control_generation + int(terminal)
                     and (terminal or identity.topology_sha256 == source.topology_sha256),
                     "waypoint_boundary", "native motion cannot change source, owner or topology within an invocation")
            if terminal:
                _require(offset == 5 and tick + 1 - self._initial_tick == self.goal.deadline_ticks
                         and applied.get("action_available") is False and applied.get("terminal_revocation") is True
                         and applied.get("action") is None,
                         "waypoint_terminal", "only the final intrinsic deadline may revoke the native readback")
            else:
                _require(applied.get("action_available") is True and applied.get("terminal_revocation") is False
                         and _same_action(action, applied.get("action")),
                         "waypoint_consumption", "actual held readback must match the issued request")
            sample = _root(view, terminal=terminal)
            _number(math.dist(sample[0], self.goal.target), "distance")
            validated.append((identity, sample))
        return validated, sequence

    def consume(self, wire_action: WireAction, actual_trace: Sequence[Mapping[str, Any]], *,
                inference_ns: int = 0) -> dict[str, float]:
        _require(not self.done, "waypoint_finished", "a completed or invalid task cannot be rewarded twice")
        try:
            _require(type(inference_ns) is int and inference_ns >= 0,
                     "waypoint_inference", "actual nonnegative inference nanoseconds required")
            rows, sequence = self._validate_trace(wire_action, actual_trace)
        except (ContractError, KeyError, TypeError, ValueError):
            self.integrity_failure = True
            raise
        terms = {key: 0. for key in self.reward_totals}
        squared_request = sum(item * item for item in (*wire_action.move, wire_action.trim, wire_action.effort))
        self.inference_ns.append(inference_ns)
        sign = 1 if wire_action.trim >= .05 else -1 if wire_action.trim <= -.05 else 0
        if sign:
            self.trim_reversals += int(self._trim_sign != 0 and self._trim_sign != sign)
            self._trim_sign = sign
        for identity, (position, velocity, angle, _, support) in rows:
            self._identity = identity
            self.elapsed_ticks = identity.tick - self._initial_tick
            delta = tuple(target - current for target, current in zip(self.goal.target, position))
            distance, speed = math.hypot(*delta), math.hypot(*velocity)
            terms["progress"] += (self._last_distance - distance) / 48.
            self._last_distance = distance
            terms["time"] -= .01 / 60.
            terms["request_cost"] -= .002 * squared_request / 60.
            useful = max(0., sum(v * (d / distance) for v, d in zip(velocity, delta))) if distance else 0.
            self.peak_useful_speed = max(self.peak_useful_speed, useful)
            if self.first_response_tick is None and useful >= 3.:
                self.first_response_tick = self.elapsed_ticks
            angle = abs(angle)
            self.absolute_angle_sum += angle
            self.peak_absolute_angle = max(self.peak_absolute_angle, angle)
            settled = speed <= 6. and angle <= .35
            quiet = distance <= 4. and settled
            self.quiet_ticks = self.quiet_ticks + 1 if quiet else 0
            if not self.succeeded and self.quiet_ticks >= 12:
                self.succeeded, self.success_tick = True, self.elapsed_ticks
                terms["arrival"] += 1.
            if self.elapsed_ticks == 240 and not self.succeeded:
                self.deadline_missed = True
                terms["deadline"] -= .5
            if self._support is True and support is False:
                self._observed_support_loss = True
                self._landing_tick, self._landing_quiet = None, 0
            if self._observed_support_loss and support is True:
                self._observed_support_loss = False
                self._landing_tick, self._landing_quiet = self.elapsed_ticks, 0
            if self._landing_tick is not None:
                self._landing_quiet = self._landing_quiet + 1 if support is True and settled else 0
                if self._landing_quiet >= 12 and self._settled_after_landing is None:
                    self._settled_after_landing = (self.elapsed_ticks - self._landing_tick) / 60.
            self._support = support
        self._sequence = sequence
        for key, value in terms.items():
            self.reward_totals[key] += value
        terms["total"] = sum(terms.values())
        return terms

    def metrics(self, *, administratively_capped: bool = False) -> dict[str, Any]:
        _require(type(administratively_capped) is bool, "waypoint_cap", "explicit administrative cap flag required")
        def quantile(fraction):
            if not self.inference_ns:
                return None
            ordered = sorted(self.inference_ns)
            position = (len(ordered) - 1) * fraction
            low, high = math.floor(position), math.ceil(position)
            return (ordered[low] + (ordered[high] - ordered[low]) * (position - low)) / 1e6
        success = self.succeeded and not self.integrity_failure
        return {"success": success,
                "reason": "integrity_failure" if self.integrity_failure else "arrival" if success
                          else "task_deadline" if self.deadline_missed else "administrative_cutoff" if administratively_capped else "incomplete",
                "administratively_capped": administratively_capped, "integrity_failure": self.integrity_failure,
                "failed_admissions": self.failed_admissions, "elapsed_ticks": self.elapsed_ticks,
                "return": sum(self.reward_totals.values()), "reward_terms": dict(self.reward_totals),
                "final_distance_cells": self._last_distance,
                "success_seconds": self.success_tick / 60. if success else None,
                "capped_completion_seconds": self.success_tick / 60. if success else 4.,
                "response_seconds": self.first_response_tick / 60. if self.first_response_tick is not None else None,
                "peak_useful_speed_cells_per_second": self.peak_useful_speed,
                "ordinary_trim_reversals": self.trim_reversals,
                "ordinary_trim_reversals_per_second": self.trim_reversals * 60. / max(1, self.elapsed_ticks),
                "mean_absolute_angle_radians": self.absolute_angle_sum / self.elapsed_ticks if self.elapsed_ticks else None,
                "peak_absolute_angle_radians": self.peak_absolute_angle if self.elapsed_ticks else None,
                "settled_after_landing_seconds": self._settled_after_landing,
                "inference_p50_ms": quantile(.5), "inference_p95_ms": quantile(.95),
                "owner_feel_accepted": False}


def qualification_cases(profiles: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Resolve only the two training profiles; heldout profile geometry is unused."""
    spec = TASK_SPEC["qualification"]
    result = []
    for name, profile, seed, x, dx in zip(spec["case_ids"], spec["profiles"], spec["host_seeds"],
                                         spec["spawn_x"], spec["target_dx"]):
        row = profiles["profiles"][profile]
        adapter = MorphologyAdapter(row["definition"], expected_sha256=row["profile_sha256"])
        adapter.identity(row["observation"])
        root_position, _, _, _, _ = _root(row["observation"])
        clearance = _number(adapter.profile["spawn_floor_clearance"], "spawn_floor_clearance")
        offset_x = _pair(adapter.parts[0]["offset"], "root.offset")[0]
        result.append({"id": name, "profile": profile, "spawn": [x, 352. - clearance],
                       "target": [x + offset_x + dx, root_position[1]], "host_seed": seed})
    return result


def evaluate_qualification(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Conservative row predicate, subordinate to the caller's raw-artifact audit."""
    spec = TASK_SPEC["qualification"]
    expected = dict(zip(spec["case_ids"], zip(spec["profiles"], spec["host_seeds"])))
    errors, checks, seen = [], {}, set()
    if not isinstance(rows, (list, tuple)):
        return {"complete": False, "passed": False, "errors": ["qualification rows must be a list"], "checks": {}}
    for index, row in enumerate(rows):
        label = f"row[{index}]"
        row_errors = []
        def check(ok, reason):
            if not ok:
                row_errors.append(reason)
        if not isinstance(row, Mapping):
            errors.append(f"{label}: missing result object")
            continue
        name = row.get("scenario_family")
        check(type(name) is str and name in expected and name not in seen, "unknown or duplicate case")
        if type(name) is str and name in expected:
            seen.add(name)
            profile, seed = expected[name]
            check(row.get("profile") == profile and type(row.get("host_seed")) is int and row["host_seed"] == seed,
                  "wrong profile or frozen host seed")
        check(row.get("kind") == "reference" and row.get("phase") == "qualification", "wrong controller or phase")
        check(row.get("policy_sha256") == sha256_json(REFERENCE_SPEC)
              and row.get("task_spec_sha256") == sha256_json(TASK_SPEC), "reference or task specification differs")
        check("error" in row and row["error"] is None and row.get("actual_owner_closed") is True,
              "episode error or missing actual owner closure")
        check(type(row.get("source_boundary_regressions")) is int and row["source_boundary_regressions"] == 0,
              "source boundary regressions are nonzero or unavailable")
        for key in ("record_sha256", "artifact_sha256", "boundary_audit_sha256", "profile_sha256"):
            value = row.get(key)
            check(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                  f"missing {key}")
        decisions, ticks = row.get("decisions"), row.get("completed_ticks")
        counters = type(decisions) is int and 1 <= decisions <= 40 and type(ticks) is int and ticks == decisions * 6
        check(counters, "incomplete or out-of-allocation native counters")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            row_errors.append("missing task metrics")
        else:
            for key, value in (("success", True), ("administratively_capped", False),
                               ("integrity_failure", False), ("owner_feel_accepted", False)):
                check(metrics.get(key) is value, f"invalid {key}")
            check(metrics.get("reason") == "arrival", "no quiet-arrival outcome")
            check(type(metrics.get("failed_admissions")) is int and metrics["failed_admissions"] == 0, "failed or missing admissions")
            check(type(metrics.get("elapsed_ticks")) is int and counters and metrics["elapsed_ticks"] == ticks,
                  "task and native counters differ")
            def finite(key, minimum, maximum):
                value = metrics.get(key)
                return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum
            check(finite("success_seconds", .2, 4.) and counters and metrics.get("success_seconds", 5.) * 60 <= ticks
                  and finite("capped_completion_seconds", .2, 4.)
                  and metrics.get("capped_completion_seconds") == metrics.get("success_seconds"), "arrival time is invalid")
            check(finite("inference_p50_ms", 0., 5.) and finite("inference_p95_ms", 0., 5.)
                  and metrics.get("inference_p50_ms", 6.) <= metrics.get("inference_p95_ms", -1.), "unmeasured or excessive inference latency")
            for key in ("ordinary_trim_reversals_per_second", "mean_absolute_angle_radians", "peak_useful_speed_cells_per_second"):
                check(finite(key, 0., float("inf")), f"unmeasured or invalid {key}")
        key = name if type(name) is str and name not in checks else label
        checks[key] = {"passed": not row_errors, "errors": row_errors}
        errors.extend(f"{key}: {error}" for error in row_errors)
    missing = sorted(set(expected) - seen)
    if missing:
        errors.append("missing cases: " + ", ".join(missing))
    complete = len(rows) == 4 and len(seen) == 4
    return {"complete": complete, "passed": complete and not errors, "errors": errors, "checks": checks,
            "missing_cases": missing, "maximum_game_requests": 160,
            "task_spec_sha256": sha256_json(TASK_SPEC), "reference_spec_sha256": sha256_json(REFERENCE_SPEC),
            "evidence_limit": spec["evidence_limit"]}


def summarize_comparison(rows, declaration, *, run_completed: bool):
    """Stable pure orchestration entry point; the comparison has its own module."""
    from .morphology_metrics import summarize_comparison as summarize
    return summarize(rows, declaration, run_completed=run_completed)
