"""Repeatable authored-control fixtures and honest response measurements.

Fixture descriptions are requests to a host-owned test harness, not a simulator.
Unsupported perturbations are explicit failures, never successful no-op tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Protocol

from nca.contracts import WireAction
from nca.controllers import AuthoredController, BaselineTuning, FeedbackObservation, SkillGoal
from nca.controllers.shaping import finite, unit


@dataclass(frozen=True)
class FixtureEvent:
    tick: int
    kind: str
    parameters: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if type(self.tick) is not int or self.tick < 0 or self.tick % 6:
            raise ValueError("fixture events must be on a 10 Hz decision boundary")
        counts = {"goal": 2, "cancel": 0, "angular_disturbance": 1,
                  "remove_support": 0, "restore_support": 0, "soft_stop": 2,
                  "recoil": 2, "payload_change": 1}
        if self.kind not in counts or len(self.parameters) != counts[self.kind]:
            raise ValueError("unknown fixture event or wrong parameter shape")
        object.__setattr__(self, "parameters", tuple(finite(value, "fixture parameter") for value in self.parameters))


@dataclass(frozen=True)
class FeelFixture:
    id: str
    actor: str
    duration_ticks: int
    events: tuple[FixtureEvent, ...]
    seed: int = 1703
    base_hz: int = 60
    decision_ticks: int = 6

    def __post_init__(self) -> None:
        if not self.id or self.actor not in ("jev", "astra"):
            raise ValueError("fixture requires an identity and declared actor")
        if type(self.duration_ticks) is not int or self.duration_ticks < 6 or self.duration_ticks % 6:
            raise ValueError("fixture duration must be a positive decision multiple")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.base_hz != 60 or self.decision_ticks != 6:
            raise ValueError("v1 fixtures freeze 60 Hz host and 10 Hz decisions")
        object.__setattr__(self, "events", tuple(self.events))
        ticks = [event.tick for event in self.events]
        if ticks != sorted(ticks) or any(tick >= self.duration_ticks for tick in ticks):
            raise ValueError("fixture events must be ordered within its duration")

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()


def fixture_suite(actor: str = "jev") -> tuple[FeelFixture, ...]:
    """Freeze eight short scenarios; perturbation units belong to the host.

    Goal offsets are in host cells, recoil values are requested velocity changes
    (cells/s) and angular disturbance is requested angular speed (rad/s). A host
    fixture owner must acknowledge and record their realized effects separately.
    """
    e = FixtureEvent
    return (
        FeelFixture("step/v1", actor, 360, (e(0, "goal", (48.0, 0.0)),)),
        FeelFixture("pulse/v1", actor, 240, (e(0, "goal", (48.0, 0.0)), e(30, "cancel"))),
        FeelFixture("reversal/v1", actor, 480, (e(0, "goal", (48.0, 0.0)), e(180, "goal", (-32.0, 0.0)))),
        FeelFixture("neutral-disturbance/v1", actor, 300,
                    (e(0, "cancel"), e(60, "angular_disturbance", (1.2,)), e(120, "goal", (0.0, 0.0)))),
        FeelFixture("support-loss/v1", actor, 360,
                    (e(0, "goal", (48.0, 0.0)), e(60, "remove_support"), e(180, "restore_support"))),
        FeelFixture("soft-stop/v1", actor, 360,
                    (e(0, "soft_stop", (-.3, .3)), e(0, "goal", (48.0, 0.0)), e(180, "goal", (-24.0, 0.0)))),
        FeelFixture("recoil-payload/v1", actor, 420,
                    (e(0, "goal", (48.0, 0.0)), e(90, "payload_change", (1.5,)), e(180, "recoil", (-14.0, -5.0)))),
        FeelFixture("recovery/v1", actor, 300,
                    (e(0, "angular_disturbance", (4.5,)), e(0, "goal", (0.0, 0.0)))),
    )


@dataclass(frozen=True)
class MotionSample:
    tick: int
    position: tuple[float, float]
    velocity: tuple[float, float]
    angle: float
    angular_velocity: float
    requested: WireAction | None
    phase: str
    supported: bool | None
    response_limited: bool = False
    valid: bool = True

    def __post_init__(self) -> None:
        if type(self.tick) is not int or self.tick < 0:
            raise ValueError("sample tick must be nonnegative")
        for vector in (self.position, self.velocity):
            if not isinstance(vector, (tuple, list)) or len(vector) != 2:
                raise ValueError("motion samples require two-dimensional vectors")
            for value in vector:
                finite(value, "sample motion")
        object.__setattr__(self, "position", tuple(self.position))
        object.__setattr__(self, "velocity", tuple(self.velocity))
        finite(self.angle, "angle")
        finite(self.angular_velocity, "angular_velocity")
        if self.requested is not None and not isinstance(self.requested, WireAction):
            raise ValueError("sample request must use the wire contract")
        if self.supported is not None and type(self.supported) is not bool:
            raise ValueError("support must be bool or unavailable")
        if type(self.valid) is not bool or type(self.response_limited) is not bool:
            raise ValueError("sample validity must be boolean")


@dataclass(frozen=True)
class ResponseMetrics:
    sample_count: int
    invalid_samples: int
    duration_seconds: float
    response_delay_seconds: float | None
    peak_speed: float
    net_useful_progress: float | None
    maximum_overshoot: float
    settling_seconds: float | None
    corrective_sign_changes: int
    request_variation_per_second: float
    request_slew_limited_fraction: float


def measure_response(samples: Iterable[MotionSample], *, origin: tuple[float, float],
                     target: tuple[float, float], command_tick: int = 0,
                     base_hz: int = 60, response_speed: float = 2.0,
                     tolerance: float = 4.0, settle_speed: float = 6.0,
                     settling_dwell: float = .2) -> ResponseMetrics:
    """Measure one fixed-target segment, with declared thresholds.

    Reversal fixtures must be split into their two target segments. Progress is
    final signed displacement, so oscillation cannot farm a movement score.
    Chatter counts trim reversals only outside launch/recoil/recovery phases.
    None means a response or settle event was not observed, never zero latency.
    Invalid trajectories retain their failure count and cannot earn timing or
    useful-progress measurements from a selected valid subset.
    """
    rows = tuple(samples)
    if not rows:
        raise ValueError("cannot measure an empty trace")
    if type(base_hz) is not int or base_hz < 1 or type(command_tick) is not int or command_tick < 0:
        raise ValueError("invalid clock")
    for name, value in (("response_speed", response_speed), ("tolerance", tolerance),
                        ("settle_speed", settle_speed), ("settling_dwell", settling_dwell)):
        if finite(value, name) < 0.0:
            raise ValueError("metric thresholds must be nonnegative")
    if len(origin) != 2 or len(target) != 2:
        raise ValueError("metric reference points must be two-dimensional")
    origin = tuple(finite(value, "origin") for value in origin)
    target = tuple(finite(value, "target") for value in target)
    if any(second.tick <= first.tick for first, second in zip(rows, rows[1:])):
        raise ValueError("samples must have unique increasing ticks")
    if rows[0].tick < command_tick:
        raise ValueError("a response segment must start at or after its command")
    direction = unit((target[0] - origin[0], target[1] - origin[1]))
    length = math.dist(origin, target)
    projections = [sum((row.position[i] - origin[i]) * direction[i] for i in range(2)) for row in rows]
    duration = (rows[-1].tick - rows[0].tick) / base_hz
    response = next(((row.tick - command_tick) / base_hz for row in rows
                     if row.valid and sum(row.velocity[i] * direction[i] for i in range(2)) >= response_speed), None)
    invalid = sum(not row.valid for row in rows)
    settle_start: int | None = None
    for row in rows:
        quiet = row.valid and math.dist(row.position, target) <= tolerance and math.hypot(*row.velocity) <= settle_speed
        if not quiet:
            settle_start = None
        elif settle_start is None:
            settle_start = row.tick
    settled = ((settle_start - command_tick) / base_hz if settle_start is not None and
               (rows[-1].tick - settle_start) / base_hz + 1e-12 >= settling_dwell else None)
    variation = sum(math.hypot(second.requested.move[0] - first.requested.move[0],
                               second.requested.move[1] - first.requested.move[1]) +
                    abs(second.requested.trim - first.requested.trim)
                    for first, second in zip(rows, rows[1:])
                    if first.requested is not None and second.requested is not None)
    changes = 0
    previous_sign = 0
    for row in rows:
        if row.requested is None or row.phase in {"launch", "burst", "recoil", "recovery", "airborne"}:
            previous_sign = 0
            continue
        sign = 1 if row.requested.trim > .05 else -1 if row.requested.trim < -.05 else 0
        if sign and previous_sign and sign != previous_sign:
            changes += 1
        if sign:
            previous_sign = sign
    return ResponseMetrics(len(rows), invalid, duration, None if invalid else response,
                           max(math.hypot(*row.velocity) for row in rows),
                           projections[-1] if not invalid else None,
                           max(0.0, max(projections) - length), None if invalid else settled,
                           changes, variation / duration if duration else 0.0,
                           sum(row.response_limited for row in rows) / len(rows))


class FixtureHost(Protocol):
    """A thin wrapper around a compatible host, including its evidence lane."""

    evidence_lane: str

    def observe(self) -> Mapping[str, Any]: ...
    def step(self, request: WireAction, ticks: int) -> Mapping[str, Any]: ...
    def cancel(self) -> None: ...
    def fixture_event(self, event: FixtureEvent) -> Mapping[str, Any]: ...


class CybersandFixtureHost:
    """Use only public accepted actor-owner APIs for ordinary command fixtures.

    The caller resets a CybersandEnv with the fixture's seed/config first. The
    accepted host exposes no arbitrary test-force, payload or support setter;
    those perturbations are explicitly refused by this wrapper.
    """

    evidence_lane = "physical"

    def __init__(self, environment: Any) -> None:
        if environment.host.lane != "act004" or environment.host.mode != "stepped":
            raise ValueError("feel fixtures require the ACT-004 stepped owner")
        self.environment = environment
        self.identity = environment.host.identity
        self.receipts: list[dict[str, Any]] = []

    def observe(self) -> Mapping[str, Any]:
        return self.environment.observe()

    def step(self, request: WireAction, ticks: int) -> Mapping[str, Any]:
        packet = self.environment.command(request.to_dict())
        view, _, terminated, truncated, info = self.environment.step_packet(packet, ticks=ticks)
        self.receipts.append({"requested_packet": packet, "tick": view["tick"],
                              "actual_ticks": info["actual_ticks"], "trace": info["trace"],
                              "terminated": terminated, "truncated": truncated, "reason": info["reason"]})
        if terminated or truncated:
            view["terminal_reason"] = info["reason"]
        return view

    def cancel(self) -> None:
        self.environment.cancel()

    def fixture_event(self, event: FixtureEvent) -> Mapping[str, Any]:
        return {"accepted": False, "event": event.kind,
                "reason": "accepted host has no public arbitrary perturbation interface"}


def run_fixture(host: FixtureHost, fixture: FeelFixture, *,
                tuning: BaselineTuning | None = None) -> dict[str, Any]:
    """Run a pre-reset host once. Never emulate an unsupported host perturbation.

    The wrapper must return the last valid observation on termination and mark
    it with terminal_reason; partial intervals are retained and not stepped on.
    Capture/replay receipts belong to the data module or wrapper independently.
    """
    if host.evidence_lane not in {"recorded", "sampled", "physical"}:
        raise ValueError("host must declare recorded, sampled or physical evidence")
    if host.evidence_lane == "physical" and not getattr(host, "identity", {}).get("host_revision"):
        raise ValueError("physical fixture needs an identified actual host")
    controller = AuthoredController(fixture.actor, tuning)
    view = host.observe()
    observation = FeedbackObservation.from_host(view)
    if observation.tick != 0:
        raise ValueError("fixtures require a fresh episode at tick zero")
    origin = observation.position
    samples: list[dict[str, Any]] = []
    perturbations: list[Mapping[str, Any]] = []
    goal: SkillGoal | None = None
    reason = "duration_complete"
    for boundary in range(0, fixture.duration_ticks, fixture.decision_ticks):
        if observation.tick != boundary:
            raise ValueError("host violated the declared causal decision cadence")
        for index, event in enumerate(fixture.events):
            if event.tick != boundary:
                continue
            if event.kind == "goal":
                skill = "recover_level" if fixture.id in {"recovery/v1", "neutral-disturbance/v1"} else (
                    "directed_hop" if fixture.actor == "jev" else "burst_traverse")
                goal = SkillGoal(f"{fixture.id}:{index}", skill,
                                 target=(origin[0] + event.parameters[0], origin[1] + event.parameters[1]),
                                 timeout_seconds=fixture.duration_ticks / fixture.base_hz)
            elif event.kind == "cancel":
                host.cancel()
                controller.cancel()
                goal = None
            else:
                receipt = host.fixture_event(event)
                if receipt.get("accepted") is not True:
                    raise ValueError(f"host did not accept fixture perturbation {event.kind}")
                perturbations.append(dict(receipt))
        # Cancellation or perturbation can issue a different current view.
        observation = FeedbackObservation.from_host(host.observe())
        if goal is None:
            action = WireAction(effort=0.0)
            phase = "neutral"
            limited = False
            status, status_reason = "idle", "no_active_goal"
        else:
            frame = controller.step(observation, goal)
            action, phase, limited = frame.action, frame.phase, frame.response_limited
            status, status_reason = frame.status, frame.reason
        sample = MotionSample(observation.tick, observation.position, observation.velocity,
                              observation.angle, observation.angular_velocity, action, phase,
                              observation.supported, limited, observation.valid)
        sample_dict = asdict(sample)
        sample_dict["requested"] = action.to_dict()
        sample_dict.update(controller_status=status, status_reason=status_reason,
                           goal_id=goal.id if goal else None, goal_target=goal.target if goal else None)
        samples.append(sample_dict)
        next_view = host.step(action, fixture.decision_ticks)
        observation = FeedbackObservation.from_host(next_view)
        if next_view.get("terminal_reason"):
            reason = str(next_view["terminal_reason"])
            break
    # Keep the state after the last consumed interval, not merely its source.
    samples.append({"tick": observation.tick, "position": observation.position,
                    "velocity": observation.velocity, "angle": observation.angle,
                    "angular_velocity": observation.angular_velocity,
                    "requested": None, "phase": "final_observation",
                    "supported": observation.supported, "response_limited": False,
                    "valid": observation.valid})
    return {"schema": "nca.feel-trace/v1", "fixture": asdict(fixture),
            "fixture_hash": fixture.digest, "controller_id": controller.controller_id,
            "tuning_hash": controller.tuning.digest, "tuning": asdict(controller.tuning),
            "evidence_lane": host.evidence_lane, "owner_feel_judgment": "not_recorded",
            "host_identity": getattr(host, "identity", {}),
            "terminal_reason": reason, "perturbations": perturbations, "samples": samples}


def render_pose_comparison(traces: Mapping[str, Mapping[str, Any]], output: str) -> None:
    """Render matched recorded body poses as a GIF; this is not game footage.

    Pillow is an optional artifact dependency. Every rectangle, orientation and
    local terrain cell comes from a consumed tick receipt. All panels share the
    same fixed camera and scale; unobserved terrain is deliberately omitted.
    """
    from PIL import Image, ImageDraw, ImageFont

    if not 1 <= len(traces) <= 3:
        raise ValueError("a matched comparison supports one to three variants")
    runs = list(traces.values())
    hashes = {run["fixture_hash"] for run in runs}
    revisions = {run["host_identity"]["host_revision"] for run in runs}
    if len(hashes) != 1 or len(revisions) != 1 or any(run["evidence_lane"] != "physical" for run in runs):
        raise ValueError("pose clips need the same physical fixture and host revision")
    rows = [[tick["observation"] for receipt in run["consumed_receipts"] for tick in receipt["trace"]]
            for run in runs]
    if not rows or any(not row for row in rows):
        raise ValueError("missing actual body receipts")
    ticks = [row["tick"] for row in rows[0]]
    if any([row["tick"] for row in other] != ticks for other in rows[1:]):
        raise ValueError("comparison timelines differ; inspect partial runs separately")
    points = [part["position"] for row in rows for view in row for part in view["parts"]]
    low_x, high_x = min(point[0] for point in points) - 18, max(point[0] for point in points) + 18
    low_y, high_y = min(point[1] for point in points) - 18, max(point[1] for point in points) + 18
    panel_width, height = 420, 336
    scale = min(4.0, (panel_width - 36) / max(1.0, high_x - low_x), 224 / max(1.0, high_y - low_y))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
        small = ImageFont.truetype("DejaVuSans.ttf", 10)
    except OSError:
        font = small = ImageFont.load_default()
    terrain: list[dict[tuple[int, int], int]] = [{} for _ in rows]
    images = []
    for index in range(len(ticks)):
        for panel, observations in enumerate(rows):
            patch = observations[index].get("cell_patch", {})
            width = int(patch.get("size", [0, 0])[0])
            start = patch.get("origin", [0, 0])
            body_ids = patch.get("body_ids", [])
            for cell, material in enumerate(patch.get("materials", [])):
                if cell < len(body_ids) and body_ids[cell] != 0:
                    continue
                coordinate = int(start[0] + cell % width), int(start[1] + cell // width)
                terrain[panel][coordinate] = material
        if (index + 1) % 6:
            continue
        image = Image.new("RGB", (panel_width * len(rows), height), "#101722")
        draw = ImageDraw.Draw(image)
        for panel, (label, run) in enumerate(traces.items()):
            x_offset = panel * panel_width
            view = rows[panel][index]

            def pixel(point):
                return (x_offset + 18 + (point[0] - low_x) * scale,
                        64 + (point[1] - low_y) * scale)

            draw.text((x_offset + 14, 10), f"{run['fixture']['actor']} / {label}", font=font, fill="#eef5ff")
            draw.text((x_offset + 14, 30), f"Recorded ACT-004 poses | tick {view['tick']} | {view['tick']/60:.2f}s",
                      font=small, fill="#9badc8")
            for (x, y), material in terrain[panel].items():
                if not material:
                    continue
                first, second = pixel((x, y)), pixel((x + 1, y + 1))
                draw.rectangle((first, second), fill="#3c4758" if material == 1 else "#536373")
            first_goal = next((event for event in run["fixture"]["events"] if event["kind"] == "goal"), None)
            if first_goal is not None:
                active_goals = [event for event in run["fixture"]["events"]
                                if event["kind"] == "goal" and event["tick"] <= view["tick"]]
                event = active_goals[-1] if active_goals else first_goal
                origin = run["samples"][0]["position"]
                target = pixel((origin[0] + event["parameters"][0], origin[1] + event["parameters"][1]))
                draw.line((target[0] - 5, target[1], target[0] + 5, target[1]), fill="#d6a752", width=2)
                draw.line((target[0], target[1] - 5, target[0], target[1] + 5), fill="#d6a752", width=2)
            for part in view["parts"]:
                cosine, sine = math.cos(part["rotation"]), math.sin(part["rotation"])
                half_x, half_y = part["size"][0] / 2.0, part["size"][1] / 2.0
                corners = [pixel((part["position"][0] + x * cosine - y * sine,
                                  part["position"][1] + x * sine + y * cosine))
                           for x, y in ((-half_x, -half_y), (half_x, -half_y),
                                        (half_x, half_y), (-half_x, half_y))]
                draw.polygon(corners, fill="#f0748e" if part["index"] == 0 else "#75bde8", outline="#e7f1ff")
            velocity = math.hypot(*view["velocity"])
            draw.text((x_offset + 14, 296), f"speed {velocity:.1f} cells/s | support {bool(view['grounded'])}",
                      font=small, fill="#c3cee0")
            draw.text((x_offset + 14, 314), "Pose replay; local cells only; no owner feel verdict", font=small, fill="#9badc8")
            if panel:
                draw.line((x_offset, 0, x_offset, height), fill="#59687e")
        images.append(image)
    if not images:
        raise ValueError("clip contains fewer than six completed ticks")
    images[0].save(output, save_all=True, append_images=images[1:], duration=100,
                   loop=0, optimize=False, disposal=2)
