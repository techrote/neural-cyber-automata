"""Bounded nominal IK and a separately derived two-link comparator.

These routines neither measure a compliant host pose nor grant actuation. The
numerical routine never calls the analytic solution. Its finite local search can
stop without establishing that a target is geometrically impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Literal, Protocol

from nca.contracts import ContractError, Record
from .contracts import (IKAuthority, IKObservation, IKResult, IKTarget,
                        InfeasibilityProof, JointReference)
from .kinematics import PlanarChain, chain_from_profile, cpu_fk
from .profiles import IKProfile, nominal_outer_reach, nominal_pose


@dataclass(frozen=True)
class SolverConfig(Record):
    """One finite allocation; a call does not retry or enlarge these limits.

    damping is the maximum; each iteration uses damping times
    max(0.01, min(1, sqrt(residual_cells))) to retain boundary convergence.
    """

    max_iterations: int = 64
    max_seconds: float = 0.05
    damping: float = 0.02
    max_step: float = 0.25
    max_backtracks: int = 8
    tolerance_cells: float = 1e-6
    posture_weight: float = 0.002
    singular_probe: float = 0.05

    def validate(self) -> None:
        if not 0 <= self.max_iterations <= 256 or not 0 <= self.max_seconds <= 5:
            raise ContractError("ik_solver_budget", "at most 256 iterations and five seconds per call")
        if not 1 <= self.max_backtracks <= 16:
            raise ContractError("ik_solver_budget", "one to sixteen line-search trials required")
        if not 1e-8 <= self.damping <= 10 or not 0 < self.max_step <= 0.25:
            raise ContractError("ik_solver_step", "positive bounded damping and trust step required")
        if not 0 < self.tolerance_cells or not 0 <= self.posture_weight <= 0.1:
            raise ContractError("ik_solver_tolerance", "invalid tolerance or posture weight")
        if not 0 < self.singular_probe <= self.max_step:
            raise ContractError("ik_solver_step", "singular probe must fit inside the trust step")


@dataclass(frozen=True)
class NumericSolution(Record):
    """Diagnostic candidate; only solved/best_effort are actionable statuses.

    Timed-out candidates are retained for inspection, never emitted as an
    IKResult reference. Residual uses only selected position task components.
    Iterative correction and initial/bound projection are separately counted.
    """

    status: Literal["solved", "best_effort", "iteration_budget_exhausted",
                    "time_budget_exhausted", "invalid_input", "unsupported_profile",
                    "missing_required_measurement", "nonfinite_result"]
    reason: str
    configuration: tuple[float, ...] | None = None
    point: tuple[float, float] | None = None
    orientation_radians: float | None = None
    residual_cells: float | None = None
    iterations: int = 0
    evaluations: int = 0
    active_limits: tuple[int, ...] = ()
    projected_coordinates: tuple[int, ...] = ()
    projection_events: int = 0
    projection_correction_norm: float = 0.0
    correction_norm: float = 0.0
    maximum_step_norm: float = 0.0
    elapsed_seconds: float = 0.0
    emission_adjusted_coordinates: tuple[int, ...] = ()

    @property
    def converged(self) -> bool:
        return self.status == "solved"

    @property
    def actionable(self) -> bool:
        return self.status in ("solved", "best_effort")


def _vector(value, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ContractError("ik_shape", f"{name} must follow the declared coordinate order")
    if any(type(v) not in (float, int) for v in value):
        raise ContractError("ik_numeric", f"{name} must contain finite numbers")
    try:
        result = tuple(float(v) for v in value)
    except OverflowError as exc:
        raise ContractError("ik_numeric", f"{name} is nonfinite") from exc
    if not all(math.isfinite(v) for v in result):
        raise ContractError("ik_numeric", f"{name} is nonfinite")
    return result


def _mask(value, size: int, default, name: str) -> tuple[bool, ...]:
    result = tuple(default) if value is None else value
    if (not isinstance(result, (tuple, list)) or len(result) != size
            or any(type(v) is not bool for v in result)):
        raise ContractError("ik_mask", f"{name} must be an explicit boolean mask")
    return tuple(result)


def _bounds(chain, lower, upper):
    n = len(chain.joints)
    lo = tuple(j.lower for j in chain.joints) if lower is None else _vector(lower, n, "lower")
    hi = tuple(j.upper for j in chain.joints) if upper is None else _vector(upper, n, "upper")
    if any(not j.lower <= a <= b <= j.upper for j, a, b in zip(chain.joints, lo, hi)):
        raise ContractError("ik_bounds", "solve bounds must be a nonempty subset of hard bounds")
    return lo, hi


def _equivalents(value: float, lower: float, upper: float) -> tuple[float, ...]:
    """Equivalent rotations inside an interval; no shortcut through its stops."""
    # Reduce first so even a finite enormous source angle cannot allocate a huge
    # enumeration or lose a small interval in a huge integer multiplication.
    value = math.remainder(value, math.tau)
    start = math.ceil((lower - value) / math.tau - 1e-13)
    stop = math.floor((upper - value) / math.tau + 1e-13)
    return tuple(min(upper, max(lower, value + k * math.tau))
                 for k in range(start, stop + 1))


def _initial_coordinate(value, joint, lower, upper):
    if lower <= value <= upper:
        return value
    if joint.kind == "rotary":
        choices = _equivalents(value, lower, upper)
        if choices:
            return min(choices, key=lambda candidate: abs(candidate - value))
        value = math.remainder(value, math.tau)
    return min(upper, max(lower, value))


def _left_solve(jacobian, vector, damping):
    """J.T (J J.T + damping**2 I)^-1 vector for a two-row task."""
    scale = max(1.0, *(abs(v) for row in jacobian for v in row))
    x, y = ([v / scale for v in row] for row in jacobian)
    ex, ey = vector[0] / scale, vector[1] / scale
    lam = max(damping / scale, 1e-12)
    # (a+l)(d+l)-b² is evaluated as a*d-b² + l*(a+d+l).
    a, b, d = sum(v * v for v in x), sum(u * v for u, v in zip(x, y)), sum(v * v for v in y)
    regularizer = lam * lam
    det = max(0.0, a * d - b * b) + regularizer * (a + d + regularizer)
    ux = ((d + regularizer) * ex - b * ey) / det
    uy = ((a + regularizer) * ey - b * ex) / det
    return tuple(u * ux + v * uy for u, v in zip(x, y))


def _trust(step, maximum):
    norm = math.hypot(*step)
    factor = min(1.0, maximum / norm) if norm > 0 else 1.0
    return tuple(v * factor for v in step)


def solve_numeric(chain: PlanarChain, target, current, *, config: SolverConfig | None = None,
                  lower=None, upper=None, posture=None, present=None, measured=None,
                  active=None, task_mask=(True, True)) -> NumericSolution:
    """Bounded damped least squares from the current configuration.

    A fixed active-set pass keeps steps pointing into joint intervals. A bounded
    line search accepts endpoint improvement; at a singularity a small projected
    null-space probe may escape a flat first derivative. Neither analytic IK nor
    random restarts is used. All coordinates on the path must be measured even
    when their Jacobian columns are masked from the solve.
    """
    started = time.perf_counter()

    def reject(status, reason):
        return NumericSolution(status, reason, elapsed_seconds=max(0.0, time.perf_counter() - started))

    config = SolverConfig() if config is None else config
    if not isinstance(config, SolverConfig):
        return reject("invalid_input", "solver configuration type is unsupported")
    if not isinstance(chain, PlanarChain):
        return reject("unsupported_profile", "a validated planar serial chain is required")
    n = len(chain.joints)
    try:
        target = _vector(target, 2, "target")
        lo, hi = _bounds(chain, lower, upper)
        declared = tuple(j.participation == "active" for j in chain.joints)
        present = _mask(present, n, (True,) * n, "presence")
        measured = _mask(measured, n, (True,) * n, "measurement")
        active = _mask(active, n, declared, "activity")
        task_mask = _mask(task_mask, 2, (True, True), "task")
        if any(a and (not p or not d) or m and not p
               for a, p, d, m in zip(active, present, declared, measured)) or not any(task_mask):
            raise ContractError("ik_mask", "activity, presence and task masks are inconsistent")
        if not all(present) or not all(measured):
            return reject("missing_required_measurement", "the complete endpoint path must be measured")
        current = _vector(current, n, "current")
        preferred = current if posture is None else _vector(posture, n, "posture")
    except ContractError as exc:
        return reject("invalid_input", exc.code)

    projected: set[int] = set()
    projection_events = 0
    projection_norm = 0.0
    q = tuple(_initial_coordinate(v, j, a, b) for v, j, a, b in zip(current, chain.joints, lo, hi))
    preferred = tuple(_initial_coordinate(v, j, a, b) for v, j, a, b in zip(preferred, chain.joints, lo, hi))
    # Inactive coordinates describe measured state, not a new actuator. They may
    # not be projected to invent a different passive configuration.
    if any(not enabled and v != initial for enabled, v, initial in zip(active, current, q)):
        return reject("invalid_input", "inactive coordinates cannot be projected")
    differences = tuple(a - b for a, b in zip(q, current))
    if any(differences):
        projected.update(i for i, value in enumerate(differences) if value != 0)
        projection_events = 1
        projection_norm = math.hypot(*differences)
        if not math.isfinite(projection_norm):
            return reject("invalid_input", "initial projection is not representable")
    initial_q = q
    iterations, evaluations, maximum_step = 0, 0, 0.0
    best_q = best_pose = None
    best_error = None

    def expired():
        return time.perf_counter() - started >= config.max_seconds

    def finish(status, reason):
        elapsed = max(0.0, time.perf_counter() - started)
        return NumericSolution(
            status, reason, best_q, None if best_pose is None else best_pose.point,
            None if best_pose is None else best_pose.orientation_radians, best_error,
            iterations, evaluations,
            () if best_q is None else tuple(i for i, (v, a, b) in enumerate(zip(best_q, lo, hi))
                                           if min(abs(v - a), abs(v - b)) <= 1e-10),
            tuple(sorted(projected)), projection_events, projection_norm,
            0.0 if best_q is None else math.dist(best_q, initial_q), maximum_step, elapsed)

    def evaluate(candidate):
        nonlocal evaluations, best_q, best_pose, best_error
        if expired():
            return None
        pose = cpu_fk(chain, candidate, present=present, measured=measured,
                      active=active, task_mask=task_mask)
        evaluations += 1
        if not pose.valid:
            raise ArithmeticError("nominal FK rejected the bounded candidate")
        error = tuple((t - p) if enabled else 0.0
                      for t, p, enabled in zip(target, pose.point, task_mask))
        residual = math.hypot(*error)
        if not math.isfinite(residual):
            raise ArithmeticError("endpoint residual is not representable")
        if best_error is None or residual < best_error:
            best_q, best_pose, best_error = candidate, pose, residual
        return pose, error, residual

    def project(candidate):
        nonlocal projection_events, projection_norm
        result = tuple(min(b, max(a, v)) for v, a, b in zip(candidate, lo, hi))
        changes = tuple(b - a for a, b in zip(candidate, result))
        if any(changes):
            projected.update(i for i, value in enumerate(changes) if value != 0)
            projection_events += 1
            projection_norm += math.hypot(*changes)
        return result

    try:
        state = evaluate(q)
        if state is None or expired():
            return finish("time_budget_exhausted", "wall-time limit reached; reachability is unknown")
        for iteration in range(config.max_iterations + 1):
            pose, error, residual = state
            if expired():
                return finish("time_budget_exhausted", "wall-time limit reached; reachability is unknown")
            if residual <= config.tolerance_cells:
                return finish("solved", "selected nominal endpoint components meet tolerance")
            if iteration == config.max_iterations:
                return finish("iteration_budget_exhausted", "iteration ceiling reached; reachability is unknown")
            iterations += 1
            damping = config.damping * max(0.01, min(1.0, math.sqrt(residual)))
            free = list(active)
            step = (0.0,) * n
            # At most n+1 passes: each unsuccessful pass removes a coordinate.
            for _ in range(n + 1):
                jac = tuple(tuple(v if enabled else 0.0 for v, enabled in zip(row, free))
                            for row in pose.jacobian)
                step = _left_solve(jac, error, damping)
                weight = config.posture_weight * min(1.0, residual * residual)
                bias = tuple(weight * (p - v) if enabled else 0.0
                             for p, v, enabled in zip(preferred, q, free))
                projected_bias = _left_solve(jac, tuple(sum(v * b for v, b in zip(row, bias))
                                                       for row in jac), damping)
                step = tuple(a + b - c for a, b, c in zip(step, bias, projected_bias))
                remove = tuple(enabled and (v <= a + 1e-12 and s < 0 or v >= b - 1e-12 and s > 0)
                               for enabled, v, a, b, s in zip(free, q, lo, hi, step))
                if not any(remove):
                    break
                free = [enabled and not blocked for enabled, blocked in zip(free, remove)]
            if not all(math.isfinite(v) for v in step):
                return finish("nonfinite_result", "damped step is nonfinite; no actionable result")
            step = _trust(step, config.max_step)
            accepted = None
            for backtrack in range(config.max_backtracks):
                candidate = project(tuple(v + s * 0.5 ** backtrack for v, s in zip(q, step)))
                if math.dist(candidate, q) <= 1e-14:
                    break
                value = evaluate(candidate)
                if value is None or expired():
                    return finish("time_budget_exhausted", "wall-time limit reached; reachability is unknown")
                if value[2] < residual - max(1e-15, residual * 1e-14):
                    accepted = candidate, value
                    break
            if accepted is None:
                # Test at most 2*n null-space probes. A single flat step is
                # allowed only when it changes the configuration while retaining
                # endpoint error, so an exactly folded chain can leave its flat
                # first derivative. The ordinary iteration ceiling still applies.
                jac = pose.jacobian
                candidates = []
                for index in range(n):
                    if not active[index]:
                        continue
                    basis = tuple(1.0 if i == index else 0.0 for i in range(n))
                    along = _left_solve(jac, (jac[0][index], jac[1][index]), damping)
                    null = tuple((v - a) if enabled else 0.0
                                 for v, a, enabled in zip(basis, along, active))
                    norm = math.hypot(*null)
                    if norm <= 1e-8:
                        continue
                    for sign in (1.0, -1.0):
                        probe = tuple(sign * config.singular_probe * v / norm for v in null)
                        candidate = project(tuple(v + s for v, s in zip(q, probe)))
                        if math.dist(candidate, q) <= 1e-12:
                            continue
                        value = evaluate(candidate)
                        if value is None or expired():
                            return finish("time_budget_exhausted", "wall-time limit reached; reachability is unknown")
                        if value[2] <= residual + 1e-14:
                            candidates.append((candidate, value))
                if candidates:
                    improving = [item for item in candidates if
                                 item[1][2] < residual - max(1e-15, residual * 1e-14)]
                    if improving:
                        accepted = min(improving, key=lambda item: (item[1][2], math.dist(item[0], preferred)))
                    else:
                        # At an exactly folded equal-link pose, different root
                        # rotations can have indistinguishable endpoint error.
                        # Prefer the flat probe whose NEW feasible gradient can
                        # actually descend, instead of oscillating toward the
                        # original posture or following round-off differences.
                        def plateau_key(item):
                            values, (new_pose, new_error, _) = item
                            gradient = tuple(sum(row[i] * e for row, e in zip(new_pose.jacobian, new_error))
                                             for i in range(n))
                            feasible = tuple(g if enabled and not (
                                v <= a + 1e-12 and g < 0 or v >= b - 1e-12 and g > 0) else 0.0
                                for g, enabled, v, a, b in zip(gradient, active, values, lo, hi))
                            return -math.hypot(*feasible), math.dist(values, preferred)
                        accepted = min(candidates, key=plateau_key)
            if accepted is None:
                return finish("best_effort", "bounded local steps cannot improve residual; global feasibility is unknown")
            candidate, state = accepted
            maximum_step = max(maximum_step, math.dist(candidate, q))
            q = candidate
    except (ArithmeticError, OverflowError, ValueError) as exc:
        if isinstance(exc, ContractError):
            return finish("invalid_input", exc.code)
        return finish("nonfinite_result", "numerical calculation failed; no actionable result")
    return finish("iteration_budget_exhausted", "iteration ceiling reached; reachability is unknown")


@dataclass(frozen=True)
class AnalyticSolution(Record):
    """Independent two-vector geometry, ordered by feasible current-pose distance."""

    status: Literal["solved", "joint_limited", "geometrically_infeasible", "unsupported_profile", "invalid_input"]
    configurations: tuple[tuple[float, ...], ...] = ()
    points: tuple[tuple[float, float], ...] = ()
    proof_kind: Literal["outside_outer_radius", "inside_inner_radius"] | None = None
    center: tuple[float, float] | None = None
    inner_radius: float | None = None
    outer_radius: float | None = None
    reason: str = ""

    @property
    def configuration(self):
        return self.configurations[0] if self.configurations else None


def analytic_two_link(chain: PlanarChain, target, current, *, lower=None, upper=None) -> AnalyticSolution:
    """Solve two rotated vectors by cosine law, without calling FK or DLS.

    Both elbow branches and their legal angle representatives are retained. A
    point in the annulus with no legal branch has status joint_limited. The
    equal-length folded case has a continuum; retain the current first angle.
    """
    if (not isinstance(chain, PlanarChain) or len(chain.joints) != 2
            or any(j.kind != "rotary" or j.participation != "active" for j in chain.joints)):
        return AnalyticSolution("unsupported_profile", reason="requires two active rotary joints")
    try:
        target, current = _vector(target, 2, "target"), _vector(current, 2, "current")
        lo, hi = _bounds(chain, lower, upper)
    except ContractError as exc:
        return AnalyticSolution("invalid_input", reason=exc.code)
    current = tuple(_initial_coordinate(v, j, a, b)
                    for v, j, a, b in zip(current, chain.joints, lo, hi))
    first, second = chain.joints

    def rot(x, y, angle):
        c, s = math.cos(angle), math.sin(angle)
        return c * x - s * y, s * x + c * y

    a = rot(second.parent_anchor[0] - first.child_anchor[0],
            second.parent_anchor[1] - first.child_anchor[1], first.zero_radians)
    b = rot(chain.effector_local[0] - second.child_anchor[0],
            chain.effector_local[1] - second.child_anchor[1], first.zero_radians + second.zero_radians)
    length_a, length_b = math.hypot(*a), math.hypot(*b)
    center = first.parent_anchor
    inner, outer = abs(length_a - length_b), length_a + length_b
    if not math.isfinite(outer) or min(length_a, length_b) <= 1e-12:
        return AnalyticSolution("unsupported_profile", reason="nondegenerate finite link vectors required")
    dx, dy = target[0] - center[0], target[1] - center[1]
    radius = math.hypot(dx, dy)
    if not math.isfinite(radius):
        return AnalyticSolution("invalid_input", reason="target radius is not representable")
    margin = 1e-12 * max(1.0, outer)
    facts = dict(center=center, inner_radius=inner, outer_radius=outer)
    if radius > outer + margin or radius < inner - margin:
        return AnalyticSolution("geometrically_infeasible", proof_kind=(
            "outside_outer_radius" if radius > outer else "inside_inner_radius"),
            reason="the unrestricted two-link annulus excludes the target", **facts)
    # Normalize lengths before squaring to avoid intermediate overflow.
    scale = max(length_a, length_b, radius)
    la, lb, distance = length_a / scale, length_b / scale, radius / scale
    cosine = (distance * distance - la * la - lb * lb) / (2 * la * lb)
    bend = math.acos(max(-1.0, min(1.0, cosine)))
    phi_a, phi_b = math.atan2(a[1], a[0]), math.atan2(b[1], b[0])
    configurations = []
    for elbow in (bend, -bend):
        q2 = elbow + phi_a - phi_b
        q1 = (current[0] if radius <= margin and inner <= margin else
              math.atan2(dy, dx) - math.atan2(lb * math.sin(elbow), la + lb * math.cos(elbow)) - phi_a)
        for qh in _equivalents(q1, lo[0], hi[0]):
            for qa in _equivalents(q2, lo[1], hi[1]):
                candidate = qh, qa
                if not any(math.dist(candidate, old) < 1e-12 for old in configurations):
                    configurations.append(candidate)
    configurations.sort(key=lambda q: (math.dist(q, current), q))
    points = []
    for q1, q2 in configurations:
        ra, rb = rot(*a, q1), rot(*b, q1 + q2)
        points.append((center[0] + ra[0] + rb[0], center[1] + ra[1] + rb[1]))
    return AnalyticSolution("solved" if configurations else "joint_limited",
                            tuple(configurations), tuple(points),
                            reason="legal branches ordered from current pose" if configurations else
                                   "the target is in the annulus but no branch satisfies the solve bounds", **facts)


class IKSolver(Protocol):
    """Replaceable source-bound solver; the caller still owns dispatch checks."""

    def solve(self, profile: IKProfile, target: IKTarget, observation: IKObservation,
              authority: IKAuthority) -> IKResult: ...


class DampedLeastSquares:
    """Source-bound Runner adapter for the numerical reference, with no host IO."""

    id = "planar-dls-v1"

    def __init__(self, *, profile: IKProfile | None = None, config: SolverConfig | None = None):
        if profile is not None and not isinstance(profile, IKProfile):
            raise ContractError("unsupported_profile", "a validated Runner IK profile is required")
        self.profile = profile
        self.config = SolverConfig() if config is None else config
        if not isinstance(self.config, SolverConfig):
            raise ContractError("ik_solver_config", "a validated SolverConfig is required")

    def solve(self, profile, target, observation, authority):
        return self.solve_with_diagnostics(profile, target, observation, authority)[0]

    def solve_with_diagnostics(self, profile, target, observation, authority):
        if not isinstance(target, IKTarget):
            raise ContractError("ik_target", "a structurally valid target is required for an identified result")

        def failure(status, reason, **kwargs):
            return IKResult(target.sha256, self.id, status, reason, **kwargs), None

        if not isinstance(profile, IKProfile):
            return failure("unsupported_profile", "a validated Runner IK profile is required")
        if self.profile is not None and profile.sha256 != self.profile.sha256:
            return failure("unsupported_profile", "solver is bound to a different profile identity")
        if not isinstance(observation, IKObservation) or not isinstance(authority, IKAuthority):
            return failure("invalid_input", "issued observation and trusted authority are required")
        try:
            target.validate_against(profile, observation, authority)
            chain = chain_from_profile(profile)
            coordinates = observation.configuration(profile)
        except ContractError as exc:
            status = ("missing_required_measurement" if exc.code == "missing_required_measurement" else
                      "unsupported_profile" if exc.code == "unsupported_profile" else "invalid_input")
            return failure(status, exc.code)
        center, radius = nominal_outer_reach(profile)
        distance = math.dist(center, target.point_root)
        if not math.isfinite(distance):
            return failure("nonfinite_result", "target radius is not representable")
        if distance > radius + 1e-9:
            return failure("geometrically_infeasible", "target lies outside the unrestricted nominal radial bound",
                           proof=InfeasibilityProof(center, radius, distance))
        current = tuple(coordinates[name] for name in profile.solver_joint_ids)
        lo, hi = tuple(j.target_lower for j in profile.effector_chain), tuple(j.target_upper for j in profile.effector_chain)
        if target.output_kind == "joint_delta":
            lo = tuple(max(a, q - profile.max_delta_radians) for a, q in zip(lo, current))
            hi = tuple(min(b, q + profile.max_delta_radians) for b, q in zip(hi, current))
            if any(a > b for a, b in zip(lo, hi)):
                return failure("invalid_input", "no legal reference lies inside the current bounded-delta envelope")
        numerical = solve_numeric(chain, target.point_root, current,
                                  config=replace(self.config, tolerance_cells=target.tolerance_cells),
                                  lower=lo, upper=hi, posture=target.posture_preference)
        if not numerical.actionable:
            status = ("iteration_budget_exhausted" if numerical.status == "time_budget_exhausted"
                      else numerical.status)
            return IKResult(target.sha256, self.id, status, numerical.reason,
                            iterations=numerical.iterations, correction_iterations=numerical.iterations), numerical
        positions = numerical.configuration
        values = positions
        if target.output_kind == "joint_delta":
            values, resolved, adjusted_indexes = [], [], []
            for index, (old, desired, joint) in enumerate(zip(current, positions, profile.effector_chain)):
                raw_delta = desired - old
                delta = min(profile.max_delta_radians, max(-profile.max_delta_radians, raw_delta))
                # Floating subtraction can turn an exact .25 trust boundary
                # into .25000000000000006. Emit the exact bounded delta, then
                # reconstruct the pose from precisely what the consumer adds.
                value = old + delta
                for _ in range(4):
                    if joint.target_lower <= value <= joint.target_upper:
                        break
                    delta = math.nextafter(delta, math.inf if value < joint.target_lower else -math.inf)
                    value = old + delta
                if (abs(delta) > profile.max_delta_radians or
                        not joint.target_lower <= value <= joint.target_upper):
                    return failure("nonfinite_result", "bounded delta has no representable admitted position")
                values.append(delta)
                resolved.append(value)
                if delta != raw_delta or value != desired:
                    adjusted_indexes.append(index)
            values, resolved = tuple(values), tuple(resolved)
            if adjusted_indexes:
                adjusted = cpu_fk(chain, resolved)
                adjusted_residual = math.dist(adjusted.point, target.point_root)
                initial = tuple(_initial_coordinate(v, j, a, b)
                                for v, j, a, b in zip(current, chain.joints, lo, hi))
                numerical = replace(
                    numerical, status="solved" if adjusted_residual <= target.tolerance_cells else "best_effort",
                    reason=numerical.reason + "; emitted delta rounded inward",
                    configuration=resolved, point=adjusted.point,
                    orientation_radians=adjusted.orientation_radians, residual_cells=adjusted_residual,
                    active_limits=tuple(i for i, (v, a, b) in enumerate(zip(resolved, lo, hi))
                                        if min(abs(v - a), abs(v - b)) <= 1e-10),
                    projected_coordinates=tuple(sorted(set(numerical.projected_coordinates) |
                        {i for i, (a, b) in enumerate(zip(positions, resolved)) if a != b})),
                    projection_events=numerical.projection_events + int(resolved != positions),
                    projection_correction_norm=numerical.projection_correction_norm + math.dist(positions, resolved),
                    correction_norm=math.dist(resolved, initial),
                    emission_adjusted_coordinates=tuple(adjusted_indexes))
            positions = resolved
        # Independent contract FK validates the exact emitted position/delta;
        # diagnostic candidate poses are not trusted as the output receipt.
        pose = nominal_pose(profile, dict(zip(profile.solver_joint_ids, positions)))
        residual = math.dist(pose.point, target.point_root)
        result = IKResult(
            target.sha256, self.id, "solved" if residual <= target.tolerance_cells else "best_effort",
            numerical.reason, JointReference(profile.solver_joint_ids, values, target.output_kind),
            pose.point, residual, pose.orientation_radians,
            iterations=numerical.iterations, correction_iterations=numerical.iterations)
        result.validate_against(profile, target, observation, authority)
        return result, numerical


class AnalyticTwoLink:
    """Replaceable analytic geometry comparator, with no host target authority."""

    id = "planar-analytic-2r-v1"

    def solve(self, chain, target, current, **kwargs):
        return analytic_two_link(chain, target, current, **kwargs)
