"""Small nominal planar-chain FK implementations, without physical actuation.

The binary64 path multiplies homogeneous transforms. The optional Torch path
traverses anchors in a fixed joint loop, vectorizing every batch dimension. It
never reads a tensor value on the host. Neither path estimates compliant pose.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Any, Literal, Protocol

from nca.contracts import ContractError, Record, digest, identifier, unique
from .profiles import IKProfile


# Per-row bit flags; invalid numeric results are finite masked zeros, not poses.
NONFINITE = 1
MISSING = 2
INVALID_ACTIVITY = 4
OUT_OF_BOUNDS = 8
EMPTY_TASK = 16


@dataclass(frozen=True)
class PlanarJoint(Record):
    """Numerical joint only. Rotary q is radians; prismatic q is game cells.

    A slide translates the child's anchor along a parent-local unit axis. Its
    relative orientation is fixed at zero_radians. This ideal fixture is not an
    additional accepted host profile or a claim of host actuator capability.
    """

    id: str
    parent_id: str
    child_id: str
    parent_anchor: tuple[float, float]
    child_anchor: tuple[float, float]
    zero_radians: float
    lower: float
    upper: float
    kind: Literal["rotary", "prismatic"] = "rotary"
    axis: tuple[float, float] = (1., 0.)
    participation: Literal["active", "passive", "disabled"] = "active"

    def validate(self) -> None:
        for name in ("id", "parent_id", "child_id"):
            identifier(getattr(self, name), "joint." + name)
        if self.parent_id == self.child_id:
            raise ContractError("ik_topology", "joint endpoints must differ")
        if self.lower >= self.upper:
            raise ContractError("ik_joint_bounds", "nonempty finite joint interval required")
        if self.kind == "rotary" and not -math.pi <= self.lower < self.upper <= math.pi:
            raise ContractError("ik_joint_bounds", "rotary intervals must lie in [-pi, pi]")
        if abs(self.zero_radians) > math.pi:
            raise ContractError("ik_zero", "authored zero must lie in [-pi, pi]")
        if not math.isclose(math.hypot(*self.axis), 1., rel_tol=0., abs_tol=1e-12):
            raise ContractError("ik_axis", "parent-local axis must be a unit vector")
        # Bound representational scale, not reachability: protect both supported
        # dtypes against overflow even for deliberately malformed fixture data.
        if max(abs(v) for v in (*self.parent_anchor, *self.child_anchor,
                                self.lower, self.upper)) > 1e6:
            raise ContractError("ik_scale", "numerical fixture dimensions exceed 1e6")

    @property
    def coordinate_unit(self) -> str:
        return "radians" if self.kind == "rotary" else "game_cells"


@dataclass(frozen=True)
class PlanarChain(Record):
    """One ordered root-to-effector path, with at most seven joints.

    A serial path can be selected from a valid tree. Disconnected paths, loops,
    multi-parent mechanisms and unrecognized joint kinds are explicitly refused.
    Source/profile identity is metadata, never an extra learned coordinate.
    """

    id: str
    root_id: str
    joints: tuple[PlanarJoint, ...]
    effector_local: tuple[float, float]
    profile_sha256: str | None = None
    evidence_kind: Literal["nominal_geometry"] = "nominal_geometry"
    schema_version: Literal[1] = 1

    def validate(self) -> None:
        identifier(self.id, "chain.id")
        identifier(self.root_id, "chain.root_id")
        if not 1 <= len(self.joints) <= 7:
            raise ContractError("ik_capacity", "one to seven serial joints required")
        unique(tuple(j.id for j in self.joints), "chain.joint_ids")
        seen, parent = {self.root_id}, self.root_id
        for joint in self.joints:
            if joint.parent_id != parent or joint.child_id in seen:
                raise ContractError("ik_topology", "ordered connected acyclic serial path required")
            seen.add(joint.child_id)
            parent = joint.child_id
        if max(abs(v) for v in self.effector_local) > 1e6:
            raise ContractError("ik_scale", "effector coordinates exceed 1e6")
        if self.profile_sha256 is not None:
            digest(self.profile_sha256, "chain.profile_sha256")

    @property
    def joint_ids(self) -> tuple[str, ...]:
        return tuple(j.id for j in self.joints)

    @property
    def geometry_scale_cells(self) -> float:
        return max(1., math.hypot(*self.effector_local) + sum(
            math.hypot(*j.parent_anchor) + math.hypot(*j.child_anchor)
            + (max(abs(j.lower), abs(j.upper)) if j.kind == "prismatic" else 0.)
            for j in self.joints))


def chain_from_profile(profile: IKProfile) -> PlanarChain:
    """Keep the accepted descriptor unchanged and select its actual effector path."""
    if not isinstance(profile, IKProfile):
        raise ContractError("unsupported_profile", "validated IKProfile required")
    return PlanarChain(
        id=profile.id, root_id=profile.root_part_id,
        joints=tuple(PlanarJoint(
            id=j.id, parent_id=j.parent_part_id, child_id=j.child_part_id,
            parent_anchor=j.parent_anchor_local, child_anchor=j.child_anchor_local,
            zero_radians=j.zero_radians, lower=j.lower, upper=j.upper,
            kind=j.kind, participation=j.participation) for j in profile.effector_chain),
        effector_local=profile.effector.point_local, profile_sha256=profile.sha256)


@dataclass(frozen=True)
class KinematicsResult:
    """Pose/Jacobians and validity, with tuples on CPU and tensors on Torch.

    transform is the effector frame, not the last body origin. Orientation is
    unwrapped and diagnostic. jacobian has the two point rows; activity masks
    remove columns and task masks remove rows. Measured passive q still affects
    geometry. Invalid rows have zero numeric output and nonzero invalid_code.
    """

    point: Any
    orientation_radians: Any
    jacobian: Any
    orientation_jacobian: Any
    transform: Any
    valid: Any
    invalid_code: Any


class ForwardKinematics(Protocol):
    def __call__(self, q: Any, *, present: Any = None, measured: Any = None,
                 active: Any = None, task_mask: Any = None) -> KinematicsResult: ...


def _mask(values, size: int, name: str, default: tuple[bool, ...]) -> tuple[bool, ...]:
    if values is None:
        return default
    if (not isinstance(values, (tuple, list)) or len(values) != size
            or any(type(v) is not bool for v in values)):
        raise ContractError("ik_mask_shape", name + " requires exact boolean coordinates")
    return tuple(values)


def _product(a, b):
    return tuple(tuple(sum(a[r][k] * b[k][c] for k in range(3))
                       for c in range(3)) for r in range(3))


def _translation(x, y):
    return ((1., 0., x), (0., 1., y), (0., 0., 1.))


def _rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return ((c, -s, 0.), (s, c, 0.), (0., 0., 1.))


def _empty(n: int, code: int) -> KinematicsResult:
    return KinematicsResult((0., 0.), 0., ((0.,) * n, (0.,) * n),
                            (0.,) * n, ((0.,) * 3,) * 3, False, code)


def cpu_fk(chain: PlanarChain, q, *, present=None, measured=None, active=None,
           task_mask=(True, True)) -> KinematicsResult:
    """Independent Python binary64 homogeneous-transform FK and geometric J.

    Structural mistakes raise ContractError. Invalid measurement rows return an
    explicit invalid finite placeholder; callers must inspect valid before use.
    The supplied q order is chain.joint_ids, never guessed from padding slots.
    """
    if not isinstance(chain, PlanarChain):
        raise ContractError("unsupported_profile", "validated PlanarChain required")
    n = len(chain.joints)
    if not isinstance(q, (tuple, list)) or len(q) != n:
        raise ContractError("ik_shape", "q must follow every ordered path joint")
    allowed = tuple(j.participation == "active" for j in chain.joints)
    present = _mask(present, n, "present", (True,) * n)
    measured = _mask(measured, n, "measured", (True,) * n)
    active = _mask(active, n, "active", allowed)
    task = _mask(task_mask, 2, "task_mask", (True, True))
    code = 0
    if not all(present) or not all(measured):
        code |= MISSING
    if any((m and not p) or (a and (not p or not permit))
           for p, m, a, permit in zip(present, measured, active, allowed)):
        code |= INVALID_ACTIVITY
    if not any(task):
        code |= EMPTY_TASK
    numbers = []
    for value, joint in zip(q, chain.joints):
        try:
            number = float(value) if type(value) in (int, float) else math.nan
        except (OverflowError, ValueError):
            number = math.nan
        if not math.isfinite(number):
            code |= NONFINITE
        elif not joint.lower <= number <= joint.upper:
            code |= OUT_OF_BOUNDS
        numbers.append(number)
    if code:
        return _empty(n, code)
    transform, angle = _translation(0., 0.), 0.
    pivots, axes = [], []
    for value, joint in zip(numbers, chain.joints):
        at_anchor = _product(transform, _translation(*joint.parent_anchor))
        pivots.append((at_anchor[0][2], at_anchor[1][2]))
        axes.append((transform[0][0] * joint.axis[0] + transform[0][1] * joint.axis[1],
                     transform[1][0] * joint.axis[0] + transform[1][1] * joint.axis[1]))
        if joint.kind == "prismatic":
            at_anchor = _product(at_anchor, _translation(value * joint.axis[0], value * joint.axis[1]))
        rotation = joint.zero_radians + (value if joint.kind == "rotary" else 0.)
        angle += rotation
        transform = _product(_product(at_anchor, _rotation(rotation)),
                             _translation(-joint.child_anchor[0], -joint.child_anchor[1]))
    transform = _product(transform, _translation(*chain.effector_local))
    point = transform[0][2], transform[1][2]
    columns = [(-(point[1] - p[1]), point[0] - p[0]) if j.kind == "rotary" else axis
               for j, p, axis in zip(chain.joints, pivots, axes)]
    jacobian = tuple(tuple(col[r] if task[r] and enabled else 0.
                           for col, enabled in zip(columns, active)) for r in range(2))
    orientation_jacobian = tuple(float(j.kind == "rotary" and enabled)
                                 for j, enabled in zip(chain.joints, active))
    return KinematicsResult(point, angle, jacobian, orientation_jacobian, transform, True, 0)


class CPUFK:
    """Replaceable dependency-free binary64 backend."""

    def __init__(self, chain: PlanarChain):
        if not isinstance(chain, PlanarChain):
            raise ContractError("unsupported_profile", "validated PlanarChain required")
        self.chain = chain

    def __call__(self, q, **kwargs) -> KinematicsResult:
        return cpu_fk(self.chain, q, **kwargs)


def _float32_bound(value: float, *, lower: bool) -> float:
    """Round a finite exact descriptor bound inward, without optional imports.

    Float32 .3 is slightly larger than binary64 .3. Nearest rounding would
    otherwise admit a tensor coordinate that the CPU reference correctly rejects.
    One neighboring representable value is sufficient; negative zero is handled
    by its IEEE sign bit. Descriptor scale checks prevent float32 overflow.
    """
    packed = struct.pack(">f", value)
    rounded = struct.unpack(">f", packed)[0]
    if (lower and rounded < value) or (not lower and rounded > value):
        bits = struct.unpack(">I", packed)[0]
        if lower:
            bits += -1 if bits & 0x80000000 else 1
        else:
            bits += 1 if bits & 0x80000000 else -1
            # Positive zero's immediate lower neighbor is negative subnormal.
            if bits == -1:
                bits = 0x80000001
        rounded = struct.unpack(">f", struct.pack(">I", bits))[0]
    return rounded


class TorchFK:
    """Optional batched differentiable anchor FK on an explicit DeviceContext.

    Construct with configure_device('cpu' or 'cuda:<index>', seed) from #51.
    The hot path has only a fixed <=7-joint host loop, no batch/sample host loop,
    .item(), .cpu(), .tolist(), tensor truth tests, or synchronization. Constants
    are allocated once on the chosen device. No input is silently transferred.
    """

    def __init__(self, chain: PlanarChain, context, *, dtype=None):
        from .device import DeviceContext

        if not isinstance(chain, PlanarChain):
            raise ContractError("unsupported_profile", "validated PlanarChain required")
        if not isinstance(context, DeviceContext):
            raise ContractError("ik_device", "explicit IK DeviceContext required")
        self.chain, self.context, self.torch = chain, context, context.torch
        self.device, self.dtype = context.device, dtype or context.torch.float32
        torch = self.torch
        if self.dtype not in (torch.float32, torch.float64):
            raise ContractError("ik_dtype", "FK supports named float32 and float64 lanes only")
        tensor = lambda values: torch.tensor(values, dtype=self.dtype, device=self.device)
        self.parent_anchors = tensor([j.parent_anchor for j in chain.joints])
        self.child_anchors = tensor([j.child_anchor for j in chain.joints])
        self.zeros = tensor([j.zero_radians for j in chain.joints])
        self.axes = tensor([j.axis for j in chain.joints])
        lows, highs = [j.lower for j in chain.joints], [j.upper for j in chain.joints]
        if self.dtype == torch.float32:
            lows = [_float32_bound(v, lower=True) for v in lows]
            highs = [_float32_bound(v, lower=False) for v in highs]
            if any(lo > hi for lo, hi in zip(lows, highs)):
                raise ContractError("ik_dtype", "a joint interval contains no representable float32 coordinate")
        self.lower, self.upper = tensor(lows), tensor(highs)
        self.effector = tensor(chain.effector_local)
        self.allowed = torch.tensor([j.participation == "active" for j in chain.joints],
                                    dtype=torch.bool, device=self.device)

    def _mask(self, values, shape, default, name):
        torch = self.torch
        if values is None:
            if isinstance(default, bool):
                return torch.full(shape, default, dtype=torch.bool, device=self.device)
            return default.expand(shape)
        if (not isinstance(values, torch.Tensor) or values.shape != shape
                or values.dtype != torch.bool or values.device != self.device):
            raise ContractError("ik_mask_shape", name + " must match batch, bool dtype and selected device")
        return values

    def __call__(self, q, *, present=None, measured=None, active=None,
                 task_mask=None) -> KinematicsResult:
        torch, n = self.torch, len(self.chain.joints)
        if (not isinstance(q, torch.Tensor) or q.ndim < 1 or q.shape[-1] != n
                or q.dtype != self.dtype or q.device != self.device):
            raise ContractError("ik_tensor", "q must have matching ordered joints, dtype and selected device")
        present = self._mask(present, q.shape, True, "present")
        measured = self._mask(measured, q.shape, True, "measured")
        active = self._mask(active, q.shape, self.allowed, "active")
        task = self._mask(task_mask, q.shape[:-1] + (2,), True, "task_mask")
        finite = torch.isfinite(q)
        code = ((~finite.all(dim=-1)).to(torch.int64) * NONFINITE
                + (~(present & measured).all(dim=-1)).to(torch.int64) * MISSING
                + ((measured & ~present) | (active & (~present | ~self.allowed))).any(dim=-1).to(torch.int64) * INVALID_ACTIVITY
                + (((q < self.lower) | (q > self.upper)) & finite).any(dim=-1).to(torch.int64) * OUT_OF_BOUNDS
                + (~task.any(dim=-1)).to(torch.int64) * EMPTY_TASK)
        valid = code == 0
        # Replace invalid rows *before* trigonometry: NaN * masked zero in an
        # unselected autograd branch would otherwise contaminate gradients.
        safe = torch.where(valid[..., None] & finite, q, torch.zeros_like(q))
        angle = torch.zeros_like(safe[..., 0])
        origin = torch.zeros_like(safe[..., :1]).expand(safe.shape[:-1] + (2,))
        pivots, axes = [], []

        def rotate(vector, theta):
            c, s = torch.cos(theta), torch.sin(theta)
            return torch.stack((c * vector[..., 0] - s * vector[..., 1],
                                s * vector[..., 0] + c * vector[..., 1]), dim=-1)

        for i, joint in enumerate(self.chain.joints):
            pivot = origin + rotate(self.parent_anchors[i], angle)
            axis = rotate(self.axes[i], angle)
            pivots.append(pivot)
            axes.append(axis)
            if joint.kind == "rotary":
                angle = angle + self.zeros[i] + safe[..., i]
            else:
                pivot = pivot + axis * safe[..., i, None]
                angle = angle + self.zeros[i]
            origin = pivot - rotate(self.child_anchors[i], angle)
        point = origin + rotate(self.effector, angle)
        columns = []
        for joint, pivot, axis in zip(self.chain.joints, pivots, axes):
            delta = point - pivot
            columns.append(torch.stack((-delta[..., 1], delta[..., 0]), dim=-1)
                           if joint.kind == "rotary" else axis)
        jacobian = torch.stack(columns, dim=-1)
        jacobian = torch.where(task[..., :, None] & active[..., None, :], jacobian, torch.zeros_like(jacobian))
        orientation_jacobian = torch.stack([
            active[..., i].to(self.dtype) if j.kind == "rotary" else torch.zeros_like(angle)
            for i, j in enumerate(self.chain.joints)], dim=-1)
        c, s, zero, one = torch.cos(angle), torch.sin(angle), torch.zeros_like(angle), torch.ones_like(angle)
        transform = torch.stack((torch.stack((c, -s, point[..., 0]), dim=-1),
                                 torch.stack((s, c, point[..., 1]), dim=-1),
                                 torch.stack((zero, zero, one), dim=-1)), dim=-2)
        return KinematicsResult(
            torch.where(valid[..., None], point, torch.zeros_like(point)),
            torch.where(valid, angle, torch.zeros_like(angle)),
            torch.where(valid[..., None, None], jacobian, torch.zeros_like(jacobian)),
            torch.where(valid[..., None], orientation_jacobian, torch.zeros_like(orientation_jacobian)),
            torch.where(valid[..., None, None], transform, torch.zeros_like(transform)), valid, code)
