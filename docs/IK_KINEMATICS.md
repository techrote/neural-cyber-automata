---
title: Independent planar kinematics and bounded IK reference
kind: reference
summary: CPU and differentiable Torch geometry, independent analytic comparison, strict masks and bounded numerical solving for Runner.
owns: [ik.kinematics, ik.reference_solver]
verified: 2026-10-21
---

# Independent planar kinematics and bounded IK reference

[IK-CUDA-003 / #52](https://github.com/techrote/neural-cyber-automata/issues/52)
implements nominal planar geometry and a numerical inverse reference on the
accepted [Runner contract](IK_LAYER.md). The original profile and
[acceptance declaration](IK_ACCEPTANCE.md) retain their identities. The
[evidence record](evidence/2026-10-21-ik-kinematics.md) distinguishes conformance,
reference measurements and remaining device/physical gates.

## Implementations and interfaces

The existing registry exposes these versioned extensions without importing
Torch or NumPy during discovery:

| Kind and ID | Interface | Evidence boundary |
|---|---|---|
| `kinematics:planar-cpu-f64@1` | `CPUFK(chain)(q, **masks)` | Python binary64 homogeneous transforms and geometric Jacobian |
| `kinematics:planar-torch@1` | `TorchFK(chain, context, dtype=...)(q, **masks)` | Batched differentiable anchor transforms on the explicitly selected device |
| `ik_solver:runner-dls@1` | `DampedLeastSquares(...).solve(profile, target, observation, authority)` | Source-bound Runner result; no host dispatch |
| `ik_solver:planar-analytic-2r@1` | `AnalyticTwoLink().solve(chain, target, current, **bounds)` | Independent two-rotary-joint comparator, including both legal branches |

`ForwardKinematics` and `IKSolver` protocols describe replaceable interfaces.
The low-level `cpu_fk`, `solve_numeric` and `analytic_two_link` functions also
support small numerical fixtures. Their diagnostics are not host commands.
`solve_with_diagnostics` returns the strict `IKResult` and separate numerical
diagnostics without changing the frozen result contract.

`chain_from_profile` selects the actual root-to-sole path from the accepted
descriptor, preserving both local anchors, rotary zeros, hard bounds, joint IDs
and profile hash. Its observed root stays at identity in canonical root-local
geometry; root translation/rotation is never an inverse variable. Missing
off-chain left-leg measurements do not become synthesized coordinates.

## Descriptor, units and supported mechanisms

`PlanarChain` describes one connected, ordered serial path, at most seven joints,
with an effector point in the last part's local frame. `PlanarJoint` records both
anchors, relative zero orientation, hard bounds, participation and joint type.
Coordinates use cells and radians with x right, y down and positive rotations
clockwise on screen. Passive or disabled participation never grants a variable
to the solver; its measured coordinate still contributes to endpoint geometry.

The accepted physical profile remains two rotary joints only. Separately named
numerical prismatic/mixed fixtures use a parent-local unit slide axis and travel
in cells. They do not establish an ACT004 slide target lane. Springs, closed
loops, disconnected paths, branching descriptors and unknown joint classes are
refused; a selected serial path through an already validated tree is supported.
Finite numerical fixture dimensions are limited to `1e6` cells. This prevents
representational overflow, not a claim of useful motion at that scale.

For a rotary joint, the CPU reference calculates:

```text
T_child = T_parent * Translate(parent_anchor)
          * Rotate(zero_radians + q) * Translate(-child_anchor)
T_effector = T_last_part * Translate(effector_local)
```

For a prismatic joint, insert `Translate(q * parent_local_axis)` after the
parent-anchor translation and use only `Rotate(zero_radians)`. The Torch
implementation separately propagates origins and angles through the anchors.
The two code paths do not call a shared transform implementation.

The result contains the effector point, unwrapped diagnostic orientation,
effector-frame transform, point Jacobian, orientation Jacobian and validity.
A rotary Jacobian column is `(-(point-pivot).y, (point-pivot).x)`; a slide
column is the parent axis rotated into root-local space. Orientation derivatives
are one for rotary and zero for prismatic coordinates, before activity masking.
Point x/y are the only task objectives. An orientation diagnostic does not grant
a third independently controlled degree of freedom to Runner.

## Masks, device residency and gradients

Every coordinate on the selected chain needs presence and measurement.
Activity is separate: it masks Jacobian columns without replacing known passive
geometry. The task mask selects point rows. Activating a passive/disabled joint,
measuring an absent joint, an empty task, a missing path coordinate, a nonfinite
coordinate or an out-of-bound value is invalid. Shape, type and device mistakes
raise `ContractError`; per-row data failures have `valid=False` and nonzero
`invalid_code` rather than silently yielding an admitted pose.

| Invalid flag | Meaning |
|---:|---|
| 1 | Nonfinite or nonnumeric coordinate |
| 2 | Missing path presence or measurement |
| 4 | Inconsistent measurement/activity or unauthorized participation |
| 8 | Coordinate exceeds the exact descriptor's hard bounds |
| 16 | No selected task component |

Flags can combine. Invalid numeric outputs, including transforms, are masked
zeros and **must not be interpreted as measured zero poses**. Invalid coordinates
are sanitized before trigonometry, keeping invalid-row gradients finite and
zero. Ordinary autograd describes geometric dependence on measured inputs;
activity-masked Jacobians describe the permitted optimization variables.

Torch accepts `q` shaped `(..., joint_count)` and returns outputs with those
batch dimensions. Masks have the same shape as q; task masks end in dimension
two. Scalar, multidimensional, empty and noncontiguous batches are covered.
The backend has fixed loops over at most seven joints, with no per-sample Python
loop, tensor truth test, host scalar extraction, device transfer or synchronization
in its hot path. Constants are allocated once on the chosen device.

Construct the backend using #51's [explicit device context](IK_RUNTIME.md).
Inputs must already have the selected device and named dtype. Float32 is the
default; float64 is a separately named numerical comparison lane. There is no
silent CPU fallback. CUDA parity remains unmeasured unless the explicit manual
test is actually run on an identified qualified device.

Float32 bounds are rounded **inward** once at construction. For example, the
usual FP32 representation of `0.3` exceeds the descriptor's binary64 bound and
is refused, just as the CPU reference refuses that same numeric coordinate.
The adjacent interior FP32 value is admitted. An interval with no representable
FP32 coordinate is explicitly rejected. This preserves exact same-input bound
semantics without adding float64 arithmetic to the FP32 hot path.

## Independent geometric checks

The pinned host audit confirms the authored hip/ankle zeros `+0.18/-0.18` and
the local-anchor conversion described in [IK_LAYER.md](IK_LAYER.md#frames-and-hand-checks).
Independently derived world-space vectors give:

```text
h = (3.6, 2.8); a = (0, 4.95); b = (1, 1.25)
p = h + R(qh)*a + R(qh+qa)*b
foot_orientation = qh + qa
```

The authored `0.18` is already represented in these vectors. Adding it again
would be an error. Hand fixtures also use displaced anchors and nonzero zeros
on a separately declared mechanism. World conversion uses the issued root pose,
and independent checks verify `J_world = R(root_angle) * J_root`.

Runner has no singularity or second branch within its accepted intervals.
Straight, folded, near-singular and multiple-branch cases therefore use synthetic
two-link fixtures. They do not enlarge the claimed Runner coverage.

| Comparison | Prospective tolerance |
|---|---|
| Independent float64 point / angle | `1e-8 cell` / `1e-9 rad` |
| Jacobian vs central difference at interior nonsingular points | Absolute `1e-5` plus relative `1e-5`; step `1e-6` in the coordinate's unit |
| Float32 point / Jacobian parity | Absolute `2e-5 * max(1, fixture_scale_cells)` plus relative `2e-6` |
| Float32 orientation parity | Absolute `2e-6 rad` |
| Feasible Runner reference solve | Both `1e-6 cell` and normalized-reach error `1e-6` |

Scale checks use independently declared two-link scales `0.001`, `1` and
`1000` cells. Boundary and singular behavior is tested separately from interior
derivatives. Dtype parity does not relax Runner's inverse-solve acceptance.

## Bounded numerical reference

`solve_numeric` starts from the measured current configuration. It applies
damped least squares, a finite active-set pass, a joint-space trust limit and a
bounded backtracking search. A small posture preference acts in the damped
null space; it cannot average incompatible solutions. At a flat singular start,
bounded deterministic null-space probes can find a descending direction. The
numerical method never calls the analytic solver or performs random restarts.

Default search limits are 64 iterations, 0.05 seconds, eight backtracking trials,
trust norm 0.25, maximum damping 0.02, posture weight 0.002 and singular probe
0.05. Primary-step damping is `damping * max(0.01, min(1, sqrt(residual_cells)))`,
so its default floor is `0.0002`. The executable declaration records the exact
settings. The API refuses more than 256 iterations, five
seconds or sixteen backtracking trials. Active-set and singular-probe loops
are also bounded by chain capacity. Wall-time checks are cooperative at each
FK evaluation and iteration; Python/OS scheduling is not a hard real-time bound.

The trust norm uses each coordinate's native numerical unit (radians or cells).
The mixed fixtures establish arithmetic only; physical mixed-mechanism tuning
requires a separately declared scaling and profile. Initial projection, projected
coordinates, active limits, correction count/norm, maximum accepted step,
evaluation count, residual and elapsed time are reported separately.
Exact-delta emission adjustments have their own coordinate list, and their
resolved endpoint/residual is recomputed after binary64 rounding.

Equivalent rotary inputs are normalized into the declared interval when one
exists. Otherwise the initial active coordinate is projected and reported.
Subsequent steps stay inside that interval; no wrap shortcut crosses its hard
stops. `[-pi,pi]` alone does not declare a physically periodic actuator. Passive
coordinates cannot be projected into invented measurements.

`solved` meets the requested endpoint tolerance. `best_effort` is a finite bounded
candidate where the local search cannot improve; it does not prove infeasibility.
Iteration/time exhaustion remains distinct and carries no actionable `IKResult`
payload. The strict adapter maps time exhaustion to the existing budget-exhausted
status while retaining its precise reason and separate diagnostics.

The Runner adapter validates original observation, profile/source/topology,
authority, ownership and deadline contracts. Desired positions use the inner
command bounds. Delta outputs also remain within the original observation's
per-coordinate `0.25`-radian envelope. The exact emitted references are resolved
and independently recomputed with the contract FK before result validation.
Geometric validity never sends a packet or grants a host capability.

## Analytic comparison and physical limits

The independent analytic comparator reduces two rotary joints to two rotated
vectors, uses the cosine law, enumerates both elbows and legal `2*pi` equivalents,
filters bounds and ranks legal branches by the normalized current configuration.
At an equal-length folded continuum it keeps a legal nearest current shoulder
angle. It never calls numerical FK or the numerical inverse solver.

Outside the unrestricted two-link annulus is a geometric certificate; absence
of legal branches within the annulus is separately `joint_limited`. The frozen
Runner `IKResult` supports only its existing outer-radius proof. Inner-annulus
proofs from synthetic analytic fixtures do not change that contract. Numerical
timeouts or local failures never acquire either certificate.

Nominal endpoint residual assumes coincident ideal anchors. Actual host
`anchor_error`, angular play, measured sole pose, contact, collision clearance,
balance and dynamic attainability are separate measurements. This implementation
does not overwrite body transforms, solve contacts or move the root. Learned
data/training, exact CPU export and physical tracking retain #53–#57's gates.

## Running conformance and using the API

These commands work from the installed checkout in PowerShell or a shell:

```text
python -m nca plugins
python -m unittest discover -s tests -p "test_ik_*.py" -v
python tools/check_docs.py --strict
```

```python
from nca.builtins import builtin_registry
from nca.ik.kinematics import chain_from_profile
from nca.ik.profiles import runner_right_sole_profile

registry = builtin_registry()
profile = runner_right_sole_profile()
chain = chain_from_profile(profile)
fk = registry.create("kinematics", "planar-cpu-f64", chain=chain)
pose = fk((0.0, 0.0))
assert pose.valid
assert pose.point == (4.6, 9.0)
```

For the optional backend, first install the pinned [IK environment](IK_RUNTIME.md),
construct `context = configure_device("cpu", 5203)` and pass `chain` and `context`
to `registry.create("kinematics", "planar-torch", ...)`. The manual CUDA fixture
requires explicit `NCA_IK_CUDA_TEST_DEVICE=cuda:0`; a skipped case is unmeasured.
Reference cost and its prospective declaration are linked from the evidence page.
