"""Independent planar geometry fixtures, not physical or CUDA qualification.

Expected Runner values come from authored pivot-to-pivot vectors, not the
production FK. Synthetic slide/mixed mechanisms carry no host capability claim.
The optional CUDA case is explicit opt-in and otherwise remains unmeasured.
"""

from dataclasses import replace
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from nca.contracts import ContractError, LifecycleStamp
from nca.ik.contracts import IKObservation, RootPose, point_to_root
from nca.ik.device import configure_device
from nca.ik.kinematics import (PlanarChain, PlanarJoint, TorchFK,
                               chain_from_profile, cpu_fk)
from nca.ik.profiles import runner_right_sole_profile


OPTIONAL = (importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("numpy") is not None)
CUDA_SELECTION = os.environ.get("NCA_IK_CUDA_TEST_DEVICE")
FLOAT64_POINT_ATOL = 1e-8
FLOAT64_ANGLE_ATOL = 1e-9
JACOBIAN_ATOL = 1e-5
JACOBIAN_RTOL = 1e-5
DIFFERENCE_STEP = 1e-6
FLOAT32_SCALE_ATOL = 2e-5
FLOAT32_RTOL = 2e-6
FLOAT32_ANGLE_ATOL = 2e-6


def two_link(scale=1.0):
    return PlanarChain(
        id="synthetic-two-link", root_id="base",
        joints=(
            PlanarJoint("shoulder", "base", "upper", (0., 0.), (0., 0.),
                        0., -math.pi, math.pi),
            PlanarJoint("elbow", "upper", "lower", (2. * scale, 0.), (0., 0.),
                        0., -math.pi, math.pi),
        ), effector_local=(scale, 0.))


def displaced_chain():
    return PlanarChain(
        id="synthetic-displaced-anchors", root_id="base",
        joints=(
            PlanarJoint("first", "base", "upper", (1., 2.), (.5, -1.),
                        math.pi / 2, -math.pi, math.pi),
            PlanarJoint("second", "upper", "lower", (-1., .5), (.25, .5),
                        -math.pi / 4, -math.pi, math.pi),
        ), effector_local=(.75, -.25))


def mixed_chain():
    return PlanarChain(
        id="synthetic-rotary-slide", root_id="base",
        joints=(
            PlanarJoint("hinge", "base", "upper", (1., 2.), (0., 0.),
                        math.pi / 2, -math.pi, math.pi),
            PlanarJoint("slide", "upper", "lower", (2., 0.), (.5, 0.),
                        0., -1., 2., kind="prismatic", axis=(1., 0.)),
        ), effector_local=(1.5, 0.))


def runner_point(q):
    h, a = q
    return (3.6 - 4.95 * math.sin(h) + math.cos(h + a) - 1.25 * math.sin(h + a),
            2.8 + 4.95 * math.cos(h) + math.sin(h + a) + 1.25 * math.cos(h + a))


def runner_jacobian(q):
    h, a = q
    dx = -math.sin(h + a) - 1.25 * math.cos(h + a)
    dy = math.cos(h + a) - 1.25 * math.sin(h + a)
    return ((-4.95 * math.cos(h) + dx, dx),
            (-4.95 * math.sin(h) + dy, dy))


def flat(values):
    if isinstance(values, (tuple, list)):
        for value in values:
            yield from flat(value)
    else:
        yield values


class IKGeometryCPUConformanceTests(unittest.TestCase):
    def assertNumbers(self, actual, expected, atol=FLOAT64_POINT_ATOL, rtol=0.):
        actual, expected = list(flat(actual)), list(flat(expected))
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertLessEqual(abs(left - right), atol + rtol * abs(right))

    def assertFiniteInvalid(self, result):
        self.assertFalse(result.valid)
        self.assertNotEqual(result.invalid_code, 0)
        values = (result.point, result.orientation_radians, result.jacobian,
                  result.orientation_jacobian, result.transform)
        self.assertTrue(all(math.isfinite(value) for value in flat(values)))
        self.assertNumbers(result.point, (0., 0.), atol=0.)
        self.assertTrue(all(value == 0. for value in flat(result.jacobian)))

    def test_base_import_registry_and_cpu_fk_do_not_load_numpy_or_torch(self):
        program = """
import importlib.abc
import sys
class RefuseOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy'):
            raise AssertionError('optional runtime imported by CPU geometry')
sys.meta_path.insert(0, RefuseOptional())
from nca.ik.kinematics import chain_from_profile, cpu_fk
from nca.ik.profiles import runner_right_sole_profile
from nca.builtins import builtin_registry
from nca.contracts import ContractError
chain = chain_from_profile(runner_right_sole_profile())
assert cpu_fk(chain, (0., 0.)).valid
registry = builtin_registry()
assert {item.id for item in registry.list('kinematics')} == {'planar-cpu-f64', 'planar-torch'}
assert {item.id for item in registry.list('ik_solver')} == {'runner-dls', 'planar-analytic-2r'}
backend = registry.create('kinematics', 'planar-cpu-f64', require=('binary64',), chain=chain)
result = backend((0., 0.))
assert result.valid and abs(result.point[0] - 4.6) < 1e-8 and abs(result.point[1] - 9.) < 1e-8
for name in ('runner-dls', 'planar-analytic-2r'):
    assert callable(registry.create('ik_solver', name).solve)
try:
    registry.create('kinematics', 'planar-torch', chain=chain, context=None)
except ContractError as error:
    assert error.code == 'ik_device'
else:
    raise AssertionError('Torch FK accepted no explicitly selected device context')
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run([sys.executable, "-c", program], env=environment,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_profile_adapter_preserves_hard_bounds_zero_and_source_identity(self):
        profile = runner_right_sole_profile()
        chain = chain_from_profile(profile)
        self.assertEqual(chain.profile_sha256, profile.sha256)
        self.assertEqual(chain.root_id, profile.root_part_id)
        self.assertEqual(chain.joint_ids, profile.solver_joint_ids)
        self.assertEqual(chain.effector_local, (0., 1.))
        for mapped, authored in zip(chain.joints, profile.effector_chain):
            self.assertEqual((mapped.parent_id, mapped.child_id),
                             (authored.parent_part_id, authored.child_part_id))
            self.assertEqual((mapped.parent_anchor, mapped.child_anchor),
                             (authored.parent_anchor_local, authored.child_anchor_local))
            self.assertEqual((mapped.lower, mapped.upper, mapped.zero_radians),
                             (authored.lower, authored.upper, authored.zero_radians))

    def test_two_link_hand_poses_and_neutral_task_jacobian(self):
        chain = two_link()
        for q, point in (((0., 0.), (3., 0.)),
                         ((math.pi / 2, 0.), (0., 3.)),
                         ((0., math.pi / 2), (2., 1.)),
                         ((math.pi / 2, -math.pi / 2), (1., 2.))):
            with self.subTest(q=q):
                actual = cpu_fk(chain, q)
                self.assertTrue(actual.valid)
                self.assertNumbers(actual.point, point)
                self.assertAlmostEqual(actual.orientation_radians, sum(q), delta=FLOAT64_ANGLE_ATOL)
        actual = cpu_fk(chain, (0., math.pi / 2))
        self.assertNumbers(actual.jacobian, ((-1., -1.), (2., 0.)))
        self.assertNumbers(actual.orientation_jacobian, (1., 1.))

    def test_authored_nonzero_angles_and_displaced_child_anchors(self):
        chain = displaced_chain()
        root_two = math.sqrt(2.)
        for q, point, angle in (
            ((0., 0.), (-.5 + 1.25 / root_two, .5 - .25 / root_two), math.pi / 4),
            ((math.pi / 2, 0.), (2.5 + .25 / root_two, .5 + 1.25 / root_two), 3 * math.pi / 4),
        ):
            with self.subTest(q=q):
                result = cpu_fk(chain, q)
                self.assertTrue(result.valid)
                self.assertNumbers(result.point, point)
                self.assertAlmostEqual(result.orientation_radians, angle, delta=FLOAT64_ANGLE_ATOL)
                c, s = math.cos(angle), math.sin(angle)
                self.assertNumbers(result.transform,
                                   ((c, -s, point[0]), (s, c, point[1]), (0., 0., 1.)))

    def test_runner_fk_against_independent_authored_vectors_including_limits(self):
        chain = chain_from_profile(runner_right_sole_profile())
        for h in (-.75, -.72, -.6, -.2, 0., .2, .6, .75):
            for a in (-.3, -.27, -.2, 0., .2, .27, .3):
                with self.subTest(hip=h, ankle=a):
                    actual = cpu_fk(chain, (h, a))
                    self.assertTrue(actual.valid)
                    self.assertNumbers(actual.point, runner_point((h, a)))
                    self.assertAlmostEqual(actual.orientation_radians, h + a,
                                           delta=FLOAT64_ANGLE_ATOL)
                    self.assertNumbers(actual.jacobian, runner_jacobian((h, a)))

    def test_jacobian_matches_independent_finite_differences_at_interior_points(self):
        chain = chain_from_profile(runner_right_sole_profile())
        for q in ((0., 0.), (-.55, -.21), (.4, .11), (.67, -.25)):
            result = cpu_fk(chain, q)
            for column in range(2):
                plus, minus = list(q), list(q)
                plus[column] += DIFFERENCE_STEP
                minus[column] -= DIFFERENCE_STEP
                expected = tuple((p - m) / (2 * DIFFERENCE_STEP)
                                 for p, m in zip(runner_point(plus), runner_point(minus)))
                self.assertNumbers(tuple(row[column] for row in result.jacobian), expected,
                                   atol=JACOBIAN_ATOL, rtol=JACOBIAN_RTOL)

    def test_world_frame_conversion_and_jacobian_rotate_together(self):
        profile = runner_right_sole_profile()
        observation = IKObservation(
            id="frame-fixture", tick=0, lifecycle=LifecycleStamp("ik-frame", "runner-0", 1),
            profile_sha256=profile.sha256, source_sha256=profile.source_sha256,
            topology_sha256=profile.topology_sha256, root_pose=RootPose((10., 20.), math.pi / 2),
            part_present=(True,) * 5 + (False,) * 3,
            joint_present=(True,) * 4 + (False,) * 3,
            joint_measured=(False, False, True, True, False, False, False),
            joint_active=(False, False, True, True, False, False, False),
            joint_positions=(None, None, 0., 0., None, None, None))
        self.assertNumbers(point_to_root((1., 24.6), "world", observation), (4.6, 9.))
        result = cpu_fk(chain_from_profile(profile), (0., 0.))
        world_jacobian = (tuple(-v for v in result.jacobian[1]), result.jacobian[0])
        self.assertNumbers(world_jacobian, ((-1., -1.), (-6.2, -1.25)))
        # Independent world points: root + ninety-degree rotation of the
        # authored-vector endpoint, differentiated without production transforms.
        for column in range(2):
            plus, minus = [0., 0.], [0., 0.]
            plus[column], minus[column] = DIFFERENCE_STEP, -DIFFERENCE_STEP
            p, m = runner_point(plus), runner_point(minus)
            difference = (-(p[1] - m[1]) / (2 * DIFFERENCE_STEP),
                          (p[0] - m[0]) / (2 * DIFFERENCE_STEP))
            self.assertNumbers(tuple(row[column] for row in world_jacobian), difference,
                               atol=JACOBIAN_ATOL, rtol=JACOBIAN_RTOL)

    def test_prismatic_fixture_uses_cells_and_parent_local_axis(self):
        chain = PlanarChain(
            id="synthetic-slide", root_id="base",
            joints=(PlanarJoint("slide", "base", "carriage", (1., 2.), (.5, 0.),
                                math.pi / 2, -2., 3., kind="prismatic", axis=(0., 1.)),),
            effector_local=(1.5, 0.))
        for travel in (-2., 0., 1.25, 3.):
            result = cpu_fk(chain, (travel,))
            self.assertTrue(result.valid)
            self.assertNumbers(result.point, (1., 3. + travel))
            self.assertNumbers(result.jacobian, ((0.,), (1.,)))
            self.assertNumbers(result.orientation_jacobian, (0.,))
            self.assertAlmostEqual(result.orientation_radians, math.pi / 2,
                                   delta=FLOAT64_ANGLE_ATOL)

    def test_mixed_fixture_rotates_slide_axis_with_parent(self):
        for hinge, travel in ((0., 0.), (.2, .3), (-.7, -.5), (math.pi / 2, 2.)):
            angle = math.pi / 2 + hinge
            c, s, length = math.cos(angle), math.sin(angle), 3. + travel
            actual = cpu_fk(mixed_chain(), (hinge, travel))
            self.assertNumbers(actual.point, (1. + length * c, 2. + length * s))
            self.assertNumbers(actual.jacobian, ((-length * s, c), (length * c, s)))
            self.assertNumbers(actual.orientation_jacobian, (1., 0.))

    def test_singular_and_near_singular_geometry_stays_finite(self):
        chain = two_link()
        for elbow in (0., 1e-10, -1e-10, math.pi, -math.pi):
            with self.subTest(elbow=elbow):
                result = cpu_fk(chain, (0., elbow))
                self.assertTrue(result.valid)
                self.assertTrue(all(math.isfinite(value) for value in flat(
                    (result.point, result.jacobian, result.transform))))
                determinant = (result.jacobian[0][0] * result.jacobian[1][1]
                               - result.jacobian[0][1] * result.jacobian[1][0])
                self.assertAlmostEqual(determinant, 2 * math.sin(elbow), delta=1e-10)
        # Runner's actual declared domain has no singularity.
        runner = chain_from_profile(runner_right_sole_profile())
        for elbow in (-.3, 0., .3):
            j = cpu_fk(runner, (0., elbow)).jacobian
            self.assertGreater(abs(j[0][0] * j[1][1] - j[0][1] * j[1][0]), 2.9)

    def test_exact_bounds_pass_and_coordinates_beyond_bounds_are_masked(self):
        chain = chain_from_profile(runner_right_sole_profile())
        for q in ((-.75, -.3), (.75, .3)):
            self.assertTrue(cpu_fk(chain, q).valid)
        for q in ((math.nextafter(.75, math.inf), 0.),
                  (0., math.nextafter(-.3, -math.inf))):
            self.assertFiniteInvalid(cpu_fk(chain, q))

    def test_known_passive_configuration_contributes_pose_without_jacobian_authority(self):
        chain = two_link()
        passive = replace(chain, joints=(chain.joints[0],
                                         replace(chain.joints[1], participation="passive")))
        q = (.2, .4)
        full, result = cpu_fk(chain, q), cpu_fk(passive, q)
        self.assertTrue(result.valid)
        self.assertNumbers(result.point, full.point)
        self.assertNumbers(result.jacobian, ((full.jacobian[0][0], 0.),
                                            (full.jacobian[1][0], 0.)))
        self.assertNumbers(result.orientation_jacobian, (1., 0.))
        self.assertFiniteInvalid(cpu_fk(passive, q, active=(True, True)))
        self.assertFiniteInvalid(cpu_fk(passive, q, measured=(True, False)))

    def test_presence_measurement_activity_and_task_masks_are_distinct(self):
        chain, q = two_link(), (.2, .4)
        full = cpu_fk(chain, q)
        reduced = cpu_fk(chain, q, active=(True, False), task_mask=(True, False))
        self.assertTrue(reduced.valid)
        self.assertNumbers(reduced.point, full.point)
        self.assertNumbers(reduced.jacobian, ((full.jacobian[0][0], 0.), (0., 0.)))
        for masks in ({"present": (True, False)}, {"measured": (False, True)},
                      {"task_mask": (False, False)}):
            with self.subTest(masks=masks):
                self.assertFiniteInvalid(cpu_fk(chain, q, **masks))

    def test_invalid_nonfinite_coordinates_produce_no_finite_looking_valid_pose(self):
        for value in (math.nan, math.inf, -math.inf, 1e308):
            with self.subTest(value=value):
                self.assertFiniteInvalid(cpu_fk(two_link(), (value, 0.)))

    def test_structural_shapes_and_nonboolean_masks_are_rejected(self):
        for q in ((), (0.,), (0., 0., 0.)):
            with self.subTest(q=q), self.assertRaises(ContractError):
                cpu_fk(two_link(), q)
        for masks in ({"present": (True,)}, {"measured": (True, True, True)},
                      {"active": (1, 0)}, {"task_mask": (True,)},
                      {"task_mask": (1, 1)}):
            with self.subTest(masks=masks), self.assertRaises(ContractError):
                cpu_fk(two_link(), (0., 0.), **masks)

    def test_unsupported_disconnected_cyclic_and_branching_descriptors_reject(self):
        chain = two_link()
        for joints in ((chain.joints[1], chain.joints[0]),
                       (chain.joints[0], replace(chain.joints[1], parent_id="base")),
                       (chain.joints[0], replace(chain.joints[1], child_id="base")),
                       (chain.joints[0], replace(chain.joints[1], id="shoulder"))):
            with self.subTest(joints=joints), self.assertRaises(ContractError):
                replace(chain, joints=joints)
        for kind in ("spring", "closed_loop", "fixed", "spherical"):
            with self.subTest(kind=kind), self.assertRaises(ContractError):
                replace(chain.joints[0], kind=kind)
        for axis in ((0., 0.), (2., 0.), (math.nan, 0.)):
            with self.subTest(axis=axis), self.assertRaises(ContractError):
                replace(chain.joints[0], kind="prismatic", axis=axis)

    def test_seven_joint_capacity_and_descriptor_roundtrip(self):
        joints = tuple(PlanarJoint(f"j{i}", "base" if i == 0 else f"p{i-1}", f"p{i}",
                                  (0., 0.), (0., 0.), .05, -.5, .5)
                       for i in range(7))
        chain = PlanarChain("synthetic-capacity", "base", joints, (1., 0.))
        self.assertEqual(PlanarChain.from_json(chain.to_json()), chain)
        result = cpu_fk(chain, (.05,) * 7)
        self.assertNumbers(result.point, (math.cos(.7), math.sin(.7)))
        self.assertNumbers(result.jacobian, ((-math.sin(.7),) * 7, (math.cos(.7),) * 7))
        eighth = PlanarJoint("j7", "p6", "p7", (0., 0.), (0., 0.), 0., -.5, .5)
        with self.assertRaises(ContractError):
            replace(chain, joints=joints + (eighth,))


@unittest.skipUnless(OPTIONAL, "optional pinned IK Torch/NumPy runtime is not installed")
class IKGeometryTorchCPUConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = configure_device("cpu", 5203)
        cls.torch = cls.context.torch

    def test_float64_batched_runner_matches_independent_points_and_jacobians(self):
        torch = self.torch
        pairs = [(h, a) for h in (-.75, -.6, 0., .6, .75) for a in (-.3, 0., .3)]
        q = torch.tensor(pairs, dtype=torch.float64).reshape(3, 5, 2)
        result = TorchFK(chain_from_profile(runner_right_sole_profile()), self.context,
                         dtype=torch.float64)(q)
        expected = torch.tensor([runner_point(pair) for pair in pairs], dtype=torch.float64).reshape(3, 5, 2)
        jacobian = torch.tensor([runner_jacobian(pair) for pair in pairs], dtype=torch.float64).reshape(3, 5, 2, 2)
        self.assertEqual(result.point.shape, (3, 5, 2))
        self.assertEqual(result.jacobian.shape, (3, 5, 2, 2))
        self.assertEqual(result.valid.shape, (3, 5))
        self.assertTrue(bool(result.valid.all()))
        torch.testing.assert_close(result.point, expected, atol=FLOAT64_POINT_ATOL, rtol=0.)
        torch.testing.assert_close(result.jacobian, jacobian, atol=FLOAT64_POINT_ATOL, rtol=0.)
        torch.testing.assert_close(result.orientation_radians, q.sum(-1),
                                   atol=FLOAT64_ANGLE_ATOL, rtol=0.)

    def test_float32_parity_has_named_length_scale_tolerance(self):
        torch = self.torch
        pairs = ((.1, -.2), (.3, .7), (-.6, 1.4), (1.1, -.9))
        for scale in (.001, 1., 1000.):
            with self.subTest(scale=scale):
                chain = two_link(scale)
                result = TorchFK(chain, self.context)(torch.tensor(pairs, dtype=torch.float32))
                expected = [cpu_fk(chain, pair) for pair in pairs]
                atol = FLOAT32_SCALE_ATOL * max(1., scale)
                torch.testing.assert_close(result.point,
                    torch.tensor([row.point for row in expected], dtype=torch.float32),
                    atol=atol, rtol=FLOAT32_RTOL)
                torch.testing.assert_close(result.jacobian,
                    torch.tensor([row.jacobian for row in expected], dtype=torch.float32),
                    atol=atol, rtol=FLOAT32_RTOL)
                torch.testing.assert_close(result.orientation_radians,
                    torch.tensor([sum(pair) for pair in pairs]), atol=FLOAT32_ANGLE_ATOL, rtol=0.)
                self.assertEqual(result.point.dtype, torch.float32)
                self.assertEqual(result.point.device, self.context.device)

    def test_float32_runner_bounds_match_cpu_for_the_actual_tensor_values(self):
        torch = self.torch
        chain = chain_from_profile(runner_right_sole_profile())
        fk = TorchFK(chain, self.context)
        rows = []
        for column, endpoints in ((0, (-.75, .75)), (1, (-.3, .3))):
            for endpoint in endpoints:
                rounded = torch.tensor(endpoint, dtype=torch.float32)
                for value in (torch.nextafter(rounded, torch.tensor(-math.inf)),
                              rounded,
                              torch.nextafter(rounded, torch.tensor(math.inf))):
                    q = torch.zeros(2, dtype=torch.float32)
                    q[column] = value
                    rows.append(q)
        q = torch.stack(rows)
        result = fk(q)
        # Compare identical binary32 values promoted exactly to Python float,
        # not the unrounded decimal literals used to construct the tensors.
        expected = [cpu_fk(chain, tuple(row)) for row in q.tolist()]
        self.assertEqual(result.valid.tolist(), [row.valid for row in expected])
        self.assertEqual(result.invalid_code.tolist(), [row.invalid_code for row in expected])
        torch.testing.assert_close(result.point,
            torch.tensor([row.point for row in expected], dtype=torch.float32),
            atol=FLOAT32_SCALE_ATOL * chain.geometry_scale_cells, rtol=FLOAT32_RTOL)
        # Rounding +/-0.3 to nearest binary32 crosses the real hard interval;
        # the adjacent value toward zero is admitted instead.
        self.assertFalse(bool(fk(torch.tensor([0., .3])).valid))
        self.assertFalse(bool(fk(torch.tensor([0., -.3])).valid))
        for endpoint in (-.3, .3):
            inside = torch.nextafter(torch.tensor(endpoint), torch.tensor(0.))
            self.assertTrue(bool(fk(torch.stack((torch.tensor(0.), inside))).valid))

    def test_float32_signed_prismatic_limits_are_inward_and_maximally_permissive(self):
        torch = self.torch
        for lower, upper in ((.1, .3), (-.3, -.1), (-.5, .5)):
            with self.subTest(lower=lower, upper=upper):
                chain = PlanarChain(
                    "synthetic-dtype-bounds", "base",
                    (PlanarJoint("slide", "base", "carriage", (0., 0.), (0., 0.),
                                 0., lower, upper, kind="prismatic"),), (1., 0.))
                fk = TorchFK(chain, self.context)
                lo, hi = fk.lower[0], fk.upper[0]
                self.assertGreaterEqual(lo.item(), lower)
                self.assertLessEqual(hi.item(), upper)
                below = torch.nextafter(lo, torch.tensor(-math.inf))
                above = torch.nextafter(hi, torch.tensor(math.inf))
                self.assertLess(below.item(), lower)
                self.assertGreater(above.item(), upper)
                q = torch.stack((below, lo, hi, above)).reshape(4, 1)
                result = fk(q)
                expected = [cpu_fk(chain, tuple(row)) for row in q.tolist()]
                self.assertEqual(result.valid.tolist(), [False, True, True, False])
                self.assertEqual(result.valid.tolist(), [row.valid for row in expected])
                self.assertEqual(result.invalid_code.tolist(), [row.invalid_code for row in expected])

    def test_displaced_and_mixed_fixtures_match_independent_hand_geometry(self):
        torch = self.torch
        for chain, pairs in ((displaced_chain(), ((0., 0.), (math.pi / 2, 0.))),
                             (mixed_chain(), ((0., 0.), (.2, .3), (-.7, -.5)))):
            q = torch.tensor(pairs, dtype=torch.float64)
            result = TorchFK(chain, self.context, dtype=torch.float64)(q)
            expected = [cpu_fk(chain, pair) for pair in pairs]
            for name in ("point", "jacobian", "transform", "orientation_jacobian"):
                torch.testing.assert_close(getattr(result, name),
                    torch.tensor([getattr(row, name) for row in expected], dtype=torch.float64),
                    atol=FLOAT64_POINT_ATOL, rtol=0.)

    def test_autograd_and_reported_jacobian_agree_with_independent_derivative(self):
        torch = self.torch
        fk = TorchFK(chain_from_profile(runner_right_sole_profile()), self.context, dtype=torch.float64)
        q = torch.tensor([[-.3, .12], [.5, -.18]], dtype=torch.float64, requires_grad=True)
        result = fk(q)
        for row in range(2):
            expected = torch.tensor(runner_jacobian(q[row].detach().tolist()), dtype=torch.float64)
            for component in range(2):
                gradient = torch.autograd.grad(result.point[row, component], q, retain_graph=True)[0]
                torch.testing.assert_close(gradient[row], expected[component], atol=FLOAT64_POINT_ATOL, rtol=0.)
                torch.testing.assert_close(gradient[1-row], torch.zeros(2, dtype=torch.float64), atol=0., rtol=0.)
        self.assertTrue(torch.autograd.gradcheck(lambda values: fk(values).point, (q,),
                                                eps=DIFFERENCE_STEP, atol=JACOBIAN_ATOL,
                                                rtol=JACOBIAN_RTOL))

    def test_invalid_rows_are_masked_without_poisoning_valid_rows_or_gradients(self):
        torch = self.torch
        q = torch.tensor([[.2, -.1], [math.nan, 0.], [0., math.inf], [.76, 0.]],
                         dtype=torch.float64, requires_grad=True)
        result = TorchFK(chain_from_profile(runner_right_sole_profile()), self.context,
                         dtype=torch.float64)(q)
        self.assertEqual(result.valid.tolist(), [True, False, False, False])
        self.assertEqual(result.invalid_code[0].item(), 0)
        self.assertTrue(bool((result.invalid_code[1:] != 0).all()))
        for name in ("point", "orientation_radians", "jacobian", "orientation_jacobian", "transform"):
            self.assertTrue(bool(torch.isfinite(getattr(result, name)).all()), name)
        torch.testing.assert_close(result.point[1:], torch.zeros((3, 2), dtype=torch.float64), atol=0., rtol=0.)
        result.point.sum().backward()
        self.assertTrue(bool(torch.isfinite(q.grad).all()))
        torch.testing.assert_close(q.grad[1:], torch.zeros((3, 2), dtype=torch.float64), atol=0., rtol=0.)

    def test_batched_masks_keep_unknown_state_distinct_from_known_passive_state(self):
        torch = self.torch
        fk = TorchFK(two_link(), self.context, dtype=torch.float64)
        q = torch.tensor([[.2, .4]] * 4, dtype=torch.float64)
        present = torch.tensor([[True, True], [True, False], [True, True], [True, True]])
        measured = torch.tensor([[True, True], [True, False], [False, True], [True, True]])
        active = torch.tensor([[True, False], [True, False], [True, True], [True, True]])
        task = torch.tensor([[True, False], [True, True], [True, True], [False, False]])
        result = fk(q, present=present, measured=measured, active=active, task_mask=task)
        self.assertEqual(result.valid.tolist(), [True, False, False, False])
        expected = cpu_fk(two_link(), (.2, .4), active=(True, False), task_mask=(True, False))
        torch.testing.assert_close(result.point[0], torch.tensor(expected.point, dtype=torch.float64),
                                   atol=FLOAT64_POINT_ATOL, rtol=0.)
        torch.testing.assert_close(result.jacobian[0], torch.tensor(expected.jacobian, dtype=torch.float64),
                                   atol=FLOAT64_POINT_ATOL, rtol=0.)

    def test_scalar_noncontiguous_and_empty_batches_preserve_shapes(self):
        torch = self.torch
        fk = TorchFK(two_link(), self.context)
        scalar = fk(torch.tensor([0., 0.]))
        self.assertEqual(scalar.point.shape, (2,))
        self.assertEqual(scalar.valid.shape, ())
        q = torch.zeros((4, 3, 2)).transpose(0, 1)
        self.assertFalse(q.is_contiguous())
        result = fk(q)
        self.assertEqual(result.point.shape, (3, 4, 2))
        torch.testing.assert_close(result.point[..., 0], torch.full((3, 4), 3.), atol=0., rtol=0.)
        empty = fk(torch.zeros((0, 2)))
        self.assertEqual(empty.point.shape, (0, 2))
        self.assertEqual(empty.jacobian.shape, (0, 2, 2))
        self.assertEqual(empty.valid.shape, (0,))

    def test_hot_path_has_no_host_value_or_host_tensor_transfer(self):
        torch = self.torch
        fk = TorchFK(two_link(), self.context)
        q = torch.zeros((13, 2), requires_grad=True)
        failure = AssertionError("FK hot path inspected or transferred tensor data on host")
        with patch.object(torch.Tensor, "item", side_effect=failure), \
                patch.object(torch.Tensor, "__bool__", side_effect=failure), \
                patch.object(torch.Tensor, "tolist", side_effect=failure), \
                patch.object(torch.Tensor, "cpu", side_effect=failure), \
                patch.object(torch.Tensor, "numpy", side_effect=failure):
            result = fk(q)
            result.point.sum().backward()
        self.assertEqual(result.point.shape, (13, 2))
        self.assertTrue(bool(torch.isfinite(q.grad).all()))

    def test_structural_dtype_shape_device_and_mask_mismatches_reject(self):
        torch = self.torch
        fk = TorchFK(two_link(), self.context)
        for q in (torch.zeros((3, 1)), torch.zeros((3, 2), dtype=torch.float64),
                  torch.zeros((3, 2), dtype=torch.int64), torch.zeros((3, 2), device="meta")):
            with self.subTest(shape=q.shape, dtype=q.dtype, device=q.device), self.assertRaises(ValueError):
                fk(q)
        q = torch.zeros((3, 2))
        for masks in ({"present": torch.ones((3, 1), dtype=torch.bool)},
                      {"measured": torch.ones((3, 2), dtype=torch.int64)},
                      {"active": torch.ones((3, 2), device="meta", dtype=torch.bool)},
                      {"task_mask": torch.ones((2,), dtype=torch.bool)}):
            with self.subTest(masks=list(masks)), self.assertRaises(ValueError):
                fk(q, **masks)


@unittest.skipUnless(CUDA_SELECTION, "actual CUDA FK parity unmeasured; explicit NCA_IK_CUDA_TEST_DEVICE required")
class IKGeometryManualCUDAConformanceTests(unittest.TestCase):
    def test_requested_device_runs_batched_fk_and_backward_without_cpu_fallback(self):
        context = configure_device(CUDA_SELECTION, 5203)
        self.assertEqual(context.device.type, "cuda")
        torch = context.torch
        pairs = [(-.6, -.2), (0., 0.), (.4, .1), (.6, .2)]
        q = torch.tensor(pairs, device=context.device, dtype=torch.float32, requires_grad=True)
        result = TorchFK(chain_from_profile(runner_right_sole_profile()), context)(q)
        result.point.square().sum().backward()
        context.synchronize()
        context.verify_tensors({"configuration": q, "point": result.point,
                                "jacobian": result.jacobian, "gradient": q.grad})
        expected = torch.tensor([runner_point(pair) for pair in pairs], dtype=torch.float32,
                                device=context.device)
        torch.testing.assert_close(result.point, expected,
                                   atol=FLOAT32_SCALE_ATOL * 6.550781059358212,
                                   rtol=FLOAT32_RTOL)
        self.assertTrue(bool(result.valid.all()))


if __name__ == "__main__":
    unittest.main()
