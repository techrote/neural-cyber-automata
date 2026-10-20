"""Hand-calculated source geometry; these checks execute no game or training."""

import math
import unittest
from dataclasses import FrozenInstanceError, replace

from nca.contracts import ContractError
from nca.ik.profiles import (
    IKProfile, nominal_outer_reach, nominal_pose, relative_rotary_angle, rotate,
    runner_right_sole_profile,
)


class IKProfileTests(unittest.TestCase):
    def setUp(self):
        self.profile = runner_right_sole_profile()

    def assertPoint(self, actual, expected, places=12):
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=places)

    def pose(self, hip, ankle, profile=None):
        return nominal_pose(profile or self.profile, {"hip_1": hip, "ankle_1": ankle})

    def test_exact_catalogue_parts_and_capacity(self):
        expected = (
            ("chassis", (0, 0), 0, (6, 8)),
            ("leg_-1", (-3.6, 5.5), -.18, (2, 5)),
            ("foot_-1", (-4.6, 8), 0, (4, 2)),
            ("leg_1", (3.6, 5.5), .18, (2, 5)),
            ("foot_1", (4.6, 8), 0, (4, 2)),
        )
        for index, (part, (name, point, angle, size)) in enumerate(zip(self.profile.parts, expected)):
            self.assertEqual((part.id, part.index, part.initial_position,
                              part.initial_angle_radians, part.size), (name, index, point, angle, size))
        self.assertEqual(len(self.profile.parts), 5)
        self.assertEqual(len(self.profile.joints), 4)
        self.assertEqual((self.profile.part_capacity, self.profile.joint_capacity), (8, 7))
        self.assertEqual(self.profile.effector.part_id, "foot_1")
        self.assertEqual(self.profile.effector.point_local, (0, 1))
        self.assertEqual(self.profile.host_target_lane, "absent_requires_separate_host_integration")

    def test_both_local_anchors_and_rotary_zero_references(self):
        hip, ankle = self.profile.effector_chain
        s, c = math.sin(.18), math.cos(.18)
        self.assertPoint(hip.parent_anchor_local, (3.6, 2.8))
        self.assertPoint(hip.child_anchor_local, (-2.7 * s, -2.7 * c))
        self.assertPoint(ankle.parent_anchor_local, (2.25 * s, 2.25 * c))
        self.assertPoint(ankle.child_anchor_local, (-1, -.25))
        self.assertEqual((hip.zero_radians, ankle.zero_radians), (.18, -.18))
        self.assertPoint((hip.lower + hip.zero_radians, hip.upper + hip.zero_radians), (-.57, .93))
        self.assertPoint((ankle.lower + ankle.zero_radians, ankle.upper + ankle.zero_radians), (-.48, .12))
        for joint in self.profile.joints:
            self.assertAlmostEqual(joint.target_lower - joint.lower, .03)
            self.assertAlmostEqual(joint.upper - joint.target_upper, .03)
            self.assertEqual(joint.motor_speed_radians_per_second, 7)

    def test_neutral_and_independently_calculated_poses(self):
        cases = (
            (0, 0, (4.6, 9)),
            (.1, 0, (3.976036982067691, 9.068859241370589)),
            (0, .1, (4.470212394469491, 9.093588623244361)),
            (.2, -.1, (3.486799207033938, 8.994918183558507)),
        )
        for hip, ankle, point in cases:
            with self.subTest(hip=hip, ankle=ankle):
                pose = self.pose(hip, ankle)
                self.assertPoint(pose.point, point)
                self.assertAlmostEqual(pose.orientation_radians, hip + ankle)

    def test_anchor_fk_matches_independent_closed_formula_at_limits(self):
        for hip in (-.75, -.6, 0, .6, .75):
            for ankle in (-.3, -.2, 0, .2, .3):
                with self.subTest(hip=hip, ankle=ankle):
                    angle = hip + ankle
                    expected = (3.6 - 4.95 * math.sin(hip) + math.cos(angle) - 1.25 * math.sin(angle),
                                2.8 + 4.95 * math.cos(hip) + math.sin(angle) + 1.25 * math.cos(angle))
                    self.assertPoint(self.pose(hip, ankle).point, expected)

    def test_down_y_rotation_and_neutral_jacobian_signs(self):
        self.assertPoint(rotate((1, 0), math.pi / 2), (0, 1))
        self.assertPoint(rotate((0, 1), math.pi / 2), (-1, 0))
        self.assertPoint(rotate((1, 2), 2 * math.pi), (1, 2))
        h = 1e-6
        for plus, minus, expected in (
            (self.pose(h, 0), self.pose(-h, 0), (-6.2, 1)),
            (self.pose(0, h), self.pose(0, -h), (-1.25, 1)),
        ):
            self.assertPoint(tuple((a - b) / (2 * h) for a, b in zip(plus.point, minus.point)), expected, 8)

    def test_rotary_angle_uses_authored_zero_and_half_open_wrap(self):
        self.assertEqual(relative_rotary_angle(.18, 0, .18), 0)
        self.assertEqual(relative_rotary_angle(0, .18, -.18), 0)
        self.assertAlmostEqual(relative_rotary_angle(.38, .1, .18), .1)
        self.assertAlmostEqual(relative_rotary_angle(-math.pi + .1, math.pi - .1, 0), .2)
        self.assertAlmostEqual(relative_rotary_angle(math.pi - .1, -math.pi + .1, 0), -.2)
        self.assertAlmostEqual(relative_rotary_angle(math.pi - .05, -math.pi + .05, .18), -.28)
        for angle in (math.pi, -math.pi, 3 * math.pi, -3 * math.pi):
            with self.subTest(angle=angle):
                self.assertEqual(relative_rotary_angle(angle, 0, 0), -math.pi)
        below = relative_rotary_angle(math.nextafter(math.pi, -math.inf), 0, 0)
        above = relative_rotary_angle(math.nextafter(math.pi, math.inf), 0, 0)
        self.assertTrue(0 < below < math.pi)
        self.assertTrue(-math.pi < above < 0)

    def test_rotary_angle_rejects_invalid_inputs_and_subtraction_overflow(self):
        for args in ((True, 0, 0), (0, "0", 0), (0, 0, math.nan),
                     (math.inf, 0, 0), (0, -math.inf, 0), (1e308, -1e308, 0),
                     (1e308, 0, -1e308), (10 ** 400, 0, 0)):
            with self.subTest(args=args), self.assertRaises(ContractError):
                relative_rotary_angle(*args)

    def test_mirrored_authored_geometry_has_reflected_motion(self):
        reflect = lambda p: (-p[0], p[1])
        # A synthetic symmetry check of the mathematics, not another host profile.
        mirrored = replace(self.profile,
            parts=tuple(replace(p, initial_position=reflect(p.initial_position),
                                initial_angle_radians=-p.initial_angle_radians) for p in self.profile.parts),
            joints=tuple(replace(j, authored_pivot=reflect(j.authored_pivot),
                                 parent_anchor_local=reflect(j.parent_anchor_local),
                                 child_anchor_local=reflect(j.child_anchor_local),
                                 zero_radians=-j.zero_radians) for j in self.profile.joints))
        for hip, ankle in ((0, 0), (.2, -.1), (-.6, .2)):
            original, mirror = self.pose(hip, ankle), self.pose(-hip, -ankle, mirrored)
            self.assertPoint(mirror.point, reflect(original.point))
            self.assertAlmostEqual(mirror.orientation_radians, -original.orientation_radians)

    def test_workspace_is_a_joint_pushforward_with_nonzero_rank(self):
        workspace = self.profile.workspace
        self.assertEqual(workspace.joint_ids, self.profile.solver_joint_ids)
        self.assertEqual(workspace.lower, (-.6, -.2))
        self.assertEqual(workspace.upper, (.6, .2))
        center, radius = nominal_outer_reach(self.profile)
        self.assertEqual(center, (3.6, 2.8))
        self.assertAlmostEqual(radius, 4.95 + math.hypot(1, 1.25))
        for hip in (-.6, 0, .6):
            for ankle in (-.2, 0, .2):
                point = self.pose(hip, ankle).point
                self.assertLessEqual(math.dist(center, point), radius + 1e-12)
                self.assertGreater(4.95 * (math.cos(ankle) - 1.25 * math.sin(ankle)), 3.62)
        self.assertEqual(workspace.claim, "nominal_geometric_reachability_only")

    def test_only_solver_measurements_are_required_and_no_silent_defaults(self):
        self.assertEqual(tuple(j.participation for j in self.profile.joints),
                         ("passive", "passive", "active", "active"))
        for values in ({}, {"hip_1": 0}, {"hip_1": 0, "ankle_1": 0, "hip_-1": 0},
                       {"hip_1": True, "ankle_1": 0}, {"hip_1": math.inf, "ankle_1": 0},
                       {"hip_1": .750001, "ankle_1": 0}, {"hip_1": 0, "ankle_1": -.300001}):
            with self.subTest(values=values), self.assertRaises(ContractError):
                nominal_pose(self.profile, values)
        disabled_context = replace(self.profile,
            joints=(replace(self.profile.joints[0], participation="disabled"), *self.profile.joints[1:]))
        self.assertEqual(self.pose(.1, .1, disabled_context), self.pose(.1, .1))
        self.assertNotEqual(disabled_context.topology_sha256, self.profile.topology_sha256)

    def test_roundtrip_immutability_and_source_identity(self):
        p = self.profile
        self.assertEqual(IKProfile.from_json(p.to_json()), p)
        self.assertEqual(p.source.commit, "f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497")
        self.assertEqual(tuple(s.git_blob_sha1 for s in p.source.files),
                         ("397ac8a3103134da276590c5a409b17429394836", "ba2f00d018fa7db849927bb0a8bfef2aaa875c03"))
        self.assertEqual(p.sha256, "6cb15832e240f82f4bcd582d2c26015ef986e44f335da683abaf9746e31503c4")
        with self.assertRaises(FrozenInstanceError):
            p.max_delta_radians = .1
        detached = p.to_dict()
        detached["parts"][0]["size"][0] = 99
        self.assertEqual(p.parts[0].size, (6, 8))

    def test_hashes_bind_distinct_source_geometry_and_topology(self):
        p = self.profile
        new_source = replace(p, source=replace(p.source, commit="1" * 40))
        self.assertNotEqual(new_source.source_sha256, p.source_sha256)
        self.assertNotEqual(new_source.sha256, p.sha256)
        self.assertEqual(new_source.topology_sha256, p.topology_sha256)
        changed_point = replace(p, effector=replace(p.effector, point_local=(.1, 1)))
        self.assertNotEqual(changed_point.sha256, p.sha256)
        self.assertEqual(changed_point.topology_sha256, p.topology_sha256)
        self.assertEqual(changed_point.source_sha256, p.source_sha256)
        self.assertNotEqual(replace(p, max_delta_radians=.1).sha256, p.sha256)

    def test_malformed_topologies_and_slots_are_rejected(self):
        def edits():
            data = self.profile.to_dict(); data["parts"] += data["parts"]
            yield "capacity", data
            data = self.profile.to_dict(); data["parts"][1]["index"] = 0
            yield "duplicate slot", data
            data = self.profile.to_dict(); data["parts"][1]["id"] = "chassis"
            yield "duplicate ID", data
            data = self.profile.to_dict(); data["joints"][0]["parent_part_id"] = "foot_-1"
            yield "disconnected cycle", data
            data = self.profile.to_dict(); data["joints"].pop()
            yield "disconnected part", data
            data = self.profile.to_dict(); data["joints"][0]["parent_part_id"] = "absent"
            yield "missing parent", data
            data = self.profile.to_dict(); data["joints"][1]["child_part_id"] = "leg_1"
            yield "multiple parent", data
            data = self.profile.to_dict(); data["joints"][0]["kind"] = "slide"
            yield "unsupported joint", data
            data = self.profile.to_dict(); data["solver_joint_ids"].reverse()
            yield "solver order", data
        for name, data in edits():
            with self.subTest(name=name), self.assertRaises(ContractError):
                IKProfile.from_dict(data)

    def test_malformed_frames_anchors_units_and_numeric_fields_are_rejected(self):
        edits = (
            ("unknown", lambda d: d.update(extra=True)),
            ("boolean index", lambda d: d["parts"][0].update(index=False)),
            ("nonfinite", lambda d: d["parts"][1].update(mass=math.nan)),
            ("moved root", lambda d: d["parts"][0].update(initial_position=[1, 0])),
            ("wrong handedness", lambda d: d["convention"].update(axes="x_right_y_up")),
            ("wrong zero", lambda d: d["joints"][2].update(zero_radians=-.18)),
            ("wrong local anchor", lambda d: d["joints"][2].update(child_anchor_local=[0, -2.7])),
            ("target beyond hard", lambda d: d["joints"][3].update(target_upper=.31)),
            ("degrees", lambda d: d["joints"][3].update(coordinate_unit="degrees")),
            ("missing active", lambda d: d["joints"][3].update(participation="disabled")),
            ("extra active", lambda d: d["joints"][0].update(participation="active")),
            ("orientation task", lambda d: d["effector"].update(task_components=["position_x", "position_y", "angle"])),
            ("point outside foot", lambda d: d["effector"].update(point_local=[0, 2])),
            ("workspace beyond target", lambda d: d["workspace"].update(upper=[.74, .2])),
            ("bad source digest", lambda d: d["source"]["files"][0].update(content_sha256="unknown")),
        )
        for name, mutate in edits:
            with self.subTest(name=name), self.assertRaises(ContractError):
                data = self.profile.to_dict()
                mutate(data)
                IKProfile.from_dict(data)


if __name__ == "__main__":
    unittest.main()
